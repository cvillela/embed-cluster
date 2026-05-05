"""cuML HDBSCAN pipeline with optional PCA."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DatasetInfo, HdbscanConfig, RunPaths, SharedConfig
from ..export import (
    export_jsonl_with_labels,
    write_cluster_summary,
    write_hdbscan_persistence,
    write_labels_parquet,
    write_metrics_json,
    write_preflight_json,
    write_run_config_json,
)
from ..gpu import choose_batch_size, estimate_run_memory, get_gpu_info, run_preflight_check
from ..io import load_embeddings_mmap
from ..logging_utils import configure_logging, get_logger
from ..preprocessing import (
    compute_l2_norms,
    create_l2_normalized_view,
    fit_transform_pca_optional,
    renormalize_matrix,
)

logger = get_logger(__name__)


def run_hdbscan_pipeline(
    shared_cfg: SharedConfig,
    hdbscan_cfg: HdbscanConfig,
    run_paths: RunPaths,
    dataset_info: DatasetInfo,
) -> None:
    """End-to-end cuML HDBSCAN pipeline."""
    configure_logging(log_file=run_paths.run_log)

    n = dataset_info.n_rows
    d = dataset_info.n_dims
    zero_norm_mask = dataset_info.zero_norm_mask
    min_cluster_size = hdbscan_cfg.min_cluster_size
    min_samples = hdbscan_cfg.min_samples

    dtype_bytes = np.dtype(dataset_info.dtype).itemsize
    if shared_cfg.batch_size == "auto":
        batch_size = choose_batch_size(n, d, dtype_bytes)
    else:
        batch_size = int(shared_cfg.batch_size)

    logger.info("HDBSCAN pipeline start: n=%d, d=%d, batch_size=%d", n, d, batch_size)

    raw = load_embeddings_mmap(dataset_info.embeddings_path)

    # Always compute norms
    norms = compute_l2_norms(raw, run_paths.norms_npy, batch_size)

    # Working matrix
    if shared_cfg.normalize:
        X_work = create_l2_normalized_view(raw, norms, run_paths.normalized_npy, batch_size)
        logger.info("Using L2-normalized working matrix: %s", run_paths.normalized_npy)
    else:
        X_work = raw
        logger.info("--no-normalize: using raw embeddings as working matrix.")

    # Optional PCA → re-normalize
    if hdbscan_cfg.pca_components is not None:
        X_pca = fit_transform_pca_optional(
            X_work,
            hdbscan_cfg.pca_components,
            run_paths.intermediate,
            batch_size,
            shared_cfg.random_state,
        )
        X_work = renormalize_matrix(X_pca, run_paths.pca_normalized_npy, batch_size)
        work_d = hdbscan_cfg.pca_components
        logger.info(
            "PCA(%d) + renormalized working matrix: %s",
            hdbscan_cfg.pca_components,
            run_paths.pca_normalized_npy,
        )
    else:
        work_d = d

    # Preflight check uses actual working-matrix dimensions (post-PCA if any)
    estimates = estimate_run_memory(n, work_d, None, "hdbscan")
    preflight_data = run_preflight_check(estimates)
    gpu_info = get_gpu_info()
    preflight_data.update(
        {
            "n_rows": n,
            "n_dims": d,
            "working_dims": work_d,
            "embedding_dtype": dataset_info.dtype,
            "pipeline": "hdbscan",
            "pca_components": hdbscan_cfg.pca_components,
            "gpu_name": gpu_info.get("gpu_name"),
            "free_gpu_memory_gb": gpu_info.get("free_memory_gb"),
        }
    )
    write_preflight_json(run_paths, preflight_data)

    # Build-algo selection (not exposed in CLI)
    if n >= 1_000_000:
        build_algo = "nn_descent"
        build_kwds: dict = {
            "knn_n_clusters": 4 if n < 2_000_000 else 8,
            "knn_overlap_factor": 2,
            "nnd_graph_degree": max(64, (min_samples or min_cluster_size) + 1),
        }
    else:
        build_algo = "brute_force"
        build_kwds = {}

    logger.info(
        "HDBSCAN: min_cluster_size=%d, min_samples=%s, cluster_selection=%s, "
        "build_algo=%s, build_kwds=%s",
        min_cluster_size,
        min_samples,
        hdbscan_cfg.cluster_selection,
        build_algo,
        build_kwds,
    )

    # Load full working matrix to GPU — HDBSCAN requires it
    import cupy as cp  # type: ignore

    logger.info("Loading working matrix to GPU (%d × %d) ...", n, work_d)
    X_gpu = cp.asarray(np.asarray(X_work, dtype=np.float32))

    from cuml.cluster import HDBSCAN  # type: ignore

    def _build_clusterer(use_build_kwds: bool) -> HDBSCAN:
        kwargs: dict = dict(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method=hdbscan_cfg.cluster_selection,
            prediction_data=False,
            build_algo=build_algo,
            output_type="cupy",
        )
        if use_build_kwds and build_kwds:
            kwargs["build_kwds"] = build_kwds
        return HDBSCAN(**kwargs)

    try:
        clusterer = _build_clusterer(use_build_kwds=True)
        labels_gpu = clusterer.fit_predict(X_gpu)
    except TypeError as exc:
        logger.warning(
            "HDBSCAN TypeError with build_kwds=%s; retrying without. Error: %s",
            build_kwds,
            exc,
        )
        clusterer = _build_clusterer(use_build_kwds=False)
        labels_gpu = clusterer.fit_predict(X_gpu)

    labels_np = np.asarray(labels_gpu.get(), dtype=np.int64)
    probs_np = np.asarray(clusterer.probabilities_.get(), dtype=np.float32)

    # Zero-norm rows → cluster_id = -1, probability = 0
    labels_np[zero_norm_mask] = -1
    probs_np[zero_norm_mask] = 0.0

    # Cluster persistence
    persistence_raw = clusterer.cluster_persistence_
    try:
        persistence_arr = np.asarray(persistence_raw.get(), dtype=np.float32)
    except AttributeError:
        persistence_arr = np.asarray(persistence_raw, dtype=np.float32)
    persistence_df = pd.DataFrame(
        {
            "cluster_id": np.arange(len(persistence_arr), dtype=np.int64),
            "persistence": persistence_arr,
        }
    )
    write_hdbscan_persistence(run_paths, persistence_df)
    logger.info(
        "Wrote hdbscan_cluster_persistence.parquet (%d entries)", len(persistence_df)
    )

    # Labels DataFrame
    is_noise = labels_np == -1
    labels_df = pd.DataFrame(
        {
            "row_id": np.arange(n, dtype=np.int64),
            "cluster_id": labels_np,
            "method": "hdbscan",
            "is_noise": is_noise,
            "probability": probs_np,
            "embedding_norm": np.asarray(norms, dtype=np.float32),
        }
    )
    write_labels_parquet(run_paths, labels_df)
    logger.info("Wrote labels.parquet (%d rows)", n)

    # Cluster summary
    valid_rows = labels_df[~labels_df["is_noise"]]
    if len(valid_rows) > 0:
        summary = (
            valid_rows.groupby("cluster_id")
            .agg(
                size=("row_id", "count"),
                embedding_norm_mean=("embedding_norm", "mean"),
                embedding_norm_std=("embedding_norm", "std"),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame(
            columns=["cluster_id", "size", "embedding_norm_mean", "embedding_norm_std"]
        )
    write_cluster_summary(run_paths, summary)
    logger.info("Wrote cluster_summary.parquet (%d clusters)", len(summary))

    # Metrics
    unique_cids = np.unique(labels_np[labels_np >= 0])
    n_clusters = len(unique_cids)
    n_noise = int(is_noise.sum())
    noise_fraction = n_noise / n if n > 0 else 0.0

    metrics = {
        "method": "hdbscan",
        "n_rows": n,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_fraction": round(noise_fraction, 6),
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "pca_components": hdbscan_cfg.pca_components,
        "cluster_selection_method": hdbscan_cfg.cluster_selection,
    }
    write_metrics_json(run_paths, metrics)

    # Run config (includes internal build choices)
    run_config = {
        "pipeline": "hdbscan",
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "pca_components": hdbscan_cfg.pca_components,
        "cluster_selection": hdbscan_cfg.cluster_selection,
        "normalize": shared_cfg.normalize,
        "batch_size": batch_size,
        "random_state": shared_cfg.random_state,
        "build_algo": build_algo,
        "build_kwds": build_kwds,
    }
    write_run_config_json(run_paths, run_config)

    if shared_cfg.export_jsonl:
        export_jsonl_with_labels(
            dataset_info.metadata_path,
            run_paths.metadata_with_labels_jsonl,
            labels_np,
            method="hdbscan",
            extra_columns={"probability": probs_np},
        )

    logger.info(
        "HDBSCAN pipeline done: %d clusters, %d noise rows (%.2f%%)",
        n_clusters,
        n_noise,
        noise_fraction * 100,
    )
