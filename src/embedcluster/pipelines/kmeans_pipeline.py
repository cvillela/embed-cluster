"""FAISS GPU spherical KMeans pipeline."""

from __future__ import annotations

from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DatasetInfo, KmeansConfig, RunPaths, SharedConfig
from ..export import (
    export_jsonl_with_labels,
    write_cluster_summary,
    write_labels_parquet,
    write_metrics_json,
    write_preflight_json,
    write_run_config_json,
)
from ..gpu import choose_batch_size, estimate_run_memory, run_preflight_check
from ..io import load_embeddings_mmap
from ..logging_utils import configure_logging, get_logger
from ..preprocessing import compute_l2_norms, create_l2_normalized_view

logger = get_logger(__name__)


def run_kmeans_pipeline(
    shared_cfg: SharedConfig,
    kmeans_cfg: KmeansConfig,
    run_paths: RunPaths,
    dataset_info: DatasetInfo,
) -> None:
    """End-to-end FAISS GPU spherical KMeans pipeline."""
    import faiss  # type: ignore

    configure_logging(log_file=run_paths.run_log)

    n = dataset_info.n_rows
    d = dataset_info.n_dims
    zero_norm_mask = dataset_info.zero_norm_mask
    valid_mask = ~zero_norm_mask

    dtype_bytes = np.dtype(dataset_info.dtype).itemsize
    if shared_cfg.batch_size == "auto":
        batch_size = choose_batch_size(n, d, dtype_bytes)
    else:
        batch_size = int(shared_cfg.batch_size)

    logger.info("KMeans pipeline start: n=%d, d=%d, batch_size=%d", n, d, batch_size)

    raw = load_embeddings_mmap(dataset_info.embeddings_path)

    # Preflight
    estimates = estimate_run_memory(n, d, None, "kmeans")
    preflight_data = run_preflight_check(estimates)
    preflight_data.update(
        {
            "n_rows": n,
            "n_dims": d,
            "embedding_dtype": dataset_info.dtype,
            "pipeline": "kmeans",
        }
    )
    write_preflight_json(run_paths, preflight_data)

    # Always compute and export norms
    norms = compute_l2_norms(raw, run_paths.norms_npy, batch_size)

    # Working matrix
    if shared_cfg.normalize:
        X = create_l2_normalized_view(raw, norms, run_paths.normalized_npy, batch_size)
        logger.info("Using L2-normalized working matrix: %s", run_paths.normalized_npy)
    else:
        logger.warning(
            "--no-normalize: spherical KMeans is designed for directional data; "
            "results may be suboptimal without normalization."
        )
        X = raw

    # Resolve n_clusters
    n_valid = int(valid_mask.sum())
    if kmeans_cfg.n_clusters is not None:
        n_clusters = kmeans_cfg.n_clusters
    else:
        n_clusters = ceil(n_valid / kmeans_cfg.target_cluster_size)
        n_clusters = min(max(n_clusters, 2), 10_000)

    logger.info("n_clusters=%d (n_valid=%d)", n_clusters, n_valid)

    # Sub-sample valid rows for training
    valid_indices = np.flatnonzero(valid_mask)
    max_train = min(len(valid_indices), 256 * n_clusters)
    if len(valid_indices) > max_train:
        rng = np.random.default_rng(shared_cfg.random_state)
        train_idx = rng.choice(valid_indices, size=max_train, replace=False)
        train_idx.sort()
    else:
        train_idx = valid_indices

    logger.info(
        "Training sub-sample: %d rows (from %d valid)", len(train_idx), len(valid_indices)
    )

    X_train = np.asarray(X[train_idx], dtype=np.float32, order="C")

    # Train
    kmeans_obj = faiss.Kmeans(
        d=d,
        k=n_clusters,
        niter=kmeans_cfg.max_iter,
        nredo=kmeans_cfg.nredo,
        spherical=True,
        gpu=True,
        seed=shared_cfg.random_state,
        verbose=True,
    )
    logger.info("Training FAISS GPU spherical KMeans (k=%d, niter=%d) ...", n_clusters, kmeans_cfg.max_iter)
    kmeans_obj.train(X_train)
    logger.info("Training complete.")

    # Save raw centroids
    centers_raw = np.asarray(kmeans_obj.centroids, dtype=np.float32)
    np.save(run_paths.cluster_centers_npy, centers_raw)
    logger.info("Saved cluster_centers.npy shape=%s", centers_raw.shape)

    # Normalized centroids for cosine computation
    centers_norms = np.linalg.norm(centers_raw, axis=1, keepdims=True)
    centers_normed = centers_raw / np.maximum(centers_norms, 1e-12)

    # Chunked assignment over all rows
    cluster_ids = np.full(n, -1, dtype=np.int64)
    cosine_to_centroid = np.full(n, np.nan, dtype=np.float32)

    n_chunks = max(1, (n + batch_size - 1) // batch_size)
    for chunk_i, start in enumerate(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        batch_valid_mask = valid_mask[start:end]
        if not batch_valid_mask.any():
            continue

        # Global indices of valid rows in this batch
        batch_global_idx = np.arange(start, end, dtype=np.int64)[batch_valid_mask]
        chunk = np.asarray(X[start:end][batch_valid_mask], dtype=np.float32, order="C")

        _, batch_labels = kmeans_obj.index.search(chunk, 1)
        batch_cids = batch_labels[:, 0].astype(np.int64)
        cluster_ids[batch_global_idx] = batch_cids

        # Cosine to centroid uses normalized embeddings
        if shared_cfg.normalize:
            chunk_normed = chunk  # already unit-norm
        else:
            chunk_row_norms = np.linalg.norm(chunk, axis=1, keepdims=True)
            chunk_normed = chunk / np.maximum(chunk_row_norms, 1e-12)

        assigned_centers = centers_normed[batch_cids]
        cos_vals = np.einsum("id,id->i", chunk_normed, assigned_centers).astype(np.float32)
        cosine_to_centroid[batch_global_idx] = cos_vals

        if chunk_i % 10 == 0:
            logger.info("Assignment: chunk %d/%d", chunk_i + 1, n_chunks)

    logger.info("Assignment complete.")

    # Labels DataFrame
    is_noise = cluster_ids == -1
    labels_df = pd.DataFrame(
        {
            "row_id": np.arange(n, dtype=np.int64),
            "cluster_id": cluster_ids,
            "method": "kmeans",
            "is_noise": is_noise,
            "embedding_norm": np.asarray(norms, dtype=np.float32),
            "cosine_to_centroid": cosine_to_centroid,
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
    unique_cids = np.unique(cluster_ids[cluster_ids >= 0])
    actual_n_clusters = len(unique_cids)

    metrics = {
        "method": "kmeans",
        "n_rows": n,
        "n_clusters": actual_n_clusters,
        "target_cluster_size": kmeans_cfg.target_cluster_size,
        "max_iter": kmeans_cfg.max_iter,
        "nredo": kmeans_cfg.nredo,
        "backend": "faiss_gpu_spherical",
    }
    write_metrics_json(run_paths, metrics)

    # Run config
    run_config = {
        "pipeline": "kmeans",
        "n_clusters": n_clusters,
        "target_cluster_size": kmeans_cfg.target_cluster_size,
        "max_iter": kmeans_cfg.max_iter,
        "nredo": kmeans_cfg.nredo,
        "normalize": shared_cfg.normalize,
        "batch_size": batch_size,
        "random_state": shared_cfg.random_state,
        "n_train_rows": int(len(train_idx)),
        "backend": "faiss_gpu_spherical",
    }
    write_run_config_json(run_paths, run_config)

    if shared_cfg.export_jsonl:
        export_jsonl_with_labels(
            dataset_info.metadata_path,
            run_paths.metadata_with_labels_jsonl,
            cluster_ids,
            method="kmeans",
            extra_columns={"cosine_to_centroid": cosine_to_centroid},
        )

    n_noise = int(is_noise.sum())
    logger.info(
        "KMeans pipeline done: %d clusters, %d invalid (zero-norm) rows.",
        actual_n_clusters,
        n_noise,
    )
