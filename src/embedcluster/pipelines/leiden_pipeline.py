"""Leiden graph clustering pipeline.

Flow:
    raw embeddings
    -> compute_l2_norms          (norms.npy)
    -> create_l2_normalized_view (normalized.npy, if normalize=True)
    -> cuVS kNN                  (knn_indices.npy, knn_scores.npy)
    -> mutual weighted kNN graph (mutual_edges.parquet)
    -> cuGraph Leiden            -> labels.parquet (+ summaries, metrics)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DatasetInfo, LeidenConfig, RunPaths, SharedConfig
from ..export import (
    export_jsonl_with_labels,
    write_cluster_summary,
    write_labels_parquet,
    write_metrics_json,
    write_preflight_json,
    write_run_config_json,
)
from ..gpu import choose_batch_size, estimate_run_memory, get_gpu_info, run_preflight_check
from ..graph.leiden import run_leiden
from ..graph.mutual_knn import build_mutual_graph
from ..io import load_embeddings_mmap
from ..logging_utils import configure_logging, get_logger
from ..neighbors.cuvs_knn import build_and_search_knn
from ..preprocessing import compute_l2_norms, create_l2_normalized_view

logger = get_logger(__name__)


def _per_row_graph_stats(
    edges_df: pd.DataFrame,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute graph_degree and mean_neighbor_similarity per row from undirected edges."""
    degree = np.zeros(n, dtype=np.int64)
    weight_sum = np.zeros(n, dtype=np.float64)

    if len(edges_df) > 0:
        src = edges_df["src"].to_numpy().astype(np.int64, copy=False)
        dst = edges_df["dst"].to_numpy().astype(np.int64, copy=False)
        w = edges_df["weight"].to_numpy().astype(np.float64, copy=False)

        np.add.at(degree, src, 1)
        np.add.at(degree, dst, 1)
        np.add.at(weight_sum, src, w)
        np.add.at(weight_sum, dst, w)

    mean_sim = np.full(n, np.nan, dtype=np.float32)
    has_edge = degree > 0
    mean_sim[has_edge] = (weight_sum[has_edge] / degree[has_edge]).astype(np.float32)
    return degree, mean_sim


def run_leiden_pipeline(
    shared_cfg: SharedConfig,
    leiden_cfg: LeidenConfig,
    run_paths: RunPaths,
    dataset_info: DatasetInfo,
) -> None:
    """End-to-end Leiden pipeline."""
    configure_logging(log_file=run_paths.run_log)

    n = dataset_info.n_rows
    d = dataset_info.n_dims
    zero_norm_mask = dataset_info.zero_norm_mask
    k = leiden_cfg.k

    dtype_bytes = np.dtype(dataset_info.dtype).itemsize
    if shared_cfg.batch_size == "auto":
        batch_size = choose_batch_size(n, d, dtype_bytes)
    else:
        batch_size = int(shared_cfg.batch_size)

    logger.info(
        "Leiden pipeline start: n=%d, d=%d, k=%d, resolution=%.4f, "
        "min_similarity=%.4f, batch_size=%d",
        n,
        d,
        k,
        leiden_cfg.resolution,
        leiden_cfg.min_similarity,
        batch_size,
    )

    raw = load_embeddings_mmap(dataset_info.embeddings_path)

    # Preflight (uses k for kNN buffer estimate)
    estimates = estimate_run_memory(n, d, k, "leiden")
    preflight_data = run_preflight_check(estimates)
    gpu_info = get_gpu_info()
    preflight_data.update(
        {
            "n_rows": n,
            "n_dims": d,
            "embedding_dtype": dataset_info.dtype,
            "pipeline": "leiden",
            "k": k,
            "gpu_name": gpu_info.get("gpu_name"),
            "free_gpu_memory_gb": gpu_info.get("free_memory_gb"),
        }
    )
    write_preflight_json(run_paths, preflight_data)

    # Always compute and export norms.
    norms = compute_l2_norms(raw, run_paths.norms_npy, batch_size)

    # Working matrix.
    if shared_cfg.normalize:
        X_work = create_l2_normalized_view(
            raw, norms, run_paths.normalized_npy, batch_size
        )
        logger.info("Using L2-normalized working matrix: %s", run_paths.normalized_npy)
    else:
        X_work = raw
        logger.info("--no-normalize: using raw embeddings as working matrix.")

    # cuVS kNN search (with reuse if outputs already exist).
    knn_indices, knn_scores = build_and_search_knn(
        X=X_work,
        k=k,
        normalize=shared_cfg.normalize,
        batch_size=batch_size,
        indices_out=run_paths.knn_indices_npy,
        scores_out=run_paths.knn_scores_npy,
    )

    # Zero-norm rows produce meaningless similarity scores; mark their outgoing
    # edges so they get filtered in the mutual graph step.
    if zero_norm_mask.any():
        zero_idx = np.flatnonzero(zero_norm_mask)
        # Need writable memmaps for masking — reopen in r+ mode.
        idx_writable = np.memmap(
            run_paths.knn_indices_npy, dtype=np.int64, mode="r+", shape=(n, k)
        )
        sc_writable = np.memmap(
            run_paths.knn_scores_npy, dtype=np.float32, mode="r+", shape=(n, k)
        )
        idx_writable[zero_idx] = -1
        sc_writable[zero_idx] = np.nan
        idx_writable.flush()
        sc_writable.flush()
        logger.info(
            "Masked %d zero-norm rows in kNN outputs (set indices=-1).",
            zero_idx.size,
        )
        # Reopen read-only for downstream use.
        knn_indices = np.memmap(
            run_paths.knn_indices_npy, dtype=np.int64, mode="r", shape=(n, k)
        )
        knn_scores = np.memmap(
            run_paths.knn_scores_npy, dtype=np.float32, mode="r", shape=(n, k)
        )

    # Mutual weighted kNN graph.
    tmp_partitions_dir = run_paths.intermediate / "_mutual_partitions_tmp"
    build_mutual_graph(
        indices_path=run_paths.knn_indices_npy,
        scores_path=run_paths.knn_scores_npy,
        n=n,
        k=k,
        min_similarity=leiden_cfg.min_similarity,
        out_path=run_paths.mutual_edges_parquet,
        tmp_dir=tmp_partitions_dir,
    )

    edges_df = pd.read_parquet(run_paths.mutual_edges_parquet)
    n_mutual_edges = len(edges_df)
    logger.info("Mutual graph: %d undirected edges.", n_mutual_edges)

    # Per-row graph stats from the undirected edge list.
    graph_degree, mean_neighbor_sim = _per_row_graph_stats(edges_df, n)

    # Run Leiden.
    leiden_df, modularity = run_leiden(
        edges_path=run_paths.mutual_edges_parquet,
        n=n,
        resolution=leiden_cfg.resolution,
        random_state=shared_cfg.random_state,
    )
    cluster_ids = leiden_df["cluster_id"].to_numpy().astype(np.int64, copy=False)

    # Zero-norm rows must be cluster_id = -1.
    cluster_ids[zero_norm_mask] = -1

    # Rows absent from the graph (degree==0) and not assigned by Leiden remain
    # cluster_id = -1 (run_leiden already initializes to -1 for missing vertices).

    is_noise = cluster_ids == -1
    labels_df = pd.DataFrame(
        {
            "row_id": np.arange(n, dtype=np.int64),
            "cluster_id": cluster_ids,
            "method": "leiden",
            "is_noise": is_noise,
            "embedding_norm": np.asarray(norms, dtype=np.float32),
            "graph_degree": graph_degree,
            "mean_neighbor_similarity": mean_neighbor_sim,
        }
    )
    write_labels_parquet(run_paths, labels_df)
    logger.info("Wrote labels.parquet (%d rows)", n)

    # Cluster summary.
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

    # Metrics.
    unique_cids = np.unique(cluster_ids[cluster_ids >= 0])
    n_clusters = int(unique_cids.size)
    n_unassigned = int(is_noise.sum())

    metrics = {
        "method": "leiden",
        "n_rows": n,
        "n_clusters": n_clusters,
        "n_unassigned": n_unassigned,
        "modularity": round(modularity, 6),
        "k": k,
        "resolution": leiden_cfg.resolution,
        "min_similarity": leiden_cfg.min_similarity,
        "n_mutual_edges": n_mutual_edges,
    }
    write_metrics_json(run_paths, metrics)

    # Run config.
    run_config = {
        "pipeline": "leiden",
        "k": k,
        "resolution": leiden_cfg.resolution,
        "min_similarity": leiden_cfg.min_similarity,
        "normalize": shared_cfg.normalize,
        "batch_size": batch_size,
        "random_state": shared_cfg.random_state,
        "knn_metric": "inner_product" if shared_cfg.normalize else "cosine",
    }
    write_run_config_json(run_paths, run_config)

    if shared_cfg.export_jsonl:
        export_jsonl_with_labels(
            dataset_info.metadata_path,
            run_paths.metadata_with_labels_jsonl,
            cluster_ids,
            method="leiden",
        )

    logger.info(
        "Leiden pipeline done: %d clusters, %d unassigned (%.2f%%), modularity=%.4f",
        n_clusters,
        n_unassigned,
        (n_unassigned / n) * 100 if n else 0.0,
        modularity,
    )
