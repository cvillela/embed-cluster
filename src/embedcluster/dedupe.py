"""GPU-accelerated near-duplicate detection on raw embeddings.

Range search via chunked matmul on cupy: returns all pairs with
``cosine(X[i], X[j]) >= threshold``. Connected components over those pairs
identifies duplicate groups. Canonical row per group is the smallest ``row_id``.

Output (under ``out``):
    run_config.json   inputs (threshold, N, D, chunk_size)
    metrics.json      summary stats (n_edges, n_groups, n_duplicate_rows)
    dedupe.parquet    one row per embedding: row_id, dup_group_id,
                      group_size, is_canonical
    logs/run.log      structured run log

VRAM budget at chunk_size=2048, N=434k, D=1536:
    X_gpu        ~2.7 GB resident (full matrix on device)
    scores buf   ~3.5 GB per chunk (chunk * N * 4)
    bool mask    ~0.85 GB per chunk
    peak         ~8 GB; safe on 16 GB GPUs.
Bigger GPUs: raise ``--chunk-size`` for fewer chunks (linear speedup).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .io import load_embeddings_mmap, write_json, write_parquet
from .logging_utils import configure_logging, get_logger

logger = get_logger(__name__)


def _normalize_inplace_gpu(X_gpu):
    """L2-normalize on GPU. Zero-norm rows become all zeros."""
    import cupy as cp

    norms = cp.linalg.norm(X_gpu, axis=1, keepdims=True)
    safe = cp.where(norms == 0, 1.0, norms)
    X_gpu /= safe
    zero_mask = (norms == 0).reshape(-1)
    if bool(zero_mask.any()):
        X_gpu[zero_mask] = 0
        logger.info("dedupe: zeroed %d zero-norm rows", int(zero_mask.sum()))
    return X_gpu


def range_search_gpu(
    X: np.ndarray,
    threshold: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find all upper-triangle pairs (i, j), i < j, with cosine >= threshold.

    Returns three 1D arrays: ``i_idx`` (int64), ``j_idx`` (int64), ``score`` (float32).
    """
    import cupy as cp

    n, d = int(X.shape[0]), int(X.shape[1])
    logger.info(
        "range_search_gpu: N=%d D=%d threshold=%.4f chunk_size=%d",
        n,
        d,
        threshold,
        chunk_size,
    )

    X_host = np.ascontiguousarray(X, dtype=np.float32)
    X_gpu = cp.asarray(X_host)
    _normalize_inplace_gpu(X_gpu)
    Xt = X_gpu.T  # (D, N) view, no copy

    i_buf: list[np.ndarray] = []
    j_buf: list[np.ndarray] = []
    s_buf: list[np.ndarray] = []

    n_chunks = (n + chunk_size - 1) // chunk_size
    total_pairs = 0
    for ci, start in enumerate(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        Q = X_gpu[start:end]                       # (c, D) view
        scores = Q @ Xt                             # (c, N) float32
        mask = scores >= threshold                  # (c, N) bool
        flat = cp.flatnonzero(mask)
        del mask
        if flat.size == 0:
            del scores, flat
            if ci % 10 == 0:
                logger.info(
                    "chunk %d/%d rows %d-%d: 0 pairs",
                    ci + 1, n_chunks, start, end,
                )
            continue
        ii_local = (flat // n).astype(cp.int64)
        jj = (flat % n).astype(cp.int64)
        global_i = ii_local + start
        keep = jj > global_i
        ii_local = ii_local[keep]
        jj = jj[keep]
        global_i = global_i[keep]
        ss = scores.ravel()[flat[keep]]
        del flat, scores, keep

        n_pairs = int(global_i.size)
        total_pairs += n_pairs
        if n_pairs > 0:
            i_buf.append(cp.asnumpy(global_i))
            j_buf.append(cp.asnumpy(jj))
            s_buf.append(cp.asnumpy(ss).astype(np.float32, copy=False))
        if ci % 10 == 0:
            logger.info(
                "chunk %d/%d rows %d-%d: %d pairs (cum=%d)",
                ci + 1, n_chunks, start, end, n_pairs, total_pairs,
            )

    del X_gpu, Xt
    cp.get_default_memory_pool().free_all_blocks()

    if not i_buf:
        return (
            np.empty(0, np.int64),
            np.empty(0, np.int64),
            np.empty(0, np.float32),
        )
    return (
        np.concatenate(i_buf),
        np.concatenate(j_buf),
        np.concatenate(s_buf),
    )


def connected_component_labels(
    n: int,
    i_arr: np.ndarray,
    j_arr: np.ndarray,
) -> np.ndarray:
    """Run scipy CC over the duplicate edge set. Returns labels[n] (int64)."""
    if i_arr.size == 0:
        return np.arange(n, dtype=np.int64)
    data = np.ones(i_arr.size, dtype=np.int8)
    g = csr_matrix((data, (i_arr, j_arr)), shape=(n, n))
    n_comp, labels = connected_components(g, directed=False, return_labels=True)
    logger.info("connected_components: %d components (incl singletons)", n_comp)
    return labels.astype(np.int64, copy=False)


def build_manifest(n: int, comp_labels: np.ndarray) -> pd.DataFrame:
    """Build dedupe manifest: row_id, dup_group_id, group_size, is_canonical."""
    df = pd.DataFrame(
        {
            "row_id": np.arange(n, dtype=np.int64),
            "dup_group_id": comp_labels.astype(np.int64, copy=False),
        }
    )
    sizes = df.groupby("dup_group_id").size().rename("group_size")
    df = df.merge(sizes, on="dup_group_id", how="left")
    canonical = df.groupby("dup_group_id")["row_id"].min().rename("_min_row")
    df = df.merge(canonical, on="dup_group_id", how="left")
    df["is_canonical"] = df["row_id"] == df["_min_row"]
    df = df.drop(columns=["_min_row"])
    return df[["row_id", "dup_group_id", "group_size", "is_canonical"]]


def run_dedupe(
    embeddings_path: Path,
    out: Path,
    threshold: float,
    chunk_size: int,
) -> None:
    """End-to-end dedupe: range search -> CC -> manifest."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=out / "logs" / "run.log")

    logger.info(
        "dedupe start: embeddings=%s out=%s threshold=%.4f chunk_size=%d",
        embeddings_path,
        out,
        threshold,
        chunk_size,
    )

    X = load_embeddings_mmap(Path(embeddings_path))
    if X.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {X.shape!r}")
    n, d = int(X.shape[0]), int(X.shape[1])

    i_arr, j_arr, s_arr = range_search_gpu(
        X, threshold=threshold, chunk_size=chunk_size
    )
    n_edges = int(i_arr.size)
    logger.info("range search done: %d pairs", n_edges)

    comp_labels = connected_component_labels(n, i_arr, j_arr)
    manifest = build_manifest(n, comp_labels)

    n_groups = int(manifest["dup_group_id"].nunique())
    n_dup_rows = int((manifest["group_size"] > 1).sum())
    n_canonical = int(manifest["is_canonical"].sum())
    n_removable = n - n_canonical
    n_multi_groups = int(
        (manifest.groupby("dup_group_id")["group_size"].first() > 1).sum()
    )
    logger.info(
        "dedupe summary: n=%d edges=%d groups=%d (multi=%d) "
        "duplicate_rows=%d removable=%d",
        n,
        n_edges,
        n_groups,
        n_multi_groups,
        n_dup_rows,
        n_removable,
    )

    write_parquet(manifest, out / "dedupe.parquet")
    write_json(
        {
            "pipeline": "dedupe",
            "embeddings": str(embeddings_path),
            "threshold": threshold,
            "chunk_size": chunk_size,
            "n_rows": n,
            "n_dims": d,
        },
        out / "run_config.json",
    )
    write_json(
        {
            "method": "dedupe",
            "n_rows": n,
            "threshold": threshold,
            "n_edges": n_edges,
            "n_groups": n_groups,
            "n_multi_member_groups": n_multi_groups,
            "n_duplicate_rows": n_dup_rows,
            "n_canonical_rows": n_canonical,
            "n_removable_rows": n_removable,
        },
        out / "metrics.json",
    )

    logger.info(
        "dedupe done: wrote %s, %s, %s",
        out / "dedupe.parquet",
        out / "run_config.json",
        out / "metrics.json",
    )
