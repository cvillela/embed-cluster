"""cuVS GPU brute-force kNN search with self-neighbor removal.

Outputs are written to disk as float32 / int64 memmaps so they can be reused
across runs (e.g., re-running Leiden with a different `resolution`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..logging_utils import get_logger

logger = get_logger(__name__)


def _existing_files_match(
    indices_path: Path,
    scores_path: Path,
    n: int,
    k: int,
) -> bool:
    """Return True if both files exist and have the expected shape/dtype."""
    if not (indices_path.exists() and scores_path.exists()):
        return False
    try:
        idx_mm = np.memmap(indices_path, dtype=np.int64, mode="r")
        sc_mm = np.memmap(scores_path, dtype=np.float32, mode="r")
    except Exception:
        return False
    return idx_mm.size == n * k and sc_mm.size == n * k


def build_and_search_knn(
    X: np.ndarray,
    k: int,
    normalize: bool,
    batch_size: int,
    indices_out: Path,
    scores_out: Path,
) -> tuple[np.memmap, np.memmap]:
    """Build a cuVS brute-force index and search top-k neighbors per row.

    Parameters
    ----------
    X : np.ndarray | np.memmap
        Working matrix of shape [N, D]. Should already be L2-normalized when
        ``normalize=True``.
    k : int
        Number of neighbors to keep per row, after self-neighbor removal.
    normalize : bool
        If True, use ``inner_product`` (cosine similarity for unit vectors).
        Otherwise, use ``cosine`` distance and convert to similarity.
    batch_size : int
        Number of query rows per cuVS search call.
    indices_out, scores_out : Path
        Destination paths for the kNN arrays. If both already exist with the
        expected shape, they are reused and the search is skipped.

    Returns
    -------
    indices : np.memmap[int64], shape [N, k]
    scores  : np.memmap[float32], shape [N, k]   (cosine similarity)
    """
    indices_out = Path(indices_out)
    scores_out = Path(scores_out)
    indices_out.parent.mkdir(parents=True, exist_ok=True)
    scores_out.parent.mkdir(parents=True, exist_ok=True)

    n = int(X.shape[0])
    d = int(X.shape[1])

    if _existing_files_match(indices_out, scores_out, n, k):
        logger.info(
            "Reusing existing kNN arrays at %s / %s (N=%d, k=%d).",
            indices_out,
            scores_out,
            n,
            k,
        )
        return (
            np.memmap(indices_out, dtype=np.int64, mode="r", shape=(n, k)),
            np.memmap(scores_out, dtype=np.float32, mode="r", shape=(n, k)),
        )

    metric = "inner_product" if normalize else "cosine"
    logger.info(
        "cuVS brute-force kNN: N=%d, D=%d, k=%d, metric=%s, batch_size=%d",
        n,
        d,
        k,
        metric,
        batch_size,
    )

    import cupy as cp  # type: ignore
    from cuvs.neighbors import brute_force  # type: ignore

    X_gpu = cp.asarray(np.asarray(X, dtype=np.float32))
    index = brute_force.build(X_gpu, metric=metric)

    indices_mm = np.memmap(indices_out, dtype=np.int64, mode="w+", shape=(n, k))
    scores_mm = np.memmap(scores_out, dtype=np.float32, mode="w+", shape=(n, k))

    k_search = k + 1
    n_chunks = max(1, (n + batch_size - 1) // batch_size)

    for chunk_i, start in enumerate(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        q_gpu = X_gpu[start:end]
        dists_gpu, neigh_gpu = brute_force.search(index, q_gpu, k_search)
        neigh = cp.asarray(neigh_gpu).get().astype(np.int64, copy=False)
        dists = cp.asarray(dists_gpu).get().astype(np.float32, copy=False)

        # Convert distances to similarity scores.
        if metric == "inner_product":
            sims = dists
        else:  # cosine distance: similarity = 1 - distance
            sims = (1.0 - dists).astype(np.float32, copy=False)

        # Remove self-neighbor per row. Build a mask of "keep" columns.
        row_ids = np.arange(start, end, dtype=np.int64)[:, None]  # [B, 1]
        is_self = neigh == row_ids  # [B, k+1]

        # Default keep mask: keep first k columns.
        # If a row contains itself in its first k columns, keep cols where
        # is_self is False, then take the first k.
        out_neigh = np.empty((end - start, k), dtype=np.int64)
        out_sims = np.empty((end - start, k), dtype=np.float32)

        any_self = is_self.any(axis=1)
        # Fast path: rows where self is at column 0 (typical for normalized data).
        self_at_zero = is_self[:, 0]
        out_neigh[self_at_zero] = neigh[self_at_zero, 1 : k + 1]
        out_sims[self_at_zero] = sims[self_at_zero, 1 : k + 1]

        # Rows without self in the top-(k+1): just drop the last column.
        no_self = ~any_self
        out_neigh[no_self] = neigh[no_self, :k]
        out_sims[no_self] = sims[no_self, :k]

        # Remaining rows: self exists but not at column 0. Filter per row.
        slow = any_self & ~self_at_zero
        if slow.any():
            slow_idx = np.flatnonzero(slow)
            for r in slow_idx:
                keep = ~is_self[r]
                # Take first k of the kept columns.
                kept_neigh = neigh[r][keep][:k]
                kept_sims = sims[r][keep][:k]
                if kept_neigh.size < k:
                    # Extremely rare: pad with -1 / NaN.
                    pad = k - kept_neigh.size
                    kept_neigh = np.concatenate(
                        [kept_neigh, np.full(pad, -1, dtype=np.int64)]
                    )
                    kept_sims = np.concatenate(
                        [kept_sims, np.full(pad, np.nan, dtype=np.float32)]
                    )
                out_neigh[r] = kept_neigh
                out_sims[r] = kept_sims

        indices_mm[start:end] = out_neigh
        scores_mm[start:end] = out_sims

        if chunk_i % 10 == 0:
            logger.info("cuvs_knn: chunk %d/%d", chunk_i + 1, n_chunks)

    indices_mm.flush()
    scores_mm.flush()
    del index
    del X_gpu
    cp.get_default_memory_pool().free_all_blocks()

    logger.info(
        "cuVS kNN done: wrote %s and %s (shape [%d, %d]).",
        indices_out,
        scores_out,
        n,
        k,
    )
    return (
        np.memmap(indices_out, dtype=np.int64, mode="r", shape=(n, k)),
        np.memmap(scores_out, dtype=np.float32, mode="r", shape=(n, k)),
    )
