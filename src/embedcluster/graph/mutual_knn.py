"""Build a mutual weighted kNN graph from cuVS kNN outputs.

The directed edge stream is partitioned by ``min(i, j) % num_partitions`` so each
partition fits in GPU memory. Each partition is processed independently with
cuDF: directed edges are aggregated by undirected key ``(min, max)`` and only
mutual edges (count == 2) are kept. The final per-edge weight is the mean of
the two directional similarity scores.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..logging_utils import get_logger

logger = get_logger(__name__)


def _choose_num_partitions(n: int, k: int) -> int:
    """Target ~10M directed edges per partition; minimum of 4 partitions."""
    return max(4, (n * k) // 10_000_000)


def _emit_directed_edges_to_partitions(
    indices: np.ndarray,
    scores: np.ndarray,
    chunk_size: int,
    num_partitions: int,
    tmp_dir: Path,
) -> list[Path]:
    """Stream directed edges (i, j, w) into partitioned parquet files.

    Partitioning key: ``min(i, j) % num_partitions`` so both directions of a
    given undirected edge land in the same partition.

    Self-loops and rows with invalid neighbor IDs (-1) are dropped here.
    """
    n, k = indices.shape
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    writers: dict[int, pq.ParquetWriter] = {}
    partition_paths: dict[int, Path] = {}
    schema = pa.schema(
        [
            ("src", pa.int64()),
            ("dst", pa.int64()),
            ("weight", pa.float32()),
            ("a", pa.int64()),
            ("b", pa.int64()),
        ]
    )

    try:
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_idx = np.asarray(indices[start:end], dtype=np.int64)
            chunk_sc = np.asarray(scores[start:end], dtype=np.float32)
            row_ids = np.arange(start, end, dtype=np.int64)
            src = np.repeat(row_ids, k)
            dst = chunk_idx.reshape(-1)
            w = chunk_sc.reshape(-1)

            # Drop self-loops and invalid neighbors (-1, NaN).
            valid = (dst >= 0) & (src != dst) & np.isfinite(w)
            src = src[valid]
            dst = dst[valid]
            w = w[valid]

            if src.size == 0:
                continue

            a = np.minimum(src, dst)
            b = np.maximum(src, dst)
            partition = (a % num_partitions).astype(np.int64)

            for p in np.unique(partition):
                p_int = int(p)
                mask = partition == p_int
                table = pa.table(
                    {
                        "src": pa.array(src[mask], type=pa.int64()),
                        "dst": pa.array(dst[mask], type=pa.int64()),
                        "weight": pa.array(w[mask], type=pa.float32()),
                        "a": pa.array(a[mask], type=pa.int64()),
                        "b": pa.array(b[mask], type=pa.int64()),
                    },
                    schema=schema,
                )
                if p_int not in writers:
                    p_path = tmp_dir / f"partition_{p_int:05d}.parquet"
                    writers[p_int] = pq.ParquetWriter(p_path, schema)
                    partition_paths[p_int] = p_path
                writers[p_int].write_table(table)

            if (start // chunk_size) % 10 == 0:
                logger.info(
                    "mutual_knn: partitioned chunk rows %d-%d (%d partitions active)",
                    start,
                    end,
                    len(writers),
                )
    finally:
        for w_ in writers.values():
            w_.close()

    return [partition_paths[p] for p in sorted(partition_paths)]


def _process_partition_cudf(part_path: Path, min_similarity: float) -> "pd.DataFrame":
    """Process one partition with cuDF: keep mutual edges, average weights."""
    import cudf  # type: ignore

    df = cudf.read_parquet(part_path)
    if len(df) == 0:
        return pd.DataFrame({"src": [], "dst": [], "weight": []})

    # Group by undirected key (a, b).
    grouped = df.groupby(["a", "b"]).agg({"weight": ["mean", "count"]}).reset_index()
    grouped.columns = ["a", "b", "weight_mean", "edge_count"]

    mutual = grouped[grouped["edge_count"] == 2]
    if min_similarity > -np.inf:
        mutual = mutual[mutual["weight_mean"] >= min_similarity]

    out_pdf = mutual.to_pandas()[["a", "b", "weight_mean"]].rename(
        columns={"a": "src", "b": "dst", "weight_mean": "weight"}
    )
    out_pdf["weight"] = out_pdf["weight"].astype("float32")
    out_pdf["src"] = out_pdf["src"].astype("int64")
    out_pdf["dst"] = out_pdf["dst"].astype("int64")
    return out_pdf


def build_mutual_graph(
    indices_path: Path,
    scores_path: Path,
    n: int,
    k: int,
    min_similarity: float,
    out_path: Path,
    tmp_dir: Path,
    num_partitions: int | None = None,
    chunk_size: int = 65_536,
) -> Path:
    """Build the mutual weighted kNN graph and write ``mutual_edges.parquet``.

    Returns
    -------
    out_path : Path
        Path to the written parquet file.
    """
    indices = np.memmap(indices_path, dtype=np.int64, mode="r", shape=(n, k))
    scores = np.memmap(scores_path, dtype=np.float32, mode="r", shape=(n, k))

    if num_partitions is None:
        num_partitions = _choose_num_partitions(n, k)
    logger.info(
        "Building mutual kNN graph: N=%d, k=%d, num_partitions=%d, min_similarity=%.4f",
        n,
        k,
        num_partitions,
        min_similarity,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tmp_dir)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    partition_paths = _emit_directed_edges_to_partitions(
        indices=indices,
        scores=scores,
        chunk_size=chunk_size,
        num_partitions=num_partitions,
        tmp_dir=tmp_dir,
    )
    logger.info("Emitted %d partition files.", len(partition_paths))

    final_frames: list[pd.DataFrame] = []
    for i, p_path in enumerate(partition_paths):
        part_df = _process_partition_cudf(p_path, min_similarity=min_similarity)
        final_frames.append(part_df)
        logger.info(
            "Partition %d/%d: %d mutual edges (path=%s)",
            i + 1,
            len(partition_paths),
            len(part_df),
            p_path.name,
        )

    if final_frames:
        final = pd.concat(final_frames, ignore_index=True)
    else:
        final = pd.DataFrame(
            {
                "src": np.array([], dtype=np.int64),
                "dst": np.array([], dtype=np.int64),
                "weight": np.array([], dtype=np.float32),
            }
        )

    final.to_parquet(out_path, engine="pyarrow", index=False)
    logger.info("Wrote %s with %d mutual edges.", out_path, len(final))

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_path
