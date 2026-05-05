"""cuGraph Leiden clustering on a mutual weighted kNN edge list."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

logger = get_logger(__name__)


def run_leiden(
    edges_path: Path,
    n: int,
    resolution: float,
    random_state: int,
    max_iter: int = 100,
) -> tuple[pd.DataFrame, float]:
    """Run cuGraph Leiden on the mutual edge list.

    Parameters
    ----------
    edges_path : Path
        Parquet file with columns ``src: int64``, ``dst: int64``, ``weight: float32``.
    n : int
        Total number of rows in the original dataset. Rows that are absent from
        the mutual graph are assigned ``cluster_id = -1``.
    resolution : float
    random_state : int
    max_iter : int

    Returns
    -------
    df : pandas.DataFrame
        Columns: ``row_id`` (int64), ``cluster_id`` (int64).
    modularity : float
    """
    import cudf  # type: ignore
    import cugraph  # type: ignore

    edges = cudf.read_parquet(edges_path)
    n_edges = len(edges)
    logger.info("Loaded %d mutual edges from %s", n_edges, edges_path)

    cluster_ids = np.full(n, -1, dtype=np.int64)

    if n_edges == 0:
        logger.warning("Mutual graph is empty: all rows will be cluster_id = -1.")
        return (
            pd.DataFrame(
                {
                    "row_id": np.arange(n, dtype=np.int64),
                    "cluster_id": cluster_ids,
                }
            ),
            0.0,
        )

    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(
        edges,
        source="src",
        destination="dst",
        edge_attr="weight",
        renumber=True,
    )

    parts, modularity = cugraph.leiden(
        G,
        max_iter=max_iter,
        resolution=resolution,
        random_state=random_state,
    )
    logger.info("Leiden modularity: %.6f", float(modularity))

    # In recent cugraph versions (25.x/26.x) the `vertex` column already
    # contains the *original* vertex IDs (un-renumbered). Try `unrenumber`
    # defensively for older versions; on failure, use parts as-is.
    try:
        parts_unrenum = G.unrenumber(parts, "vertex")
        parts_pdf = parts_unrenum.to_pandas()
        if "vertex" not in parts_pdf.columns:
            parts_pdf = parts_pdf.rename(columns={parts_pdf.columns[0]: "vertex"})
    except Exception as exc:
        logger.info("Using leiden parts directly (unrenumber unnecessary): %s", exc)
        parts_pdf = parts.to_pandas()

    if "vertex" not in parts_pdf.columns or "partition" not in parts_pdf.columns:
        raise RuntimeError(
            f"Unexpected leiden output columns: {parts_pdf.columns.tolist()}"
        )

    vertices = parts_pdf["vertex"].to_numpy().astype(np.int64, copy=False)
    partitions = parts_pdf["partition"].to_numpy().astype(np.int64, copy=False)

    # Remap partition IDs to a contiguous 0..C-1 range.
    unique_parts, remapped = np.unique(partitions, return_inverse=True)
    logger.info("Leiden produced %d clusters", unique_parts.size)

    # Bounds-check vertex IDs.
    if vertices.size > 0:
        v_max = int(vertices.max())
        v_min = int(vertices.min())
        if v_min < 0 or v_max >= n:
            raise RuntimeError(
                f"Leiden returned vertex IDs outside [0, {n}): "
                f"min={v_min}, max={v_max}"
            )

    cluster_ids[vertices] = remapped.astype(np.int64, copy=False)

    df = pd.DataFrame(
        {
            "row_id": np.arange(n, dtype=np.int64),
            "cluster_id": cluster_ids,
        }
    )
    return df, float(modularity)
