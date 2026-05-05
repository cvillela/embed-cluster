"""Run-directory creation and structured output writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import orjson
import pandas as pd

from .config import RunPaths
from .io import iter_metadata_lines, write_json, write_parquet
from .logging_utils import get_logger

logger = get_logger(__name__)


def create_run_dirs(out: Path) -> RunPaths:
    """Create the canonical run directory layout."""
    paths = RunPaths.from_out(Path(out))
    paths.out.mkdir(parents=True, exist_ok=True)
    paths.intermediate.mkdir(parents=True, exist_ok=True)
    paths.logs.mkdir(parents=True, exist_ok=True)
    return paths


def write_preflight_json(paths: RunPaths, data: dict) -> None:
    write_json(data, paths.preflight_json)


def write_run_config_json(paths: RunPaths, data: dict) -> None:
    write_json(data, paths.run_config_json)


def write_metrics_json(paths: RunPaths, data: dict) -> None:
    write_json(data, paths.metrics_json)


def write_labels_parquet(paths: RunPaths, df: pd.DataFrame) -> None:
    write_parquet(df, paths.labels_parquet)


def write_cluster_summary(paths: RunPaths, df: pd.DataFrame) -> None:
    write_parquet(df, paths.cluster_summary_parquet)


def write_hdbscan_persistence(paths: RunPaths, df: pd.DataFrame) -> None:
    write_parquet(df, paths.hdbscan_persistence_parquet)


def export_jsonl_with_labels(
    metadata_path: Path,
    out_path: Path,
    cluster_ids: np.ndarray,
    method: str,
    extra_columns: dict[str, np.ndarray] | None = None,
) -> None:
    """Stream metadata.jsonl and append clustering labels per row.

    Never loads the full metadata file into memory.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    extra_columns = extra_columns or {}
    n = len(cluster_ids)

    with open(out_path, "wb") as fout:
        for row_id, record in enumerate(iter_metadata_lines(metadata_path)):
            if row_id >= n:
                raise RuntimeError(
                    f"Metadata has more lines than cluster labels (cluster_ids has "
                    f"{n}, metadata reached row {row_id})."
                )
            record["row_id"] = row_id
            record["cluster_id"] = int(cluster_ids[row_id])
            record["clustering_method"] = method
            for col, arr in extra_columns.items():
                v = arr[row_id]
                # Convert numpy scalars / NaN to JSON-friendly types.
                if isinstance(v, (np.floating,)):
                    fv = float(v)
                    record[col] = None if np.isnan(fv) else fv
                elif isinstance(v, (np.integer,)):
                    record[col] = int(v)
                elif isinstance(v, (np.bool_,)):
                    record[col] = bool(v)
                else:
                    record[col] = v
            fout.write(orjson.dumps(record))
            fout.write(b"\n")
        if row_id + 1 != n:
            raise RuntimeError(
                f"Metadata line count ({row_id + 1}) does not match cluster label "
                f"count ({n})."
            )
