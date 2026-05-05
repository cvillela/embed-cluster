"""I/O helpers: mmap loader, JSONL streaming, parquet/json writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import orjson
import pandas as pd


def load_embeddings_mmap(path: Path) -> np.memmap:
    """Open the embeddings file read-only via mmap. Never write back."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    arr = np.load(path, mmap_mode="r")
    if not isinstance(arr, np.memmap) and not isinstance(arr, np.ndarray):
        raise TypeError(f"Unexpected object loaded from {path}: {type(arr)}")
    return arr  # type: ignore[return-value]


def count_metadata_lines(path: Path) -> int:
    """Count newline-terminated lines without loading the whole file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def iter_metadata_lines(path: Path) -> Iterator[dict]:
    """Stream metadata.jsonl line by line. Uses orjson for speed."""
    path = Path(path)
    with open(path, "rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            yield orjson.loads(raw)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)


def write_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = orjson.dumps(obj, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with open(path, "rb") as f:
        return json.loads(f.read())
