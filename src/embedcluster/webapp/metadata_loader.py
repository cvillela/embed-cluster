"""Row-aligned ``metadata.jsonl`` loader + audio path resolution."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

_AUDIO_EXTS: tuple[str, ...] = (
    ".wav", ".mp3", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".aiff", ".aif",
)


@st.cache_resource(show_spinner="Loading metadata.jsonl…")
def _load_metadata_cached(path_str: str, mtime: float) -> pd.DataFrame:
    df = pd.read_json(path_str, lines=True)
    if "row_id" in df.columns:
        df = df.set_index("row_id", drop=False)
    else:
        df.index = pd.RangeIndex(len(df), name="row_id")
        df["row_id"] = df.index
    return df


def load_metadata(path: Path) -> pd.DataFrame:
    """Load metadata.jsonl as a row-id-indexed DataFrame, cached on (path, mtime)."""
    p = Path(path)
    return _load_metadata_cached(str(p), p.stat().st_mtime)


def _looks_like_audio(value: object) -> bool:
    return isinstance(value, str) and value.lower().endswith(_AUDIO_EXTS)


def detect_audio_fields(df: pd.DataFrame) -> list[str]:
    """Return string columns whose first non-null value ends with an audio extension."""
    out: list[str] = []
    for col in df.columns:
        if col == "row_id":
            continue
        if df[col].dtype != object:
            continue
        sample = df[col].dropna()
        if sample.empty:
            continue
        if _looks_like_audio(sample.iloc[0]):
            out.append(col)
    return out
