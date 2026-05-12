"""Discover and load embedcluster run artifacts (read-only)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

REQUIRED_FILES = ("run_config.json", "labels.parquet")


@dataclass(frozen=True)
class RunSummary:
    name: str
    path: Path
    method: str
    n_rows: int
    n_clusters: int
    mtime: float


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _is_run_dir(p: Path) -> bool:
    return p.is_dir() and all((p / f).exists() for f in REQUIRED_FILES)


def discover_runs(runs_root: Path) -> list[RunSummary]:
    """List runs under ``runs_root``, sorted by mtime desc.

    A run is any subdir containing both ``run_config.json`` and ``labels.parquet``.
    Reads only small JSONs — does not touch parquet or .npy.
    """
    runs_root = Path(runs_root)
    if not runs_root.exists() or not runs_root.is_dir():
        return []
    summaries: list[RunSummary] = []
    for child in sorted(runs_root.iterdir()):
        if not _is_run_dir(child):
            continue
        cfg_path = child / "run_config.json"
        metrics_path = child / "metrics.json"
        try:
            cfg = _read_json(cfg_path)
            metrics = _read_json(metrics_path) if metrics_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            continue
        method = metrics.get("method") or cfg.get("pipeline") or "unknown"
        n_rows = int(metrics.get("n_rows", 0))
        n_clusters = int(metrics.get("n_clusters", 0))
        summaries.append(
            RunSummary(
                name=child.name,
                path=child,
                method=method,
                n_rows=n_rows,
                n_clusters=n_clusters,
                mtime=cfg_path.stat().st_mtime,
            )
        )
    summaries.sort(key=lambda r: r.mtime, reverse=True)
    return summaries


class RunBundle:
    """Eagerly loads small artifacts; lazily loads parquet / npy on access."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.run_config: dict[str, Any] = _read_json(self.path / "run_config.json")
        preflight_path = self.path / "preflight.json"
        metrics_path = self.path / "metrics.json"
        self.preflight: dict[str, Any] = _read_json(preflight_path) if preflight_path.exists() else {}
        self.metrics: dict[str, Any] = _read_json(metrics_path) if metrics_path.exists() else {}
        cs_path = self.path / "cluster_summary.parquet"
        self.cluster_summary: pd.DataFrame = (
            pd.read_parquet(cs_path) if cs_path.exists() else pd.DataFrame()
        )
        self._labels: pd.DataFrame | None = None
        self._hdbscan_persistence: pd.DataFrame | None = None
        self._kmeans_centers: np.ndarray | None = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def method(self) -> str:
        return self.metrics.get("method") or self.run_config.get("pipeline") or "unknown"

    @property
    def labels(self) -> pd.DataFrame:
        if self._labels is None:
            self._labels = pd.read_parquet(self.path / "labels.parquet")
        return self._labels

    @property
    def hdbscan_persistence(self) -> pd.DataFrame | None:
        if self._hdbscan_persistence is None:
            p = self.path / "hdbscan_cluster_persistence.parquet"
            if not p.exists():
                return None
            self._hdbscan_persistence = pd.read_parquet(p)
        return self._hdbscan_persistence

    @property
    def kmeans_centers(self) -> np.ndarray | None:
        if self._kmeans_centers is None:
            p = self.path / "cluster_centers.npy"
            if not p.exists():
                return None
            self._kmeans_centers = np.load(p, mmap_mode="r")
        return self._kmeans_centers


@st.cache_resource(show_spinner=False)
def _load_run_cached(path_str: str, mtime: float) -> RunBundle:
    return RunBundle(Path(path_str))


def load_run(path: Path) -> RunBundle:
    """Load a run bundle, cached by (path, run_config.json mtime)."""
    path = Path(path)
    mtime = (path / "run_config.json").stat().st_mtime
    return _load_run_cached(str(path), mtime)


def clear_cache() -> None:
    _load_run_cached.clear()
