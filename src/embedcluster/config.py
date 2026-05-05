"""Pydantic configs and dataclasses shared across pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator


BatchSizeSetting = int | Literal["auto"]


class SharedConfig(BaseModel):
    embeddings: Path
    metadata: Path
    out: Path
    normalize: bool = True
    batch_size: BatchSizeSetting = "auto"
    random_state: int = 42
    sample_metrics: int = 50_000
    export_jsonl: bool = False

    @field_validator("embeddings", "metadata", mode="before")
    @classmethod
    def _coerce_path(cls, v):
        return Path(v) if v is not None else v


class LeidenConfig(BaseModel):
    k: int = 50
    resolution: float = 1.0
    min_similarity: float = 0.0


class HdbscanConfig(BaseModel):
    min_cluster_size: int = 50
    min_samples: Optional[int] = None
    pca_components: Optional[int] = 128
    cluster_selection: Literal["eom", "leaf"] = "eom"


class KmeansConfig(BaseModel):
    n_clusters: Optional[int] = None
    target_cluster_size: int = 1000
    max_iter: int = 300
    nredo: int = 1


@dataclass
class DatasetInfo:
    n_rows: int
    n_dims: int
    dtype: str
    embeddings_path: Path
    metadata_path: Path
    zero_norm_mask: np.ndarray = field(repr=False)
    has_nan: bool = False
    has_inf: bool = False

    @property
    def n_zero_norm(self) -> int:
        return int(self.zero_norm_mask.sum())


@dataclass
class RunPaths:
    out: Path
    intermediate: Path
    logs: Path
    run_config_json: Path
    preflight_json: Path
    metrics_json: Path
    labels_parquet: Path
    cluster_summary_parquet: Path
    norms_npy: Path
    normalized_npy: Path
    pca_npy: Path
    pca_normalized_npy: Path
    knn_indices_npy: Path
    knn_scores_npy: Path
    mutual_edges_parquet: Path
    cluster_centers_npy: Path
    hdbscan_persistence_parquet: Path
    metadata_with_labels_jsonl: Path
    run_log: Path

    @classmethod
    def from_out(cls, out: Path) -> "RunPaths":
        out = Path(out)
        intermediate = out / "intermediate"
        logs = out / "logs"
        return cls(
            out=out,
            intermediate=intermediate,
            logs=logs,
            run_config_json=out / "run_config.json",
            preflight_json=out / "preflight.json",
            metrics_json=out / "metrics.json",
            labels_parquet=out / "labels.parquet",
            cluster_summary_parquet=out / "cluster_summary.parquet",
            norms_npy=intermediate / "norms.npy",
            normalized_npy=intermediate / "normalized.npy",
            pca_npy=intermediate / "pca.npy",
            pca_normalized_npy=intermediate / "pca_normalized.npy",
            knn_indices_npy=intermediate / "knn_indices.npy",
            knn_scores_npy=intermediate / "knn_scores.npy",
            mutual_edges_parquet=intermediate / "mutual_edges.parquet",
            cluster_centers_npy=out / "cluster_centers.npy",
            hdbscan_persistence_parquet=out / "hdbscan_cluster_persistence.parquet",
            metadata_with_labels_jsonl=out / "metadata_with_labels.jsonl",
            run_log=logs / "run.log",
        )
