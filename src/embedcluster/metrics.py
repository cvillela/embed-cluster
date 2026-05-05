"""Standalone metrics and cluster-summary utilities."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


_KNOWN_LABEL_FIELDS = frozenset(
    {"species", "label", "scientific_name", "common_name", "class", "category"}
)


def compute_global_metrics(
    cluster_ids: np.ndarray,
    norms: np.ndarray,
    method: str,
    pipeline_cfg: dict,
) -> dict:
    """Return global metrics dict (n_rows, n_clusters, size distribution, norms, ...).

    pipeline_cfg is merged in as extra method-specific fields.
    """
    n = len(cluster_ids)
    valid_mask = cluster_ids >= 0
    noise_mask = ~valid_mask

    n_noise = int(noise_mask.sum())
    noise_fraction = n_noise / n if n > 0 else 0.0

    unique_cids = np.unique(cluster_ids[valid_mask])
    n_clusters = int(len(unique_cids))

    size_stats: dict = {}
    if n_clusters > 0:
        sizes = (
            pd.Series(cluster_ids[valid_mask])
            .value_counts()
            .values.astype(np.int64)
        )
        size_stats = {
            "cluster_size_min": int(sizes.min()),
            "cluster_size_p25": float(np.percentile(sizes, 25)),
            "cluster_size_median": float(np.median(sizes)),
            "cluster_size_p75": float(np.percentile(sizes, 75)),
            "cluster_size_max": int(sizes.max()),
        }

    norms_f = np.asarray(norms, dtype=np.float32)
    result: dict = {
        "method": method,
        "n_rows": n,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_fraction": round(float(noise_fraction), 6),
        "embedding_norm_mean": round(float(np.mean(norms_f)), 6) if len(norms_f) > 0 else 0.0,
        "embedding_norm_std": round(float(np.std(norms_f)), 6) if len(norms_f) > 0 else 0.0,
        **size_stats,
    }
    result.update(pipeline_cfg)
    return result


def compute_cluster_summary(
    labels_df: pd.DataFrame,
    norms: np.ndarray,
    metadata_iter: Iterator[dict] | None = None,
) -> pd.DataFrame:
    """Per-cluster summary with optional top-label columns scanned from metadata.

    If metadata_iter is provided and a record contains a known label field
    (species / label / scientific_name / common_name / class / category),
    appends top_label_field, top_label_value, top_label_count, top_label_fraction.
    """
    valid = labels_df[~labels_df["is_noise"]]

    if len(valid) == 0:
        return pd.DataFrame(
            columns=["cluster_id", "size", "embedding_norm_mean", "embedding_norm_std"]
        )

    summary = (
        valid.groupby("cluster_id")
        .agg(
            size=("row_id", "count"),
            embedding_norm_mean=("embedding_norm", "mean"),
            embedding_norm_std=("embedding_norm", "std"),
        )
        .reset_index()
    )

    if metadata_iter is None:
        return summary

    row_to_cluster: dict[int, int] = dict(
        zip(valid["row_id"].tolist(), valid["cluster_id"].tolist())
    )
    label_field: str | None = None
    cluster_label_counts: dict[int, dict[str, int]] = {}

    for row_id, record in enumerate(metadata_iter):
        if label_field is None:
            for fld in _KNOWN_LABEL_FIELDS:
                if fld in record:
                    label_field = fld
                    break

        cid = row_to_cluster.get(row_id)
        if cid is None or label_field is None:
            continue
        label_val = record.get(label_field)
        if label_val is None:
            continue
        label_str = str(label_val)
        counts = cluster_label_counts.setdefault(cid, {})
        counts[label_str] = counts.get(label_str, 0) + 1

    if label_field is None or not cluster_label_counts:
        return summary

    top_rows = []
    for _, row in summary.iterrows():
        cid = int(row["cluster_id"])
        counts = cluster_label_counts.get(cid, {})
        if counts:
            top_label = max(counts, key=lambda k: counts[k])
            top_count = counts[top_label]
            cluster_size = int(row["size"])
            top_rows.append(
                {
                    "cluster_id": cid,
                    "top_label_field": label_field,
                    "top_label_value": top_label,
                    "top_label_count": top_count,
                    "top_label_fraction": round(top_count / cluster_size, 6),
                }
            )
        else:
            top_rows.append(
                {
                    "cluster_id": cid,
                    "top_label_field": label_field,
                    "top_label_value": None,
                    "top_label_count": 0,
                    "top_label_fraction": 0.0,
                }
            )

    top_df = pd.DataFrame(top_rows)
    return summary.merge(top_df, on="cluster_id", how="left")


def compute_sampled_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    sample_n: int = 50_000,
    random_state: int = 42,
) -> float | None:
    """Silhouette score on a random sample. Returns None if skipped.

    Skipped when: sklearn unavailable, n_clusters < 2, sample_n < 2 * n_clusters,
    or any exception during scoring.
    """
    try:
        from sklearn.metrics import silhouette_score  # type: ignore
    except ImportError:
        return None

    valid_mask = labels >= 0
    X_valid = np.asarray(X[valid_mask], dtype=np.float32)
    labels_valid = labels[valid_mask]

    unique_cids = np.unique(labels_valid)
    n_clusters = len(unique_cids)
    if n_clusters < 2 or sample_n < 2 * n_clusters:
        return None

    n = len(X_valid)
    if n <= sample_n:
        X_sample = X_valid
        labels_sample = labels_valid
    else:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=sample_n, replace=False)
        idx.sort()
        X_sample = X_valid[idx]
        labels_sample = labels_valid[idx]

    if len(np.unique(labels_sample)) < 2:
        return None

    try:
        return float(silhouette_score(X_sample, labels_sample, metric="cosine"))
    except Exception:
        return None
