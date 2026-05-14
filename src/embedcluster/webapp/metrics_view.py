"""Method-aware metrics + cluster aggregations.

Exposes:
- ``aggregate_cluster_table(bundle)``: per-cluster quality table (cached).
- ``render(bundle)``: lean overview — counts + cluster-size histogram only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from embedcluster.webapp.run_loader import RunBundle


# --------------------------------------------------------------------------- #
# Per-method aggregations
# --------------------------------------------------------------------------- #

def _agg_kmeans(labels: pd.DataFrame) -> pd.DataFrame:
    g = labels.groupby("cluster_id", sort=True)
    return pd.DataFrame(
        {
            "size": g.size(),
            "cosine_to_centroid_mean": g["cosine_to_centroid"].mean(),
            "cosine_to_centroid_p10": g["cosine_to_centroid"].quantile(0.10),
            "embedding_norm_mean": g["embedding_norm"].mean(),
        }
    ).reset_index()


def _agg_hdbscan(
    labels: pd.DataFrame, persistence_df: pd.DataFrame | None
) -> pd.DataFrame:
    g = labels.groupby("cluster_id", sort=True)
    out = pd.DataFrame(
        {
            "size": g.size(),
            "probability_mean": g["probability"].mean(),
            "embedding_norm_mean": g["embedding_norm"].mean(),
        }
    ).reset_index()
    if persistence_df is not None:
        out = out.merge(
            persistence_df[["cluster_id", "persistence"]], on="cluster_id", how="left"
        )
    else:
        out["persistence"] = np.nan
    return out[
        ["cluster_id", "size", "persistence", "probability_mean", "embedding_norm_mean"]
    ]


def _agg_leiden(labels: pd.DataFrame) -> pd.DataFrame:
    g = labels.groupby("cluster_id", sort=True)
    return pd.DataFrame(
        {
            "size": g.size(),
            "mean_neighbor_similarity_mean": g["mean_neighbor_similarity"].mean(),
            "graph_degree_mean": g["graph_degree"].mean(),
            "embedding_norm_mean": g["embedding_norm"].mean(),
        }
    ).reset_index()


@st.cache_data(show_spinner=False)
def _aggregate_cached(
    path_str: str, mtime: float, method: str, _labels: pd.DataFrame, _persistence
) -> pd.DataFrame:
    if method == "kmeans":
        return _agg_kmeans(_labels)
    if method == "hdbscan":
        return _agg_hdbscan(_labels, _persistence)
    if method == "leiden":
        return _agg_leiden(_labels)
    raise ValueError(f"unknown method: {method}")


def aggregate_cluster_table(bundle: RunBundle) -> pd.DataFrame:
    """Method-aware per-cluster quality table. Cached on (path, labels mtime)."""
    labels_path = bundle.path / "labels.parquet"
    mtime = labels_path.stat().st_mtime
    persistence = bundle.hdbscan_persistence if bundle.method == "hdbscan" else None
    return _aggregate_cached(
        str(bundle.path), mtime, bundle.method, bundle.labels, persistence
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def render(bundle: RunBundle) -> pd.DataFrame:
    """Render the lean overview block. Returns the cluster table."""
    table = aggregate_cluster_table(bundle)

    metrics = bundle.metrics
    n_rows = int(metrics.get("n_rows", int(table["size"].sum())))
    n_clusters_real = int((table["cluster_id"] >= 0).sum())
    if bundle.method == "hdbscan":
        n_noise = int(metrics.get("n_noise", 0))
        noise_label = "Noise rows"
    elif bundle.method == "leiden":
        n_noise = int(metrics.get("n_unassigned", 0))
        noise_label = "Unassigned rows"
    else:
        n_noise = 0
        noise_label = "Noise rows"

    c1, c2, c3 = st.columns(3)
    c1.metric("N rows", f"{n_rows:,}")
    c2.metric("N clusters", f"{n_clusters_real:,}")
    if bundle.method in {"hdbscan", "leiden"} and n_rows:
        c3.metric(noise_label, f"{n_noise:,} ({n_noise / n_rows:.1%})")
    else:
        c3.metric(noise_label, "0")

    sizes = table.loc[table["cluster_id"] >= 0, "size"].astype(int)
    if len(sizes):
        log_y = st.toggle("Log-y", value=True, key="size_hist_log_y")
        fig = px.histogram(sizes, nbins=min(60, max(10, len(sizes) // 4)))
        fig.update_layout(
            xaxis_title="cluster size",
            yaxis_title="# clusters",
            yaxis_type="log" if log_y else "linear",
            showlegend=False,
            height=260,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, width="stretch")

    return table
