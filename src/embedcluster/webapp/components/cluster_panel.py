"""Per-cluster drill-in panel: lean summary chips + audio (Phase 3)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from embedcluster.webapp.run_loader import RunBundle


def _summary_chip(method: str, row: pd.Series) -> None:
    chips: list[tuple[str, str]] = [
        ("cluster_id", str(int(row["cluster_id"]))),
        ("size", f"{int(row['size']):,}"),
    ]
    if method == "kmeans":
        chips.append(("cosine_mean", f"{row['cosine_to_centroid_mean']:.4f}"))
    elif method == "hdbscan":
        if pd.notna(row.get("persistence")):
            chips.append(("persistence", f"{row['persistence']:.4f}"))
        chips.append(("prob_mean", f"{row['probability_mean']:.4f}"))
    elif method == "leiden":
        chips.append(("nbr_sim_mean", f"{row['mean_neighbor_similarity_mean']:.4f}"))
    chips.append(("emb_norm_mean", f"{row['embedding_norm_mean']:.3f}"))

    cols = st.columns(len(chips))
    for col, (label, value) in zip(cols, chips):
        col.metric(label, value)


def render(bundle: RunBundle, table: pd.DataFrame, cluster_id: int | None) -> None:
    st.subheader("Cluster detail")
    if cluster_id is None:
        st.info("Select a cluster from the table to inspect it.")
        return

    row_match = table[table["cluster_id"] == cluster_id]
    if row_match.empty:
        st.warning(f"cluster_id {cluster_id} not found in the cluster table.")
        return
    row = row_match.iloc[0]

    if cluster_id < 0:
        st.warning(
            "This is the noise / unassigned bucket — quality metrics that require "
            "cluster membership are not meaningful here."
        )

    _summary_chip(bundle.method, row)
