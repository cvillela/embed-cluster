"""Cluster picker — sort + selectbox, no big table."""
from __future__ import annotations

import pandas as pd
import streamlit as st


_DEFAULT_SORT: dict[str, str] = {
    "kmeans": "cosine_to_centroid_mean",
    "hdbscan": "persistence",
    "leiden": "mean_neighbor_similarity_mean",
}


def select_cluster(method: str, table: pd.DataFrame) -> int | None:
    """Render sort controls + a cluster selectbox; return the chosen ``cluster_id``."""
    if table.empty:
        st.info("No clusters to select.")
        return None

    sortable = [c for c in table.columns if c != "cluster_id"]
    default_sort = _DEFAULT_SORT.get(method, "size")
    if default_sort not in sortable:
        default_sort = "size"

    c1, c2 = st.columns([1, 1])
    sort_col = c1.selectbox(
        "Sort clusters by",
        sortable,
        index=sortable.index(default_sort),
        key=f"cluster_sort_{method}",
    )
    descending = c2.toggle("Descending", value=True, key=f"cluster_sort_dir_{method}")

    sorted_df = table.sort_values(sort_col, ascending=not descending, na_position="last")
    cluster_ids = sorted_df["cluster_id"].astype(int).tolist()

    prev = st.session_state.get("selected_cluster_id")
    idx = cluster_ids.index(prev) if prev in cluster_ids else 0
    cluster_id = int(
        st.selectbox(
            "Cluster",
            cluster_ids,
            index=idx,
            format_func=lambda cid: f"cluster {cid}  (size={int(sorted_df.loc[sorted_df.cluster_id == cid, 'size'].iloc[0]):,})",
            key=f"cluster_select_{method}",
        )
    )
    st.session_state["selected_cluster_id"] = cluster_id
    return cluster_id
