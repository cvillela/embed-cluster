"""Cluster validation page: existing single-page body, packaged as a function."""

from __future__ import annotations

import streamlit as st

from embedcluster.webapp import metrics_view, run_loader
from embedcluster.webapp.components import (
    audio_panel,
    cluster_panel,
    cluster_table,
    sidebar,
    similarity_panel,
)


def _hyperparam_keys(method: str, run_config: dict) -> list[str]:
    if method == "leiden":
        return [
            "k",
            "resolution",
            "min_similarity",
            "knn_metric",
            "normalize",
            "random_state",
        ]
    if method == "hdbscan":
        return [
            "min_cluster_size",
            "min_samples",
            "cluster_selection",
            "pca_components",
            "build_algo",
            "normalize",
            "random_state",
        ]
    if method == "kmeans":
        return [
            "n_clusters",
            "target_cluster_size",
            "max_iter",
            "nredo",
            "backend",
            "normalize",
            "random_state",
        ]
    return list(run_config.keys())


_PICKED_RUN_KEY = "selected_run_name"


def render() -> None:
    st.title("embedcluster validation")

    state = sidebar.render()

    summaries = run_loader.discover_runs(state.runs_root)
    if not summaries:
        st.warning(f"No runs found under `{state.runs_root}`.")
        st.stop()

    labels = [
        f"{s.name}  ({s.method}, N={s.n_rows:,}, k={s.n_clusters})"
        for s in summaries
    ]
    names = [s.name for s in summaries]
    prev = st.session_state.get(_PICKED_RUN_KEY)
    default_idx = names.index(prev) if prev in names else 0
    chosen_label = st.selectbox(
        "Run",
        labels,
        index=default_idx,
        key="run_selectbox",
    )
    selected = summaries[labels.index(chosen_label)]
    st.session_state[_PICKED_RUN_KEY] = selected.name

    bundle = run_loader.load_run(selected.path)

    c1, c2 = st.columns([1, 3])
    c1.metric("Method", bundle.method)
    c2.metric("Run", bundle.name)

    st.divider()

    st.header("Cluster sizes")
    table = metrics_view.render(bundle)

    st.divider()

    st.header("Audio per cluster")
    selected_cluster_id = cluster_table.select_cluster(bundle.method, table)
    cluster_panel.render(bundle, table, selected_cluster_id)
    audio_panel.render(
        bundle,
        selected_cluster_id,
        metadata_path=state.metadata_path,
        audio_field=state.audio_field,
        extra_cols=state.extra_metadata_cols,
    )

    st.divider()

    similarity_panel.render(
        bundle,
        embeddings_path=state.embeddings_path,
        metadata_path=state.metadata_path,
        audio_field=state.audio_field,
        extra_cols=state.extra_metadata_cols,
    )

    st.divider()

    st.header("UMAP")
    if state.umap_path:
        st.info("UMAP rendering lands in Phase 4.")
    else:
        st.caption("No `umap_6d.npy` configured in the sidebar.")

    st.divider()

    with st.expander("Hyperparameters"):
        present = [
            (k, bundle.run_config[k])
            for k in _hyperparam_keys(bundle.method, bundle.run_config)
            if k in bundle.run_config
        ]
        if present:
            cols = st.columns(min(len(present), 6))
            for i, (k, v) in enumerate(present):
                cols[i % len(cols)].metric(k, "—" if v is None else str(v))
        else:
            st.caption("No hyperparameters reported.")

    with st.expander("Dataset / preflight"):
        preflight = bundle.preflight
        n_dims = preflight.get("n_dims")
        working_dims = preflight.get("working_dims", n_dims)
        dtype = preflight.get("embedding_dtype", "—")
        cols = st.columns(3)
        cols[0].metric("D (raw)", "—" if n_dims is None else str(n_dims))
        cols[1].metric(
            "D (working)", "—" if working_dims is None else str(working_dims)
        )
        cols[2].metric("dtype", dtype)

    with st.expander("Raw run_config.json / metrics.json / preflight.json"):
        st.json(
            {
                "run_config": bundle.run_config,
                "metrics": bundle.metrics,
                "preflight": bundle.preflight,
            }
        )
