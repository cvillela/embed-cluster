import argparse
import os
import sys

import streamlit as st


def _apply_cli_defaults() -> None:
    """Parse CLI args (streamlit run app.py -- --metadata ...) into env vars
    consumed by sidebar._init_session_defaults."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--umap", type=str, default=None)
    parser.add_argument("--embeddings", type=str, default=None)
    parser.add_argument("--runs-root", type=str, default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    if args.metadata is not None:
        os.environ.setdefault("EMBEDCLUSTER_METADATA_PATH", args.metadata)
    if args.umap is not None:
        os.environ.setdefault("EMBEDCLUSTER_UMAP_PATH", args.umap)
    if args.embeddings is not None:
        os.environ.setdefault("EMBEDCLUSTER_EMBEDDINGS_PATH", args.embeddings)
    if args.runs_root is not None:
        os.environ.setdefault("EMBEDCLUSTER_RUNS_ROOT", args.runs_root)


_apply_cli_defaults()

from embedcluster.webapp import metrics_view, run_loader  # noqa: E402
from embedcluster.webapp.components import (  # noqa: E402
    audio_panel,
    cluster_panel,
    cluster_table,
    sidebar,
    similarity_panel,
)

st.set_page_config(page_title="embedcluster validation", layout="wide")
st.title("embedcluster validation")

state = sidebar.render()

if state.selected is None:
    st.info("Select a run from the sidebar to begin.")
    st.stop()

bundle = run_loader.load_run(state.selected.path)


def _hyperparam_keys(method: str) -> list[str]:
    if method == "leiden":
        return ["k", "resolution", "min_similarity", "knn_metric", "normalize", "random_state"]
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
    return list(bundle.run_config.keys())


# 1) Method
c1, c2 = st.columns([1, 3])
c1.metric("Method", bundle.method)
c2.metric("Run", bundle.name)

st.divider()

# 2) Cluster size
st.header("Cluster sizes")
table = metrics_view.render(bundle)

st.divider()

# 3) Audio listening per cluster (with cluster selection)
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

# 4) UMAP
st.header("UMAP")
if state.umap_path:
    st.info("UMAP rendering lands in Phase 4.")
else:
    st.caption("No `umap_6d.npy` configured in the sidebar.")

st.divider()

# 5) Rest — hyperparams, dataset counts, raw config
with st.expander("Hyperparameters"):
    present = [(k, bundle.run_config[k]) for k in _hyperparam_keys(bundle.method) if k in bundle.run_config]
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
    cols[1].metric("D (working)", "—" if working_dims is None else str(working_dims))
    cols[2].metric("dtype", dtype)

with st.expander("Raw run_config.json / metrics.json / preflight.json"):
    st.json({"run_config": bundle.run_config, "metrics": bundle.metrics, "preflight": bundle.preflight})
