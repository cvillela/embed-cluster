import argparse
import os
import sys

import streamlit as st


def _apply_cli_defaults() -> None:
    """Parse CLI args (streamlit run app.py -- --metadata ...) into env vars
    consumed by sidebar._init_session_defaults."""
    parser = argparse.ArgumentParser(
        prog="run_webapp.sh -- ",
        description="embedcluster webapp CLI defaults (override sidebar values).",
    )
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

from embedcluster.webapp.app_pages import cluster_page, dedupe_page  # noqa: E402

st.set_page_config(page_title="embedcluster", layout="wide")

pages = [
    st.Page(
        cluster_page.render,
        title="Cluster validation",
        url_path="cluster",
        icon=":material/scatter_plot:",
        default=True,
    ),
    st.Page(
        dedupe_page.render,
        title="Dedupe",
        url_path="dedupe",
        icon=":material/content_copy:",
    ),
]
nav = st.navigation(pages)
nav.run()
