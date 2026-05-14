"""Dedupe page: launcher (subprocess) + browser (paginated audio audition).

Independent of any clustering run. Only requires:
- embeddings .npy path (sidebar) — to launch new dedupe runs
- metadata.jsonl path + audio field (sidebar) — to audition group members
- a dedupe run dir under runs root — to browse
"""

from __future__ import annotations

import streamlit as st

from embedcluster.webapp import run_loader
from embedcluster.webapp.components import dedupe_launcher, dup_group_panel, sidebar


_PICKED_RUN_KEY = "dedupe_picked_run_name"


def _select_run(name: str) -> None:
    st.session_state[_PICKED_RUN_KEY] = name


def render() -> None:
    st.title("Dedupe")
    st.caption(
        "Find near-duplicate embeddings via GPU range search + connected components. "
        "Independent of clustering runs."
    )

    state = sidebar.render()

    dedupe_launcher.render(
        runs_root=state.runs_root,
        embeddings_path=state.embeddings_path,
        on_finish_select=_select_run,
    )

    st.divider()

    summaries = run_loader.discover_dedupe_runs(state.runs_root)
    if not summaries:
        st.info(
            f"No dedupe runs found under `{state.runs_root}`. "
            "Launch one above."
        )
        return

    labels = [
        f"{s.name}  (τ={s.threshold:.4f}, dup_groups={s.n_multi_member_groups}, "
        f"removable={s.n_removable_rows})"
        for s in summaries
    ]
    names = [s.name for s in summaries]
    prev = st.session_state.get(_PICKED_RUN_KEY)
    default_idx = names.index(prev) if prev in names else 0
    chosen = st.selectbox(
        "dedupe run",
        labels,
        index=default_idx,
        key="dedupe_run_selectbox",
    )
    selected = summaries[labels.index(chosen)]
    st.session_state[_PICKED_RUN_KEY] = selected.name

    bundle = run_loader.load_dedupe_run(selected.path)
    dup_group_panel.render(
        bundle=bundle,
        metadata_path=state.metadata_path,
        audio_field=state.audio_field,
        extra_cols=state.extra_metadata_cols,
    )
