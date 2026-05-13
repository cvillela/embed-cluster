"""Per-cluster audio listening grid."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from embedcluster.webapp import audio_sampler, metadata_loader
from embedcluster.webapp.components import similarity_panel
from embedcluster.webapp.run_loader import RunBundle

_DEFAULT_N = 8
_MAX_N = 24
_COLS_PER_ROW = 2


def _seed_key(method: str) -> str:
    return f"audio_seed_{method}"


def _page_key(method: str, cluster_id: int) -> str:
    return f"audio_page_{method}_{cluster_id}"


def render(
    bundle: RunBundle,
    cluster_id: int | None,
    metadata_path: str,
    audio_field: str | None,
    extra_cols: list[str],
) -> None:
    if cluster_id is None:
        return
    if not metadata_path:
        st.info("Configure `metadata.jsonl` in the sidebar to listen to cluster samples.")
        return
    p = Path(metadata_path).expanduser()
    if not p.exists():
        st.warning(f"metadata path not found: `{p}`")
        return
    try:
        meta = metadata_loader.load_metadata(p)
    except Exception as e:  # noqa: BLE001
        st.error(f"failed to load metadata: {e}")
        return
    if not audio_field or audio_field not in meta.columns:
        st.warning("Pick an audio-path field in the sidebar.")
        return

    strategies = audio_sampler.strategies_for(cluster_id)
    seed_key = _seed_key(bundle.method)
    st.session_state.setdefault(seed_key, 0)

    c1, c2, c3 = st.columns([2, 1, 1])
    strategy = c1.selectbox(
        "Strategy", strategies, key=f"audio_strategy_{bundle.method}_{cluster_id < 0}"
    )
    per_page = c2.slider(
        "Per page",
        min_value=1,
        max_value=_MAX_N,
        value=_DEFAULT_N,
        key=f"audio_n_{bundle.method}",
    )
    if c3.button("Refresh seed", key=f"audio_refresh_{bundle.method}"):
        st.session_state[seed_key] += 1
        st.session_state[_page_key(bundle.method, cluster_id)] = 1
    seed = int(st.session_state[seed_key])

    dedupe_choices = ["(none)"] + [c for c in meta.columns if c != "row_id"]
    dedupe_field = st.selectbox(
        "Dedupe by metadata field (one example per unique value)",
        dedupe_choices,
        index=0,
        key=f"audio_dedupe_{bundle.method}",
    )
    dedupe_active = dedupe_field != "(none)"

    cluster_size = int((bundle.labels["cluster_id"] == cluster_id).sum())

    try:
        sub = audio_sampler.sample_cluster(
            bundle.labels, bundle.method, cluster_id, strategy, cluster_size, seed=seed
        )
    except ValueError as e:
        st.error(str(e))
        return
    if sub.empty:
        st.info("No rows in this cluster.")
        return

    keep_meta_cols = [audio_field]
    keep_meta_cols += [c for c in extra_cols if c in meta.columns and c != audio_field]
    if dedupe_active and dedupe_field not in keep_meta_cols:
        keep_meta_cols.append(dedupe_field)
    joined = sub.set_index("row_id").join(meta[keep_meta_cols], how="left")
    if dedupe_active:
        joined = joined.dropna(subset=[dedupe_field]).drop_duplicates(
            subset=[dedupe_field], keep="first"
        )
        if joined.empty:
            st.info(f"No rows with non-null `{dedupe_field}` in this cluster.")
            return

    total = len(joined)
    n_pages = max(1, (total + per_page - 1) // per_page)
    pkey = _page_key(bundle.method, cluster_id)
    cur_page = int(st.session_state.get(pkey, 1))
    if cur_page > n_pages:
        cur_page = 1
        st.session_state[pkey] = 1

    pcol1, pcol2 = st.columns([1, 3])
    page = pcol1.number_input(
        "Page",
        min_value=1,
        max_value=n_pages,
        value=cur_page,
        step=1,
        key=pkey,
    )
    start = (int(page) - 1) * per_page
    end = start + per_page
    pcol2.caption(
        f"Showing {start + 1}–{min(end, total)} of {total}"
        + (" (after dedupe)" if dedupe_active else "")
        + f" · cluster size {cluster_size}"
    )
    joined = joined.iloc[start:end]

    quality_col = audio_sampler.QUALITY_COLUMN.get(bundle.method)
    items = list(joined.iterrows())

    for i in range(0, len(items), _COLS_PER_ROW):
        cols = st.columns(_COLS_PER_ROW, gap="medium")
        for col, (row_id, row) in zip(cols, items[i : i + _COLS_PER_ROW]):
            with col:
                with st.container(border=True):
                    _render_card(
                        row_id,
                        row,
                        audio_field,
                        quality_col,
                        extra_cols,
                        button_key=f"sim_btn_{bundle.method}_{cluster_id}_{row_id}",
                    )
        st.write("")


def _render_card(
    row_id: int,
    row: pd.Series,
    audio_field: str,
    quality_col: str | None,
    extra_cols: list[str],
    *,
    button_key: str | None = None,
) -> None:
    raw = row[audio_field]
    has_path = isinstance(raw, str) and bool(raw)
    resolved = Path(raw).expanduser() if has_path else None

    title = resolved.name if resolved is not None else "(no path)"
    st.markdown(f"**{title}**")

    meta_bits = [f"`row_id` `{row_id}`"]
    if quality_col and quality_col in row and pd.notna(row[quality_col]):
        meta_bits.append(f"`{quality_col}` `{float(row[quality_col]):.4f}`")
    st.caption(" · ".join(meta_bits))

    if not has_path:
        st.warning("missing audio path in metadata")
    elif not resolved.exists():
        st.warning(f"file not found: `{resolved}`")
    else:
        try:
            st.audio(str(resolved))
        except Exception as e:  # noqa: BLE001
            st.warning(f"audio failed: {e}")

    extras = [(c, row[c]) for c in extra_cols if c in row and pd.notna(row[c])]
    if extras:
        st.markdown(
            "\n".join(f"- **{c}**: {v}" for c, v in extras)
        )

    if resolved is not None:
        with st.expander("path", expanded=False):
            st.code(str(resolved), language=None)

    if button_key is not None:
        if st.button("🔎 Find similar", key=button_key, width="stretch"):
            st.session_state[similarity_panel.QUERY_KEY] = int(row_id)
            st.session_state[similarity_panel.AUTO_RUN_KEY] = True
            st.rerun()
