"""Per-cluster audio listening grid."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from embedcluster.webapp import audio_sampler, metadata_loader
from embedcluster.webapp.run_loader import RunBundle

_DEFAULT_N = 8
_MAX_N = 24
_COLS_PER_ROW = 2


def _seed_key(method: str) -> str:
    return f"audio_seed_{method}"


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
    n = c2.slider(
        "N", min_value=1, max_value=_MAX_N, value=_DEFAULT_N, key=f"audio_n_{bundle.method}"
    )
    if c3.button("Refresh seed", key=f"audio_refresh_{bundle.method}"):
        st.session_state[seed_key] += 1
    seed = int(st.session_state[seed_key])

    try:
        sub = audio_sampler.sample_cluster(
            bundle.labels, bundle.method, cluster_id, strategy, n, seed=seed
        )
    except ValueError as e:
        st.error(str(e))
        return
    if sub.empty:
        st.info("No rows in this cluster.")
        return

    keep_meta_cols = [audio_field] + [c for c in extra_cols if c in meta.columns and c != audio_field]
    joined = sub.set_index("row_id").join(meta[keep_meta_cols], how="left")

    quality_col = audio_sampler.QUALITY_COLUMN.get(bundle.method)
    items = list(joined.iterrows())

    for i in range(0, len(items), _COLS_PER_ROW):
        cols = st.columns(_COLS_PER_ROW)
        for col, (row_id, row) in zip(cols, items[i : i + _COLS_PER_ROW]):
            with col:
                _render_card(row_id, row, audio_field, quality_col, extra_cols)


def _render_card(
    row_id: int,
    row: pd.Series,
    audio_field: str,
    quality_col: str | None,
    extra_cols: list[str],
) -> None:
    header = [f"row_id={row_id}"]
    if quality_col and quality_col in row and pd.notna(row[quality_col]):
        header.append(f"{quality_col}={float(row[quality_col]):.4f}")
    st.caption(" · ".join(header))

    raw = row[audio_field]
    if not isinstance(raw, str) or not raw:
        st.warning("missing audio path in metadata")
        return
    resolved = Path(raw).expanduser()
    if not resolved.exists():
        st.warning(f"file not found: `{resolved}`")
    else:
        try:
            st.audio(str(resolved))
        except Exception as e:  # noqa: BLE001
            st.warning(f"audio failed: {e}")

    for c in extra_cols:
        if c in row and pd.notna(row[c]):
            st.caption(f"{c}: {row[c]}")
    st.code(str(resolved), language=None)
