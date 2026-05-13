"""Similarity search panel: FAISS top-k over raw embeddings."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from embedcluster.webapp import metadata_loader
from embedcluster.webapp.run_loader import RunBundle

QUERY_KEY = "sim_query_row_id"
AUTO_RUN_KEY = "sim_auto_run"
_TEXT_KEY = "sim_query_text"
_TOPK_KEY = "sim_topk"
_NORMALIZE_KEY = "sim_normalize"


@st.cache_resource(show_spinner="Loading embeddings (mmap)…")
def _load_embeddings(path_str: str, mtime: float) -> np.ndarray:
    arr = np.load(path_str, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {arr.shape!r}")
    return arr


@st.cache_resource(show_spinner="Building FAISS index…")
def _build_index(path_str: str, mtime: float, normalize: bool):
    """Return ('faiss', index, X_used) or ('numpy', X_used, None)."""
    raw = _load_embeddings(path_str, mtime)
    X = np.ascontiguousarray(np.asarray(raw, dtype=np.float32))
    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms
    try:
        import faiss  # type: ignore

        index = faiss.IndexFlatIP(int(X.shape[1])) if normalize else faiss.IndexFlatL2(int(X.shape[1]))
        index.add(X)
        return ("faiss", index, X)
    except Exception:
        return ("numpy", X, None)


def _search(backend, k: int, q: np.ndarray, normalize: bool) -> tuple[np.ndarray, np.ndarray]:
    kind = backend[0]
    if kind == "faiss":
        index = backend[1]
        d, i = index.search(q.reshape(1, -1).astype(np.float32, copy=False), k)
        return i[0], d[0]
    X = backend[1]
    if normalize:
        scores = X @ q
        order = np.argpartition(-scores, k - 1)[:k]
        order = order[np.argsort(-scores[order])]
        return order, scores[order]
    diff = X - q
    d2 = np.einsum("ij,ij->i", diff, diff)
    order = np.argpartition(d2, k - 1)[:k]
    order = order[np.argsort(d2[order])]
    return order, d2[order]


def _resolve_row_id(value: str, n_rows: int, meta: pd.DataFrame, audio_field: str | None) -> int | None:
    v = value.strip()
    if not v:
        return None
    if v.isdigit():
        rid = int(v)
        if 0 <= rid < n_rows:
            return rid
    if audio_field and audio_field in meta.columns:
        col = meta[audio_field].astype(str)
        match = meta[col == v]
        if not match.empty:
            return int(match.iloc[0]["row_id"])
        match = meta[col.str.endswith("/" + v) | (col.apply(lambda s: Path(s).name) == v)]
        if not match.empty:
            return int(match.iloc[0]["row_id"])
    return None


def _render_result_card(
    row_id: int,
    row: pd.Series,
    audio_field: str,
    extra_cols: list[str],
    score_label: str,
    score_value: float,
) -> None:
    raw = row.get(audio_field)
    has_path = isinstance(raw, str) and bool(raw)
    resolved = Path(raw).expanduser() if has_path else None
    title = resolved.name if resolved is not None else "(no path)"

    st.markdown(f"**{title}**")
    st.caption(f"`row_id` `{row_id}` · `{score_label}` `{score_value:.4f}`")

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
        st.markdown("\n".join(f"- **{c}**: {v}" for c, v in extras))


def render(
    bundle: RunBundle,
    embeddings_path: str,
    metadata_path: str,
    audio_field: str | None,
    extra_cols: list[str],
) -> None:
    st.header("Similarity search")

    if not embeddings_path:
        st.info("Configure `embeddings .npy path` in the sidebar to enable similarity search.")
        return
    emb_p = Path(embeddings_path).expanduser()
    if not emb_p.exists():
        st.warning(f"embeddings path not found: `{emb_p}`")
        return
    if not metadata_path:
        st.info("Configure `metadata.jsonl path` in the sidebar.")
        return
    meta_p = Path(metadata_path).expanduser()
    if not meta_p.exists():
        st.warning(f"metadata path not found: `{meta_p}`")
        return
    try:
        meta = metadata_loader.load_metadata(meta_p)
    except Exception as e:  # noqa: BLE001
        st.error(f"failed to load metadata: {e}")
        return

    try:
        emb = _load_embeddings(str(emb_p), emb_p.stat().st_mtime)
    except Exception as e:  # noqa: BLE001
        st.error(f"failed to load embeddings: {e}")
        return

    n_rows = int(emb.shape[0])
    if n_rows != len(bundle.labels):
        st.warning(
            f"embeddings rows ({n_rows}) != run rows ({len(bundle.labels)}). "
            "row_id alignment may be off."
        )

    pending = st.session_state.pop(QUERY_KEY, None)
    if pending is not None:
        st.session_state[_TEXT_KEY] = str(int(pending))

    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    query_str = c1.text_input(
        "row_id or audio path / filename",
        key=_TEXT_KEY,
    )
    k = c2.slider("top-k", min_value=1, max_value=50, value=8, key=_TOPK_KEY)
    normalize = c3.checkbox("cosine (normalize)", value=True, key=_NORMALIZE_KEY)
    go = c4.button("Search", key="sim_search_btn", width="stretch")

    auto_run = st.session_state.pop(AUTO_RUN_KEY, False)
    if not (go or auto_run):
        return

    rid = _resolve_row_id(query_str, n_rows, meta, audio_field)
    if rid is None:
        st.warning(
            "Could not resolve query. Enter a numeric row_id, full audio path, or filename "
            "matching the audio-path metadata field."
        )
        return

    with st.spinner("Searching…"):
        backend = _build_index(str(emb_p), emb_p.stat().st_mtime, normalize)
        X = backend[2] if backend[0] == "faiss" else backend[1]
        q = np.asarray(X[rid], dtype=np.float32)
        idx, scores = _search(backend, k + 1, q, normalize)

    pairs = [(int(i), float(s)) for i, s in zip(idx, scores) if int(i) != rid][:k]
    if not pairs:
        st.info("No results.")
        return

    score_label = "cosine" if normalize else "L2²"

    if audio_field and audio_field in meta.columns and rid in meta.index:
        st.caption(f"Query: row_id `{rid}` · `{meta.loc[rid, audio_field]}` · backend `{backend[0]}`")
    else:
        st.caption(f"Query: row_id `{rid}` · backend `{backend[0]}`")

    keep_cols = [c for c in [audio_field, *extra_cols] if c and c in meta.columns]
    keep_cols = list(dict.fromkeys(keep_cols))
    sub = pd.DataFrame({"row_id": [p[0] for p in pairs], "_score": [p[1] for p in pairs]})
    joined = sub.set_index("row_id").join(meta[keep_cols], how="left")

    items = list(joined.iterrows())
    cols_per_row = 2
    for i in range(0, len(items), cols_per_row):
        cols = st.columns(cols_per_row, gap="medium")
        for col, (row_id, row) in zip(cols, items[i : i + cols_per_row]):
            with col:
                with st.container(border=True):
                    _render_result_card(
                        int(row_id),
                        row,
                        audio_field or "",
                        extra_cols,
                        score_label,
                        float(row["_score"]),
                    )
        st.write("")
