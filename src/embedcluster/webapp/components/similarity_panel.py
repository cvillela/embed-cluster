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
_PERPAGE_KEY = "sim_perpage"
_MAXRESULTS_KEY = "sim_max_results"
_NORMALIZE_KEY = "sim_normalize"
_DEDUPE_KEY = "sim_dedupe_field"
_PAGE_KEY = "sim_page"
_RESULTS_KEY = "sim_results"


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


def _fmt_cluster(cid) -> str:
    if cid is None:
        return "—"
    try:
        return str(int(cid))
    except (TypeError, ValueError):
        return str(cid)


def _render_result_card(
    row_id: int,
    row: pd.Series,
    audio_field: str,
    extra_cols: list[str],
    score_label: str,
    score_value: float,
    cluster_id=None,
) -> None:
    raw = row.get(audio_field)
    has_path = isinstance(raw, str) and bool(raw)
    resolved = Path(raw).expanduser() if has_path else None
    title = resolved.name if resolved is not None else "(no path)"

    st.markdown(f"**{title}**")
    st.caption(
        f"`row_id` `{row_id}` · `cluster` `{_fmt_cluster(cluster_id)}` · "
        f"`{score_label}` `{score_value:.4f}`"
    )

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

    c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1])
    query_str = c1.text_input(
        "row_id or audio path / filename",
        key=_TEXT_KEY,
    )
    per_page = c2.slider(
        "Per page", min_value=1, max_value=50, value=8, key=_PERPAGE_KEY
    )
    max_results = c3.slider(
        "Max results",
        min_value=10,
        max_value=min(2000, max(10, n_rows)),
        value=min(200, max(10, n_rows)),
        step=10,
        key=_MAXRESULTS_KEY,
    )
    normalize = c4.checkbox("cosine (normalize)", value=True, key=_NORMALIZE_KEY)
    go = c5.button("Search", key="sim_search_btn", width="stretch")

    dedupe_choices = ["(none)"] + [c for c in meta.columns if c != "row_id"]
    dedupe_field = st.selectbox(
        "Dedupe by metadata field (one example per unique value)",
        dedupe_choices,
        index=0,
        key=_DEDUPE_KEY,
    )
    dedupe_active = dedupe_field != "(none)"

    auto_run = st.session_state.pop(AUTO_RUN_KEY, False)
    trigger = go or auto_run

    if trigger:
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
            k_fetch = min(int(max_results) + 1, n_rows)
            idx, scores = _search(backend, k_fetch, q, normalize)

        pairs = [(int(i), float(s)) for i, s in zip(idx, scores) if int(i) != rid][
            : int(max_results)
        ]
        st.session_state[_RESULTS_KEY] = {
            "rid": int(rid),
            "pairs": pairs,
            "backend": backend[0],
            "normalize": bool(normalize),
        }
        st.session_state[_PAGE_KEY] = 1

    cached = st.session_state.get(_RESULTS_KEY)
    if not cached:
        return

    pairs = cached["pairs"]
    rid = cached["rid"]
    score_label = "cosine" if cached["normalize"] else "L2²"
    backend_kind = cached["backend"]

    if not pairs:
        st.info("No results.")
        return

    keep_cols = [c for c in [audio_field, *extra_cols] if c and c in meta.columns]
    if dedupe_active and dedupe_field not in keep_cols:
        keep_cols.append(dedupe_field)
    keep_cols = list(dict.fromkeys(keep_cols))

    sub = pd.DataFrame(
        {"row_id": [p[0] for p in pairs], "_score": [p[1] for p in pairs]}
    )
    joined = sub.set_index("row_id").join(meta[keep_cols], how="left")
    if dedupe_active:
        joined = joined.dropna(subset=[dedupe_field]).drop_duplicates(
            subset=[dedupe_field], keep="first"
        )
        if joined.empty:
            st.info(f"No results with non-null `{dedupe_field}`.")
            return

    total = len(joined)
    n_pages = max(1, (total + per_page - 1) // per_page)
    cur_page = int(st.session_state.get(_PAGE_KEY, 1))
    if cur_page > n_pages:
        cur_page = 1
        st.session_state[_PAGE_KEY] = 1

    cluster_series = bundle.labels.set_index("row_id")["cluster_id"]

    def _cluster_for(r: int):
        try:
            return cluster_series.at[r]
        except KeyError:
            return None

    q_cluster = _cluster_for(rid)
    if audio_field and audio_field in meta.columns and rid in meta.index:
        st.caption(
            f"Query: row_id `{rid}` · cluster `{_fmt_cluster(q_cluster)}` · "
            f"`{meta.loc[rid, audio_field]}` · backend `{backend_kind}`"
        )
    else:
        st.caption(
            f"Query: row_id `{rid}` · cluster `{_fmt_cluster(q_cluster)}` · "
            f"backend `{backend_kind}`"
        )

    pcol1, pcol2 = st.columns([1, 3])
    page = pcol1.number_input(
        "Page",
        min_value=1,
        max_value=n_pages,
        value=cur_page,
        step=1,
        key=_PAGE_KEY,
    )
    start = (int(page) - 1) * per_page
    end = start + per_page
    pcol2.caption(
        f"Showing {start + 1}–{min(end, total)} of {total}"
        + (" (after dedupe)" if dedupe_active else "")
        + f" · pool {len(pairs)}"
    )
    page_df = joined.iloc[start:end]

    items = list(page_df.iterrows())
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
                        cluster_id=_cluster_for(int(row_id)),
                    )
        st.write("")
