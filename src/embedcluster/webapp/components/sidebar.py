"""Sidebar: runs root, run picker, external input paths."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st

from embedcluster.webapp import metadata_loader, run_loader

DEFAULT_RUNS_ROOT = "./runs"
DEFAULT_METADATA_PATH = ""

_PERSIST_KEYS = (
    "runs_root",
    "metadata_path",
    "umap_path",
    "embeddings_path",
    "audio_field",
    "extra_metadata_cols",
    "selected_run_name",
    "dedupe_picked_run_name",
    "dedupe_emb_path",
    "dedupe_threshold",
    "dedupe_chunk_size",
    "dedupe_out_name",
    "dedupe_last_launched_out",
)


def _reanchor_session_state() -> None:
    """Force-rebind keys so streamlit retains them across page switches.

    Streamlit can clear widget-keyed session_state when a widget unmounts
    (e.g. switching pages). Reassigning to itself anchors the value for the
    next rerun.
    """
    for k in _PERSIST_KEYS:
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]


@dataclass
class SidebarState:
    runs_root: Path
    metadata_path: str
    umap_path: str
    embeddings_path: str
    audio_field: str | None = None
    extra_metadata_cols: list[str] = field(default_factory=list)


def _init_session_defaults() -> None:
    st.session_state.setdefault(
        "runs_root",
        os.environ.get("EMBEDCLUSTER_RUNS_ROOT", DEFAULT_RUNS_ROOT),
    )
    st.session_state.setdefault(
        "metadata_path",
        os.environ.get("EMBEDCLUSTER_METADATA_PATH", DEFAULT_METADATA_PATH),
    )
    st.session_state.setdefault("umap_path", os.environ.get("EMBEDCLUSTER_UMAP_PATH", ""))
    st.session_state.setdefault(
        "embeddings_path",
        os.environ.get("EMBEDCLUSTER_EMBEDDINGS_PATH", ""),
    )
    st.session_state.setdefault("selected_run_name", None)
    st.session_state.setdefault("audio_field", None)
    st.session_state.setdefault("extra_metadata_cols", [])


def _on_pick(key: str, kinds: list[str], targets: list[str], idx_key: str, cur_key: str, show_key: str) -> None:
    """Button callback: navigate (dir) or pick (file). Runs before next rerun."""
    idx = int(st.session_state.get(idx_key, 0))
    if idx >= len(kinds):
        return
    kind = kinds[idx]
    target = targets[idx]
    if kind == "dir":
        st.session_state[cur_key] = target
        st.session_state.pop(idx_key, None)  # reset selection in new dir
    else:
        st.session_state[key] = target
        st.session_state[show_key] = False


def _inapp_browser(key: str, exts: tuple[str, ...]) -> None:
    """Inline directory walker — fallback when no native dialog is available."""
    cur_key = f"{key}_browser_dir"
    show_key = f"{key}_show_inapp"
    if cur_key not in st.session_state:
        existing = st.session_state.get(key, "")
        start = Path(existing).expanduser().parent if existing else Path.cwd()
        st.session_state[cur_key] = str(start if start.exists() else Path.cwd())

    st.text_input("directory", key=cur_key)
    cur = Path(st.session_state[cur_key]).expanduser()
    if not cur.exists() or not cur.is_dir():
        st.warning(f"not a directory: `{cur}`")
        return

    try:
        entries = sorted(cur.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError as e:
        st.warning(str(e))
        return

    dirs = [p for p in entries if p.is_dir() and not p.name.startswith(".")]
    files = [
        p for p in entries
        if p.is_file()
        and not p.name.startswith(".")
        and (not exts or p.suffix.lower() in exts)
    ]

    labels: list[str] = [".. (up one level)"]
    kinds: list[str] = ["dir"]
    targets: list[str] = [str(cur.parent)]
    for d in dirs:
        labels.append(f"[dir]  {d.name}/")
        kinds.append("dir")
        targets.append(str(d))
    for f in files:
        labels.append(f"[file] {f.name}")
        kinds.append("file")
        targets.append(str(f))

    idx_key = f"{key}_browser_sel"
    st.selectbox(
        "entries",
        range(len(labels)),
        format_func=lambda i: labels[i],
        key=idx_key,
    )
    st.button(
        "Open / Pick",
        key=f"{key}_browser_pick",
        on_click=_on_pick,
        args=(key, kinds, targets, idx_key, cur_key, show_key),
        width="stretch",
    )


def _path_input_with_browse(
    label: str,
    key: str,
    exts: tuple[str, ...],
) -> None:
    """Text input + 'Browse…' button toggling inline directory walker."""
    show_key = f"{key}_show_inapp"
    pending_key = f"{key}_pending"

    if pending_key in st.session_state:
        st.session_state[key] = st.session_state.pop(pending_key)

    st.text_input(label, key=key)
    is_open = st.session_state.get(show_key, False)
    if st.button("Close browser" if is_open else "Browse…", key=f"{key}_browse", width="stretch"):
        st.session_state[show_key] = not is_open

    if st.session_state.get(show_key, False):
        with st.container(border=True):
            _inapp_browser(key, exts)


def _metadata_controls(metadata_path: str) -> tuple[str | None, list[str]]:
    if not metadata_path:
        return None, []
    p = Path(metadata_path).expanduser()
    if not p.exists():
        st.caption(f"metadata path not found: `{p}`")
        return None, []
    try:
        meta = metadata_loader.load_metadata(p)
    except Exception as e:  # noqa: BLE001
        st.caption(f"failed to read metadata: {e}")
        return None, []

    audio_like = metadata_loader.detect_audio_fields(meta)
    other = [c for c in meta.columns if c != "row_id" and c not in audio_like]
    candidates = audio_like + other  # audio-like promoted to top, but show every field
    if not candidates:
        st.caption("metadata has no usable columns.")
        return None, []
    prev_field = st.session_state.get("audio_field")
    default_idx = candidates.index(prev_field) if prev_field in candidates else 0
    audio_field = st.selectbox("audio-path field", candidates, index=default_idx, key="audio_field")

    other_cols = [c for c in meta.columns if c not in {audio_field, "row_id"}]
    st.session_state["extra_metadata_cols"] = [
        c for c in st.session_state.get("extra_metadata_cols", []) if c in other_cols
    ]
    extra = st.multiselect(
        "extra metadata columns",
        other_cols,
        key="extra_metadata_cols",
    )
    return audio_field, list(extra)


def render() -> SidebarState:
    _init_session_defaults()
    _reanchor_session_state()

    with st.sidebar:
        st.header("Run")
        st.text_input("Runs root", key="runs_root")
        runs_root = Path(st.session_state["runs_root"]).expanduser()

        st.divider()
        st.header("External inputs")
        _path_input_with_browse(
            "metadata.jsonl path",
            "metadata_path",
            exts=(".jsonl", ".json"),
        )
        _path_input_with_browse(
            "umap_6d.npy path (optional)",
            "umap_path",
            exts=(".npy",),
        )
        _path_input_with_browse(
            "embeddings .npy path (raw, for similarity search)",
            "embeddings_path",
            exts=(".npy",),
        )

        audio_field, extra_cols = _metadata_controls(st.session_state["metadata_path"])

        st.divider()
        if st.button("Reload (clear caches)"):
            run_loader.clear_cache()
            st.rerun()

    return SidebarState(
        runs_root=runs_root,
        metadata_path=st.session_state["metadata_path"],
        umap_path=st.session_state["umap_path"],
        embeddings_path=st.session_state["embeddings_path"],
        audio_field=audio_field,
        extra_metadata_cols=extra_cols,
    )
