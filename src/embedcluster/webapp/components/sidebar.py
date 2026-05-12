"""Sidebar: runs root, run picker, external input paths."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st

from embedcluster.webapp import metadata_loader, run_loader
from embedcluster.webapp.run_loader import RunSummary

DEFAULT_RUNS_ROOT = "./runs"


@dataclass
class SidebarState:
    runs_root: Path
    selected: RunSummary | None
    metadata_path: str
    umap_path: str
    audio_field: str | None = None
    extra_metadata_cols: list[str] = field(default_factory=list)


def _init_session_defaults() -> None:
    st.session_state.setdefault("runs_root", DEFAULT_RUNS_ROOT)
    st.session_state.setdefault("metadata_path", "")
    st.session_state.setdefault("umap_path", "")
    st.session_state.setdefault("selected_run_name", None)
    st.session_state.setdefault("audio_field", None)
    st.session_state.setdefault("extra_metadata_cols", [])


class _FileDialogUnavailable(RuntimeError):
    pass


def _pick_file_dialog(title: str, initial: str, filetypes: list[tuple[str, str]]) -> str | None:
    """Native file picker via tkinter. Returns chosen path, None on user-cancel.

    Raises ``_FileDialogUnavailable`` if tkinter cannot open a window
    (e.g. headless / no $DISPLAY).
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:  # noqa: BLE001
        raise _FileDialogUnavailable(str(e)) from e
    try:
        root = tk.Tk()
    except Exception as e:  # noqa: BLE001
        raise _FileDialogUnavailable(str(e)) from e
    try:
        root.withdraw()
        root.attributes("-topmost", True)
        initialdir = str(Path(initial).expanduser().parent) if initial else "."
        path = filedialog.askopenfilename(title=title, initialdir=initialdir, filetypes=filetypes)
    finally:
        root.destroy()
    return path or None


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


def _on_close(show_key: str) -> None:
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
    c1, c2 = st.columns([1, 1])
    c1.button(
        "Open / Pick",
        key=f"{key}_browser_pick",
        on_click=_on_pick,
        args=(key, kinds, targets, idx_key, cur_key, show_key),
    )
    c2.button("Close", key=f"{key}_browser_close", on_click=_on_close, args=(show_key,))


def _path_input_with_browse(
    label: str,
    key: str,
    filetypes: list[tuple[str, str]],
    exts: tuple[str, ...],
) -> None:
    """Text input + 'Browse…' button.

    Tries native (tkinter) file dialog first; on failure (e.g. headless), falls
    back to an inline in-app directory walker. Once the native dialog has failed
    in this session, the button just toggles the in-app browser visibility.
    """
    show_key = f"{key}_show_inapp"
    failed_key = f"{key}_tk_failed"

    c1, c2 = st.columns([4, 1])
    c1.text_input(label, key=key)
    if c2.button("Browse…", key=f"{key}_browse"):
        if st.session_state.get(failed_key, False):
            st.session_state[show_key] = not st.session_state.get(show_key, False)
        else:
            try:
                chosen = _pick_file_dialog(label, st.session_state.get(key, ""), filetypes)
            except _FileDialogUnavailable as e:
                st.info(f"native file dialog unavailable ({e}); using in-app browser.")
                st.session_state[failed_key] = True
                st.session_state[show_key] = True
            else:
                if chosen:
                    st.session_state[key] = chosen
                    st.rerun()

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
    prev_extra = [c for c in st.session_state.get("extra_metadata_cols", []) if c in other_cols]
    extra = st.multiselect(
        "extra metadata columns",
        other_cols,
        default=prev_extra,
        key="extra_metadata_cols",
    )
    return audio_field, list(extra)


def render() -> SidebarState:
    _init_session_defaults()

    with st.sidebar:
        st.header("Run")
        st.text_input("Runs root", key="runs_root")
        runs_root = Path(st.session_state["runs_root"]).expanduser()

        summaries = run_loader.discover_runs(runs_root)
        if not summaries:
            st.warning(f"No runs found under `{runs_root}`.")
            selected: RunSummary | None = None
        else:
            labels = [
                f"{s.name}  ({s.method}, N={s.n_rows:,}, k={s.n_clusters})"
                for s in summaries
            ]
            names = [s.name for s in summaries]
            prev = st.session_state.get("selected_run_name")
            default_idx = names.index(prev) if prev in names else 0
            chosen_label = st.selectbox("Run", labels, index=default_idx, key="run_selectbox")
            selected = summaries[labels.index(chosen_label)]
            st.session_state["selected_run_name"] = selected.name

        st.divider()
        st.header("External inputs")
        _path_input_with_browse(
            "metadata.jsonl path",
            "metadata_path",
            filetypes=[("JSONL", "*.jsonl"), ("JSON", "*.json"), ("All files", "*.*")],
            exts=(".jsonl", ".json"),
        )
        _path_input_with_browse(
            "umap_6d.npy path (optional)",
            "umap_path",
            filetypes=[("NumPy", "*.npy"), ("All files", "*.*")],
            exts=(".npy",),
        )

        audio_field, extra_cols = _metadata_controls(st.session_state["metadata_path"])

        st.divider()
        if st.button("Reload (clear caches)"):
            run_loader.clear_cache()
            st.rerun()

    return SidebarState(
        runs_root=runs_root,
        selected=selected,
        metadata_path=st.session_state["metadata_path"],
        umap_path=st.session_state["umap_path"],
        audio_field=audio_field,
        extra_metadata_cols=extra_cols,
    )
