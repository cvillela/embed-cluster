"""Launcher panel: configure and start ``embedcluster dedupe-remove``.

Requires a previously completed dedupe run (provides the ``dedupe.parquet``
manifest) plus the source embeddings.npy and metadata.jsonl that were used
to produce that run. Strategies: canonical | metadata | duration | limit-k.
``--k`` stacks on top of metadata as an optional limit-k filter.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import streamlit as st

from embedcluster.webapp import jobs, metadata_loader

_REFRESH_SECONDS = 3.0
_STRATEGIES = ("canonical", "metadata", "duration", "limit-k")
_DURATION_ORDERS = ("longest", "shortest")
_SELECTIONS = ("most_similar", "most_distant", "random")


def _suggest_out_dir(
    embeddings_path: str,
    dedupe_run_name: str,
    strategy: str,
) -> str:
    """Default output dir: sibling of embeddings.npy under ``deduped/``.

    Falls back to CWD if no embeddings path is set yet.
    """
    base_name = f"{dedupe_run_name}__remove_{strategy.replace('-', '_')}"
    if embeddings_path:
        parent = Path(embeddings_path).expanduser().parent
    else:
        parent = Path.cwd()
    candidate = parent / "deduped" / base_name
    if not candidate.exists():
        return str(candidate)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return str(candidate.with_name(f"{base_name}_{stamp}"))


def _validate(
    embeddings: str,
    metadata: str,
    manifest: Path,
    out_dir_str: str,
    strategy: str,
    metadata_field: Optional[str],
    k: Optional[int],
) -> str | None:
    if not embeddings:
        return "Configure `embeddings .npy path` in the sidebar."
    if not Path(embeddings).expanduser().exists():
        return f"embeddings path not found: `{embeddings}`"
    if not metadata:
        return "Configure `metadata.jsonl path` in the sidebar."
    if not Path(metadata).expanduser().exists():
        return f"metadata path not found: `{metadata}`"
    if not manifest.exists():
        return f"dedupe manifest not found: `{manifest}`"
    if not out_dir_str.strip():
        return "Choose an output directory."
    out_path = Path(out_dir_str).expanduser()
    if not out_path.is_absolute():
        return f"Output dir must be an absolute path: `{out_dir_str}`"
    parent = out_path.parent
    if not parent.exists():
        return f"Parent directory does not exist: `{parent}`"
    if strategy == "metadata" and not metadata_field:
        return "strategy=metadata requires a --metadata-field selection."
    if strategy in ("duration", "limit-k") and (k is None or k < 1):
        return f"strategy={strategy} requires K >= 1."
    if strategy == "metadata" and k is not None and k < 1:
        return "K must be >= 1 when stacked on metadata."
    return None


def _render_status(status: jobs.JobStatus, out_dir: Path) -> None:
    st.markdown(f"**Job pid `{status.pid}`** · out `{out_dir.name}`")
    elapsed = time.time() - status.started
    st.caption(f"running for {elapsed:.0f}s · log `{status.log}`")
    log_text = jobs.tail_log(Path(status.log), n_lines=200)
    st.code(log_text or "(no log output yet)", language="log")
    if st.button("Cancel", key="dedupe_remove_cancel_btn", type="secondary"):
        jobs.cancel_job(status.pid)
        st.warning("Sent SIGTERM. Allow a few seconds for the process to exit.")
        time.sleep(1.0)
        st.rerun()


def _render_finished(status: jobs.JobStatus, out_dir: Path) -> None:
    elapsed = (status.finished or time.time()) - status.started
    if status.returncode == 0:
        st.success(
            f"dedupe-remove finished in {elapsed:.0f}s. "
            f"Outputs in `{out_dir}`:"
        )
        files = [
            "deduped_embeddings.npy",
            "deduped_embeddings.jsonl",
            "kept_indices.npy",
            "removed_indices.json",
            "run_config.json",
            "metrics.json",
        ]
        st.code("\n".join(str(out_dir / f) for f in files), language=None)
    else:
        st.error(
            f"Job exited with returncode={status.returncode} after {elapsed:.0f}s."
        )
        st.code(jobs.tail_log(Path(status.log), n_lines=200), language="log")


def _strategy_form(
    metadata_path: str,
) -> tuple[str, Optional[str], str, Optional[int], str, int]:
    """Render strategy + parameter widgets. Returns the chosen values."""
    strategy = st.radio(
        "strategy",
        _STRATEGIES,
        horizontal=True,
        key="dedupe_remove_strategy",
        help=(
            "canonical: keep canonical only. "
            "metadata: keep canonical + same-field members "
            "(stack --k to also limit). "
            "duration: keep top-K by duration order. "
            "limit-k: keep canonical + (K-1) ranked by selection."
        ),
    )

    metadata_field: Optional[str] = None
    duration_order = "longest"
    k: Optional[int] = None
    selection = "most_similar"
    random_state = int(st.session_state.get("dedupe_remove_random_state", 42))

    if strategy == "metadata":
        meta_field = _metadata_field_picker(metadata_path)
        metadata_field = meta_field
        c1, c2 = st.columns([1, 2])
        use_k = c1.checkbox(
            "stack --k limit",
            value=bool(st.session_state.get("dedupe_remove_meta_use_k", False)),
            key="dedupe_remove_meta_use_k",
            help="Also cap each group to canonical + (K-1) same-field members.",
        )
        if use_k:
            kc1, kc2 = c2.columns(2)
            k = int(kc1.number_input(
                "K (total per group)",
                min_value=1,
                value=int(st.session_state.get("dedupe_remove_k", 2)),
                step=1,
                key="dedupe_remove_k",
            ))
            selection = kc2.selectbox(
                "selection",
                _SELECTIONS,
                index=_SELECTIONS.index(
                    st.session_state.get("dedupe_remove_selection", "most_similar")
                ),
                key="dedupe_remove_selection",
            )
    elif strategy == "duration":
        c1, c2 = st.columns(2)
        duration_order = c1.selectbox(
            "duration order",
            _DURATION_ORDERS,
            index=_DURATION_ORDERS.index(
                st.session_state.get("dedupe_remove_duration_order", "longest")
            ),
            key="dedupe_remove_duration_order",
        )
        k = int(c2.number_input(
            "K (per group)",
            min_value=1,
            value=int(st.session_state.get("dedupe_remove_k", 1)),
            step=1,
            key="dedupe_remove_k",
        ))
    elif strategy == "limit-k":
        c1, c2 = st.columns(2)
        k = int(c1.number_input(
            "K (total per group, incl. canonical)",
            min_value=1,
            value=int(st.session_state.get("dedupe_remove_k", 2)),
            step=1,
            key="dedupe_remove_k",
        ))
        selection = c2.selectbox(
            "selection",
            _SELECTIONS,
            index=_SELECTIONS.index(
                st.session_state.get("dedupe_remove_selection", "most_similar")
            ),
            key="dedupe_remove_selection",
        )

    if selection == "random" or (strategy == "limit-k" and selection == "random"):
        random_state = int(st.number_input(
            "random seed",
            min_value=0,
            value=random_state,
            step=1,
            key="dedupe_remove_random_state",
        ))

    return strategy, metadata_field, duration_order, k, selection, random_state


def _metadata_field_picker(metadata_path: str) -> Optional[str]:
    if not metadata_path:
        st.caption("(set metadata.jsonl in sidebar to pick a field)")
        return None
    p = Path(metadata_path).expanduser()
    if not p.exists():
        st.caption(f"metadata path not found: `{p}`")
        return None
    try:
        meta = metadata_loader.load_metadata(p)
    except Exception as e:  # noqa: BLE001
        st.caption(f"failed to load metadata: {e}")
        return None
    cols = [c for c in meta.columns if c != "row_id"]
    if not cols:
        st.caption("metadata has no usable fields.")
        return None
    prev = st.session_state.get("dedupe_remove_metadata_field")
    default_idx = cols.index(prev) if prev in cols else 0
    return st.selectbox(
        "metadata field (must match canonical's value)",
        cols,
        index=default_idx,
        key="dedupe_remove_metadata_field",
    )


def render(
    embeddings_path: str,
    metadata_path: str,
    dedupe_run_name: str,
    manifest_path: Path,
    on_finish_select: Callable[[str], None] | None = None,
) -> None:
    """Render the dedupe-remove launcher.

    Output dir is a free-form absolute path; defaults to a sibling of
    ``embeddings.npy`` under ``deduped/<run>__remove_<strategy>``.

    Parameters
    ----------
    embeddings_path, metadata_path : str
        Source embeddings.npy / metadata.jsonl (sidebar values).
    dedupe_run_name : str
        Name of the dedupe run providing the manifest.
    manifest_path : Path
        Resolved path to that run's ``dedupe.parquet``.
    on_finish_select : callable | None
        Optional hook invoked with the new dir basename on success.
    """
    with st.expander("Apply dedupe (remove duplicates)", expanded=False):
        # Drain any pending suggested-path swap before the text_input binds.
        if "dedupe_remove_out_dir_pending" in st.session_state:
            st.session_state["dedupe_remove_out_dir"] = (
                st.session_state.pop("dedupe_remove_out_dir_pending")
            )
        last_out = st.session_state.get("dedupe_remove_last_launched_out")
        if last_out:
            status = jobs.reconcile(Path(last_out))
            if status and status.finished is None:
                _render_status(status, Path(last_out))
                time.sleep(_REFRESH_SECONDS)
                st.rerun()
                return
            if status and status.finished is not None:
                _render_finished(status, Path(last_out))
                if status.returncode == 0:
                    st.session_state.pop("dedupe_remove_last_launched_out", None)
                    if on_finish_select:
                        on_finish_select(Path(last_out).name)

        st.caption(
            f"Source manifest: `{manifest_path}` "
            f"(from dedupe run `{dedupe_run_name}`)"
        )

        (
            strategy,
            metadata_field,
            duration_order,
            k,
            selection,
            random_state,
        ) = _strategy_form(metadata_path)

        suggested_default = _suggest_out_dir(
            embeddings_path, dedupe_run_name, strategy
        )
        st.session_state.setdefault("dedupe_remove_out_dir", suggested_default)
        c_name, c_btn = st.columns([3, 1])
        out_dir_str = c_name.text_input(
            "output directory (absolute path; defaults next to embeddings.npy)",
            key="dedupe_remove_out_dir",
            help=(
                "Free-form absolute path. Click 'suggest path' to default to "
                "<embeddings_dir>/deduped/<run>__remove_<strategy>."
            ),
        )
        if c_btn.button("suggest path", key="dedupe_remove_suggest_path"):
            st.session_state["dedupe_remove_out_dir_pending"] = suggested_default
            st.rerun()

        run_btn = st.button(
            "Run dedupe-remove",
            key="dedupe_remove_run_btn",
            type="primary",
            width="stretch",
        )
        if not run_btn:
            return

        err = _validate(
            embeddings_path,
            metadata_path,
            manifest_path,
            out_dir_str,
            strategy,
            metadata_field,
            k,
        )
        if err:
            st.error(err)
            return

        out_dir = Path(out_dir_str).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        if (out_dir / "deduped_embeddings.npy").exists():
            st.warning(
                f"`{out_dir}` already contains deduped_embeddings.npy. "
                "Pick a new path or delete the existing output before re-running."
            )
            return

        status = jobs.start_dedupe_remove_job(
            embeddings_path=Path(embeddings_path).expanduser(),
            metadata_path=Path(metadata_path).expanduser(),
            manifest_path=manifest_path,
            out_dir=out_dir,
            strategy=strategy,
            metadata_field=metadata_field,
            duration_order=duration_order,
            k=k,
            selection=selection,
            random_state=random_state,
        )
        st.session_state["dedupe_remove_last_launched_out"] = str(out_dir)
        st.success(
            f"Launched dedupe-remove (pid {status.pid}). Output dir: {out_dir}"
        )
        time.sleep(0.5)
        st.rerun()
