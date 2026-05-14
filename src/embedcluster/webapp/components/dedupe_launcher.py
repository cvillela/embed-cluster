"""Launcher panel: configure and start ``embedcluster dedupe`` from the webapp.

While a job is alive, shows a tailed log + cancel button and auto-reruns to
update. On completion, the run dir appears in the dedupe run picker.
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from embedcluster.webapp import jobs, run_loader

_REFRESH_SECONDS = 3.0
_DEFAULT_CHUNK_SIZE = 2048
_DEFAULT_THRESHOLD = 0.98


def _suggest_out_name(runs_root: Path, threshold: float) -> str:
    base = f"dedupe_t{int(round(threshold * 100)):03d}"
    candidate = runs_root / base
    if not candidate.exists():
        return base
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{base}_{stamp}"


def _validate_inputs(
    embeddings: str, out_name: str, threshold: float, chunk_size: int
) -> str | None:
    if not embeddings:
        return "Configure `embeddings .npy path` in the sidebar first."
    p = Path(embeddings).expanduser()
    if not p.exists():
        return f"embeddings path not found: `{p}`"
    if not out_name.strip():
        return "Choose an output run name."
    if "/" in out_name or out_name.startswith("."):
        return "Output name must be a single directory name (no slashes, no leading dot)."
    if not (0.0 < threshold <= 1.0):
        return "Threshold must be in (0, 1]."
    if chunk_size <= 0:
        return "chunk_size must be positive."
    return None


def _render_status(status: jobs.JobStatus, out_dir: Path) -> None:
    st.markdown(f"**Job pid `{status.pid}`** · out `{out_dir.name}`")
    elapsed = time.time() - status.started
    st.caption(f"running for {elapsed:.0f}s · log `{status.log}`")
    log_text = jobs.tail_log(Path(status.log), n_lines=200)
    st.code(log_text or "(no log output yet)", language="log")

    cancel = st.button("Cancel", key="dedupe_cancel_btn", type="secondary")
    if cancel:
        jobs.cancel_job(status.pid)
        st.warning("Sent SIGTERM. Allow a few seconds for the process to exit.")
        time.sleep(1.0)
        st.rerun()


def _render_finished(status: jobs.JobStatus, out_dir: Path) -> None:
    elapsed = (status.finished or time.time()) - status.started
    if status.returncode == 0:
        st.success(
            f"Job completed in {elapsed:.0f}s. Run `{out_dir.name}` is ready below."
        )
    else:
        st.error(
            f"Job exited with returncode={status.returncode} after {elapsed:.0f}s. "
            "See log tail."
        )
        st.code(jobs.tail_log(Path(status.log), n_lines=200), language="log")


def render(
    runs_root: Path,
    embeddings_path: str,
    on_finish_select: callable,
) -> None:
    """Render the dedupe launcher.

    Parameters
    ----------
    runs_root : Path
        Where new dedupe run directories are created.
    embeddings_path : str
        Default embeddings.npy path (from sidebar).
    on_finish_select : callable(name: str) -> None
        Invoked with the new run dir name when a job has just completed; lets
        the page auto-select it in the run picker.
    """
    runs_root = Path(runs_root).expanduser()

    with st.expander("Run new dedupe", expanded=True):
        active = jobs.find_active_jobs(runs_root)
        if active:
            status = jobs.reconcile(Path(active[0].out)) or active[0]
            _render_status(status, Path(status.out))
            time.sleep(_REFRESH_SECONDS)
            st.rerun()
            return

        last_out = st.session_state.get("dedupe_last_launched_out")
        if last_out:
            status = jobs.reconcile(Path(last_out))
            if status and status.finished is not None:
                _render_finished(status, Path(last_out))
                if status.returncode == 0:
                    st.session_state.pop("dedupe_last_launched_out", None)
                    on_finish_select(Path(last_out).name)

        st.text_input(
            "embeddings .npy path",
            value=embeddings_path or "",
            key="dedupe_emb_path",
            help="Set in sidebar; can override here.",
        )
        c1, c2, c3 = st.columns([2, 1, 1])
        threshold = c1.slider(
            "threshold (cosine)",
            min_value=0.80,
            max_value=1.00,
            step=0.005,
            value=_DEFAULT_THRESHOLD,
            key="dedupe_threshold",
        )
        chunk_size = c2.number_input(
            "chunk_size",
            min_value=128,
            max_value=32768,
            step=128,
            value=_DEFAULT_CHUNK_SIZE,
            key="dedupe_chunk_size",
        )
        if c3.button("suggest name", key="dedupe_suggest_name"):
            st.session_state["dedupe_out_name"] = _suggest_out_name(
                runs_root, float(threshold)
            )

        st.session_state.setdefault(
            "dedupe_out_name", _suggest_out_name(runs_root, float(threshold))
        )
        out_name = st.text_input(
            "output run dir name (under runs root)", key="dedupe_out_name"
        )

        run_btn = st.button(
            "Run dedupe", key="dedupe_run_btn", type="primary", width="stretch"
        )
        if not run_btn:
            return

        emb = st.session_state["dedupe_emb_path"]
        err = _validate_inputs(emb, out_name, float(threshold), int(chunk_size))
        if err:
            st.error(err)
            return

        out_dir = runs_root / out_name
        out_dir.mkdir(parents=True, exist_ok=True)
        if (out_dir / "dedupe.parquet").exists():
            st.warning(
                f"`{out_dir}` already contains dedupe.parquet. Pick a new name "
                "or delete the existing run before re-running."
            )
            return

        status = jobs.start_dedupe_job(
            embeddings_path=Path(emb).expanduser(),
            out_dir=out_dir,
            threshold=float(threshold),
            chunk_size=int(chunk_size),
        )
        st.session_state["dedupe_last_launched_out"] = str(out_dir)
        st.success(f"Launched dedupe (pid {status.pid}). Output dir: {out_dir}")
        run_loader.clear_cache()
        time.sleep(0.5)
        st.rerun()
