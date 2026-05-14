"""Background subprocess lifecycle for webapp-launched pipeline jobs.

Used by the dedupe page to spawn ``python -m embedcluster dedupe`` as a
detached process whose status survives Streamlit reruns. Status is mirrored
to a sidecar JSON in the run output directory so refreshes recover it.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

STATUS_FILE_NAME = "_job_status.json"


@dataclass
class JobStatus:
    pid: int
    out: str
    log: str
    cmd: list[str]
    started: float
    finished: Optional[float] = None
    returncode: Optional[int] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_path(cls, path: Path) -> Optional["JobStatus"]:
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            return None
        try:
            return cls(**data)
        except TypeError:
            return None


def _status_path(out_dir: Path) -> Path:
    return Path(out_dir) / STATUS_FILE_NAME


def write_status(out_dir: Path, status: JobStatus) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = _status_path(out_dir)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(status.to_json())
    tmp.replace(p)


def read_status(out_dir: Path) -> Optional[JobStatus]:
    return JobStatus.from_path(_status_path(out_dir))


def is_running(pid: int) -> bool:
    """True iff a live (non-zombie) process exists with this pid.

    Streamlit reruns lose the Popen handle, so the child is never reaped and
    becomes a zombie after exit. ``os.kill(pid, 0)`` returns success for
    zombies, so we additionally consult /proc/<pid>/status on Linux.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("State:"):
                    state = line.split()[1]
                    return state not in ("Z", "X", "x")
    except (FileNotFoundError, PermissionError, OSError):
        pass
    return True


def cancel_job(pid: int) -> None:
    if pid <= 0:
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def tail_log(log_path: Path, n_lines: int = 200, n_bytes: int = 64 * 1024) -> str:
    p = Path(log_path)
    if not p.exists():
        return ""
    try:
        with p.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - n_bytes))
            data = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    lines = data.splitlines()
    return "\n".join(lines[-n_lines:])


def start_dedupe_job(
    embeddings_path: Path,
    out_dir: Path,
    threshold: float,
    chunk_size: int,
) -> JobStatus:
    """Spawn ``embedcluster dedupe`` as detached process. Returns initial status.

    The child writes its own logs to ``out_dir/logs/run.log`` via the pipeline's
    configure_logging. We additionally redirect stdout/stderr to the same file
    in append mode so any unhandled startup errors land in the same place.
    """
    out_dir = Path(out_dir)
    log_path = out_dir / "logs" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "embedcluster",
        "dedupe",
        "--embeddings",
        str(embeddings_path),
        "--out",
        str(out_dir),
        "--threshold",
        str(threshold),
        "--chunk-size",
        str(chunk_size),
    ]
    log_fh = open(log_path, "ab")
    proc = subprocess.Popen(  # noqa: S603 — args are list, no shell
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=str(Path.cwd()),
    )
    status = JobStatus(
        pid=proc.pid,
        out=str(out_dir),
        log=str(log_path),
        cmd=cmd,
        started=time.time(),
    )
    write_status(out_dir, status)
    return status


def reconcile(out_dir: Path) -> Optional[JobStatus]:
    """Refresh status for a run dir.

    If the recorded pid is no longer alive, marks the job as finished. Returns
    the (possibly updated) status, or None if no status file exists.
    """
    status = read_status(out_dir)
    if status is None:
        return None
    if status.finished is not None:
        return status
    if is_running(status.pid):
        return status
    status.finished = time.time()
    artifacts = (Path(out_dir) / "dedupe.parquet").exists() and (
        Path(out_dir) / "metrics.json"
    ).exists()
    status.returncode = 0 if artifacts else 1
    write_status(out_dir, status)
    return status


def is_job_active(out_dir: Path) -> bool:
    status = read_status(out_dir)
    return bool(status and status.finished is None and is_running(status.pid))


def find_active_jobs(runs_root: Path) -> list[JobStatus]:
    """Scan runs_root for any dirs with an active job (alive pid)."""
    runs_root = Path(runs_root)
    if not runs_root.exists():
        return []
    out: list[JobStatus] = []
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        s = read_status(child)
        if s and s.finished is None and is_running(s.pid):
            out.append(s)
    return out
