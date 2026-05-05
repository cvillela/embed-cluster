"""Structured logging: file handler + Rich console handler."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler


_FILE_FMT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_CONSOLE_FMT = "%(message)s"
_DATE_FMT = "%Y-%m-%dT%H:%M:%S"

_configured = False


def configure_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> None:
    """Attach handlers to the root logger. Idempotent across calls."""
    global _configured

    root = logging.getLogger()
    root.setLevel(level)

    # Remove handlers we added previously to keep this idempotent.
    for h in list(root.handlers):
        if getattr(h, "_embedcluster_managed", False):
            root.removeHandler(h)

    console = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
        log_time_format="[%X]",
    )
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_CONSOLE_FMT))
    console._embedcluster_managed = True  # type: ignore[attr-defined]
    root.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))
        fh._embedcluster_managed = True  # type: ignore[attr-defined]
        root.addHandler(fh)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger; ensures default config is in place."""
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
