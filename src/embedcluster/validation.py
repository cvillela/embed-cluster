"""Input validation for embeddings.npy and metadata.jsonl."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .config import DatasetInfo
from .io import count_metadata_lines, load_embeddings_mmap
from .logging_utils import get_logger

logger = get_logger(__name__)


_ALLOWED_DTYPES = {np.float16, np.float32, np.float64}


class InputValidationError(ValueError):
    """Raised when input validation fails irrecoverably."""


def _scan_chunks(
    embeddings: np.ndarray,
    chunk_rows: int,
    fail_on_nan_inf: bool,
) -> tuple[np.ndarray, bool, bool]:
    """Scan for NaN/Inf/zero-norm rows in chunks.

    Returns
    -------
    zero_norm_mask : np.ndarray[bool] of length N
    has_nan : bool
    has_inf : bool
    """
    n = embeddings.shape[0]
    zero_norm_mask = np.zeros(n, dtype=bool)
    has_nan = False
    has_inf = False

    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        chunk = np.asarray(embeddings[start:end], dtype=np.float32, order="C")

        nan_rows = np.isnan(chunk).any(axis=1)
        inf_rows = np.isinf(chunk).any(axis=1)
        if nan_rows.any():
            has_nan = True
        if inf_rows.any():
            has_inf = True

        # Compute norms only where the row is finite to avoid NaN/Inf propagation.
        finite_mask = ~(nan_rows | inf_rows)
        norms = np.zeros(end - start, dtype=np.float32)
        if finite_mask.any():
            norms[finite_mask] = np.linalg.norm(chunk[finite_mask], axis=1)
        # zero-norm = finite & (norm == 0).
        zero_norm_mask[start:end] = finite_mask & (norms == 0.0)

        if fail_on_nan_inf and (has_nan or has_inf):
            bad_rows = np.flatnonzero(nan_rows | inf_rows)[:5] + start
            raise InputValidationError(
                f"NaN/Inf detected in embeddings (first offending rows: "
                f"{bad_rows.tolist()}). Refusing to proceed."
            )

    return zero_norm_mask, has_nan, has_inf


def validate_inputs(
    embeddings_path: Path,
    metadata_path: Path,
    chunk_rows: int = 65_536,
    fail_on_nan_inf: bool = True,
) -> DatasetInfo:
    """Validate inputs and return a DatasetInfo summary.

    Rules:
    1. embeddings_path must exist; loaded read-only via mmap.
    2. matrix is 2D with dtype in {float16, float32, float64}.
    3. metadata_path must exist; line count must equal N.
    4. NaN/Inf -> hard error by default.
    5. zero-norm rows are flagged for downstream cluster_id = -1.
    """
    embeddings_path = Path(embeddings_path)
    metadata_path = Path(metadata_path)

    if not embeddings_path.exists():
        raise InputValidationError(f"Embeddings file not found: {embeddings_path}")
    if not metadata_path.exists():
        raise InputValidationError(f"Metadata file not found: {metadata_path}")

    arr = load_embeddings_mmap(embeddings_path)

    if arr.ndim != 2:
        raise InputValidationError(
            f"Embeddings must be a 2D array, got shape {arr.shape!r}"
        )
    if arr.dtype.type not in _ALLOWED_DTYPES:
        raise InputValidationError(
            f"Unsupported dtype {arr.dtype}. Expected float16/float32/float64."
        )

    n_rows, n_dims = int(arr.shape[0]), int(arr.shape[1])

    n_meta = count_metadata_lines(metadata_path)
    if n_meta != n_rows:
        raise InputValidationError(
            f"Metadata line count ({n_meta}) does not match embeddings row count "
            f"({n_rows}). They must be exactly aligned."
        )

    zero_norm_mask, has_nan, has_inf = _scan_chunks(
        arr, chunk_rows=chunk_rows, fail_on_nan_inf=fail_on_nan_inf
    )

    logger.info(
        "Validated %d rows x %d dims (dtype=%s); zero-norm=%d, has_nan=%s, has_inf=%s",
        n_rows,
        n_dims,
        arr.dtype,
        int(zero_norm_mask.sum()),
        has_nan,
        has_inf,
    )

    return DatasetInfo(
        n_rows=n_rows,
        n_dims=n_dims,
        dtype=str(arr.dtype),
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        zero_norm_mask=zero_norm_mask,
        has_nan=has_nan,
        has_inf=has_inf,
    )
