"""CPU-only validation tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import orjson
import pytest

from embedcluster.validation import InputValidationError, validate_inputs


def _write_metadata(path: Path, n: int) -> None:
    with open(path, "wb") as f:
        for i in range(n):
            f.write(orjson.dumps({"row_id": i, "label": f"item_{i}"}))
            f.write(b"\n")


def _make_dataset(
    tmp_path: Path,
    n: int = 200,
    d: int = 16,
    dtype: str = "float32",
    *,
    inject_nan_row: int | None = None,
    inject_inf_row: int | None = None,
    inject_zero_rows: tuple[int, ...] = (),
    metadata_lines: int | None = None,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d)).astype(dtype)
    for r in inject_zero_rows:
        X[r] = 0
    if inject_nan_row is not None:
        X[inject_nan_row, 0] = np.nan
    if inject_inf_row is not None:
        X[inject_inf_row, 1] = np.inf

    emb_path = tmp_path / "embeddings.npy"
    np.save(emb_path, X)

    meta_path = tmp_path / "metadata.jsonl"
    _write_metadata(meta_path, metadata_lines if metadata_lines is not None else n)

    return emb_path, meta_path


def test_validates_npy_shape_and_metadata_count(tmp_path):
    emb, meta = _make_dataset(tmp_path, n=50, d=8)
    info = validate_inputs(emb, meta)
    assert info.n_rows == 50
    assert info.n_dims == 8
    assert info.dtype == "float32"
    assert info.zero_norm_mask.shape == (50,)
    assert info.has_nan is False
    assert info.has_inf is False


def test_rejects_non_2d_array(tmp_path):
    arr = np.zeros((10, 4, 2), dtype=np.float32)
    emb = tmp_path / "embeddings.npy"
    np.save(emb, arr)
    meta = tmp_path / "metadata.jsonl"
    _write_metadata(meta, 10)
    with pytest.raises(InputValidationError, match="2D"):
        validate_inputs(emb, meta)


def test_rejects_unsupported_dtype(tmp_path):
    arr = np.zeros((4, 4), dtype=np.int32)
    emb = tmp_path / "embeddings.npy"
    np.save(emb, arr)
    meta = tmp_path / "metadata.jsonl"
    _write_metadata(meta, 4)
    with pytest.raises(InputValidationError, match="dtype"):
        validate_inputs(emb, meta)


def test_rejects_metadata_count_mismatch(tmp_path):
    emb, meta = _make_dataset(tmp_path, n=20, d=4, metadata_lines=19)
    with pytest.raises(InputValidationError, match="line count"):
        validate_inputs(emb, meta)


def test_catches_nan(tmp_path):
    emb, meta = _make_dataset(tmp_path, n=30, d=4, inject_nan_row=7)
    with pytest.raises(InputValidationError, match="NaN/Inf"):
        validate_inputs(emb, meta)


def test_catches_inf(tmp_path):
    emb, meta = _make_dataset(tmp_path, n=30, d=4, inject_inf_row=11)
    with pytest.raises(InputValidationError, match="NaN/Inf"):
        validate_inputs(emb, meta)


def test_handles_zero_norm_rows(tmp_path):
    emb, meta = _make_dataset(tmp_path, n=40, d=8, inject_zero_rows=(0, 5, 9))
    info = validate_inputs(emb, meta)
    assert info.n_zero_norm == 3
    assert info.zero_norm_mask[0] is np.True_ or info.zero_norm_mask[0]
    assert info.zero_norm_mask[5]
    assert info.zero_norm_mask[9]
    assert not info.zero_norm_mask[1]


def test_missing_embeddings_file(tmp_path):
    meta = tmp_path / "metadata.jsonl"
    _write_metadata(meta, 1)
    with pytest.raises(InputValidationError, match="Embeddings file not found"):
        validate_inputs(tmp_path / "missing.npy", meta)


def test_missing_metadata_file(tmp_path):
    arr = np.ones((4, 4), dtype=np.float32)
    emb = tmp_path / "embeddings.npy"
    np.save(emb, arr)
    with pytest.raises(InputValidationError, match="Metadata file not found"):
        validate_inputs(emb, tmp_path / "missing.jsonl")


def test_allows_nan_inf_when_disabled(tmp_path):
    emb, meta = _make_dataset(tmp_path, n=30, d=4, inject_nan_row=2, inject_inf_row=3)
    info = validate_inputs(emb, meta, fail_on_nan_inf=False)
    assert info.has_nan
    assert info.has_inf
