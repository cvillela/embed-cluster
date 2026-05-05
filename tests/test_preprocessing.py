"""CPU-only tests for preprocessing norms and normalized view.

PCA and re-normalization tests are GPU-marked (Phase 4+).
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def embed_fixture(tmp_path):
    """200 x 16 float32 matrix with 2 zero-norm rows saved as .npy."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 16)).astype(np.float32)
    X[0] = 0.0
    X[1] = 0.0
    path = tmp_path / "embeddings.npy"
    np.save(path, X)
    arr = np.load(path, mmap_mode="r")
    return arr, tmp_path


# ---------------------------------------------------------------------------
# compute_l2_norms
# ---------------------------------------------------------------------------


def test_norms_shape(embed_fixture):
    from embedcluster.preprocessing import compute_l2_norms

    arr, tmp = embed_fixture
    norms = compute_l2_norms(arr, tmp / "norms.npy", batch_size=64)
    assert norms.shape == (200,)
    assert norms.dtype == np.float32


def test_norms_values(embed_fixture):
    from embedcluster.preprocessing import compute_l2_norms

    arr, tmp = embed_fixture
    norms = compute_l2_norms(arr, tmp / "norms.npy", batch_size=64)
    expected = np.linalg.norm(np.asarray(arr, dtype=np.float32), axis=1)
    np.testing.assert_allclose(norms, expected, rtol=1e-5)


def test_norms_zero_rows(embed_fixture):
    from embedcluster.preprocessing import compute_l2_norms

    arr, tmp = embed_fixture
    norms = compute_l2_norms(arr, tmp / "norms.npy", batch_size=64)
    assert norms[0] == pytest.approx(0.0)
    assert norms[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# create_l2_normalized_view
# ---------------------------------------------------------------------------


def test_normalized_shape_dtype(embed_fixture):
    from embedcluster.preprocessing import compute_l2_norms, create_l2_normalized_view

    arr, tmp = embed_fixture
    norms = compute_l2_norms(arr, tmp / "norms.npy", batch_size=64)
    normed = create_l2_normalized_view(arr, norms, tmp / "normalized.npy", batch_size=64)
    assert normed.shape == arr.shape
    assert normed.dtype == np.float32


def test_normalized_unit_norms(embed_fixture):
    from embedcluster.preprocessing import compute_l2_norms, create_l2_normalized_view

    arr, tmp = embed_fixture
    norms = compute_l2_norms(arr, tmp / "norms.npy", batch_size=64)
    normed = create_l2_normalized_view(arr, norms, tmp / "normalized.npy", batch_size=64)
    # Skip the two zero-norm rows.
    row_norms = np.linalg.norm(np.asarray(normed[2:], dtype=np.float32), axis=1)
    np.testing.assert_allclose(row_norms, 1.0, atol=1e-5)


def test_normalized_zero_rows_stay_zero(embed_fixture):
    from embedcluster.preprocessing import compute_l2_norms, create_l2_normalized_view

    arr, tmp = embed_fixture
    norms = compute_l2_norms(arr, tmp / "norms.npy", batch_size=64)
    normed = create_l2_normalized_view(arr, norms, tmp / "normalized.npy", batch_size=64)
    assert np.all(normed[0] == 0.0), "zero-norm row 0 must remain all-zero"
    assert np.all(normed[1] == 0.0), "zero-norm row 1 must remain all-zero"


def test_normalized_immutable_input(embed_fixture):
    """Raw embeddings memmap must not be modified."""
    from embedcluster.preprocessing import compute_l2_norms, create_l2_normalized_view

    arr, tmp = embed_fixture
    original_row2 = np.array(arr[2], copy=True)
    norms = compute_l2_norms(arr, tmp / "norms.npy", batch_size=64)
    create_l2_normalized_view(arr, norms, tmp / "normalized.npy", batch_size=64)
    np.testing.assert_array_equal(arr[2], original_row2)


# ---------------------------------------------------------------------------
# PCA + renormalize_matrix — GPU required
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_pca_output_shape(embed_fixture):
    from embedcluster.preprocessing import (
        compute_l2_norms,
        create_l2_normalized_view,
        fit_transform_pca_optional,
    )

    arr, tmp = embed_fixture
    norms = compute_l2_norms(arr, tmp / "norms.npy", batch_size=64)
    normed = create_l2_normalized_view(arr, norms, tmp / "normalized.npy", batch_size=64)
    pca_out = fit_transform_pca_optional(normed, n_components=8, out_dir=tmp, batch_size=64, random_state=42)
    assert pca_out.shape == (200, 8)
    assert pca_out.dtype == np.float32


@pytest.mark.gpu
def test_pca_renormalized_unit_norms(embed_fixture):
    from embedcluster.preprocessing import (
        compute_l2_norms,
        create_l2_normalized_view,
        fit_transform_pca_optional,
        renormalize_matrix,
    )

    arr, tmp = embed_fixture
    norms = compute_l2_norms(arr, tmp / "norms.npy", batch_size=64)
    normed = create_l2_normalized_view(arr, norms, tmp / "normalized.npy", batch_size=64)
    pca_out = fit_transform_pca_optional(normed, n_components=8, out_dir=tmp, batch_size=64, random_state=42)
    renormed = renormalize_matrix(pca_out, tmp / "pca_normalized.npy", batch_size=64)
    row_norms = np.linalg.norm(np.asarray(renormed, dtype=np.float32), axis=1)
    np.testing.assert_allclose(row_norms, 1.0, atol=1e-5)
