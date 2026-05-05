"""Preprocessing: L2 norms, normalized working view, optional PCA, re-normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .logging_utils import get_logger

logger = get_logger(__name__)


def compute_l2_norms(
    embeddings: np.ndarray,
    out_path: Path,
    batch_size: int,
) -> np.memmap:
    """Write one float32 norm per row to out_path. Returns read-only memmap."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = embeddings.shape[0]

    norms_out = np.memmap(out_path, dtype="float32", mode="w+", shape=(n,))
    n_chunks = max(1, (n + batch_size - 1) // batch_size)

    for i, start in enumerate(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        chunk = np.asarray(embeddings[start:end], dtype=np.float32, order="C")
        norms_out[start:end] = np.linalg.norm(chunk, axis=1)
        if i % 10 == 0:
            logger.info(
                "compute_l2_norms: chunk %d/%d (rows %d–%d)", i + 1, n_chunks, start, end
            )

    norms_out.flush()
    return np.memmap(out_path, dtype="float32", mode="r", shape=(n,))


def create_l2_normalized_view(
    embeddings: np.ndarray,
    norms: np.ndarray,
    out_path: Path,
    batch_size: int,
) -> np.memmap:
    """Create float32 normalized matrix. Zero-norm rows become all zeros."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n, d = embeddings.shape[0], embeddings.shape[1]

    normed_out = np.memmap(out_path, dtype="float32", mode="w+", shape=(n, d))
    n_chunks = max(1, (n + batch_size - 1) // batch_size)

    for i, start in enumerate(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        chunk = np.asarray(embeddings[start:end], dtype=np.float32, order="C")
        chunk_norms = np.asarray(norms[start:end], dtype=np.float32)
        zero_mask = chunk_norms == 0.0
        safe_norms = np.where(zero_mask, 1.0, chunk_norms)[:, np.newaxis]
        result = chunk / safe_norms
        result[zero_mask] = 0.0
        normed_out[start:end] = result
        if i % 10 == 0:
            logger.info("create_l2_normalized_view: chunk %d/%d", i + 1, n_chunks)

    normed_out.flush()
    return np.memmap(out_path, dtype="float32", mode="r", shape=(n, d))


def fit_transform_pca_optional(
    X: np.ndarray,
    n_components: int | None,
    out_dir: Path,
    batch_size: int,
    random_state: int,
) -> np.ndarray:
    """If n_components is None, return X unchanged. Else PCA-transform and write pca.npy."""
    if n_components is None:
        return X

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pca.npy"

    if not (50 <= n_components <= 256):
        logger.warning(
            "pca_components=%d is outside recommended range [50, 256]. Proceeding anyway.",
            n_components,
        )

    # Lazy import — cuML only needed at runtime, not during CPU tests.
    from cuml.decomposition import PCA  # type: ignore

    n = X.shape[0]
    max_fit_rows = min(n, 500_000)
    if n > max_fit_rows:
        rng = np.random.default_rng(random_state)
        fit_idx = rng.choice(n, size=max_fit_rows, replace=False)
        fit_idx.sort()
        fit_data = np.asarray(X[fit_idx], dtype=np.float32, order="C")
    else:
        fit_data = np.asarray(X, dtype=np.float32, order="C")

    pca = PCA(
        n_components=n_components,
        svd_solver="auto",
        output_type="numpy",
    )
    logger.info("Fitting PCA (n_components=%d) on %d rows ...", n_components, len(fit_data))
    pca.fit(fit_data)

    explained = float(np.asarray(pca.explained_variance_ratio_).sum())
    logger.info("PCA explained variance ratio sum: %.4f", explained)

    pca_out = np.memmap(out_path, dtype="float32", mode="w+", shape=(n, n_components))
    n_chunks = max(1, (n + batch_size - 1) // batch_size)

    for i, start in enumerate(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        batch = np.asarray(X[start:end], dtype=np.float32, order="C")
        transformed = np.asarray(pca.transform(batch), dtype=np.float32)
        pca_out[start:end] = transformed
        if i % 10 == 0:
            logger.info("PCA transform: chunk %d/%d", i + 1, n_chunks)

    pca_out.flush()
    return np.memmap(out_path, dtype="float32", mode="r", shape=(n, n_components))


def renormalize_matrix(
    X: np.ndarray,
    out_path: Path,
    batch_size: int,
) -> np.memmap:
    """L2-normalize an arbitrary working matrix and write to out_path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n, d = X.shape[0], X.shape[1]

    normed_out = np.memmap(out_path, dtype="float32", mode="w+", shape=(n, d))
    n_chunks = max(1, (n + batch_size - 1) // batch_size)

    for i, start in enumerate(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        chunk = np.asarray(X[start:end], dtype=np.float32, order="C")
        chunk_norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        safe_norms = np.maximum(chunk_norms, 1e-12)
        normed_out[start:end] = chunk / safe_norms
        if i % 10 == 0:
            logger.info("renormalize_matrix: chunk %d/%d", i + 1, n_chunks)

    normed_out.flush()
    return np.memmap(out_path, dtype="float32", mode="r", shape=(n, d))
