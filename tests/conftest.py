"""Shared pytest fixtures and marker registration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: requires a CUDA GPU and RAPIDS / FAISS GPU runtime",
    )


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory):
    """Write synthetic embeddings.npy + metadata.jsonl to a temp directory.

    N=200, D=1536: 3 tight clusters + 10 noise-like points + 2 zero-norm rows.
    """
    base = tmp_path_factory.mktemp("synthetic")
    N, D = 200, 1536

    rng = np.random.default_rng(42)
    centers = rng.standard_normal((3, D))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    X = np.vstack(
        [
            centers[0] + 0.05 * rng.standard_normal((60, D)),
            centers[1] + 0.05 * rng.standard_normal((60, D)),
            centers[2] + 0.05 * rng.standard_normal((70, D)),
            rng.standard_normal((10, D)),
        ]
    ).astype("float32")

    # Two zero-norm rows
    X[0] = 0.0
    X[1] = 0.0

    emb_path = base / "embeddings.npy"
    np.save(emb_path, X)

    meta_path = base / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for i in range(N):
            f.write(json.dumps({"idx": i}) + "\n")

    return base, emb_path, meta_path
