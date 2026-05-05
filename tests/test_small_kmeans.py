"""GPU integration test: FAISS GPU spherical KMeans pipeline (N=200, D=1536)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from embedcluster.cli import app


runner = CliRunner()


@pytest.fixture(scope="session")
def kmeans_run(synthetic_dataset, tmp_path_factory):
    """Run the KMeans pipeline once; share outputs across all tests in this module."""
    _, emb_path, meta_path = synthetic_dataset
    out = tmp_path_factory.mktemp("kmeans_out")

    result = runner.invoke(
        app,
        [
            "kmeans",
            "--embeddings", str(emb_path),
            "--metadata", str(meta_path),
            "--out", str(out),
            "--target-cluster-size", "50",
        ],
    )
    return out, result


@pytest.mark.gpu
def test_kmeans_cli_runs(kmeans_run):
    _, result = kmeans_run
    assert result.exit_code == 0, (
        f"CLI failed (exit {result.exit_code}):\n{result.output}\n{result.exception}"
    )


@pytest.mark.gpu
def test_kmeans_labels_parquet_exists(kmeans_run):
    out, _ = kmeans_run
    assert (out / "labels.parquet").exists()

    df = pd.read_parquet(out / "labels.parquet")
    assert len(df) == 200
    assert set(df["row_id"].tolist()) == set(range(200))


@pytest.mark.gpu
def test_kmeans_cluster_centers_npy(kmeans_run):
    out, _ = kmeans_run
    centers_path = out / "cluster_centers.npy"
    assert centers_path.exists(), "cluster_centers.npy not written"

    centers = np.load(centers_path)
    assert centers.ndim == 2
    assert centers.shape[1] == 1536


@pytest.mark.gpu
def test_kmeans_cosine_to_centroid_column(kmeans_run):
    out, _ = kmeans_run
    df = pd.read_parquet(out / "labels.parquet")
    assert "cosine_to_centroid" in df.columns


@pytest.mark.gpu
def test_kmeans_valid_rows_non_negative_cluster_ids(kmeans_run):
    out, _ = kmeans_run
    df = pd.read_parquet(out / "labels.parquet")

    valid = df[~df["is_noise"]]
    assert (valid["cluster_id"] >= 0).all(), "Valid rows must have non-negative cluster IDs"
    # rows 0 and 1 are zero-norm → noise; all 198 others should be assigned
    assert len(valid) == 198


@pytest.mark.gpu
def test_kmeans_required_output_files(kmeans_run):
    out, _ = kmeans_run
    for rel in [
        "labels.parquet",
        "cluster_centers.npy",
        "cluster_summary.parquet",
        "run_config.json",
        "preflight.json",
        "metrics.json",
        "intermediate/norms.npy",
        "intermediate/normalized.npy",
        "logs/run.log",
    ]:
        assert (out / rel).exists(), f"Missing required output: {rel}"
