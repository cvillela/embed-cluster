"""GPU integration test: cuGraph Leiden pipeline (N=200, D=1536)."""

from __future__ import annotations

import pandas as pd
import pytest
from typer.testing import CliRunner

from embedcluster.cli import app


runner = CliRunner()


@pytest.fixture(scope="session")
def leiden_run(synthetic_dataset, tmp_path_factory):
    """Run the Leiden pipeline once; share outputs across tests in this module."""
    _, emb_path, meta_path = synthetic_dataset
    out = tmp_path_factory.mktemp("leiden_out")

    result = runner.invoke(
        app,
        [
            "leiden",
            "--embeddings", str(emb_path),
            "--metadata", str(meta_path),
            "--out", str(out),
            "--k", "20",
            "--resolution", "1.0",
        ],
    )
    return out, result


@pytest.mark.gpu
def test_leiden_cli_runs(leiden_run):
    _, result = leiden_run
    assert result.exit_code == 0, (
        f"CLI failed (exit {result.exit_code}):\n{result.output}\n{result.exception}"
    )


@pytest.mark.gpu
def test_leiden_labels_parquet_exists(leiden_run):
    out, _ = leiden_run
    assert (out / "labels.parquet").exists()

    df = pd.read_parquet(out / "labels.parquet")
    assert len(df) == 200
    assert set(df["row_id"].tolist()) == set(range(200))


@pytest.mark.gpu
def test_leiden_some_non_negative_clusters(leiden_run):
    out, _ = leiden_run
    df = pd.read_parquet(out / "labels.parquet")
    assert (df["cluster_id"] >= 0).any(), "Expected at least one non-noise cluster"


@pytest.mark.gpu
def test_leiden_noise_rows_have_minus_one(leiden_run):
    out, _ = leiden_run
    df = pd.read_parquet(out / "labels.parquet")
    # Zero-norm rows (0 and 1) must be noise
    assert df.loc[df["row_id"].isin([0, 1]), "is_noise"].all()
    assert df.loc[df["row_id"].isin([0, 1]), "cluster_id"].eq(-1).all()


@pytest.mark.gpu
def test_leiden_required_output_files(leiden_run):
    out, _ = leiden_run
    for rel in [
        "labels.parquet",
        "cluster_summary.parquet",
        "run_config.json",
        "preflight.json",
        "metrics.json",
        "intermediate/norms.npy",
        "intermediate/normalized.npy",
        "intermediate/knn_indices.npy",
        "intermediate/knn_scores.npy",
        "intermediate/mutual_edges.parquet",
        "logs/run.log",
    ]:
        assert (out / rel).exists(), f"Missing required output: {rel}"
