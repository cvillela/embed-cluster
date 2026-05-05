"""GPU integration test: cuML HDBSCAN pipeline (N=200, D=1536)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from embedcluster.cli import app


runner = CliRunner()


@pytest.fixture(scope="session")
def hdbscan_run(synthetic_dataset, tmp_path_factory):
    """Run the HDBSCAN pipeline once; share outputs across tests in this module."""
    _, emb_path, meta_path = synthetic_dataset
    out = tmp_path_factory.mktemp("hdbscan_out")

    result = runner.invoke(
        app,
        [
            "hdbscan",
            "--embeddings", str(emb_path),
            "--metadata", str(meta_path),
            "--out", str(out),
            "--min-cluster-size", "10",
            "--pca-components", "64",
        ],
    )
    return out, result


@pytest.mark.gpu
def test_hdbscan_cli_runs(hdbscan_run):
    _, result = hdbscan_run
    assert result.exit_code == 0, (
        f"CLI failed (exit {result.exit_code}):\n{result.output}\n{result.exception}"
    )


@pytest.mark.gpu
def test_hdbscan_labels_parquet_exists(hdbscan_run):
    out, _ = hdbscan_run
    assert (out / "labels.parquet").exists()

    df = pd.read_parquet(out / "labels.parquet")
    assert len(df) == 200
    assert set(df["row_id"].tolist()) == set(range(200))


@pytest.mark.gpu
def test_hdbscan_probability_column_exists(hdbscan_run):
    out, _ = hdbscan_run
    df = pd.read_parquet(out / "labels.parquet")
    assert "probability" in df.columns


@pytest.mark.gpu
def test_hdbscan_noise_rows_allowed(hdbscan_run):
    out, _ = hdbscan_run
    df = pd.read_parquet(out / "labels.parquet")
    assert "is_noise" in df.columns
    # Noise rows must carry cluster_id == -1
    assert df[df["is_noise"]]["cluster_id"].eq(-1).all()
    # Zero-norm rows (0 and 1) must be noise
    assert df.loc[df["row_id"].isin([0, 1]), "is_noise"].all()


@pytest.mark.gpu
def test_hdbscan_pca_files_exist(hdbscan_run):
    out, _ = hdbscan_run
    assert (out / "intermediate" / "pca.npy").exists(), "pca.npy not written"
    # pca.npy is a raw memmap binary (not npy-format); load via np.memmap
    pca = np.memmap(out / "intermediate" / "pca.npy", dtype="float32", mode="r", shape=(200, 64))
    assert pca.shape == (200, 64), f"Expected (200, 64), got {pca.shape}"


@pytest.mark.gpu
def test_hdbscan_required_output_files(hdbscan_run):
    out, _ = hdbscan_run
    for rel in [
        "labels.parquet",
        "hdbscan_cluster_persistence.parquet",
        "cluster_summary.parquet",
        "run_config.json",
        "preflight.json",
        "metrics.json",
        "intermediate/norms.npy",
        "intermediate/normalized.npy",
        "intermediate/pca.npy",
        "intermediate/pca_normalized.npy",
        "logs/run.log",
    ]:
        assert (out / rel).exists(), f"Missing required output: {rel}"
