"""Typer CLI entrypoint. Pipeline functions are stubbed in Phase 1."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .config import (
    HdbscanConfig,
    KmeansConfig,
    LeidenConfig,
    SharedConfig,
)


app = typer.Typer(
    name="embedcluster",
    no_args_is_help=True,
    add_completion=False,
    help="GPU-accelerated clustering for pre-extracted embeddings.",
)


def _parse_batch_size(value: str) -> int | str:
    if value == "auto":
        return "auto"
    try:
        n = int(value)
    except ValueError as e:
        raise typer.BadParameter("--batch-size must be 'auto' or an integer") from e
    if n <= 0:
        raise typer.BadParameter("--batch-size must be positive")
    return n


def _parse_optional_int(value: str) -> Optional[int]:
    if value is None:
        return None
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError as e:
        raise typer.BadParameter(f"expected integer or 'none', got {value!r}") from e


@app.command()
def leiden(
    embeddings: Path = typer.Option(..., "--embeddings", exists=False, dir_okay=False),
    metadata: Path = typer.Option(..., "--metadata", exists=False, dir_okay=False),
    out: Path = typer.Option(..., "--out", file_okay=False),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize"),
    batch_size: str = typer.Option("auto", "--batch-size"),
    random_state: int = typer.Option(42, "--random-state"),
    sample_metrics: int = typer.Option(50_000, "--sample-metrics"),
    export_jsonl: bool = typer.Option(False, "--export-jsonl"),
    k: int = typer.Option(50, "--k"),
    resolution: float = typer.Option(1.0, "--resolution"),
    min_similarity: float = typer.Option(0.0, "--min-similarity"),
):
    """Run Leiden graph clustering (cuVS kNN -> mutual kNN -> cuGraph Leiden)."""
    shared_cfg = SharedConfig(
        embeddings=embeddings,
        metadata=metadata,
        out=out,
        normalize=normalize,
        batch_size=_parse_batch_size(batch_size),
        random_state=random_state,
        sample_metrics=sample_metrics,
        export_jsonl=export_jsonl,
    )
    leiden_cfg = LeidenConfig(k=k, resolution=resolution, min_similarity=min_similarity)
    raise NotImplementedError("Leiden pipeline implemented in Phase 5.")


@app.command()
def hdbscan(
    embeddings: Path = typer.Option(..., "--embeddings", exists=False, dir_okay=False),
    metadata: Path = typer.Option(..., "--metadata", exists=False, dir_okay=False),
    out: Path = typer.Option(..., "--out", file_okay=False),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize"),
    batch_size: str = typer.Option("auto", "--batch-size"),
    random_state: int = typer.Option(42, "--random-state"),
    sample_metrics: int = typer.Option(50_000, "--sample-metrics"),
    export_jsonl: bool = typer.Option(False, "--export-jsonl"),
    min_cluster_size: int = typer.Option(50, "--min-cluster-size"),
    min_samples: str = typer.Option("none", "--min-samples"),
    pca_components: str = typer.Option("128", "--pca-components"),
    cluster_selection: str = typer.Option("eom", "--cluster-selection"),
):
    """Run cuML HDBSCAN, optionally preceded by PCA."""
    if cluster_selection not in ("eom", "leaf"):
        raise typer.BadParameter("--cluster-selection must be 'eom' or 'leaf'")
    shared_cfg = SharedConfig(
        embeddings=embeddings,
        metadata=metadata,
        out=out,
        normalize=normalize,
        batch_size=_parse_batch_size(batch_size),
        random_state=random_state,
        sample_metrics=sample_metrics,
        export_jsonl=export_jsonl,
    )
    hdbscan_cfg = HdbscanConfig(
        min_cluster_size=min_cluster_size,
        min_samples=_parse_optional_int(min_samples),
        pca_components=_parse_optional_int(pca_components),
        cluster_selection=cluster_selection,  # type: ignore[arg-type]
    )
    raise NotImplementedError("HDBSCAN pipeline implemented in Phase 4.")


@app.command()
def kmeans(
    embeddings: Path = typer.Option(..., "--embeddings", exists=False, dir_okay=False),
    metadata: Path = typer.Option(..., "--metadata", exists=False, dir_okay=False),
    out: Path = typer.Option(..., "--out", file_okay=False),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize"),
    batch_size: str = typer.Option("auto", "--batch-size"),
    random_state: int = typer.Option(42, "--random-state"),
    sample_metrics: int = typer.Option(50_000, "--sample-metrics"),
    export_jsonl: bool = typer.Option(False, "--export-jsonl"),
    n_clusters: Optional[int] = typer.Option(None, "--n-clusters"),
    target_cluster_size: int = typer.Option(1000, "--target-cluster-size"),
    max_iter: int = typer.Option(300, "--max-iter"),
    nredo: int = typer.Option(1, "--nredo"),
):
    """Run FAISS GPU spherical KMeans."""
    shared_cfg = SharedConfig(
        embeddings=embeddings,
        metadata=metadata,
        out=out,
        normalize=normalize,
        batch_size=_parse_batch_size(batch_size),
        random_state=random_state,
        sample_metrics=sample_metrics,
        export_jsonl=export_jsonl,
    )
    kmeans_cfg = KmeansConfig(
        n_clusters=n_clusters,
        target_cluster_size=target_cluster_size,
        max_iter=max_iter,
        nredo=nredo,
    )
    raise NotImplementedError("KMeans pipeline implemented in Phase 3.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
