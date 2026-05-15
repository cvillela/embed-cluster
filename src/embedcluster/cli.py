"""Typer CLI entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .config import (
    HdbscanConfig,
    KmeansConfig,
    LeidenConfig,
    RunPaths,
    SharedConfig,
)
from .export import create_run_dirs
from .validation import validate_inputs


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
    from .pipelines.leiden_pipeline import run_leiden_pipeline

    run_paths = create_run_dirs(out)
    dataset_info = validate_inputs(embeddings, metadata)
    run_leiden_pipeline(shared_cfg, leiden_cfg, run_paths, dataset_info)


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
    from .pipelines.hdbscan_pipeline import run_hdbscan_pipeline

    run_paths = create_run_dirs(out)
    dataset_info = validate_inputs(embeddings, metadata)
    run_hdbscan_pipeline(shared_cfg, hdbscan_cfg, run_paths, dataset_info)


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
    from .pipelines.kmeans_pipeline import run_kmeans_pipeline

    run_paths = create_run_dirs(out)
    dataset_info = validate_inputs(embeddings, metadata)
    run_kmeans_pipeline(shared_cfg, kmeans_cfg, run_paths, dataset_info)


@app.command()
def dedupe(
    embeddings: Path = typer.Option(..., "--embeddings", exists=False, dir_okay=False),
    out: Path = typer.Option(..., "--out", file_okay=False),
    threshold: float = typer.Option(0.98, "--threshold"),
    chunk_size: int = typer.Option(2048, "--chunk-size"),
):
    """Find near-duplicate embeddings via GPU range search + connected components.

    Writes ``dedupe.parquet`` (row_id, dup_group_id, group_size, is_canonical)
    plus ``run_config.json`` and ``metrics.json``. Singletons are kept in the
    manifest with ``is_canonical=True`` so downstream joins are trivial.

    --chunk-size controls VRAM use: chunk * N * 4 bytes for the scores buffer
    plus ~2 * (N * D * 4) for X and temps. Default tuned for 16 GB GPUs;
    raise on bigger cards for fewer chunks (linear speedup).
    """
    if not (0.0 < threshold <= 1.0):
        raise typer.BadParameter("--threshold must be in (0, 1]")
    if chunk_size <= 0:
        raise typer.BadParameter("--chunk-size must be positive")
    from .dedupe import run_dedupe

    run_dedupe(
        embeddings_path=embeddings,
        out=out,
        threshold=threshold,
        chunk_size=chunk_size,
    )


@app.command("dedupe-remove")
def dedupe_remove(
    embeddings: Path = typer.Option(..., "--embeddings", exists=False, dir_okay=False),
    metadata: Path = typer.Option(..., "--metadata", exists=False, dir_okay=False),
    manifest: Path = typer.Option(..., "--manifest", exists=False, dir_okay=False),
    out: Path = typer.Option(..., "--out", file_okay=False),
    strategy: str = typer.Option(..., "--strategy"),
    metadata_field: Optional[str] = typer.Option(None, "--metadata-field"),
    duration_order: str = typer.Option("longest", "--duration-order"),
    k: Optional[int] = typer.Option(None, "--k"),
    selection: str = typer.Option("most_similar", "--selection"),
    random_state: int = typer.Option(42, "--random-state"),
    write_chunk: int = typer.Option(4096, "--write-chunk"),
):
    """Apply a dedupe manifest to produce deduped_embeddings.npy + .jsonl.

    Strategies (one per run):
      canonical | metadata | duration | limit-k. See module docstring for
      semantics. ``--k`` is required for duration / limit-k and optional for
      metadata (stacks limit-k on the same-field survivors).
    """
    if strategy not in ("canonical", "metadata", "duration", "limit-k"):
        raise typer.BadParameter(
            "--strategy must be canonical|metadata|duration|limit-k"
        )
    if duration_order not in ("longest", "shortest"):
        raise typer.BadParameter("--duration-order must be longest|shortest")
    if selection not in ("most_similar", "most_distant", "random"):
        raise typer.BadParameter(
            "--selection must be most_similar|most_distant|random"
        )
    if strategy == "metadata" and not metadata_field:
        raise typer.BadParameter("--metadata-field required for strategy=metadata")
    if strategy in ("duration", "limit-k") and (k is None or k < 1):
        raise typer.BadParameter(f"--k (>=1) required for strategy={strategy}")
    if strategy == "metadata" and k is not None and k < 1:
        raise typer.BadParameter("--k must be >= 1 when stacked on metadata")
    if write_chunk <= 0:
        raise typer.BadParameter("--write-chunk must be positive")

    from .dedupe_remove import run_dedupe_remove

    run_dedupe_remove(
        embeddings_path=embeddings,
        metadata_path=metadata,
        manifest_path=manifest,
        out=out,
        strategy=strategy,  # type: ignore[arg-type]
        metadata_field=metadata_field,
        duration_order=duration_order,  # type: ignore[arg-type]
        k=k,
        selection=selection,  # type: ignore[arg-type]
        random_state=random_state,
        write_chunk=write_chunk,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
