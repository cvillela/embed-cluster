#!/usr/bin/env python3
"""Run the FAISS GPU spherical KMeans pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def _parse_batch_size(value: str) -> int | str:
    if value == "auto":
        return "auto"
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("--batch-size must be positive")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FAISS GPU spherical KMeans."
    )
    # Shared parameters
    parser.add_argument("--embeddings", required=True, type=Path, metavar="PATH")
    parser.add_argument("--metadata", required=True, type=Path, metavar="PATH")
    parser.add_argument("--out", required=True, type=Path, metavar="PATH")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=str, default="auto", metavar="INT|auto")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample-metrics", type=int, default=50_000)
    parser.add_argument("--export-jsonl", action="store_true", default=False)
    # KMeans-specific
    parser.add_argument("--n-clusters", type=int, default=None, metavar="INT")
    parser.add_argument("--target-cluster-size", type=int, default=1000)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--nredo", type=int, default=1)
    args = parser.parse_args()

    from embedcluster.config import KmeansConfig, SharedConfig
    from embedcluster.export import create_run_dirs
    from embedcluster.pipelines.kmeans_pipeline import run_kmeans_pipeline
    from embedcluster.validation import validate_inputs

    shared_cfg = SharedConfig(
        embeddings=args.embeddings,
        metadata=args.metadata,
        out=args.out,
        normalize=args.normalize,
        batch_size=_parse_batch_size(args.batch_size),
        random_state=args.random_state,
        sample_metrics=args.sample_metrics,
        export_jsonl=args.export_jsonl,
    )
    kmeans_cfg = KmeansConfig(
        n_clusters=args.n_clusters,
        target_cluster_size=args.target_cluster_size,
        max_iter=args.max_iter,
        nredo=args.nredo,
    )
    run_paths = create_run_dirs(args.out)
    dataset_info = validate_inputs(args.embeddings, args.metadata)
    run_kmeans_pipeline(shared_cfg, kmeans_cfg, run_paths, dataset_info)


if __name__ == "__main__":
    main()
