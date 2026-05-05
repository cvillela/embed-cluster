#!/usr/bin/env python3
"""Run the Leiden graph clustering pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_batch_size(value: str) -> int | str:
    if value == "auto":
        return "auto"
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("--batch-size must be positive")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Leiden graph clustering (cuVS kNN → mutual kNN → cuGraph Leiden)."
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
    # Leiden-specific
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--min-similarity", type=float, default=0.0)
    args = parser.parse_args()

    from embedcluster.config import LeidenConfig, SharedConfig
    from embedcluster.export import create_run_dirs
    from embedcluster.pipelines.leiden_pipeline import run_leiden_pipeline
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
    leiden_cfg = LeidenConfig(
        k=args.k,
        resolution=args.resolution,
        min_similarity=args.min_similarity,
    )
    run_paths = create_run_dirs(args.out)
    dataset_info = validate_inputs(args.embeddings, args.metadata)
    run_leiden_pipeline(shared_cfg, leiden_cfg, run_paths, dataset_info)


if __name__ == "__main__":
    main()
