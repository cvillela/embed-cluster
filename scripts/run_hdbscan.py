#!/usr/bin/env python3
"""Run the cuML HDBSCAN clustering pipeline."""

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


def _parse_optional_int(value: str) -> Optional[int]:
    if value is None or value.lower() == "none":
        return None
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cuML HDBSCAN, optionally preceded by PCA."
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
    # HDBSCAN-specific
    parser.add_argument("--min-cluster-size", type=int, default=50)
    parser.add_argument("--min-samples", type=str, default="none", metavar="INT|none")
    parser.add_argument("--pca-components", type=str, default="128", metavar="INT|none")
    parser.add_argument(
        "--cluster-selection", type=str, default="eom", choices=["eom", "leaf"]
    )
    args = parser.parse_args()

    from embedcluster.config import HdbscanConfig, SharedConfig
    from embedcluster.export import create_run_dirs
    from embedcluster.pipelines.hdbscan_pipeline import run_hdbscan_pipeline
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
    hdbscan_cfg = HdbscanConfig(
        min_cluster_size=args.min_cluster_size,
        min_samples=_parse_optional_int(args.min_samples),
        pca_components=_parse_optional_int(args.pca_components),
        cluster_selection=args.cluster_selection,  # type: ignore[arg-type]
    )
    run_paths = create_run_dirs(args.out)
    dataset_info = validate_inputs(args.embeddings, args.metadata)
    run_hdbscan_pipeline(shared_cfg, hdbscan_cfg, run_paths, dataset_info)


if __name__ == "__main__":
    main()
