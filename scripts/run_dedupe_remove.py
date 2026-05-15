#!/usr/bin/env python3
"""Apply a dedupe manifest to produce stripped embeddings + metadata files."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build deduped_embeddings.npy + deduped_embeddings.jsonl by "
            "removing rows according to a dedupe manifest and a strategy."
        )
    )
    parser.add_argument("--embeddings", required=True, type=Path, metavar="PATH")
    parser.add_argument("--metadata", required=True, type=Path, metavar="PATH")
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        metavar="PATH",
        help="dedupe.parquet from a prior `embedcluster dedupe` run.",
    )
    parser.add_argument("--out", required=True, type=Path, metavar="PATH")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["canonical", "metadata", "duration", "limit-k"],
        help=(
            "canonical: keep canonical only. metadata: keep canonical + members "
            "matching canonical's --metadata-field. duration: keep top-K by "
            "duration order. limit-k: keep canonical + (K-1) ranked by --selection."
        ),
    )
    parser.add_argument(
        "--metadata-field",
        type=str,
        default=None,
        help="Field to match against canonical's value (strategy=metadata).",
    )
    parser.add_argument(
        "--duration-order",
        choices=["longest", "shortest"],
        default="longest",
        help="Sort order for strategy=duration.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help=(
            "Total rows kept per multi-member group. Required for "
            "strategy=duration|limit-k. Optional for strategy=metadata "
            "(stacked: cap same-field survivors)."
        ),
    )
    parser.add_argument(
        "--selection",
        choices=["most_similar", "most_distant", "random"],
        default="most_similar",
        help="Member ranking for limit-k (used by limit-k strategy and stacked metadata+k).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="RNG seed for selection=random.",
    )
    parser.add_argument(
        "--write-chunk",
        type=int,
        default=4096,
        help="Rows per chunk when writing deduped_embeddings.npy.",
    )
    args = parser.parse_args()

    if args.strategy == "metadata" and not args.metadata_field:
        parser.error("--metadata-field is required for strategy=metadata")
    if args.strategy in ("duration", "limit-k") and (args.k is None or args.k < 1):
        parser.error(f"--k (>=1) required for strategy={args.strategy}")
    if args.strategy == "metadata" and args.k is not None and args.k < 1:
        parser.error("--k must be >= 1 when stacked on metadata")
    if args.write_chunk <= 0:
        parser.error("--write-chunk must be positive")

    from embedcluster.dedupe_remove import run_dedupe_remove

    run_dedupe_remove(
        embeddings_path=args.embeddings,
        metadata_path=args.metadata,
        manifest_path=args.manifest,
        out=args.out,
        strategy=args.strategy,
        metadata_field=args.metadata_field,
        duration_order=args.duration_order,
        k=args.k,
        selection=args.selection,
        random_state=args.random_state,
        write_chunk=args.write_chunk,
    )


if __name__ == "__main__":
    main()
