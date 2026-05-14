#!/usr/bin/env python3
"""Run the near-duplicate detection pipeline (GPU range search + CC)."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find near-duplicate embeddings via GPU range search + connected "
            "components. Writes dedupe.parquet, run_config.json, metrics.json."
        )
    )
    parser.add_argument("--embeddings", required=True, type=Path, metavar="PATH")
    parser.add_argument("--out", required=True, type=Path, metavar="PATH")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.98,
        help="Cosine similarity threshold; pairs >= threshold form duplicate edges.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help=(
            "Query rows per matmul chunk. Controls VRAM (chunk*N*4 bytes for "
            "scores). Default tuned for 16 GB GPUs."
        ),
    )
    args = parser.parse_args()

    if not (0.0 < args.threshold <= 1.0):
        parser.error("--threshold must be in (0, 1]")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")

    from embedcluster.dedupe import run_dedupe

    run_dedupe(
        embeddings_path=args.embeddings,
        out=args.out,
        threshold=args.threshold,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
