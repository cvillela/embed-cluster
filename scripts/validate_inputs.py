#!/usr/bin/env python3
"""Validate embeddings.npy and metadata.jsonl; print DatasetInfo as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate embeddings.npy and metadata.jsonl for embedcluster."
    )
    parser.add_argument("--embeddings", required=True, type=Path, metavar="PATH")
    parser.add_argument("--metadata", required=True, type=Path, metavar="PATH")
    args = parser.parse_args()

    from embedcluster.validation import InputValidationError, validate_inputs

    try:
        info = validate_inputs(args.embeddings, args.metadata)
    except InputValidationError as exc:
        print(f"Validation FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

    result = {
        "n_rows": info.n_rows,
        "n_dims": info.n_dims,
        "dtype": info.dtype,
        "n_zero_norm": info.n_zero_norm,
        "has_nan": info.has_nan,
        "has_inf": info.has_inf,
        "embeddings_path": str(info.embeddings_path),
        "metadata_path": str(info.metadata_path),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
