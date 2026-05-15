"""Apply a dedupe manifest to produce stripped (embeddings.npy + jsonl) outputs.

Reads the dedupe manifest written by :mod:`embedcluster.dedupe`
(``dedupe.parquet``: row_id, dup_group_id, group_size, is_canonical) and
removes a subset of rows from the paired ``embeddings.npy`` /
``embeddings.jsonl`` according to a chosen strategy.

Strategies (one per run, --strategy):
    canonical      Keep canonical row only per multi-member group.
    metadata       Keep canonical + members whose --metadata-field value
                   matches the canonical row's value. If --k is also given,
                   additionally cap each group to canonical + (K-1)
                   same-field members ranked by --selection (stacked
                   metadata + limit-k).
    duration       Keep top-K by duration (end_s - start_s) per group,
                   ordered by --duration-order (longest|shortest).
                   Canonical not forced.
    limit-k        Keep canonical + (K-1) non-canonical members ranked by
                   --selection (most_similar | most_distant | random) on
                   cosine to canonical embedding.

Singletons (group_size == 1) always kept.

Outputs (under ``out``):
    deduped_embeddings.npy   stripped float32 array, rows in original order
    deduped_embeddings.jsonl original metadata lines for kept rows, in order
    kept_indices.npy         int64 original row indices retained
    removed_indices.json     audit log: dropped rows + reason + filepath
    run_config.json          inputs (strategy params, source paths)
    metrics.json             counts (n_in, n_kept, n_removed, ...)
    logs/run.log             structured run log
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Optional

import numpy as np
import orjson
import pandas as pd

from .io import (
    iter_metadata_lines,
    load_embeddings_mmap,
    write_json,
)
from .logging_utils import configure_logging, get_logger

logger = get_logger(__name__)

Strategy = Literal["canonical", "metadata", "duration", "limit-k"]
DurationOrder = Literal["longest", "shortest"]
KSelection = Literal["most_similar", "most_distant", "random"]


def _validate_manifest(df: pd.DataFrame, n_emb: int) -> None:
    needed = {"row_id", "dup_group_id", "group_size", "is_canonical"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"dedupe manifest missing columns: {sorted(missing)}")
    if len(df) != n_emb:
        raise ValueError(
            f"manifest rows ({len(df)}) != embeddings rows ({n_emb})"
        )
    rids = df["row_id"].to_numpy()
    if rids.min() != 0 or rids.max() != n_emb - 1 or len(np.unique(rids)) != n_emb:
        raise ValueError("manifest row_id must be a permutation of [0, n)")


def _multi_group_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return manifest rows belonging to multi-member groups, sorted by row_id."""
    return manifest[manifest["group_size"] > 1].sort_values("row_id")


def _read_field_for_rows(
    metadata_path: Path,
    row_ids: set[int],
    field: str,
) -> dict[int, object]:
    """Stream JSONL once, collect ``field`` for the listed row_ids.

    Row id is the line index (0-based). Raises if any requested row missing
    the field. Returns ``{row_id: value}``.
    """
    out: dict[int, object] = {}
    target = len(row_ids)
    for i, rec in enumerate(iter_metadata_lines(metadata_path)):
        if i in row_ids:
            if field not in rec:
                raise KeyError(
                    f"metadata row {i} missing field {field!r}"
                )
            out[i] = rec[field]
            if len(out) == target:
                break
    if len(out) != target:
        raise RuntimeError(
            f"reached EOF after {i + 1} lines but only collected "
            f"{len(out)}/{target} requested rows for field {field!r}"
        )
    return out


def _apply_canonical(manifest: pd.DataFrame) -> np.ndarray:
    """Boolean mask: singletons + canonical rows of multi-member groups."""
    return ((manifest["group_size"] == 1) | manifest["is_canonical"]).to_numpy()


def _select_extras(
    canon_row: int,
    candidates: list[int],
    extras: int,
    selection: KSelection,
    embeddings: Optional[np.ndarray],
    rng: np.random.Generator,
) -> list[int]:
    """Pick up to ``extras`` rows from ``candidates`` (excludes canonical).

    For ``most_similar`` / ``most_distant``, ``embeddings`` must be supplied;
    rows are ranked by cosine to ``canon_row``.
    """
    if extras <= 0 or not candidates:
        return []
    if extras >= len(candidates):
        return list(candidates)
    if selection == "random":
        picked = rng.choice(len(candidates), size=extras, replace=False)
        return [candidates[i] for i in picked]
    if embeddings is None:
        raise ValueError(f"selection={selection!r} requires embeddings")
    v = np.asarray(embeddings[canon_row], dtype=np.float32)
    v_n = np.linalg.norm(v)
    if v_n > 0:
        v = v / v_n
    M = np.asarray(embeddings[candidates], dtype=np.float32)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    M = M / norms
    sims = M @ v
    order_idx = np.argsort(
        -sims if selection == "most_similar" else sims,
        kind="stable",
    )
    return [candidates[i] for i in order_idx[:extras]]


def _apply_metadata(
    manifest: pd.DataFrame,
    metadata_path: Path,
    field: str,
    embeddings: Optional[np.ndarray] = None,
    k: Optional[int] = None,
    selection: KSelection = "most_similar",
    random_state: int = 42,
) -> np.ndarray:
    """Keep canonical + same-field members per multi-member group.

    If ``k`` is given (>=1), additionally cap each group to canonical +
    (k-1) same-field members, ranked by ``selection``.
    """
    if k is not None and k < 1:
        raise ValueError("--k must be >= 1")
    multi = _multi_group_rows(manifest)
    needed = set(int(r) for r in multi["row_id"].tolist())
    values = _read_field_for_rows(metadata_path, needed, field)

    keep = (manifest["group_size"] == 1).to_numpy().copy()
    keep |= manifest["is_canonical"].to_numpy()

    rng = np.random.default_rng(random_state)
    by_group = multi.groupby("dup_group_id")
    n_field_drop = 0
    n_k_drop = 0
    for _, sub in by_group:
        canon_row = int(sub.loc[sub["is_canonical"], "row_id"].iloc[0])
        canon_val = values[canon_row]
        same_field: list[int] = []
        for rid in sub["row_id"].tolist():
            rid = int(rid)
            if rid == canon_row:
                continue
            if values[rid] == canon_val:
                same_field.append(rid)
            else:
                n_field_drop += 1
        if k is None:
            for rid in same_field:
                keep[rid] = True
            continue
        chosen = _select_extras(
            canon_row, same_field, k - 1, selection, embeddings, rng
        )
        n_k_drop += len(same_field) - len(chosen)
        for rid in chosen:
            keep[rid] = True
    if k is None:
        logger.info(
            "metadata strategy: dropped %d members on field=%r",
            n_field_drop, field,
        )
    else:
        logger.info(
            "metadata+limit-k strategy: field=%r dropped_field=%d "
            "dropped_k=%d k=%d selection=%s",
            field, n_field_drop, n_k_drop, k, selection,
        )
    return keep


def _apply_duration(
    manifest: pd.DataFrame,
    metadata_path: Path,
    order: DurationOrder,
    k: int,
) -> np.ndarray:
    """Keep top-K by (end_s - start_s) per multi-member group."""
    if k < 1:
        raise ValueError("--k must be >= 1")
    multi = _multi_group_rows(manifest)
    needed = set(int(r) for r in multi["row_id"].tolist())

    starts: dict[int, float] = {}
    ends: dict[int, float] = {}
    target = len(needed)
    for i, rec in enumerate(iter_metadata_lines(metadata_path)):
        if i in needed:
            if "start_s" not in rec or "end_s" not in rec:
                raise KeyError(f"metadata row {i} missing start_s/end_s")
            starts[i] = float(rec["start_s"])
            ends[i] = float(rec["end_s"])
            if len(starts) == target:
                break
    if len(starts) != target:
        raise RuntimeError(
            f"only collected {len(starts)}/{target} duration rows from JSONL"
        )

    keep = (manifest["group_size"] == 1).to_numpy().copy()
    by_group = multi.groupby("dup_group_id")
    descending = order == "longest"
    for _, sub in by_group:
        rids = [int(r) for r in sub["row_id"].tolist()]
        durs = np.array([ends[r] - starts[r] for r in rids], dtype=np.float64)
        order_idx = np.argsort(-durs if descending else durs, kind="stable")
        chosen = [rids[i] for i in order_idx[:k]]
        for rid in chosen:
            keep[rid] = True
    logger.info("duration strategy: order=%s k=%d", order, k)
    return keep


def _apply_limit_k(
    manifest: pd.DataFrame,
    embeddings: np.ndarray,
    selection: KSelection,
    k: int,
    random_state: int,
) -> np.ndarray:
    """Keep canonical + (K-1) members per multi-member group ranked by selection."""
    if k < 1:
        raise ValueError("--k must be >= 1")
    rng = np.random.default_rng(random_state)
    multi = _multi_group_rows(manifest)

    keep = (manifest["group_size"] == 1).to_numpy().copy()
    keep |= manifest["is_canonical"].to_numpy()

    by_group = multi.groupby("dup_group_id")
    for _, sub in by_group:
        canon_row = int(sub.loc[sub["is_canonical"], "row_id"].iloc[0])
        non_canon = [int(r) for r in sub["row_id"].tolist() if int(r) != canon_row]
        chosen = _select_extras(
            canon_row, non_canon, k - 1, selection, embeddings, rng
        )
        for rid in chosen:
            keep[rid] = True
    logger.info("limit-k strategy: selection=%s k=%d", selection, k)
    return keep


def _write_npy_chunked(
    src: np.ndarray,
    kept_indices: np.ndarray,
    out_path: Path,
    chunk: int = 4096,
) -> None:
    """Write ``src[kept_indices]`` to ``out_path`` in chunks via open_memmap.

    Avoids materializing the whole stripped matrix in RAM.
    """
    n_keep = int(kept_indices.size)
    d = int(src.shape[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    out = np.lib.format.open_memmap(
        tmp,
        mode="w+",
        dtype=src.dtype,
        shape=(n_keep, d),
    )
    for start in range(0, n_keep, chunk):
        end = min(start + chunk, n_keep)
        out[start:end] = src[kept_indices[start:end]]
    out.flush()
    del out
    tmp.replace(out_path)


def _write_jsonl_kept(
    metadata_path: Path,
    kept_set: set[int],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    written = 0
    with open(metadata_path, "rb") as fin, open(tmp, "wb") as fout:
        for i, raw in enumerate(fin):
            if i in kept_set:
                if not raw.endswith(b"\n"):
                    raw = raw + b"\n"
                fout.write(raw)
                written += 1
    tmp.replace(out_path)
    if written != len(kept_set):
        raise RuntimeError(
            f"jsonl write count mismatch: wrote {written}, expected {len(kept_set)}"
        )


def _build_removed_audit(
    manifest: pd.DataFrame,
    keep: np.ndarray,
    metadata_path: Path,
    strategy: Strategy,
) -> list[dict]:
    """Stream JSONL, emit one record per dropped row with file/filepath fields."""
    removed_idx = np.where(~keep)[0]
    removed_set = set(int(r) for r in removed_idx.tolist())
    gid_lookup = manifest.set_index("row_id")["dup_group_id"].to_dict()
    audit: list[dict] = []
    target = len(removed_set)
    if target == 0:
        return audit
    for i, rec in enumerate(iter_metadata_lines(metadata_path)):
        if i in removed_set:
            audit.append({
                "row_id": int(i),
                "dup_group_id": int(gid_lookup[i]),
                "reason": strategy,
                "file": rec.get("file"),
                "filepath": rec.get("filepath"),
            })
            if len(audit) == target:
                break
    return audit


def run_dedupe_remove(
    embeddings_path: Path,
    metadata_path: Path,
    manifest_path: Path,
    out: Path,
    strategy: Strategy,
    metadata_field: Optional[str] = None,
    duration_order: DurationOrder = "longest",
    k: Optional[int] = None,
    selection: KSelection = "most_similar",
    random_state: int = 42,
    write_chunk: int = 4096,
) -> None:
    """End-to-end stripped output build."""
    embeddings_path = Path(embeddings_path)
    metadata_path = Path(metadata_path)
    manifest_path = Path(manifest_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=out / "logs" / "run.log")

    logger.info(
        "dedupe-remove start: embeddings=%s metadata=%s manifest=%s out=%s "
        "strategy=%s field=%s duration_order=%s k=%s selection=%s",
        embeddings_path, metadata_path, manifest_path, out,
        strategy, metadata_field, duration_order, k, selection,
    )

    X = load_embeddings_mmap(embeddings_path)
    if X.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {X.shape!r}")
    n, d = int(X.shape[0]), int(X.shape[1])

    manifest = pd.read_parquet(manifest_path)
    _validate_manifest(manifest, n)
    manifest = manifest.sort_values("row_id").reset_index(drop=True)

    if strategy == "canonical":
        keep = _apply_canonical(manifest)
    elif strategy == "metadata":
        if not metadata_field:
            raise ValueError("--metadata-field required for strategy=metadata")
        keep = _apply_metadata(
            manifest,
            metadata_path,
            metadata_field,
            embeddings=X,
            k=k if k is not None and k >= 1 else None,
            selection=selection,
            random_state=random_state,
        )
    elif strategy == "duration":
        if k is None:
            raise ValueError("--k required for strategy=duration")
        keep = _apply_duration(manifest, metadata_path, duration_order, k)
    elif strategy == "limit-k":
        if k is None:
            raise ValueError("--k required for strategy=limit-k")
        keep = _apply_limit_k(manifest, X, selection, k, random_state)
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")

    if keep.shape != (n,):
        raise RuntimeError(f"keep mask shape {keep.shape} != ({n},)")
    kept_indices = np.where(keep)[0].astype(np.int64)
    n_kept = int(kept_indices.size)
    n_removed = n - n_kept
    logger.info("kept=%d removed=%d (n=%d)", n_kept, n_removed, n)

    out_npy = out / "deduped_embeddings.npy"
    out_jsonl = out / "deduped_embeddings.jsonl"
    out_kept = out / "kept_indices.npy"
    out_removed = out / "removed_indices.json"

    logger.info("writing %s", out_npy)
    _write_npy_chunked(X, kept_indices, out_npy, chunk=write_chunk)

    logger.info("writing %s", out_jsonl)
    _write_jsonl_kept(metadata_path, set(int(i) for i in kept_indices.tolist()), out_jsonl)

    logger.info("writing %s", out_kept)
    np.save(out_kept, kept_indices)

    logger.info("building removed audit -> %s", out_removed)
    audit = _build_removed_audit(manifest, keep, metadata_path, strategy)
    with open(out_removed, "wb") as f:
        f.write(orjson.dumps(audit, option=orjson.OPT_INDENT_2))

    write_json(
        {
            "pipeline": "dedupe-remove",
            "embeddings": str(embeddings_path),
            "metadata": str(metadata_path),
            "manifest": str(manifest_path),
            "strategy": strategy,
            "metadata_field": metadata_field,
            "duration_order": duration_order if strategy == "duration" else None,
            "k": k if strategy in ("duration", "limit-k", "metadata") else None,
            "selection": (
                selection if strategy == "limit-k"
                or (strategy == "metadata" and k is not None)
                else None
            ),
            "random_state": (
                random_state if strategy == "limit-k"
                or (strategy == "metadata" and k is not None)
                else None
            ),
            "n_rows_in": n,
            "n_dims": d,
            "n_rows_kept": n_kept,
            "n_rows_removed": n_removed,
        },
        out / "run_config.json",
    )
    write_json(
        {
            "method": "dedupe-remove",
            "strategy": strategy,
            "n_rows_in": n,
            "n_rows_kept": n_kept,
            "n_rows_removed": n_removed,
            "n_groups_total": int(manifest["dup_group_id"].nunique()),
            "n_multi_member_groups": int(
                (manifest.groupby("dup_group_id")["group_size"].first() > 1).sum()
            ),
        },
        out / "metrics.json",
    )
    logger.info("dedupe-remove done")
