"""Per-cluster sample selection strategies for the audio panel.

`sample_cluster` returns a slice of the labels DataFrame; the panel joins it with
metadata to obtain audio paths. Strategies are method-aware via `QUALITY_COLUMN`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QUALITY_COLUMN: dict[str, str] = {
    "kmeans": "cosine_to_centroid",
    "hdbscan": "probability",
    "leiden": "mean_neighbor_similarity",
}

ALL_STRATEGIES: tuple[str, ...] = (
    "representative",
    "boundary",
    "random",
    "stratified-by-norm",
)
NOISE_STRATEGIES: tuple[str, ...] = ("random", "stratified-by-norm")


def strategies_for(cluster_id: int) -> tuple[str, ...]:
    return NOISE_STRATEGIES if cluster_id < 0 else ALL_STRATEGIES


def quality_column(method: str) -> str:
    try:
        return QUALITY_COLUMN[method]
    except KeyError as e:
        raise ValueError(f"unknown method: {method!r}") from e


def _topk(df: pd.DataFrame, col: str, n: int, ascending: bool) -> pd.DataFrame:
    return df.sort_values(col, ascending=ascending, kind="mergesort", na_position="last").head(n)


def _stratified_by_norm(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(df) <= n:
        return df.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
    n_bins = min(n, 8)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(df["embedding_norm"].to_numpy(), qs))
    if len(edges) <= 2:
        take = min(n, len(df))
        return df.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1)))
    bins = np.digitize(df["embedding_norm"].to_numpy(), edges[1:-1], right=True)
    tagged = df.assign(_bin=bins)
    n_unique_bins = int(tagged["_bin"].nunique())
    per_bin = max(1, n // n_unique_bins)
    parts: list[pd.DataFrame] = []
    for _, sub in tagged.groupby("_bin", sort=True):
        take = min(per_bin, len(sub))
        parts.append(sub.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
    out = pd.concat(parts).drop(columns="_bin")
    if len(out) < n:
        rest = df.drop(out.index, errors="ignore")
        if len(rest):
            extra = rest.sample(
                n=min(n - len(out), len(rest)),
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            out = pd.concat([out, extra])
    return out.head(n)


def sample_cluster(
    labels: pd.DataFrame,
    method: str,
    cluster_id: int,
    strategy: str,
    n: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Return up to ``n`` rows of ``labels`` belonging to ``cluster_id``.

    For ``representative``/``boundary``, rows are ordered by the method-aware quality
    column (descending / ascending). For ``random``/``stratified-by-norm`` the order
    is unspecified. Noise clusters (``cluster_id < 0``) accept only random / stratified.
    """
    if strategy not in ALL_STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy!r}")
    if cluster_id < 0 and strategy not in NOISE_STRATEGIES:
        raise ValueError(
            f"strategy {strategy!r} is not valid for noise cluster {cluster_id}"
        )
    if n <= 0:
        return labels.iloc[0:0]

    sub = labels.loc[labels["cluster_id"] == cluster_id]
    if sub.empty:
        return sub

    rng = np.random.default_rng(seed)
    if strategy == "representative":
        return _topk(sub, quality_column(method), n, ascending=False)
    if strategy == "boundary":
        return _topk(sub, quality_column(method), n, ascending=True)
    if strategy == "random":
        take = min(n, len(sub))
        return sub.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1)))
    if strategy == "stratified-by-norm":
        return _stratified_by_norm(sub, n, rng)
    raise AssertionError("unreachable")
