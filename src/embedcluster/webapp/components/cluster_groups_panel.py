"""Browse clusters as a paginated grid with audio audition.

Mirrors the dedupe group panel: multiple clusters per page, one global
metadata-dedupe filter applied to all, per-cluster member pagination.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from embedcluster.webapp import audio_sampler, metadata_loader
from embedcluster.webapp.components import similarity_panel
from embedcluster.webapp.run_loader import RunBundle

_DEFAULT_CLUSTERS_PER_PAGE = 5
_MAX_CLUSTERS_PER_PAGE = 30
_DEFAULT_MEMBERS_PER_PAGE = 8
_MAX_MEMBERS_PER_PAGE = 32
_COLS_PER_GROUP = 2

_DEFAULT_SORT: dict[str, str] = {
    "kmeans": "cosine_to_centroid_mean",
    "hdbscan": "persistence",
    "leiden": "mean_neighbor_similarity_mean",
}


def _seed_key(method: str) -> str:
    return f"cgrid_seed_{method}"


def _member_page_key(method: str, cluster_id: int) -> str:
    return f"cgrid_member_page_{method}_{cluster_id}"


def _summary_chips(method: str, row: pd.Series) -> None:
    chips: list[tuple[str, str]] = [
        ("cluster_id", str(int(row["cluster_id"]))),
        ("size", f"{int(row['size']):,}"),
    ]
    if method == "kmeans":
        chips.append(("cosine_mean", f"{row['cosine_to_centroid_mean']:.4f}"))
    elif method == "hdbscan":
        if pd.notna(row.get("persistence")):
            chips.append(("persistence", f"{row['persistence']:.4f}"))
        chips.append(("prob_mean", f"{row['probability_mean']:.4f}"))
    elif method == "leiden":
        chips.append(("nbr_sim_mean", f"{row['mean_neighbor_similarity_mean']:.4f}"))
    chips.append(("emb_norm_mean", f"{row['embedding_norm_mean']:.3f}"))

    cols = st.columns(len(chips))
    for col, (label, value) in zip(cols, chips):
        col.metric(label, value)


def _render_audio_card(
    row_id: int,
    row: pd.Series,
    audio_field: str,
    quality_col: str | None,
    extra_cols: list[str],
    *,
    button_key: str,
) -> None:
    raw = row.get(audio_field)
    has_path = isinstance(raw, str) and bool(raw)
    resolved = Path(raw).expanduser() if has_path else None

    title = resolved.name if resolved is not None else "(no path)"
    st.markdown(f"**{title}**")

    meta_bits = [f"`row_id` `{int(row_id)}`"]
    if quality_col and quality_col in row and pd.notna(row[quality_col]):
        meta_bits.append(f"`{quality_col}` `{float(row[quality_col]):.4f}`")
    st.caption(" · ".join(meta_bits))

    if not has_path:
        st.warning("missing audio path in metadata")
    elif not resolved.exists():
        st.warning(f"file not found: `{resolved}`")
    else:
        try:
            st.audio(str(resolved))
        except Exception as e:  # noqa: BLE001
            st.warning(f"audio failed: {e}")

    extras = [(c, row[c]) for c in extra_cols if c in row and pd.notna(row[c])]
    if extras:
        st.markdown("\n".join(f"- **{c}**: {v}" for c, v in extras))

    if resolved is not None:
        with st.expander("path", expanded=False):
            st.code(str(resolved), language=None)

    if st.button("🔎 Find similar", key=button_key, width="stretch"):
        st.session_state[similarity_panel.QUERY_KEY] = int(row_id)
        st.session_state[similarity_panel.AUTO_RUN_KEY] = True
        st.rerun()


def _render_cluster_card(
    bundle: RunBundle,
    table_row: pd.Series,
    meta: pd.DataFrame,
    audio_field: str,
    extra_cols: list[str],
    strategy: str,
    members_per_page: int,
    seed: int,
    dedupe_field: str | None,
) -> int:
    """Render one cluster card. Returns number of audio members shown after dedupe."""
    cluster_id = int(table_row["cluster_id"])
    cluster_size = int(table_row["size"])

    st.markdown(
        f"### Cluster `{cluster_id}` · size `{cluster_size:,}`"
        + ("  · noise/unassigned" if cluster_id < 0 else "")
    )
    _summary_chips(bundle.method, table_row)

    eff_strategy = strategy
    if cluster_id < 0 and strategy not in audio_sampler.NOISE_STRATEGIES:
        eff_strategy = audio_sampler.NOISE_STRATEGIES[0]
        st.caption(
            f"strategy `{strategy}` invalid for noise — using `{eff_strategy}`"
        )

    try:
        sub = audio_sampler.sample_cluster(
            bundle.labels,
            bundle.method,
            cluster_id,
            eff_strategy,
            cluster_size,
            seed=seed,
        )
    except ValueError as e:
        st.error(str(e))
        return 0
    if sub.empty:
        st.info("No rows in this cluster.")
        return 0

    keep_cols = [audio_field]
    keep_cols += [c for c in extra_cols if c in meta.columns and c != audio_field]
    if dedupe_field and dedupe_field in meta.columns and dedupe_field not in keep_cols:
        keep_cols.append(dedupe_field)
    joined = sub.set_index("row_id").join(meta[keep_cols], how="left")

    if dedupe_field and dedupe_field in joined.columns:
        before = len(joined)
        joined = joined.dropna(subset=[dedupe_field]).drop_duplicates(
            subset=[dedupe_field], keep="first"
        )
        after = len(joined)
        if after < before:
            st.caption(
                f"deduped by `{dedupe_field}`: {after} unique of {before} members"
            )
        if joined.empty:
            st.info(f"No rows with non-null `{dedupe_field}` in this cluster.")
            return 0

    total = len(joined)
    n_pages = max(1, (total + int(members_per_page) - 1) // int(members_per_page))
    pkey = _member_page_key(bundle.method, cluster_id)
    cur_page = int(st.session_state.get(pkey, 1))
    if cur_page > n_pages:
        cur_page = 1
        st.session_state[pkey] = 1

    if n_pages > 1:
        pcol1, pcol2 = st.columns([1, 3])
        page = pcol1.number_input(
            "members page",
            min_value=1,
            max_value=n_pages,
            value=cur_page,
            step=1,
            key=pkey,
        )
        start = (int(page) - 1) * int(members_per_page)
        end = start + int(members_per_page)
        pcol2.caption(
            f"Showing members {start + 1}–{min(end, total)} of {total}"
        )
    else:
        start, end = 0, total

    page_df = joined.iloc[start:end]
    quality_col = audio_sampler.QUALITY_COLUMN.get(bundle.method)
    items = list(page_df.iterrows())
    for i in range(0, len(items), _COLS_PER_GROUP):
        cols = st.columns(_COLS_PER_GROUP, gap="medium")
        for col, (row_id, row) in zip(cols, items[i : i + _COLS_PER_GROUP]):
            with col:
                with st.container(border=True):
                    _render_audio_card(
                        int(row_id),
                        row,
                        audio_field,
                        quality_col,
                        extra_cols,
                        button_key=f"sim_btn_{bundle.method}_{cluster_id}_{int(row_id)}",
                    )
        st.write("")

    return total


def render(
    bundle: RunBundle,
    table: pd.DataFrame,
    metadata_path: str,
    audio_field: str | None,
    extra_cols: list[str],
) -> None:
    st.subheader(f"Browse clusters · `{bundle.name}`")

    if table.empty:
        st.info("No clusters in this run.")
        return

    if not metadata_path:
        st.info("Configure `metadata.jsonl` in the sidebar to audition clusters.")
        return
    p = Path(metadata_path).expanduser()
    if not p.exists():
        st.warning(f"metadata path not found: `{p}`")
        return
    try:
        meta = metadata_loader.load_metadata(p)
    except Exception as e:  # noqa: BLE001
        st.error(f"failed to load metadata: {e}")
        return
    if not audio_field or audio_field not in meta.columns:
        st.warning("Pick an audio-path field in the sidebar.")
        return

    sortable = [c for c in table.columns if c != "cluster_id"]
    default_sort = _DEFAULT_SORT.get(bundle.method, "size")
    if default_sort not in sortable:
        default_sort = "size"

    f1, f2, f3, f4, f5 = st.columns([1, 1, 1, 1, 1])
    min_size = f1.number_input(
        "min cluster size",
        min_value=1,
        max_value=int(table["size"].max()),
        value=1,
        key=f"cgrid_min_size_{bundle.method}",
    )
    sort_col = f2.selectbox(
        "sort by",
        sortable,
        index=sortable.index(default_sort),
        key=f"cgrid_sort_{bundle.method}",
    )
    descending = f3.toggle(
        "desc",
        value=True,
        key=f"cgrid_sort_dir_{bundle.method}",
    )
    per_page = f4.slider(
        "clusters per page",
        min_value=1,
        max_value=_MAX_CLUSTERS_PER_PAGE,
        value=_DEFAULT_CLUSTERS_PER_PAGE,
        key=f"cgrid_per_page_{bundle.method}",
    )
    members_per_page = f5.slider(
        "members per page (per cluster)",
        min_value=2,
        max_value=_MAX_MEMBERS_PER_PAGE,
        value=_DEFAULT_MEMBERS_PER_PAGE,
        key=f"cgrid_members_per_page_{bundle.method}",
        help=(
            "Cap audio elements rendered per cluster. Big clusters paginate "
            "internally to avoid loading too many <audio> elements at once."
        ),
    )

    seed_key = _seed_key(bundle.method)
    st.session_state.setdefault(seed_key, 0)
    s1, s2, s3 = st.columns([2, 1, 1])
    include_noise = s1.toggle(
        "include noise / unassigned",
        value=False,
        key=f"cgrid_include_noise_{bundle.method}",
    )
    strategy_pool = list(audio_sampler.ALL_STRATEGIES)
    strategy = s2.selectbox(
        "strategy",
        strategy_pool,
        key=f"cgrid_strategy_{bundle.method}",
    )
    if s3.button("Refresh seed", key=f"cgrid_refresh_{bundle.method}"):
        st.session_state[seed_key] += 1
    seed = int(st.session_state[seed_key])

    dedupe_choices = ["(off)"] + [c for c in meta.columns if c != "row_id"]
    dedupe_field_choice = st.selectbox(
        "Dedupe by metadata field (one example per unique value)",
        dedupe_choices,
        index=0,
        key=f"cgrid_dedupe_{bundle.method}",
        help=(
            "When set, each cluster shows only one member per unique value of "
            "the chosen metadata field. Applies to all clusters."
        ),
    )
    dedupe_field = None if dedupe_field_choice == "(off)" else dedupe_field_choice

    filtered = table[table["size"] >= int(min_size)]
    if not include_noise:
        filtered = filtered[filtered["cluster_id"] >= 0]

    if sort_col in filtered.columns:
        filtered = filtered.sort_values(
            sort_col, ascending=not descending, na_position="last", kind="mergesort"
        )
    filtered = filtered.reset_index(drop=True)

    total_clusters = len(filtered)
    if total_clusters == 0:
        st.info("No clusters match the filters.")
        return
    total_files = int(filtered["size"].sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("clusters (filtered)", f"{total_clusters:,}")
    m2.metric("files in filtered clusters", f"{total_files:,}")
    m3.metric("of total clusters", f"{len(table):,}")

    n_pages = max(1, (total_clusters + int(per_page) - 1) // int(per_page))
    page_state_key = f"cgrid_page_{bundle.method}"
    cur_page = int(st.session_state.get(page_state_key, 1))
    if cur_page > n_pages:
        cur_page = 1
        st.session_state[page_state_key] = 1

    pcol1, pcol2 = st.columns([1, 3])
    page = pcol1.number_input(
        "page",
        min_value=1,
        max_value=n_pages,
        value=cur_page,
        step=1,
        key=page_state_key,
    )
    start = (int(page) - 1) * int(per_page)
    end = start + int(per_page)
    pcol2.caption(
        f"Showing clusters {start + 1}–{min(end, total_clusters)} of {total_clusters} "
        f"(filtered from {len(table)})"
    )

    page_df = filtered.iloc[start:end]
    for _, row in page_df.iterrows():
        with st.container(border=True):
            _render_cluster_card(
                bundle,
                row,
                meta,
                audio_field,
                extra_cols,
                strategy=strategy,
                members_per_page=int(members_per_page),
                seed=seed,
                dedupe_field=dedupe_field,
            )
