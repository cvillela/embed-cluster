"""Browse duplicate groups from a DedupeBundle: paginated list + audio audition."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from embedcluster.webapp import metadata_loader
from embedcluster.webapp.run_loader import DedupeBundle

_DEFAULT_PER_PAGE = 5
_MAX_PER_PAGE = 30
_DEFAULT_MEMBERS_PER_PAGE = 8
_MAX_MEMBERS_PER_PAGE = 32
_COLS_PER_GROUP = 2


def _audio_card(
    row_id: int,
    row: pd.Series,
    audio_field: str,
    extra_cols: list[str],
    is_canonical: bool,
) -> None:
    raw = row.get(audio_field) if audio_field else None
    has_path = isinstance(raw, str) and bool(raw)
    resolved = Path(raw).expanduser() if has_path else None
    title = resolved.name if resolved is not None else "(no path)"

    star = " ★ canonical" if is_canonical else ""
    st.markdown(f"**{title}**{star}")
    st.caption(f"`row_id` `{row_id}`")

    if not audio_field:
        st.caption("no audio field configured")
    elif not has_path:
        st.warning("missing audio path")
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


def _render_group(
    bundle: DedupeBundle,
    dup_group_id: int,
    group_size: int,
    canonical_row_id: int,
    meta: pd.DataFrame,
    audio_field: str | None,
    extra_cols: list[str],
    members_per_page: int,
) -> None:
    members = bundle.members(int(dup_group_id))
    st.markdown(
        f"### Group `{int(dup_group_id)}` · size `{group_size}` · "
        f"canonical `row_id={int(canonical_row_id)}`"
    )

    keep_cols = [c for c in [audio_field, *extra_cols] if c and c in meta.columns]
    keep_cols = list(dict.fromkeys(keep_cols))
    if keep_cols:
        joined = members.set_index("row_id").join(meta[keep_cols], how="left")
    else:
        joined = members.set_index("row_id")

    total = len(joined)
    n_pages = max(1, (total + int(members_per_page) - 1) // int(members_per_page))
    page_key = f"dup_member_page_{int(dup_group_id)}"
    cur_page = int(st.session_state.get(page_key, 1))
    if cur_page > n_pages:
        cur_page = 1
        st.session_state[page_key] = 1

    if n_pages > 1:
        pcol1, pcol2 = st.columns([1, 3])
        page = pcol1.number_input(
            "members page",
            min_value=1,
            max_value=n_pages,
            value=cur_page,
            step=1,
            key=page_key,
        )
        start = (int(page) - 1) * int(members_per_page)
        end = start + int(members_per_page)
        pcol2.caption(
            f"Showing members {start + 1}–{min(end, total)} of {total} "
            f"(canonical always on page 1)"
        )
    else:
        start, end = 0, total

    page_df = joined.iloc[start:end]
    items = list(page_df.iterrows())
    for i in range(0, len(items), _COLS_PER_GROUP):
        cols = st.columns(_COLS_PER_GROUP, gap="medium")
        for col, (row_id, row) in zip(cols, items[i : i + _COLS_PER_GROUP]):
            with col:
                with st.container(border=True):
                    _audio_card(
                        int(row_id),
                        row,
                        audio_field or "",
                        extra_cols,
                        is_canonical=bool(row.get("is_canonical", False)),
                    )
        st.write("")


def render(
    bundle: DedupeBundle,
    metadata_path: str,
    audio_field: str | None,
    extra_cols: list[str],
) -> None:
    st.subheader(f"Browse duplicate groups · `{bundle.name}`")
    cols = st.columns(5)
    cols[0].metric("threshold", f"{bundle.threshold:.4f}")
    cols[1].metric("rows", f"{int(bundle.metrics.get('n_rows', 0)):,}")
    cols[2].metric("dup groups", int(bundle.metrics.get("n_multi_member_groups", 0)))
    cols[3].metric("dup rows", int(bundle.metrics.get("n_duplicate_rows", 0)))
    cols[4].metric("removable", int(bundle.metrics.get("n_removable_rows", 0)))

    groups = bundle.groups
    if groups.empty:
        st.info("No duplicate groups in this run.")
        return

    if not metadata_path:
        st.info("Configure `metadata.jsonl` in the sidebar to audition groups.")
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

    f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
    min_size = f1.number_input(
        "min group_size",
        min_value=2,
        max_value=int(groups["group_size"].max()),
        value=2,
        key="dup_min_size",
    )
    sort_choice = f2.selectbox(
        "sort",
        ["size desc", "size asc", "group_id asc"],
        index=0,
        key="dup_sort",
    )
    per_page = f3.slider(
        "groups per page",
        min_value=1,
        max_value=_MAX_PER_PAGE,
        value=_DEFAULT_PER_PAGE,
        key="dup_per_page",
    )
    members_per_page = f4.slider(
        "members per page (per group)",
        min_value=2,
        max_value=_MAX_MEMBERS_PER_PAGE,
        value=_DEFAULT_MEMBERS_PER_PAGE,
        key="dup_members_per_page",
        help=(
            "Cap audio elements rendered per group. Big groups paginate "
            "internally to avoid loading too many <audio> elements at once."
        ),
    )

    filtered = groups[groups["group_size"] >= int(min_size)]
    if sort_choice == "size desc":
        filtered = filtered.sort_values(
            ["group_size", "dup_group_id"], ascending=[False, True]
        )
    elif sort_choice == "size asc":
        filtered = filtered.sort_values(
            ["group_size", "dup_group_id"], ascending=[True, True]
        )
    else:
        filtered = filtered.sort_values("dup_group_id", ascending=True)
    filtered = filtered.reset_index(drop=True)

    total = len(filtered)
    if total == 0:
        st.info("No groups match the filters.")
        return
    n_pages = max(1, (total + int(per_page) - 1) // int(per_page))
    cur_page = int(st.session_state.get("dup_page", 1))
    if cur_page > n_pages:
        cur_page = 1
        st.session_state["dup_page"] = 1

    pcol1, pcol2 = st.columns([1, 3])
    page = pcol1.number_input(
        "page",
        min_value=1,
        max_value=n_pages,
        value=cur_page,
        step=1,
        key="dup_page",
    )
    start = (int(page) - 1) * int(per_page)
    end = start + int(per_page)
    pcol2.caption(
        f"Showing groups {start + 1}–{min(end, total)} of {total} "
        f"(filtered from {len(groups)})"
    )

    page_df = filtered.iloc[start:end]
    for _, g in page_df.iterrows():
        with st.container(border=True):
            _render_group(
                bundle,
                int(g["dup_group_id"]),
                int(g["group_size"]),
                int(g["canonical_row_id"]),
                meta,
                audio_field,
                extra_cols,
                members_per_page=int(members_per_page),
            )
