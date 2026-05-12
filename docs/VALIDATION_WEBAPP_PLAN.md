# Validation Web App — Implementation Plan

A local web tool to **load**, **inspect**, and **audibly validate** clustering runs produced by the `embedcluster` pipelines (Leiden / HDBSCAN / KMeans).

This document is the source of truth for what to build. Progress is tracked in `VALIDATION_WEBAPP_PROGRESS.md`. Per-phase agent instructions are in `VALIDATION_WEBAPP_AGENT_PROMPT.md`.

---

## 0. Goals & non-goals

### Goals
1. **Easy run loading** — point at a `runs/` directory, pick a run, load all artifacts.
2. **Method-aware display** — show only the metrics that make sense for the run's method.
3. **Per-cluster audio listening** — sample audio rows from a chosen cluster with meaningful selection strategies (representative / boundary / random).
4. **Interactive 3D UMAP** — when a 6D UMAP (`[x, y, z, r, g, b]`) file is provided, render an orbitable / zoomable scatter colored either by UMAP-RGB or by `cluster_id`.

### Non-goals
- No re-clustering, no embedding extraction, no model inference inside the app.
- No write-back to run directories (read-only). The app may write a small `validation_notes/` sidecar for user annotations (Phase 5, optional).
- No multi-user auth, no remote deploy. Local single-user tool.

---

## 1. Run artifact reference (what the app reads)

For every run under `runs/<run_name>/`:

| File | Always present | Use |
|---|---|---|
| `run_config.json` | yes | pipeline + hyperparams |
| `preflight.json`  | yes | GPU + N, D, dtype |
| `metrics.json`    | yes | method-aware quality numbers |
| `labels.parquet`  | yes | row → cluster + per-row quality |
| `cluster_summary.parquet` | yes | per-cluster size + norm stats |
| `cluster_centers.npy` | KMeans only | centroids `[k, D]` |
| `hdbscan_cluster_persistence.parquet` | HDBSCAN only | per-cluster persistence |
| `intermediate/mutual_edges.parquet` | Leiden only | optional, large — do **not** auto-load |
| `intermediate/knn_*.npy`, `normalized.npy`, `pca*.npy` | varies | **never** auto-load (multi-GB) |
| `logs/run.log` | yes | optional view in UI |

`labels.parquet` columns by method (always `row_id`, `cluster_id`, `method`, `is_noise`, `embedding_norm`):

| Method | Extra columns |
|---|---|
| Leiden  | `graph_degree`, `mean_neighbor_similarity` |
| HDBSCAN | `probability` |
| KMeans  | `cosine_to_centroid` |

External inputs (provided per-session by the user, not stored in the run):
- `metadata.jsonl` — N lines aligned with `row_id`. Contains the **audio path field** (user picks which key, e.g. `audio_path`, `path`, `file`).
- `umap_6d.npy` *(optional)* — shape `[N, 6]`, columns `[x, y, z, r, g, b]`. RGB in `[0, 1]` or `[0, 255]` (auto-detect).

---

## 2. Tech stack

- **Streamlit** — fastest path to a clean local UI with audio + sliders + selectors.
- **Plotly** — `Scatter3d` for the orbitable UMAP (native rotate/zoom, no extra JS).
- **Pandas + PyArrow** — parquet IO.
- **NumPy** — `mmap_mode='r'` for any `.npy` we touch.
- **soundfile** (or just `st.audio` with file paths) — audio playback.
- **No frontend build step. No FastAPI. No React.** Keep it boring.

Add a separate `requirements-webapp.txt` so the clustering env is not polluted unless desired.

---

## 3. Directory layout to create

```
src/embedcluster/webapp/
  __init__.py
  app.py                     # streamlit entrypoint
  run_loader.py              # discover + load run artifacts (cached)
  metadata_loader.py         # metadata.jsonl streaming + row-id resolution
  audio_sampler.py           # per-cluster sample selection strategies
  umap_view.py               # 3D plotly scatter + color modes
  metrics_view.py            # method-aware panels
  components/
    __init__.py
    sidebar.py               # run picker, paths, options
    cluster_table.py
    cluster_panel.py
    audio_panel.py

requirements-webapp.txt
scripts/run_webapp.sh        # `streamlit run src/embedcluster/webapp/app.py`
docs/VALIDATION_WEBAPP_PLAN.md          # this file
docs/VALIDATION_WEBAPP_PROGRESS.md
docs/VALIDATION_WEBAPP_AGENT_PROMPT.md
```

No changes to existing `embedcluster` pipeline code. The webapp imports nothing from the pipelines — it only reads files.

---

## 4. Phases

Each phase ends with the phase tickbox in `VALIDATION_WEBAPP_PROGRESS.md` checked. Phases are sequential — later ones assume earlier ones land.

---

### Phase 0 — Scaffolding & dependencies

**Outcome:** `bash scripts/run_webapp.sh` opens an empty page titled "embedcluster validation".

Tasks:
1. Create `requirements-webapp.txt` with loose pins:
   ```
   streamlit>=1.32
   plotly>=5.20
   pandas>=2.0
   pyarrow>=15
   numpy>=1.24
   soundfile>=0.12
   ```
2. Create `src/embedcluster/webapp/` package skeleton with empty modules listed in §3.
3. Create `app.py` with `st.set_page_config(page_title="embedcluster validation", layout="wide")` and a placeholder `st.title(...)`.
4. Create `scripts/run_webapp.sh`:
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail
   exec streamlit run "$(dirname "$0")/../src/embedcluster/webapp/app.py" "$@"
   ```
   `chmod +x`.

Acceptance:
- App boots locally with no run selected.
- No regression in existing pipeline tests (`pytest -q`).

**Validation**

Agent self-validation (must all pass before ticking agent box):
- [ ] `python -m pip install -r requirements-webapp.txt` completes in the active env.
- [ ] `python -c "import embedcluster.webapp.app"` exits 0.
- [ ] `pytest -q` passes.
- [ ] `bash scripts/run_webapp.sh --server.headless=true` starts Streamlit without errors (kill after the "You can now view your Streamlit app" line). If you cannot run Streamlit at all, say so explicitly.
- [ ] `git status` shows only the intended new files; no `__pycache__`, no `.streamlit/`.

Human validation (user only):
- [ ] Open the local Streamlit URL. Page renders with the title "embedcluster validation". No tracebacks in the terminal.

---

### Phase 1 — Pillar 1: Run discovery & loading

**Outcome:** Sidebar lets the user pick a `runs/` root, lists runs, and selects one. Header shows method, N, dims, k/n_clusters, and timestamps.

Tasks:
1. `run_loader.discover_runs(runs_root: Path) -> list[RunSummary]`:
   - A run is any subdir containing both `run_config.json` and `labels.parquet`.
   - `RunSummary` = `{name, path, method, n_rows, n_clusters, mtime}` (cheap — read only the small JSONs).
2. `run_loader.load_run(path: Path) -> RunBundle`:
   - Lazy fields. Eagerly parse: `run_config.json`, `preflight.json`, `metrics.json`, `cluster_summary.parquet`.
   - Lazy properties: `labels` (parquet), `hdbscan_persistence`, `kmeans_centers` — only read on access.
   - Cache with `@st.cache_resource` keyed by `(path, mtime)` so re-selecting a run is instant.
3. `components/sidebar.py`:
   - Text input: runs root (default `./runs`, persists in `st.session_state`).
   - Selectbox of discovered runs, sorted by mtime desc, labeled `name (method, N, k_or_clusters)`.
   - Text inputs for: `metadata.jsonl` path, `audio_root_dir` (optional prefix to prepend if metadata audio paths are relative), `umap_6d.npy` path (optional).
   - "Reload" button to clear caches.
4. `app.py` header: when a run is selected, render a 3-row strip with method, hyperparams (from `run_config.json`), and basic counts (from `metrics.json` + `preflight.json`).

Acceptance:
- All three existing runs (`hdbscan`, `kmeans_coarse`, `leiden_k50_res1`) appear and load.
- Switching runs does not re-read `labels.parquet` if the same run is reselected.
- Missing optional inputs (metadata, UMAP) do not crash anything — the relevant pillars show "not configured".

**Validation**

Agent self-validation:
- [ ] `python -c "from embedcluster.webapp.run_loader import discover_runs; print(len(discover_runs(__import__('pathlib').Path('runs'))))"` returns `3`.
- [ ] For each of the three runs, write a small ad-hoc test (in-process, do **not** add to `tests/`) that calls `load_run(path)` and asserts `bundle.method`, `len(bundle.cluster_summary) > 0`, and `bundle.metrics["n_rows"] == 630362`.
- [ ] `bash scripts/run_webapp.sh --server.headless=true` starts; sidebar imports do not error in the logs.
- [ ] `pytest -q` still passes.

Human validation:
- [ ] Sidebar lists all three runs with method + N + n_clusters.
- [ ] Selecting each run renders a header strip with the right method and hyperparams.
- [ ] Re-selecting the same run is instant (cache hit).
- [ ] Pointing the metadata / UMAP path inputs at nonexistent paths does not crash the app.

---

### Phase 2 — Pillar 2: Method-aware validation & cluster display

**Outcome:** For each method, only the metrics and panels that make sense are shown. The user can sort/filter clusters and pick one to drill into.

Tasks:
1. `metrics_view.render(bundle)` dispatches by `bundle.method`:
   - **Common header (all methods):**
     - N rows, n_clusters, noise/unassigned count + fraction.
     - Cluster size distribution histogram (log-y toggle).
     - Embedding-norm boxplot per top-N clusters.
   - **Leiden:** `modularity`, `n_mutual_edges`, mean `graph_degree`, distribution of `mean_neighbor_similarity` (overall + per cluster).
   - **HDBSCAN:** `noise_fraction` callout, `min_cluster_size`, persistence table (top/bottom 10 by persistence, joined with size from `cluster_summary`), probability distribution histogram.
   - **KMeans:** `n_clusters`, `target_cluster_size`, `cosine_to_centroid` distribution overall + per-cluster mean/median, "tightest vs loosest cluster" callouts.
2. `components/cluster_table.py`:
   - Sortable dataframe of clusters with method-aware quality columns:
     - Leiden:  `cluster_id, size, mean_neighbor_similarity_mean, graph_degree_mean, embedding_norm_mean`
     - HDBSCAN: `cluster_id, size, persistence, probability_mean, embedding_norm_mean`
     - KMeans:  `cluster_id, size, cosine_to_centroid_mean, cosine_to_centroid_p10, embedding_norm_mean`
   - Quality columns are computed from `labels.parquet` once per run, cached.
   - Row click (or selectbox under the table) sets `st.session_state.selected_cluster_id`.
   - Always include the noise row (`-1`) when applicable.
3. `components/cluster_panel.py`:
   - Given the selected cluster, render: size, method-specific quality summary, top-K representative rows by quality, and a placeholder for the audio panel (filled in Phase 3).

Acceptance:
- Switching from a Leiden run to an HDBSCAN run swaps panels with no leftover Leiden-only widgets.
- Sorting the cluster table by quality picks plausibly tight clusters first.
- No method shows "N/A" rows for metrics it does not produce — those panels are simply absent.

**Validation**

Agent self-validation:
- [ ] For each of the three runs, programmatically call the per-method aggregation function used by the cluster table and assert the expected columns are present and contain no NaN in non-noise rows.
- [ ] Asserts: KMeans table has `cosine_to_centroid_mean`; HDBSCAN table has `persistence` and `probability_mean`; Leiden table has `mean_neighbor_similarity_mean` and `graph_degree_mean`. None of these columns appear cross-method.
- [ ] HDBSCAN cluster table includes a row for `cluster_id == -1` with the noise count from `metrics.json["n_noise"]`.
- [ ] `pytest -q` still passes.

Human validation:
- [ ] Load each run in turn. For each, confirm only its method-appropriate panels appear.
- [ ] Cluster table sort by quality column returns plausible top/bottom clusters (eyeball check).
- [ ] Selecting a cluster from the table updates the cluster panel (size + summary) — audio panel is still a placeholder until Phase 3.
- [ ] HDBSCAN run shows the noise callout with ~81% noise (matches `metrics.json`).

---

### Phase 3 — Pillar 3: Audio fetching & per-cluster listening

**Outcome:** With a selected cluster, the user picks a sampling strategy and gets a clean grid of N audio players plus per-row metadata. Looks polished.

Tasks:
1. `metadata_loader.py`:
   - `load_metadata(path: Path) -> pd.DataFrame` — streams `metadata.jsonl`, returns a DataFrame indexed by row_id. Cache with `@st.cache_resource`.
   - Auto-detect candidate audio-path fields: any string column whose values look like file paths or end with `.wav/.mp3/.flac/.ogg/.m4a`. Let the user override.
   - `resolve_audio_path(rel_or_abs: str, audio_root: Path | None) -> Path` — joins `audio_root` if path is relative.
2. `audio_sampler.py`:
   - `sample_cluster(labels_df, metadata_df, cluster_id, strategy, n) -> list[Sample]`:
     - **representative** — top-N by method-aware quality (kmeans: `cosine_to_centroid`; hdbscan: `probability`; leiden: `mean_neighbor_similarity`).
     - **boundary** — bottom-N by the same column.
     - **random** — uniform sample with fixed seed (toggleable).
     - **stratified-by-norm** — mix across `embedding_norm` quantiles.
   - For noise cluster `-1`: only random / stratified strategies are valid; representative/boundary are disabled.
3. `components/audio_panel.py`:
   - Strategy selector + N selector (default 8, max 24).
   - Refresh-seed button.
   - Grid (e.g. 2 cols × N/2 rows) of cards. Each card:
     - Audio player (`st.audio` with the resolved file path).
     - Compact metadata block: row_id, the quality value used, plus 2–3 user-selected metadata columns (configurable in sidebar).
     - "Open file" link / copy-path button.
   - Top of panel: cluster summary chip (id, size, mean quality).
4. Edge cases:
   - File not found → render the card with a warning, do not crash the grid.
   - Metadata not loaded → audio panel shows a friendly "configure metadata.jsonl in the sidebar" message.

Acceptance:
- For a KMeans cluster, "representative" picks rows with the highest `cosine_to_centroid`; "boundary" picks the lowest.
- For an HDBSCAN noise cluster, only random / stratified strategies are offered.
- Switching clusters keeps strategy + N; resamples automatically.

**Validation**

Agent self-validation:
- [ ] Unit-style assertions on `audio_sampler.sample_cluster` for each method+strategy combination using the actual `labels.parquet` from each run:
  - KMeans + representative: returned `cosine_to_centroid` values are monotonically non-increasing and equal to the top-N when sorted.
  - KMeans + boundary: monotonically non-decreasing.
  - HDBSCAN + representative on a real cluster: returned `probability` ≥ all unselected probabilities in that cluster.
  - Leiden + representative: returned `mean_neighbor_similarity` ≥ all unselected values in that cluster.
  - Random with a fixed seed is deterministic across two calls.
- [ ] `metadata_loader.load_metadata` on a small synthetic JSONL (you write inline, do not commit) returns a row-aligned DataFrame.
- [ ] Audio-path resolver: relative path + `audio_root` joins correctly; absolute path is returned unchanged.
- [ ] If the user has not pointed `metadata.jsonl` at a real file in your environment, skip the live audio check and say so.

Human validation:
- [ ] Configure `metadata.jsonl` and `audio_root` in the sidebar.
- [ ] Pick a KMeans cluster, "representative", N=8 — eight audio players render and the audio sounds intra-cluster consistent.
- [ ] Same cluster, "boundary" — audio sounds clearly less consistent.
- [ ] HDBSCAN noise cluster (`-1`): only random/stratified strategies are offered.
- [ ] A row whose audio file is missing on disk renders a warning card, not a traceback, and the rest of the grid still plays.

---

### Phase 4 — Pillar 4: 3D UMAP plot

**Outcome:** When `umap_6d.npy` is provided, an orbitable Plotly `Scatter3d` renders the cloud, color-toggleable between **UMAP-RGB** and **cluster_id**. Clicking a cluster in the Phase 2 table highlights its points.

Tasks:
1. `umap_view.load_umap(path: Path, n_expected: int) -> np.ndarray`:
   - mmap-load, validate `shape == (n_expected, 6)`, normalize RGB to `[0, 1]` if values exceed 1.
2. `umap_view.render(umap, labels_df, selected_cluster_id, color_mode, max_points)`:
   - **Performance:** if N > `max_points` (default 60_000), subsample with a deterministic seed; expose the slider in the sidebar.
   - **Color modes:**
     - `umap_rgb` — per-point color from columns 3:6.
     - `cluster` — discrete palette by `cluster_id`; noise (`-1`) rendered light gray + smaller marker.
     - `highlight` — selected cluster bright, others gray-faded.
   - Plotly `Scatter3d` with `mode='markers'`, marker size 2–3, no axis labels, dark template, equal aspect.
   - Hover text: `row_id`, `cluster_id`, plus optional 1–2 metadata fields if metadata loaded.
3. Sidebar additions:
   - UMAP path (already added in Phase 1, just wire it up).
   - Color mode radio.
   - Subsample slider.
4. Cross-pillar wiring:
   - When a cluster is selected in Phase 2's table, default the UMAP color mode to `highlight` for that cluster (do not force — user can override).
   - When the UMAP is shown but no cluster is selected, default to `umap_rgb`.

Acceptance:
- Renders fluidly at 60k points on the workstation. No browser hangs.
- Toggling color modes does not re-load the npy (cached).
- Missing UMAP → the panel collapses with a single "no UMAP file configured" message; everything else still works.

**Validation**

Agent self-validation:
- [ ] Generate a synthetic UMAP `[N, 6]` array (N = `metrics.json["n_rows"]` for one run) inline, save to a tmp path, and confirm `umap_view.load_umap` returns the right shape and normalizes RGB if values were in `[0, 255]`.
- [ ] Building the Plotly figure with each color mode (`umap_rgb`, `cluster`, `highlight`) does not raise. (You don't have to render in a browser.)
- [ ] If N > `max_points`, the figure trace length equals `max_points` and the seed is stable across calls.
- [ ] Wrong-shape UMAP (e.g. `[N, 5]` or `[N-1, 6]`) raises a clear validation error, not an opaque numpy error.
- [ ] If you cannot generate / test a real UMAP file (e.g. disk space, time), say so and skip — do not tick this section.

Human validation:
- [ ] Provide a real (or synthetic) `umap_6d.npy` via the sidebar. The 3D scatter renders.
- [ ] Mouse-drag rotates; scroll zooms; the figure stays responsive.
- [ ] Switch color mode between `umap_rgb` and `cluster` — colors change without reloading the npy.
- [ ] Select a cluster in the Phase 2 table → UMAP defaults to `highlight` for that cluster (selected cluster is bright, others gray).
- [ ] Remove the UMAP path → only the UMAP panel disappears; the rest of the app keeps working.

---

### Phase 5 — Polish, caching, errors, docs

**Outcome:** The app is presentable. Caches are sized sensibly. Errors are user-friendly. A short README explains how to launch.

Tasks:
1. **Caching review:**
   - `@st.cache_resource` for: parquet/json loads, metadata DataFrame, UMAP array.
   - `@st.cache_data` for: per-cluster aggregates used by the table.
   - Bound caches by run path + mtime so editing a run on disk invalidates cleanly.
2. **Error boundaries:**
   - Wrap each pillar's render in a `try/except` that prints a compact `st.error` instead of a full traceback. Full traceback behind an expander.
3. **Empty states:** every panel has a meaningful empty-state message (no run, no metadata, no UMAP, empty cluster).
4. **README section** in `docs/VALIDATION_WEBAPP_PLAN.md` (append) **and** a short paragraph + link in the top-level `README.md` under a new "12. Validation web app" section pointing at the launcher script.
5. **Optional sidecar (only if asked):** `runs/<run>/validation_notes.json` — user notes per cluster persisted from the cluster panel. Off by default.

Acceptance:
- Cold start to first render < 3s for a typical run on the workstation (no UMAP).
- All four pillars are in place and don't crash when their inputs are missing.
- README has clear launch instructions.

**Validation**

Agent self-validation:
- [ ] Force-trigger each pillar's empty state in code (no run, no metadata, no UMAP, empty cluster) and confirm the rendered widget is a single `st.info`/`st.warning`, not a traceback.
- [ ] Inject an exception into one pillar's render path and confirm the error boundary catches it: user sees `st.error("…")` plus an expander labeled "Traceback".
- [ ] Cache invalidation: touch `runs/<one_run>/labels.parquet` (`os.utime`) and confirm `load_run` returns a fresh bundle on the next call.
- [ ] `pytest -q` passes.
- [ ] README diff includes a "12. Validation web app" section with the launch command.

Human validation:
- [ ] Cold-start the app, select a run with no UMAP and no metadata configured — first render under ~3s, all four pillar areas show clear empty states or working content.
- [ ] Configure metadata + UMAP, switch between all three runs — no leftover panels from the previous method, no broken caches.
- [ ] Read the new README section; the launch instructions work as written on a fresh shell.

---

## 5. Out-of-scope (record so we don't sprawl)

- Producing the 6D UMAP (separate offline script — not part of this app).
- Comparing two runs side-by-side (could be Phase 6 later).
- Writing labels back to `metadata.jsonl`.
- Authentication, deployment, dockerization.
