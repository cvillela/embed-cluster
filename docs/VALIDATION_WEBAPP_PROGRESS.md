# Validation Web App — Progress

Plan: `VALIDATION_WEBAPP_PLAN.md`
Per-phase agent instructions: `VALIDATION_WEBAPP_AGENT_PROMPT.md`

Each phase has **two** checkboxes:
- **Agent self-validation** — ticked by the implementing agent only after every "Agent self-validation" step in that phase's plan section passes. If a step couldn't be run (e.g. no display for Streamlit), the agent must say so in the notes and **leave the box unchecked**.
- **Human validation** — ticked **only by the human user** after running the "Human validation" steps in that phase's plan section. Agents must never tick this box.

Do not start phase N+1 until **both** boxes for phase N are ticked.

---

- [ ] **Phase 0 — Scaffolding & dependencies**
  - [x] Agent self-validation passed
  - [ ] Human validation passed
  - Notes: Scaffolding only; all module files created empty per plan §3 (placeholders for later phases). Streamlit booted headlessly on port 8765 and printed the standard "You can now view your Streamlit app" banner with no tracebacks. No deviations from the plan.

- [ ] **Phase 1 — Pillar 1: Run discovery & loading**
  - [x] Agent self-validation passed
  - [ ] Human validation passed
  - Notes: discover_runs returns 3 runs (hdbscan/kmeans_coarse/leiden_k50_res1); load_run on each yields method ∈ {leiden,hdbscan,kmeans}, non-empty cluster_summary, metrics.n_rows == 630362. Streamlit booted headlessly on :8767 with no tracebacks. labels.parquet is loaded lazily via @property; @st.cache_resource keys on (path, run_config.json mtime). No deviations from the plan.

- [ ] **Phase 2 — Pillar 2: Method-aware validation & cluster display**
  - [x] Agent self-validation passed
  - [ ] Human validation passed
  - Notes: aggregate_cluster_table verified for all 3 runs — expected method-specific columns present, no cross-method leakage, no NaN in non-noise rows, n_clusters matches metrics.json. HDBSCAN/Leiden tables include cluster_id == -1 with size matching n_noise / n_unassigned. pytest -q: 36 passed. Streamlit booted clean. **Deviation from plan §4 Phase 2 (per user direction):** UI is intentionally minimal — no per-method histogram/boxplot/persistence/tightest-loosest panels, no sortable cluster dataframe. Cluster picker is a sort-control + selectbox above the audio panel. Layout order: (1) method, (2) cluster size, (3) audio per cluster, (4) UMAP, (5) hyperparams/dataset/raw config in expanders.

- [ ] **Phase 3 — Pillar 3: Audio fetching & per-cluster listening**
  - [x] Agent self-validation passed
  - [ ] Human validation passed
  - Notes: sample_cluster verified on real labels for all 3 methods × all 4 strategies — representative monotonically non-increasing & equals top-N by quality col, boundary monotonically non-decreasing & equals bottom-N, random with fixed seed is deterministic, stratified-by-norm returns N rows from the right cluster. Noise (cluster_id=-1) accepts only random/stratified; representative/boundary raise ValueError. metadata_loader.load_metadata + detect_audio_fields + resolve_audio_path verified on synthetic JSONL (with and without explicit row_id). pytest -q: 36 passed. Streamlit booted headlessly on :8771 with no tracebacks. **Deviation from plan §4 Phase 3:** sample_cluster returns a labels-DataFrame slice rather than `list[Sample]` — keeps the API testable and lets the panel join metadata cleanly; the monotonicity / top-N validation criteria all read naturally off the DataFrame. **Could not run live audio check** — no `metadata.jsonl` available in this environment, so the rendered grid (st.audio cards, missing-file warning path, extra-cols captions) is unverified by the agent and must be checked by the human user.

- [ ] **Phase 4 — Pillar 4: 3D UMAP plot**
  - [ ] Agent self-validation passed
  - [ ] Human validation passed
  - Notes:

- [ ] **Phase 5 — Polish, caching, errors, docs**
  - [ ] Agent self-validation passed
  - [ ] Human validation passed
  - Notes:
