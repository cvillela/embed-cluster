# Build Plan: embedding-cluster / embedcluster

## Context

Build a Python package (`embedcluster`) that clusters pre-extracted embedding matrices using three GPU-accelerated pipelines: Leiden graph clustering (cuVS kNN → mutual kNN → cuGraph Leiden), HDBSCAN (cuML, optionally preceded by PCA), and spherical KMeans (FAISS GPU). The canonical spec is `docs/IMPLEMENTATION_BRIEF.md`. The repo is currently empty except for that brief.

---

## Cross-Cutting Invariants (enforce in every phase)

1. **Raw `.npy` immutability** — embeddings are always opened with `np.load(..., mmap_mode="r")`. No function may accept a writable reference to the raw file or write back to it.
2. **Row-ID alignment** — `row_id = np.arange(N)` is the universal join key. Every output parquet and every intermediate array must be positionally aligned to `row_id`. Rows that are absent from a clustering result (e.g., Leiden graph) are filled with `cluster_id = -1`.
3. **No dense N×N matrix** — kNN is sparse (cuVS returns top-k only), mutual graph is sparse (edge list), distance matrices are never materialized beyond a single batch.
4. **Working-view paths** — normalized embeddings (`normalized.npy`), PCA output (`pca.npy`), and PCA-renormalized output (`pca_normalized.npy`) are written to `{out}/intermediate/`. They are opened with `mmap_mode="r+"` only when creating them; downstream readers open them read-only.
5. **Zero-norm rows** — detected during norm computation; propagated through all pipelines as `cluster_id = -1`, `is_noise = True`. Never silently passed to GPU kernels.
6. **Chunked GPU transfers** — `batch_size` controls the maximum rows sent to GPU at once in every matmul, norm computation, and assignment loop. Never load the full embedding matrix into VRAM unless it demonstrably fits (checked via `gpu.py::estimate_run_memory`).
7. **Structured outputs** — every run writes `preflight.json`, `run_config.json`, `metrics.json`, `labels.parquet`, `cluster_summary.parquet`, and `logs/run.log`. Paths are canonical and not configurable.

---

## API Risks & Ambiguities

### A1 — cuVS kNN call signature
The brief says "cuVS kNN" without specifying the index type. cuVS exposes several index families: `cuvs.neighbors.brute_force`, `cuvs.neighbors.ivf_flat`, `cuvs.neighbors.cagra`. For exact kNN at N up to ~5M and D=1536, `brute_force` is the safe default (no approximation error). The Python API (circa cuVS 24.x/25.x) is:
```python
from cuvs.neighbors import brute_force
index = brute_force.build(dataset_cupy, metric="inner_product")
distances, indices = brute_force.search(index, queries_cupy, k)
```
**Risk**: The RAPIDS release installed by the user may expose a slightly different import path or keyword argument name. Wrap the cuVS call in `neighbors/cuvs_knn.py` behind a single function so the call site is isolated and easy to patch.

### A2 — FAISS GPU `Kmeans.train()` chunked behavior
`faiss.Kmeans.train(X)` expects the full training matrix as a contiguous `float32` C-array. FAISS internally sub-samples if N > a threshold (controlled by `max_points_per_centroid`, default 256 per cluster). It does **not** natively accept chunked training calls. The brief's "chunked" requirement applies to **assignment** (`kmeans.index.search()` in batches), not to training. Training is always done on the full (or FAISS-sub-sampled) matrix passed at once.
**Risk**: For very large N (~10M rows, D=1536) the float32 matrix is ~60 GB host RAM. Passing it directly to `train()` would OOM. Mitigation: sub-sample up to `max_points_per_centroid * n_clusters` rows for training (standard FAISS practice), then do chunked assignment over all rows.

### A3 — cuGraph Leiden vertex mapping with `renumber=True`
`G.from_cudf_edgelist(..., renumber=True)` remaps vertex IDs to a contiguous range [0, |V|). The `leiden()` return DataFrame has columns `vertex` (renumbered) and `partition`. To recover original `row_id` values, use `G.unrenumber(parts, "vertex")` or inspect `G.edgelist.renumber_map` depending on cuGraph version.
**Risk**: The unrenumber API changed between cuGraph 23.x and 25.x. In recent versions the renumber map is accessed via `G.renumber_map.legacy_src_id` or via `renumber_map` DataFrame returned alongside the edgelist. Implement the mapping with a defensive fallback: if `G.unrenumber` is not available, merge `parts` against the renumber map stored in `G.edgelist`.

### A4 — cuML HDBSCAN `build_kwds` key names
The brief specifies:
```python
build_kwds = {
    "knn_n_clusters": ...,
    "knn_overlap_factor": ...,
    "nnd_graph_degree": ...,
}
```
cuML's HDBSCAN `nn_descent` kwarg names may differ (`nnd_n_clusters`, `nnd_overlap_factor`). The cuML 24.x/25.x docs show `nn_descent_params` as a dict with keys that may not exactly match the brief.
**Risk**: Passing unknown keys silently ignored vs. raising TypeError. Wrap the `HDBSCAN(...)` constructor call so `build_kwds` is logged to `run_config.json` at construction time; add a try/except that retries without `build_kwds` and warns loudly if the first attempt raises `TypeError`.

### A5 — cuML PCA memory for large N
`cuml.decomposition.PCA` with `svd_solver="full"` loads the full matrix to GPU. For N=5M, D=1536 as float32 → 30 GB. The brief's default PCA-128 is only for HDBSCAN. Use `svd_solver="jacobi"` or `"auto"` (cuML picks the right solver). Alternatively, use `fit` on a sub-sample and `transform` in batches using the fitted components (cuML PCA supports this).
**Risk**: Large N with D=1536 may not fit in VRAM for a full-GPU PCA. Mitigation: sub-sample for `fit`, then batch-transform.

### A6 — cuDF partitioned mutual kNN memory
The partitioned mutual kNN construction emits temporary parquet files per partition. The brief says "write temporary edge partitions, then process each partition independently." The number of partitions must be chosen so each partition's directed edge table fits in GPU memory. Default `num_partitions = max(1, N * k // 20_000_000)` is a reasonable heuristic.

---

## Phase 1 — Foundation: Package Skeleton, CLI Stub, Validation, Output Infrastructure
*(Brief §1, §2, §3, §4, §6, §15)*

### Files to create

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata, build system (setuptools), `[project.scripts]` entrypoint |
| `requirements.txt` | Non-RAPIDS deps per §2 |
| `src/embedcluster/__init__.py` | Package version string only |
| `src/embedcluster/cli.py` | Typer app with `leiden`, `hdbscan`, `kmeans` commands; shared params wired up; pipeline calls stubbed with `NotImplementedError` |
| `src/embedcluster/config.py` | Pydantic models: `SharedConfig`, `LeidenConfig`, `HdbscanConfig`, `KmeansConfig`; dataclasses: `DatasetInfo`, `RunPaths` |
| `src/embedcluster/logging_utils.py` | Structured logger: file handler → `logs/run.log`, rich console handler, `get_logger(name)` factory |
| `src/embedcluster/io.py` | `load_embeddings_mmap()`, `iter_metadata_lines()` (streaming iterator), `count_metadata_lines()`, `write_parquet()`, `write_json()` |
| `src/embedcluster/validation.py` | `validate_inputs()` → `DatasetInfo`; chunk scan for NaN/Inf/zero-norm; returns `zero_norm_mask` |
| `src/embedcluster/export.py` | `create_run_dirs()`, `write_preflight_json()`, `write_run_config_json()`, `write_labels_parquet()`, `write_cluster_summary()`, `write_metrics_json()`, `export_jsonl_with_labels()` |
| `src/embedcluster/gpu.py` | `get_gpu_info()`, `choose_batch_size()`, `estimate_run_memory()`, `run_preflight_check()` |
| `src/embedcluster/neighbors/__init__.py` | Empty |
| `src/embedcluster/graph/__init__.py` | Empty |
| `src/embedcluster/pipelines/__init__.py` | Empty |
| `tests/test_validation.py` | CPU-only pytest for validation module |
| `tests/conftest.py` | Registers `gpu` pytest mark |

### Key design decisions

- `DatasetInfo` holds: `n_rows`, `n_dims`, `dtype`, `zero_norm_mask: np.ndarray[bool]`, `embeddings_path`, `metadata_path`.
- `RunPaths` holds all canonical output paths derived from `out: Path`. Computed once in `export.create_run_dirs()`.
- `cli.py` uses `typer.Typer()` with `no_args_is_help=True`. Each sub-command receives `SharedConfig` + method-specific config assembled from CLI flags, then calls the pipeline function (stubbed in Phase 1).
- `gpu.py` uses `nvidia-ml-py` (`pynvml`) for GPU info, falling back gracefully if pynvml fails.
- `run_preflight_check()` raises `RuntimeError` with actionable message if estimated memory exceeds 85% of free VRAM.

---

## Phase 2 — Preprocessing: Norms and L2-Normalized Working View
*(Brief §7)*

### Files to create/modify

| File | Purpose |
|------|---------|
| `src/embedcluster/preprocessing.py` | All four required functions |

### Implementation notes

**`compute_l2_norms(embeddings, out_path, batch_size)`**
- Open `out_path` as writable `np.memmap` shape `[N]`, dtype `float32`.
- Iterate in `batch_size` chunks; compute `np.linalg.norm(batch.astype("float32"), axis=1)`.
- Write each chunk's norms. Log progress every 10 chunks.
- Return the resulting memmap (reopened read-only after writing).

**`create_l2_normalized_view(embeddings, norms, out_path, batch_size)`**
- Open `out_path` as writable `np.memmap` shape `[N, D]`, dtype `float32`.
- For each chunk: divide by norm (clip denominator to `max(norm, 1e-12)`); zero-norm rows become all-zeros.
- Return the resulting memmap (reopened read-only).

**`fit_transform_pca_optional(X, n_components, out_dir, batch_size, random_state)`**
- If `n_components is None`, return `X` unchanged.
- Validate `50 <= n_components <= 256`; warn but continue if out of range.
- Sub-sample up to `min(N, 500_000)` rows for `cuml.decomposition.PCA.fit()`.
- Batch-transform all rows in `batch_size` chunks via `pca.transform(batch_gpu)`.
- Write result to `out_dir / "pca.npy"` as `float32` memmap.
- Log explained variance sum.

**`renormalize_matrix(X, out_path, batch_size)`**
- Same logic as `create_l2_normalized_view` but operates on an arbitrary matrix (not necessarily the raw embeddings).
- Writes to `out_dir / "pca_normalized.npy"`.

### Policy enforcement
- Both `fit_transform_pca_optional` and `renormalize_matrix` always open their outputs as new memmaps; they never write back to their input memmap.

---

## Phase 3 — Pipeline: FAISS GPU Spherical KMeans
*(Brief §11)*

### Files to create/modify

| File | Purpose |
|------|---------|
| `src/embedcluster/pipelines/kmeans_pipeline.py` | `run_kmeans_pipeline(shared_cfg, kmeans_cfg, run_paths, dataset_info)` |

### Implementation notes

**n_clusters calculation** (when `--n-clusters` not given):
```python
from math import ceil
n_clusters = ceil(N / target_cluster_size)
n_clusters = min(max(n_clusters, 2), 10_000)
```

**Training sub-sample**: compute `max_train = min(N, 256 * n_clusters)`. If `N > max_train`, randomly sample `max_train` rows (using `random_state` seed). Pass the sub-sample as contiguous `float32` C-array to `kmeans.train()`.

**Chunked assignment**: iterate over all rows in `batch_size` chunks, call `kmeans.index.search(chunk.astype("float32"), 1)`, accumulate `cluster_id` and distances.

**`cosine_to_centroid` computation**:
- Normalize `kmeans.centroids` (shape `[k, D]`) row-wise.
- For each batch: `cos[i] = dot(X_normed[i], centers[cluster_id[i]])`. Use numpy gather + einsum.

**`--no-normalize` warning**: emit `logger.warning(...)` before training; still proceeds.

**Zero-norm rows**: excluded from training sub-sample; assigned `cluster_id = -1`, `cosine_to_centroid = NaN` in output.

**Outputs**: `labels.parquet`, `cluster_centers.npy` (raw `kmeans.centroids`), `cluster_summary.parquet`, `run_config.json`, `preflight.json`, `metrics.json`.

---

## Phase 4 — Pipeline: HDBSCAN with Optional PCA
*(Brief §10, §7 PCA subsections)*

### Files to create/modify

| File | Purpose |
|------|---------|
| `src/embedcluster/pipelines/hdbscan_pipeline.py` | `run_hdbscan_pipeline(shared_cfg, hdbscan_cfg, run_paths, dataset_info)` |

### Implementation notes

**Working matrix construction**:
```
compute_l2_norms → norms.npy
create_l2_normalized_view → normalized.npy
fit_transform_pca_optional → pca.npy         (if pca_components is not None)
renormalize_matrix → pca_normalized.npy      (if pca_components is not None)
```
The final working matrix (`X_work`) is `pca_normalized.npy` if PCA is used, else `normalized.npy`.

**Build algo selection**:
```python
build_algo = "nn_descent" if N >= 1_000_000 else "brute_force"
build_kwds = {
    "knn_n_clusters": 4 if N < 2_000_000 else 8,
    "knn_overlap_factor": 2,
    "nnd_graph_degree": max(64, (min_samples or min_cluster_size) + 1),
} if build_algo == "nn_descent" else {}
```

**GPU transfer**: load entire `X_work` to cupy array. HDBSCAN requires the full matrix in VRAM (no chunking at fit time). Preflight check must raise if this won't fit.

**`build_kwds` defensive construction**: try the cuML constructor with `build_kwds`; if `TypeError`, retry without and log a warning.

**Label extraction**: `labels = clusterer.fit_predict(X_gpu)` returns cupy array; convert via `.get()`. `probabilities = clusterer.probabilities_.get()`. `persistence = clusterer.cluster_persistence_` → parquet.

**Outputs**: `labels.parquet` (with `probability` column), `hdbscan_cluster_persistence.parquet`, `cluster_summary.parquet`, `run_config.json` (includes `build_algo`, `build_kwds`, `pca_components`), `preflight.json`, `metrics.json`.

---

## Phase 5 — Pipeline: Leiden Graph Clustering
*(Brief §9, including cuVS kNN, mutual kNN, cuGraph Leiden)*

### Files to create/modify

| File | Purpose |
|------|---------|
| `src/embedcluster/neighbors/cuvs_knn.py` | `build_and_search_knn(X, k, normalize, batch_size, indices_out, scores_out)` |
| `src/embedcluster/graph/mutual_knn.py` | `build_mutual_graph(indices_path, scores_path, N, k, min_similarity, out_path, num_partitions, tmp_dir)` |
| `src/embedcluster/graph/leiden.py` | `run_leiden(edges_path, N, resolution, random_state)` → cuDF DataFrame with `row_id`, `cluster_id` |
| `src/embedcluster/pipelines/leiden_pipeline.py` | `run_leiden_pipeline(shared_cfg, leiden_cfg, run_paths, dataset_info)` |

### cuVS kNN (`neighbors/cuvs_knn.py`)

**Metric**: `"inner_product"` if `normalize=True`, else `"cosine"`.
**Self-neighbor removal**: request `k+1` neighbors, filter columns where `indices[i, j] == i`, keep first `k` remaining. Handle edge case where self is not in top-k.
**Output**: write `knn_indices.npy` (int32, `[N, k]`) and `knn_scores.npy` (float32, `[N, k]`) to `run_paths.intermediate`.
**Reuse check**: if both files exist and match expected shape `[N, k]`, log "reusing existing kNN" and skip recomputation.

### Mutual kNN graph (`graph/mutual_knn.py`)

**Algorithm**:
1. Stream `knn_indices` and `knn_scores` in chunks; emit directed edges `(i, j, w)` to partition files based on `min(i, j) % num_partitions`.
2. For each partition: load all directed edges into cuDF; group by `(a=min(i,j), b=max(i,j))`; keep only pairs with count == 2 (mutual); compute `weight = mean(w_forward, w_backward)`.
3. Filter by `weight >= min_similarity`.
4. Concatenate all partition results; write `mutual_edges.parquet` with columns `src: int64`, `dst: int64`, `weight: float32`.
5. Clean up partition tmp files.

**num_partitions heuristic**: `max(4, N * k // 10_000_000)` (target ~10M directed edges per partition).

### cuGraph Leiden (`graph/leiden.py`)

```python
import cugraph, cudf

edge_df = cudf.read_parquet(edges_path)
G = cugraph.Graph(directed=False)
G.from_cudf_edgelist(edge_df, source="src", destination="dst", edge_attr="weight", renumber=True)
parts, modularity = cugraph.leiden(G, max_iter=100, resolution=resolution, random_state=random_state)
```

**Vertex remapping**: After Leiden, `parts` has columns `vertex` (renumbered internal IDs) and `partition`. Use `G.unrenumber(parts, "vertex")` to recover original row IDs. If `unrenumber` is unavailable (older cuGraph), merge `parts` against `G.edgelist.renumber_map`.

**Missing rows**: rows absent from `mutual_edges.parquet` are filled with `cluster_id = -1`, `is_noise = True`, `graph_degree = 0`, `mean_neighbor_similarity = NaN`.

**Reuse logic**: if `mutual_edges.parquet` exists and `normalized.npy` / `knn_indices.npy` / `knn_scores.npy` all exist, only re-run Leiden (with potentially different `resolution`). Log clearly which steps are reused.

---

## Phase 6 — Polish: Metrics, Scripts, Tests, README
*(Brief §12, §13, §17, §18, §19)*

### Files to create/modify

| File | Purpose |
|------|---------|
| `src/embedcluster/metrics.py` | `compute_global_metrics()`, `compute_cluster_summary()`, `compute_sampled_silhouette()` |
| `scripts/validate_inputs.py` | Thin wrapper calling `validate_inputs()` + printing `DatasetInfo` |
| `scripts/run_leiden.py` | Thin wrapper: parse same args as CLI leiden command, call `run_leiden_pipeline()` |
| `scripts/run_hdbscan.py` | Same pattern for HDBSCAN |
| `scripts/run_kmeans.py` | Same pattern for KMeans |
| `tests/test_preprocessing.py` | Pytest: normalized norms ≈ 1, zero rows stay zero, PCA shape, PCA-renorm norms ≈ 1 |
| `tests/test_small_leiden.py` | `@pytest.mark.gpu` — CLI runs, labels.parquet exists, all row_ids present, some non-negative clusters |
| `tests/test_small_hdbscan.py` | `@pytest.mark.gpu` — CLI runs, labels.parquet exists, probability column exists |
| `tests/test_small_kmeans.py` | `@pytest.mark.gpu` — CLI runs, labels.parquet + cluster_centers.npy + cosine_to_centroid column, valid rows non-negative |
| `README.md` | Per §19 structure |

### Metrics (`metrics.py`)

**`compute_global_metrics(labels, norms, method, pipeline_cfg)`**:
- `n_rows`, `n_clusters = len(unique(cluster_id[cluster_id >= 0]))`, `n_noise/n_unassigned`, `noise_fraction`.
- Cluster size distribution: `min`, `p25`, `median`, `p75`, `max` (over non-noise clusters).
- `embedding_norm_mean`, `embedding_norm_std` (over all rows including noise).

**`compute_cluster_summary(labels_df, norms, metadata_iter)`**:
- Group by `cluster_id`; compute `size`, `embedding_norm_mean`, `embedding_norm_std`.
- Optionally scan metadata lines (streaming) for known label fields: `species`, `label`, `scientific_name`, `common_name`, `class`, `category`. If found, compute `top_label_*` columns.

**`compute_sampled_silhouette(X, labels, sample_n, random_state)`**:
- Sample `min(N, sample_n)` rows stratified by cluster.
- Use `sklearn.metrics.silhouette_score` on the sample (CPU numpy). Mark as skipped if `n_clusters < 2` or `sample_n < 2 * n_clusters`.

### Synthetic test fixtures (shared `conftest.py`)

```python
N, D = 200, 1536
rng = np.random.default_rng(42)
centers = rng.standard_normal((3, D))
centers /= np.linalg.norm(centers, axis=1, keepdims=True)
X = np.vstack([
    centers[0] + 0.05 * rng.standard_normal((60, D)),   # cluster 0
    centers[1] + 0.05 * rng.standard_normal((60, D)),   # cluster 1
    centers[2] + 0.05 * rng.standard_normal((70, D)),   # cluster 2
    rng.standard_normal((10, D)),                        # noise-like
]).astype("float32")
# Insert 2 zero-norm rows
X[0] = 0.0
X[1] = 0.0
```

Tests that don't require GPU (`test_validation.py`, `test_preprocessing.py`) must run on CPU only — no cuml/cuvs imports at module level.

### Scripts pattern

Each script in `scripts/` uses `argparse`, parses the same flags as the corresponding CLI command, constructs `SharedConfig` + method config objects, and calls the same pipeline function. The CLI and scripts share zero duplicated logic.

### README sections (§19)

1. What this repo does
2. What this repo does not do (no audio, no extraction, no RAPIDS install)
3. Input file requirements
4. Dependency assumptions (CUDA 12, RAPIDS pre-installed)
5. Running validation
6. Running Leiden
7. Running HDBSCAN
8. Running FAISS spherical KMeans
9. Outputs (run directory contract from §15)
10. How to join labels back to metadata
11. Known limitations (cuML HDBSCAN full-matrix VRAM requirement, FAISS training sub-sample, cuVS brute-force index memory)

---

## Implementation Order (matches §20)

| Step | Phase | Deliverable |
|------|-------|-------------|
| 1 | 1 | `pyproject.toml`, `requirements.txt`, `src/embedcluster/__init__.py`, `cli.py` (stubbed), `config.py`, `logging_utils.py` |
| 2 | 1 | `validation.py` |
| 3 | 1 | `export.py`, `io.py`, `gpu.py` |
| 4 | 2 | `preprocessing.py::compute_l2_norms` |
| 5 | 2 | `preprocessing.py::create_l2_normalized_view` |
| 6 | 3 | `pipelines/kmeans_pipeline.py`, `cli.py` kmeans command wired up |
| 7 | 4 | `pipelines/hdbscan_pipeline.py` (without PCA), `cli.py` hdbscan wired up |
| 8 | 4 | `preprocessing.py::fit_transform_pca_optional` + `renormalize_matrix`; HDBSCAN pipeline updated |
| 9 | 5 | `neighbors/cuvs_knn.py` |
| 10 | 5 | `graph/mutual_knn.py` |
| 11 | 5 | `graph/leiden.py`, `pipelines/leiden_pipeline.py`, `cli.py` leiden wired up |
| 12 | 6 | `metrics.py`, wired into all three pipelines |
| 13 | 6 | `scripts/*.py` |
| 14 | 6 | `tests/*.py` (preprocessing + GPU integration tests) |
| 15 | 6 | `README.md` |

---

## Verification

After each phase, verify:

- **Phase 1**: `python -m embedcluster --help` shows three commands; `embedcluster leiden --help` shows all shared + method-specific flags; `pytest tests/test_validation.py` passes on CPU.
- **Phase 2**: `pytest tests/test_preprocessing.py` passes (CPU-only); confirm normalized rows have `norm ≈ 1.0 ± 1e-5`; confirm zero rows stay zero.
- **Phase 3**: `pytest -m gpu tests/test_small_kmeans.py` passes; inspect `labels.parquet` for `cosine_to_centroid` column; verify `cluster_centers.npy` shape `[k, D]`.
- **Phase 4**: `pytest -m gpu tests/test_small_hdbscan.py` passes; verify `hdbscan_cluster_persistence.parquet` exists; check `pca.npy` shape `[N, 128]`.
- **Phase 5**: `pytest -m gpu tests/test_small_leiden.py` passes; verify `mutual_edges.parquet` edge count > 0; verify all 200 `row_id` values present in `labels.parquet`.
- **Phase 6**: `pytest tests/test_validation.py tests/test_preprocessing.py` (no GPU); `pytest -m gpu` for integration tests; confirm all required output files exist for each pipeline.
