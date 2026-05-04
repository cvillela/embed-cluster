# Engineer implementation brief: embedding clustering repository

Build a Python repository that clusters **already-extracted embeddings** from:

```text
embeddings.npy      # shape: [N, D], usually D = 1536
metadata.jsonl      # exactly N lines; row i describes embeddings[i]
```

Do **not** implement audio loading, segmentation, denoising, embedding extraction, or model inference.

The repository must implement exactly three clustering pipelines:

```text
1. Leiden graph clustering
   L2-normalized working view → cuVS kNN → mutual weighted kNN graph → cuGraph Leiden

2. HDBSCAN noise/outlier discovery
   L2-normalized working view → optional PCA 50–256 → L2 re-normalization → cuML HDBSCAN

3. Spherical KMeans coarse baseline
   L2-normalized working view → FAISS GPU spherical KMeans
```

Use **CUDA 12 only**.

The user will handle the RAPIDS installation. The engineer should not add RAPIDS install instructions, CUDA-selection logic, or CUDA-version branching. Assume the runtime already has working CUDA 12-compatible installations of:

```text
cudf
cuml
cugraph
cuvs
cupy
rmm
```

The repository should still provide a plain `requirements.txt` for non-RAPIDS Python dependencies and required FAISS GPU support.

---

# 1. Repository structure

Implement this structure:

```text
embedding-cluster/
  README.md
  pyproject.toml
  requirements.txt

  src/
    embedcluster/
      __init__.py

      cli.py
      config.py
      logging_utils.py

      io.py
      validation.py
      gpu.py
      preprocessing.py
      metrics.py
      export.py

      neighbors/
        __init__.py
        cuvs_knn.py

      graph/
        __init__.py
        mutual_knn.py
        leiden.py

      pipelines/
        __init__.py
        leiden_pipeline.py
        hdbscan_pipeline.py
        kmeans_pipeline.py

  scripts/
    run_leiden.py
    run_hdbscan.py
    run_kmeans.py
    validate_inputs.py

  tests/
    test_validation.py
    test_preprocessing.py
    test_small_leiden.py
    test_small_hdbscan.py
    test_small_kmeans.py
```

Do not create an `examples/` folder.

Do not use bird-specific names anywhere in package names, file names, CLI names, or module names.

The package and CLI should be called:

```text
embedcluster
```

---

# 2. Environment and installation files

Do **not** create `environment.yml`.

Create only:

```text
requirements.txt
pyproject.toml
```

The user will install RAPIDS separately. Therefore, do **not** include `cudf-cu12`, `cuml-cu12`, `cugraph-cu12`, or `cuvs-cu12` in `requirements.txt`.

## `requirements.txt`

Use non-locking version ranges:

```txt
numpy>=1.26
pandas>=2.2
pyarrow>=15.0
orjson>=3.10
tqdm>=4.66
typer>=0.12
rich>=13.7
pydantic>=2.7
nvidia-ml-py>=12.535
pytest>=8.0

faiss-gpu-cu12>=1.10
```

Because the project requires CUDA 12 and FAISS GPU spherical KMeans, FAISS GPU is a required dependency. The PyPI `faiss-gpu-cu12` package is built for CUDA 12.x and exposes CUDA 12 GPU support, but the engineer should verify compatibility on the actual deployment GPU during validation. ([PyPI][4])

## `pyproject.toml`

Expose the CLI:

```toml
[project.scripts]
embedcluster = "embedcluster.cli:app"
```

---

# 3. CLI design

Implement one CLI entrypoint:

```bash
embedcluster <command> [args]
```

Commands:

```bash
embedcluster leiden \
  --embeddings embeddings.npy \
  --metadata metadata.jsonl \
  --out runs/leiden_default

embedcluster hdbscan \
  --embeddings embeddings.npy \
  --metadata metadata.jsonl \
  --out runs/hdbscan_default

embedcluster kmeans \
  --embeddings embeddings.npy \
  --metadata metadata.jsonl \
  --out runs/kmeans_default
```

Keep exposed parameters minimal.

---

# 4. Shared CLI parameters

All commands must accept:

```text
--embeddings PATH              required
--metadata PATH                required
--out PATH                     required
--normalize / --no-normalize   default: --normalize
--batch-size INT               default: auto
--random-state INT             default: 42
--sample-metrics INT           default: 50000
```

`--normalize` creates a normalized working matrix under the run directory.

It must **never** overwrite the raw input `.npy`.

The raw embeddings must always remain immutable.

---

# 5. Pipeline-specific CLI parameters

## Leiden

Expose only:

```text
--k INT                        default: 50
--resolution FLOAT             default: 1.0
--min-similarity FLOAT         default: 0.0
```

Use cuVS as the only kNN backend.

Do not expose FAISS kNN.

Do not add alternative graph clustering algorithms.

## HDBSCAN

Expose only:

```text
--min-cluster-size INT         default: 50
--min-samples INT|none         default: none
--pca-components INT|none      default: 128
--cluster-selection eom|leaf   default: eom
```

Default HDBSCAN path uses PCA-128.

Allow `--pca-components none`.

Allowed PCA range is:

```text
50–256
```

If the user passes a value outside that range, warn but allow execution.

## KMeans

Expose only:

```text
--n-clusters INT               optional
--target-cluster-size INT      default: 1000
--max-iter INT                 default: 300
--nredo INT                    default: 1
```

If `--n-clusters` is not provided:

```python
n_clusters = ceil(N / target_cluster_size)
n_clusters = min(max(n_clusters, 2), 10000)
```

Use **FAISS GPU spherical KMeans only**.

Do not implement cuML KMeans.

Do not implement a KMeans backend switch.

---

# 6. Input validation

Implement `validation.py`.

Required function:

```python
def validate_inputs(
    embeddings_path: Path,
    metadata_path: Path,
) -> DatasetInfo:
    ...
```

Validation rules:

1. `embeddings.npy` must exist.

2. Load with:

   ```python
   np.load(path, mmap_mode="r")
   ```

3. Embedding matrix must be 2D.

4. Matrix dtype must be one of:

   ```text
   float16
   float32
   float64
   ```

5. `metadata.jsonl` must exist.

6. Metadata line count must equal `N`.

7. Do not require specific metadata keys.

8. Add internal:

   ```python
   row_id = np.arange(N)
   ```

9. Scan embeddings in chunks for:

   ```text
   NaN
   Inf
   zero-norm rows
   ```

10. Fail on NaN or Inf by default.

11. Zero-norm rows should be marked invalid and assigned:

```text
cluster_id = -1
```

Metadata labels must not be trusted for clustering. They are only row-aligned metadata for later joining or summaries.

---

# 7. Preprocessing implementation

Implement `preprocessing.py`.

The repository must preserve raw embeddings.

Normalization and PCA are working views saved under the run directory.

## Required functions

```python
def compute_l2_norms(
    embeddings: np.memmap,
    out_path: Path,
    batch_size: int,
) -> np.memmap:
    """Write one float32 norm per row."""
```

```python
def create_l2_normalized_view(
    embeddings: np.memmap,
    norms: np.memmap,
    out_path: Path,
    batch_size: int,
) -> np.memmap:
    """Create float32 normalized matrix. Zero-norm rows become all zeros."""
```

```python
def fit_transform_pca_optional(
    X: np.memmap,
    n_components: int | None,
    out_dir: Path,
    batch_size: int,
    random_state: int,
) -> np.memmap:
    """If n_components is None, return X. Else write PCA-transformed matrix."""
```

```python
def renormalize_matrix(
    X: np.memmap,
    out_path: Path,
    batch_size: int,
) -> np.memmap:
    """L2-normalize a working matrix."""
```

## Normalization policy

Default for all three pipelines:

```text
raw embeddings
→ compute raw norms
→ create L2-normalized working matrix
```

If `--no-normalize` is passed:

```text
raw embeddings
→ compute raw norms
→ use raw matrix as working matrix
```

Still export embedding norms either way.

## PCA policy

Only HDBSCAN uses PCA by default.

Leiden and KMeans do not use PCA.

HDBSCAN default:

```text
raw embeddings
→ L2-normalized working matrix
→ PCA-128
→ L2 re-normalize PCA output
→ HDBSCAN
```

---

# 8. GPU and memory utilities

Implement `gpu.py`.

Required functions:

```python
def get_gpu_info() -> dict:
    """Return GPU name, total memory, free memory, and driver/runtime info if available."""
```

```python
def choose_batch_size(
    n_rows: int,
    n_dims: int,
    dtype_bytes: int,
    target_fraction_gpu_mem: float = 0.25,
) -> int:
    """Choose a conservative batch size for GPU transfers."""
```

```python
def estimate_run_memory(
    n_rows: int,
    n_dims: int,
    k: int | None,
    pipeline: str,
) -> dict:
    """Return approximate memory estimates for logs and preflight checks."""
```

Every run must write:

```text
out/preflight.json
```

Example:

```json
{
  "n_rows": 1234567,
  "n_dims": 1536,
  "embedding_dtype": "float32",
  "pipeline": "leiden",
  "cuda_major": 12,
  "gpu_name": "NVIDIA ...",
  "free_gpu_memory_gb": 70.1,
  "estimated_edges_before_mutual": 61728350
}
```

Do not silently continue when a clear GPU-memory issue is detected.

Raise a clear error suggesting one of:

```text
smaller batch size
smaller k
HDBSCAN with PCA
smaller target cluster size is not a memory fix for KMeans; do not suggest it as one
```

---

# 9. Pipeline 1: Leiden graph clustering

Implement in:

```text
neighbors/cuvs_knn.py
graph/mutual_knn.py
graph/leiden.py
pipelines/leiden_pipeline.py
```

## 9.1 Working matrix

Default:

```text
X_raw = embeddings.npy
norms = compute_l2_norms(X_raw)
X = create_l2_normalized_view(X_raw, norms)
```

If `--no-normalize` is passed:

```text
X = X_raw
```

Still compute and export norms.

## 9.2 cuVS kNN search

Use cuVS as the only kNN backend.

Implement:

```python
def build_and_search_knn(
    X: np.memmap,
    k: int,
    normalize: bool,
    batch_size: int,
    indices_out: Path,
    scores_out: Path,
) -> tuple[np.memmap, np.memmap]:
    ...
```

Metric behavior:

```python
if normalize:
    metric = "inner_product"
else:
    metric = "cosine"
```

Request `k + 1` neighbors internally and remove self-neighbors.

Output:

```text
out/intermediate/knn_indices.npy   # int64 or int32, shape [N, k]
out/intermediate/knn_scores.npy    # float32, shape [N, k]
```

Default:

```text
k = 50
```

Do not create a dense distance matrix.

## 9.3 Mutual weighted kNN graph

Implement `graph/mutual_knn.py`.

Input:

```text
knn_indices[N, k]
knn_scores[N, k]
```

Output:

```text
out/intermediate/mutual_edges.parquet
```

Columns:

```text
src: int64
dst: int64
weight: float32
```

Rules:

1. Remove self edges.

2. Remove invalid neighbor IDs.

3. Remove edges with:

   ```python
   weight < min_similarity
   ```

4. Convert each directed edge `(i, j)` to an undirected key:

   ```python
   a = min(i, j)
   b = max(i, j)
   ```

5. Keep only mutual edges where both directions exist.

6. Final edge weight:

   ```python
   weight = mean(weight_i_to_j, weight_j_to_i)
   ```

7. Write one final row per undirected edge.

For memory safety, implement partitioned graph construction:

```python
partition_id = a % num_partitions
```

Write temporary edge partitions, then process each partition independently with cuDF groupby.

Do not require the full directed edge table to fit in GPU memory.

## 9.4 cuGraph Leiden

Implement `graph/leiden.py`.

Read:

```text
out/intermediate/mutual_edges.parquet
```

Create weighted undirected cuGraph graph:

```python
G = cugraph.Graph(directed=False)

G.from_cudf_edgelist(
    edge_df,
    source="src",
    destination="dst",
    edge_attr="weight",
    renumber=True,
)
```

Run:

```python
parts, modularity = cugraph.leiden(
    G,
    max_iter=100,
    resolution=resolution,
    random_state=random_state,
)
```

Rows absent from the mutual graph must still appear in the final output with:

```text
cluster_id = -1
is_noise = true
```

## 9.5 Leiden outputs

Write:

```text
out/labels.parquet
out/cluster_summary.parquet
out/run_config.json
out/preflight.json
out/metrics.json
```

`labels.parquet` columns:

```text
row_id
cluster_id
method                  # "leiden"
is_noise                # true if cluster_id == -1
embedding_norm
graph_degree
mean_neighbor_similarity
```

`metrics.json`:

```json
{
  "method": "leiden",
  "n_rows": 123,
  "n_clusters": 12,
  "n_unassigned": 3,
  "modularity": 0.73,
  "k": 50,
  "resolution": 1.0,
  "min_similarity": 0.0
}
```

---

# 10. Pipeline 2: HDBSCAN

Implement in:

```text
pipelines/hdbscan_pipeline.py
```

## 10.1 Working matrix

Default:

```text
raw embeddings
→ compute raw norms
→ L2-normalized working matrix
→ PCA-128
→ L2 re-normalized PCA matrix
→ cuML HDBSCAN
```

Allow:

```bash
--pca-components none
```

If PCA is disabled:

```text
raw embeddings
→ compute raw norms
→ L2-normalized working matrix
→ cuML HDBSCAN
```

## 10.2 HDBSCAN implementation

Use cuML HDBSCAN.

cuML HDBSCAN currently documents `metric='euclidean'` as the allowed metric, so the pipeline should use Euclidean HDBSCAN on the L2-normalized or PCA-renormalized working matrix. cuML HDBSCAN also exposes `labels_`, `probabilities_`, and `cluster_persistence_`; noisy samples receive label `-1`. ([RAPIDS Docs][5])

Implementation:

```python
from cuml.cluster import HDBSCAN

clusterer = HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
    metric="euclidean",
    cluster_selection_method=cluster_selection,
    prediction_data=False,
    build_algo=build_algo,
    build_kwds=build_kwds,
    output_type="cupy",
)

labels = clusterer.fit_predict(X_gpu)
```

Defaults:

```python
min_cluster_size = 50
min_samples = None
cluster_selection_method = "eom"
pca_components = 128
```

Internal backend choice:

```python
if N >= 1_000_000:
    build_algo = "nn_descent"
else:
    build_algo = "brute_force"
```

For large datasets:

```python
build_kwds = {
    "knn_n_clusters": 4 if N < 2_000_000 else 8,
    "knn_overlap_factor": 2,
    "nnd_graph_degree": max(64, (min_samples or min_cluster_size) + 1),
}
```

Do not expose `build_algo` or `build_kwds` in the CLI for v1.

Write these internal choices to:

```text
out/run_config.json
```

## 10.3 HDBSCAN outputs

Write:

```text
out/labels.parquet
out/cluster_summary.parquet
out/hdbscan_cluster_persistence.parquet
out/run_config.json
out/preflight.json
out/metrics.json
```

`labels.parquet` columns:

```text
row_id
cluster_id              # -1 means noise
method                  # "hdbscan"
is_noise
probability
embedding_norm
```

`hdbscan_cluster_persistence.parquet` columns:

```text
cluster_id
persistence
```

`metrics.json`:

```json
{
  "method": "hdbscan",
  "n_rows": 123,
  "n_clusters": 12,
  "n_noise": 8,
  "noise_fraction": 0.065,
  "min_cluster_size": 50,
  "min_samples": null,
  "pca_components": 128,
  "cluster_selection_method": "eom"
}
```

---

# 11. Pipeline 3: FAISS GPU spherical KMeans

Implement in:

```text
pipelines/kmeans_pipeline.py
```

Use FAISS GPU spherical KMeans as the only KMeans implementation.

Do not implement cuML KMeans.

Do not expose a backend switch.

## 11.1 Working matrix

Default:

```text
raw embeddings
→ compute raw norms
→ L2-normalized working matrix
→ FAISS GPU spherical KMeans
```

If `--no-normalize` is passed:

```text
raw embeddings
→ compute raw norms
→ raw working matrix
→ FAISS GPU spherical KMeans
```

However, log a warning when KMeans is run with `--no-normalize`, because spherical KMeans is intended for directional/cosine-style clustering.

## 11.2 FAISS implementation

Use:

```python
import faiss

kmeans = faiss.Kmeans(
    d=X.shape[1],
    k=n_clusters,
    niter=max_iter,
    nredo=nredo,
    spherical=True,
    gpu=True,
    seed=random_state,
    verbose=True,
)

kmeans.train(X_train_float32)
```

Then assign every valid row:

```python
distances, labels = kmeans.index.search(X_float32, 1)
```

Export:

```python
cluster_id = labels[:, 0]
```

Use chunked assignment for large matrices.

The implementation must not load all embeddings into GPU memory at once unless the matrix is small enough.

## 11.3 KMeans diagnostics

Because KMeans assigns every valid row, do not mark low-quality points as noise.

Export:

```text
cosine_to_centroid
```

Compute it using normalized embeddings and normalized centroids:

```python
centers = kmeans.centroids.astype("float32")
centers = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)

cosine_to_centroid[i] = dot(X_normed[i], centers[cluster_id[i]])
```

Invalid zero-norm rows get:

```text
cluster_id = -1
is_noise = true
cosine_to_centroid = null
```

Valid KMeans rows get:

```text
is_noise = false
```

## 11.4 KMeans outputs

Write:

```text
out/labels.parquet
out/cluster_centers.npy
out/cluster_summary.parquet
out/run_config.json
out/preflight.json
out/metrics.json
```

`labels.parquet` columns:

```text
row_id
cluster_id
method                  # "kmeans"
is_noise
embedding_norm
cosine_to_centroid
```

`metrics.json`:

```json
{
  "method": "kmeans",
  "n_rows": 123,
  "n_clusters": 10,
  "target_cluster_size": 1000,
  "max_iter": 300,
  "nredo": 1,
  "backend": "faiss_gpu_spherical"
}
```

---

# 12. Output format and metadata handling

Do not rewrite the full metadata JSONL by default.

Always output labels by row ID:

```text
row_id → cluster_id
```

Downstream code can join labels back to metadata by row order or `row_id`.

Implement optional export:

```bash
--export-jsonl
```

If enabled, write:

```text
out/metadata_with_labels.jsonl
```

This must stream through `metadata.jsonl` line by line and append:

```json
{
  "row_id": 123,
  "cluster_id": 45,
  "clustering_method": "leiden"
}
```

Do not load the full metadata JSONL into memory.

---

# 13. Metrics and summaries

Implement lightweight metrics only.

`metrics.py` should compute:

```text
n_rows
n_clusters
n_noise / n_unassigned
noise_fraction
cluster_size_min
cluster_size_p25
cluster_size_median
cluster_size_p75
cluster_size_max
embedding_norm_mean
embedding_norm_std
```

For each cluster, write:

```text
cluster_summary.parquet
```

Columns:

```text
cluster_id
size
embedding_norm_mean
embedding_norm_std
```

If metadata contains any of these fields:

```text
species
label
scientific_name
common_name
class
category
```

then include optional top-label summaries:

```text
top_label_field
top_label_value
top_label_count
top_label_fraction
```

Do not require these metadata fields.

Optional sampled metric:

```text
sampled_silhouette
```

Compute only on a sample, default:

```text
sample_metrics = 50000
```

Skip if too expensive.

---

# 14. Handling noise/artifact rows

Do not build a separate artifact-removal model.

Implement this behavior:

```text
HDBSCAN:
  cluster_id = -1 from HDBSCAN means noise.

Leiden:
  rows absent from the mutual kNN graph get cluster_id = -1.
  export graph_degree and mean_neighbor_similarity.

KMeans:
  assign every valid row.
  export cosine_to_centroid.
  low cosine_to_centroid can be reviewed downstream.

All methods:
  zero-norm invalid rows get cluster_id = -1.
```

Do not discard rows based on metadata labels.

---

# 15. Run directory contract

Every run must produce:

```text
runs/<run_name>/
  run_config.json
  preflight.json
  metrics.json

  labels.parquet
  cluster_summary.parquet

  intermediate/
    norms.npy
    normalized.npy              # if normalization enabled
    pca.npy                     # only if PCA enabled
    pca_normalized.npy          # only if PCA enabled
    knn_indices.npy             # only Leiden
    knn_scores.npy              # only Leiden
    mutual_edges.parquet        # only Leiden

  logs/
    run.log
```

Intermediate files should be reusable.

If the user reruns Leiden with a different `resolution`, reuse existing normalized embeddings, kNN results, and mutual graph when compatible.

Do not hide intermediate paths.

Log them clearly.

---

# 16. Code quality requirements

Use:

```text
type hints
dataclasses or pydantic configs
chunked processing
explicit random_state
structured JSON run configs
clear exceptions
pytest tests
```

Do not:

```text
overwrite input .npy
load full metadata into memory
create dense N x N distance matrices
standardize embedding dimensions
silently convert NaN/Inf rows
add embedding extraction logic
add audio logic
add extra clustering algorithms
add alternative KMeans backend
add environment.yml
add examples/ folder
```

---

# 17. Minimum tests

Create synthetic test data inside test fixtures, not inside an `examples/` folder.

Synthetic data:

```text
N = 200
D = 1536
3 obvious clusters + 10 noise-like points
```

Required tests:

```text
test_validation.py
  validates .npy shape
  validates metadata row count
  catches NaN
  catches Inf
  handles zero-norm rows

test_preprocessing.py
  normalized rows have norm approximately 1
  zero rows remain zero
  PCA output shape is correct
  PCA-renormalized rows have norm approximately 1

test_small_leiden.py
  CLI runs
  labels.parquet exists
  all row_id values are present
  some non-negative clusters exist

test_small_hdbscan.py
  CLI runs
  labels.parquet exists
  probability column exists
  noise rows are allowed

test_small_kmeans.py
  CLI runs
  labels.parquet exists
  cluster_centers.npy exists
  cosine_to_centroid column exists
  valid rows have non-negative cluster IDs
```

Mark GPU integration tests with:

```python
@pytest.mark.gpu
```

---

# 18. Scripts

Add these scripts under `scripts/`:

```text
scripts/validate_inputs.py
scripts/run_leiden.py
scripts/run_hdbscan.py
scripts/run_kmeans.py
```

Each script should be a thin wrapper around the package code.

They should not duplicate pipeline logic.

Example behavior:

```bash
python scripts/validate_inputs.py \
  --embeddings /data/embeddings.npy \
  --metadata /data/metadata.jsonl

python scripts/run_leiden.py \
  --embeddings /data/embeddings.npy \
  --metadata /data/metadata.jsonl \
  --out runs/leiden_k50_res1
```

The CLI and scripts must call the same underlying functions.

---

# 19. README content

README should include:

```text
1. What this repo does
2. What this repo does not do
3. Input file requirements
4. Dependency assumptions
5. Running validation
6. Running Leiden
7. Running HDBSCAN
8. Running FAISS spherical KMeans
9. Outputs
10. How to join labels back to metadata
11. Known limitations
```

The README must state:

```text
This repository assumes CUDA 12.
This repository assumes RAPIDS has already been installed by the user.
This repository does not install RAPIDS.
This repository does not extract embeddings.
```

Example README commands:

```bash
pip install -r requirements.txt
pip install -e .

embedcluster leiden \
  --embeddings /data/embeddings.npy \
  --metadata /data/metadata.jsonl \
  --out runs/leiden_k50_res1

embedcluster hdbscan \
  --embeddings /data/embeddings.npy \
  --metadata /data/metadata.jsonl \
  --out runs/hdbscan_pca128 \
  --pca-components 128

embedcluster kmeans \
  --embeddings /data/embeddings.npy \
  --metadata /data/metadata.jsonl \
  --out runs/kmeans_coarse \
  --target-cluster-size 1000
```

---

# 20. Implementation order

Implement in this order:

```text
1. Package skeleton, pyproject.toml, CLI stub.
2. Input validation.
3. Run directory creation and output writer.
4. Norm computation.
5. L2-normalized working matrix creation.
6. FAISS GPU spherical KMeans pipeline.
7. HDBSCAN pipeline.
8. PCA and PCA re-normalization for HDBSCAN.
9. cuVS kNN backend.
10. Mutual kNN graph builder.
11. cuGraph Leiden pipeline.
12. Metrics and cluster summaries.
13. Scripts.
14. Tests.
15. README.
```

This order gives a working baseline first, then adds noise-aware clustering, then the graph clustering pipeline.

---

# 21. Final default configuration

Use these defaults unless explicitly overridden:

```json
{
  "shared": {
    "normalize": true,
    "batch_size": "auto",
    "random_state": 42,
    "sample_metrics": 50000
  },
  "leiden": {
    "k": 50,
    "resolution": 1.0,
    "min_similarity": 0.0
  },
  "hdbscan": {
    "pca_components": 128,
    "min_cluster_size": 50,
    "min_samples": null,
    "cluster_selection_method": "eom"
  },
  "kmeans": {
    "target_cluster_size": 1000,
    "max_iter": 300,
    "nredo": 1,
    "backend": "faiss_gpu_spherical"
  }
}
```

The most important implementation rule:

```text
Raw embeddings are immutable.
Normalization and PCA are explicit working views.
Every output is row-aligned through row_id.
No dense all-pairs distance matrix is ever created.
```

[1]: https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks%3A-clustering%2C-PCA%2C-quantization "Faiss building blocks: clustering, PCA, quantization · facebookresearch/faiss Wiki · GitHub"
[2]: https://docs.rapids.ai/api/cugraph/stable/api_docs/api/cugraph/cugraph.leiden/ "cugraph.leiden — cugraph-docs 26.04.00 documentation"
[3]: https://docs.rapids.ai/api/cuvs/stable/ "cuVS: Vector Search and Clustering on the GPU — cuvs"
[4]: https://pypi.org/project/faiss-gpu-cu12/ "faiss-gpu-cu12 · PyPI"
[5]: https://docs.rapids.ai/api/cuml/stable/api/generated/cuml.cluster.hdbscan.HDBSCAN/ "HDBSCAN — cuml 26.04.00 documentation"

