# embedcluster

GPU-accelerated clustering for pre-extracted embedding matrices.

---

## 1. What this repo does

Clusters embeddings stored in `embeddings.npy` (shape `[N, D]`) using three GPU-accelerated pipelines:

- **Leiden** — cuVS kNN → mutual weighted kNN graph → cuGraph Leiden community detection
- **HDBSCAN** — optional PCA + L2 re-normalization → cuML HDBSCAN
- **Spherical KMeans** — FAISS GPU spherical KMeans (cosine/directional)

Every run writes row-aligned labels (`labels.parquet`), per-cluster summaries, and structured metadata to a run directory.

---

## 2. What this repo does not do

- Does **not** load audio, images, or raw media.
- Does **not** extract, compute, or fine-tune embeddings.
- Does **not** install RAPIDS.
- Does **not** select CUDA versions or manage drivers.

---

## 3. Input file requirements

```
embeddings.npy    — shape [N, D], dtype float16/float32/float64
metadata.jsonl    — exactly N newline-delimited JSON objects (one per embedding row)
```

- Raw embeddings are **never** modified. All working views are written to the run directory.
- Metadata keys are not required to follow any schema. They are joined back to results by row position (`row_id`).

---

## 4. Dependency assumptions

This repository assumes:

- **CUDA 12** is installed and working.
- **RAPIDS** (cuDF, cuML, cuGraph, cuVS, CuPy, RMM) has already been installed by the user.
- This repository **does not install RAPIDS**.

Install non-RAPIDS dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## 5. Running validation

```bash
embedcluster --help

python scripts/validate_inputs.py \
  --embeddings /data/embeddings.npy \
  --metadata /data/metadata.jsonl
```

Validation checks: file existence, 2D shape, supported dtype (float16/32/64), metadata line count match, NaN/Inf scan, zero-norm row detection.

---

## 6. Running Leiden

```bash
embedcluster leiden \
  --embeddings /data/embeddings.npy \
  --metadata /data/metadata.jsonl \
  --out runs/leiden_k50_res1

# Or via script:
python scripts/run_leiden.py \
  --embeddings /data/embeddings.npy \
  --metadata /data/metadata.jsonl \
  --out runs/leiden_k50_res1
```

Key parameters:

```
--k INT               k-nearest neighbors (default: 50)
--resolution FLOAT    Leiden resolution (default: 1.0)
--min-similarity FLOAT  minimum edge weight to keep (default: 0.0)
```

---

## 7. Running HDBSCAN

```bash
embedcluster hdbscan \
  --embeddings /data/embeddings.npy \
  --metadata /data/metadata.jsonl \
  --out runs/hdbscan_pca128 \
  --pca-components 128

# Or via script:
python scripts/run_hdbscan.py \
  --embeddings /data/embeddings.npy \
  --metadata /data/metadata.jsonl \
  --out runs/hdbscan_pca128 \
  --pca-components 128
```

Key parameters:

```
--min-cluster-size INT      minimum cluster size (default: 50)
--min-samples INT|none      HDBSCAN min_samples (default: none)
--pca-components INT|none   PCA before HDBSCAN; use 'none' to disable (default: 128)
--cluster-selection eom|leaf  cluster selection method (default: eom)
```

---

## 8. Running FAISS Spherical KMeans

```bash
embedcluster kmeans \
  --embeddings /data/embeddings.npy \
  --metadata /data/metadata.jsonl \
  --out runs/kmeans_coarse \
  --target-cluster-size 1000

# Or via script:
python scripts/run_kmeans.py \
  --embeddings /data/embeddings.npy \
  --metadata /data/metadata.jsonl \
  --out runs/kmeans_coarse \
  --target-cluster-size 1000
```

Key parameters:

```
--n-clusters INT            explicit cluster count (optional; computed from target if omitted)
--target-cluster-size INT   target rows per cluster (default: 1000)
--max-iter INT              training iterations (default: 300)
--nredo INT                 independent restarts (default: 1)
```

---

## 9. Outputs

Every run produces a canonical directory layout:

```
runs/<run_name>/
  run_config.json             — full pipeline configuration
  preflight.json              — GPU info + memory estimates
  metrics.json                — cluster quality metrics

  labels.parquet              — one row per embedding; always row-aligned
  cluster_summary.parquet     — per-cluster size and norm statistics

  cluster_centers.npy         — KMeans only: raw centroid vectors [k, D]
  hdbscan_cluster_persistence.parquet  — HDBSCAN only

  intermediate/
    norms.npy                 — L2 norm per embedding row
    normalized.npy            — L2-normalized working matrix (if normalization enabled)
    pca.npy                   — PCA-transformed matrix (HDBSCAN only, if PCA enabled)
    pca_normalized.npy        — L2-renormalized PCA output (HDBSCAN only)
    knn_indices.npy           — Leiden only: [N, k] kNN indices
    knn_scores.npy            — Leiden only: [N, k] kNN scores
    mutual_edges.parquet      — Leiden only: undirected mutual kNN graph

  logs/
    run.log                   — structured run log
```

`labels.parquet` columns (all methods):

```
row_id          — integer index aligned to embedding row
cluster_id      — assigned cluster; -1 means noise/unassigned
method          — "leiden" | "hdbscan" | "kmeans"
is_noise        — true if cluster_id == -1
embedding_norm  — L2 norm of the raw embedding
```

Additional method-specific columns:

| Method   | Extra columns |
|----------|---------------|
| Leiden   | `graph_degree`, `mean_neighbor_similarity` |
| HDBSCAN  | `probability` |
| KMeans   | `cosine_to_centroid` |

---

## 10. Joining labels back to metadata

Labels are row-aligned: `labels.parquet.row_id[i]` corresponds to `metadata.jsonl` line `i`.

**Python (pandas):**

```python
import pandas as pd, json

labels = pd.read_parquet("runs/leiden_k50_res1/labels.parquet")
meta = pd.DataFrame(
    json.loads(line) for line in open("metadata.jsonl")
)
meta["row_id"] = range(len(meta))
result = meta.merge(labels[["row_id", "cluster_id"]], on="row_id")
```

**Streaming export** (avoids loading metadata into memory):

```bash
embedcluster leiden \
  --embeddings /data/embeddings.npy \
  --metadata /data/metadata.jsonl \
  --out runs/leiden_k50_res1 \
  --export-jsonl
# writes: runs/leiden_k50_res1/metadata_with_labels.jsonl
```

---

## 11. Known limitations

- **cuML HDBSCAN** requires the full working matrix (post-PCA if enabled) in GPU VRAM. For large N and high-dimensional embeddings without PCA, memory requirements can be substantial. Use `--pca-components 128` (the default) to reduce VRAM usage.
- **FAISS KMeans training** sub-samples up to `256 × k` rows. For large N, training quality depends on the sub-sample being representative. Assignments are always done over all rows in batches.
- **cuVS brute-force kNN** (Leiden pipeline) loads the full matrix plus k-neighbor buffers to VRAM. For N > 5M at D=1536, verify available VRAM with `preflight.json` before running.
- **Silhouette score** (`sampled_silhouette` in `metrics.json`) is computed on a CPU sample of up to `--sample-metrics` rows using cosine distance. It is skipped automatically if `sklearn` is unavailable or if the number of clusters is too small.
- Intermediate files (`knn_indices.npy`, `normalized.npy`, etc.) are reused across re-runs with the same output directory. Changing `--k` or `--normalize` requires a fresh `--out` path.
