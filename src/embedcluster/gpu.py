"""GPU info, batch-size heuristics, memory estimates, preflight checks."""

from __future__ import annotations

from typing import Any

from .logging_utils import get_logger

logger = get_logger(__name__)


_BYTES_PER_GB = 1024 ** 3


def _try_pynvml() -> tuple[Any, Any]:
    """Initialize pynvml. Returns (pynvml, handle) or (None, None) on failure."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml, handle
    except Exception as e:  # pragma: no cover - depends on driver
        logger.warning("pynvml unavailable: %s", e)
        return None, None


def get_gpu_info() -> dict:
    """Return GPU name, total memory, free memory, driver/CUDA versions.

    All keys are present even if pynvml fails; missing values are None.
    """
    info: dict[str, Any] = {
        "available": False,
        "gpu_name": None,
        "total_memory_gb": None,
        "free_memory_gb": None,
        "driver_version": None,
        "cuda_runtime_version": None,
    }
    pynvml, handle = _try_pynvml()
    if pynvml is None or handle is None:
        return info
    try:
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode("utf-8", errors="replace")
        try:
            cuda_runtime = pynvml.nvmlSystemGetCudaDriverVersion()
        except Exception:
            cuda_runtime = None
        info.update(
            {
                "available": True,
                "gpu_name": name,
                "total_memory_gb": round(mem.total / _BYTES_PER_GB, 3),
                "free_memory_gb": round(mem.free / _BYTES_PER_GB, 3),
                "driver_version": driver,
                "cuda_runtime_version": cuda_runtime,
            }
        )
    except Exception as e:  # pragma: no cover - depends on driver
        logger.warning("Failed to read GPU info: %s", e)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return info


def choose_batch_size(
    n_rows: int,
    n_dims: int,
    dtype_bytes: int,
    target_fraction_gpu_mem: float = 0.25,
) -> int:
    """Choose a conservative batch size for GPU transfers.

    Targets ~`target_fraction_gpu_mem` of free GPU memory per batch.
    Falls back to a static heuristic if GPU info is unavailable.
    """
    info = get_gpu_info()
    if info.get("free_memory_gb") is not None:
        free_bytes = float(info["free_memory_gb"]) * _BYTES_PER_GB
        budget = free_bytes * target_fraction_gpu_mem
        per_row = max(n_dims * dtype_bytes, 1)
        bs = int(budget // per_row)
    else:
        # No GPU info: use a static fallback (~256 MB per batch).
        bs = int((256 * 1024 * 1024) // max(n_dims * dtype_bytes, 1))
    bs = max(1024, min(bs, n_rows if n_rows > 0 else 1024))
    return bs


def estimate_run_memory(
    n_rows: int,
    n_dims: int,
    k: int | None,
    pipeline: str,
) -> dict:
    """Approximate VRAM/host memory estimates for logs and preflight checks.

    Returned values are rough upper bounds, in gigabytes.
    """
    out: dict[str, Any] = {
        "pipeline": pipeline,
        "n_rows": n_rows,
        "n_dims": n_dims,
    }
    f32 = 4
    matrix_gb = n_rows * n_dims * f32 / _BYTES_PER_GB
    out["working_matrix_gb_fp32"] = round(matrix_gb, 3)

    if pipeline == "leiden":
        if k is None:
            k = 50
        knn_idx_gb = n_rows * k * 4 / _BYTES_PER_GB
        knn_dist_gb = n_rows * k * f32 / _BYTES_PER_GB
        directed_edges = n_rows * k
        out.update(
            {
                "k": k,
                "knn_indices_gb": round(knn_idx_gb, 3),
                "knn_scores_gb": round(knn_dist_gb, 3),
                "estimated_edges_before_mutual": int(directed_edges),
            }
        )
    elif pipeline == "hdbscan":
        # cuML HDBSCAN currently requires the full matrix in VRAM.
        out["hdbscan_full_matrix_required_gb"] = round(matrix_gb, 3)
    elif pipeline == "kmeans":
        # FAISS GPU spherical KMeans handles batched assignment.
        out["kmeans_train_subsample_note"] = (
            "FAISS sub-samples up to max_points_per_centroid * n_clusters rows."
        )
    return out


def run_preflight_check(
    estimates: dict,
    safety_fraction: float = 0.85,
) -> dict:
    """Compare memory estimates against free GPU memory.

    Raises RuntimeError if a clear over-budget condition is detected.
    Returns a dict combining GPU info + estimates suitable for preflight.json.
    """
    info = get_gpu_info()
    free_gb = info.get("free_memory_gb")
    pipeline = estimates.get("pipeline", "")

    out = {
        "gpu": info,
        "estimates": estimates,
    }

    if free_gb is None:
        logger.warning("Free GPU memory unknown; skipping preflight memory check.")
        return out

    budget = float(free_gb) * safety_fraction

    def _hint(extra: str) -> str:
        base = (
            "Reduce memory usage. Suggestions: smaller --batch-size, smaller --k "
            "(Leiden), or run HDBSCAN with --pca-components 128. Note: changing "
            "--target-cluster-size is NOT a memory fix for KMeans."
        )
        return f"{extra} {base}"

    if pipeline == "hdbscan":
        need = float(estimates.get("hdbscan_full_matrix_required_gb", 0.0))
        if need > budget:
            raise RuntimeError(
                _hint(
                    f"HDBSCAN requires the full working matrix in VRAM "
                    f"(~{need:.2f} GB), but only {budget:.2f} GB is usable "
                    f"out of {free_gb:.2f} GB free."
                )
            )
    elif pipeline == "leiden":
        need = float(estimates.get("knn_indices_gb", 0.0)) + float(
            estimates.get("knn_scores_gb", 0.0)
        )
        if need > budget:
            raise RuntimeError(
                _hint(
                    f"Leiden kNN buffers (~{need:.2f} GB) exceed the usable "
                    f"VRAM budget ({budget:.2f} GB of {free_gb:.2f} GB free)."
                )
            )
    elif pipeline == "kmeans":
        # FAISS handles its own batching; only warn.
        logger.info(
            "Preflight: kmeans relies on FAISS internal batching; free=%.2f GB",
            free_gb,
        )

    return out
