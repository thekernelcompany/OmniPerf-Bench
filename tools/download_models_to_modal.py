"""
Download HuggingFace models from OmniPerf dataset to Modal volume.

This script downloads all models required by the Ayushnangia/omniperf_v1 dataset
to a Modal volume for use in benchmarks.

Usage:
    # Deploy the Modal app first
    modal deploy tools/download_models_to_modal.py

    # Run the download (this will take hours for large models)
    modal run tools/download_models_to_modal.py::download_all_models

    # Check download status
    modal run tools/download_models_to_modal.py::list_downloaded_models
"""

import modal
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# Modal app configuration
app = modal.App("omniperf-model-downloader")

# Volume for storing downloaded models (persistent storage)
# Using 'omniperf-models' as a dedicated volume for benchmark models
models_volume = modal.Volume.from_name("omniperf-models", create_if_missing=True)

# High-end CPU image for downloading models
# Using lots of RAM for handling large model files during download
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub>=0.24.0",
        "hf_transfer",  # For faster downloads
        "transformers>=4.40.0",
        "torch>=2.0.0",
        "datasets>=2.18.0",
        "tqdm",
        "requests",
    )
    .env({
        "HF_HOME": "/models/.cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Faster downloads
    })
)

# All valid models from the omniperf_v1 dataset
# Including ALL models - no filtering by size
OMNIPERF_MODELS = [
    # OpenGVLab models (Vision)
    "OpenGVLab/InternVL2_5-8B",

    # Qwen models
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2-VL-2B",
    "Qwen/Qwen2-VL-7B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-VL-3B",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-30B-A3B-FP8",
    "Qwen/Qwen3-7B-Instruct",

    # RedHatAI models
    "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",

    # DeepSeek models (ALL including large ones)
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-V2",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-V3-0324",

    # Facebook/Meta OPT models
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",

    # Google models
    "google/gemma-2b",
    "google/gemma-2-2b",
    "google/gemma-3-12b-it",

    # Huggyllama models
    "huggyllama/llama-7b",

    # IBM models
    "ibm-ai-platform/Bamba-9B",
    "ibm-ai-platform/Bamba-9B-v2",
    "ibm-granite/granite-4.0-tiny-preview",

    # LLaVA models (Vision-Language)
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-1.5-13b-hf",

    # lmsys models
    "lmsys/sglang-ci-dsv3-test",

    # Meta Llama models (ALL including large ones)
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Llama-3-8B",
    "meta-llama/Llama-3-70B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B",

    # Microsoft models
    "microsoft/phi-1_5",
    "microsoft/Phi-3-medium-128k-instruct",

    # Mistral models
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",

    # NeuralMagic models (quantized/optimized)
    "neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    "nm-testing/Meta-Llama-3-8B-Instruct-FP8-KV",
    "nm-testing/Mixtral-8x7B-Instruct-v0.1-FP8",
    "nm-testing/Meta-Llama-3-70B-Instruct-FP8",

    # NVIDIA models
    "nvidia/Nemotron-4-340B-Instruct",
]

# Smaller models for quick testing
SMALL_MODELS = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "microsoft/phi-1_5",
    "meta-llama/Llama-3.2-1B-Instruct",
]

# Medium models (7B-13B)
MEDIUM_MODELS = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B",
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2b",
    "google/gemma-2-2b",
    "huggyllama/llama-7b",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-1.5-13b-hf",
]

# Large models (requires significant storage and time)
LARGE_MODELS = [
    "Qwen/Qwen3-30B-A3B",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Llama-3-70B",
    "meta-llama/Meta-Llama-3-70B",
    "deepseek-ai/DeepSeek-V2",
    "deepseek-ai/DeepSeek-V3",
    "nvidia/Nemotron-4-340B-Instruct",
]


@app.function(
    image=download_image,
    volumes={"/models": models_volume},
    timeout=14400,  # 4 hours per model
    cpu=8,
    memory=65536,  # 64GB RAM for handling large files
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_single_model(model_id: str, force: bool = False) -> Dict[str, Any]:
    """
    Download a single HuggingFace model to the Modal volume.

    Args:
        model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3-8B")
        force: If True, re-download even if model exists

    Returns:
        Dict with download status and metadata
    """
    from huggingface_hub import snapshot_download, HfApi
    import time
    import shutil

    result = {
        "model_id": model_id,
        "status": "error",
        "path": None,
        "size_gb": None,
        "error": None,
        "duration_s": 0,
    }

    start_time = time.time()

    # Target path maintains HuggingFace directory structure
    # e.g., /models/meta-llama/Llama-3-8B
    model_path = Path(f"/models/{model_id}")

    # Check if already downloaded
    if model_path.exists() and not force:
        # Verify it has files
        files = list(model_path.rglob("*.safetensors")) + list(model_path.rglob("*.bin"))
        if files:
            print(f"[SKIP] {model_id} already downloaded ({len(files)} model files)")
            result["status"] = "already_exists"
            result["path"] = str(model_path)
            result["duration_s"] = time.time() - start_time
            return result

    print(f"[DOWNLOAD] Starting download of {model_id}...")

    try:
        # Create parent directory
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Get model info first
        api = HfApi()
        try:
            model_info = api.model_info(model_id)
            print(f"  Model type: {model_info.pipeline_tag or 'unknown'}")
            print(f"  Downloads: {model_info.downloads or 'unknown'}")
        except Exception as e:
            print(f"  Could not get model info: {e}")

        # Download using snapshot_download for reliability
        # This handles large files, resumable downloads, and proper caching
        downloaded_path = snapshot_download(
            repo_id=model_id,
            local_dir=str(model_path),
            local_dir_use_symlinks=False,  # Copy files, don't symlink
            resume_download=True,
            # Download only necessary files (skip .git, etc.)
            ignore_patterns=[
                "*.md",
                "*.txt",
                ".git*",
                "*.h5",  # Skip TensorFlow weights if pytorch available
                "tf_*",
                "flax_*",
                "rust_*",
            ],
        )

        # Calculate total size
        total_size = 0
        for f in model_path.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size

        size_gb = total_size / (1024**3)

        print(f"[SUCCESS] Downloaded {model_id} ({size_gb:.2f} GB)")

        # Commit volume to persist
        models_volume.commit()

        result["status"] = "success"
        result["path"] = str(model_path)
        result["size_gb"] = round(size_gb, 2)

    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Failed to download {model_id}: {error_msg}")
        result["error"] = error_msg

        # Clean up partial download
        if model_path.exists():
            shutil.rmtree(model_path, ignore_errors=True)

    result["duration_s"] = round(time.time() - start_time, 1)
    return result


@app.function(
    image=download_image,
    volumes={"/models": models_volume},
    timeout=3600,
    cpu=2,
    memory=8192,
)
def list_downloaded_models() -> List[Dict[str, Any]]:
    """
    List all models currently downloaded in the volume.

    Returns:
        List of dicts with model info (path, size, files)
    """
    from pathlib import Path

    models = []
    base_path = Path("/models")

    if not base_path.exists():
        print("No models directory found")
        return []

    # Find all directories that look like model repos (org/model structure)
    for org_dir in base_path.iterdir():
        if org_dir.is_dir() and not org_dir.name.startswith("."):
            for model_dir in org_dir.iterdir():
                if model_dir.is_dir():
                    model_id = f"{org_dir.name}/{model_dir.name}"

                    # Count files
                    safetensors = list(model_dir.rglob("*.safetensors"))
                    bins = list(model_dir.rglob("*.bin"))
                    configs = list(model_dir.rglob("config.json"))

                    # Calculate size
                    total_size = 0
                    for f in model_dir.rglob("*"):
                        if f.is_file():
                            total_size += f.stat().st_size

                    size_gb = total_size / (1024**3)

                    models.append({
                        "model_id": model_id,
                        "path": str(model_dir),
                        "size_gb": round(size_gb, 2),
                        "safetensors_files": len(safetensors),
                        "bin_files": len(bins),
                        "has_config": len(configs) > 0,
                    })

    # Sort by model_id
    models.sort(key=lambda x: x["model_id"])

    print(f"\n{'='*60}")
    print(f"Downloaded Models Summary")
    print(f"{'='*60}")

    total_size = 0
    for m in models:
        total_size += m["size_gb"]
        status = "✓" if m["has_config"] else "?"
        files = m["safetensors_files"] or m["bin_files"]
        print(f"{status} {m['model_id']:<50} {m['size_gb']:>8.2f} GB ({files} model files)")

    print(f"{'='*60}")
    print(f"Total: {len(models)} models, {total_size:.2f} GB")

    return models


@app.function(
    image=download_image,
    timeout=300,
)
def get_models_to_download(
    include_large: bool = False,
    only_small: bool = False,
) -> List[str]:
    """
    Get list of models to download based on size filters.

    Args:
        include_large: Include large models (70B+)
        only_small: Only include small models (<3B)
    """
    if only_small:
        return SMALL_MODELS

    models = SMALL_MODELS + MEDIUM_MODELS

    if include_large:
        models.extend(LARGE_MODELS)

    # Remove duplicates while preserving order
    seen = set()
    unique_models = []
    for m in models:
        if m not in seen:
            seen.add(m)
            unique_models.append(m)

    return unique_models


@app.local_entrypoint()
def download_all_models(
    include_large: bool = False,
    only_small: bool = False,
    force: bool = False,
    parallel: int = 2,
):
    """
    Download all models from the OmniPerf dataset.

    Args:
        include_large: Include large models (70B+, requires significant storage)
        only_small: Only download small models for testing
        force: Re-download even if model exists
        parallel: Number of parallel downloads (be careful with rate limits)
    """
    import json

    # Get list of models to download
    if only_small:
        models = SMALL_MODELS
        print(f"Downloading {len(models)} small models (for testing)...")
    elif include_large:
        models = OMNIPERF_MODELS + LARGE_MODELS
        print(f"Downloading {len(models)} models (including large)...")
    else:
        models = OMNIPERF_MODELS
        print(f"Downloading {len(models)} models...")

    # Remove duplicates
    models = list(dict.fromkeys(models))

    print(f"\nModels to download:")
    for i, m in enumerate(models, 1):
        print(f"  {i:2d}. {m}")

    print(f"\nStarting downloads (parallel={parallel})...")

    # Download models (can be parallelized with Modal's map)
    results = []

    # Use Modal's parallel execution for multiple models
    download_fn = modal.Function.from_name("omniperf-model-downloader", "download_single_model")

    # Process in batches to avoid overwhelming HuggingFace
    batch_size = parallel
    for i in range(0, len(models), batch_size):
        batch = models[i:i + batch_size]
        print(f"\n--- Batch {i//batch_size + 1}/{(len(models) + batch_size - 1)//batch_size} ---")

        # Run batch in parallel using starmap
        batch_results = list(download_fn.map(batch, kwargs={"force": force}))
        results.extend(batch_results)

        # Print batch results
        for r in batch_results:
            status_icon = "✓" if r["status"] == "success" else ("⏭" if r["status"] == "already_exists" else "✗")
            size_str = f"{r.get('size_gb', 0):.2f} GB" if r.get('size_gb') else "N/A"
            print(f"  {status_icon} {r['model_id']}: {r['status']} ({size_str}, {r['duration_s']:.1f}s)")

    # Summary
    print(f"\n{'='*60}")
    print(f"Download Summary")
    print(f"{'='*60}")

    success = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] == "already_exists"]
    failed = [r for r in results if r["status"] == "error"]

    print(f"  Successful: {len(success)}")
    print(f"  Already existed: {len(skipped)}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print(f"\nFailed downloads:")
        for r in failed:
            print(f"  - {r['model_id']}: {r.get('error', 'Unknown error')[:100]}")

    total_size = sum(r.get("size_gb", 0) for r in results if r.get("size_gb"))
    print(f"\nTotal downloaded: {total_size:.2f} GB")

    # Save results to file
    results_file = "/tmp/download_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


@app.local_entrypoint()
def check_status():
    """Check the status of downloaded models."""
    list_fn = modal.Function.from_name("omniperf-model-downloader", "list_downloaded_models")
    models = list_fn.remote()

    print(f"\n{'='*60}")
    print(f"Downloaded Models in Volume")
    print(f"{'='*60}")

    total_size = 0
    for m in models:
        total_size += m.get("size_gb", 0)
        status = "✓" if m.get("has_config") else "?"
        files = m.get("safetensors_files", 0) or m.get("bin_files", 0)
        print(f"{status} {m['model_id']:<50} {m.get('size_gb', 0):>8.2f} GB ({files} model files)")

    print(f"{'='*60}")
    print(f"Total: {len(models)} models, {total_size:.2f} GB")

    return models


if __name__ == "__main__":
    # For local testing
    print("Run this script using Modal:")
    print("  modal deploy tools/download_models_to_modal.py")
    print("  modal run tools/download_models_to_modal.py::download_all_models --only-small")
    print("  modal run tools/download_models_to_modal.py::check_status")
