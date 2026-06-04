#!/usr/bin/env python3
"""
Download/cache all Hugging Face models used by the local RAG notebook.

Usage:
  python download_hf_models.py --profile defaults
  python download_hf_models.py --profile all-open
  python download_hf_models.py --profile all --include-gated
"""

from __future__ import annotations

import argparse
import os
from huggingface_hub import snapshot_download

DEFAULT_MODELS = [
    # Text summarization
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    # Image/table summarization + multimodal generation
    "Qwen/Qwen2.5-VL-3B-Instruct",
    # Embeddings
    "BAAI/bge-m3",
    # Reranking
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    # Audio
    "laion/larger_clap_general",
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    "openai/whisper-large-v3",
    # Evaluation
    "distilbert-base-uncased",
]

ALL_OPEN_MODELS = [
    # Local substitutes/defaults
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",

    # Image summarization options
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",

    # Embedding options
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1",
    "sentence-transformers/all-MiniLM-L6-v2",

    # Reranker
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",

    # Audio
    "laion/larger_clap_general",
    "laion/larger_clap_music",
    "laion/clap-htsat-unfused",
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    "openai/whisper-large-v3",

    # Evaluation
    "distilbert-base-uncased",

    # Video
    "OpenGVLab/InternVideo2-Stage2_1B-224p-f4",
    "OpenGVLab/InternVideo2_CLIP_S",

    # Generator
    "meta-llama/Llama-3.1-8B-Instruct",
]

def download(model_id: str, token: str | None = None) -> None:
    print(f"\n=== Downloading {model_id} ===")
    snapshot_download(
        repo_id=model_id,
        token=token,
        resume_download=True,
        local_files_only=False,
    )
    print(f"Cached: {model_id}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["defaults", "all-open", "all"],
        default="defaults",
        help="defaults = only models used by the local notebook; all-open = every open option in config; all = all-open plus gated/huge when --include-gated is set",
    )
    parser.add_argument(
        "--include-gated",
        action="store_true",
        help="Also attempt Meta Llama gated/huge models. Requires huggingface-cli login and license access.",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if args.profile == "defaults":
        models = DEFAULT_MODELS
    elif args.profile == "all-open":
        models = ALL_OPEN_MODELS
    else:
        models = list(ALL_OPEN_MODELS)

    print("HF_HOME:", os.environ.get("HF_HOME", "~/.cache/huggingface"))
    print("Models to download:", len(models))

    failures = []
    for model_id in models:
        try:
            download(model_id, token=token)
        except Exception as exc:
            print(f"FAILED: {model_id}\n{exc}")
            failures.append(model_id)

    if failures:
        print("\nSome models failed:")
        for model_id in failures:
            print(" -", model_id)
        raise SystemExit(1)

    print("\nAll requested models are cached.")

if __name__ == "__main__":
    main()
