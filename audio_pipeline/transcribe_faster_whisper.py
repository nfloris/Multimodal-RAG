#!/usr/bin/env python
"""Run faster-whisper outside the Jupyter kernel and persist a JSON transcript."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language", default="")
    parser.add_argument("--inference-mode", choices=("sequential", "batched"), default="sequential")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--vad-filter", default="true")
    return parser


def _transcribe(args: argparse.Namespace) -> dict:
    from faster_whisper import WhisperModel

    print(
        "Loading isolated faster-whisper model "
        f"on {args.device} (compute={args.compute_type}, mode={args.inference_mode}) ...",
        flush=True,
    )
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )

    language = args.language or None
    common_kwargs = {
        "language": language,
        "vad_filter": _parse_bool(args.vad_filter),
        "beam_size": args.beam_size,
        "word_timestamps": False,
    }
    if args.inference_mode == "batched":
        from faster_whisper import BatchedInferencePipeline

        runner = BatchedInferencePipeline(model=model)
        segments_iter, info = runner.transcribe(
            args.audio,
            batch_size=args.batch_size,
            **common_kwargs,
        )
    else:
        segments_iter, info = model.transcribe(args.audio, **common_kwargs)

    chunks = []
    texts = []
    for segment in segments_iter:
        text = segment.text.strip()
        if not text:
            continue
        chunks.append(
            {
                "text": text,
                "timestamp": [float(segment.start), float(segment.end)],
            }
        )
        texts.append(text)

    detected_language = getattr(info, "language", None)
    language_probability = getattr(info, "language_probability", None)
    if detected_language:
        probability_label = (
            f"{language_probability:.2f}" if language_probability is not None else "unknown"
        )
        print(f"Detected language: {detected_language} ({probability_label})", flush=True)

    return {
        "text": " ".join(texts).strip(),
        "chunks": chunks,
        "metadata": {
            "backend": "faster-whisper",
            "model": args.model,
            "device": args.device,
            "compute_type": args.compute_type,
            "inference_mode": args.inference_mode,
            "batch_size": args.batch_size if args.inference_mode == "batched" else None,
            "beam_size": args.beam_size,
            "vad_filter": _parse_bool(args.vad_filter),
            "detected_language": detected_language,
            "language_probability": language_probability,
            "isolated_subprocess": True,
        },
    }


def main() -> int:
    args = _build_parser().parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)

    transcript = _transcribe(args)
    temporary_path.write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)
    print(f"Saved transcript cache: {output_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
