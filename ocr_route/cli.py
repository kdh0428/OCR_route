"""Command line interface for the OCR routing pipeline."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

from tqdm import tqdm

from .main import Pipeline, create_pipeline
from .routing import RoutingResult
from .utils import append_jsonl, ensure_directory, iter_csv_records
from .utils.text import extract_assistant_answer

LOGGER = logging.getLogger(__name__)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Uncertainty-triggered OCR routing CLI")
    parser.add_argument("--image", help="Image path or URL", type=str, default=None)
    parser.add_argument("--question", help="Question text", type=str, default=None)
    parser.add_argument("--csv", help="Batch mode CSV with columns image,question", type=str, default=None)
    parser.add_argument("--out", help="Output JSONL path", type=str, default="outputs/results.jsonl")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument(
        "--cuda-memory-fraction",
        type=float,
        default=None,
        help="Optional per-process CUDA memory cap as a fraction (0-1). Applied before model load.",
    )
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable KV cache during generation to reduce VRAM usage (slower).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=-1,
        help="Max new tokens during generation (-1 to disable limit).",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Clamp image area (width*height) to this many pixels (omit or set negative to disable clamping).",
    )
    parser.add_argument("--entropy-th", type=float, default=0.7)
    parser.add_argument("--margin-th", type=float, default=1.0)
    parser.add_argument("--entropy-a", type=float, help="Intercept for length-aware entropy threshold")
    parser.add_argument("--entropy-b", type=float, help="Slope for length-aware entropy threshold")
    parser.add_argument("--margin-tau", type=float, help="Dynamic margin threshold override")
    parser.add_argument("--history-k", type=int, default=4, help="Number of trailing tokens to enforce hysteresis")
    parser.add_argument(
        "--ocr-engine",
        type=str,
        choices=["paddleocr", "easyocr", "tesseract", "doctr"],
        default="paddleocr",
    )
    parser.add_argument("--no-routing", action="store_true")
    parser.add_argument("--always-ocr", action="store_true")
    parser.add_argument("--return-json", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--enable-wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, help="Weights & Biases project name.")
    parser.add_argument("--wandb-entity", type=str, help="Weights & Biases entity (team) name.")
    parser.add_argument("--wandb-run-name", type=str, help="Optional explicit W&B run name.")
    parser.add_argument(
        "--wandb-tag",
        dest="wandb_tags",
        action="append",
        help="Additional W&B tag (repeatable). Dataset/model tags are appended automatically.",
    )
    parser.add_argument(
        "--temperature-scale",
        type=float,
        default=1.0,
        help="Temperature scaling factor applied to decoder scores for calibration (default: 1.0).",
    )
    parser.add_argument(
        "--ocr-prompt-mode",
        choices=["default", "verify"],
        default="default",
        help="How to inject OCR text in the second pass (default: legacy prompt, verify: cross-check/fix mode).",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    """Create the pipeline using parsed arguments."""
    cuda_fraction = getattr(args, "cuda_memory_fraction", None)
    if cuda_fraction is not None:
        try:
            import torch

            if not (0.0 < float(cuda_fraction) <= 1.0):
                raise ValueError("--cuda-memory-fraction must be in (0, 1].")
            if torch.cuda.is_available():  # pragma: no cover - hardware dependent
                torch.cuda.set_per_process_memory_fraction(float(cuda_fraction))
                LOGGER.info("Set CUDA per-process memory fraction to %.3f", float(cuda_fraction))
            else:
                LOGGER.warning("--cuda-memory-fraction set but CUDA is not available; ignoring.")
        except Exception as exc:
            raise ValueError(f"Failed to apply --cuda-memory-fraction={cuda_fraction}: {exc}") from exc

    device_map = args.device_map
    max_pixels = args.max_pixels
    if max_pixels is not None and max_pixels < 0:
        max_pixels = None
    dataset_id = getattr(args, "dataset", None)
    return create_pipeline(
        model_id=args.model_id,
        dtype=args.dtype,
        device_map=device_map,
        min_pixels=args.min_pixels,
        max_pixels=max_pixels,
        entropy_th=args.entropy_th,
        margin_th=args.margin_th,
        entropy_a=args.entropy_a,
        entropy_b=args.entropy_b,
        margin_tau=args.margin_tau,
        history_k=args.history_k,
        ocr_engine=args.ocr_engine,
        no_routing=args.no_routing,
        always_ocr=args.always_ocr,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        use_kv_cache=not bool(getattr(args, "no_kv_cache", False)),
        seed=args.seed,
        enable_wandb=args.enable_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=tuple(args.wandb_tags) if args.wandb_tags else None,
        temperature_scale=args.temperature_scale,
        ocr_prompt_mode=args.ocr_prompt_mode,
        dataset_id=dataset_id,
    )


def result_to_dict(image_source: str, result: RoutingResult, model_id: str, ocr_engine: str) -> Dict[str, object]:
    """Convert a RoutingResult into a serializable dictionary."""
    first_metrics = result.first.metrics
    second_metrics = result.second.metrics if result.second else None
    answer_clean = extract_assistant_answer(result.answer)
    first_answer_clean = extract_assistant_answer(result.first.answer)
    second_answer_clean = extract_assistant_answer(result.second.answer) if result.second else None
    payload: Dict[str, object] = {
        "answer": answer_clean,
        "used_ocr": result.used_ocr,
        "answer_first": first_answer_clean,
        "answer_second": second_answer_clean,
        "raw_answer": result.answer,
        "raw_answer_first": result.first.answer,
        "raw_answer_second": result.second.answer if result.second else None,
        "mean_entropy_first": first_metrics.mean_entropy,
        "normalized_entropy_first": first_metrics.normalized_entropy,
        "min_margin_first": first_metrics.min_margin,
        "mean_entropy_second": second_metrics.mean_entropy if second_metrics else None,
        "normalized_entropy_second": second_metrics.normalized_entropy if second_metrics else None,
        "min_margin_second": second_metrics.min_margin if second_metrics else None,
        "triggered": result.triggered,
        "ocr_engine": ocr_engine if result.used_ocr else None,
        "latency_ms_first": int(result.timings.get("latency_ms_first", 0.0)),
        "latency_ms_ocr": int(result.timings.get("latency_ms_ocr", 0.0)),
        "latency_ms_second": int(result.timings.get("latency_ms_second", 0.0)),
        "total_latency_ms": int(result.total_latency_ms),
        "total_latency_s": float(result.total_latency_ms / 1000.0),
        "model_id": model_id,
        "image_source": image_source,
        "entropies_first": first_metrics.filtered_entropies,
        "entropies_second": second_metrics.filtered_entropies if second_metrics else None,
        "final_normalized_entropy": result.final_metrics.normalized_entropy,
        "sequence_confidence_first": first_metrics.sequence_confidence,
        "sequence_confidence_second": second_metrics.sequence_confidence if second_metrics else None,
        "sequence_confidence_final": result.final_metrics.sequence_confidence,
        "token_confidences_first": first_metrics.token_confidences,
        "token_confidences_second": second_metrics.token_confidences if second_metrics else None,
        "final_mean_entropy": result.final_metrics.mean_entropy,
        "final_min_margin": result.final_metrics.min_margin,
    }
    if result.second is None:
        payload["min_margin_second"] = None
        payload["mean_entropy_second"] = None
        payload["entropies_second"] = None
        payload["sequence_confidence_second"] = None
        payload["token_confidences_second"] = None
    return payload


def run_single(pipeline: Pipeline, args: argparse.Namespace) -> int:
    """Execute a single example."""
    image = args.image
    question = args.question
    if not image or not question:
        LOGGER.error("Single mode requires both --image and --question")
        return 1

    result = pipeline.run(image=image, question=question)
    payload = result_to_dict(image, result, args.model_id, args.ocr_engine)

    if args.return_json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        entropy_second = (
            f"{payload['mean_entropy_second']:.3f}"
            if payload["mean_entropy_second"] is not None
            else "n/a"
        )
        margin_second = (
            f"{payload['min_margin_second']:.3f}"
            if payload["min_margin_second"] is not None
            else "n/a"
        )
        print(f"Answer: {payload['answer']}")
        print(f"Triggered: {payload['triggered']} | Used OCR: {payload['used_ocr']}")
        print(f"First pass answer: {payload['answer_first']}")
        second_answer = payload.get("answer_second") or "n/a"
        print(f"Second pass answer: {second_answer}")
        print(f"Entropy (first/second): {payload['mean_entropy_first']:.3f}/{entropy_second}")
        print(f"Margin (first/second): {payload['min_margin_first']:.3f}/{margin_second}")
        print(f"Latency total(ms): {payload['total_latency_ms']}")
        ent_first = payload.get("entropies_first") or []
        if ent_first:
            print("Token entropies - first pass:")
            for idx, value in enumerate(ent_first, start=1):
                print(f"  [{idx:02d}] {value:.4f}")
        ent_second = payload.get("entropies_second") or []
        if ent_second:
            print("Token entropies - second pass:")
            for idx, value in enumerate(ent_second, start=1):
                print(f"  [{idx:02d}] {value:.4f}")
    return 0


def run_batch(pipeline: Pipeline, args: argparse.Namespace) -> int:
    """Execute batch mode from a CSV file."""
    if not args.csv:
        LOGGER.error("Batch mode requires --csv")
        return 1
    csv_path = Path(args.csv)
    if not csv_path.exists():
        LOGGER.error("CSV file not found: %s", csv_path)
        return 1

    out_path = Path(args.out)
    ensure_directory(out_path.parent)

    with open(out_path, "w", encoding="utf-8") as _:
        pass  # truncate existing file

    records = list(iter_csv_records(csv_path))
    for row in tqdm(records, desc="Routing"):
        image = row.get("image")
        question = row.get("question")
        if not image or not question:
            LOGGER.warning("Skipping row with missing fields: %s", row)
            continue
        try:
            result = pipeline.run(image=image, question=question)
            payload = result_to_dict(image, result, args.model_id, args.ocr_engine)
            append_jsonl(out_path, payload)
        except Exception as exc:  # pragma: no cover - runtime errors
            LOGGER.exception("Failed to process row %s: %s", row, exc)
    LOGGER.info("Wrote batch results to %s", out_path)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    batch_mode = bool(args.csv)
    if batch_mode:
        LOGGER.info("Running in batch mode")
    else:
        LOGGER.info("Running in single mode")

    pipeline = build_pipeline(args)

    try:
        if batch_mode:
            return run_batch(pipeline, args)
        return run_single(pipeline, args)
    except KeyboardInterrupt:  # pragma: no cover - CLI interaction
        LOGGER.warning("Interrupted by user")
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
