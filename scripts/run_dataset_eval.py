#!/usr/bin/env python3
"""Run OCR routing experiments on multiple OCR datasets."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
import re
import os

from tqdm import tqdm

# Ensure local package is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import base64
import io

from ocr_route.cli import build_pipeline, parse_args, result_to_dict, setup_logging
from ocr_route.main import Pipeline
from ocr_route.routing import route
from ocr_route.utils import append_jsonl, ensure_directory
from ocr_route.dataset_registry import load_dataset_by_id


def _load_env(path: Path) -> None:
    """Lightweight .env loader for HF tokens and other secrets."""
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val
    except Exception:
        # Best-effort; ignore malformed lines
        pass


# Load env (e.g., HF tokens) from project root if present
env_path = PROJECT_ROOT / ".env"
example_path = PROJECT_ROOT / ".env.example"
_load_env(env_path if env_path.exists() else example_path)

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


def normalize(text: str) -> str:
    """Lowercase and strip whitespace for comparison."""
    return text.strip().lower()


def normalize_loose(text: str) -> str:
    """Lowercase, strip, and remove punctuation/spaces for relaxed matching."""
    lowered = normalize(text)
    return re.sub(r"[^a-z0-9]", "", lowered)


def _levenshtein(a: List[str], b: List[str]) -> int:
    """Compute Levenshtein distance between two sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, token_b in enumerate(b, start=1):
            cost = 0 if token_a == token_b else 1
            prev, dp[j] = dp[j], min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost,    # substitution
            )
    return dp[-1]


def _token_fscore(pred_tokens: List[str], ref_tokens: List[str]) -> Tuple[float, float, float]:
    """Compute precision/recall/F1 over token counters."""
    if not pred_tokens and not ref_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    precision = overlap / sum(pred_counts.values()) if pred_counts else 0.0
    recall = overlap / sum(ref_counts.values()) if ref_counts else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _rouge_l(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    """Compute ROUGE-L F1 based on LCS."""
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    prec = lcs / m
    rec = lcs / n
    return (2 * prec * rec) / (prec + rec) if prec + rec else 0.0


def evaluate_prediction(prediction: str, references: List[str]) -> Dict[str, float]:
    """Compute multiple string similarity metrics between prediction and references."""
    if not references:
        return {
            "exact_match": 0.0,
            "accuracy": 0.0,
            "cer": 1.0,
            "wer": 1.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "rouge_l": 0.0,
            "anls": 0.0,
            "relaxed_accuracy": 0.0,
        }
    pred_norm = normalize(prediction)
    pred_loose = normalize_loose(prediction)
    pred_chars = list(pred_norm)
    pred_tokens = pred_norm.split()

    best = None
    for ref in references:
        ref_norm = normalize(ref)
        ref_loose = normalize_loose(ref)
        ref_chars = list(ref_norm)
        ref_tokens = ref_norm.split()
        char_dist = _levenshtein(pred_chars, ref_chars)
        word_dist = _levenshtein(pred_tokens, ref_tokens)
        cer = char_dist / max(len(ref_chars), 1)
        wer = word_dist / max(len(ref_tokens), 1)
        precision, recall, f1 = _token_fscore(pred_tokens, ref_tokens)
        rouge_l = _rouge_l(pred_tokens, ref_tokens)
        exact = 1.0 if pred_norm == ref_norm else 0.0
        # Treat as correct if the gold answer is contained in the prediction (or vice versa), punctuation-insensitive
        contains = 1.0 if (ref_loose in pred_loose or pred_loose in ref_loose) else 0.0
        # ANLS (Average Normalized Levenshtein Similarity) using loose normalization
        pred_chars_loose = list(pred_loose)
        ref_chars_loose = list(ref_loose)
        char_dist_loose = _levenshtein(pred_chars_loose, ref_chars_loose)
        max_len = max(len(pred_chars_loose), len(ref_chars_loose), 1)
        anls = 1.0 - (char_dist_loose / max_len)
        # If the normalized strings contain each other, treat as a perfect hit.
        if contains >= 1.0:
            anls = 1.0
        elif anls < 0.5:
            anls = 0.0
        relaxed_acc = 1.0 if anls > 0 else 0.0
        metrics = {
            "exact_match": exact,
            "accuracy": contains,  # relaxed: counts containment as correct
            "cer": cer,
            "wer": wer,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "rouge_l": rouge_l,
            "anls": anls,
            "relaxed_accuracy": relaxed_acc,
        }
        if best is None or metrics["f1"] > best["f1"]:
            best = metrics
    assert best is not None
    return best


# ---------------------------------------------------------------------------
# TextVQA helpers
# ---------------------------------------------------------------------------

def load_textvqa(path: Path) -> List[Dict[str, object]]:
    """Load a TextVQA json file and return the list of samples."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "data" in payload:
        return list(payload["data"])
    if isinstance(payload, list):
        return payload
    raise ValueError("Unsupported TextVQA file format.")


def resolve_image_path(root: Path, image_id: str, template: str) -> Path:
    """Build the image path from template."""
    relative = template.format(image_id=image_id)
    return (root / relative).resolve()


def iter_textvqa_samples(
    dataset_path: Path,
    images_root: Path,
    image_template: str,
) -> Tuple[Iterator[Dict[str, Any]], Optional[int]]:
    """Yield samples from a locally stored TextVQA JSON plus image directory."""
    samples = load_textvqa(dataset_path)
    total = len(samples)

    def generator() -> Iterator[Dict[str, Any]]:
        for sample in samples:
            question = sample.get("question")
            image_id = sample.get("image_id")
            answers = sample.get("answers", [])
            if not question or not image_id:
                continue
            image_path = resolve_image_path(images_root, image_id, image_template)
            yield {
                "question": question,
                "image_id": image_id,
                "answers": answers,
                "image_source": str(image_path),
            }

    return generator(), total


# ---------------------------------------------------------------------------
# TrainingDataPro helpers
# ---------------------------------------------------------------------------

def extract_trainingdatapro_answers(sample: Dict[str, Any]) -> List[str]:
    """Heuristically collect ground-truth text from TrainingDataPro samples."""
    answers: List[str] = []
    text_field = sample.get("text")
    if isinstance(text_field, str):
        answers.append(text_field)

    for key in ("labels", "texts", "annotations"):
        if key not in sample:
            continue
        value = sample[key]
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, str):
                    answers.append(entry)
                elif isinstance(entry, dict):
                    text = entry.get("text") or entry.get("label")
                    if text:
                        answers.append(str(text))

    seen = set()
    unique: List[str] = []
    for ans in answers:
        key = ans.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(ans)
    return unique


def _to_pil_image(image: Any) -> Any:
    """Convert dataset image payloads into PIL.Image when possible."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - optional dependency
        return image

    if isinstance(image, Image.Image):
        return image
    if isinstance(image, dict):
        if "bytes" in image:
            image = image["bytes"]
        elif "path" in image:
            return Image.open(image["path"])
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    if isinstance(image, str):
        try:
            data = base64.b64decode(image)
            return Image.open(io.BytesIO(data))
        except Exception:  # pragma: no cover - fallback to raw string
            return image
    return image


def iter_trainingdatapro_samples(
    split: str,
    question_template: str,
) -> Tuple[Iterator[Dict[str, Any]], Optional[int]]:
    """Yield samples from the TrainingDataPro OCR text detection dataset."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("The 'datasets' package is required for TrainingDataPro experiments.") from exc

    try:
        dataset = load_dataset("TrainingDataPro/ocr-text-detection-in-the-documents", split=split)
    except Exception as exc:  # pragma: no cover - remote dataset availability
        raise RuntimeError(
            "Failed to load TrainingDataPro dataset. Verify access to "
            "'TrainingDataPro/ocr-text-detection-in-the-documents'."
        ) from exc

    total = len(dataset)

    def generator() -> Iterator[Dict[str, Any]]:
        for sample in dataset:
            image = sample.get("image")
            if image is None:
                continue
            image = _to_pil_image(image)
            answers = extract_trainingdatapro_answers(sample)
            fmt_mapping = {k: v for k, v in sample.items() if isinstance(k, str)}
            try:
                question = question_template.format_map(fmt_mapping)
            except Exception:
                question = question_template
            yield {
                "question": question,
                "image_id": sample.get("image_id"),
                "answers": answers,
                "image_source": image,
            }

    return generator(), total


def _extract_textocr_answers(sample: Dict[str, Any]) -> List[str]:
    """Gather reference strings from TextOCR-like samples."""
    answers: List[str] = []
    candidates = [
        sample.get("answers"),
        sample.get("answer"),
        sample.get("text"),
        sample.get("words"),
    ]
    for cand in candidates:
        if isinstance(cand, list):
            for v in cand:
                if isinstance(v, str) and v.strip():
                    answers.append(v)
                elif isinstance(v, dict):
                    for val in v.values():
                        if isinstance(val, str) and val.strip():
                            answers.append(val)
        elif isinstance(cand, str) and cand.strip():
            answers.append(cand)
        if answers:
            break
    return answers


def iter_textocr_samples(
    split: str,
    mode: str,
    question_template: str,
) -> Tuple[Iterator[Dict[str, Any]], Optional[int]]:
    """Yield samples from TextOCR in transcription or VQA mode."""
    try:
        dataset = load_dataset_by_id("textocr", split=split)
    except Exception as exc:
        raise RuntimeError("Failed to load TextOCR. Ensure it is cached or accessible.") from exc

    total = len(dataset)

    def generator() -> Iterator[Dict[str, Any]]:
        for sample in dataset:
            image = sample.get("image")
            if image is None:
                continue
            image = _to_pil_image(image)
            answers = _extract_textocr_answers(sample)
            if mode == "transcribe":
                question = question_template
            else:
                question = sample.get("question", question_template)
            yield {
                "question": question,
                "image_id": sample.get("image_id") or sample.get("id"),
                "answers": answers,
                "image_source": image,
            }

    return generator(), total


def iter_stvqa_samples(split: str) -> Tuple[Iterator[Dict[str, Any]], Optional[int]]:
    """Yield samples from the HF ST-VQA dataset (vikhyatk/st-vqa)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("The 'datasets' package is required for ST-VQA experiments.") from exc

    try:
        dataset = load_dataset("lmms-lab/ST-VQA", split=split)
    except Exception as exc:  # pragma: no cover - remote dataset availability
        raise RuntimeError("Failed to load lmms-lab/ST-VQA. Check network/access.") from exc

    total = len(dataset)

    def generator() -> Iterator[Dict[str, Any]]:
        for sample in dataset:
            image = sample.get("image")
            if image is None:
                continue
            image = _to_pil_image(image)
            question_raw = sample.get("question", "")
            if isinstance(question_raw, str):
                question = question_raw
            elif isinstance(question_raw, (list, tuple)):
                question = " ".join(str(x) for x in question_raw)
            else:
                question = str(question_raw)
            answers_field = sample.get("answers") or sample.get("answer") or []
            answers: List[str] = []
            if isinstance(answers_field, list):
                for a in answers_field:
                    if isinstance(a, dict) and "answer" in a and isinstance(a["answer"], str):
                        answers.append(a["answer"])
                    elif isinstance(a, str):
                        answers.append(a)
            elif isinstance(answers_field, str):
                answers.append(answers_field)
            image_id = sample.get("image_id") or sample.get("question_id")
            yield {
                "question": question,
                "image_id": image_id,
                "answers": answers,
                "image_source": image,
            }

    return generator(), total


def iter_registry_vqa_samples(
    dataset_id: str,
    split: str,
) -> Tuple[Iterator[Dict[str, Any]], Optional[int]]:
    """Generic VQA iterator for registry-backed HF datasets."""
    dataset = load_dataset_by_id(dataset_id, split=split)
    total = len(dataset)

    def generator() -> Iterator[Dict[str, Any]]:
        for sample in dataset:
            image = sample.get("image")
            if image is None and isinstance(sample.get("images"), list) and sample["images"]:
                # Fallback: use the first page/image if provided as a list (e.g., multi-page docs)
                image = sample["images"][0]
            if image is None:
                continue
            image = _to_pil_image(image)
            question = sample.get("question", "")
            answers_field = sample.get("answers") or sample.get("answer") or []
            answers: List[str] = []
            if isinstance(answers_field, list):
                for a in answers_field:
                    if isinstance(a, dict) and "answer" in a and isinstance(a["answer"], str):
                        answers.append(a["answer"])
                    elif isinstance(a, str):
                        answers.append(a)
            elif isinstance(answers_field, str):
                answers.append(answers_field)
            image_id = sample.get("image_id") or sample.get("question_id")
            yield {
                "question": question,
                "image_id": image_id,
                "answers": answers,
                "image_source": image,
            }

    return generator(), total


# ---------------------------------------------------------------------------
# CC-OCR helpers
# ---------------------------------------------------------------------------

def _collect_strings(value: Any, bucket: List[str], max_items: int = 200) -> None:
    """Recursively collect string snippets from nested structures."""
    if len(bucket) >= max_items:
        return
    if isinstance(value, str):
        if value.strip():
            bucket.append(value)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            if len(bucket) >= max_items:
                break
            _collect_strings(item, bucket, max_items=max_items)
    elif isinstance(value, dict):
        for item in value.values():
            if len(bucket) >= max_items:
                break
            _collect_strings(item, bucket, max_items=max_items)


def extract_ccocr_answers(sample: Dict[str, Any]) -> List[str]:
    """Gather available transcription text from CC-OCR samples."""
    bucket: List[str] = []
    answer_field = sample.get("answer")
    if isinstance(answer_field, str) and answer_field.strip():
        bucket.append(answer_field)
    elif isinstance(answer_field, list):
        for item in answer_field:
            if isinstance(item, str) and item.strip():
                bucket.append(item)

    for key in ("text", "texts", "gt_text", "ground_truth", "words", "annotations"):
        if key in sample:
            _collect_strings(sample[key], bucket)
    if not bucket:
        # Fallback: scan entire sample but limit recursion depth via _collect_strings cap.
        _collect_strings(sample, bucket)

    seen: set[str] = set()
    unique: List[str] = []
    for text in bucket:
        norm = text.strip().lower()
        if norm and norm not in seen:
            seen.add(norm)
            unique.append(text)
    return unique


def _sanitize_wandb_tag(tag: str) -> str:
    """Convert an arbitrary label into a W&B-safe tag token."""
    safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", tag)
    safe = safe.strip("_")
    return safe or "untitled"


@dataclass
class ExperimentSpec:
    """Defines a routing configuration sweep experiment."""

    name: str
    overrides: Dict[str, Any]
    tags: List[str]
    description: str = ""


def _format_float(value: float) -> str:
    """Format floats consistently for experiment names/tags."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def build_experiment_specs(args: argparse.Namespace, cli_args: argparse.Namespace) -> List[ExperimentSpec]:
    """Construct the list of experiments to execute based on CLI options."""
    specs: List[ExperimentSpec] = []

    include_baseline = getattr(args, "include_baseline", False)
    include_always = getattr(args, "include_always", False)
    include_margin_ablation = getattr(args, "include_margin_ablation", False)
    include_verify_combo = getattr(args, "include_verify_combo", False)

    if args.experiment_set == "paper":
        include_baseline = True
        include_always = True
        include_margin_ablation = True
        entropy_grid = args.entropy_th_grid or [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        history_grid = args.history_k_grid or [0, 3, 5]
        temp_grid = args.temperature_scale_grid or [0.6, 0.7, 0.8]
        margin_grid = args.margin_th_grid or [0.5, 1.0, 1.5]
    elif args.experiment_set == "quick_verify":
        include_baseline = True
        include_always = include_always
        include_margin_ablation = False
        include_verify_combo = True
        entropy_grid = []
        history_grid = []
        temp_grid = []
        margin_grid = []
    else:
        entropy_grid = args.entropy_th_grid or []
        history_grid = args.history_k_grid or []
        temp_grid = args.temperature_scale_grid or []
        margin_grid = args.margin_th_grid or []

    # Baseline and Always-OCR configurations
    if include_baseline:
        specs.append(
            ExperimentSpec(
                name="baseline_no_ocr",
                overrides={"no_routing": True, "always_ocr": False},
                tags=["baseline", "no_ocr"],
                description="First-pass only; OCR disabled.",
            )
        )

    if include_always:
        specs.append(
            ExperimentSpec(
                name="always_ocr",
                overrides={"no_routing": False, "always_ocr": True},
                tags=["always_ocr"],
                description="Always invoke OCR-assisted second pass.",
            )
        )

    # Default entropy-only routing (ours)
    specs.append(
        ExperimentSpec(
            name="entropy_routing_default",
            overrides={
                "no_routing": False,
                "always_ocr": False,
                "entropy_th": cli_args.entropy_th,
                "history_k": cli_args.history_k,
                "margin_th": float("-inf"),
            },
            tags=[
                "entropy",
                f"entropy-{_format_float(cli_args.entropy_th)}",
                f"history-{cli_args.history_k}",
            ],
            description="Entropy-only routing with CLI defaults.",
        )
    )

    if include_verify_combo or args.experiment_set == "quick_verify":
        specs.append(
            ExperimentSpec(
                name="verify_combo",
                overrides={
                    "no_routing": False,
                    "always_ocr": False,
                    "entropy_th": cli_args.entropy_th,
                    "history_k": 3,
                    "margin_th": 0.5,
                    "temperature_scale": 0.7,
                    "ocr_prompt_mode": "verify",
                },
                tags=[
                    "quick_verify",
                    "verify_prompt",
                    "temp-scale-0.7",
                    "margin-0.5",
                    "history-3",
                ],
                description="Recommended verify-mode combo: history=3, temp_scale=0.7, margin_th=0.5, prompt=verify.",
            )
        )

    # Entropy threshold/history sweeps
    for entropy_th in entropy_grid:
        for history_k in history_grid or [cli_args.history_k]:
            specs.append(
                ExperimentSpec(
                    name=f"entropy_e{_format_float(entropy_th)}_h{history_k}",
                    overrides={
                        "no_routing": False,
                        "always_ocr": False,
                        "entropy_th": entropy_th,
                        "history_k": history_k,
                        "margin_th": float("-inf"),
                    },
                    tags=[
                        "entropy",
                        f"entropy-{_format_float(entropy_th)}",
                        f"history-{history_k}",
                    ],
                    description="Entropy threshold/history sweep.",
                )
            )

    # Temperature scaling calibration
    for temp_scale in temp_grid:
        specs.append(
            ExperimentSpec(
                name=f"entropy_calibrated_t{_format_float(temp_scale)}",
                overrides={
                    "no_routing": False,
                    "always_ocr": False,
                    "entropy_th": cli_args.entropy_th,
                    "history_k": cli_args.history_k,
                    "margin_th": float("-inf"),
                    "temperature_scale": temp_scale,
                },
                tags=[
                    "entropy",
                    "calibration",
                    f"temp-scale-{_format_float(temp_scale)}",
                ],
                description="Entropy-only routing with temperature scaling calibration.",
            )
        )

    # Margin ablation experiments
    if include_margin_ablation:
        for margin_th in margin_grid or [0.5, 1.0, 1.5]:
            specs.append(
                ExperimentSpec(
                    name=f"entropy_margin_{_format_float(margin_th)}",
                    overrides={
                        "no_routing": False,
                        "always_ocr": False,
                        "entropy_th": cli_args.entropy_th,
                        "history_k": cli_args.history_k,
                        "margin_th": margin_th,
                    },
                    tags=[
                        "entropy_margin",
                        f"margin-{_format_float(margin_th)}",
                    ],
                    description="Margin ablation: reintroduce margin threshold.",
                )
            )
            specs.append(
                ExperimentSpec(
                    name=f"margin_only_{_format_float(margin_th)}",
                    overrides={
                        "no_routing": False,
                        "always_ocr": False,
                        "entropy_th": float("inf"),
                        "history_k": 0,
                        "margin_th": margin_th,
                    },
                    tags=[
                        "margin_only",
                        f"margin-{_format_float(margin_th)}",
                    ],
                    description="Margin-only ablation with entropy disabled.",
                )
            )

    # If no sweeps requested and baseline/alway not selected, fall back to single config
    if not specs:
        specs.append(
            ExperimentSpec(
                name="default",
                overrides={},
                tags=[],
                description="Single configuration (no sweep).",
            )
        )

    # Ensure unique names
    unique: Dict[str, ExperimentSpec] = {}
    for spec in specs:
        unique[spec.name] = spec
    return list(unique.values())


def derive_out_path(base_path: Path, experiment_name: str, multi: bool) -> Path:
    """Derive a per-experiment output path."""
    if not multi:
        return base_path
    stem = base_path.stem
    suffix = base_path.suffix or ".jsonl"
    return base_path.with_name(f"{stem}__{experiment_name}{suffix}")


def compute_calibration_stats(
    confidences: List[float],
    correctness: List[float],
    num_bins: int = 10,
) -> Tuple[float, float, List[Dict[str, float]]]:
    """Compute Expected Calibration Error (ECE) and Brier score."""
    total = len(confidences)
    if total == 0:
        return 0.0, 0.0, []

    bins = [0] * num_bins
    conf_sum = [0.0] * num_bins
    acc_sum = [0.0] * num_bins

    for conf, corr in zip(confidences, correctness):
        clipped = min(max(conf, 0.0), 1.0)
        idx = min(int(clipped * num_bins), num_bins - 1)
        bins[idx] += 1
        conf_sum[idx] += clipped
        acc_sum[idx] += corr

    ece = 0.0
    bin_stats: List[Dict[str, float]] = []
    for idx, count in enumerate(bins):
        if count == 0:
            bin_stats.append({"bin": idx, "count": 0, "avg_confidence": 0.0, "avg_accuracy": 0.0})
            continue
        avg_conf = conf_sum[idx] / count
        avg_acc = acc_sum[idx] / count
        weight = count / total
        ece += weight * abs(avg_conf - avg_acc)
        bin_stats.append(
            {
                "bin": idx,
                "count": count,
                "avg_confidence": avg_conf,
                "avg_accuracy": avg_acc,
            }
        )

    brier = sum((min(max(c, 0.0), 1.0) - corr) ** 2 for c, corr in zip(confidences, correctness)) / total
    return ece, brier, bin_stats


def iter_ccocr_samples(
    split: str,
    config: str,
    question_template: str,
) -> Tuple[Iterator[Dict[str, Any]], Optional[int]]:
    """Yield samples from the wulipc/CC-OCR dataset on Hugging Face."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("The 'datasets' package is required for CC-OCR experiments.") from exc

    try:
        dataset = load_dataset("wulipc/CC-OCR", config, split=split)
    except Exception as exc:  # pragma: no cover - remote dataset availability
        raise RuntimeError(
            "Failed to load wulipc/CC-OCR dataset. Verify access to 'wulipc/CC-OCR'."
        ) from exc

    total = len(dataset)

    def generator() -> Iterator[Dict[str, Any]]:
        for sample in dataset:
            image = sample.get("image") or sample.get("img")
            if image is None:
                image_path = sample.get("image_path") or sample.get("img_path")
                if image_path is None:
                    continue
                image = image_path
            image = _to_pil_image(image)

            answers = extract_ccocr_answers(sample)
            fmt_mapping = {k: v for k, v in sample.items() if isinstance(k, str)}
            raw_question = sample.get("question")
            if isinstance(raw_question, str) and raw_question.strip():
                question = raw_question
            else:
                try:
                    question = question_template.format_map(fmt_mapping)
                except Exception:
                    question = question_template

            yield {
                "question": question,
                "image_id": sample.get("image_id"),
                "answers": answers,
                "image_source": image,
            }

    return generator(), total


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def describe_image_source(image: Any, image_id: Optional[Any]) -> str:
    """Return a human-readable string for logging/serialization."""
    if isinstance(image, (str, Path)):
        return str(image)
    filename = getattr(image, "filename", None)
    if filename:
        return str(filename)
    return f"in_memory:{image_id}"


def iter_samples(args: argparse.Namespace) -> Tuple[Iterator[Dict[str, Any]], Optional[int]]:
    """Dispatch to the selected dataset iterator."""
    dataset = args.dataset.lower()
    if dataset == "textvqa":
        if not args.questions or not args.images_root:
            raise ValueError("--questions and --images-root must be provided for dataset=textvqa.")
        dataset_path = Path(args.questions)
        images_root = Path(args.images_root)
        return iter_textvqa_samples(dataset_path, images_root, args.image_template)

    if dataset == "trainingdatapro":
        return iter_trainingdatapro_samples(args.split, args.question_template)

    if dataset == "cc-ocr":
        return iter_ccocr_samples(args.split, args.config, args.question_template)

    if dataset == "st-vqa":
        return iter_stvqa_samples(args.split)

    if dataset in {"docvqa", "mp_docvqa", "infovqa", "chartqa", "ocrbench_v2"}:
        return iter_registry_vqa_samples(dataset, args.split)

    if dataset == "textocr_transcribe":
        return iter_textocr_samples(args.split, mode="transcribe", question_template=args.question_template)

    if dataset == "textocr_vqa":
        return iter_textocr_samples(args.split, mode="vqa", question_template=args.question_template)

    raise ValueError(f"Unsupported dataset: {args.dataset}")


def run_experiment(
    experiment: ExperimentSpec,
    base_pipeline: Pipeline,
    samples: List[Dict[str, Any]],
    cli_args: argparse.Namespace,
    base_out_path: Path,
    multi_output: bool,
    dataset_name: str,
    print_comparisons: bool,
) -> Dict[str, Any]:
    """Execute a routing experiment and collect per-sample / aggregate metrics."""
    out_path = derive_out_path(base_out_path, experiment.name, multi_output)
    ensure_directory(out_path.parent)
    if out_path.exists():
        out_path.unlink()

    base_tags = list(getattr(base_pipeline.routing_config, "wandb_tags", ()) or [])
    tag_set = { _sanitize_wandb_tag(tag) for tag in base_tags if tag }
    tag_set.add(_sanitize_wandb_tag(f"experiment-{experiment.name}"))
    for extra in experiment.tags:
        tag_set.add(_sanitize_wandb_tag(extra))
    tag_set.add(_sanitize_wandb_tag(f"dataset-{dataset_name}"))
    model_id = getattr(cli_args, "model_id", None)
    if model_id:
        tag_set.add(_sanitize_wandb_tag(f"model-{model_id}"))

    overrides = dict(experiment.overrides)
    overrides.setdefault("no_routing", False)
    overrides.setdefault("always_ocr", False)
    if overrides.get("no_routing"):
        overrides["always_ocr"] = False
    overrides["wandb_tags"] = tuple(sorted(tag_set))

    config = replace(base_pipeline.routing_config, **overrides)
    pipeline = Pipeline(model=base_pipeline.model, processor=base_pipeline.processor, routing_config=config)

    metric_keys = [
        "exact_match",
        "accuracy",
        "cer",
        "wer",
        "precision",
        "recall",
        "f1",
        "rouge_l",
        "anls",
        "relaxed_accuracy",
    ]
    totals = {k: 0.0 for k in metric_keys}
    ocr_count = 0
    latency_ms_total = 0.0
    latency_vs_quality: List[Dict[str, Any]] = []
    triggered_entropies: List[float] = []
    non_triggered_entropies: List[float] = []
    confidences: List[float] = []
    correctness: List[float] = []

    total_count = 0
    for sample in tqdm(samples, desc=f"Experiment {experiment.name}", leave=False):
        question = sample["question"]
        answers = sample.get("answers", [])
        image_source = sample["image_source"]
        image_id = sample.get("image_id")
        # Normalize question to string to avoid str/list concat issues downstream.
        if not isinstance(question, str):
            if isinstance(question, (list, tuple)):
                question = " ".join(str(x) for x in question)
            else:
                question = str(question)
        try:
            routed_result = pipeline.run(image=image_source, question=question)
        except Exception as exc:  # pragma: no cover - runtime variability
            import traceback
            print(f"Failed on {image_id}: {exc}")
            traceback.print_exc()
            continue

        image_repr = describe_image_source(image_source, image_id)
        routed_payload = result_to_dict(image_repr, routed_result, cli_args.model_id, cli_args.ocr_engine)
        routed_payload["experiment"] = experiment.name

        routed_metrics = evaluate_prediction(routed_payload.get("answer", ""), answers)
        for key in metric_keys:
            totals[key] += routed_metrics[key]

        if print_comparisons:
            answer_display = "; ".join(answers) if answers else "<no reference>"
            verdict = "CORRECT" if routed_metrics.get("accuracy", 0.0) >= 0.5 else "WRONG"
            print(f"[{experiment.name}] image={image_id or 'n/a'} used_ocr={routed_result.used_ocr}")
            print(f"  verdict   : {verdict}")
            print(f"  question   : {question}")
            print(f"  prediction : {routed_payload.get('answer', '')}")
            print(f"  references : {answer_display}")
            print(
                "  metrics    : CER={:.3f} WER={:.3f} EM={:.3f} ANLS={:.3f}".format(
                    routed_metrics.get("cer", 0.0),
                    routed_metrics.get("wer", 0.0),
                    routed_metrics.get("exact_match", 0.0),
                    routed_metrics.get("anls", 0.0),
                )
            )
            print("-" * 80)

        append_jsonl(out_path, {
            "question": question,
            "image_id": image_id,
            "answers": answers,
            "experiment": experiment.name,
            "routed": routed_payload,
            "routed_metrics": routed_metrics,
        })

        total_count += 1
        if routed_result.used_ocr:
            ocr_count += 1
        latency_ms_total += routed_result.total_latency_ms
        latency_vs_quality.append(
            {
                "latency_s": routed_result.total_latency_ms / 1000.0,
                "cer": routed_metrics.get("cer", 0.0),
                "wer": routed_metrics.get("wer", 0.0),
                "used_ocr": routed_result.used_ocr,
                "image_id": image_id,
            }
        )
        mean_entropy = routed_result.first.metrics.mean_entropy
        if routed_result.triggered:
            triggered_entropies.append(mean_entropy)
        else:
            non_triggered_entropies.append(mean_entropy)

        confidences.append(routed_result.final_metrics.sequence_confidence)
        correctness.append(routed_metrics.get("exact_match", 0.0))

    avg_metrics = {k: (totals[k] / total_count) if total_count else 0.0 for k in metric_keys}
    ocr_ratio = (ocr_count / total_count) if total_count else 0.0
    avg_latency_s = (latency_ms_total / 1000.0 / total_count) if total_count else 0.0
    ece, brier, calibration_bins = compute_calibration_stats(confidences, correctness)
    accuracy = avg_metrics.get("accuracy", avg_metrics.get("exact_match", 0.0))
    risk = 1.0 - accuracy

    summary = {
        "experiment": experiment.name,
        "description": experiment.description,
        "total": total_count,
        "avg_metrics": avg_metrics,
        "ocr_call_ratio": ocr_ratio,
        "avg_latency_s": avg_latency_s,
        "ece": ece,
        "brier": brier,
        "coverage": ocr_ratio,
        "risk": risk,
        "latency_vs_quality": latency_vs_quality,
        "entropy_trigger_distribution": {
            "triggered": triggered_entropies,
            "not_triggered": non_triggered_entropies,
        },
        "calibration_bins": calibration_bins,
        "output_path": str(out_path),
    }

    if wandb is not None and wandb.run is not None:
        log_payload: Dict[str, Any] = {
            f"experiment/{experiment.name}/accuracy": accuracy,
            f"experiment/{experiment.name}/cer": avg_metrics.get("cer", 0.0),
            f"experiment/{experiment.name}/wer": avg_metrics.get("wer", 0.0),
            f"experiment/{experiment.name}/anls": avg_metrics.get("anls", 0.0),
            f"experiment/{experiment.name}/relaxed_accuracy": avg_metrics.get("relaxed_accuracy", 0.0),
            f"experiment/{experiment.name}/ocr_call_ratio": ocr_ratio,
            f"experiment/{experiment.name}/avg_latency_s": avg_latency_s,
            f"experiment/{experiment.name}/ece": ece,
            f"experiment/{experiment.name}/brier": brier,
            f"experiment/{experiment.name}/coverage": ocr_ratio,
            f"experiment/{experiment.name}/risk": risk,
        }
        wandb.log(log_payload)

        if triggered_entropies:
            wandb.log(
                {
                    f"experiment/{experiment.name}/entropy_triggered_hist": wandb.Histogram(triggered_entropies)
                }
            )
        if non_triggered_entropies:
            wandb.log(
                {
                    f"experiment/{experiment.name}/entropy_not_triggered_hist": wandb.Histogram(
                        non_triggered_entropies
                    )
                }
            )
        if latency_vs_quality:
            max_rows = min(len(latency_vs_quality), 512)
            table = wandb.Table(
                columns=["latency_s", "cer", "wer", "used_ocr"],
                rows=[
                    [
                        latency_vs_quality[idx]["latency_s"],
                        latency_vs_quality[idx]["cer"],
                        latency_vs_quality[idx]["wer"],
                        int(latency_vs_quality[idx]["used_ocr"]),
                    ]
                    for idx in range(max_rows)
                ],
            )
            wandb.log({f"experiment/{experiment.name}/latency_vs_quality": table})

    return summary


def build_argparser() -> argparse.ArgumentParser:
    """Command line arguments."""
    parser = argparse.ArgumentParser(description="Run OCR routing experiments on benchmark datasets.")
    parser.add_argument(
        "--dataset",
        default="textvqa",
        choices=[
            "textvqa",
            "trainingdatapro",
            "cc-ocr",
            "st-vqa",
            "docvqa",
            "mp_docvqa",
            "infovqa",
            "chartqa",
            "textocr_transcribe",
            "textocr_vqa",
            "ocrbench_v2",
        ],
        help="Dataset to evaluate.",
    )
    parser.add_argument("--questions", help="Path to TextVQA JSON (required for dataset=textvqa).")
    parser.add_argument("--images-root", help="Root directory containing TextVQA images.")
    parser.add_argument(
        "--image-template",
        default="{image_id}.jpg",
        help="Format string to convert image_id to path relative to images-root (TextVQA).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split used for Hugging Face hosted datasets.",
    )
    parser.add_argument(
        "--config",
        default="multi_scene_ocr",
        help="Dataset configuration (e.g., CC-OCR configs: multi_scene_ocr, multi_lan_ocr, kie, doc_parsing).",
    )
    parser.add_argument(
        "--question-template",
        default="Please transcribe all visible text in this document image.",
        help="Prompt template for datasets without native questions.",
    )
    parser.add_argument(
        "--out",
        default="outputs/ocr_routing_results.jsonl",
        help="Where to store detailed JSONL predictions.",
    )
    parser.add_argument("--limit", type=int, help="Optional number of samples to evaluate.")
    parser.add_argument(
        "--experiment-set",
        type=str,
        choices=["none", "paper", "quick_verify"],
        default="none",
        help="Preset sweep configuration to run (e.g., 'paper' full ablation; 'quick_verify' baseline + verify-mode combo).",
    )
    parser.add_argument(
        "--entropy-th-grid",
        type=float,
        nargs="+",
        help="Override entropy threshold sweep values.",
    )
    parser.add_argument(
        "--history-k-grid",
        type=int,
        nargs="+",
        help="Override history length sweep values.",
    )
    parser.add_argument(
        "--temperature-scale-grid",
        type=float,
        nargs="+",
        help="Temperature scale factors for calibration sweeps.",
    )
    parser.add_argument(
        "--margin-th-grid",
        type=float,
        nargs="+",
        help="Margin threshold sweep values for ablation.",
    )
    parser.add_argument("--include-baseline", action="store_true", help="Explicitly include baseline (no OCR) run.")
    parser.add_argument("--include-always", action="store_true", help="Explicitly include always-OCR run.")
    parser.add_argument(
        "--include-margin-ablation",
        action="store_true",
        help="Include margin threshold ablation even if experiment-set is not 'paper'.",
    )
    parser.add_argument(
        "--include-verify-combo",
        action="store_true",
        help="Include the recommended verify-mode combo (history_k=3, temp_scale=0.7, margin_th=0.5, ocr_prompt_mode=verify).",
    )
    parser.add_argument(
        "--print-comparisons",
        action="store_true",
        help="Print reference answers and predictions for every sample in each experiment.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for model/tokenizer init.")
    return parser


def main() -> None:
    parser = build_argparser()
    args, remaining = parser.parse_known_args()

    cli_args = parse_args(remaining)
    setup_logging(cli_args.log_level)
    setattr(cli_args, "dataset", args.dataset)
    if not hasattr(cli_args, "seed") or cli_args.seed is None:
        cli_args.seed = args.seed
    if getattr(cli_args, "wandb_tags", None):
        sanitized_tags: List[str] = []
        seen: set[str] = set()
        for tag in cli_args.wandb_tags:
            clean = _sanitize_wandb_tag(tag)
            if clean and clean not in seen:
                seen.add(clean)
                sanitized_tags.append(clean)
        cli_args.wandb_tags = sanitized_tags
    else:
        cli_args.wandb_tags = []

    out_path = Path(args.out)

    try:
        sample_iter, dataset_total = iter_samples(args)
    except ValueError as exc:
        parser.error(str(exc))
        return

    if args.limit is not None:
        samples = list(itertools.islice(sample_iter, args.limit))
    else:
        samples = list(sample_iter)

    evaluated_total = len(samples)
    if evaluated_total == 0:
        print(json.dumps({"dataset": args.dataset, "split": args.split, "samples_evaluated": 0}, indent=2))
        return

    pipeline = build_pipeline(cli_args)
    experiments = build_experiment_specs(args, cli_args)
    multi_output = len(experiments) > 1

    summaries: List[Dict[str, Any]] = []
    for spec in experiments:
        summary = run_experiment(
            experiment=spec,
            base_pipeline=pipeline,
            samples=samples,
            cli_args=cli_args,
            base_out_path=out_path,
            multi_output=multi_output,
            dataset_name=args.dataset,
            print_comparisons=args.print_comparisons,
        )
        summaries.append(summary)

    report = {
        "dataset": args.dataset,
        "split": args.split,
        "samples_available": dataset_total if dataset_total is not None else evaluated_total,
        "samples_evaluated": evaluated_total,
        "limit": args.limit,
        "output_base": str(out_path),
        "experiments": summaries,
        "metrics_table": [
            {
                "experiment": summary["experiment"],
                "accuracy": summary["avg_metrics"].get("accuracy", summary["avg_metrics"].get("exact_match", 0.0)),
                "cer": summary["avg_metrics"].get("cer", 0.0),
                "wer": summary["avg_metrics"].get("wer", 0.0),
                "anls": summary["avg_metrics"].get("anls", 0.0),
                "relaxed_accuracy": summary["avg_metrics"].get("relaxed_accuracy", 0.0),
                "coverage": summary["ocr_call_ratio"],
                "latency_s": summary["avg_latency_s"],
                "ece": summary["ece"],
                "brier": summary["brier"],
                "risk": summary["risk"],
            }
            for summary in summaries
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
