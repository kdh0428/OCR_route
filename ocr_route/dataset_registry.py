"""Dataset registry for OCR/VLM benchmarks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

try:  # optional dependency
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover - handled at call time
    load_dataset = None


@dataclass(frozen=True)
class DatasetEntry:
    id: str
    category: str
    main_metrics: List[str]
    description: str
    default_metric: Optional[str] = None
    huggingface_id: Optional[str] = None
    huggingface_id_train: Optional[str] = None
    huggingface_id_test: Optional[str] = None
    huggingface_config: Optional[str] = None


DATASET_REGISTRY: Dict[str, DatasetEntry] = {
    # Scene text VQA benchmarks
    "textvqa": DatasetEntry(
        id="textvqa",
        huggingface_id="facebook/textvqa",
        category="scene_text_vqa",
        main_metrics=["accuracy"],
        default_metric="accuracy",
        description="Scene text visual question answering benchmark; requires reading text in natural images.",
    ),
    "stvqa": DatasetEntry(
        id="stvqa",
        huggingface_id="lmms-lab/ST-VQA",
        category="scene_text_vqa",
        main_metrics=["anls", "accuracy"],
        default_metric="anls",
        description="ST-VQA: scene text VQA benchmark with ANLS as the main metric.",
    ),
    # Document / infographic VQA benchmarks
    "docvqa": DatasetEntry(
        id="docvqa",
        huggingface_id="lmms-lab/DocVQA",
        category="document_vqa",
        main_metrics=["anls"],
        default_metric="anls",
        huggingface_config="DocVQA",
        description="Single-page document VQA benchmark; measures document understanding and OCR via ANLS.",
    ),
    "mp_docvqa": DatasetEntry(
        id="mp_docvqa",
        huggingface_id="rubentito/mp-docvqa",
        category="document_vqa_multi_page",
        main_metrics=["anls"],
        default_metric="anls",
        description="Multi-page DocVQA; requires cross-page document understanding and OCR.",
    ),
    "infovqa": DatasetEntry(
        id="infovqa",
        huggingface_id="Ahren09/InfoVQA",
        category="infographic_vqa",
        main_metrics=["anls", "accuracy"],
        default_metric="anls",
        description="Infographic / text-rich image VQA; combines text, graphics, and numerical reasoning.",
    ),
    # Chart QA
    "chartqa": DatasetEntry(
        id="chartqa",
        huggingface_id="lmms-lab/ChartQA",
        category="chart_qa",
        main_metrics=["relaxed_accuracy", "accuracy"],
        default_metric="relaxed_accuracy",
        description="ChartQA: QA over bar/line/pie charts; evaluates reading chart text and numeric reasoning.",
    ),
    # TextOCR (transcription + VQA modes)
    "textocr_transcribe": DatasetEntry(
        id="textocr_transcribe",
        huggingface_id="textocr",
        category="ocr_transcribe",
        main_metrics=["cer", "wer", "accuracy"],
        default_metric="cer",
        description="TextOCR transcription split; fixed prompt to transcribe all visible text.",
    ),
    "textocr_vqa": DatasetEntry(
        id="textocr_vqa",
        huggingface_id="textocr",
        category="ocr_vqa",
        main_metrics=["accuracy", "anls"],
        default_metric="accuracy",
        description="TextOCR VQA split; uses provided QA pairs to evaluate text reading in images.",
    ),
    # Generic transcription dataset (TrainingDataPro)
    "trainingdatapro": DatasetEntry(
        id="trainingdatapro",
        huggingface_id="TrainingDataPro/ocr-text-detection-in-the-documents",
        category="ocr_transcribe",
        main_metrics=["cer", "wer", "f1"],
        default_metric="cer",
        description="TrainingDataPro OCR text detection/recognition dataset formatted as transcription.",
    ),
    # CC-OCR benchmark (multi configs)
    "cc-ocr": DatasetEntry(
        id="cc-ocr",
        huggingface_id="wulipc/CC-OCR",
        huggingface_config="multi_scene_ocr",
        category="scene_text_vqa",
        main_metrics=["cer", "wer", "f1", "anls", "relaxed_accuracy"],
        default_metric="cer",
        description="CC-OCR benchmark with multiple tracks (scene text, doc parsing, KIE, etc.).",
    ),
    # OCRBench placeholder (metadata only; requires manual loader)
    "ocrbench": DatasetEntry(
        id="ocrbench",
        huggingface_id=None,
        category="ocr_benchmark",
        main_metrics=["score"],
        default_metric="score",
        description="OCRBench aggregated benchmark (Recog/VQA/KIE/HMER); requires manual data setup.",
    ),
    "ocrbench_v2": DatasetEntry(
        id="ocrbench_v2",
        huggingface_id="lmms-lab/OCRBench-v2",
        category="ocr_benchmark",
        main_metrics=["score"],
        default_metric="score",
        description="OCRBench v2 benchmark with multiple OCR/VQA/KIE/HMER tasks.",
    ),
}


def load_dataset_by_id(dataset_id: str, *, split: str = "train", **kwargs):
    """Helper to load a Hugging Face dataset using the registry.

    Falls back to split-specific IDs when provided (e.g., infovqa train/test).
    """
    if load_dataset is None:
        raise RuntimeError("The 'datasets' package is required to load benchmark datasets.")
    entry = DATASET_REGISTRY.get(dataset_id)
    if entry is None:
        raise KeyError(f"Unknown dataset id: {dataset_id}")

    # Resolve dataset identifier based on split hints
    ds_id = entry.huggingface_id
    split_lower = split.lower()
    if entry.huggingface_id_train and split_lower.startswith("train"):
        ds_id = entry.huggingface_id_train
    elif entry.huggingface_id_test and ("test" in split_lower or "val" in split_lower):
        ds_id = entry.huggingface_id_test

    if ds_id is None:
        raise ValueError(f"Dataset '{dataset_id}' does not define a huggingface_id for split '{split}'.")

    hf_name = entry.huggingface_config
    if hf_name is not None:
        return load_dataset(ds_id, hf_name, split=split, **kwargs)
    return load_dataset(ds_id, split=split, **kwargs)
