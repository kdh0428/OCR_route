"""Run all benchmark combos and aggregate summaries.

This script sequentially launches run_dataset_eval.py for a fixed set of
model/dataset combinations (CC-OCR, DocVQA, InfoVQA, OCRBench-v2) and then
collects the last JSONL record from each run into outputs/summary_all.jsonl.

Usage:
    python3 scripts/run_all_bench.py

Before running, set CUDA_VISIBLE_DEVICES if you want to pin GPUs globally.
Max-new-tokens defaults follow project conventions:
    - CC-OCR: 2048
    - Others: 256
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass
class RunSpec:
    dataset: str
    split: str
    max_new_tokens: int
    model_id: str
    out_name: str


RUNS: List[RunSpec] = [
    # Qwen2.5-VL
    RunSpec("cc-ocr", "test[:10%]", 2048, "Qwen/Qwen2.5-VL-7B-Instruct", "cc_ocr_test10_qwen25vl.jsonl"),
    RunSpec("ocrbench_v2", "test[:10%]", 256, "Qwen/Qwen2.5-VL-7B-Instruct", "ocrbench_v2_test10_qwen25vl.jsonl"),
    RunSpec("docvqa", "validation[:10%]", 256, "Qwen/Qwen2.5-VL-7B-Instruct", "docvqa_val10_qwen25vl.jsonl"),
    RunSpec("infovqa", "val[:10%]", 256, "Qwen/Qwen2.5-VL-7B-Instruct", "infovqa_val10_qwen25vl.jsonl"),

    # LLaVA 1.6 (vicuna local path)
    RunSpec("cc-ocr", "test[:10%]", 2048, str(PROJECT_ROOT / "llava-v1.6-vicuna-7b-hf"), "cc_ocr_test10_llava16.jsonl"),
    RunSpec("ocrbench_v2", "test[:10%]", 256, str(PROJECT_ROOT / "llava-v1.6-vicuna-7b-hf"), "ocrbench_v2_test10_llava16.jsonl"),
    RunSpec("docvqa", "validation[:10%]", 256, str(PROJECT_ROOT / "llava-v1.6-vicuna-7b-hf"), "docvqa_val10_llava16.jsonl"),
    RunSpec("infovqa", "val[:10%]", 256, str(PROJECT_ROOT / "llava-v1.6-vicuna-7b-hf"), "infovqa_val10_llava16.jsonl"),

    # InstructBLIP
    RunSpec("cc-ocr", "test[:10%]", 2048, "Salesforce/instructblip-vicuna-7b", "cc_ocr_test10_instructblip.jsonl"),
    RunSpec("ocrbench_v2", "test[:10%]", 256, "Salesforce/instructblip-vicuna-7b", "ocrbench_v2_test10_instructblip.jsonl"),
    RunSpec("docvqa", "validation[:10%]", 256, "Salesforce/instructblip-vicuna-7b", "docvqa_val10_instructblip.jsonl"),
    RunSpec("infovqa", "val[:10%]", 256, "Salesforce/instructblip-vicuna-7b", "infovqa_val10_instructblip.jsonl"),

    # InternVL2.5
    RunSpec("cc-ocr", "test[:10%]", 2048, "OpenGVLab/InternVL2_5-8B", "cc_ocr_test10_internvl2_5.jsonl"),
    RunSpec("ocrbench_v2", "test[:10%]", 256, "OpenGVLab/InternVL2_5-8B", "ocrbench_v2_test10_internvl2_5.jsonl"),
    RunSpec("docvqa", "validation[:10%]", 256, "OpenGVLab/InternVL2_5-8B", "docvqa_val10_internvl2_5.jsonl"),
    RunSpec("infovqa", "val[:10%]", 256, "OpenGVLab/InternVL2_5-8B", "infovqa_val10_internvl2_5.jsonl"),
]


def question_template_for(dataset: str) -> List[str]:
    if dataset == "cc-ocr":
        return ["--config", "multi_scene_ocr", "--question-template", "Please transcribe all visible text in this chart."]
    if dataset == "docvqa":
        return ["--question-template", "Please answer the question by reading the document."]
    if dataset == "infovqa":
        return ["--question-template", "Answer the question based on the infographic."]
    if dataset == "ocrbench_v2":
        return ["--question-template", "Please answer the question by reading the document or scene."]
    return []


def run_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in RUNS:
        cmd = [
            "python3",
            "scripts/run_dataset_eval.py",
            "--dataset",
            spec.dataset,
            "--split",
            spec.split,
            "--out",
            str(OUTPUT_DIR / spec.out_name),
            "--model-id",
            spec.model_id,
            "--ocr-engine",
            "easyocr",
            "--experiment-set",
            "none",
            "--include-baseline",
            "--include-always",
            "--max-new-tokens",
            str(spec.max_new_tokens),
            "--print-comparisons",
        ]
        cmd.extend(question_template_for(spec.dataset))
        print(f"=== Running {spec.out_name} ===")
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    # Aggregate summaries
    summary_path = OUTPUT_DIR / "summary_all.jsonl"
    files = sorted(OUTPUT_DIR.glob("*.jsonl"))
    summaries = []
    for fp in files:
        last = None
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                last = json.loads(line)
        if last is None:
            continue
        if "metrics_table" in last:
            summaries.append({"file": fp.name, "metrics_table": last["metrics_table"]})
        else:
            summaries.append({"file": fp.name, "summary": last})

    with summary_path.open("w", encoding="utf-8") as f:
        for s in summaries:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved summary to {summary_path.resolve()}")
    for s in summaries:
        print(json.dumps(s, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_all()
