"""Model management for Qwen vision-language models."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    InstructBlipForConditionalGeneration,
    InternVLForConditionalGeneration,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    PreTrainedModel,
    Qwen2VLForConditionalGeneration,
)

LOGGER = logging.getLogger(__name__)


def _resolve_dtype(dtype: str | None) -> torch.dtype | None:
    """Map string dtype flags to torch dtypes; return None for auto."""
    if not dtype or dtype == "auto":
        return None

    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "full": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype flag: {dtype}")
    return mapping[dtype]


@dataclass
class ModelConfig:
    """Configuration options for model loading."""

    model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    dtype: str | None = "auto"
    device_map: str | dict | None = "auto"


def load_model(config: ModelConfig) -> PreTrainedModel:
    """Load a Qwen-family vision-language model according to the configuration."""
    torch_dtype = _resolve_dtype(config.dtype)
    model_id = config.model_id
    model_id_lower = model_id.lower()

    # InternVL3/3.5: HF 포맷 리포로 강제 매핑
    internvl_aliases = {
        "opengvlab/internvl3_5-8b": "OpenGVLab/InternVL3_5-8B-HF",
        "opengvlab/internvl3-8b": "OpenGVLab/InternVL3_5-8B-HF",
        "opengvlab/internvl3_5-8b-hf": "OpenGVLab/InternVL3_5-8B-HF",
        "opengvlab/internvl3-8b-hf": "OpenGVLab/InternVL3_5-8B-HF",
    }
    if model_id_lower in internvl_aliases:
        model_id = internvl_aliases[model_id_lower]
        model_id_lower = model_id.lower()

    LOGGER.info(
        "Loading model %s with dtype=%s device_map=%s",
        model_id,
        torch_dtype or "auto",
        config.device_map,
    )
    # InternVL, LLaVA, and Ocean all rely on custom code; force trust_remote_code for them.
    trust_remote = any(keyword in model_id_lower for keyword in ("ocean-ocr", "llava", "internvl"))

    hf_config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=trust_remote,
    )

    auto_map = getattr(hf_config, "auto_map", {}) or {}

    if hf_config.model_type == "ocean":
        loader = AutoModelForCausalLM
    elif hf_config.model_type == "internvl" or "internvl" in model_id_lower:
        # InternVL requires a generation-capable wrapper; AutoModel returns InternVLModel (no .generate()).
        loader = InternVLForConditionalGeneration
    elif hf_config.model_type in {"instructblip"} or "instructblip" in model_id_lower:
        loader = InstructBlipForConditionalGeneration
    elif hf_config.model_type in {"llava_next"} or "llava-1.6" in model_id_lower or "llava-v1.6" in model_id_lower:
        loader = LlavaNextForConditionalGeneration
    elif (hf_config.model_type and hf_config.model_type.startswith("llava")) or "llava" in model_id_lower:
        loader = LlavaForConditionalGeneration
    elif "AutoModelForImageTextToText" in auto_map:
        loader = AutoModelForImageTextToText
    elif "AutoModelForVision2Seq" in auto_map or "qwen2.5" in model_id_lower or "qwen2_5" in model_id_lower:
        loader = AutoModelForVision2Seq
    else:
        loader = Qwen2VLForConditionalGeneration

    try:
        common_kwargs = dict(
            torch_dtype=torch_dtype,
            device_map=config.device_map,
            trust_remote_code=trust_remote,
            low_cpu_mem_usage=True,
        )
        model = loader.from_pretrained(
            model_id,
            **common_kwargs,
        )
    except RuntimeError as exc:
        if loader is Qwen2VLForConditionalGeneration:
            LOGGER.info("Falling back to AutoModelForVision2Seq for %s", config.model_id)
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=config.device_map,
                trust_remote_code=trust_remote,
            )
        else:
            raise
    model.eval()
    return model


def set_seed(seed: Optional[int]) -> None:
    """Reproducibly seed torch and related libraries."""
    if seed is None:
        return
    LOGGER.info("Setting random seed: %s", seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - hardware dependent
        torch.cuda.manual_seed_all(seed)
