"""Processor configuration and input preparation for Qwen2-VL."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from PIL import Image

from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor

from .utils.image_io import load_image

LOGGER = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """Configuration for the Qwen2-VL processor."""

    model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None


def load_processor(config: ProcessorConfig) -> AutoProcessor:
    """Instantiate the AutoProcessor with dynamic resolution options."""
    # InternVL3.5 GitHub 포맷은 HF 포맷으로 강제 매핑
    if config.model_id in {
        "OpenGVLab/InternVL3_5-8B",
        "opengvlab/internvl3_5-8b",
        "OpenGVLab/InternVL3-8B",
        "opengvlab/internvl3-8b",
    }:
        resolved_model_id = "OpenGVLab/InternVL3_5-8B-HF"
    else:
        resolved_model_id = config.model_id
    model_id_lower = resolved_model_id.lower()
    kwargs: Dict[str, int] = {}
    if config.min_pixels:
        kwargs["min_pixels"] = config.min_pixels
    if config.max_pixels:
        kwargs["max_pixels"] = config.max_pixels
    trust_remote = any(keyword in model_id_lower for keyword in ("ocean-ocr", "llava", "internvl"))
    LOGGER.info(
        "Loading processor %s with options %s",
        resolved_model_id,
        kwargs if kwargs else "default",
    )
    processor = None
    for use_fast in (True, False):
        try:
            processor = AutoProcessor.from_pretrained(
                resolved_model_id,
                trust_remote_code=trust_remote,
                use_fast=use_fast,
                **kwargs,
            )
            break
        except Exception as exc:
            if use_fast and "internvl" in model_id_lower:
                LOGGER.warning(
                    "Fast processor failed for %s with %s; retrying use_fast=False",
                    resolved_model_id,
                    exc,
                )
                continue
            raise
    setattr(processor, "_ocr_route_model_id", resolved_model_id)
    return processor


def _infer_model_id(processor: AutoProcessor) -> str:
    return getattr(processor, "_ocr_route_model_id", getattr(processor, "name_or_path", "") or "")


def build_conversation(
    model_id: str,
    question: str,
    ocr_text: Optional[str] = None,
    ocr_prompt_mode: str = "default",
    system_prompt: Optional[str] = None,
) -> list[dict]:
    """Construct a conversation compatible with Qwen2-VL processors."""
    model_id_lower = model_id.lower()
    # Normalize question/ocr_text to strings to avoid concat errors.
    question_str = question if isinstance(question, str) else str(question)
    if ocr_text is not None:
        if isinstance(ocr_text, (list, tuple)):
            ocr_text_str = "\n".join(str(x) for x in ocr_text)
        else:
            ocr_text_str = str(ocr_text)
    else:
        ocr_text_str = None
    use_verify_prompt = ocr_text_str and ocr_prompt_mode == "verify"
    if ocr_text_str:
        if use_verify_prompt:
            ocr_block = (
                "Below is raw OCR text from the image (may contain errors). "
                "Cross-check with the image and only output the corrected final text; "
                "fix any wrong numbers or missing items:\n"
                f"{ocr_text_str}"
            )
        else:
            ocr_block = (
                "Here is OCR-extracted text from the image (may contain errors). "
                "Verify it against the visual content before finalizing your answer:\n"
                f"{ocr_text_str}"
            )
    else:
        ocr_block = None

    system_msg = None
    if system_prompt:
        # Most HF chat templates (including Qwen2/2.5-VL) expect system content as a plain string.
        system_msg = {"role": "system", "content": system_prompt}

    if "internvl" in model_id_lower:
        # InternVL: use plain text prompt (no chat template).
        prompt_lines = []
        if system_prompt:
            prompt_lines.append(system_prompt)
        prompt_lines.append(question_str)
        if ocr_block:
            prompt_lines.append(ocr_block)
        return "\n".join(prompt_lines)
    if "ocean-ocr" in model_id_lower:
        content = [
            {"type": "text", "text": question_str},
            {"type": "image"},
        ]
    elif "instructblip" in model_id_lower:
        # InstructBLIP expects a plain prompt like "Question: ... Answer:" (no chat template).
        parts = []
        # Keep prompt minimal to avoid degenerate repetition on some LLM backends.
        parts.append(f"Question: {question_str.strip()}")
        if ocr_block:
            parts.append(ocr_block.strip())
        parts.append("Answer:")
        return "\n".join(p for p in parts if p)
    elif "llava" in model_id_lower:
        content = [{"type": "image"}]
        appended_text = question_str
        if ocr_block:
            appended_text += "\n" + ocr_block
        content.append({"type": "text", "text": appended_text})
        convo = []
        if system_msg:
            convo.append(system_msg)
        convo.append({"role": "user", "content": content})
        return convo
    else:
        content = [
            {"type": "image"},
            {"type": "text", "text": question_str},
        ]
    if ocr_block:
        content.append({"type": "text", "text": ocr_block})
    convo = []
    if system_msg:
        convo.append(system_msg)
    convo.append({"role": "user", "content": content})
    return convo


def prepare_inputs(
    processor: AutoProcessor,
    image_source: str | Image.Image,
    question: str,
    ocr_text: Optional[str] = None,
    ocr_prompt_mode: str = "default",
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Load the image and tokenize the conversation for model.generate."""
    image = load_image(image_source)
    model_id_lower = _infer_model_id(processor).lower()
    conversation = None

    # InternVL 계열은 이미지 플레이스홀더 오류를 피하기 위해 직접 프롬프트를 구성한다.
    if "internvl" in model_id_lower:
        question_str = question if isinstance(question, str) else str(question)
        if ocr_text is not None:
            if isinstance(ocr_text, (list, tuple)):
                ocr_block = "\n".join(str(x) for x in ocr_text)
            else:
                ocr_block = str(ocr_text)
            if ocr_prompt_mode == "verify":
                ocr_block = (
                    "Below is raw OCR text from the image (may contain errors). "
                    "Cross-check with the image and only output the corrected final text; "
                    "fix any wrong numbers or missing items:\n"
                    f"{ocr_block}"
                )
            else:
                ocr_block = (
                    "Here is OCR-extracted text from the image (may contain errors). "
                    "Verify it against the visual content before finalizing your answer:\n"
                    f"{ocr_block}"
                )
        else:
            ocr_block = None

        tok = getattr(processor, "tokenizer", None)
        # InternVLProcessor expects its own `image_token` placeholder (e.g., "<IMG_CONTEXT>") in the prompt.
        placeholder = (
            getattr(processor, "image_token", None)
            or getattr(tok, "image_token", None)
            or "<image>"
        )
        parts = []
        if system_prompt:
            parts.append(system_prompt)
        parts.append(question_str)
        if ocr_block:
            parts.append(ocr_block)
        base_prompt = "\n".join(p for p in parts if p)

        def ensure_placeholder(prompt: str, placeholder_token: str, n_images: int) -> str:
            cnt = prompt.count(placeholder_token)
            if n_images == 1:
                if cnt == 0:
                    return f"{placeholder_token}\n{prompt}"
                if cnt == 1:
                    return prompt
                # cnt > 1: placeholder를 모두 제거하고 하나만 붙인다.
                stripped = prompt.replace(placeholder_token, "").strip()
                return f"{placeholder_token}\n{stripped}"
            else:
                # n_images == 0
                return prompt.replace(placeholder_token, "").strip()

        if image is None:
            prompt = ensure_placeholder(base_prompt, placeholder, 0)
            inputs = processor(
                text=prompt,
                return_tensors="pt",
            )
        else:
            prompt = ensure_placeholder(base_prompt, placeholder, 1)
            inputs = processor(
                text=prompt,
                images=[image],
                return_tensors="pt",
            )
    elif "instructblip" in model_id_lower:
        # InstructBLIP uses a dual-tokenizer setup (Q-Former BERT + LLM tokenizer). The Q-Former side has a hard
        # max length (typically 512). Long OCR text can exceed this and crash with position-embedding mismatches.
        question_str = question if isinstance(question, str) else str(question)
        parts = [f"Question: {question_str.strip()}"]
        if ocr_text is not None:
            ocr_text_str = "\n".join(str(x) for x in ocr_text) if isinstance(ocr_text, (list, tuple)) else str(ocr_text)
            ocr_text_str = ocr_text_str.strip()
            if ocr_text_str:
                qtok = getattr(processor, "qformer_tokenizer", None)
                if qtok is not None and getattr(qtok, "model_max_length", None):
                    max_len = int(getattr(qtok, "model_max_length", 512) or 512)
                    prefix = "\n".join(parts + ["OCR text:"]) + "\n"
                    suffix = "\nAnswer:"
                    prefix_ids = qtok.encode(prefix, add_special_tokens=False)
                    suffix_ids = qtok.encode(suffix, add_special_tokens=False)
                    remaining = max_len - len(prefix_ids) - len(suffix_ids)
                    if remaining > 0:
                        ocr_ids = qtok.encode(ocr_text_str, add_special_tokens=False)
                        ocr_trunc = qtok.decode(ocr_ids[:remaining], skip_special_tokens=True).strip()
                        if ocr_trunc:
                            parts.extend(["OCR text:", ocr_trunc])
                else:
                    parts.extend(["OCR text:", ocr_text_str])
        parts.append("Answer:")
        prompt = "\n".join(p for p in parts if p)
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            truncation=True,
        )
    else:
        conversation = build_conversation(
            model_id_lower,
            question,
            ocr_text=ocr_text,
            ocr_prompt_mode=ocr_prompt_mode,
            system_prompt=system_prompt,
        )
        if isinstance(conversation, str):
            # InstructBLIP-style plain prompt.
            prompt = conversation
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            )
        else:
            prompt = processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            )
    # Keep only tensor-like entries to avoid passing raw images/metadata to generate.
    inputs = {k: v for k, v in inputs.items() if hasattr(v, "to")}
    # InstructBLIP processors can emit pixel_values with an extra singleton image dim: (B, 1, C, H, W).
    pix = inputs.get("pixel_values")
    if pix is not None and hasattr(pix, "dim") and pix.dim() == 5 and pix.shape[1] == 1:
        inputs["pixel_values"] = pix[:, 0]
    # Remove any lingering raw-image keys.
    for bad_key in ("images", "image", "pixel_values_varying_size"):
        while bad_key in inputs:
            inputs.pop(bad_key, None)
    return inputs
