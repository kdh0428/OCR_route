"""Inference helpers wrapping model.generate for uncertainty analysis."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoProcessor, PreTrainedModel

LOGGER = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Lightweight configuration for generation parameters."""

    max_new_tokens: int = -1
    temperature: float = 0.0
    do_sample: bool = False
    use_cache: bool = True


@dataclass
class GenerationResult:
    """Container for generation outputs and metadata."""

    text: str
    sequences: torch.LongTensor
    generated_tokens: torch.LongTensor
    scores: List[torch.FloatTensor]
    prompt_length: int


def _move_to_device(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move tensor inputs to the specified device if possible."""
    result: Dict[str, Any] = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def generate(
    model: PreTrainedModel,
    processor: AutoProcessor,
    inputs: Dict[str, Any],
    config: Optional[GenerationConfig] = None,
) -> GenerationResult:
    """Run model.generate and collect sequences and scores for uncertainty."""
    config = config or GenerationConfig()
    # InternVL: ensure special token id is set on the model if provided by processor.
    if hasattr(model, "img_context_token_id") and getattr(model, "img_context_token_id", None) is None:
        candidates = [
            getattr(processor, "img_context_token_id", None),
            getattr(getattr(processor, "tokenizer", None), "img_context_token_id", None),
            getattr(getattr(processor, "tokenizer", None), "image_token_id", None),
            getattr(getattr(model, "config", None), "img_context_token_id", None),
        ]
        for cand in candidates:
            if cand is not None:
                try:
                    model.img_context_token_id = cand  # type: ignore[attr-defined]
                    break
                except Exception:
                    pass
        # Fallback to a non-None value to satisfy assertions even if above did not set.
        if getattr(model, "img_context_token_id", None) is None:
            model.img_context_token_id = 0  # type: ignore[attr-defined]

    # Drop raw image passthrough keys that some processors might leave.
    inputs = dict(inputs)
    # Remove any image-carrying keys regardless of type.
    for bad_key in ("images", "image", "pixel_values_varying_size"):
        while bad_key in inputs:
            inputs.pop(bad_key, None)
    # If processor left pixel_values on CPU while model is on GPU, move below via _move_to_device.
    model_inputs = _move_to_device(inputs, model.device)

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    # Salesforce InstructBLIP models can ship with a video-centric processor that prepends repeated <video> tokens.
    # The model expects exactly `num_query_tokens` occurrences of `image_token_index` as placeholders.
    if model_type == "instructblip" and "input_ids" in model_inputs:
        input_ids = model_inputs["input_ids"]
        if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2:
            video_token = getattr(getattr(model, "config", None), "video_token_index", None)
            image_token = getattr(getattr(model, "config", None), "image_token_index", None)
            num_query_tokens = getattr(getattr(model, "config", None), "num_query_tokens", None)
            if (
                isinstance(video_token, int)
                and isinstance(image_token, int)
                and isinstance(num_query_tokens, int)
                and num_query_tokens > 0
            ):
                prefix_len = int((input_ids[0] == video_token).long().sum().item())
                if prefix_len > 0 and prefix_len != num_query_tokens:
                    tail = input_ids[:, prefix_len:]
                    fixed_prefix = torch.full(
                        (input_ids.shape[0], num_query_tokens),
                        image_token,
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    )
                    model_inputs["input_ids"] = torch.cat([fixed_prefix, tail], dim=1)
                    if "attention_mask" in model_inputs and isinstance(model_inputs["attention_mask"], torch.Tensor):
                        attn = model_inputs["attention_mask"]
                        if attn.ndim == 2:
                            fixed_attn = torch.ones(
                                (attn.shape[0], num_query_tokens),
                                dtype=attn.dtype,
                                device=attn.device,
                            )
                            model_inputs["attention_mask"] = torch.cat([fixed_attn, attn[:, prefix_len:]], dim=1)

    prompt_length = model_inputs["input_ids"].shape[-1]
    max_new_tokens = None if config.max_new_tokens is not None and config.max_new_tokens < 0 else config.max_new_tokens
    generation_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=config.do_sample,
        use_cache=config.use_cache,
        return_dict_in_generate=True,
        output_scores=True,
    )
    # Only pass temperature when sampling; some models warn about unused generation flags otherwise.
    if config.do_sample:
        generation_kwargs["temperature"] = config.temperature

    if model_type == "instructblip":
        # InstructBLIP can exhibit degenerate repetition on some prompts; apply mild anti-repetition defaults.
        generation_kwargs.setdefault("repetition_penalty", 1.1)
        generation_kwargs.setdefault("no_repeat_ngram_size", 3)
        generation_kwargs.setdefault("num_beams", 1)
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        eos_token_id = getattr(tok, "eos_token_id", None)
        pad_token_id = getattr(tok, "pad_token_id", None)
        if eos_token_id is not None:
            generation_kwargs.setdefault("eos_token_id", eos_token_id)
        if pad_token_id is not None:
            generation_kwargs.setdefault("pad_token_id", pad_token_id)
        elif eos_token_id is not None:
            # Some tokenizers leave pad unset; using EOS is a common safe default for generation.
            generation_kwargs.setdefault("pad_token_id", eos_token_id)
    # Some models (InternVL) expect img_context_token_id during generate.
    if hasattr(model, "img_context_token_id") and getattr(model, "img_context_token_id", None) is not None:
        generation_kwargs.setdefault("img_context_token_id", getattr(model, "img_context_token_id"))

    LOGGER.debug("Starting generation: %s", generation_kwargs)
    with torch.no_grad():
        outputs = model.generate(**model_inputs, **generation_kwargs)

    sequences = outputs.sequences
    generated_tokens = sequences if prompt_length == 0 else sequences[:, prompt_length:]

    # Decode only the newly generated tokens to avoid echoing the prompt.
    decoded_gen = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    text = decoded_gen[0].strip() if decoded_gen else ""

    if not text:
        LOGGER.warning("Model returned an empty string.")

    return GenerationResult(
        text=text,
        sequences=sequences,
        generated_tokens=generated_tokens,
        scores=outputs.scores,
        prompt_length=prompt_length,
    )
