"""Routing policy that decides when to invoke OCR-assisted generation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence
import math
import re

from transformers import AutoProcessor, PreTrainedModel

from .inference import GenerationConfig, GenerationResult, generate
from .ocr_engines import run_ocr
from .processor import prepare_inputs
from .uncertainty import UncertaintyMetrics, compute_metrics, should_trigger
from .utils import load_image, now_ms, track

LOGGER = logging.getLogger(__name__)

try:  # Optional dependency for experiment tracking
    import wandb
except ImportError:  # pragma: no cover - telemetry is best-effort
    wandb = None
WANDB_UNAVAILABLE_WARNED = False


@dataclass
class PassResult:
    """Holds outputs for a single generation pass."""

    answer: str
    generation: GenerationResult
    metrics: UncertaintyMetrics


@dataclass
class RoutingConfig:
    """Configuration knobs for routing decisions."""

    entropy_th: float = 0.7
    margin_th: float = 1.0
    entropy_a: Optional[float] = None
    entropy_b: Optional[float] = None
    margin_tau: Optional[float] = None
    history_k: int = 4
    ocr_engine: str = "paddleocr"
    no_routing: bool = False
    always_ocr: bool = False
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    # Use -1 to disable max_new_tokens limit (GenerationConfig will get None)
    max_new_tokens: int = -1
    temperature: float = 0.0
    do_sample: bool = False
    use_kv_cache: bool = True
    structured_keywords: Sequence[str] = ("latex", "table", "tabular", "equation", "matrix")
    structured_patterns: Sequence[str] = (
        r"\\begin\{",
        r"\\frac",
        r"\\section",
        r"\\tabular",
        r"\\matrix",
    )
    enable_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: Sequence[str] = ()
    temperature_scale: float = 1.0
    entropy_use_zscore: bool = True
    entropy_ema_decay: float = 0.1
    entropy_ema_mean: float = 0.0
    entropy_ema_sq: float = 0.0
    entropy_ema_initialized: bool = False
    ocr_prompt_mode: str = "default"  # default | verify
    dataset_id: Optional[str] = None


@dataclass
class RoutingResult:
    """Final decision containing per-pass details and telemetry."""

    answer: str
    used_ocr: bool
    triggered: bool
    first: PassResult
    second: Optional[PassResult]
    ocr_text: Optional[str]
    timings: dict[str, float]
    total_latency_ms: int
    final_metrics: UncertaintyMetrics


def _ensure_wandb_run(config: RoutingConfig) -> None:
    """Best-effort initialization of a W&B run when telemetry is requested."""
    global WANDB_UNAVAILABLE_WARNED
    if not config.enable_wandb:
        return
    if wandb is None:
        if not WANDB_UNAVAILABLE_WARNED:
            WANDB_UNAVAILABLE_WARNED = True
            LOGGER.warning("Weights & Biases not installed; telemetry disabled.")
        return
    if wandb.run is not None:
        return
    init_kwargs: dict[str, object] = {}
    if config.wandb_project:
        init_kwargs["project"] = config.wandb_project
    if config.wandb_entity:
        init_kwargs["entity"] = config.wandb_entity
    if config.wandb_run_name:
        init_kwargs["name"] = config.wandb_run_name
    if config.wandb_tags:
        init_kwargs["tags"] = list(config.wandb_tags)
    try:
        wandb.init(**init_kwargs)
    except Exception:  # pragma: no cover - telemetry is best-effort
        LOGGER.exception("Failed to initialize Weights & Biases run.")


def _wandb_log(payload: dict[str, float], commit: bool = False) -> None:
    """Log metrics to Weights & Biases without impacting control flow."""
    if wandb is None or wandb.run is None:
        return
    try:
        wandb.log(payload, commit=commit)
    except Exception:  # pragma: no cover - logging must not interrupt routing
        LOGGER.exception("Failed to send metrics to Weights & Biases.")


def _wandb_log_pass(pass_name: str, result: PassResult, latency_ms: Optional[float]) -> None:
    """Emit uncertainty statistics for an individual generation pass."""
    payload = {
        f"routing/{pass_name}/mean_entropy": float(result.metrics.mean_entropy),
        f"routing/{pass_name}/normalized_entropy": float(result.metrics.normalized_entropy),
        f"routing/{pass_name}/min_margin": float(result.metrics.min_margin),
        f"routing/{pass_name}/token_count": float(len(result.metrics.entropies)),
        f"routing/{pass_name}/sequence_confidence": float(result.metrics.sequence_confidence),
    }
    if latency_ms is not None:
        payload[f"routing/{pass_name}/latency_ms"] = float(latency_ms)
    _wandb_log(payload)


def _compute_entropy_zscore(config: RoutingConfig, current_entropy: float) -> tuple[float, float, float]:
    """Return z-scored entropy along with current EMA mean/std (before updating)."""
    if not config.entropy_use_zscore or not config.entropy_ema_initialized:
        std = 1.0
        mean = config.entropy_ema_mean if config.entropy_ema_initialized else current_entropy
        normalized = (current_entropy - mean) / std if config.entropy_ema_initialized else 0.0
        return normalized, mean, std

    mean = config.entropy_ema_mean
    variance = max(config.entropy_ema_sq - mean * mean, 1e-6)
    std = math.sqrt(variance)
    normalized = (current_entropy - mean) / std
    return normalized, mean, std


def _update_entropy_ema(config: RoutingConfig, current_entropy: float) -> None:
    """Update EMA statistics with the latest entropy measurement."""
    if not config.entropy_use_zscore:
        return
    decay = config.entropy_ema_decay
    if not config.entropy_ema_initialized:
        config.entropy_ema_mean = current_entropy
        config.entropy_ema_sq = current_entropy * current_entropy
        config.entropy_ema_initialized = True
        return
    config.entropy_ema_mean = (1.0 - decay) * config.entropy_ema_mean + decay * current_entropy
    config.entropy_ema_sq = (1.0 - decay) * config.entropy_ema_sq + decay * (current_entropy * current_entropy)


def _run_pass(
    model: PreTrainedModel,
    processor: AutoProcessor,
    image_source,
    question: str,
    generation_cfg: GenerationConfig,
    config: RoutingConfig,
    ocr_text: Optional[str] = None,
    timings: Optional[dict[str, float]] = None,
    timing_key: str = "latency_ms",
    ) -> PassResult:
    # Only DocVQA에 한해서 엄격한 단문/ANSWER 포맷을 강제한다.
    system_prompt = None
    # Base system prompt used everywhere
    base_prompt = (
        "You are an OCR-style vision-language assistant.\n"
        "Answer the question according to the image using a single word or phrase.\n"
        "Do NOT add fillers or extra sentences.\n"
        "The last line of your response should be of the form \"ANSWER: [ANSWER]\" (without quotes) "
        "where [ANSWER] is the answer to the question."
    )
    system_prompt = base_prompt

    model_id_lower = (config.model_id or "").lower()
    # InstructBLIP follows a simpler "Question/Answer:" prompting style; avoid strict ANSWER: formatting.
    if "instructblip" in model_id_lower:
        system_prompt = None
    # Infographic VQA (InfoVQA): simpler wording, single word/phrase requirement.
    if config.dataset_id == "infovqa" and "instructblip" not in model_id_lower:
        system_prompt = (
            "You are an OCR-style vision-language assistant.\n"
            "Answer the question using a single word or phrase.\n"
            "Do NOT add fillers or extra sentences.\n"
            "The last line of your response should be of the form \"ANSWER: [ANSWER]\" (without quotes) "
            "where [ANSWER] is the answer to the question."
        )

    if config.dataset_id == "docvqa" and "llava-1.5" in model_id_lower:
        # DocVQA + LLaVA 1.5: add an extra brevity hint.
        system_prompt = base_prompt + "\nKeep the answer as short as possible (ideally one word or number)."
    with track(timing_key, timings if timings is not None else {}):
        inputs = prepare_inputs(
            processor,
            image_source,
            question,
            ocr_text=ocr_text,
            ocr_prompt_mode=config.ocr_prompt_mode if config else "default",
            system_prompt=system_prompt,
        )
        result = generate(model, processor, inputs, config=generation_cfg)
        special_tokens = processor.tokenizer.all_special_ids
        metrics = compute_metrics(
            result.scores,
            result.generated_tokens,
            special_tokens,
            temperature_scale=config.temperature_scale,
        )
    return PassResult(answer=result.text, generation=result, metrics=metrics)


def _compute_entropy_threshold(config: RoutingConfig, generated_length: int) -> float:
    if config.entropy_a is not None and config.entropy_b is not None:
        length_term = math.log1p(max(generated_length, 1))
        return config.entropy_a + config.entropy_b * length_term
    return config.entropy_th


def _compute_margin_threshold(config: RoutingConfig) -> float:
    if config.margin_tau is not None:
        return config.margin_tau
    return config.margin_th


def _entropy_hysteresis(metrics: UncertaintyMetrics, threshold: float, history_k: int) -> bool:
    if not metrics.entropies:
        return False
    if history_k <= 0 or len(metrics.entropies) <= history_k:
        return metrics.mean_entropy > threshold
    recent = metrics.entropies[-history_k:]
    return sum(recent) / len(recent) > threshold


def _should_trigger_dynamic(
    metrics: UncertaintyMetrics,
    entropy_threshold: float,
    margin_threshold: float,
    history_k: int,
    normalized_entropy: float,
    normalized_recent: Optional[float],
    use_zscore: bool,
) -> bool:
    if use_zscore:
        entropy_flag = normalized_entropy > entropy_threshold
        hysteresis_flag = normalized_recent is not None and normalized_recent > entropy_threshold
    else:
        entropy_flag = metrics.mean_entropy > entropy_threshold
        hysteresis_flag = _entropy_hysteresis(metrics, entropy_threshold, history_k)
    margin_flag = metrics.min_margin < margin_threshold
    return (entropy_flag or hysteresis_flag) or margin_flag


def _contains_structured_pattern(
    question: str,
    answer_hint: Optional[str],
    config: RoutingConfig,
) -> bool:
    corpus = " ".join(filter(None, [question, answer_hint or ""]))
    lowered = corpus.lower()
    if any(keyword in lowered for keyword in config.structured_keywords):
        return True
    for pattern in config.structured_patterns:
        if re.search(pattern, corpus):
            return True
    return False


def route(
    model: PreTrainedModel,
    processor: AutoProcessor,
    image_path: str,
    question: str,
    config: RoutingConfig,
) -> RoutingResult:
    """Execute routing logic and return detailed results."""
    timings: dict[str, float] = {}
    overall_start = now_ms()
    generation_cfg = GenerationConfig(
        max_new_tokens=None if config.max_new_tokens is not None and config.max_new_tokens < 0 else config.max_new_tokens,
        temperature=config.temperature,
        do_sample=config.do_sample,
        use_cache=config.use_kv_cache,
    )

    _ensure_wandb_run(config)

    image = load_image(image_path)

    first_pass = _run_pass(
        model,
        processor,
        image,
        question,
        generation_cfg,
        config,
        timings=timings,
        timing_key="latency_ms_first",
    )

    normalized_entropy, ema_mean, ema_std = _compute_entropy_zscore(config, first_pass.metrics.mean_entropy)
    entropy_value_for_threshold = (
        normalized_entropy if config.entropy_use_zscore else first_pass.metrics.mean_entropy
    )
    first_pass.metrics.normalized_entropy = entropy_value_for_threshold
    if config.enable_wandb:
        _wandb_log_pass("first_pass", first_pass, timings.get("latency_ms_first"))

    normalized_recent = None
    if (
        config.entropy_use_zscore
        and config.entropy_ema_initialized
        and config.history_k > 0
        and first_pass.metrics.entropies
    ):
        window = first_pass.metrics.entropies[-min(config.history_k, len(first_pass.metrics.entropies)) :]
        if window:
            window_mean = sum(window) / len(window)
            if ema_std > 1e-6:
                normalized_recent = (window_mean - ema_mean) / ema_std

    legacy_trigger = (
        entropy_value_for_threshold > config.entropy_th or first_pass.metrics.min_margin < config.margin_th
    )
    generated_length = first_pass.generation.generated_tokens.shape[-1]
    entropy_threshold = _compute_entropy_threshold(config, generated_length)
    margin_threshold = _compute_margin_threshold(config)
    dynamic_trigger = _should_trigger_dynamic(
        first_pass.metrics,
        entropy_threshold,
        margin_threshold,
        config.history_k,
        entropy_value_for_threshold,
        normalized_recent,
        config.entropy_use_zscore,
    )
    force_structured = _contains_structured_pattern(question, first_pass.answer, config)
    LOGGER.info(
        (
            "Routing decision: mean_entropy=%.3f min_margin=%.3f legacy=%s "
            "dynamic=%s tau_H=%.3f tau_m=%.3f structured=%s"
        ),
        first_pass.metrics.mean_entropy,
        first_pass.metrics.min_margin,
        legacy_trigger,
        dynamic_trigger,
        entropy_threshold,
        margin_threshold,
        force_structured,
    )

    second_pass: Optional[PassResult] = None
    ocr_text: Optional[str] = None
    execute_second = False

    if not config.no_routing:
        if config.always_ocr or force_structured:
            execute_second = True
        elif dynamic_trigger:
            execute_second = True

    if execute_second:
        with track("latency_ms_ocr", timings):
            ocr_text = run_ocr(config.ocr_engine, image)
        LOGGER.info("OCR text length: %d", len(ocr_text) if ocr_text else 0)
        second_pass = _run_pass(
            model,
            processor,
            image,
            question,
            generation_cfg,
            config,
            ocr_text=ocr_text,
            timings=timings,
            timing_key="latency_ms_second",
        )
        if config.entropy_use_zscore and ema_std > 1e-6:
            second_pass.metrics.normalized_entropy = (second_pass.metrics.mean_entropy - ema_mean) / ema_std
        else:
            second_pass.metrics.normalized_entropy = (
                second_pass.metrics.mean_entropy if not config.entropy_use_zscore else 0.0
            )
        if config.enable_wandb:
            _wandb_log_pass("second_pass", second_pass, timings.get("latency_ms_second"))

    final_answer = first_pass.answer
    final_metrics = first_pass.metrics
    if second_pass:
        improved = False
        if second_pass.answer and not first_pass.answer:
            improved = True
        elif second_pass.metrics.mean_entropy < first_pass.metrics.mean_entropy:
            improved = True
        elif second_pass.metrics.min_margin > first_pass.metrics.min_margin:
            improved = True

        if improved:
            final_answer = second_pass.answer
            final_metrics = second_pass.metrics
            LOGGER.info("Second pass selected based on uncertainty metrics.")
        else:
            LOGGER.info("First pass retained after comparing uncertainty metrics.")

    total_latency_ms = now_ms() - overall_start
    if config.enable_wandb:
        _wandb_log(
            {
                "routing/triggered": float(execute_second),
                "routing/used_ocr": float(bool(second_pass)),
                "routing/ocr_text_length": float(len(ocr_text) if ocr_text else 0),
                "routing/legacy_trigger": float(legacy_trigger),
                "routing/dynamic_trigger": float(dynamic_trigger),
                "routing/entropy_threshold": float(entropy_threshold),
                "routing/margin_threshold": float(margin_threshold),
                "routing/total_latency_ms": float(total_latency_ms),
                "routing/final/normalized_entropy": float(final_metrics.normalized_entropy),
                "routing/final/sequence_confidence": float(final_metrics.sequence_confidence),
                "routing/final/mean_entropy": float(final_metrics.mean_entropy),
            },
            commit=True,
        )
    _update_entropy_ema(config, first_pass.metrics.mean_entropy)
    return RoutingResult(
        answer=final_answer,
        used_ocr=bool(second_pass),
        triggered=execute_second,
        first=first_pass,
        second=second_pass,
        ocr_text=ocr_text,
        timings=timings,
        total_latency_ms=total_latency_ms,
        final_metrics=final_metrics,
    )
