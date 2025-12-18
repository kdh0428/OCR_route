"""High-level orchestration for uncertainty-triggered OCR routing."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

from transformers import AutoProcessor, PreTrainedModel

from .model import ModelConfig, load_model, set_seed
from .processor import ProcessorConfig, load_processor
from .routing import RoutingConfig, RoutingResult, route

LOGGER = logging.getLogger(__name__)


@dataclass
class Pipeline:
    """A simple container bundling model, processor, and routing config."""

    model: PreTrainedModel
    processor: AutoProcessor
    routing_config: RoutingConfig

    def run(self, image: str, question: str) -> RoutingResult:
        """Execute the routing pipeline for a single query."""
        LOGGER.debug("Running pipeline for image=%s question=%s", image, question)
        return route(
            model=self.model,
            processor=self.processor,
            image_path=image,
            question=question,
            config=self.routing_config,
        )


def create_pipeline(
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
    dtype: Optional[str] = "auto",
    device_map: str | dict | None = "auto",
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    entropy_th: float = 0.7,
    margin_th: float = 1.0,
    entropy_a: Optional[float] = None,
    entropy_b: Optional[float] = None,
    margin_tau: Optional[float] = None,
    history_k: int = 4,
    ocr_engine: str = "paddleocr",
    no_routing: bool = False,
    always_ocr: bool = False,
    max_new_tokens: int = -1,
    temperature: float = 0.0,
    do_sample: bool = False,
    use_kv_cache: bool = True,
    seed: Optional[int] = None,
    enable_wandb: bool = True,
    wandb_project: Optional[str] = "ocr-routing",
    wandb_entity: Optional[str] = "dhkang0428-sungkyunkwan-university",
    wandb_run_name: Optional[str] = "run-20251103",
    wandb_tags: Optional[Sequence[str]] = ("cc-ocr", "baseline", "entropy-0.7"),
    temperature_scale: float = 1.0,
    ocr_prompt_mode: str = "default",
    dataset_id: Optional[str] = None,
) -> Pipeline:
    """Factory that creates and wires together pipeline components."""
    set_seed(seed)

    model_cfg = ModelConfig(model_id=model_id, dtype=dtype, device_map=device_map)
    processor_cfg = ProcessorConfig(model_id=model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    routing_cfg = RoutingConfig(
        entropy_th=entropy_th,
        margin_th=margin_th,
        entropy_a=entropy_a,
        entropy_b=entropy_b,
        margin_tau=margin_tau,
        history_k=history_k,
        ocr_engine=ocr_engine,
        no_routing=no_routing,
        always_ocr=always_ocr,
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        use_kv_cache=use_kv_cache,
        enable_wandb=enable_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        wandb_tags=tuple(wandb_tags) if wandb_tags else (),
        temperature_scale=temperature_scale,
        ocr_prompt_mode=ocr_prompt_mode,
        dataset_id=dataset_id,
    )

    model = load_model(model_cfg)
    processor = load_processor(processor_cfg)

    return Pipeline(model=model, processor=processor, routing_config=routing_cfg)


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point that forwards to the CLI implementation."""
    from .cli import main as cli_main  # Local import to avoid circular dependency

    return cli_main(argv)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    import sys

    sys.exit(main())
