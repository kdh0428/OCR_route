"""Uncertainty computation utilities based on model output scores."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


@dataclass
class UncertaintyMetrics:
    """Aggregate statistics derived from per-token entropies and logit margins."""

    mean_entropy: float
    min_margin: float
    entropies: list[float]
    margins: list[float]
    filtered_entropies: list[float]
    filtered_margins: list[float]
    filtered_indices: list[int]
    token_confidences: list[float]
    sequence_confidence: float


def compute_metrics(
    scores: Sequence[torch.Tensor],
    generated_tokens: torch.LongTensor,
    special_token_ids: Iterable[int],
    temperature_scale: float = 1.0,
) -> UncertaintyMetrics:
    """Compute entropy, margin, and confidence statistics for generated tokens."""
    if not scores:
        LOGGER.warning("No decoder scores provided; returning neutral metrics.")
        return UncertaintyMetrics(0.0, 0.0, [], [], [], [], [], [], 0.0)

    special_set = set(int(t) for t in special_token_ids)
    entropies: list[float] = []
    margins: list[float] = []
    filtered_entropies: list[float] = []
    filtered_margins: list[float] = []
    filtered_indices: list[int] = []
    token_confidences: list[float] = []

    scale = temperature_scale if temperature_scale and temperature_scale > 0 else 1.0
    sum_log_conf = 0.0
    valid_conf_count = 0

    for idx, score in enumerate(scores):
        logits = score[0] / scale
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum().item()
        topk = torch.topk(logits, k=2)
        margin = (topk.values[0] - topk.values[1]).item()
        entropies.append(entropy)
        margins.append(margin)

        current_token = generated_tokens[0, idx].item() if generated_tokens.numel() else None
        if current_token is not None:
            confidence = float(probs[current_token].item())
            token_confidences.append(confidence)
            if confidence > 0.0:
                sum_log_conf += math.log(confidence)
                valid_conf_count += 1

        if current_token is None or current_token not in special_set:
            filtered_entropies.append(entropy)
            filtered_margins.append(margin)
            filtered_indices.append(idx)

    if not filtered_entropies:
        # Fallback: use all tokens if filtering removed everything.
        filtered_entropies = entropies
        filtered_margins = margins
        filtered_indices = list(range(len(entropies)))

    mean_entropy = float(sum(filtered_entropies) / max(len(filtered_entropies), 1))
    min_margin = float(min(filtered_margins)) if filtered_margins else 0.0

    if valid_conf_count > 0:
        sequence_confidence = float(math.exp(sum_log_conf / valid_conf_count))
    else:
        sequence_confidence = 0.0

    return UncertaintyMetrics(
        mean_entropy=mean_entropy,
        min_margin=min_margin,
        entropies=entropies,
        margins=margins,
        filtered_entropies=filtered_entropies,
        filtered_margins=filtered_margins,
        filtered_indices=filtered_indices,
        token_confidences=token_confidences,
        sequence_confidence=sequence_confidence,
    )


def should_trigger(metrics: UncertaintyMetrics, entropy_th: float, margin_th: float) -> bool:
    """Return True if uncertainty exceeds configured thresholds."""
    return metrics.mean_entropy > entropy_th or metrics.min_margin < margin_th
