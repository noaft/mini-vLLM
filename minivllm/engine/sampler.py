"""Token sampling utilities."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from minivllm.engine.request import SamplingParams


class Sampler:
    """Select the next token id from model logits."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def sample_next_token(
        self,
        logits: object,
        params: SamplingParams | None = None,
    ) -> int:
        params = params or SamplingParams()
        values = _as_float_list(logits)

        if not values:
            raise ValueError("logits must not be empty")

        if params.temperature == 0:
            return _argmax(values)

        candidates = list(enumerate(values))

        if params.top_k is not None:
            candidates = sorted(candidates, key=lambda item: item[1], reverse=True)
            candidates = candidates[: params.top_k]

        token_ids, filtered_logits = zip(*candidates, strict=True)
        probabilities = _softmax(
            [logit / params.temperature for logit in filtered_logits]
        )

        if params.top_p is not None:
            token_ids, probabilities = _apply_top_p(token_ids, probabilities, params.top_p)

        return self.rng.choices(list(token_ids), weights=list(probabilities), k=1)[0]


def _as_float_list(logits: object) -> list[float]:
    if hasattr(logits, "detach"):
        logits = logits.detach()
    if hasattr(logits, "cpu"):
        logits = logits.cpu()
    if hasattr(logits, "tolist"):
        logits = logits.tolist()

    while isinstance(logits, list) and logits and isinstance(logits[0], list):
        logits = logits[-1]

    if not isinstance(logits, Sequence):
        raise TypeError("logits must be a sequence or tensor-like object")

    return [float(value) for value in logits]


def _argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def _softmax(values: Sequence[float]) -> list[float]:
    max_value = max(values)
    exp_values = [math.exp(value - max_value) for value in values]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def _apply_top_p(
    token_ids: Sequence[int],
    probabilities: Sequence[float],
    top_p: float,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    ranked = sorted(
        zip(token_ids, probabilities, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )

    kept: list[tuple[int, float]] = []
    cumulative = 0.0
    for token_id, probability in ranked:
        kept.append((token_id, probability))
        cumulative += probability
        if cumulative >= top_p:
            break

    kept_token_ids, kept_probabilities = zip(*kept, strict=True)
    total = sum(kept_probabilities)
    normalized = tuple(probability / total for probability in kept_probabilities)
    return tuple(kept_token_ids), normalized
