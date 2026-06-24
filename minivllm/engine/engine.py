"""Naive generation engine wiring model runner, cache, and sampler together."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from minivllm.engine.kv_cache import KVCacheStore
from minivllm.engine.request import Request, SamplingParams
from minivllm.engine.sampler import Sampler


@dataclass(frozen=True)
class GenerationResult:
    """Completed generation output."""

    request_id: str
    prompt: str
    text: str
    token_ids: list[int]


class Engine:
    """Single-request generation engine.

    This first engine is intentionally simple. It proves the serving loop:
    prefill once, then repeatedly decode one token using the KV cache.
    """

    def __init__(self, runner: Any, sampler: Sampler | None = None) -> None:
        self.runner = runner
        self.sampler = sampler or Sampler()
        self.cache_store = KVCacheStore()

    def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        *,
        request_id: str | None = None,
    ) -> GenerationResult:
        params = sampling_params or SamplingParams()
        request = Request(
            request_id=request_id or str(uuid4()),
            prompt=prompt,
            sampling_params=params,
        )

        encoded = self.runner.encode(prompt)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        request.prompt_token_ids = _flatten_token_ids(input_ids)

        prefill_output = self.runner.prefill(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cache = self.cache_store.add_prefill_cache(
            request_id=request.request_id,
            value=prefill_output.kv_cache,
            prompt_tokens=len(request.prompt_token_ids),
        )

        logits = prefill_output.logits
        last_token_id: int | None = None

        while not request.is_finished:
            if last_token_id is not None:
                decode_output = self.runner.decode_step(
                    input_ids=_single_token_batch(last_token_id),
                    kv_cache=cache.value,
                )
                cache = self.cache_store.update_after_decode(
                    request.request_id,
                    decode_output.kv_cache,
                )
                logits = decode_output.logits

            next_token_id = self.sampler.sample_next_token(logits, params)
            request.append_token(next_token_id)
            last_token_id = next_token_id

        self.cache_store.remove(request.request_id)
        return GenerationResult(
            request_id=request.request_id,
            prompt=prompt,
            text=self.runner.decode_tokens(request.output_token_ids),
            token_ids=list(request.output_token_ids),
        )


def _single_token_batch(token_id: int) -> list[list[int]]:
    return [[token_id]]


def _flatten_token_ids(input_ids: Any) -> list[int]:
    if hasattr(input_ids, "detach"):
        input_ids = input_ids.detach()
    if hasattr(input_ids, "cpu"):
        input_ids = input_ids.cpu()
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()

    while isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]

    return [int(token_id) for token_id in input_ids]
