"""Sequence-level KV cache bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class KVCache:
    """KV cache owned by one active sequence."""

    request_id: str
    value: Any
    prompt_tokens: int
    generated_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.generated_tokens

    def update_after_decode(self, value: Any) -> None:
        self.value = value
        self.generated_tokens += 1


class KVCacheStore:
    """In-memory mapping from request id to sequence cache."""

    def __init__(self) -> None:
        self._caches: dict[str, KVCache] = {}

    def add_prefill_cache(
        self,
        *,
        request_id: str,
        value: Any,
        prompt_tokens: int,
    ) -> KVCache:
        if prompt_tokens < 1:
            raise ValueError("prompt_tokens must be >= 1")
        if request_id in self._caches:
            raise ValueError(f"cache already exists for request: {request_id}")

        cache = KVCache(
            request_id=request_id,
            value=value,
            prompt_tokens=prompt_tokens,
        )
        self._caches[request_id] = cache
        return cache

    def get(self, request_id: str) -> KVCache:
        try:
            return self._caches[request_id]
        except KeyError as exc:
            raise KeyError(f"cache not found for request: {request_id}") from exc

    def update_after_decode(self, request_id: str, value: Any) -> KVCache:
        cache = self.get(request_id)
        cache.update_after_decode(value)
        return cache

    def remove(self, request_id: str) -> KVCache:
        try:
            return self._caches.pop(request_id)
        except KeyError as exc:
            raise KeyError(f"cache not found for request: {request_id}") from exc

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._caches

    def __len__(self) -> int:
        return len(self._caches)
