import pytest

from minivllm.engine import KVCacheStore


def test_store_adds_prefill_cache() -> None:
    store = KVCacheStore()

    cache = store.add_prefill_cache(
        request_id="req-1",
        value="prefill-cache",
        prompt_tokens=4,
    )

    assert cache.request_id == "req-1"
    assert cache.value == "prefill-cache"
    assert cache.total_tokens == 4
    assert "req-1" in store


def test_store_updates_cache_after_decode() -> None:
    store = KVCacheStore()
    store.add_prefill_cache(request_id="req-1", value="prefill-cache", prompt_tokens=4)

    cache = store.update_after_decode("req-1", value="decode-cache")

    assert cache.value == "decode-cache"
    assert cache.generated_tokens == 1
    assert cache.total_tokens == 5


def test_store_removes_finished_cache() -> None:
    store = KVCacheStore()
    store.add_prefill_cache(request_id="req-1", value="prefill-cache", prompt_tokens=4)

    removed = store.remove("req-1")

    assert removed.request_id == "req-1"
    assert len(store) == 0


def test_store_rejects_duplicate_cache() -> None:
    store = KVCacheStore()
    store.add_prefill_cache(request_id="req-1", value="prefill-cache", prompt_tokens=4)

    with pytest.raises(ValueError):
        store.add_prefill_cache(
            request_id="req-1",
            value="other-cache",
            prompt_tokens=4,
        )
