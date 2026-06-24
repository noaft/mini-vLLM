from types import SimpleNamespace

from minivllm.engine import Engine, SamplingParams


class FakeRunner:
    def __init__(self):
        self.decode_calls = []

    def encode(self, text: str):
        return {"input_ids": [[ord(char) for char in text]], "attention_mask": [[1]]}

    def decode_tokens(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)

    def prefill(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=[0.0, 100.0, 99.0], kv_cache="prompt-cache")

    def decode_step(self, input_ids, kv_cache, attention_mask=None):
        self.decode_calls.append((input_ids, kv_cache))
        return SimpleNamespace(logits=[0.0, 99.0, 100.0], kv_cache="decode-cache")


def test_engine_generates_with_prefill_then_decode() -> None:
    runner = FakeRunner()
    engine = Engine(runner)

    result = engine.generate(
        "hi",
        SamplingParams(max_tokens=2, temperature=0),
        request_id="req-1",
    )

    assert result.request_id == "req-1"
    assert result.token_ids == [1, 2]
    assert result.text == "\x01\x02"
    assert runner.decode_calls == [([[1]], "prompt-cache")]
    assert len(engine.cache_store) == 0
