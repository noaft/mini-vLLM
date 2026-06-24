from types import SimpleNamespace

from minivllm.model import HFRunner


class FakeTokenizer:
    pad_token_id = None
    eos_token_id = 0
    eos_token = "<eos>"
    pad_token = None

    def __call__(self, text: str, return_tensors: str):
        assert return_tensors == "pt"
        return {"input_ids": [[ord(char) for char in text]]}

    def decode(self, token_ids, skip_special_tokens: bool):
        assert skip_special_tokens is True
        return "".join(chr(token_id) for token_id in token_ids)


class FakeModel:
    def __init__(self):
        self.eval_called = False
        self.calls = []

    def eval(self):
        self.eval_called = True

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(logits="fake-logits", past_key_values="fake-cache")


def test_runner_sets_model_to_eval_mode() -> None:
    model = FakeModel()

    HFRunner(tokenizer=FakeTokenizer(), model=model)

    assert model.eval_called is True


def test_encode_and_decode_delegate_to_tokenizer() -> None:
    runner = HFRunner(tokenizer=FakeTokenizer(), model=FakeModel())

    encoded = runner.encode("hi")
    decoded = runner.decode_tokens([104, 105])

    assert encoded == {"input_ids": [[104, 105]]}
    assert decoded == "hi"


def test_prefill_calls_model_with_cache_enabled() -> None:
    model = FakeModel()
    runner = HFRunner(tokenizer=FakeTokenizer(), model=model)

    output = runner.prefill(input_ids=[[1, 2, 3]], attention_mask=[[1, 1, 1]])

    assert output.logits == "fake-logits"
    assert output.kv_cache == "fake-cache"
    assert model.calls[-1] == {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
        "past_key_values": None,
        "use_cache": True,
    }


def test_decode_step_passes_existing_cache() -> None:
    model = FakeModel()
    runner = HFRunner(tokenizer=FakeTokenizer(), model=model)

    output = runner.decode_step(input_ids=[[4]], kv_cache="existing-cache")

    assert output.logits == "fake-logits"
    assert output.kv_cache == "fake-cache"
    assert model.calls[-1] == {
        "input_ids": [[4]],
        "attention_mask": None,
        "past_key_values": "existing-cache",
        "use_cache": True,
    }
