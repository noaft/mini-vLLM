"""Hugging Face model adapter used by the serving engine.

The runner intentionally exposes low-level forward calls instead of using
`model.generate()`. That keeps prefill, decode, cache ownership, and sampling in
mini-vLLM code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelOutput:
    """Normalized output from a causal language model forward pass."""

    logits: Any
    kv_cache: Any


class HFRunner:
    """Small adapter around a Hugging Face causal language model."""

    def __init__(self, tokenizer: Any, model: Any) -> None:
        self.tokenizer = tokenizer
        self.model = model

        if hasattr(self.model, "eval"):
            self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "sshleifer/tiny-gpt2",
        *,
        device: str | None = None,
        dtype: Any | None = None,
    ) -> HFRunner:
        """Load a tokenizer and causal LM from Hugging Face.

        Imports are kept inside this method so the package can be imported in a
        fresh environment before optional runtime dependencies are installed.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_kwargs: dict[str, Any] = {}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if device is not None:
            model = model.to(device)

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        return cls(tokenizer=tokenizer, model=model)

    def encode(self, text: str) -> Any:
        """Encode prompt text into model input tensors."""
        return self.tokenizer(text, return_tensors="pt")

    def decode_tokens(self, token_ids: Any) -> str:
        """Decode token ids into text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def prefill(self, input_ids: Any, attention_mask: Any | None = None) -> ModelOutput:
        """Run the prompt through the model and return logits plus KV cache."""
        return self._forward(input_ids=input_ids, attention_mask=attention_mask)

    def decode_step(
        self,
        input_ids: Any,
        kv_cache: Any,
        attention_mask: Any | None = None,
    ) -> ModelOutput:
        """Run one decode step using an existing KV cache."""
        return self._forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=kv_cache,
        )

    def _forward(
        self,
        *,
        input_ids: Any,
        attention_mask: Any | None = None,
        past_key_values: Any | None = None,
    ) -> ModelOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return ModelOutput(logits=outputs.logits, kv_cache=outputs.past_key_values)
