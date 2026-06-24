# mini-vLLM Overview

mini-vLLM is organized around the parts of an LLM serving system that sit around
the model. Hugging Face `transformers` will provide the tokenizer and model
forward pass. This repository owns the serving mechanics.

## What This Repo Owns

- Request lifecycle: accepted, running, finished, failed.
- Scheduling: choosing which requests get compute in each step.
- Batching: grouping compatible model calls.
- KV cache ownership: tracking cache objects for active sequences.
- Sampling: choosing the next token from logits.
- API surface: exposing a minimal completion endpoint.

## What Hugging Face Owns

- Tokenizer vocabulary and encode/decode behavior.
- Model architecture and weights.
- The low-level forward pass.
- The initial `past_key_values` tensor format.

The engine must not call `model.generate()`. That method would hide prefill,
decode, sampling, and cache handling, which are the exact concepts this project
exists to study.

## First Runtime Shape

The first model runner should expose two calls:

```python
prefill(input_ids) -> logits, kv_cache
decode_step(input_ids, kv_cache) -> logits, kv_cache
```

`prefill` processes the full prompt. `decode_step` processes only the latest
token while reusing prior keys and values.

## First Model Adapter

`minivllm.model.HFRunner` is the first runtime adapter. It does three small
jobs:

- load `AutoTokenizer` and `AutoModelForCausalLM`
- normalize model output into `logits` and `kv_cache`
- keep `model.generate()` out of the engine path

The initial cache object is whatever Hugging Face returns as `past_key_values`.
Later engine phases can wrap that object in a richer sequence-level cache
manager.

## Request Scheduling

`Request` tracks user-visible generation state: prompt, output token ids,
sampling parameters, lifecycle status, and errors.

`Scheduler` starts as a FIFO scheduler with a fixed `max_batch_size`. This is
intentionally simple. It lets the engine study request admission and decode
batching before adding more complex policies such as priority, preemption, or
memory-aware scheduling.

## Sampling

`Sampler` owns token selection after the model returns logits. Keeping sampling
outside the model runner makes the engine loop explicit:

1. run model forward
2. take final-token logits
3. sample the next token
4. append token to request state

The first sampler supports greedy decoding through `temperature=0`, plus
temperature, top-k, and top-p filtering.

## Learning Boundary

The project should stay small enough that each system concept is visible. When a
feature starts hiding the mechanism, prefer a simpler implementation and explain
what production systems do differently.
