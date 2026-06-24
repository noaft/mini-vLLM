# mini-vLLM

mini-vLLM is a small learning-first LLM serving engine inspired by
[vLLM](https://github.com/vllm-project/vllm). The goal is not to reimplement
vLLM completely. The goal is to build a compact system that makes the core
ideas easy to inspect, modify, and reason about.

This repository is for learning how an inference serving system works around a
model: request scheduling, prefill and decode phases, KV cache management,
batching, sampling, and a small serving API.

## Project Goals

- Build a minimal LLM serving engine around Hugging Face `transformers`.
- Use real pretrained models without calling `model.generate()`.
- Own the decode loop, request lifecycle, KV cache handling, batching, and
  sampling logic.
- Keep each subsystem small enough to read in one sitting.
- Prefer explicit code over clever abstractions.
- Document the system tradeoffs as the code evolves.
- Make the project runnable from the command line and later through an HTTP API.

## Non-Goals

This project intentionally avoids production complexity at the start:

- No distributed inference.
- No tensor parallelism.
- No custom CUDA kernels.
- No custom model architecture in the main path.
- No full Hugging Face compatibility beyond a small causal language model path
  in the first phase.
- No production-grade PagedAttention implementation until the simpler KV cache
  model is clear.

Those topics can be added later as learning milestones.

## Planned Architecture

```text
mini-vLLM/
  minivllm/
    model/
      hf_runner.py
    engine/
      request.py
      scheduler.py
      kv_cache.py
      sampler.py
      engine.py
    server/
      api.py
    cli.py
  tests/
  docs/
  README.md
  PLAN.md
  AGENTS.md
```

## Core Concepts

### Prefill

The prefill phase processes the full prompt. It is compute-heavy because the
model attends over all prompt tokens.

### Decode

The decode phase generates one token at a time. It should reuse previous
attention keys and values through the KV cache instead of recomputing the whole
prompt.

### HF Model Runner

The model runner loads a tokenizer and causal language model from
`transformers`. It exposes low-level `prefill()` and `decode_step()` calls. The
repo should not use `model.generate()` because that would hide the serving
logic we want to study.

### KV Cache

The KV cache stores attention keys and values for previous tokens. In the first
version, this wraps Hugging Face `past_key_values`. Later versions can add a
more explicit cache manager and an educational paged cache.

### Scheduler

The scheduler decides which requests run together. A naive engine serves one
request at a time; a serving engine batches active requests to improve hardware
utilization.

### Sampler

The sampler turns logits into the next token. The initial implementation should
support greedy decoding, then temperature, top-k, and top-p sampling.

## Development Order

See [PLAN.md](./PLAN.md) for the full roadmap.

The first useful milestone is:

1. Create project skeleton.
2. Load a small Hugging Face causal LM.
3. Add a manual `generate()` loop without `model.generate()`.
4. Split the loop into prefill and decode.
5. Wrap Hugging Face `past_key_values` as the first KV cache model.
6. Add sampling.
7. Add request scheduling and batching.

## Setup

Create a virtual environment and install the project in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
```

## Expected Usage

Early CLI target:

```bash
python -m minivllm.cli --prompt "hello" --max-tokens 20
```

Later server target:

```bash
uvicorn minivllm.server.api:app --reload
```

Example API shape:

```http
POST /v1/completions
```

```json
{
  "prompt": "hello",
  "max_tokens": 20,
  "temperature": 0.8
}
```

## Learning Philosophy

Every feature should answer two questions:

1. What does this subsystem do in a serving engine?
2. What simplification are we making compared with production vLLM?

If a change makes the code harder to study, it should earn its complexity.

The main learning path is serving mechanics, not Transformer architecture. A
toy Transformer can be added later as a separate study module, but the core repo
should use `transformers` so the generated output is real and the system work
stays visible.
