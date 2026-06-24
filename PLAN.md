# mini-vLLM Plan

This plan is organized as learning milestones. Each phase should leave the repo
in a runnable and explainable state.

## Phase 0: Repository Foundation

Purpose: make the project understandable before implementation starts.

Deliverables:

- README with project scope and architecture.
- PLAN with implementation phases.
- AGENTS guide for future agent work.
- Initial package layout.
- Basic test and formatting setup.

Done when:

- A new contributor can explain what the repo is building.
- The next implementation step is obvious.

## Phase 1: Tokenizer

Purpose: load and use a real tokenizer through `transformers` while keeping the
serving engine in this repo.

Initial design:

- Use `AutoTokenizer`.
- Start with a small causal LM such as `sshleifer/tiny-gpt2`.
- Normalize padding/eos behavior in a small wrapper if needed.

Deliverables:

- tokenizer loading path
- encode/decode tests using the selected model tokenizer
- docs explaining what Hugging Face handles versus what mini-vLLM owns

Done when:

- Text can be encoded into token ids and decoded back into text.
- The tokenizer interface is stable enough for the engine.

## Phase 2: Hugging Face Model Runner

Purpose: load a real causal language model but expose only the low-level calls
needed by the serving engine.

Initial design:

- Use `AutoModelForCausalLM`.
- Do not call `model.generate()`.
- Call `model(input_ids=..., past_key_values=..., use_cache=True)` directly.
- Return logits and updated `past_key_values`.

Deliverables:

- `minivllm/model/hf_runner.py`
- model config/loading options
- shape tests for logits and cache outputs

Done when:

- The model accepts token ids and returns logits.
- A forward pass exposes `past_key_values` for later decode steps.

## Phase 3: Naive Generation Loop

Purpose: make the system generate tokens before optimizing anything.

Initial design:

- One request at a time.
- Use Hugging Face model calls manually.
- Recompute the full sequence each step at first if needed.
- Greedy decoding only.

Deliverables:

- `minivllm/engine/engine.py`
- `generate(prompt, max_tokens)` API
- CLI entrypoint

Done when:

- `python -m minivllm.cli --prompt "hello"` runs end to end.

## Phase 4: Prefill and Decode Split

Purpose: expose the two-phase shape of LLM serving.

Initial design:

- `prefill()` handles prompt tokens.
- `decode_step()` handles one generated token.
- Keep behavior equivalent to Phase 3 at first.

Deliverables:

- explicit prefill/decode methods
- docs explaining the difference
- tests that generation still works

Done when:

- The engine code clearly shows where prompt processing ends and token-by-token
  generation begins.

## Phase 5: KV Cache

Purpose: avoid recomputing attention history during decode.

Initial design:

- Wrap Hugging Face `past_key_values`.
- Keep one cache object per active sequence.
- Make cache ownership explicit even if storage is still handled by
  `transformers`.

Deliverables:

- `minivllm/engine/kv_cache.py`
- model runner path that can read/write cache
- tests for cache shape growth
- docs explaining memory tradeoffs

Done when:

- Decode uses cached keys and values.
- Cache behavior is easy to inspect in tests.

## Phase 6: Request and Scheduler Model

Purpose: move from single-request generation to serving multiple active
requests.

Initial design:

- Request object with prompt, sampling params, output tokens, and status.
- Scheduler that selects active requests.
- Simple batching for prefill and decode steps where feasible.
- Keep a clear fallback path for models whose cache format makes batching
  awkward at first.

Deliverables:

- `minivllm/engine/request.py`
- `minivllm/engine/scheduler.py`
- tests for request lifecycle

Done when:

- Multiple requests can be added and stepped through the engine.
- Finished requests are removed or marked complete.

## Phase 7: Sampling

Purpose: separate model execution from token selection.

Initial design:

- Greedy decoding.
- Temperature.
- Top-k.
- Top-p.

Deliverables:

- `minivllm/engine/sampler.py`
- `SamplingParams`
- deterministic tests where possible

Done when:

- The engine can switch sampling behavior without changing model code.

## Phase 8: HTTP Server

Purpose: expose the engine through a small serving interface.

Initial design:

- FastAPI.
- `/health`
- `/v1/completions`
- OpenAI-like request and response shape, but explicitly minimal.

Deliverables:

- `minivllm/server/api.py`
- server run command in README
- curl example

Done when:

- A user can submit a prompt over HTTP and receive generated text.

## Phase 9: Paged KV Cache Study

Purpose: introduce the idea behind vLLM's memory efficiency.

Initial design:

- Keep the existing simple KV cache.
- Add a separate educational paged cache implementation.
- Compare simple contiguous cache vs paged cache in docs.

Deliverables:

- paged cache prototype
- allocation/free tests
- docs explaining fragmentation and page tables

Done when:

- The repo can demonstrate why paged KV cache matters.

## Phase 10: Compatibility and Experiments

Purpose: add optional realism after the core system is clear.

Possible directions:

- Add more Hugging Face causal LM compatibility.
- Add a tiny hand-written Transformer as a separate educational module.
- Add streaming responses.
- Add benchmark scripts.
- Add memory usage instrumentation.
- Add richer scheduling policies.
- Explore speculative decoding.

Done when:

- Each experiment is isolated and documented.
- The core learning path remains simple.

## Current Next Step

Implement Phase 0 package skeleton:

- `pyproject.toml`
- `minivllm/__init__.py`
- initial test config
- `docs/01-overview.md`

Do not start with PagedAttention, CUDA kernels, or model compatibility. Those
come after the plain engine is easy to understand. The main path should start
with a tiny Hugging Face causal LM and manual serving logic.
