# AGENTS.md

This file guides future agent work in this repository.

## Project Intent

mini-vLLM is a learning project for understanding LLM inference serving systems.
Favor small, inspectable implementations over production completeness.

The repository should help a reader understand:

- how prompts become token ids
- how a decoder-only Transformer produces logits
- how prefill differs from decode
- how KV cache changes decode cost
- how request scheduling and batching work
- how sampling turns logits into tokens
- how a minimal server wraps the engine

## Working Style

- Keep changes scoped to the current phase in `PLAN.md`.
- Prefer readable code and explicit data structures.
- Add tests for behavior that teaches a system concept.
- Update docs when implementation changes the mental model.
- Avoid broad refactors unless they make the current phase clearer.
- Do not add heavy production features before the basic path works.

## Implementation Preferences

- Use Python for the first version.
- Use PyTorch for the model implementation.
- Use FastAPI only when the server phase starts.
- Start with a character-level tokenizer.
- Start with a simple contiguous KV cache before paged KV cache.
- Keep interfaces stable and boring:
  - `Tokenizer.encode()`
  - `Tokenizer.decode()`
  - `Engine.generate()`
  - `Scheduler.add_request()`
  - `Scheduler.step()`

## Documentation Expectations

When adding a subsystem, explain:

1. What it does.
2. Why a serving engine needs it.
3. What simplification this repo makes compared with vLLM.

Docs should be practical and concrete. Avoid marketing language.

## Testing Expectations

Early tests should focus on:

- tensor shapes
- tokenizer round trips
- request state transitions
- cache growth
- deterministic sampling paths

Avoid brittle tests that depend on high-quality generated text. This project is
about inference mechanics first.

## Things To Avoid Early

- CUDA-specific code
- distributed serving
- tensor parallelism
- complicated Hugging Face compatibility
- production PagedAttention before simple KV cache exists
- large dependencies that hide the mechanism being studied

## Recommended Next Action

If the repo only has planning files, implement Phase 0 from `PLAN.md`:

1. Add `pyproject.toml`.
2. Create the `minivllm/` package.
3. Add a minimal test setup.
4. Add `docs/01-overview.md`.

After that, continue with the tokenizer phase.
