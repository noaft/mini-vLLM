"""Command-line entrypoint for mini-vLLM."""

from __future__ import annotations

import argparse

from minivllm.engine import Engine, SamplingParams
from minivllm.model import HFRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mini-vLLM generation.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    runner = HFRunner.from_pretrained(args.model)
    engine = Engine(runner)
    result = engine.generate(
        args.prompt,
        SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature),
    )
    print(result.text)


if __name__ == "__main__":
    main()
