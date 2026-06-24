import random

from minivllm.engine import Sampler, SamplingParams


def test_temperature_zero_uses_greedy_argmax() -> None:
    sampler = Sampler()

    token_id = sampler.sample_next_token(
        [0.1, 3.0, 1.2],
        SamplingParams(temperature=0),
    )

    assert token_id == 1


def test_sampler_accepts_nested_logits() -> None:
    sampler = Sampler(random.Random(1))

    token_id = sampler.sample_next_token(
        [[0.0, 0.1], [5.0, 0.0]],
        SamplingParams(temperature=0),
    )

    assert token_id == 0


def test_top_k_limits_candidates() -> None:
    sampler = Sampler(random.Random(1))

    token_id = sampler.sample_next_token(
        [10.0, 9.0, -100.0],
        SamplingParams(top_k=2),
    )

    assert token_id in {0, 1}


def test_top_p_keeps_high_probability_prefix() -> None:
    sampler = Sampler(random.Random(1))

    token_id = sampler.sample_next_token(
        [10.0, 0.0, -10.0],
        SamplingParams(top_p=0.8),
    )

    assert token_id == 0
