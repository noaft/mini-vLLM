import pytest

from minivllm.engine import Request, RequestStatus, SamplingParams, Scheduler
from minivllm.engine.scheduler import active_request_ids


def test_sampling_params_validate_bounds() -> None:
    with pytest.raises(ValueError):
        SamplingParams(max_tokens=0)

    with pytest.raises(ValueError):
        SamplingParams(temperature=-1)

    with pytest.raises(ValueError):
        SamplingParams(top_k=0)

    with pytest.raises(ValueError):
        SamplingParams(top_p=1.5)


def test_request_finishes_after_max_tokens() -> None:
    request = Request(
        request_id="req-1",
        prompt="hello",
        sampling_params=SamplingParams(max_tokens=2),
    )

    request.append_token(10)
    request.append_token(11)

    assert request.status == RequestStatus.FINISHED
    assert request.output_token_ids == [10, 11]


def test_scheduler_promotes_waiting_requests_to_batch() -> None:
    scheduler = Scheduler(max_batch_size=2)
    scheduler.add_request(Request(request_id="a", prompt="one"))
    scheduler.add_request(Request(request_id="b", prompt="two"))
    scheduler.add_request(Request(request_id="c", prompt="three"))

    batch = scheduler.schedule()

    assert active_request_ids(batch) == ["a", "b"]
    assert [request.status for request in batch] == [
        RequestStatus.RUNNING,
        RequestStatus.RUNNING,
    ]
    assert active_request_ids(scheduler.waiting) == ["c"]


def test_scheduler_collects_finished_requests() -> None:
    scheduler = Scheduler(max_batch_size=1)
    scheduler.add_request(Request(request_id="a", prompt="one"))
    scheduler.schedule()

    finished = scheduler.finish_request("a")

    assert finished.status == RequestStatus.FINISHED
    assert scheduler.running == ()
    assert active_request_ids(scheduler.collect_finished()) == ["a"]
    assert scheduler.finished == ()
