"""Simple request scheduler for mini-vLLM."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from minivllm.engine.request import Request, RequestStatus


class Scheduler:
    """FIFO scheduler with a fixed decode batch size."""

    def __init__(self, max_batch_size: int = 4) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")

        self.max_batch_size = max_batch_size
        self._waiting: deque[Request] = deque()
        self._running: list[Request] = []
        self._finished: list[Request] = []

    def add_request(self, request: Request) -> None:
        if request.status != RequestStatus.WAITING:
            raise ValueError("only waiting requests can be scheduled")
        self._waiting.append(request)

    def schedule(self) -> list[Request]:
        """Return the next batch of runnable requests."""
        self._promote_waiting_requests()
        batch = [request for request in self._running if not request.is_finished]
        return batch[: self.max_batch_size]

    def finish_request(self, request_id: str) -> Request:
        request = self._find_running(request_id)
        request.finish()
        self._running.remove(request)
        self._finished.append(request)
        return request

    def fail_request(self, request_id: str, message: str) -> Request:
        request = self._find_running(request_id)
        request.fail(message)
        self._running.remove(request)
        self._finished.append(request)
        return request

    def collect_finished(self) -> list[Request]:
        finished = list(self._finished)
        self._finished.clear()
        return finished

    @property
    def waiting(self) -> tuple[Request, ...]:
        return tuple(self._waiting)

    @property
    def running(self) -> tuple[Request, ...]:
        return tuple(self._running)

    @property
    def finished(self) -> tuple[Request, ...]:
        return tuple(self._finished)

    def _promote_waiting_requests(self) -> None:
        while self._waiting and len(self._running) < self.max_batch_size:
            request = self._waiting.popleft()
            request.mark_running()
            self._running.append(request)

    def _find_running(self, request_id: str) -> Request:
        for request in self._running:
            if request.request_id == request_id:
                return request
        raise KeyError(f"running request not found: {request_id}")


def active_request_ids(requests: Iterable[Request]) -> list[str]:
    """Return request ids in scheduler order."""
    return [request.request_id for request in requests]
