"""Request and sequence state for the serving engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RequestStatus(str, Enum):
    """Lifecycle state for a generation request."""

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass(frozen=True)
class SamplingParams:
    """User-controllable generation settings."""

    max_tokens: int = 16
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1 when set")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1] when set")


@dataclass
class Request:
    """A single completion request tracked by the scheduler."""

    request_id: str
    prompt: str
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    status: RequestStatus = RequestStatus.WAITING
    prompt_token_ids: list[int] = field(default_factory=list)
    output_token_ids: list[int] = field(default_factory=list)
    error: str | None = None

    @property
    def generated_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def is_finished(self) -> bool:
        return self.status in {RequestStatus.FINISHED, RequestStatus.FAILED}

    def mark_running(self) -> None:
        if self.status == RequestStatus.WAITING:
            self.status = RequestStatus.RUNNING

    def append_token(self, token_id: int) -> None:
        if self.is_finished:
            raise RuntimeError("cannot append token to a finished request")

        self.output_token_ids.append(token_id)
        self.mark_running()

        if self.generated_tokens >= self.sampling_params.max_tokens:
            self.status = RequestStatus.FINISHED

    def finish(self) -> None:
        if self.status != RequestStatus.FAILED:
            self.status = RequestStatus.FINISHED

    def fail(self, message: str) -> None:
        self.status = RequestStatus.FAILED
        self.error = message
