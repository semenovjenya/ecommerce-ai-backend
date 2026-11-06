from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProgramSimulationLaunch:
    simulation_run_ids: tuple[str, ...]
    executor_name: str
    details: dict[str, Any] = field(default_factory=dict)


class ProgramSimulationUnsupported(RuntimeError):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})


__all__ = ["ProgramSimulationLaunch", "ProgramSimulationUnsupported"]
