"""Runtime and job-plan abstractions for task packages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class JobState(str, Enum):
    """Execution state for a planned task job."""

    READY = "ready"
    DEFERRED = "deferred"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class RuntimeProbe:
    """Best-effort dependency probe for the current machine."""

    has_torch: bool
    has_cuda: bool
    has_diffusers: bool
    has_sklearn: bool
    has_matplotlib: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommandSpec:
    """Shell command description for deferred GPU work."""

    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class JobPlan:
    """One job emitted by a task planner."""

    job_id: str
    state: JobState
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]
    reason: str | None = None
    command: CommandSpec | None = None


@dataclass(frozen=True)
class TaskPlan:
    """Plan emitted by GPU-bound task wrappers."""

    task_name: str
    jobs: tuple[JobPlan, ...]
    warnings: tuple[str, ...] = ()


def probe_runtime() -> RuntimeProbe:
    """Detect the lightweight dependency surface available on this machine."""
    warnings: list[str] = []

    try:
        import torch
    except Exception:
        torch = None  # type: ignore[assignment]
    try:
        import diffusers  # noqa: F401
        has_diffusers = True
    except Exception:
        has_diffusers = False
    try:
        import sklearn  # noqa: F401
        has_sklearn = True
    except Exception:
        has_sklearn = False
    try:
        import matplotlib  # noqa: F401
        has_matplotlib = True
    except Exception:
        has_matplotlib = False

    has_torch = torch is not None
    has_cuda = bool(has_torch and torch.cuda.is_available())

    if has_torch and not has_cuda:
        warnings.append("torch available but CUDA is unavailable")
    if not has_torch:
        warnings.append("torch is unavailable")
    if not has_diffusers:
        warnings.append("diffusers is unavailable")

    return RuntimeProbe(
        has_torch=has_torch,
        has_cuda=has_cuda,
        has_diffusers=has_diffusers,
        has_sklearn=has_sklearn,
        has_matplotlib=has_matplotlib,
        warnings=tuple(warnings),
    )
