"""Job specification: what a workload needs from a chip."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

TASK_TYPES = ("inference", "fine-tune", "training", "signal-processing")
PRECISIONS = ("fp8", "bf16", "fp16", "fp32")
OPTIMIZE_MODES = ("cost", "performance", "balanced")
SOFTWARE_PREFS = ("any", "cuda", "rocm", "tpu")

# Used only when vram_gb is omitted and model_size_gb is set.
# These are conservative single-device headroom multiples, not a
# guarantee of fit for a particular framework or batch size.
TASK_MEMORY_MULTIPLIER = {
    "inference": 1.2,
    "fine-tune": 2.5,
    "training": 4.0,
    "signal-processing": 1.2,
}


@dataclass(frozen=True)
class JobSpec:
    name: str
    task: str
    precision: str = "bf16"
    model_size_gb: Optional[float] = None
    vram_gb: Optional[float] = None
    budget_usd_per_hour: Optional[float] = None
    budget_usd_per_job: Optional[float] = None
    max_runtime_hours: Optional[float] = None
    preferred_providers: tuple[str, ...] = ()
    excluded_providers: tuple[str, ...] = ()
    allowed_providers: tuple[str, ...] = ()
    optimize: str = "balanced"
    software: str = "any"
    count: int = 1
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def required_vram_gb(self) -> float:
        if self.vram_gb is not None:
            return float(self.vram_gb)
        if self.model_size_gb is None:
            raise ValueError(
                "job is missing both vram_gb and model_size_gb; "
                "set one so the router can apply a memory floor"
            )
        multiplier = TASK_MEMORY_MULTIPLIER[self.task]
        return float(self.model_size_gb) * multiplier

    def vram_was_estimated(self) -> bool:
        return self.vram_gb is None and self.model_size_gb is not None

    def job_budget_ceiling(self) -> Optional[float]:
        """Max $/hr implied by a per-job budget and a runtime cap, if both exist."""
        if (
            self.budget_usd_per_job is not None
            and self.max_runtime_hours
            and self.max_runtime_hours > 0
        ):
            return self.budget_usd_per_job / self.max_runtime_hours
        return None


def load_job(path: str | Path) -> JobSpec:
    job_path = Path(path)
    with job_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{job_path} is not a YAML mapping")
    return job_from_dict(data)


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _opt_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def job_from_dict(data: dict[str, Any]) -> JobSpec:
    name = str(data.get("name") or data.get("id") or "unnamed-job")
    task = str(data.get("task") or data.get("task_type") or "").strip()
    if task not in TASK_TYPES:
        raise ValueError(
            f"task must be one of {TASK_TYPES}, got {task!r}"
        )
    precision = str(data.get("precision") or "bf16").strip().lower()
    if precision not in PRECISIONS:
        raise ValueError(
            f"precision must be one of {PRECISIONS}, got {precision!r}"
        )
    optimize = str(data.get("optimize") or "balanced").strip().lower()
    if optimize not in OPTIMIZE_MODES:
        raise ValueError(
            f"optimize must be one of {OPTIMIZE_MODES}, got {optimize!r}"
        )
    software = str(data.get("software") or "any").strip().lower()
    if software not in SOFTWARE_PREFS:
        raise ValueError(
            f"software must be one of {SOFTWARE_PREFS}, got {software!r}"
        )
    count = int(data.get("count") or 1)
    if count < 1:
        raise ValueError("count must be >= 1")
    return JobSpec(
        name=name,
        task=task,
        precision=precision,
        model_size_gb=_opt_float(data.get("model_size_gb")),
        vram_gb=_opt_float(data.get("vram_gb")),
        budget_usd_per_hour=_opt_float(data.get("budget_usd_per_hour")),
        budget_usd_per_job=_opt_float(data.get("budget_usd_per_job")),
        max_runtime_hours=_opt_float(data.get("max_runtime_hours")),
        preferred_providers=_as_tuple(data.get("preferred_providers")),
        excluded_providers=_as_tuple(data.get("excluded_providers")),
        allowed_providers=_as_tuple(data.get("allowed_providers")),
        optimize=optimize,
        software=software,
        count=count,
        raw=data,
    )
