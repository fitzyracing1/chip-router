"""Shared dry-run launch planning."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Protocol

from chip_router.router import Candidate, Decision


@dataclass(frozen=True)
class DryRunPlan:
    provider: str
    summary: str
    would_call: str
    env_var: str
    env_present: bool
    notes: tuple[str, ...] = ()

    def render(self) -> str:
        key_state = "set" if self.env_present else "not set"
        lines = [
            "DRY RUN ONLY — nothing was rented, no API was called, no money was spent.",
            f"Provider: {self.provider}",
            f"Plan: {self.summary}",
            f"Would call: {self.would_call}",
            f"Credential env {self.env_var}: {key_state} (value not printed)",
        ]
        lines.extend(self.notes)
        return "\n".join(lines)


class ProviderAdapter(Protocol):
    name: str
    env_var: str

    def plan(self, candidate: Candidate, count: int = 1) -> DryRunPlan: ...


def env_is_set(name: str) -> bool:
    value = os.environ.get(name)
    return bool(value)


def plan_launch(decision: Decision) -> Optional[DryRunPlan]:
    if decision.choice is None:
        return None
    from chip_router.providers import ADAPTERS

    provider = decision.choice.offer.provider
    adapter = ADAPTERS.get(provider)
    if adapter is None:
        return DryRunPlan(
            provider=provider,
            summary=(
                f"No dry-run adapter is registered for {provider}; "
                f"refusing to invent an API call."
            ),
            would_call="none",
            env_var="",
            env_present=False,
            notes=("Add a provider adapter before this offer can be planned.",),
        )
    return adapter.plan(decision.choice, count=decision.job.count)
