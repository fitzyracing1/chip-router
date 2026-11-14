"""RunPod dry-run adapter."""

from __future__ import annotations

from chip_router.providers.base import DryRunPlan, env_is_set
from chip_router.router import Candidate

ENV_VAR = "RUNPOD_API_KEY"


class RunPodAdapter:
    name = "runpod"
    env_var = ENV_VAR

    def plan(self, candidate: Candidate, count: int = 1) -> DryRunPlan:
        sku = candidate.offer.sku
        return DryRunPlan(
            provider="runpod",
            summary=(
                f"Start {count} Secure Cloud pod(s) with GPU {sku} "
                f"({candidate.chip.name}) at the catalog list price of "
                f"${candidate.offer.usd_per_hour:.2f}/hr each."
            ),
            would_call=(
                "POST https://api.runpod.io/graphql (or the REST pods API) "
                f"to create a pod with gpuType={sku!r}. "
                "Not sent."
            ),
            env_var=ENV_VAR,
            env_present=env_is_set(ENV_VAR),
            notes=(
                "This adapter prints the intended launch only. It does not "
                "create a pod, even if RUNPOD_API_KEY is set.",
            ),
        )
