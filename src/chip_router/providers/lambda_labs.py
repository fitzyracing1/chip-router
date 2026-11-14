"""Lambda Labs dry-run adapter."""

from __future__ import annotations

from chip_router.providers.base import DryRunPlan, env_is_set
from chip_router.router import Candidate

ENV_VAR = "LAMBDA_API_KEY"


class LambdaAdapter:
    name = "lambda"
    env_var = ENV_VAR

    def plan(self, candidate: Candidate, count: int = 1) -> DryRunPlan:
        sku = candidate.offer.sku
        extra = ""
        if candidate.offer.effective_at:
            extra = (
                f" Catalog price takes effect {candidate.offer.effective_at}; "
                "confirm the live rate in a Lambda Cloud account before launch."
            )
        size = candidate.offer.instance_size or "1x"
        return DryRunPlan(
            provider="lambda",
            summary=(
                f"Launch a Lambda Cloud {size} instance with {sku} "
                f"({candidate.chip.name}) at the catalog list price of "
                f"${candidate.offer.usd_per_hour:.2f} per GPU-hour. "
                f"Requested device count: {count}.{extra}"
            ),
            would_call=(
                "POST https://cloud.lambda.ai/api/v1/instance-operations/launch "
                f"with instance_type matching {sku!r}. Not sent."
            ),
            env_var=ENV_VAR,
            env_present=env_is_set(ENV_VAR),
            notes=(
                "This adapter prints the intended launch only. It does not "
                "create an instance, even if LAMBDA_API_KEY is set.",
            ),
        )
