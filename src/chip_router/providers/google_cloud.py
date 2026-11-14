"""Google Cloud TPU dry-run adapter."""

from __future__ import annotations

from chip_router.providers.base import DryRunPlan, env_is_set
from chip_router.router import Candidate

ENV_VAR = "GOOGLE_APPLICATION_CREDENTIALS"


class GoogleCloudAdapter:
    name = "google_cloud"
    env_var = ENV_VAR

    def plan(self, candidate: Candidate, count: int = 1) -> DryRunPlan:
        sku = candidate.offer.sku
        region = candidate.offer.region or "us-east5"
        return DryRunPlan(
            provider="google_cloud",
            summary=(
                f"Create {count} Cloud TPU {sku} chip(s) in {region} "
                f"({candidate.chip.name}) at the catalog on-demand rate of "
                f"${candidate.offer.usd_per_hour:.2f} per chip-hour."
            ),
            would_call=(
                f"gcloud compute tpus tpu-vm create ... --zone={region} "
                f"--accelerator-type matching {sku!r}. Not sent."
            ),
            env_var=ENV_VAR,
            env_present=env_is_set(ENV_VAR),
            notes=(
                "This adapter prints the intended launch only. It does not "
                "create a TPU, even if Google credentials are set.",
            ),
        )
