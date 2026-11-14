"""Modal dry-run adapter."""

from __future__ import annotations

from chip_router.providers.base import DryRunPlan, env_is_set
from chip_router.router import Candidate

ENV_VAR = "MODAL_TOKEN_ID"


class ModalAdapter:
    name = "modal"
    env_var = ENV_VAR

    def plan(self, candidate: Candidate, count: int = 1) -> DryRunPlan:
        sku = candidate.offer.sku
        per_sec = candidate.offer.usd_per_second
        rate = (
            f"${per_sec}/sec"
            if per_sec is not None
            else f"${candidate.offer.usd_per_hour:.4f}/hr"
        )
        return DryRunPlan(
            provider="modal",
            summary=(
                f"Run a Modal function with gpu={sku!r} "
                f"({candidate.chip.name}), count={count}, billed per second "
                f"at the catalog GPU-only rate of {rate}."
            ),
            would_call=(
                "modal.Function or the Modal API with "
                f"gpu={sku!r}. Not sent. Modal also bills CPU and memory."
            ),
            env_var=ENV_VAR,
            env_present=env_is_set(ENV_VAR) or env_is_set("MODAL_TOKEN_SECRET"),
            notes=(
                "This adapter prints the intended launch only. It does not "
                "start a Modal app, even if Modal tokens are set.",
            ),
        )
