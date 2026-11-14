"""Deterministic, explainable chip router."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from chip_router.catalog import Catalog, Chip, Offer, load_catalog
from chip_router.job import JobSpec, load_job

# In balanced mode, inference and signal-processing prefer chips whose
# memory is inside this window so a 192 GB accelerator is not chosen
# for a 16 GB job just because its throughput-per-dollar is high.
RIGHTSIZE_MEMORY_FACTOR = 3.0
RIGHTSIZE_MEMORY_PADDING_GB = 40.0


@dataclass(frozen=True)
class Candidate:
    chip: Chip
    offer: Offer
    required_vram_gb: float
    throughput: Optional[float]
    estimated_usd_per_hour: float
    estimated_usd_for_job: Optional[float]
    in_rightsize_window: bool
    skip_reasons: tuple[str, ...] = ()

    @property
    def key(self) -> tuple:
        return (self.chip.id, self.offer.provider, self.offer.sku)


@dataclass(frozen=True)
class Decision:
    job: JobSpec
    required_vram_gb: float
    choice: Optional[Candidate]
    reason: str
    runners_up: tuple[Candidate, ...]
    rejected: tuple[Candidate, ...]
    catalog_updated: str = ""
    catalog_path: str = ""

    def as_log_record(
        self,
        timestamp: str,
        actual: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        choice = self.choice
        record: dict[str, Any] = {
            "timestamp": timestamp,
            "job": self.job.name,
            "task": self.job.task,
            "precision": self.job.precision,
            "required_vram_gb": self.required_vram_gb,
            "chip": choice.chip.id if choice else None,
            "chip_name": choice.chip.name if choice else None,
            "provider": choice.offer.provider if choice else None,
            "sku": choice.offer.sku if choice else None,
            "est_cost_usd_per_hour": (
                choice.estimated_usd_per_hour if choice else None
            ),
            "est_cost_usd_for_job": (
                choice.estimated_usd_for_job if choice else None
            ),
            "pricing_kind": choice.offer.pricing_kind if choice else None,
            "reason": self.reason,
            "runners_up": [
                {
                    "chip": item.chip.id,
                    "provider": item.offer.provider,
                    "sku": item.offer.sku,
                    "usd_per_hour": item.estimated_usd_per_hour,
                }
                for item in self.runners_up
            ],
        }
        if actual is not None:
            record["actual"] = actual
        return record


class NoFitError(LookupError):
    """Raised when no catalog offer satisfies the job."""


def _software_ok(chip: Chip, job: JobSpec) -> bool:
    if job.software == "any":
        return True
    return chip.software == job.software


def _offer_allowed(offer: Offer, job: JobSpec) -> bool:
    if offer.provider in job.excluded_providers:
        return False
    if job.allowed_providers and offer.provider not in job.allowed_providers:
        return False
    return True


def _hourly_ceiling(job: JobSpec) -> Optional[float]:
    caps = [
        cap
        for cap in (job.budget_usd_per_hour, job.job_budget_ceiling())
        if cap is not None
    ]
    if not caps:
        return None
    return min(caps)


def _job_cost(offer: Offer, job: JobSpec) -> Optional[float]:
    if job.max_runtime_hours is None:
        return None
    return offer.usd_per_hour * job.max_runtime_hours * job.count


def _rightsize_cap(required_vram_gb: float) -> float:
    return max(
        required_vram_gb * RIGHTSIZE_MEMORY_FACTOR,
        required_vram_gb + RIGHTSIZE_MEMORY_PADDING_GB,
    )


def _throughput_for(chip: Chip, job: JobSpec) -> Optional[float]:
    if job.task == "signal-processing":
        return chip.memory_bandwidth_gbs
    if job.precision == "fp8":
        return chip.fp8_tflops_dense or chip.bf16_tflops_dense
    return chip.dense_throughput(job.precision)


def _collect(
    catalog: Catalog, job: JobSpec
) -> tuple[list[Candidate], list[Candidate]]:
    required = job.required_vram_gb()
    ceiling = _hourly_ceiling(job)
    rightsize_cap = _rightsize_cap(required)
    accepted: list[Candidate] = []
    rejected: list[Candidate] = []
    for chip in catalog.chips:
        skip: list[str] = []
        if chip.memory_gb < required:
            skip.append(
                f"{chip.name} has {chip.memory_gb:g} GB, below the "
                f"{required:g} GB memory floor"
            )
        if not _software_ok(chip, job):
            skip.append(
                f"{chip.name} uses the {chip.software} stack, but the "
                f"job asked for {job.software}"
            )
        if not chip.supports_precision(job.precision):
            skip.append(
                f"{chip.name} does not publish FP8 Tensor Core support, "
                f"and the job asked for {job.precision}"
            )
        offers = [offer for offer in chip.offers if _offer_allowed(offer, job)]
        if not chip.offers:
            skip.append(f"{chip.name} has no priced provider offers in the catalog")
        elif not offers:
            skip.append(
                f"{chip.name} has offers, but none are on an allowed provider"
            )
        if skip and not offers:
            rejected.append(
                Candidate(
                    chip=chip,
                    offer=Offer(
                        provider="",
                        sku="",
                        usd_per_hour=0.0,
                        pricing_kind="public_list",
                        retrieved="",
                        source_url="",
                    ),
                    required_vram_gb=required,
                    throughput=_throughput_for(chip, job),
                    estimated_usd_per_hour=0.0,
                    estimated_usd_for_job=None,
                    in_rightsize_window=chip.memory_gb <= rightsize_cap,
                    skip_reasons=tuple(skip),
                )
            )
            continue
        if skip:
            # Chip-level skip applies to every remaining offer.
            for offer in offers:
                rejected.append(
                    Candidate(
                        chip=chip,
                        offer=offer,
                        required_vram_gb=required,
                        throughput=_throughput_for(chip, job),
                        estimated_usd_per_hour=offer.usd_per_hour * job.count,
                        estimated_usd_for_job=_job_cost(offer, job),
                        in_rightsize_window=chip.memory_gb <= rightsize_cap,
                        skip_reasons=tuple(skip),
                    )
                )
            continue
        for offer in offers:
            offer_skip: list[str] = []
            hourly = offer.usd_per_hour * job.count
            job_cost = _job_cost(offer, job)
            if ceiling is not None and hourly > ceiling + 1e-9:
                offer_skip.append(
                    f"{offer.provider} {offer.sku} is ${hourly:.2f}/hr, "
                    f"above the ${ceiling:.2f}/hr budget cap"
                )
            if (
                job.budget_usd_per_job is not None
                and job_cost is not None
                and job_cost > job.budget_usd_per_job + 1e-9
            ):
                offer_skip.append(
                    f"{offer.provider} {offer.sku} would cost about "
                    f"${job_cost:.2f} over {job.max_runtime_hours:g} h, "
                    f"above the ${job.budget_usd_per_job:.2f} job cap"
                )
            candidate = Candidate(
                chip=chip,
                offer=offer,
                required_vram_gb=required,
                throughput=_throughput_for(chip, job),
                estimated_usd_per_hour=hourly,
                estimated_usd_for_job=job_cost,
                in_rightsize_window=chip.memory_gb <= rightsize_cap,
                skip_reasons=tuple(offer_skip),
            )
            if offer_skip:
                rejected.append(candidate)
            else:
                accepted.append(candidate)
    return accepted, rejected


def _preferred_rank(candidate: Candidate, job: JobSpec) -> int:
    if not job.preferred_providers:
        return 0
    return 0 if candidate.offer.provider in job.preferred_providers else 1


def _sort_key(candidate: Candidate, job: JobSpec) -> tuple:
    throughput = candidate.throughput if candidate.throughput is not None else -1.0
    price = candidate.estimated_usd_per_hour
    efficiency = (throughput / price) if price > 0 and throughput > 0 else -1.0
    # Higher efficiency should sort first, so we negate it.
    preferred = _preferred_rank(candidate, job)
    # Tie-breakers keep the result deterministic.
    stable = (candidate.offer.provider, candidate.chip.id, candidate.offer.sku)
    use_rightsize = job.optimize == "balanced" and job.task in {
        "inference",
        "signal-processing",
    }
    rightsize = 0 if (candidate.in_rightsize_window or not use_rightsize) else 1

    if job.optimize == "cost":
        return (price, -throughput, preferred, *stable)
    if job.optimize == "performance":
        return (-throughput, price, preferred, *stable)
    # balanced
    return (rightsize, -efficiency, price, preferred, *stable)


def _money(value: float) -> str:
    return f"${value:.2f}"


def _format_throughput(candidate: Candidate, job: JobSpec) -> str:
    if candidate.throughput is None:
        return "unpublished dense throughput for this precision"
    if job.task == "signal-processing":
        return f"{candidate.throughput:g} GB/s memory bandwidth"
    return f"{candidate.throughput:g} dense {job.precision.upper()} TFLOPS"


def _price_footnote(offer: Offer) -> str:
    bit = offer.price_label()
    if offer.source_url:
        bit += f" ({offer.source_url})"
    return bit


def _memory_sentence(job: JobSpec, required: float) -> str:
    if job.vram_was_estimated():
        from chip_router.job import TASK_MEMORY_MULTIPLIER

        multiplier = TASK_MEMORY_MULTIPLIER[job.task]
        return (
            f"The job did not set vram_gb, so the router estimated a "
            f"{required:g} GB floor as {job.model_size_gb:g} GB model size "
            f"times the {job.task} headroom factor of {multiplier:g}"
        )
    return f"The job set a {required:g} GB memory floor"


def _reason(
    job: JobSpec,
    required: float,
    ranked: list[Candidate],
    rejected: list[Candidate],
) -> str:
    if not ranked:
        parts = [
            f"No catalog offer fits {job.name}.",
            _memory_sentence(job, required) + ".",
        ]
        # Highlight the most useful rejections (memory, then budget).
        interesting = [
            cand
            for cand in rejected
            if cand.skip_reasons and (cand.offer.provider or cand.chip.memory_gb)
        ]
        # De-duplicate chip-level memory messages, keep up to 6.
        seen = set()
        extras = []
        for cand in interesting:
            text = cand.skip_reasons[0]
            if text in seen:
                continue
            seen.add(text)
            extras.append(text)
            if len(extras) >= 6:
                break
        if extras:
            parts.append("Closest misses: " + "; ".join(extras) + ".")
        return " ".join(parts)

    winner = ranked[0]
    offer = winner.offer
    unit = _money(winner.estimated_usd_per_hour) + "/hr"
    if job.count > 1:
        unit += f" for {job.count} devices"
    parts = [
        (
            f"Chose {winner.chip.name} on {offer.provider} "
            f"({offer.sku}) at {unit} for {job.name}, a {job.precision} "
            f"{job.task} job."
        ),
        _memory_sentence(job, required)
        + f"; {winner.chip.name} has {winner.chip.memory_gb:g} GB.",
    ]
    metric = _format_throughput(winner, job)
    if job.optimize == "cost":
        parts.append(
            f"Among offers that fit, this was the lowest price, with {metric}."
        )
    elif job.optimize == "performance":
        parts.append(
            f"Among offers that fit, this had the best published performance "
            f"({metric})."
        )
    else:
        parts.append(
            f"In balanced mode, this had the best published performance "
            f"per dollar ({metric} at {unit})."
        )
        if job.task in {"inference", "signal-processing"} and winner.in_rightsize_window:
            outsiders = [cand for cand in ranked if not cand.in_rightsize_window]
            if outsiders:
                cap = _rightsize_cap(required)
                parts.append(
                    f"Larger chips above the {cap:g} GB right-size window were "
                    f"kept out of the first rank so a much bigger accelerator "
                    f"is not chosen for a smaller job."
                )
    if job.preferred_providers:
        if offer.provider in job.preferred_providers:
            parts.append(
                f"{offer.provider} was in the preferred provider list and "
                f"won any ties."
            )
        else:
            parts.append(
                "No preferred-provider offer both fit and beat this "
                "ranking, so a non-preferred provider was chosen."
            )
    if winner.estimated_usd_for_job is not None:
        parts.append(
            f"Estimated job cost over {job.max_runtime_hours:g} h is "
            f"{_money(winner.estimated_usd_for_job)}."
        )
    parts.append("Price is a " + _price_footnote(offer) + ", not a quote.")
    if ranked[1:]:
        follow = []
        for cand in ranked[1:4]:
            follow.append(
                f"{cand.chip.name} on {cand.offer.provider} at "
                f"{_money(cand.estimated_usd_per_hour)}/hr "
                f"({_format_throughput(cand, job)})"
            )
        parts.append("Runners-up: " + "; ".join(follow) + ".")
    return " ".join(parts)


def rank_candidates(
    catalog: Catalog, job: JobSpec
) -> tuple[list[Candidate], list[Candidate]]:
    accepted, rejected = _collect(catalog, job)
    accepted.sort(key=lambda cand: _sort_key(cand, job))
    return accepted, rejected


def pick_chip(
    job: JobSpec | str,
    catalog: Catalog | str | None = None,
    *,
    raise_if_none: bool = False,
) -> Decision:
    if isinstance(catalog, str) or catalog is None:
        catalog = load_catalog(catalog)
    if isinstance(job, str):
        job = load_job(job)
    required = job.required_vram_gb()
    ranked, rejected = rank_candidates(catalog, job)
    reason = _reason(job, required, ranked, rejected)
    decision = Decision(
        job=job,
        required_vram_gb=required,
        choice=ranked[0] if ranked else None,
        reason=reason,
        runners_up=tuple(ranked[1:4]),
        rejected=tuple(rejected),
        catalog_updated=catalog.updated,
        catalog_path=str(catalog.path or ""),
    )
    if raise_if_none and decision.choice is None:
        raise NoFitError(reason)
    return decision
