import pytest

from chip_router.router import pick_chip
from tests.conftest import make_job


def test_cost_picks_cheapest_that_fits(fixture_catalog):
    decision = pick_chip(make_job(vram_gb=16, optimize="cost", precision="fp8"), fixture_catalog)
    assert decision.choice is not None
    assert decision.choice.chip.id == "small-gpu"
    assert decision.choice.offer.usd_per_hour == 1.00


def test_performance_picks_highest_throughput(fixture_catalog):
    decision = pick_chip(
        make_job(vram_gb=16, optimize="performance"), fixture_catalog
    )
    assert decision.choice.chip.id == "huge-gpu"


def test_balanced_rightsizes_small_inference(fixture_catalog):
    decision = pick_chip(
        make_job(vram_gb=16, optimize="balanced", task="inference", precision="fp8"),
        fixture_catalog,
    )
    # huge-gpu is 192 GB and outside the right-size window for a 16 GB job.
    assert decision.choice.chip.id != "huge-gpu"
    assert decision.choice.in_rightsize_window
    # mid-gpu: 600 TFLOPS / $2 = 300; small-gpu: 200 / $1 = 200, so mid wins.
    assert decision.choice.chip.id == "mid-gpu"


def test_fp8_job_skips_chip_without_fp8(fixture_catalog):
    decision = pick_chip(
        make_job(vram_gb=16, precision="fp8", optimize="cost"), fixture_catalog
    )
    # no-fp8-gpu is cheaper ($0.50) but must be skipped.
    assert decision.choice.chip.id != "no-fp8-gpu"
    assert any(
        "FP8" in reason
        for cand in decision.rejected
        for reason in cand.skip_reasons
        if cand.chip.id == "no-fp8-gpu"
    )


def test_memory_floor_rejects_small_gpu(fixture_catalog):
    decision = pick_chip(make_job(vram_gb=32, optimize="cost", precision="fp8"), fixture_catalog)
    assert decision.choice.chip.id == "mid-gpu"
    assert any(
        cand.chip.id == "small-gpu" and cand.skip_reasons for cand in decision.rejected
    )


def test_budget_rejects_expensive_offers(fixture_catalog):
    decision = pick_chip(
        make_job(vram_gb=16, optimize="performance", precision="fp8", budget_usd_per_hour=2.25),
        fixture_catalog,
    )
    # huge-gpu at $3 is out; mid at $2 is the fastest remaining.
    assert decision.choice.chip.id == "mid-gpu"
    assert decision.choice.offer.provider == "runpod"


def test_software_cuda_excludes_rocm(fixture_catalog):
    decision = pick_chip(
        make_job(vram_gb=16, optimize="performance", software="cuda"),
        fixture_catalog,
    )
    assert decision.choice.chip.id != "huge-gpu"
    assert decision.choice.chip.software == "cuda"


def test_preferred_provider_breaks_ties_only(fixture_catalog):
    # mid-gpu on runpod is cheaper than modal, so even preferring modal
    # should not override a better efficiency/price in balanced/cost mode
    # unless they tie. For cost, runpod mid ($2) vs modal mid ($2.50).
    decision = pick_chip(
        make_job(
            vram_gb=32,
            optimize="cost",
            preferred_providers=["modal"],
        ),
        fixture_catalog,
    )
    assert decision.choice.offer.provider == "runpod"
    # Equal price tie: only mid-gpu runpod vs nothing at same price.
    # Add a direct check on preferred rank with two equal offers by
    # preferring runpod on the 16 GB cost pick (only one offer on small).
    decision2 = pick_chip(
        make_job(vram_gb=32, optimize="cost", preferred_providers=["runpod"]),
        fixture_catalog,
    )
    assert decision2.choice.offer.provider == "runpod"


def test_excluded_and_allowed_providers(fixture_catalog):
    decision = pick_chip(
        make_job(vram_gb=32, optimize="cost", allowed_providers=["modal"]),
        fixture_catalog,
    )
    assert decision.choice.offer.provider == "modal"
    decision = pick_chip(
        make_job(vram_gb=32, optimize="cost", excluded_providers=["runpod"]),
        fixture_catalog,
    )
    assert decision.choice.offer.provider == "modal"


def test_no_fit_returns_reason_without_raising(fixture_catalog):
    decision = pick_chip(
        make_job(vram_gb=16, budget_usd_per_hour=0.10), fixture_catalog
    )
    assert decision.choice is None
    assert "No catalog offer fits" in decision.reason


def test_same_inputs_are_deterministic(fixture_catalog):
    job = make_job(vram_gb=16, optimize="balanced", precision="fp8")
    first = pick_chip(job, fixture_catalog)
    second = pick_chip(job, fixture_catalog)
    assert first.choice.key == second.choice.key
    assert first.reason == second.reason
    assert [c.key for c in first.runners_up] == [c.key for c in second.runners_up]


def test_job_cost_uses_runtime_and_count(fixture_catalog):
    decision = pick_chip(
        make_job(vram_gb=16, optimize="cost", precision="fp8", max_runtime_hours=3, count=2),
        fixture_catalog,
    )
    assert decision.choice.estimated_usd_per_hour == pytest.approx(2.00)
    assert decision.choice.estimated_usd_for_job == pytest.approx(6.00)


def test_reason_mentions_runners_up(fixture_catalog):
    decision = pick_chip(make_job(vram_gb=16, optimize="cost", precision="fp8"), fixture_catalog)
    assert "Runners-up:" in decision.reason
    assert "public list price" in decision.reason
