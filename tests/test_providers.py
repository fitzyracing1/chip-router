import os

from chip_router.job import load_job
from chip_router.providers.base import plan_launch
from chip_router.router import pick_chip
from tests.conftest import EXAMPLES


def test_plan_does_not_require_keys_and_does_not_print_them(catalog, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "super-secret-value")
    decision = pick_chip(load_job(EXAMPLES / "infer-8b.yaml"), catalog)
    plan = plan_launch(decision)
    assert plan is not None
    rendered = plan.render()
    assert "DRY RUN ONLY" in rendered
    assert "super-secret-value" not in rendered
    assert "RUNPOD_API_KEY: set" in rendered
    assert "Not sent" in rendered


def test_plan_reports_missing_key(catalog, monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    decision = pick_chip(load_job(EXAMPLES / "infer-8b.yaml"), catalog)
    plan = plan_launch(decision)
    assert "not set" in plan.render()


def test_adapters_exist_for_core_providers():
    from chip_router.providers import ADAPTERS

    for name in ("runpod", "modal", "lambda", "google_cloud"):
        assert name in ADAPTERS
