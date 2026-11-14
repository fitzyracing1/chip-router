import copy

import yaml

from chip_router.catalog import load_catalog
from chip_router.validate import has_errors, validate_catalog
from tests.conftest import DEFAULT_CATALOG


def test_default_catalog_has_no_errors():
    issues = validate_catalog(path=str(DEFAULT_CATALOG))
    errors = [issue for issue in issues if issue.severity == "error"]
    assert errors == [], errors
    # A10G is expected to warn about having no offers.
    warnings = [issue for issue in issues if issue.severity == "warning"]
    assert any("nvidia-a10g" in issue.path for issue in warnings)


def test_validate_rejects_duplicate_ids(tmp_path, fixture_catalog):
    raw = yaml.safe_load((fixture_catalog.path).read_text())
    raw["chips"].append(copy.deepcopy(raw["chips"][0]))
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    issues = validate_catalog(load_catalog(path))
    assert has_errors(issues)
    assert any("duplicate" in issue.message for issue in issues)


def test_validate_rejects_example_price_without_note(tmp_path, fixture_catalog):
    raw = yaml.safe_load((fixture_catalog.path).read_text())
    raw["chips"][0]["offers"][0]["pricing_kind"] = "example"
    raw["chips"][0]["offers"][0]["source_note"] = ""
    path = tmp_path / "example.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    issues = validate_catalog(load_catalog(path))
    assert any("example" in issue.message for issue in issues if issue.severity == "error")


def test_validate_rejects_mismatched_per_second(tmp_path, fixture_catalog):
    raw = yaml.safe_load((fixture_catalog.path).read_text())
    raw["chips"][0]["offers"][0]["usd_per_second"] = 0.001
    raw["chips"][0]["offers"][0]["usd_per_hour"] = 9.99
    path = tmp_path / "mismatch.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    issues = validate_catalog(load_catalog(path))
    assert any("usd_per_second" in issue.message for issue in issues)
