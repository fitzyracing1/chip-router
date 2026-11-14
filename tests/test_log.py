from chip_router.log import append_decision, read_log
from chip_router.router import pick_chip
from tests.conftest import make_job


def test_append_and_read_round_trip(fixture_catalog, tmp_path):
    decision = pick_chip(make_job(vram_gb=16, optimize="cost", precision="fp8"), fixture_catalog)
    path = tmp_path / "nested" / "log.jsonl"
    append_decision(
        decision,
        path=path,
        timestamp="2026-11-13T12:00:00+00:00",
        actual={"usd": 0.95},
    )
    records = read_log(path)
    assert len(records) == 1
    record = records[0]
    assert record["job"] == "test-job"
    assert record["chip"] == "small-gpu"
    assert record["provider"] == "runpod"
    assert record["est_cost_usd_per_hour"] == 1.0
    assert record["reason"]
    assert record["timestamp"] == "2026-11-13T12:00:00+00:00"
    assert record["actual"] == {"usd": 0.95}


def test_empty_log_is_empty_list(tmp_path):
    assert read_log(tmp_path / "missing.jsonl") == []
