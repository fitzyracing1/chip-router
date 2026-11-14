import json
from pathlib import Path

from chip_router.cli import main
from tests.conftest import DEFAULT_CATALOG, EXAMPLES


def test_validate_cli_ok():
    assert main(["validate", str(DEFAULT_CATALOG)]) == 0


def test_list_cli(capsys):
    assert main(["list", "--catalog", str(DEFAULT_CATALOG)]) == 0
    out = capsys.readouterr().out
    assert "nvidia-h100-sxm" in out
    assert "amd-mi300x" in out


def test_pick_cli_json_and_log(tmp_path, capsys):
    log_path = tmp_path / "decisions.jsonl"
    code = main(
        [
            "pick",
            str(EXAMPLES / "infer-8b.yaml"),
            "--catalog",
            str(DEFAULT_CATALOG),
            "--json",
            "--log",
            str(log_path),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["chip"] == "nvidia-rtx-4090"
    assert log_path.is_file()
    assert main(["log", "--log", str(log_path), "--json", "-n", "0"]) == 0


def test_pick_cli_plan_is_dry_run(capsys):
    code = main(
        [
            "pick",
            str(EXAMPLES / "signal-processing.yaml"),
            "--catalog",
            str(DEFAULT_CATALOG),
            "--plan",
            "--no-log",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "DRY RUN ONLY" in out
    assert "nothing was rented" in out.lower() or "DRY RUN" in out
    assert "Not sent" in out


def test_pick_fail_if_none(tmp_path):
    job = tmp_path / "tight.yaml"
    job.write_text(
        "\n".join(
            [
                "name: too-tight",
                "task: inference",
                "precision: bf16",
                "vram_gb: 16",
                "budget_usd_per_hour: 0.01",
            ]
        ),
        encoding="utf-8",
    )
    code = main(
        [
            "pick",
            str(job),
            "--catalog",
            str(DEFAULT_CATALOG),
            "--no-log",
            "--fail-if-none",
        ]
    )
    assert code == 2
