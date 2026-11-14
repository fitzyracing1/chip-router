from chip_router.mcp_server import pick_chip_tool
from tests.conftest import DEFAULT_CATALOG


def test_pick_chip_tool_from_mapping():
    result = pick_chip_tool(
        job={
            "name": "tool-job",
            "task": "inference",
            "precision": "fp8",
            "model_size_gb": 8,
            "optimize": "balanced",
        },
        catalog=str(DEFAULT_CATALOG),
    )
    assert result["chip"] == "nvidia-rtx-4090"
    assert result["provider"] == "runpod"
    assert "reason" in result


def test_pick_chip_tool_from_yaml_string():
    yaml_text = "\n".join(
        [
            "name: yaml-job",
            "task: inference",
            "precision: fp8",
            "model_size_gb: 8",
        ]
    )
    result = pick_chip_tool(job_yaml=yaml_text, catalog=str(DEFAULT_CATALOG))
    assert result["job"] == "yaml-job"
    assert result["chip"] == "nvidia-rtx-4090"
