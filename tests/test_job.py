import pytest

from chip_router.job import TASK_MEMORY_MULTIPLIER, job_from_dict, load_job
from tests.conftest import EXAMPLES


def test_load_example_jobs():
    for name in (
        "infer-8b.yaml",
        "infer-70b.yaml",
        "finetune-13b.yaml",
        "signal-processing.yaml",
    ):
        job = load_job(EXAMPLES / name)
        assert job.required_vram_gb() > 0


def test_vram_estimate_uses_task_multiplier():
    job = job_from_dict(
        {
            "name": "est",
            "task": "fine-tune",
            "precision": "bf16",
            "model_size_gb": 13,
        }
    )
    assert job.vram_was_estimated()
    assert job.required_vram_gb() == pytest.approx(13 * TASK_MEMORY_MULTIPLIER["fine-tune"])


def test_explicit_vram_wins_over_model_size():
    job = job_from_dict(
        {
            "name": "explicit",
            "task": "inference",
            "model_size_gb": 140,
            "vram_gb": 160,
        }
    )
    assert job.required_vram_gb() == 160
    assert not job.vram_was_estimated()


def test_rejects_unknown_task():
    with pytest.raises(ValueError, match="task must be"):
        job_from_dict({"name": "x", "task": "mining"})


def test_missing_memory_raises():
    job = job_from_dict({"name": "x", "task": "inference"})
    with pytest.raises(ValueError, match="vram_gb"):
        job.required_vram_gb()
