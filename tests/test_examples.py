from chip_router.job import load_job
from chip_router.router import pick_chip
from tests.conftest import EXAMPLES


def test_infer_8b_picks_rtx_4090_on_runpod(catalog):
    decision = pick_chip(load_job(EXAMPLES / "infer-8b.yaml"), catalog)
    assert decision.choice is not None
    assert decision.choice.chip.id == "nvidia-rtx-4090"
    assert decision.choice.offer.provider == "runpod"
    assert decision.choice.estimated_usd_per_hour == 0.89
    assert "4090" in decision.reason


def test_infer_70b_picks_mi300x(catalog):
    decision = pick_chip(load_job(EXAMPLES / "infer-70b.yaml"), catalog)
    assert decision.choice is not None
    # 160 GB floor: only MI300X (192) and B200 (180) fit. MI300X is cheaper.
    assert decision.choice.chip.id == "amd-mi300x"
    assert decision.choice.offer.provider == "runpod"
    assert decision.choice.estimated_usd_per_hour == 2.39
    runners = {item.chip.id for item in decision.runners_up}
    assert "nvidia-b200-sxm" in runners


def test_finetune_13b_picks_cuda_48gb_class(catalog):
    decision = pick_chip(load_job(EXAMPLES / "finetune-13b.yaml"), catalog)
    assert decision.choice is not None
    # 32.5 GB floor and software=cuda: L40S is the cheapest CUDA fit.
    assert decision.choice.chip.id == "nvidia-l40s"
    assert decision.choice.offer.provider == "runpod"
    assert decision.choice.chip.software == "cuda"
    assert decision.choice.chip.memory_gb >= 32.5


def test_signal_processing_picks_rightsized_cuda_card(catalog):
    decision = pick_chip(load_job(EXAMPLES / "signal-processing.yaml"), catalog)
    assert decision.choice is not None
    # 16 GB, cuda, $2/hr cap, balanced + right-size: RTX 4090 at $0.89
    # has 1008 GB/s and is inside the window. MI300X is outside it and
    # would also miss the $2 cap anyway ($2.39).
    assert decision.choice.chip.id == "nvidia-rtx-4090"
    assert decision.choice.offer.provider == "runpod"
    assert decision.choice.estimated_usd_per_hour <= 2.00
    assert decision.choice.chip.software == "cuda"
