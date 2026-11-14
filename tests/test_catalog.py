from chip_router.catalog import load_catalog
from tests.conftest import DEFAULT_CATALOG

REQUIRED_IDS = {
    "nvidia-h100-sxm",
    "nvidia-h200-sxm",
    "nvidia-b200-sxm",
    "nvidia-a100-80gb-sxm",
    "nvidia-l40s",
    "nvidia-a10g",
    "nvidia-rtx-4090",
    "amd-mi300x",
    "google-tpu-v5p",
}


def test_default_catalog_loads_required_chips():
    catalog = load_catalog(DEFAULT_CATALOG)
    ids = {chip.id for chip in catalog.chips}
    assert REQUIRED_IDS <= ids
    assert catalog.updated


def test_a10g_has_specs_but_no_offers():
    catalog = load_catalog(DEFAULT_CATALOG)
    chip = catalog.get("nvidia-a10g")
    assert chip.memory_gb == 24
    assert chip.memory_bandwidth_gbs == 600
    assert chip.bf16_tflops_dense == 70
    assert chip.fp8_supported is False
    assert chip.offers == ()


def test_a100_has_no_fp8():
    catalog = load_catalog(DEFAULT_CATALOG)
    chip = catalog.get("nvidia-a100-80gb-sxm")
    assert chip.fp8_supported is False
    assert chip.fp8_tflops_dense is None
    assert chip.memory_bandwidth_gbs == 2039


def test_b200_bf16_is_explicitly_unpublished():
    catalog = load_catalog(DEFAULT_CATALOG)
    chip = catalog.get("nvidia-b200-sxm")
    assert chip.memory_gb == 180
    assert chip.memory_bandwidth_gbs == 8000
    assert chip.fp8_tflops_dense == 4500
    assert chip.bf16_tflops_dense is None


def test_modal_hourly_matches_per_second():
    catalog = load_catalog(DEFAULT_CATALOG)
    for chip in catalog.chips:
        for offer in chip.offers:
            if offer.usd_per_second is not None:
                assert abs(offer.usd_per_hour - offer.usd_per_second * 3600) < 1e-6


def test_every_offer_has_a_source_and_kind():
    catalog = load_catalog(DEFAULT_CATALOG)
    kinds = {"public_list", "published_future_list", "example"}
    for chip in catalog.chips:
        assert chip.sources
        for offer in chip.offers:
            assert offer.pricing_kind in kinds
            assert offer.source_url.startswith("http")
            assert offer.retrieved
            assert offer.usd_per_hour > 0
            if offer.pricing_kind == "example":
                raise AssertionError(
                    f"{chip.id} {offer.provider} presents an example price; "
                    "the default catalog should not need invented numbers"
                )
