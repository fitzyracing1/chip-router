from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from chip_router.catalog import load_catalog
from chip_router.job import job_from_dict

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "src" / "chip_router" / "data" / "chips.yaml"
EXAMPLES = ROOT / "examples"


@pytest.fixture
def catalog():
    return load_catalog(DEFAULT_CATALOG)


@pytest.fixture
def fixture_catalog(tmp_path: Path):
    data = {
        "schema_version": 1,
        "updated": "2026-11-13",
        "notes": "tiny fixture catalog for unit tests",
        "chips": [
            {
                "id": "small-gpu",
                "name": "Small GPU",
                "vendor": "Test",
                "architecture": "test",
                "form_factor": "PCIe",
                "software": "cuda",
                "memory_gb": 24,
                "memory_type": "GDDR",
                "memory_bandwidth_gbs": 1000,
                "bf16_tflops_dense": 100,
                "bf16_tflops_sparse": 200,
                "fp8_tflops_dense": 200,
                "fp8_tflops_sparse": 400,
                "fp8_supported": True,
                "sources": [
                    {
                        "title": "fixture",
                        "url": "https://example.com/small",
                        "retrieved": "2026-11-13",
                    }
                ],
                "offers": [
                    {
                        "provider": "runpod",
                        "sku": "small",
                        "usd_per_hour": 1.00,
                        "pricing_kind": "public_list",
                        "retrieved": "2026-11-13",
                        "source_url": "https://example.com/price",
                    }
                ],
            },
            {
                "id": "mid-gpu",
                "name": "Mid GPU",
                "vendor": "Test",
                "architecture": "test",
                "form_factor": "PCIe",
                "software": "cuda",
                "memory_gb": 48,
                "memory_type": "GDDR",
                "memory_bandwidth_gbs": 800,
                "bf16_tflops_dense": 300,
                "bf16_tflops_sparse": 600,
                "fp8_tflops_dense": 600,
                "fp8_tflops_sparse": 1200,
                "fp8_supported": True,
                "sources": [
                    {
                        "title": "fixture",
                        "url": "https://example.com/mid",
                        "retrieved": "2026-11-13",
                    }
                ],
                "offers": [
                    {
                        "provider": "runpod",
                        "sku": "mid",
                        "usd_per_hour": 2.00,
                        "pricing_kind": "public_list",
                        "retrieved": "2026-11-13",
                        "source_url": "https://example.com/price",
                    },
                    {
                        "provider": "modal",
                        "sku": "mid",
                        "usd_per_hour": 2.50,
                        "pricing_kind": "public_list",
                        "retrieved": "2026-11-13",
                        "source_url": "https://example.com/price",
                    },
                ],
            },
            {
                "id": "huge-gpu",
                "name": "Huge GPU",
                "vendor": "Test",
                "architecture": "test",
                "form_factor": "SXM",
                "software": "rocm",
                "memory_gb": 192,
                "memory_type": "HBM",
                "memory_bandwidth_gbs": 5000,
                "bf16_tflops_dense": 1000,
                "bf16_tflops_sparse": 2000,
                "fp8_tflops_dense": 2000,
                "fp8_tflops_sparse": 4000,
                "fp8_supported": True,
                "sources": [
                    {
                        "title": "fixture",
                        "url": "https://example.com/huge",
                        "retrieved": "2026-11-13",
                    }
                ],
                "offers": [
                    {
                        "provider": "runpod",
                        "sku": "huge",
                        "usd_per_hour": 3.00,
                        "pricing_kind": "public_list",
                        "retrieved": "2026-11-13",
                        "source_url": "https://example.com/price",
                    }
                ],
            },
            {
                "id": "no-fp8-gpu",
                "name": "No FP8 GPU",
                "vendor": "Test",
                "architecture": "test",
                "form_factor": "PCIe",
                "software": "cuda",
                "memory_gb": 80,
                "memory_type": "HBM",
                "memory_bandwidth_gbs": 2000,
                "bf16_tflops_dense": 300,
                "bf16_tflops_sparse": 600,
                "fp8_tflops_dense": None,
                "fp8_supported": False,
                "sources": [
                    {
                        "title": "fixture",
                        "url": "https://example.com/nofp8",
                        "retrieved": "2026-11-13",
                    }
                ],
                "offers": [
                    {
                        "provider": "runpod",
                        "sku": "nofp8",
                        "usd_per_hour": 0.50,
                        "pricing_kind": "public_list",
                        "retrieved": "2026-11-13",
                        "source_url": "https://example.com/price",
                    }
                ],
            },
        ],
    }
    path = tmp_path / "chips.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_catalog(path)


def make_job(**kwargs):
    data = {
        "name": "test-job",
        "task": "inference",
        "precision": "bf16",
        "vram_gb": 16,
        "optimize": "balanced",
    }
    data.update(kwargs)
    return job_from_dict(data)
