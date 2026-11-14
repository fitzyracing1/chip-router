"""Load and represent the chip catalog."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

PACKAGE_CATALOG = Path(__file__).resolve().parent / "data" / "chips.yaml"

KNOWN_PROVIDERS = ("runpod", "modal", "lambda", "google_cloud")
PRICING_KINDS = ("public_list", "published_future_list", "example")
SOFTWARE_STACKS = ("cuda", "rocm", "tpu", "any")


@dataclass(frozen=True)
class Source:
    title: str
    url: str
    retrieved: str
    notes: str = ""


@dataclass(frozen=True)
class Offer:
    provider: str
    sku: str
    usd_per_hour: float
    pricing_kind: str
    retrieved: str
    source_url: str
    source_note: str = ""
    billing: str = ""
    pricing_tier: str = ""
    usd_per_second: Optional[float] = None
    effective_at: Optional[str] = None
    instance_size: Optional[str] = None
    region: Optional[str] = None

    def price_label(self) -> str:
        kind = {
            "public_list": "public list price",
            "published_future_list": "published future list price",
            "example": "example price (not a real quote)",
        }.get(self.pricing_kind, self.pricing_kind)
        extra = ""
        if self.effective_at:
            extra = f", effective {self.effective_at}"
        return f"{kind} retrieved {self.retrieved}{extra}"


@dataclass(frozen=True)
class Chip:
    id: str
    name: str
    vendor: str
    architecture: str
    form_factor: str
    software: str
    memory_gb: float
    memory_type: str
    memory_bandwidth_gbs: float
    fp8_supported: bool
    sources: tuple[Source, ...]
    offers: tuple[Offer, ...]
    bf16_tflops_dense: Optional[float] = None
    bf16_tflops_sparse: Optional[float] = None
    fp8_tflops_dense: Optional[float] = None
    fp8_tflops_sparse: Optional[float] = None
    memory_bandwidth_note: str = ""
    memory_note: str = ""
    throughput_derivation: str = ""
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def dense_throughput(self, precision: str) -> Optional[float]:
        """Return dense TFLOPS for the requested precision, if published."""
        precision = precision.lower()
        if precision == "fp8":
            return self.fp8_tflops_dense
        if precision in {"bf16", "fp16"}:
            return self.bf16_tflops_dense
        if precision == "fp32":
            return self.bf16_tflops_dense
        return self.bf16_tflops_dense

    def supports_precision(self, precision: str) -> bool:
        precision = precision.lower()
        if precision == "fp8":
            return bool(self.fp8_supported)
        return True


def default_catalog_path() -> Path:
    env = os.environ.get("CHIP_ROUTER_CATALOG")
    if env:
        return Path(env).expanduser()
    if PACKAGE_CATALOG.is_file():
        return PACKAGE_CATALOG
    raise FileNotFoundError(
        "No chip catalog found. Set CHIP_ROUTER_CATALOG or install the package."
    )


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _parse_source(item: dict[str, Any]) -> Source:
    return Source(
        title=str(item.get("title") or ""),
        url=str(item.get("url") or ""),
        retrieved=str(item.get("retrieved") or ""),
        notes=str(item.get("notes") or "").strip(),
    )


def _parse_offer(item: dict[str, Any]) -> Offer:
    usd_per_second = _as_float(item.get("usd_per_second"))
    usd_per_hour = item.get("usd_per_hour")
    if usd_per_hour is None and usd_per_second is not None:
        usd_per_hour = usd_per_second * 3600
    if usd_per_hour is None:
        raise ValueError(f"offer {item.get('sku')!r} is missing usd_per_hour")
    return Offer(
        provider=str(item.get("provider") or "").strip(),
        sku=str(item.get("sku") or "").strip(),
        usd_per_hour=float(usd_per_hour),
        pricing_kind=str(item.get("pricing_kind") or "").strip(),
        retrieved=str(item.get("retrieved") or "").strip(),
        source_url=str(item.get("source_url") or "").strip(),
        source_note=str(item.get("source_note") or "").strip(),
        billing=str(item.get("billing") or "").strip(),
        pricing_tier=str(item.get("pricing_tier") or "").strip(),
        usd_per_second=usd_per_second,
        effective_at=item.get("effective_at"),
        instance_size=item.get("instance_size"),
        region=item.get("region"),
    )


def _parse_chip(item: dict[str, Any]) -> Chip:
    sources = tuple(_parse_source(s) for s in item.get("sources") or [])
    offers = tuple(_parse_offer(o) for o in item.get("offers") or [])
    return Chip(
        id=str(item.get("id") or "").strip(),
        name=str(item.get("name") or "").strip(),
        vendor=str(item.get("vendor") or "").strip(),
        architecture=str(item.get("architecture") or "").strip(),
        form_factor=str(item.get("form_factor") or "").strip(),
        software=str(item.get("software") or "any").strip(),
        memory_gb=float(item.get("memory_gb")),
        memory_type=str(item.get("memory_type") or "").strip(),
        memory_bandwidth_gbs=float(item.get("memory_bandwidth_gbs")),
        fp8_supported=bool(item.get("fp8_supported")),
        sources=sources,
        offers=offers,
        bf16_tflops_dense=_as_float(item.get("bf16_tflops_dense")),
        bf16_tflops_sparse=_as_float(item.get("bf16_tflops_sparse")),
        fp8_tflops_dense=_as_float(item.get("fp8_tflops_dense")),
        fp8_tflops_sparse=_as_float(item.get("fp8_tflops_sparse")),
        memory_bandwidth_note=str(item.get("memory_bandwidth_note") or "").strip(),
        memory_note=str(item.get("memory_note") or "").strip(),
        throughput_derivation=str(item.get("throughput_derivation") or "").strip(),
        raw=item,
    )


@dataclass
class Catalog:
    schema_version: int
    updated: str
    notes: str
    chips: tuple[Chip, ...]
    path: Optional[Path] = None

    def get(self, chip_id: str) -> Chip:
        for chip in self.chips:
            if chip.id == chip_id:
                return chip
        raise KeyError(f"unknown chip id: {chip_id}")

    def __iter__(self) -> Iterable[Chip]:
        return iter(self.chips)

    def __len__(self) -> int:
        return len(self.chips)


def load_catalog(path: Optional[os.PathLike[str] | str] = None) -> Catalog:
    catalog_path = Path(path) if path else default_catalog_path()
    with catalog_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{catalog_path} is not a YAML mapping")
    chips = tuple(_parse_chip(item) for item in data.get("chips") or [])
    return Catalog(
        schema_version=int(data.get("schema_version") or 1),
        updated=str(data.get("updated") or ""),
        notes=str(data.get("notes") or "").strip(),
        chips=chips,
        path=catalog_path,
    )
