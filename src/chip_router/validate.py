"""Validate a chip catalog."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from chip_router.catalog import (
    PRICING_KINDS,
    SOFTWARE_STACKS,
    Catalog,
    Chip,
    Offer,
    load_catalog,
)

ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
URL_RE = re.compile(r"^https?://", re.I)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class ValidationIssue:
    severity: str  # "error" or "warning"
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.severity.upper()}: {self.path}: {self.message}"


def _check_source(chip: Chip, issues: list[ValidationIssue]) -> None:
    if not chip.sources:
        issues.append(
            ValidationIssue("error", chip.id, "chip has no vendor sources")
        )
        return
    for i, source in enumerate(chip.sources):
        path = f"{chip.id}.sources[{i}]"
        if not source.title:
            issues.append(ValidationIssue("error", path, "missing title"))
        if not source.url or not URL_RE.match(source.url):
            issues.append(
                ValidationIssue("error", path, "source url must be http(s)")
            )
        if not source.retrieved or not DATE_RE.match(source.retrieved):
            issues.append(
                ValidationIssue(
                    "error", path, "retrieved date must look like YYYY-MM-DD"
                )
            )


def _check_offer(chip: Chip, offer: Offer, index: int, issues: list[ValidationIssue]) -> None:
    path = f"{chip.id}.offers[{index}]"
    if not offer.provider:
        issues.append(ValidationIssue("error", path, "missing provider"))
    if not offer.sku:
        issues.append(ValidationIssue("error", path, "missing sku"))
    if offer.usd_per_hour is None or offer.usd_per_hour <= 0:
        issues.append(
            ValidationIssue("error", path, "usd_per_hour must be > 0")
        )
    if offer.pricing_kind not in PRICING_KINDS:
        issues.append(
            ValidationIssue(
                "error",
                path,
                f"pricing_kind must be one of {PRICING_KINDS}, got {offer.pricing_kind!r}",
            )
        )
    if offer.pricing_kind == "example" and not offer.source_note:
        issues.append(
            ValidationIssue(
                "error",
                path,
                "example prices must include a source_note that says they are examples",
            )
        )
    if offer.pricing_kind == "published_future_list" and not offer.effective_at:
        issues.append(
            ValidationIssue(
                "error",
                path,
                "published_future_list offers must set effective_at",
            )
        )
    if not offer.source_url or not URL_RE.match(offer.source_url):
        issues.append(
            ValidationIssue(
                "error",
                path,
                "every offer needs a source_url (http or https)",
            )
        )
    if not offer.retrieved or not DATE_RE.match(offer.retrieved):
        issues.append(
            ValidationIssue(
                "error",
                path,
                "retrieved date must look like YYYY-MM-DD",
            )
        )
    if (
        offer.usd_per_second is not None
        and abs(offer.usd_per_second * 3600 - offer.usd_per_hour) > 0.0001
    ):
        issues.append(
            ValidationIssue(
                "error",
                path,
                "usd_per_hour does not match usd_per_second * 3600",
            )
        )


def _check_chip(chip: Chip, issues: list[ValidationIssue]) -> None:
    path = chip.id or "<missing-id>"
    if not chip.id or not ID_RE.match(chip.id):
        issues.append(
            ValidationIssue(
                "error",
                path,
                "id must be lowercase kebab-case (e.g. nvidia-h100-sxm)",
            )
        )
    if not chip.name:
        issues.append(ValidationIssue("error", path, "missing name"))
    if not chip.vendor:
        issues.append(ValidationIssue("error", path, "missing vendor"))
    if chip.software not in SOFTWARE_STACKS:
        issues.append(
            ValidationIssue(
                "error",
                path,
                f"software must be one of {SOFTWARE_STACKS}",
            )
        )
    if chip.memory_gb <= 0:
        issues.append(ValidationIssue("error", path, "memory_gb must be > 0"))
    if chip.memory_bandwidth_gbs <= 0:
        issues.append(
            ValidationIssue("error", path, "memory_bandwidth_gbs must be > 0")
        )
    for field_name, dense, sparse in (
        ("bf16", chip.bf16_tflops_dense, chip.bf16_tflops_sparse),
        ("fp8", chip.fp8_tflops_dense, chip.fp8_tflops_sparse),
    ):
        for label, value in (("dense", dense), ("sparse", sparse)):
            if value is not None and value < 0:
                issues.append(
                    ValidationIssue(
                        "error",
                        path,
                        f"{field_name}_tflops_{label} must be >= 0",
                    )
                )
        if dense is not None and sparse is not None and sparse < dense:
            issues.append(
                ValidationIssue(
                    "error",
                    path,
                    f"{field_name} sparse throughput is lower than dense",
                )
            )
    if chip.fp8_supported and chip.fp8_tflops_dense is None:
        issues.append(
            ValidationIssue(
                "warning",
                path,
                "fp8_supported is true but no dense FP8 throughput is published",
            )
        )
    if not chip.fp8_supported and chip.fp8_tflops_dense:
        issues.append(
            ValidationIssue(
                "error",
                path,
                "fp8_supported is false but fp8_tflops_dense is set",
            )
        )
    if not chip.offers:
        issues.append(
            ValidationIssue(
                "warning",
                path,
                "chip has no provider offers; it cannot be chosen until a priced offer is added",
            )
        )
    _check_source(chip, issues)
    for i, offer in enumerate(chip.offers):
        _check_offer(chip, offer, i, issues)


def validate_catalog(catalog: Catalog | None = None, path: Optional[str] = None) -> list[ValidationIssue]:
    if catalog is None:
        catalog = load_catalog(path)
    issues: list[ValidationIssue] = []
    if catalog.schema_version < 1:
        issues.append(
            ValidationIssue("error", "schema_version", "must be >= 1")
        )
    seen: set[str] = set()
    for chip in catalog.chips:
        if chip.id in seen:
            issues.append(
                ValidationIssue("error", chip.id, "duplicate chip id")
            )
        seen.add(chip.id)
        _check_chip(chip, issues)
    return issues


def has_errors(issues: Iterable[ValidationIssue]) -> bool:
    return any(issue.severity == "error" for issue in issues)
