"""chip-router: pick the AI chip each job runs on, on purpose."""

from chip_router.catalog import Catalog, Chip, Offer, load_catalog
from chip_router.job import JobSpec, load_job
from chip_router.log import append_decision, read_log
from chip_router.router import Decision, pick_chip
from chip_router.validate import ValidationIssue, validate_catalog

__version__ = "0.1.0"
__all__ = [
    "Catalog",
    "Chip",
    "Decision",
    "JobSpec",
    "Offer",
    "ValidationIssue",
    "append_decision",
    "load_catalog",
    "load_job",
    "pick_chip",
    "read_log",
    "validate_catalog",
    "__version__",
]
