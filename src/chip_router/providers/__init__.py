"""Dry-run provider adapters. They never rent a GPU or call a paid API."""

from chip_router.providers.base import DryRunPlan, plan_launch
from chip_router.providers.lambda_labs import LambdaAdapter
from chip_router.providers.modal import ModalAdapter
from chip_router.providers.runpod import RunPodAdapter
from chip_router.providers.google_cloud import GoogleCloudAdapter

ADAPTERS = {
    "runpod": RunPodAdapter(),
    "modal": ModalAdapter(),
    "lambda": LambdaAdapter(),
    "google_cloud": GoogleCloudAdapter(),
}

__all__ = [
    "ADAPTERS",
    "DryRunPlan",
    "GoogleCloudAdapter",
    "LambdaAdapter",
    "ModalAdapter",
    "RunPodAdapter",
    "plan_launch",
]
