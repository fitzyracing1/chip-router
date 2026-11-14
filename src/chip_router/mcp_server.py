"""Optional tiny MCP server exposing pick_chip over stdio.

The MCP Python SDK is an optional extra (`pip install chip-router[mcp]`).
This module also works as a plain library helper used by tests.
"""

from __future__ import annotations

import json
from typing import Optional

from chip_router.job import job_from_dict, load_job
from chip_router.router import pick_chip


def pick_chip_tool(
    job_yaml: Optional[str] = None,
    job: Optional[dict] = None,
    catalog: Optional[str] = None,
) -> dict:
    """Return a JSON-serializable pick for a job YAML string or mapping."""
    if job is not None and job_yaml:
        raise ValueError("pass job or job_yaml, not both")
    if job is not None:
        spec = job_from_dict(job)
    elif job_yaml:
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(job_yaml)
            path = Path(handle.name)
        try:
            spec = load_job(path)
        finally:
            path.unlink(missing_ok=True)
    else:
        raise ValueError("job_yaml or job is required")
    decision = pick_chip(spec, catalog=catalog)
    choice = decision.choice
    return {
        "job": decision.job.name,
        "required_vram_gb": decision.required_vram_gb,
        "chip": choice.chip.id if choice else None,
        "chip_name": choice.chip.name if choice else None,
        "provider": choice.offer.provider if choice else None,
        "sku": choice.offer.sku if choice else None,
        "est_cost_usd_per_hour": choice.estimated_usd_per_hour if choice else None,
        "est_cost_usd_for_job": choice.estimated_usd_for_job if choice else None,
        "pricing_kind": choice.offer.pricing_kind if choice else None,
        "reason": decision.reason,
        "runners_up": [
            {
                "chip": item.chip.id,
                "provider": item.offer.provider,
                "sku": item.offer.sku,
                "usd_per_hour": item.estimated_usd_per_hour,
            }
            for item in decision.runners_up
        ],
    }


def serve_mcp(catalog_path: Optional[str] = None) -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit(
            "The MCP extra is not installed. Run: pip install 'chip-router[mcp]'"
        ) from exc

    server = FastMCP("chip-router")

    @server.tool()
    def pick_chip(job_yaml: str) -> str:  # noqa: F811 — MCP tool name
        """Pick a rented AI chip and provider for a job YAML document."""
        result = pick_chip_tool(job_yaml=job_yaml, catalog=catalog_path)
        return json.dumps(result, indent=2, sort_keys=True)

    server.run()
