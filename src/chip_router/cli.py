"""Command-line interface for chip-router."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from chip_router import __version__
from chip_router.catalog import default_catalog_path, load_catalog
from chip_router.job import load_job
from chip_router.log import append_decision, default_log_path, read_log
from chip_router.providers.base import plan_launch
from chip_router.router import Decision, pick_chip
from chip_router.validate import has_errors, validate_catalog


def _add_catalog_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--catalog",
        help="Path to chips.yaml (default: packaged catalog or CHIP_ROUTER_CATALOG)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chip-router",
        description=(
            "Pick the AI chip each job runs on, on purpose. "
            "Provider adapters are dry-run only and never rent a GPU."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"chip-router {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pick = sub.add_parser("pick", help="Choose a chip and provider for a job spec")
    pick.add_argument("job", help="Path to a job YAML file")
    _add_catalog_flag(pick)
    pick.add_argument(
        "--plan",
        action="store_true",
        help="Also print a dry-run launch plan (nothing is rented)",
    )
    pick.add_argument(
        "--no-log",
        action="store_true",
        help="Do not append this decision to the JSONL log",
    )
    pick.add_argument(
        "--log",
        dest="log_path",
        help="JSONL log path (default: ./chip-router-log.jsonl or CHIP_ROUTER_LOG)",
    )
    pick.add_argument(
        "--json",
        action="store_true",
        help="Print the decision as JSON instead of text",
    )
    pick.add_argument(
        "--fail-if-none",
        action="store_true",
        help="Exit with status 2 if no offer fits",
    )

    list_cmd = sub.add_parser("list", help="List chips and offers in the catalog")
    _add_catalog_flag(list_cmd)
    list_cmd.add_argument(
        "--json", action="store_true", help="Print catalog entries as JSON"
    )

    log_cmd = sub.add_parser("log", help="Show the decision log")
    log_cmd.add_argument(
        "--log",
        dest="log_path",
        help="JSONL log path (default: ./chip-router-log.jsonl or CHIP_ROUTER_LOG)",
    )
    log_cmd.add_argument(
        "--json", action="store_true", help="Print raw JSONL records as a JSON array"
    )
    log_cmd.add_argument(
        "-n",
        type=int,
        default=20,
        help="Show only the last N records (default: 20, 0 for all)",
    )

    validate = sub.add_parser("validate", help="Validate a chip catalog")
    validate.add_argument(
        "catalog_path",
        nargs="?",
        help="Path to chips.yaml (default: packaged catalog or CHIP_ROUTER_CATALOG)",
    )

    mcp = sub.add_parser(
        "mcp",
        help="Run a tiny MCP server exposing pick_chip over stdio",
    )
    _add_catalog_flag(mcp)
    return parser


def _print_decision(decision: Decision) -> None:
    print(decision.reason)
    if decision.choice:
        choice = decision.choice
        print()
        print(f"chip:       {choice.chip.name} ({choice.chip.id})")
        print(f"provider:   {choice.offer.provider}")
        print(f"sku:        {choice.offer.sku}")
        print(f"est. cost:  ${choice.estimated_usd_per_hour:.4f}/hr")
        if choice.estimated_usd_for_job is not None:
            print(f"            ${choice.estimated_usd_for_job:.4f} estimated for the job")
        print(f"pricing:    {choice.offer.price_label()}")
        if choice.offer.source_url:
            print(f"source:     {choice.offer.source_url}")


def _decision_json(decision: Decision) -> dict:
    record = decision.as_log_record(timestamp="")
    record.pop("timestamp", None)
    return record


def cmd_pick(args: argparse.Namespace) -> int:
    job = load_job(args.job)
    decision = pick_chip(job, catalog=args.catalog)
    if not args.no_log:
        append_decision(decision, path=args.log_path)
    if args.json:
        print(json.dumps(_decision_json(decision), indent=2, sort_keys=True))
    else:
        _print_decision(decision)
        if args.plan:
            plan = plan_launch(decision)
            print()
            if plan is None:
                print("No dry-run plan: no chip was chosen.")
            else:
                print(plan.render())
    if args.fail_if_none and decision.choice is None:
        return 2
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.catalog)
    if args.json:
        payload = []
        for chip in catalog.chips:
            payload.append(
                {
                    "id": chip.id,
                    "name": chip.name,
                    "memory_gb": chip.memory_gb,
                    "memory_bandwidth_gbs": chip.memory_bandwidth_gbs,
                    "bf16_tflops_dense": chip.bf16_tflops_dense,
                    "fp8_tflops_dense": chip.fp8_tflops_dense,
                    "fp8_supported": chip.fp8_supported,
                    "software": chip.software,
                    "offers": [
                        {
                            "provider": offer.provider,
                            "sku": offer.sku,
                            "usd_per_hour": offer.usd_per_hour,
                            "pricing_kind": offer.pricing_kind,
                            "source_url": offer.source_url,
                            "retrieved": offer.retrieved,
                        }
                        for offer in chip.offers
                    ],
                }
            )
        print(json.dumps(payload, indent=2))
        return 0
    path = catalog.path or default_catalog_path()
    print(f"Catalog: {path} (updated {catalog.updated})")
    print()
    for chip in catalog.chips:
        fp8 = (
            f"{chip.fp8_tflops_dense:g} dense FP8 TFLOPS"
            if chip.fp8_tflops_dense is not None
            else "no published dense FP8"
        )
        bf16 = (
            f"{chip.bf16_tflops_dense:g} dense BF16 TFLOPS"
            if chip.bf16_tflops_dense is not None
            else "no published dense BF16"
        )
        print(
            f"{chip.id}: {chip.name} — {chip.memory_gb:g} GB {chip.memory_type}, "
            f"{chip.memory_bandwidth_gbs:g} GB/s, {bf16}, {fp8}"
        )
        if not chip.offers:
            print("  (no priced offers)")
            continue
        for offer in chip.offers:
            future = (
                f", effective {offer.effective_at}" if offer.effective_at else ""
            )
            print(
                f"  {offer.provider}: {offer.sku} ${offer.usd_per_hour:g}/hr "
                f"[{offer.pricing_kind}{future}, {offer.retrieved}]"
            )
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    records = read_log(args.log_path)
    if args.n and args.n > 0:
        records = records[-args.n :]
    if args.json:
        print(json.dumps(records, indent=2, sort_keys=True))
        return 0
    if not records:
        print(f"No decisions logged at {args.log_path or default_log_path()}")
        return 0
    for record in records:
        when = record.get("timestamp", "")
        job = record.get("job", "")
        chip = record.get("chip_name") or record.get("chip") or "no fit"
        provider = record.get("provider") or "-"
        cost = record.get("est_cost_usd_per_hour")
        cost_s = f"${cost:.2f}/hr" if isinstance(cost, (int, float)) else "-"
        print(f"{when}  {job}: {chip} on {provider} ({cost_s})")
        if record.get("reason"):
            print(f"  {record['reason']}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    path = args.catalog_path
    catalog = load_catalog(path)
    issues = validate_catalog(catalog)
    shown = catalog.path or path or default_catalog_path()
    if not issues:
        print(f"{shown}: OK ({len(catalog)} chips, no issues)")
        return 0
    errors = sum(1 for issue in issues if issue.severity == "error")
    warnings = len(issues) - errors
    for issue in issues:
        print(issue)
    print(
        f"{shown}: {errors} error(s), {warnings} warning(s) across {len(catalog)} chips"
    )
    return 1 if has_errors(issues) else 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from chip_router.mcp_server import serve_mcp

    serve_mcp(catalog_path=args.catalog)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    commands = {
        "pick": cmd_pick,
        "list": cmd_list,
        "log": cmd_log,
        "validate": cmd_validate,
        "mcp": cmd_mcp,
    }
    try:
        return commands[args.command](args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
