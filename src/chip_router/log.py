"""JSONL decision log."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from chip_router.router import Decision

DEFAULT_LOG_NAME = "chip-router-log.jsonl"


def default_log_path() -> Path:
    env = os.environ.get("CHIP_ROUTER_LOG")
    if env:
        return Path(env).expanduser()
    return Path.cwd() / DEFAULT_LOG_NAME


def append_decision(
    decision: Decision,
    path: Optional[os.PathLike[str] | str] = None,
    actual: Optional[dict[str, Any]] = None,
    timestamp: Optional[str] = None,
) -> Path:
    log_path = Path(path) if path else default_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    record = decision.as_log_record(timestamp=timestamp, actual=actual)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return log_path


def read_log(path: Optional[os.PathLike[str] | str] = None) -> list[dict[str, Any]]:
    log_path = Path(path) if path else default_log_path()
    if not log_path.is_file():
        return []
    records = []
    with log_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{log_path}:{line_no} is not valid JSON: {exc}"
                ) from exc
    return records
