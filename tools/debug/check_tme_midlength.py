"""Check depth-27 TME mid-length smoke logs against the resolution runbook."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLD = 10.0
DEFAULT_MIN_STEP = 500


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def summarize_records(
    records: list[dict[str, Any]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_step: int = DEFAULT_MIN_STEP,
) -> dict[str, Any]:
    checked = 0
    failures: list[dict[str, Any]] = []
    max_grad_norm_tme = 0.0
    max_health_abs = 0.0
    max_step = 0

    for record in records:
        step = int(record.get("step") or 0)
        max_step = max(max_step, step)
        grad_norm_tme = _finite_float(record.get("grad_norm_tme"))
        health = record.get("grad_health_tme") or {}
        health_max_abs = _finite_float(health.get("max_abs"))
        nonfinite_tensors = int(health.get("nonfinite_tensors") or 0)
        nonfinite_values = int(health.get("nonfinite_values") or 0)
        checked += 1

        if grad_norm_tme is not None:
            max_grad_norm_tme = max(max_grad_norm_tme, grad_norm_tme)
        if health_max_abs is not None:
            max_health_abs = max(max_health_abs, health_max_abs)

        failed = (
            grad_norm_tme is None
            or grad_norm_tme >= threshold
            or health_max_abs is None
            or health_max_abs >= threshold
            or nonfinite_tensors
            or nonfinite_values
        )
        if failed:
            failures.append(
                {
                    "step": step,
                    "grad_norm_tme": record.get("grad_norm_tme"),
                    "grad_health_tme.max_abs": health.get("max_abs"),
                    "nonfinite_tensors": nonfinite_tensors,
                    "nonfinite_values": nonfinite_values,
                }
            )

    passed = bool(records) and max_step >= min_step and not failures
    return {
        "passed": passed,
        "records": checked,
        "max_step": max_step,
        "min_step": min_step,
        "threshold": threshold,
        "max_grad_norm_tme": max_grad_norm_tme,
        "max_grad_health_tme_abs": max_health_abs,
        "failure_count": len(failures),
        "failures": failures[:20],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate TME mid-length grad-health logs."
    )
    parser.add_argument("train_log", type=Path, help="Path to train_log.jsonl.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--min-step", type=int, default=DEFAULT_MIN_STEP)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write the JSON summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize_records(
        read_jsonl(args.train_log),
        threshold=args.threshold,
        min_step=args.min_step,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
