"""Run and summarize the Phase 0 TME gradient-explosion follow-up smoke.

The companion runbook is docs/debug/grad_explosion_tme_followup.md.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PASS_THRESHOLD = 1.0e3
DEFAULT_VARIANTS = (
    ("concat", "configs/config_controlnet_exp_smoke_depth18_bs2_concat_fixed.py"),
    ("additive", "configs/config_controlnet_exp_smoke_depth18_bs2_additive_fixed.py"),
    ("grouped", "configs/config_controlnet_exp_smoke_depth18_bs2_grouped_fixed.py"),
    ("per_channel", "configs/config_controlnet_exp_smoke_depth18_bs2_per_channel_fixed.py"),
)


@dataclass(frozen=True)
class VariantResult:
    name: str
    config: Path
    work_dir: Path
    train_log: Path
    log_file: Path
    returncode: int | None
    elapsed_sec: float | None
    new_records: int
    grad_norm_tme: float | None
    max_abs: float | None
    pass_threshold: float
    passed: bool
    top_tensors: list[dict]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_config(config_path: Path):
    from diffusion.utils.misc import read_config

    return read_config(str(config_path))


def resolve_work_dir(config_path: Path) -> Path:
    config = _read_config(config_path)
    work_dir = Path(str(config.work_dir))
    if not work_dir.is_absolute():
        work_dir = repo_root() / work_dir
    return work_dir


def count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def read_jsonl_records(path: Path, skip: int = 0) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < skip or not line.strip():
                continue
            records.append(json.loads(line))
    return records


def _as_finite_float(value) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return number
    return number


def summarize_variant(
    name: str,
    config_path: Path,
    returncode: int | None,
    elapsed_sec: float | None,
    previous_records: int = 0,
    pass_threshold: float = PASS_THRESHOLD,
) -> VariantResult:
    work_dir = resolve_work_dir(config_path)
    train_log = work_dir / "train_log.jsonl"
    log_file = work_dir / "train_log.log"
    records = read_jsonl_records(train_log, skip=previous_records)
    first = records[0] if records else {}
    health = first.get("grad_health_tme") or {}
    top_tensors = list(health.get("top_tensors") or [])
    max_abs = _as_finite_float(health.get("max_abs"))
    if max_abs is None and top_tensors:
        max_abs = max(float(item.get("max_abs", 0.0)) for item in top_tensors)
    grad_norm_tme = _as_finite_float(first.get("grad_norm_tme"))
    passed = (
        returncode in (None, 0)
        and bool(records)
        and grad_norm_tme is not None
        and max_abs is not None
        and math.isfinite(grad_norm_tme)
        and math.isfinite(max_abs)
        and grad_norm_tme < pass_threshold
        and max_abs < pass_threshold
    )
    return VariantResult(
        name=name,
        config=config_path,
        work_dir=work_dir,
        train_log=train_log,
        log_file=log_file,
        returncode=returncode,
        elapsed_sec=elapsed_sec,
        new_records=len(records),
        grad_norm_tme=grad_norm_tme,
        max_abs=max_abs,
        pass_threshold=pass_threshold,
        passed=passed,
        top_tensors=top_tensors,
    )


def run_variant(
    name: str,
    config_path: Path,
    python_bin: str,
    pass_threshold: float,
) -> VariantResult:
    work_dir = resolve_work_dir(config_path)
    previous_records = count_jsonl_records(work_dir / "train_log.jsonl")
    command = [python_bin, "stage2_train.py", str(config_path)]
    print(f"\n[phase0] starting {name}: {' '.join(command)}", flush=True)
    print(f"[phase0] logging to {work_dir / 'train_log.log'}", flush=True)
    start = time.monotonic()
    completed = subprocess.run(command, cwd=repo_root(), check=False)
    elapsed = time.monotonic() - start
    result = summarize_variant(
        name=name,
        config_path=config_path,
        returncode=completed.returncode,
        elapsed_sec=elapsed,
        previous_records=previous_records,
        pass_threshold=pass_threshold,
    )
    status = "PASS" if result.passed else "FAIL"
    print(
        f"[phase0] finished {name}: {status} "
        f"returncode={completed.returncode} elapsed={_format_seconds(elapsed)} "
        f"grad_norm_tme={result.grad_norm_tme} max_abs={result.max_abs}",
        flush=True,
    )
    return result


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    minutes, sec = divmod(int(round(seconds)), 60)
    return f"{minutes}m{sec:02d}s" if minutes else f"{sec}s"


def print_summary(results: Iterable[VariantResult]) -> None:
    print("\nPhase 0 TME grad-health summary")
    print("variant      pass  records  grad_norm_tme  max_abs      elapsed  log")
    for result in results:
        status = "yes" if result.passed else "no"
        grad = "n/a" if result.grad_norm_tme is None else f"{result.grad_norm_tme:.3e}"
        max_abs = "n/a" if result.max_abs is None else f"{result.max_abs:.3e}"
        print(
            f"{result.name:<12} {status:<5} {result.new_records:<7} "
            f"{grad:<14} {max_abs:<12} {_format_seconds(result.elapsed_sec):<8} "
            f"{result.train_log}"
        )
        if result.top_tensors:
            top = result.top_tensors[0]
            print(
                f"  top: {top.get('name')} "
                f"max_abs={float(top.get('max_abs', 0.0)):.3e} "
                f"finite_norm={float(top.get('finite_norm', 0.0)):.3e}"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run/summarize the TME grad-explosion Phase 0 smoke variants."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch stage2_train.py.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Do not run training; summarize existing train_log.jsonl files.",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=PASS_THRESHOLD,
        help="Pass threshold for grad_norm_tme and grad_health_tme.max_abs.",
    )
    parser.add_argument(
        "--variant",
        choices=[name for name, _ in DEFAULT_VARIANTS],
        action="append",
        help="Run only selected variant(s). May be passed more than once.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when any variant misses the pass threshold.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected = set(args.variant or [])
    variants = [
        (name, repo_root() / config)
        for name, config in DEFAULT_VARIANTS
        if not selected or name in selected
    ]
    results = []
    for name, config_path in variants:
        if args.summary_only:
            results.append(
                summarize_variant(
                    name=name,
                    config_path=config_path,
                    returncode=None,
                    elapsed_sec=None,
                    pass_threshold=args.pass_threshold,
                )
            )
        else:
            results.append(
                run_variant(
                    name=name,
                    config_path=config_path,
                    python_bin=args.python,
                    pass_threshold=args.pass_threshold,
                )
            )
    print_summary(results)
    if args.strict and not all(result.passed for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
