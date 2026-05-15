"""Probe the largest useful batch size for depth-27 grouped TME training."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


OOM_PATTERNS = (
    "CUDA out of memory",
    "CUDNN_STATUS_ALLOC_FAILED",
    "out of memory",
    "Cannot allocate memory",
)
SAMPLES_PER_SEC_RE = re.compile(r"Samples/s:\s*([0-9.]+)")


@dataclass(frozen=True)
class ProbeResult:
    batch_size: int
    status: str
    elapsed_sec: float
    samples_per_sec: float | None
    returncode: int
    work_dir: str
    config_path: str
    log_path: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_config(
    *,
    path: Path,
    base_config: Path,
    batch_size: int,
    steps: int,
    work_dir: Path,
    seed: int,
) -> None:
    sample_count = batch_size * steps
    path.write_text(
        "\n".join(
            [
                f'"""Generated batch-size probe config: bs={batch_size}, steps={steps}."""',
                "",
                f"_base_ = [{str(base_config)!r}]",
                "",
                "controlnet_depth = 27",
                "num_epochs = 1",
                f"train_batch_size = {batch_size}",
                "num_workers = 0",
                "log_interval = 1",
                "save_model_steps = 100000",
                "save_model_epochs = 1000",
                "save_final_checkpoint = False",
                "debug_tme_probe = False",
                "",
                f"max_train_samples = {sample_count}",
                f"data = dict(max_train_samples={sample_count})",
                "",
                f"work_dir = {str(work_dir)!r}",
                f"seed = {seed}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _extract_samples_per_sec(text: str) -> float | None:
    matches = SAMPLES_PER_SEC_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def _status(returncode: int, text: str) -> str:
    if returncode == 0:
        return "pass"
    lowered = text.lower()
    if any(pattern.lower() in lowered for pattern in OOM_PATTERNS):
        return "oom"
    return "fail"


def run_probe(
    *,
    batch_size: int,
    steps: int,
    base_config: Path,
    output_dir: Path,
    python_bin: str,
    seed: int,
    stop_on_oom: bool,
) -> ProbeResult:
    config_dir = output_dir / "configs"
    work_base = output_dir / "work_dirs"
    log_dir = output_dir / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    work_base.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    work_dir = work_base / f"bs{batch_size}"
    config_path = config_dir / f"config_bs{batch_size}.py"
    log_path = log_dir / f"bs{batch_size}.log"
    _write_config(
        path=config_path,
        base_config=base_config,
        batch_size=batch_size,
        steps=steps,
        work_dir=work_dir,
        seed=seed,
    )

    command = [python_bin, "-u", "stage2_train.py", str(config_path)]
    start = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=_repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.monotonic() - start
    output = completed.stdout or ""
    log_path.write_text(output, encoding="utf-8")
    status = _status(completed.returncode, output)
    if stop_on_oom and status == "oom":
        print(f"[batch-probe] bs={batch_size} OOM; stopping sweep", flush=True)
    else:
        print(
            f"[batch-probe] bs={batch_size} status={status} "
            f"elapsed={elapsed:.1f}s samples_per_sec={_extract_samples_per_sec(output)}",
            flush=True,
        )
    return ProbeResult(
        batch_size=batch_size,
        status=status,
        elapsed_sec=elapsed,
        samples_per_sec=_extract_samples_per_sec(output),
        returncode=completed.returncode,
        work_dir=str(work_dir),
        config_path=str(config_path),
        log_path=str(log_path),
    )


def write_reports(results: list[ProbeResult], output_dir: Path) -> None:
    summary_json = output_dir / "batch_size_probe_summary.json"
    summary_md = output_dir / "batch_size_probe_summary.md"
    output_dir.mkdir(parents=True, exist_ok=True)

    passed = [result for result in results if result.status == "pass"]
    largest_fit = max((result.batch_size for result in passed), default=None)
    best_throughput = max(
        passed,
        key=lambda result: result.samples_per_sec or 0.0,
        default=None,
    )
    payload = {
        "largest_fit_batch_size": largest_fit,
        "best_throughput_batch_size": None
        if best_throughput is None
        else best_throughput.batch_size,
        "best_samples_per_sec": None
        if best_throughput is None
        else best_throughput.samples_per_sec,
        "results": [asdict(result) for result in results],
    }
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# TME Depth-27 Batch-Size Probe",
        "",
        f"- Largest passing batch size: {largest_fit}",
        "- Best throughput batch size: "
        + (
            "None"
            if best_throughput is None
            else f"{best_throughput.batch_size} ({best_throughput.samples_per_sec} samples/s)"
        ),
        "",
        "| batch_size | status | samples/s | elapsed_s | log |",
        "|---:|---|---:|---:|---|",
    ]
    for result in results:
        samples = "" if result.samples_per_sec is None else f"{result.samples_per_sec:.3f}"
        lines.append(
            f"| {result.batch_size} | {result.status} | {samples} | "
            f"{result.elapsed_sec:.1f} | `{result.log_path}` |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[batch-probe] wrote {summary_json}", flush=True)
    print(f"[batch-probe] wrote {summary_md}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[2, 4, 8, 12, 16, 24, 32, 48, 64],
        help="Batch sizes to probe in order.",
    )
    parser.add_argument("--steps", type=int, default=20, help="Optimizer steps per batch size.")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=_repo_root() / "configs/config_controlnet_exp.py",
        help="Base config to probe. Default is grouped production TME.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated configs, logs, and summaries.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-stop-on-oom",
        action="store_true",
        help="Continue probing larger batch sizes after the first OOM.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results: list[ProbeResult] = []
    for batch_size in args.batch_sizes:
        result = run_probe(
            batch_size=batch_size,
            steps=args.steps,
            base_config=args.base_config.resolve(),
            output_dir=args.output_dir,
            python_bin=args.python,
            seed=args.seed,
            stop_on_oom=not args.no_stop_on_oom,
        )
        results.append(result)
        write_reports(results, args.output_dir)
        if result.status == "oom" and not args.no_stop_on_oom:
            break
    return 0 if any(result.status == "pass" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
