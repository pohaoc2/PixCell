"""Aggregate per-seed training logs for the A3 zero-init stability ablation."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import fmean, pstdev


_LOSS_RE = re.compile(r"(?:^|\s)Loss:\s*([-+0-9.eE]+|nan|NaN|inf|-inf)")
_STEP_RE = re.compile(r"(?:^|\s)Step\s*\[(\d+)[/\]]")
_GRAD_RE = re.compile(r"(?:grad_norm|GradNorm|proj_grad\[[^\]]+\])[:=]\s*([-+0-9.eE]+|nan|NaN|inf|-inf)")


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_nan(value) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _read_log(path: Path) -> list[dict]:
    entries: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
                continue
            except json.JSONDecodeError:
                pass
            loss_match = _LOSS_RE.search(line)
            step_match = _STEP_RE.search(line)
            grad_matches = [_to_float(match) for match in _GRAD_RE.findall(line)]
            if loss_match and step_match:
                rec = {"step": int(step_match.group(1)), "loss": _to_float(loss_match.group(1))}
                if grad_matches:
                    rec["grad_norm"] = max(abs(g) for g in grad_matches)
                entries.append(rec)
    return entries


def _loss_at_or_before(entries: list[dict], fixed_step: int) -> float | None:
    selected: float | None = None
    selected_step = -1
    for entry in entries:
        step = int(entry.get("step", -1))
        loss = entry.get("loss")
        if step <= fixed_step and step >= selected_step and not _is_nan(loss):
            selected = float(loss)
            selected_step = step
    return selected


def aggregate(
    seed_dirs,
    *,
    fixed_step: int,
    grad_threshold: float,
    fid_diverge_cutoff: float | None,
    fid_lookup: dict[str, float] | None = None,
) -> dict:
    """Aggregate logs and mark seeds diverged by NaN loss, grad explosion, or FID."""
    fid_lookup = fid_lookup or {}
    per_seed: list[dict] = []
    losses_at_step: list[float] = []
    divergence_count = 0

    for seed_dir_raw in seed_dirs:
        seed_dir = Path(seed_dir_raw)
        log_path = seed_dir / "train_log.jsonl"
        if not log_path.exists():
            log_path = seed_dir / "train_log.log"
        entries = _read_log(log_path)
        loss_at_fixed = _loss_at_or_before(entries, fixed_step)
        diverged = False
        reason = None

        for entry in entries:
            if _is_nan(entry.get("loss")):
                diverged = True
                reason = "nan_loss"
                break
            grad_norm = entry.get("grad_norm")
            if grad_norm is not None and abs(float(grad_norm)) > grad_threshold:
                diverged = True
                reason = "grad_explosion"
                break

        seed_name = seed_dir.name
        fid = fid_lookup.get(seed_name)
        if not diverged and fid_diverge_cutoff is not None and fid is not None:
            if float(fid) > fid_diverge_cutoff:
                diverged = True
                reason = "fid_outlier"

        if diverged:
            divergence_count += 1
        elif loss_at_fixed is not None:
            losses_at_step.append(loss_at_fixed)

        per_seed.append(
            {
                "seed_dir": str(seed_dir),
                "log_path": str(log_path),
                "loss_at_fixed_step": loss_at_fixed,
                "diverged": diverged,
                "divergence_reason": reason,
                "fid": fid,
            }
        )

    mean_loss = fmean(losses_at_step) if losses_at_step else float("nan")
    std_loss = pstdev(losses_at_step) if len(losses_at_step) > 1 else 0.0 if losses_at_step else float("nan")
    return {
        "per_seed": per_seed,
        "divergence_count": divergence_count,
        "mean_loss_at_fixed_step": mean_loss,
        "std_loss_at_fixed_step": std_loss,
        "fixed_step": fixed_step,
        "grad_threshold": grad_threshold,
        "fid_diverge_cutoff": fid_diverge_cutoff,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dirs", "--seed_dirs", dest="seed_dirs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fixed-step", "--fixed_step", dest="fixed_step", type=int, default=10000)
    parser.add_argument("--grad-threshold", "--grad_threshold", dest="grad_threshold", type=float, default=100.0)
    parser.add_argument("--fid-diverge-cutoff", "--fid_diverge_cutoff", dest="fid_diverge_cutoff", type=float, default=None)
    parser.add_argument("--fid-json", "--fid_json", dest="fid_json", default=None)
    args = parser.parse_args(argv)

    fid_lookup = json.loads(Path(args.fid_json).read_text()) if args.fid_json else {}
    summary = aggregate(
        [Path(path) for path in args.seed_dirs],
        fixed_step=args.fixed_step,
        grad_threshold=args.grad_threshold,
        fid_diverge_cutoff=args.fid_diverge_cutoff,
        fid_lookup=fid_lookup,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
