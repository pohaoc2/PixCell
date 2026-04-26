"""Lightweight helpers shared by initialize_models CLI code."""

import argparse
import os


def set_fsdp_env():
    """Set environment variables for FSDP training."""
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
    os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_PRE"
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "PixArtBlock"


def find_checkpoint(resume_dir):
    """Find the latest checkpoint file in a directory, or return a file path unchanged."""
    if os.path.isfile(resume_dir):
        return resume_dir

    checkpoints = [ckpt for ckpt in os.listdir(resume_dir) if ckpt.endswith(".pth")]
    if not checkpoints:
        raise ValueError(f"No checkpoint found in {resume_dir}")

    checkpoints = sorted(
        checkpoints, key=lambda name: int(name.split("_")[-1].replace(".pth", "")), reverse=True
    )
    return os.path.join(resume_dir, checkpoints[0])


def parse_args(args_list=None):
    """Parse command-line arguments for ControlNet training."""
    parser = argparse.ArgumentParser(description="Train ControlNet for PixCell-256")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument("--resume-from", help="the dir to resume the training")
    parser.add_argument("--load-from", help="the checkpoint to load from")
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--report-to", type=str, default="tensorboard")
    parser.add_argument("--tracker-project-name", type=str, default="pixcell_controlnet")
    parser.add_argument("--slurm-time-limit", type=float, default=float("inf"))
    parser.add_argument("--loss-report-name", type=str, default="loss")
    parser.add_argument("--skip-step", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config seed for per-run ablation reproducibility.",
    )
    return parser.parse_args(args_list)
