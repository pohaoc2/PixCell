from __future__ import annotations

import argparse
from pathlib import Path

from tools.stage3.style_mapping import load_style_mapping

from .data import load_dataset_summary
from .figures import export_report_png_pages
from .html_report import render_report_html
from .shared import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a paired vs unpaired ablation HTML report.")
    parser.add_argument(
        "--paired-metrics-root",
        type=Path,
        default=ROOT / "inference_output" / "paired_ablation" / "ablation_results",
        help="Directory containing paired per-tile metrics.json files.",
    )
    parser.add_argument(
        "--paired-dataset-root",
        type=Path,
        default=ROOT / "inference_output" / "paired_ablation",
        help="Paired dataset root used for representative figure lookup.",
    )
    parser.add_argument(
        "--paired-reference-root",
        type=Path,
        default=ROOT / "data" / "orion-crc33",
        help="Reference H&E root used to compute missing paired HED metrics.",
    )
    parser.add_argument(
        "--paired-style-mapping-json",
        type=Path,
        default=None,
        help="Optional layout->style mapping JSON for paired reference lookup.",
    )
    parser.add_argument(
        "--unpaired-metrics-root",
        type=Path,
        default=ROOT / "inference_output" / "unpaired_ablation" / "ablation_results",
        help="Directory containing unpaired per-tile metrics.json files.",
    )
    parser.add_argument(
        "--unpaired-dataset-root",
        type=Path,
        default=ROOT / "inference_output" / "unpaired_ablation",
        help="Unpaired dataset root used for representative figure lookup.",
    )
    parser.add_argument(
        "--unpaired-reference-root",
        type=Path,
        default=ROOT / "data" / "orion-crc33",
        help="Reference H&E root used to compute missing unpaired HED metrics when absent.",
    )
    parser.add_argument(
        "--unpaired-style-mapping-json",
        type=Path,
        default=None,
        help="Optional layout->style mapping JSON for unpaired reference lookup.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "ablation_scientific_report.html",
        help="HTML output path.",
    )
    parser.add_argument(
        "--min-gt-cells",
        type=int,
        default=0,
        help="Skip tiles with fewer than this many GT cell instances in each dataset (default: 0 = no filter).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Channel Ablation Scientific Report",
        help="Title shown at the top of the HTML report.",
    )
    parser.add_argument(
        "--self-contained",
        action="store_true",
        help="Embed representative evidence images so the HTML can be opened as a standalone file.",
    )
    parser.add_argument(
        "--png-dir",
        type=Path,
        default=ROOT / "figures" / "pngs",
        help="Directory where standalone PNG pages will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [
        load_dataset_summary(
            slug="paired",
            title="Paired",
            metrics_root=args.paired_metrics_root.resolve(),
            dataset_root=args.paired_dataset_root.resolve(),
            reference_root=args.paired_reference_root.resolve(),
            style_mapping=load_style_mapping(args.paired_style_mapping_json),
            min_gt_cells=args.min_gt_cells,
        ),
        load_dataset_summary(
            slug="unpaired",
            title="Unpaired",
            metrics_root=args.unpaired_metrics_root.resolve(),
            dataset_root=args.unpaired_dataset_root.resolve(),
            reference_root=args.unpaired_reference_root.resolve(),
            style_mapping=load_style_mapping(args.unpaired_style_mapping_json),
            min_gt_cells=args.min_gt_cells,
        ),
    ]
    report = render_report_html(
        args.title,
        summaries,
        args.output.resolve(),
        self_contained=args.self_contained,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    png_paths = export_report_png_pages(summaries, args.png_dir.resolve())
    print(f"Rendered ablation HTML report -> {args.output}")
    print(f"Saved {len(png_paths)} PNG pages -> {args.png_dir}")
