"""Run a2 decomposition for the a4 null tile set, loading model once."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.a2_decomposition.main import (
    DecompositionConfig,
    _load_worker_resources,
    _load_control_tensor,
    _generate_from_control,
    _resolve_uni_embedding,
    _save_png,
    DEFAULT_MODES,
)
from src._tasklib.io import ensure_directory

NULL_TILE_IDS = sorted(set([
    # eccentricity_mean
    "10752_20480", "11008_15872", "11520_19456", "11520_22016", "11520_22528",
    "11520_4096", "12032_13568", "12544_21248", "12800_21248", "14592_13568",
    "14592_26880", "14848_26368", "15360_11008", "15360_19712", "15616_7936",
    "17408_25344", "18176_26368", "19200_33280", "19712_37632", "20992_18432",
    "21248_12288", "22272_48128", "22784_43264", "23296_24576", "24064_36096",
    "25600_46080", "27136_41216", "29952_46848", "8960_7936", "9728_18688",
    # nuclear_area_mean
    "10752_16384", "10752_18944", "13568_19712", "13568_3328", "14592_4096",
    "15360_4864", "15872_30464", "16128_25600", "17664_37376", "18176_18432",
    "18944_15616", "20480_40192", "21504_17920", "21504_48896", "21760_38912",
    "22272_18176", "23808_17152", "25088_28928", "25600_42752", "26624_41728",
    "27904_38144", "29440_49408", "29952_45056", "31232_39424", "31488_45312",
    "31744_45056", "35328_47360", "35584_45568", "6912_2048", "9472_10752",
    # nuclei_density
    "0_6400", "10752_16128", "11520_16384", "12288_10496", "12288_28672",
    "12544_16640", "12544_7168", "12800_32256", "13056_4096", "13824_10240",
    "14848_27136", "16128_15872", "16128_32512", "17664_13056", "17920_18944",
    "18432_27904", "18944_34304", "20480_25088", "20992_38656", "21248_42240",
    "21760_46592", "22016_13056", "22272_44800", "23040_22016", "24064_36608",
    "24320_48896", "28928_44288", "29440_46080", "31232_45312", "3584_8704",
]))

CONFIG = DecompositionConfig(
    config_path=ROOT / "checkpoints/concat_95470_0/config.py",
    checkpoint_dir=ROOT / "checkpoints/concat_95470_0/checkpoints/step_0002600",
    data_root=ROOT / "data/orion-crc33",
    out_dir=ROOT / "inference_output/a1_concat/a4_uni_probe/uni_null",
    sample_n=len(NULL_TILE_IDS),
)

NUM_STEPS = 20
GUIDANCE_SCALE = 2.5
SEED = 42


def main() -> None:
    print(f"Loading model from {CONFIG.checkpoint_dir}", flush=True)
    resources = _load_worker_resources(CONFIG, device="cuda", num_steps=NUM_STEPS)

    total = len(NULL_TILE_IDS)
    for index, tile_id in enumerate(NULL_TILE_IDS, start=1):
        out_dir = ensure_directory(CONFIG.out_dir / "generated" / tile_id)
        mode_paths = [out_dir / f"{mode.name}.png" for mode in DEFAULT_MODES]
        if all(p.is_file() for p in mode_paths):
            print(f"[{index}/{total}] skip {tile_id} (existing)", flush=True)
            continue

        ctrl_full = _load_control_tensor(
            tile_id,
            resources.inference_config.data.active_channels,
            resources.inference_config.image_size,
            resources.exp_channels_dir,
        )

        for mode in DEFAULT_MODES:
            output_path = out_dir / f"{mode.name}.png"
            if output_path.is_file():
                continue
            uni_embeds = _resolve_uni_embedding(
                tile_id,
                feat_dir=resources.feat_dir,
                null_uni=not mode.use_uni,
            )
            active_groups = None if mode.use_tme else ()
            gen_np, _ = _generate_from_control(
                ctrl_full,
                models=resources.models,
                config=resources.inference_config,
                scheduler=resources.scheduler,
                uni_embeds=uni_embeds,
                device=resources.device,
                guidance_scale=GUIDANCE_SCALE,
                seed=SEED,
                active_groups=active_groups,
            )
            _save_png(gen_np, output_path)

        print(f"[{index}/{total}] generated {tile_id}", flush=True)

    print("Done. Run summarize:", flush=True)
    print(
        f"  python -m src.a2_decomposition.main "
        f"--out-dir {CONFIG.out_dir} "
        f"--data-root {CONFIG.data_root} "
        f"--worker summarize",
        flush=True,
    )


if __name__ == "__main__":
    main()
