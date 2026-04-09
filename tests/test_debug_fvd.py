from __future__ import annotations

import sys
import types
from pathlib import Path

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    class _DummyTensor:
        pass

    torch_stub.float16 = "float16"
    torch_stub.float32 = "float32"
    torch_stub.Tensor = _DummyTensor
    sys.modules["torch"] = torch_stub

if "diffusers" not in sys.modules:
    diffusers_stub = types.ModuleType("diffusers")

    class _DummyScheduler:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def set_timesteps(self, *args, **kwargs) -> None:
            pass

    diffusers_stub.DDPMScheduler = _DummyScheduler
    sys.modules["diffusers"] = diffusers_stub

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compute_fid import ImageFeatureRecord
from tools.debug_fvd import resolve_condition_key, split_records
from tools.stage3.ablation_vis_utils import FOUR_GROUP_ORDER, condition_metric_key


def test_resolve_condition_key_maps_all_to_full_condition() -> None:
    full_key = condition_metric_key(FOUR_GROUP_ORDER)

    assert resolve_condition_key("all", [full_key]) == full_key


def test_resolve_condition_key_preserves_explicit_key() -> None:
    key = "cell_types__cell_state"

    assert resolve_condition_key(key, [key]) == key


def test_split_records_returns_non_empty_deterministic_halves(tmp_path: Path) -> None:
    records = [
        ImageFeatureRecord(image_path=tmp_path / f"img_{idx}.png")
        for idx in range(6)
    ]

    left_a, right_a = split_records(records, seed=7)
    left_b, right_b = split_records(records, seed=7)

    assert left_a
    assert right_a
    assert [record.image_path for record in left_a] == [record.image_path for record in left_b]
    assert [record.image_path for record in right_a] == [record.image_path for record in right_b]
    assert {
        record.image_path for record in left_a + right_a
    } == {record.image_path for record in records}
