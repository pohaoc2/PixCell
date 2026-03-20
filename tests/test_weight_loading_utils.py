import torch

from train_scripts.mapping_weights_helper import (
    _extract_state_dict,
    _filter_state_dict_for_model,
    _looks_like_diffusers_pixcell_sd,
    _strip_known_prefixes,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)


def test_strip_known_prefixes_removes_common_wrappers():
    state = {
        "module.linear.weight": torch.randn(3, 4),
        "model.linear.bias": torch.randn(3),
        "_orig_mod.linear.weight": torch.randn(3, 4),
    }
    out = _strip_known_prefixes(state)
    assert "linear.weight" in out
    assert "linear.bias" in out


def test_filter_state_dict_for_model_keeps_only_matching_shapes():
    model = TinyModel()
    state = {
        "linear.weight": torch.randn(3, 4),  # match
        "linear.bias": torch.randn(99),  # wrong shape
        "extra.key": torch.randn(1),  # not in model
    }
    filtered, dropped = _filter_state_dict_for_model(model, state)
    assert "linear.weight" in filtered
    assert "linear.bias" not in filtered
    assert "extra.key" not in filtered
    assert len(dropped) == 2


def test_extract_state_dict_prefers_standard_nested_keys():
    nested = {"state_dict": {"linear.weight": torch.randn(3, 4)}}
    out = _extract_state_dict(nested)
    assert "linear.weight" in out


def test_detect_diffusers_pixcell_keys_heuristic():
    state = {"transformer_blocks.0.attn1.to_q.weight": torch.randn(4, 4)}
    assert _looks_like_diffusers_pixcell_sd(state) is True
    assert _looks_like_diffusers_pixcell_sd({"linear.weight": torch.randn(2, 2)}) is False
