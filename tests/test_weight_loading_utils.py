import torch

from train_scripts.mapping_weights_helper import (
    _filter_state_dict_for_model,
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
