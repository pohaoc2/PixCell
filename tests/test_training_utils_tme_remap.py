from diffusion.utils.tme_checkpoint_key_remap import (
    remap_tme_state_dict_cell_identity_to_cell_types,
)


def test_remap_cell_identity_checkpoint_keys_to_cell_types():
    sd = {
        "groups.cell_identity.encoder.stem.0.weight": 1,
        "groups.cell_state.encoder.stem.0.weight": 2,
        "norm_q.weight": 3,
    }
    out = remap_tme_state_dict_cell_identity_to_cell_types(sd)
    assert "groups.cell_types.encoder.stem.0.weight" in out
    assert out["groups.cell_types.encoder.stem.0.weight"] == 1
    assert "groups.cell_identity.encoder.stem.0.weight" not in out
    assert out["groups.cell_state.encoder.stem.0.weight"] == 2


def test_remap_idempotent_for_cell_types_keys():
    sd = {"groups.cell_types.encoder.weight": 0}
    out = remap_tme_state_dict_cell_identity_to_cell_types(sd)
    assert out == sd
