"""Remap legacy TME checkpoint keys (no torch dependency)."""


def remap_tme_state_dict_cell_identity_to_cell_types(state_dict):
    """
    Older checkpoints used ``groups.cell_identity``; config now uses ``cell_types``.

    Remap tensor keys so ``load_state_dict`` matches the current module layout.
    """
    out = {}
    for key, value in state_dict.items():
        if key.startswith("groups.cell_identity."):
            out["groups.cell_types." + key[len("groups.cell_identity.") :]] = value
        else:
            out[key] = value
    return out
