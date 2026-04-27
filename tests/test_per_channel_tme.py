import torch


def _make_module():
    from diffusion.model.nets.per_channel_tme import PerChannelTMEModule

    active_channels = [
        "cell_masks",
        "cell_type_healthy",
        "cell_type_cancer",
        "cell_type_immune",
        "cell_state_prolif",
        "cell_state_nonprolif",
        "cell_state_dead",
        "vasculature",
        "oxygen",
        "glucose",
    ]
    return PerChannelTMEModule(active_channels=active_channels, base_ch=16, latent_ch=16, num_heads=4)


def test_per_channel_tme_forward_shape():
    module = _make_module()
    mask_latent = torch.randn(2, 16, 32, 32)
    control_input = torch.randn(2, 10, 256, 256)
    out = module(mask_latent, control_input)
    assert out.shape == (2, 16, 32, 32)


def test_per_channel_tme_active_subset_and_residuals():
    module = _make_module()
    mask_latent = torch.randn(1, 16, 32, 32)
    control_input = torch.randn(1, 10, 256, 256)
    out, residuals = module(
        mask_latent,
        control_input,
        active_groups={"cell_type_cancer", "oxygen"},
        return_residuals=True,
    )
    assert out.shape == (1, 16, 32, 32)
    assert set(residuals.keys()) == {"cell_type_cancer", "oxygen"}


def test_per_channel_tme_empty_active_groups_returns_mask_latent():
    module = _make_module()
    mask_latent = torch.randn(1, 16, 32, 32)
    control_input = torch.randn(1, 10, 256, 256)
    out = module(mask_latent, control_input, active_groups=set())
    torch.testing.assert_close(out, mask_latent, atol=1e-6, rtol=1e-6)