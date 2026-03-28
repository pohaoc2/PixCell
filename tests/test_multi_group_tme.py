import unittest
import torch


def _make_module():
    from diffusion.model.nets.multi_group_tme import MultiGroupTMEModule
    channel_groups = [
        dict(name="cell_identity", n_channels=3),
        dict(name="cell_state", n_channels=3),
        dict(name="vasculature", n_channels=1),
        dict(name="microenv", n_channels=2),
    ]
    return MultiGroupTMEModule(channel_groups=channel_groups, base_ch=32, latent_ch=16, num_heads=4)


def _make_inputs(B=2, H=256, W=256):
    mask_latent = torch.randn(B, 16, 32, 32)
    tme_channel_dict = {
        "cell_identity": torch.randn(B, 3, H, W),
        "cell_state": torch.randn(B, 3, H, W),
        "vasculature": torch.randn(B, 1, H, W),
        "microenv": torch.randn(B, 2, H, W),
    }
    return mask_latent, tme_channel_dict


class TestMultiGroupTMEModule(unittest.TestCase):
    def test_forward_shape(self):
        module = _make_module()
        mask_latent, tme_dict = _make_inputs()
        out = module(mask_latent, tme_dict)
        self.assertEqual(out.shape, (2, 16, 32, 32))

    def test_small_init_nonzero_residuals(self):
        # proj uses small normal init (std=0.02) — residuals should be non-zero
        module = _make_module()
        mask_latent, tme_dict = _make_inputs()
        with torch.no_grad():
            out, residuals = module(mask_latent, tme_dict, return_residuals=True)
        for name, delta in residuals.items():
            self.assertFalse(
                torch.allclose(delta, torch.zeros_like(delta), atol=1e-6),
                msg=f"group '{name}' residual is unexpectedly zero — proj may have reverted to zero-init",
            )

    def test_active_groups_subset(self):
        module = _make_module()
        mask_latent, tme_dict = _make_inputs()
        out = module(mask_latent, tme_dict, active_groups={"cell_identity"})
        self.assertEqual(out.shape, (2, 16, 32, 32))

    def test_empty_active_groups_returns_mask_latent(self):
        module = _make_module()
        mask_latent, tme_dict = _make_inputs()
        with torch.no_grad():
            out = module(mask_latent, tme_dict, active_groups=set())
        torch.testing.assert_close(out, mask_latent, atol=1e-6, rtol=1e-6)

    def test_return_residuals(self):
        module = _make_module()
        mask_latent, tme_dict = _make_inputs()
        out, residuals = module(mask_latent, tme_dict, return_residuals=True)
        self.assertEqual(set(residuals.keys()), {"cell_identity", "cell_state", "vasculature", "microenv"})
        for name, delta in residuals.items():
            self.assertEqual(delta.shape, (2, 16, 32, 32))

    def test_return_attn_weights(self):
        module = _make_module()
        mask_latent, tme_dict = _make_inputs()
        out, residuals, attn_maps = module(
            mask_latent, tme_dict, return_residuals=True, return_attn_weights=True
        )
        self.assertEqual(set(attn_maps.keys()), {"cell_identity", "cell_state", "vasculature", "microenv"})
        for name, weights in attn_maps.items():
            self.assertEqual(weights.shape, (2, 4, 1024, 1024))

    def test_missing_group_in_dict_skipped(self):
        module = _make_module()
        mask_latent, _ = _make_inputs()
        partial_dict = {"cell_identity": torch.randn(2, 3, 256, 256)}
        out = module(mask_latent, partial_dict)
        self.assertEqual(out.shape, (2, 16, 32, 32))

    def test_n_params(self):
        module = _make_module()
        total = sum(p.numel() for p in module.parameters() if p.requires_grad)
        self.assertGreater(total, 1_000_000)
        self.assertLess(total, 4_000_000)


if __name__ == "__main__":
    unittest.main()
