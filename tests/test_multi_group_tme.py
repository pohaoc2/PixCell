import unittest
import torch


def _make_module():
    from diffusion.model.nets.multi_group_tme import MultiGroupTMEModule
    channel_groups = [
        dict(
            name="cell_types",
            n_channels=3,
            channels=["cell_type_healthy", "cell_type_cancer", "cell_type_immune"],
        ),
        dict(
            name="cell_state",
            n_channels=3,
            channels=["cell_state_prolif", "cell_state_nonprolif", "cell_state_dead"],
        ),
        dict(name="vasculature", n_channels=1, channels=["vasculature"]),
        dict(name="microenv", n_channels=2, channels=["oxygen", "glucose"]),
    ]
    return MultiGroupTMEModule(channel_groups=channel_groups, base_ch=32, latent_ch=16, num_heads=4)


def _make_inputs(B=2, H=256, W=256):
    mask_latent = torch.randn(B, 16, 32, 32)
    tme_channel_dict = {
        "cell_types": torch.randn(B, 3, H, W),
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

    def test_zero_init_residuals_preserve_identity(self):
        # proj is zero-initialized so TME fusion is identity at step 0.
        module = _make_module()
        mask_latent, tme_dict = _make_inputs()
        with torch.no_grad():
            out, residuals = module(mask_latent, tme_dict, return_residuals=True)
        for name, delta in residuals.items():
            torch.testing.assert_close(
                delta,
                torch.zeros_like(delta),
                atol=1e-6,
                rtol=1e-6,
                msg=f"group '{name}' residual should be zero at init",
            )
        torch.testing.assert_close(out, mask_latent, atol=1e-6, rtol=1e-6)

    def test_active_groups_subset(self):
        module = _make_module()
        mask_latent, tme_dict = _make_inputs()
        out = module(mask_latent, tme_dict, active_groups={"cell_types"})
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
        self.assertEqual(set(residuals.keys()), {"cell_types", "cell_state", "vasculature", "microenv"})
        for name, delta in residuals.items():
            self.assertEqual(delta.shape, (2, 16, 32, 32))

    def test_return_attn_weights(self):
        module = _make_module()
        mask_latent, tme_dict = _make_inputs()
        out, residuals, attn_maps = module(
            mask_latent, tme_dict, return_residuals=True, return_attn_weights=True
        )
        self.assertEqual(set(attn_maps.keys()), {"cell_types", "cell_state", "vasculature", "microenv"})
        for name, weights in attn_maps.items():
            self.assertEqual(weights.shape, (2, 4, 1024, 1024))

    def test_missing_group_in_dict_skipped(self):
        module = _make_module()
        mask_latent, _ = _make_inputs()
        partial_dict = {"cell_types": torch.randn(2, 3, 256, 256)}
        out = module(mask_latent, partial_dict)
        self.assertEqual(out.shape, (2, 16, 32, 32))

    def test_extreme_continuous_scale_keeps_post_stem_bounded(self):
        module = _make_module()
        module.eval()
        mask_latent, tme_dict = _make_inputs()
        tme_dict["microenv"][:, 0:1] = torch.randn_like(tme_dict["microenv"][:, 0:1]) * 1.0e6

        activations = {}

        def capture_post_stem(_module, _inputs, output):
            activations["post_stem"] = output.detach()

        handle = module.groups["microenv"].encoder.stem.register_forward_hook(capture_post_stem)
        try:
            with torch.no_grad():
                module(mask_latent, tme_dict, active_groups={"microenv"})
        finally:
            handle.remove()

        post_stem = activations["post_stem"]
        self.assertTrue(torch.isfinite(post_stem).all())
        self.assertLess(float(post_stem.abs().max().item()), 1.0e3)

    def test_n_params(self):
        module = _make_module()
        total = sum(p.numel() for p in module.parameters() if p.requires_grad)
        self.assertGreater(total, 1_000_000)
        self.assertLess(total, 4_000_000)


if __name__ == "__main__":
    unittest.main()
