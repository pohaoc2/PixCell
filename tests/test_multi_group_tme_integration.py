import unittest
import torch
from diffusion.utils.misc import read_config
from diffusion.model.builder import build_model
from tools.channel_group_utils import split_channels_to_groups, apply_group_dropout


class TestMultiGroupIntegration(unittest.TestCase):
    def setUp(self):
        self.config = read_config("configs/config_controlnet_exp.py")
        group_specs = [
            dict(name=g["name"], n_channels=len(g["channels"]))
            for g in self.config.channel_groups
        ]
        self.module = build_model(
            "MultiGroupTMEModule", False, False,
            channel_groups=group_specs, base_ch=32,
        )

    def test_config_to_module_to_forward(self):
        B = 2
        control_input = torch.randn(B, 10, 256, 256)
        mask_latent = torch.randn(B, 16, 32, 32)

        tme_dict = split_channels_to_groups(
            control_input,
            self.config.data.active_channels,
            self.config.channel_groups,
        )

        with torch.no_grad():
            fused = self.module(mask_latent, tme_dict)
        self.assertEqual(fused.shape, (B, 16, 32, 32))

    def test_group_dropout_integration(self):
        B = 4
        control_input = torch.randn(B, 10, 256, 256)
        mask_latent = torch.randn(B, 16, 32, 32)

        tme_dict = split_channels_to_groups(
            control_input,
            self.config.data.active_channels,
            self.config.channel_groups,
        )
        active_per_sample = apply_group_dropout(
            [g["name"] for g in self.config.channel_groups],
            self.config.group_dropout_probs,
            batch_size=B,
        )
        for b in range(B):
            for g in self.config.channel_groups:
                gname = g["name"]
                if gname not in active_per_sample[b] and gname in tme_dict:
                    tme_dict[gname][b] = 0.0

        with torch.no_grad():
            fused = self.module(mask_latent, tme_dict)
        self.assertEqual(fused.shape, (B, 16, 32, 32))

    def test_full_analysis_mode(self):
        B = 1
        control_input = torch.randn(B, 10, 256, 256)
        mask_latent = torch.randn(B, 16, 32, 32)

        tme_dict = split_channels_to_groups(
            control_input,
            self.config.data.active_channels,
            self.config.channel_groups,
        )

        with torch.no_grad():
            fused, residuals, attn_maps = self.module(
                mask_latent, tme_dict,
                return_residuals=True,
                return_attn_weights=True,
            )

        self.assertEqual(fused.shape, (B, 16, 32, 32))
        self.assertEqual(len(residuals), 4)
        self.assertEqual(len(attn_maps), 4)
        for name in ["cell_identity", "cell_state", "vasculature", "microenv"]:
            self.assertEqual(residuals[name].shape, (B, 16, 32, 32))
            self.assertEqual(attn_maps[name].shape, (B, 4, 1024, 1024))


if __name__ == "__main__":
    unittest.main()
