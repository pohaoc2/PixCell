import unittest
import torch


class TestSplitChannelsToGroups(unittest.TestCase):
    def setUp(self):
        self.active_channels = [
            "cell_mask",
            "cell_type_healthy", "cell_type_cancer", "cell_type_immune",
            "cell_state_prolif", "cell_state_nonprolif", "cell_state_dead",
            "vasculature", "oxygen", "glucose",
        ]
        self.channel_groups = [
            dict(name="cell_types", channels=["cell_type_healthy", "cell_type_cancer", "cell_type_immune"]),
            dict(name="cell_state", channels=["cell_state_prolif", "cell_state_nonprolif", "cell_state_dead"]),
            dict(name="vasculature", channels=["vasculature"]),
            dict(name="microenv", channels=["oxygen", "glucose"]),
        ]
        self.control_input = torch.randn(2, 10, 4, 4)

    def test_split_returns_all_groups(self):
        from tools.channel_group_utils import split_channels_to_groups
        result = split_channels_to_groups(self.control_input, self.active_channels, self.channel_groups)
        self.assertEqual(set(result.keys()), {"cell_types", "cell_state", "vasculature", "microenv"})

    def test_split_shapes(self):
        from tools.channel_group_utils import split_channels_to_groups
        result = split_channels_to_groups(self.control_input, self.active_channels, self.channel_groups)
        self.assertEqual(result["cell_types"].shape, (2, 3, 4, 4))
        self.assertEqual(result["cell_state"].shape, (2, 3, 4, 4))
        self.assertEqual(result["vasculature"].shape, (2, 1, 4, 4))
        self.assertEqual(result["microenv"].shape, (2, 2, 4, 4))

    def test_split_values_correct(self):
        from tools.channel_group_utils import split_channels_to_groups
        result = split_channels_to_groups(self.control_input, self.active_channels, self.channel_groups)
        torch.testing.assert_close(result["cell_types"][:, 0], self.control_input[:, 1])
        torch.testing.assert_close(result["microenv"][:, 0], self.control_input[:, 8])

    def test_split_excludes_cell_mask(self):
        from tools.channel_group_utils import split_channels_to_groups
        result = split_channels_to_groups(self.control_input, self.active_channels, self.channel_groups)
        self.assertNotIn("cell_mask", result)


if __name__ == "__main__":
    unittest.main()
