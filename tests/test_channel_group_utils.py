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


class TestApplyGroupDropout(unittest.TestCase):
    def test_returns_set_of_sets(self):
        from tools.channel_group_utils import apply_group_dropout
        group_names = ["cell_types", "cell_state", "vasculature", "microenv"]
        dropout_probs = dict(cell_types=0.0, cell_state=0.0, vasculature=0.0, microenv=0.0)
        result = apply_group_dropout(group_names, dropout_probs, batch_size=4)
        self.assertEqual(len(result), 4)
        for sample_groups in result:
            self.assertEqual(sample_groups, set(group_names))

    def test_full_dropout(self):
        from tools.channel_group_utils import apply_group_dropout
        group_names = ["cell_types", "cell_state", "vasculature", "microenv"]
        dropout_probs = dict(cell_types=1.0, cell_state=1.0, vasculature=1.0, microenv=1.0)
        result = apply_group_dropout(group_names, dropout_probs, batch_size=4)
        for sample_groups in result:
            self.assertEqual(sample_groups, set())

    def test_partial_dropout_returns_correct_length(self):
        from tools.channel_group_utils import apply_group_dropout
        group_names = ["cell_types", "cell_state"]
        dropout_probs = dict(cell_types=0.5, cell_state=0.5)
        result = apply_group_dropout(group_names, dropout_probs, batch_size=8)
        self.assertEqual(len(result), 8)
        for sample_groups in result:
            self.assertIsInstance(sample_groups, set)
            self.assertTrue(sample_groups.issubset(set(group_names)))


if __name__ == "__main__":
    unittest.main()
