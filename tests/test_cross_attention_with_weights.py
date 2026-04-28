import unittest
import torch


class TestCrossAttentionWithWeights(unittest.TestCase):
    def setUp(self):
        from diffusion.model.nets.cross_attention_with_weights import CrossAttentionWithWeights
        self.d_model = 16
        self.num_heads = 4
        self.attn = CrossAttentionWithWeights(d_model=self.d_model, num_heads=self.num_heads)
        self.B, self.N = 2, 64
        self.x = torch.randn(self.B, self.N, self.d_model)
        self.cond = torch.randn(self.B, self.N, self.d_model)

    def test_forward_without_weights_shape(self):
        out = self.attn(self.x, self.cond)
        self.assertEqual(out.shape, (self.B, self.N, self.d_model))

    def test_forward_with_weights_returns_tuple(self):
        out, weights = self.attn(self.x, self.cond, return_attn_weights=True)
        self.assertEqual(out.shape, (self.B, self.N, self.d_model))
        self.assertEqual(weights.shape, (self.B, self.num_heads, self.N, self.N))

    def test_weights_sum_to_one(self):
        _, weights = self.attn(self.x, self.cond, return_attn_weights=True)
        row_sums = weights.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5)

    def test_backward_compat_with_base_class(self):
        from diffusion.model.nets.PixArt_blocks import MultiHeadCrossAttention
        self.assertIsInstance(self.attn, MultiHeadCrossAttention)

    def test_base_attention_is_batch_isolated_without_mask(self):
        from diffusion.model.nets.PixArt_blocks import MultiHeadCrossAttention

        attn = MultiHeadCrossAttention(d_model=self.d_model, num_heads=self.num_heads)
        attn.eval()

        x = torch.randn(2, 8, self.d_model)
        cond = torch.randn(2, 8, self.d_model)

        with torch.no_grad():
            out1 = attn(x, cond)
            cond2 = cond.clone()
            cond2[1].mul_(1000)
            out2 = attn(x, cond2)

        torch.testing.assert_close(out1[0], out2[0], atol=1e-6, rtol=1e-6)
        self.assertGreater((out1[1] - out2[1]).abs().max().item(), 1e-3)

    @unittest.skipUnless(torch.cuda.is_available(), "masked xformers path requires CUDA")
    def test_base_attention_supports_flattened_conditioning_with_lengths(self):
        from diffusion.model.nets.PixArt_blocks import MultiHeadCrossAttention

        attn = MultiHeadCrossAttention(d_model=self.d_model, num_heads=self.num_heads).cuda()
        x = torch.randn(2, 8, self.d_model, device="cuda")
        cond = torch.randn(1, 10, self.d_model, device="cuda")
        lengths = [5, 5]

        out = attn(x, cond, lengths)

        self.assertEqual(out.shape, (2, 8, self.d_model))


if __name__ == "__main__":
    unittest.main()
