import torch


def test_concat_controlnet_accepts_raw_conditioning():
    from diffusion.model.nets.concat_controlnet import PixCell_ControlNet_XL_2_UNI_Concat

    model = PixCell_ControlNet_XL_2_UNI_Concat(
        input_size=32,
        hidden_size=128,
        controlnet_depth=2,
        num_heads=8,
        in_channels=16,
        caption_channels=1536,
        model_max_length=1,
    )
    model.eval()
    hidden_states = torch.randn(2, 16, 32, 32)
    conditioning = torch.randn(2, 10, 256, 256)
    encoder_hidden_states = torch.randn(2, 1, 1, 1536)
    timestep = torch.randint(0, 1000, (2,))

    outputs = model(
        hidden_states=hidden_states,
        conditioning=conditioning,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
    )

    block_samples = outputs[0]
    assert len(block_samples) == 2
    assert block_samples[0].shape == (2, 256, 128)


def test_concat_controlnet_token_count_matches_latent_path():
    from diffusion.model.nets.concat_controlnet import PixCell_ControlNet_XL_2_UNI_Concat

    model = PixCell_ControlNet_XL_2_UNI_Concat(
        input_size=32,
        hidden_size=128,
        controlnet_depth=1,
        num_heads=8,
        in_channels=16,
        caption_channels=1536,
        model_max_length=1,
    )

    assert model.x_embedder.num_patches == model.cond_embedder.num_patches == 256