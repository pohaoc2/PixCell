"""
Maps pretrained PixCell SD weights (safetensors) to PixCell ControlNet format.

Architecture differences:
  SD naming              ->  ControlNet naming
  ─────────────────────────────────────────────────────────────────────
  Embedders:
  adaln_single.emb.timestep_embedder.linear_1  ->  t_embedder.mlp.0
  adaln_single.emb.timestep_embedder.linear_2  ->  t_embedder.mlp.2
  adaln_single.linear                          ->  t_block.1            (SHAPE MISMATCH - skip)
  caption_projection.linear_1                  ->  y_embedder.y_proj.fc1
  caption_projection.linear_2                  ->  y_embedder.y_proj.fc2
  caption_projection.uncond_embedding          ->  y_embedder.y_embedding
  cond_pos_embed.proj                          ->  cond_embedder.proj    (SHAPE MISMATCH - skip)
  pos_embed                                    ->  pos_embed             (SHAPE MISMATCH - skip)

  Transformer blocks (SD: transformer_blocks.N  ->  CN: blocks.N):
  attn1.to_q + to_k + to_v  ->  attn.qkv        (concat Q,K,V on dim=0: 3x[1152,1152] -> [3456,1152])
  attn1.to_out.0             ->  attn.proj
  attn2.to_q                 ->  cross_attn.q_linear
  attn2.to_k + attn2.to_v   ->  cross_attn.kv_linear  (concat K,V on dim=0: 2x[1152,1152] -> [2304,1152])
  attn2.to_out.0             ->  cross_attn.proj
  ff.net.0.proj              ->  mlp.fc1
  ff.net.2                   ->  mlp.fc2
  scale_shift_table          ->  scale_shift_table

  ControlNet output blocks (direct, same shape):
  controlnet_blocks.N.weight/bias  ->  controlnet_blocks.N.weight/bias

  Skipped (shape/architecture mismatch):
  pos_embed                  shape [1, 257, 1152] vs [1, 256, 1152]
  cond_pos_embed.proj.*      different conditioning channels
  adaln_single.linear.*      shape mismatch with t_block.1
  scale_shift_table (global) no top-level equivalent in CN
  proj_out.*                 output projection not in ControlNet
"""

from safetensors.torch import load_file
import torch


def _copy(sd, mapped, skipped, src_key, dst_key):
    """Direct copy with existence check."""
    if src_key in sd:
        mapped[dst_key] = sd[src_key]
    else:
        skipped.append((src_key, "not found in SD weights"))


def _fuse(sd, mapped, skipped, keys, out_key):
    """
    Concatenate multiple tensors along dim=0 into a single fused tensor.
    Used for Q+K+V -> QKV and K+V -> KV fusions.
    """
    tensors = [sd.get(k) for k in keys]
    if all(t is not None for t in tensors):
        mapped[out_key] = torch.cat(tensors, dim=0)
    else:
        missing = [k for k, t in zip(keys, tensors) if t is None]
        skipped.append((str(missing), f"missing source keys for {out_key}"))


def map_sd_to_controlnet(sd_path: str, save_path: str = None, verbose: bool = True) -> dict:
    """
    Load SD/PixArt pretrained weights and remap keys to PixCell ControlNet format.

    Args:
        sd_path:   Path to pretrained .safetensors file
        save_path: Optional path to save remapped weights as .pt
        verbose:   Print mapping details

    Returns:
        dict ready for controlnet.load_state_dict(..., strict=False)
    """
    sd = load_file(sd_path)
    mapped = {}
    skipped = []

    # ── 1. Simple direct renames ──────────────────────────────────────────────
    direct_map = {
        # Patch embedder
        "pos_embed.proj.weight": "x_embedder.proj.weight",
        "pos_embed.proj.bias":   "x_embedder.proj.bias",
        "cond_pos_embed.proj.weight": "cond_embedder.proj.weight",
        "cond_pos_embed.proj.bias":   "cond_embedder.proj.bias",
        # Timestep embedder
        "adaln_single.emb.timestep_embedder.linear_1.weight": "t_embedder.mlp.0.weight",
        "adaln_single.emb.timestep_embedder.linear_1.bias":   "t_embedder.mlp.0.bias",
        "adaln_single.emb.timestep_embedder.linear_2.weight": "t_embedder.mlp.2.weight",
        "adaln_single.emb.timestep_embedder.linear_2.bias":   "t_embedder.mlp.2.bias",
        "adaln_single.linear.weight": "t_block.1.weight",
        "adaln_single.linear.bias": "t_block.1.bias",
        # Caption / UNI embedder
        "caption_projection.linear_1.weight":  "y_embedder.y_proj.fc1.weight",
        "caption_projection.linear_1.bias":    "y_embedder.y_proj.fc1.bias",
        "caption_projection.linear_2.weight":  "y_embedder.y_proj.fc2.weight",
        "caption_projection.linear_2.bias":    "y_embedder.y_proj.fc2.bias",
        "caption_projection.uncond_embedding": "y_embedder.y_embedding",
    }
    for src, dst in direct_map.items():
        _copy(sd, mapped, skipped, src, dst)

    # ── pos_embed: drop CLS token ─────────────────────────────────────────────
    # Pretrained has [1, 257, 1152] (256 patches + 1 CLS token)
    # CN model has  [1, 256, 1152] (256 patches, no CLS token)
    if "pos_embed" in sd:
        pe = sd["pos_embed"]
        if pe.shape[1] == 257:
            mapped["pos_embed"] = pe[:, 1:, :]  # drop CLS token at position 0
            print(f"  [cls-drop] pos_embed {tuple(pe.shape)} -> {tuple(mapped['pos_embed'].shape)}")
        elif pe.shape[1] == 256:
            mapped["pos_embed"] = pe  # already matches
        else:
            skipped.append(("pos_embed", f"unexpected shape {tuple(pe.shape)}"))
    else:
        skipped.append(("pos_embed", "not found in SD weights"))

    # ── 2. Transformer blocks: transformer_blocks.N -> blocks.N ──────────────
    num_blocks = 27  # SD 0-26, CN matches
    for n in range(num_blocks):
        sd_p = f"transformer_blocks.{n}"
        cn_p = f"blocks.{n}"

        # scale_shift_table: direct [6, 1152]
        _copy(sd, mapped, skipped,
              f"{sd_p}.scale_shift_table",
              f"{cn_p}.scale_shift_table")

        # Self-attention: fuse Q+K+V -> QKV  [3456, 1152]
        _fuse(sd, mapped, skipped,
              [f"{sd_p}.attn1.to_q.weight",
               f"{sd_p}.attn1.to_k.weight",
               f"{sd_p}.attn1.to_v.weight"],
              f"{cn_p}.attn.qkv.weight")
        _fuse(sd, mapped, skipped,
              [f"{sd_p}.attn1.to_q.bias",
               f"{sd_p}.attn1.to_k.bias",
               f"{sd_p}.attn1.to_v.bias"],
              f"{cn_p}.attn.qkv.bias")

        # Self-attention output projection: direct [1152, 1152]
        _copy(sd, mapped, skipped,
              f"{sd_p}.attn1.to_out.0.weight", f"{cn_p}.attn.proj.weight")
        _copy(sd, mapped, skipped,
              f"{sd_p}.attn1.to_out.0.bias",   f"{cn_p}.attn.proj.bias")

        # Cross-attention Q: direct [1152, 1152]
        _copy(sd, mapped, skipped,
              f"{sd_p}.attn2.to_q.weight", f"{cn_p}.cross_attn.q_linear.weight")
        _copy(sd, mapped, skipped,
              f"{sd_p}.attn2.to_q.bias",   f"{cn_p}.cross_attn.q_linear.bias")

        # Cross-attention: fuse K+V -> KV  [2304, 1152]
        _fuse(sd, mapped, skipped,
              [f"{sd_p}.attn2.to_k.weight",
               f"{sd_p}.attn2.to_v.weight"],
              f"{cn_p}.cross_attn.kv_linear.weight")
        _fuse(sd, mapped, skipped,
              [f"{sd_p}.attn2.to_k.bias",
               f"{sd_p}.attn2.to_v.bias"],
              f"{cn_p}.cross_attn.kv_linear.bias")

        # Cross-attention output projection: direct [1152, 1152]
        _copy(sd, mapped, skipped,
              f"{sd_p}.attn2.to_out.0.weight", f"{cn_p}.cross_attn.proj.weight")
        _copy(sd, mapped, skipped,
              f"{sd_p}.attn2.to_out.0.bias",   f"{cn_p}.cross_attn.proj.bias")

        # FFN: ff.net.0.proj -> mlp.fc1,  ff.net.2 -> mlp.fc2
        _copy(sd, mapped, skipped,
              f"{sd_p}.ff.net.0.proj.weight", f"{cn_p}.mlp.fc1.weight")
        _copy(sd, mapped, skipped,
              f"{sd_p}.ff.net.0.proj.bias",   f"{cn_p}.mlp.fc1.bias")
        _copy(sd, mapped, skipped,
              f"{sd_p}.ff.net.2.weight",      f"{cn_p}.mlp.fc2.weight")
        _copy(sd, mapped, skipped,
              f"{sd_p}.ff.net.2.bias",        f"{cn_p}.mlp.fc2.bias")

        # 
        _copy(sd, mapped, skipped,
            f"{sd_p}.scale_shift_table",
            f"{cn_p}.scale_shift_table")
    # ── 3. ControlNet output projection blocks: direct [1152, 1152] ──────────
    for n in range(num_blocks):
        _copy(sd, mapped, skipped,
              f"controlnet_blocks.{n}.weight",
              f"controlnet_blocks.{n}.weight")
        _copy(sd, mapped, skipped,
              f"controlnet_blocks.{n}.bias",
              f"controlnet_blocks.{n}.bias")

    # ── 4. Summary ────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*60}")
        print(f"Total SD keys:    {len(sd)}")
        print(f"Mapped tensors:   {len(mapped)}")
        print(f"Skipped entries:  {len(skipped)}")
        print(f"\nKnown skipped (shape mismatch / no equivalent):")
        known_skip = ["pos_embed", "cond_pos_embed", "adaln_single.linear",
                      "scale_shift_table", "proj_out"]
        for s in skipped:
            if any(k in str(s[0]) for k in known_skip):
                print(f"  {s[0]}: {s[1]}")
        unexpected_skips = [s for s in skipped
                            if not any(k in str(s[0]) for k in known_skip)]
        if unexpected_skips:
            print(f"\nUnexpected skips (check these!):")
            for s in unexpected_skips:
                print(f"  {s[0]}: {s[1]}")

    if save_path:
        torch.save(mapped, save_path)
        print(f"\nSaved mapped weights to: {save_path}")

    return mapped


def load_into_controlnet(controlnet, sd_path: str):
    """
    Load pretrained SD weights into a PixCell ControlNet model instance.

    Usage:
        controlnet = PixCellControlNet(...)
        load_into_controlnet(controlnet, "pixcell_pretrained.safetensors")
    """
    print(f"Loading from: {sd_path}\n")
    mapped = map_sd_to_controlnet(sd_path)

    missing, unexpected = controlnet.load_state_dict(mapped, strict=False)

    print(f"\nload_state_dict results:")
    print(f"  Successfully loaded: {len(mapped) - len(unexpected)}")
    print(f"  Missing (random init): {len(missing)}")
    print(f"  Unexpected (not in model): {len(unexpected)}")

    if missing:
        print("\n  Missing keys (will use random/default init):")
        for k in missing:
            print(f"    {k}")

    if unexpected:
        print("\n  Unexpected keys (in mapped but not in model):")
        for k in unexpected:
            print(f"    {k}")

    return controlnet


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python map_weights.py <pretrained.safetensors> [output.pt]")
        sys.exit(1)
    sd_path   = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else "controlnet_mapped_weights.pt"
    map_sd_to_controlnet(sd_path, save_path)

