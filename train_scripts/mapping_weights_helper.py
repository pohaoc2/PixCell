"""
Maps pretrained PixCell SD weights (safetensors) to PixCell ControlNet format.

Architecture differences:
  SD naming              ->  ControlNet naming
  ─────────────────────────────────────────────────────────────────────
  Patch embedders:
  pos_embed.proj             ->  x_embedder.proj        (1152, 16, 2, 2)
  cond_pos_embed.proj        ->  cond_embedder.proj     (1152, 16, 2, 2)

  Timestep embedder:
  adaln_single.emb.timestep_embedder.linear_1  ->  t_embedder.mlp.0
  adaln_single.emb.timestep_embedder.linear_2  ->  t_embedder.mlp.2
  adaln_single.linear                          ->  t_block.1            (6912, 1152)

  Caption / UNI embedder:
  caption_projection.linear_1                  ->  y_embedder.y_proj.fc1
  caption_projection.linear_2                  ->  y_embedder.y_proj.fc2
  caption_projection.uncond_embedding          ->  y_embedder.y_embedding

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

  Skipped (no equivalent in ControlNet architecture):
  pos_embed (standalone)     buffer not in safetensors; CN uses sincos init
  scale_shift_table (global) (2, 1152) — final layer, no equivalent in CN
  proj_out.*                 (128, 1152) — output projection not in CN
"""

from pathlib import Path

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


def _extract_state_dict(checkpoint_obj: dict) -> dict:
    """
    Extract a plausible model state dict from .pth-style checkpoint payloads.
    """
    if not isinstance(checkpoint_obj, dict):
        return checkpoint_obj
    for key in ("state_dict", "controlnet_state_dict", "model_state"):
        if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
            return checkpoint_obj[key]
    return checkpoint_obj


def _strip_known_prefixes(state_dict: dict) -> dict:
    """
    Remove common wrappers added by DDP/FSDP/checkpoint containers.
    """
    prefixes = ("module.", "model.", "_orig_mod.", "controlnet.")
    out = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        out[new_key] = value
    return out


def _filter_state_dict_for_model(model, state_dict: dict):
    """
    Keep only keys present in model with exact tensor shape match.
    """
    model_sd = model.state_dict()
    filtered = {}
    dropped = []
    for key, tensor in state_dict.items():
        if key not in model_sd:
            dropped.append((key, "missing in target model"))
            continue
        if tuple(tensor.shape) != tuple(model_sd[key].shape):
            dropped.append(
                (
                    key,
                    f"shape mismatch src={tuple(tensor.shape)} dst={tuple(model_sd[key].shape)}",
                )
            )
            continue
        filtered[key] = tensor
    return filtered, dropped


def _looks_like_diffusers_pixcell_sd(state_dict: dict) -> bool:
    """
    Heuristic: true when incoming safetensor keys are PixArt/PixCell base keys
    needing remap to ControlNet format.
    """
    probe_keys = (
        "pos_embed.proj.weight",
        "cond_pos_embed.proj.weight",
        "transformer_blocks.0.attn1.to_q.weight",
    )
    return any(key in state_dict for key in probe_keys)


def _map_sd_state_dict_to_controlnet(sd: dict, verbose: bool = True) -> tuple[dict, list]:
    """
    Remap an already-loaded SD/PixCell state dict to PixCell ControlNet keys.
    """
    mapped = {}
    skipped = []

    # ── 1. Simple direct renames ──────────────────────────────────────────────
    direct_map = {
        # Patch embedder
        "pos_embed.proj.weight": "x_embedder.proj.weight",
        "pos_embed.proj.bias": "x_embedder.proj.bias",
        "cond_pos_embed.proj.weight": "cond_embedder.proj.weight",
        "cond_pos_embed.proj.bias": "cond_embedder.proj.bias",
        # Timestep embedder
        "adaln_single.emb.timestep_embedder.linear_1.weight": "t_embedder.mlp.0.weight",
        "adaln_single.emb.timestep_embedder.linear_1.bias": "t_embedder.mlp.0.bias",
        "adaln_single.emb.timestep_embedder.linear_2.weight": "t_embedder.mlp.2.weight",
        "adaln_single.emb.timestep_embedder.linear_2.bias": "t_embedder.mlp.2.bias",
        "adaln_single.linear.weight": "t_block.1.weight",
        "adaln_single.linear.bias": "t_block.1.bias",
        # Caption / UNI embedder
        "caption_projection.linear_1.weight": "y_embedder.y_proj.fc1.weight",
        "caption_projection.linear_1.bias": "y_embedder.y_proj.fc1.bias",
        "caption_projection.linear_2.weight": "y_embedder.y_proj.fc2.weight",
        "caption_projection.linear_2.bias": "y_embedder.y_proj.fc2.bias",
        "caption_projection.uncond_embedding": "y_embedder.y_embedding",
    }
    for src, dst in direct_map.items():
        _copy(sd, mapped, skipped, src, dst)

    # ── pos_embed: drop CLS token ─────────────────────────────────────────────
    if "pos_embed" in sd:
        pe = sd["pos_embed"]
        if pe.shape[1] == 257:
            mapped["pos_embed"] = pe[:, 1:, :]
            if verbose:
                print(
                    f"  [cls-drop] pos_embed {tuple(pe.shape)} -> {tuple(mapped['pos_embed'].shape)}"
                )
        elif pe.shape[1] == 256:
            mapped["pos_embed"] = pe
        else:
            skipped.append(("pos_embed", f"unexpected shape {tuple(pe.shape)}"))
    else:
        skipped.append(("pos_embed", "not found in SD weights"))

    # ── 2. Transformer blocks: transformer_blocks.N -> blocks.N ──────────────
    num_blocks = 27
    for n in range(num_blocks):
        sd_p = f"transformer_blocks.{n}"
        cn_p = f"blocks.{n}"
        _copy(sd, mapped, skipped, f"{sd_p}.scale_shift_table", f"{cn_p}.scale_shift_table")
        _fuse(
            sd,
            mapped,
            skipped,
            [f"{sd_p}.attn1.to_q.weight", f"{sd_p}.attn1.to_k.weight", f"{sd_p}.attn1.to_v.weight"],
            f"{cn_p}.attn.qkv.weight",
        )
        _fuse(
            sd,
            mapped,
            skipped,
            [f"{sd_p}.attn1.to_q.bias", f"{sd_p}.attn1.to_k.bias", f"{sd_p}.attn1.to_v.bias"],
            f"{cn_p}.attn.qkv.bias",
        )
        _copy(sd, mapped, skipped, f"{sd_p}.attn1.to_out.0.weight", f"{cn_p}.attn.proj.weight")
        _copy(sd, mapped, skipped, f"{sd_p}.attn1.to_out.0.bias", f"{cn_p}.attn.proj.bias")
        _copy(
            sd, mapped, skipped, f"{sd_p}.attn2.to_q.weight", f"{cn_p}.cross_attn.q_linear.weight"
        )
        _copy(sd, mapped, skipped, f"{sd_p}.attn2.to_q.bias", f"{cn_p}.cross_attn.q_linear.bias")
        _fuse(
            sd,
            mapped,
            skipped,
            [f"{sd_p}.attn2.to_k.weight", f"{sd_p}.attn2.to_v.weight"],
            f"{cn_p}.cross_attn.kv_linear.weight",
        )
        _fuse(
            sd,
            mapped,
            skipped,
            [f"{sd_p}.attn2.to_k.bias", f"{sd_p}.attn2.to_v.bias"],
            f"{cn_p}.cross_attn.kv_linear.bias",
        )
        _copy(
            sd, mapped, skipped, f"{sd_p}.attn2.to_out.0.weight", f"{cn_p}.cross_attn.proj.weight"
        )
        _copy(sd, mapped, skipped, f"{sd_p}.attn2.to_out.0.bias", f"{cn_p}.cross_attn.proj.bias")
        _copy(sd, mapped, skipped, f"{sd_p}.ff.net.0.proj.weight", f"{cn_p}.mlp.fc1.weight")
        _copy(sd, mapped, skipped, f"{sd_p}.ff.net.0.proj.bias", f"{cn_p}.mlp.fc1.bias")
        _copy(sd, mapped, skipped, f"{sd_p}.ff.net.2.weight", f"{cn_p}.mlp.fc2.weight")
        _copy(sd, mapped, skipped, f"{sd_p}.ff.net.2.bias", f"{cn_p}.mlp.fc2.bias")

    # ── 3. ControlNet output projection blocks ────────────────────────────────
    for n in range(num_blocks):
        _copy(sd, mapped, skipped, f"controlnet_blocks.{n}.weight", f"controlnet_blocks.{n}.weight")
        _copy(sd, mapped, skipped, f"controlnet_blocks.{n}.bias", f"controlnet_blocks.{n}.bias")

    return mapped, skipped


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
    mapped, skipped = _map_sd_state_dict_to_controlnet(sd, verbose=verbose)

    # ── 4. Summary ────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*60}")
        print(f"Total SD keys:    {len(sd)}")
        print(f"Mapped tensors:   {len(mapped)}")
        print(f"Skipped entries:  {len(skipped)}")
        print(f"\nKnown skipped (shape mismatch / no equivalent):")
        known_skip = [
            "pos_embed",
            "cond_pos_embed",
            "adaln_single.linear",
            "scale_shift_table",
            "proj_out",
        ]
        for s in skipped:
            if any(k in str(s[0]) for k in known_skip):
                print(f"  {s[0]}: {s[1]}")
        unexpected_skips = [s for s in skipped if not any(k in str(s[0]) for k in known_skip)]
        if unexpected_skips:
            print(f"\nUnexpected skips (check these!):")
            for s in unexpected_skips:
                print(f"  {s[0]}: {s[1]}")

    if save_path:
        torch.save(mapped, save_path)
        print(f"\nSaved mapped weights to: {save_path}")

    return mapped


def load_controlnet_weights_flexible(controlnet, checkpoint_path: str, verbose: bool = True):
    """
    Robust ControlNet loader for both .pth and .safetensors payloads.
    Handles prefix mismatch and optional SD->ControlNet remapping automatically.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if path.suffix == ".safetensors":
        raw_state = load_file(str(path))
    else:
        raw_checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        raw_state = _extract_state_dict(raw_checkpoint)

    normalized = _strip_known_prefixes(raw_state)

    if _looks_like_diffusers_pixcell_sd(normalized):
        if verbose:
            print("Detected SD/PixCell transformer keys. Applying ControlNet remap.")
        remapped, _ = _map_sd_state_dict_to_controlnet(normalized, verbose=verbose)
        candidate = remapped
    else:
        candidate = normalized

    filtered, dropped = _filter_state_dict_for_model(controlnet, candidate)
    missing, unexpected = controlnet.load_state_dict(filtered, strict=False)

    if verbose:
        print("\n[Flexible ControlNet Load]")
        print(f"  source tensors:     {len(raw_state)}")
        print(f"  candidate tensors:  {len(candidate)}")
        print(f"  shape-matched load: {len(filtered)}")
        print(f"  dropped:            {len(dropped)}")
        print(f"  missing:            {len(missing)}")
        print(f"  unexpected:         {len(unexpected)}")
        if dropped:
            print(f"  dropped examples:   {dropped[:5]}")

    return {
        "loaded": len(filtered),
        "dropped": dropped,
        "missing": missing,
        "unexpected": unexpected,
    }


def load_model_weights_flexible(
    model,
    checkpoint_path: str,
    *,
    remap_pixcell_safetensors: bool = False,
    verbose: bool = True,
):
    """
    Robust generic model loader for .pth/.safetensors with prefix + shape filtering.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if path.suffix == ".safetensors":
        raw_state = load_file(str(path))
        if remap_pixcell_safetensors:
            # Import locally to avoid tight coupling at module import time.
            from diffusion.utils.checkpoint import remap_pixcell_to_pixart_alpha

            raw_state = remap_pixcell_to_pixart_alpha(raw_state)
    else:
        raw_checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        raw_state = _extract_state_dict(raw_checkpoint)

    normalized = _strip_known_prefixes(raw_state)
    filtered, dropped = _filter_state_dict_for_model(model, normalized)
    missing, unexpected = model.load_state_dict(filtered, strict=False)

    if verbose:
        print("\n[Flexible Generic Load]")
        print(f"  source tensors:     {len(raw_state)}")
        print(f"  shape-matched load: {len(filtered)}")
        print(f"  dropped:            {len(dropped)}")
        print(f"  missing:            {len(missing)}")
        print(f"  unexpected:         {len(unexpected)}")
        if dropped:
            print(f"  dropped examples:   {dropped[:5]}")

    return {
        "loaded": len(filtered),
        "dropped": dropped,
        "missing": missing,
        "unexpected": unexpected,
    }


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
    sd_path = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else "controlnet_mapped_weights.pt"
    map_sd_to_controlnet(sd_path, save_path)
