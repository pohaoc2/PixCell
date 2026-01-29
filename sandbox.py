from safetensors.torch import load_file

# Load and inspect the checkpoint
checkpoint_path = "pretrained_models/pixcell-256/transformer/diffusion_pytorch_model.safetensors"
state_dict = load_file(checkpoint_path)

print("Checkpoint keys (first 30):")
for i, key in enumerate(list(state_dict.keys())[:30]):
    print(f"  {key}: {state_dict[key].shape}")
    
print(f"\nTotal keys: {len(state_dict)}")

# Check for specific key patterns
print("\nKey patterns found:")
patterns = ['adaln_single', 'caption_projection', 't_embedder', 'y_embedder', 
            'pos_embed', 'transformer_blocks', 'blocks']
for pattern in patterns:
    matching = [k for k in state_dict.keys() if pattern in k]
    if matching:
        print(f"  {pattern}: {len(matching)} keys")