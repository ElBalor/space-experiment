import torch
import sys
import os
sys.path.append('.')

# Import Config from train_space.py before loading checkpoint (required for pickle)
from train_space import Config

# Load checkpoint
print("=== Loading Checkpoint ===")
chk = torch.load('jinx_dns_surrogate.pt', map_location='cpu', weights_only=False)

print("\n=== Checkpoint Keys ===")
for key in chk.keys():
    print(f"  - {key}")

print("\n=== File Size ===")
file_size = os.path.getsize('jinx_dns_surrogate.pt')
print(f"  {file_size / (1024**3):.2f} GB")

print("\n=== Config (from checkpoint) ===")
if 'config' in chk:
    config = chk['config']
    print(f"  Config type: {type(config)}")
    print(f"  Config value: {config}")
    if isinstance(config, dict):
        for key, value in config.items():
            print(f"  {key}: {value}")
    elif hasattr(config, '__dict__'):
        for key, value in config.__dict__.items():
            print(f"  {key}: {value}")
else:
    print("  No 'config' key found")

print("\n=== Training Step ===")
if 'step' in chk:
    print(f"  Step: {chk['step']}")
else:
    print("  No 'step' key found")

print("\n=== Timestamp ===")
if 'timestamp' in chk:
    print(f"  Timestamp: {chk['timestamp']}")
else:
    print("  No 'timestamp' key found")

print("\n=== Math Signature ===")
if 'math_signature' in chk:
    math_sig = chk['math_signature']
    if math_sig is not None:
        print(f"  Math Signature: {type(math_sig)}")
        if hasattr(math_sig, 'shape'):
            print(f"  Shape: {math_sig.shape}")
else:
    print("  No 'math_signature' key found")

print("\n=== Model State Dict (all keys and shapes) ===")
if 'model_state_dict' in chk:
    model_state = chk['model_state_dict']
    print(f"  Number of keys: {len(model_state)}")
    for key in model_state.keys():
        print(f"  {key}: {model_state[key].shape}")
elif 'model' in chk:
    model_state = chk['model']
    print(f"  Number of keys: {len(model_state)}")
    for key in model_state.keys():
        print(f"  {key}: {model_state[key].shape}")
else:
    print("  No 'model' or 'model_state_dict' key found")

print("\n=== Bridge State Dict (all keys and shapes) ===")
if 'bridge_state_dict' in chk:
    bridge_state = chk['bridge_state_dict']
    print(f"  Number of keys: {len(bridge_state)}")
    for key in bridge_state.keys():
        print(f"  {key}: {bridge_state[key].shape}")
elif 'bridge' in chk:
    bridge_state = chk['bridge']
    print(f"  Number of keys: {len(bridge_state)}")
    for key in bridge_state.keys():
        print(f"  {key}: {bridge_state[key].shape}")
else:
    print("  No 'bridge' or 'bridge_state_dict' key found")

print("\n=== Architecture Summary ===")
if 'model' in chk:
    model_state = chk['model']
    # Infer block_size from position embedding
    if 'position_embedding.weight' in model_state:
        pos_emb_shape = model_state['position_embedding.weight'].shape
        print(f"  Position Embedding: {pos_emb_shape}")
        print(f"  → Block size (tokens): {pos_emb_shape[0]}")
        print(f"  → Resolution (cube): {int(round(pos_emb_shape[0] ** (1/3)))}³")
    
    # Infer n_embd from first layer
    if 'blocks.0.ln1.weight' in model_state:
        n_embd = model_state['blocks.0.ln1.weight'].shape[0]
        print(f"  Embedding dimension: {n_embd}")
    
    # Infer n_layer
    n_layer = sum(1 for key in model_state.keys() if key.startswith('blocks.'))
    print(f"  Number of layers: {n_layer}")
