import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import sys

sys.path.append('.')
from train_space import Config

# Load checkpoint to extract architecture
print("Loading checkpoint to extract architecture...")
checkpoint = torch.load('jinx_dns_surrogate.pt', map_location='cpu', weights_only=False)

model_state = checkpoint['model_state_dict']

# Extract architecture info
layer_info = {}
for name, param in model_state.items():
    layer_info[name] = {
        'shape': param.shape,
        'dtype': param.dtype
    }

print(f"Found {len(layer_info)} layers in checkpoint")

# Analyze architecture
print("\n=== ARCHITECTURE ANALYSIS ===")

# Extract key parameters
if 'position_embedding.weight' in layer_info:
    block_size = layer_info['position_embedding.weight']['shape'][0]
    n_embd = layer_info['position_embedding.weight']['shape'][1]
    print(f"Block size: {block_size}")
    print(f"Embedding dim: {n_embd}")

# Count layers
n_layer = sum(1 for name in layer_info.keys() if name.startswith('blocks.') and name.endswith('.ln_1.weight'))
print(f"Number of transformer blocks: {n_layer}")

# Check for bias in layer norm
has_ln_bias = any('ln_1.bias' in name for name in layer_info.keys())
print(f"Layer norm has bias: {has_ln_bias}")

# Check attention structure
attn_keys = [name for name in layer_info.keys() if 'attn.bias' in name]
if attn_keys:
    attn_bias_shape = layer_info[attn_keys[0]]['shape']
    print(f"Attention bias shape: {attn_bias_shape}")

# Check for specialized components
specialized = set()
for name in layer_info.keys():
    if 'ghost_lattice' in name: specialized.add('ghost_lattice')
    if 'ego_engine' in name: specialized.add('ego_engine')
    if 'third_eye' in name: specialized.add('third_eye')
    if 'symbolic_oracle' in name: specialized.add('symbolic_oracle')
    if 'reservoir' in name: specialized.add('reservoir')
    if 'replay_buffer' in name: specialized.add('replay_buffer')
    if 'restlessness' in name: specialized.add('restlessness')
    if 'plasticity_gate' in name: specialized.add('plasticity_gate')

print(f"Specialized components: {specialized}")

print("\n=== CREATING MODEL THAT MATCHES CHECKPOINT ===")

# Create a minimal model that loads only the core layers
class MinimalCheckpointModel(nn.Module):
    def __init__(self, layer_info):
        super().__init__()
        self.params = {}  # Regular dict to preserve exact names
        self.buffers = {}
        
        # Create ALL layers from checkpoint - no skipping
        for name, info in layer_info.items():
            
            # Create parameter or buffer based on dtype
            if len(info['shape']) == 0:
                # Use appropriate default value based on dtype
                if info['dtype'] in [torch.int32, torch.int64, torch.int16, torch.int8]:
                    tensor = torch.tensor(0, dtype=info['dtype'])
                elif info['dtype'] in [torch.bool]:
                    tensor = torch.tensor(False, dtype=info['dtype'])
                else:
                    tensor = torch.tensor(0.0, dtype=info['dtype'])
            else:
                tensor = torch.zeros(*info['shape'], dtype=info['dtype'])
            
            # Use Parameter for floating point, buffer for others
            if info['dtype'] in [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.complex64, torch.complex128]:
                self.params[name] = nn.Parameter(tensor)
            else:
                self.buffers[name] = tensor
    
    def forward(self, x):
        # This is just for loading weights, not actual inference
        return x

print("Creating minimal model structure...")
model = MinimalCheckpointModel(layer_info)

print(f"Model has {len(model.params)} parameters")

# Load weights
print("\nLoading weights...")
loaded = 0
skipped = 0
for name, param in model.params.items():
    if name in model_state:
        if param.shape == model_state[name].shape:
            param.data = model_state[name]
            loaded += 1
        else:
            skipped += 1
            print(f"Shape mismatch: {name} - {param.shape} vs {model_state[name].shape}")
    else:
        skipped += 1

# Also load buffers
for name, tensor in model.buffers.items():
    if name in model_state:
        if tensor.shape == model_state[name].shape:
            tensor.copy_(model_state[name])
            loaded += 1
        else:
            skipped += 1
            print(f"Shape mismatch (buffer): {name} - {tensor.shape} vs {model_state[name].shape}")
    else:
        skipped += 1

print(f"\n=== LOADING SUMMARY ===")
print(f"Loaded: {loaded} layers")
print(f"Skipped: {skipped} layers")
print(f"Total in checkpoint: {len(model_state)}")
print(f"Total in model: {len(model.params) + len(model.buffers)}")

if loaded > 0:
    print(f"\n✅ Successfully loaded {loaded}/{len(model_state)} layers")
else:
    print(f"\n❌ Failed to load any layers")
