import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.n_embd // self.n_head
        self.block_size = config.block_size
        
        # Key parameters from checkpoint
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.k_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        
        # Attention bias - will be loaded from checkpoint
        self.bias = nn.Parameter(torch.zeros(1, 1, self.block_size, self.block_size))
        
    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.dend1 = nn.Linear(self.n_embd, 4 * self.n_embd, bias=True)
        self.dend2 = nn.Linear(4 * self.n_embd, 4 * self.n_embd, bias=True)
        self.down_proj = nn.Linear(4 * self.n_embd, self.n_embd, bias=True)
        
    def forward(self, x):
        x = F.gelu(self.dend1(x))
        x = F.gelu(self.dend2(x))
        return self.down_proj(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x, mask=None, temp_tokens=None, bridge=None):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MiniTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Position embedding - exact size from checkpoint
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = RMSNorm(config.n_embd)
        
    def forward(self, idx):
        B, T = idx.shape[:2]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos)
        x = idx + pos_emb
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        return x

class PhysicsBridge(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.proj_in = nn.Linear(3, n_embd)
        self.proj_out = nn.Linear(n_embd, 3)
        self.norm = nn.LayerNorm(n_embd, elementwise_affine=False)  # No bias to match checkpoint
    def forward_in(self, x): return self.norm(self.proj_in(x))
    def forward_out(self, x): return self.proj_out(x)

class Config:
    block_size = 4120  # Exact checkpoint size
    n_embd = 768
    n_head = 12
    n_layer = 12
    vocab_size = 151936
    dropout = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
