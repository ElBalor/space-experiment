"""
Head Lifecycle Manager for Guided Civilization Architecture.
"""
import torch
import random
from typing import Dict, Optional, List

class HeadMetrics:
    def __init__(self, head_idx: int, bias_type: str):
        self.head_idx = head_idx
        self.bias_type = bias_type
        self.entropy_ema = 2.0
        self.reward_corr_ema = 0.0
        self.activation_freq = 0.5
        self.promotion_count = 0
        self.demotion_count = 0
        self.age = 0
        self.last_promotion_step = -1
        self.last_demotion_step = -1

    def update(self, entropy: float, reward_signal: float, active: bool):
        alpha = 0.05
        self.entropy_ema = (1 - alpha) * self.entropy_ema + alpha * entropy
        if active:
            self.reward_corr_ema = (1 - alpha) * self.reward_corr_ema + alpha * reward_signal
            self.activation_freq = (1 - alpha) * self.activation_freq + alpha * 1.0
        else:
            self.activation_freq = (1 - alpha) * self.activation_freq
        self.age += 1

class HeadLifecycleManager:
    def __init__(self, n_heads, bias_types, promotion_threshold=0.7, demotion_threshold=0.2):
        self.n_heads = n_heads
        self.current_roles = bias_types.copy()
        self.metrics = [HeadMetrics(i, t) for i, t in enumerate(bias_types)]
        self.promotion_thresh = promotion_threshold
        self.demotion_thresh = demotion_threshold
        self.max_demotions = 3
        self.role_pool = ['local', 'global', 'causal', 'random', 'instruction', 'order']

    def update_metrics(self, head_idx, attention_probs, reward_signal=0.0):
        if head_idx >= len(self.metrics): return
        p = attention_probs.clamp_min(1e-8)
        entropy = -(p * p.log()).sum().item()
        active = attention_probs.max().item() > 0.1
        self.metrics[head_idx].update(entropy, reward_signal, active)

    def evaluate_promotion(self, head_idx, step):
        m = self.metrics[head_idx]
        if step - m.last_promotion_step < 100: return False
        if m.reward_corr_ema > self.promotion_thresh and m.activation_freq > 0.3:
            m.promotion_count += 1
            m.last_promotion_step = step
            m.reward_corr_ema *= 0.8
            return True
        return False

    def evaluate_demotion(self, head_idx, step):
        m = self.metrics[head_idx]
        if step - m.last_demotion_step < 100: return False
        if m.reward_corr_ema < self.demotion_thresh or m.activation_freq < 0.1:
            m.demotion_count += 1
            m.last_demotion_step = step
            return True
        return False

    def reassign_role(self, head_idx):
        m = self.metrics[head_idx]
        if m.demotion_count >= self.max_demotions:
            new_role = random.choice(self.role_pool)
            self.current_roles[head_idx] = new_role
            m.bias_type = new_role
            m.promotion_count = 0
            m.demotion_count = 0
            m.age = 0
            m.entropy_ema = 2.0
            m.reward_corr_ema = 0.0
            return new_role
        return None

    def get_plasticity_multipliers(self):
        mults = []
        for m in self.metrics:
            val = 1.0 + (m.promotion_count * 0.2) - (m.demotion_count * 0.2)
            mults.append(max(0.1, min(3.0, val)))
        return torch.tensor(mults)
