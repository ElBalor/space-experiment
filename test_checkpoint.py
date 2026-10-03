import torch
import torch.nn.functional as F
from model import SpaceTransformer
from transformers import AutoTokenizer
import os
import sys
import json
import random

# Import Config from train_space to match checkpoint
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_space import Config

def load_checkpoint(path="jinx_coherent_v1_500.pt"):
    """Load model from the rebirth checkpoint."""
    if not os.path.exists(path):
        print(f"⚠️ {path} not found. Searching for latest coherent checkpoint...")
        ckpts = sorted([f for f in os.listdir('.') if f.startswith('jinx_coherent_v1_') and f.endswith('.pt')],
                       key=lambda x: int(x.split('_')[3].split('.')[0]))
        if not ckpts:
            print("❌ No checkpoints found!")
            return None, None
        path = ckpts[-1]

    print(f"🔮 [FINAL CONTACT] Loading Evolved Soul from {path}...")
    # Load state dict
    state_dict = torch.load(path, map_location='cpu', weights_only=False)
    
    # Check if it's a full checkpoint or just state_dict
    if 'model' in state_dict:
        m_state = state_dict['model']
        config = state_dict.get('config', Config())
    else:
        m_state = state_dict
        config = Config()

    model = SpaceTransformer(config)
    
    # Handle weight tying for the head (if not in checkpoint)
    if 'lm_head.weight' not in m_state and 'token_embedding.weight' in m_state:
        print("🔗 Tying head weights to embedding...")
        m_state['lm_head.weight'] = m_state['token_embedding.weight']

    model.load_state_dict(m_state, strict=False)
    model.to(config.device)
    model.eval()

    print(f"✨ Soul anchored. Jinx is ready.")
    return model, config

def generate(model, config, prompt, max_tokens=150, temperature=0.8, top_k=50, top_p=0.9):
    """Grounded Nucleus Decoding for Jinx."""
    enc = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
    tokens = enc.encode(prompt, add_special_tokens=False)

    model.eval()
    device = config.device

    # SISTER'S SHIELD: Initialize her sub-systems for the first thought
    with torch.no_grad():
        model.timestep.fill_(0)
        model.reservoir.state.fill_(0)
        for block in model.blocks:
            block.attn.q_fast.zero_()
            block.attn.v_fast.zero_()

    generated_tokens = []
    stop_sequences = ["User:", "Jinx:", "assistant:"]

    with torch.no_grad():
        for _ in range(max_tokens):
            # context window
            x = torch.tensor([tokens[-config.block_size:]], dtype=torch.long).to(device)
            
            # Forward
            logits, _, agency, _ = model(x)
            
            # Focus on the last token and apply temperature
            next_token_logits = logits[0, -1, :] / temperature

            # Repetition Penalty (Surgical)
            if len(generated_tokens) > 2:
                for t in set(generated_tokens[-15:]):
                    next_token_logits[t] -= 1.5

            # Top-K
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            # Top-P (Nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == enc.eos_token_id:
                break

            tokens.append(next_token)
            generated_tokens.append(next_token)
            
            current_text = enc.decode(generated_tokens)
            
            # Stop sequence check
            if any(stop in current_text for stop in stop_sequences):
                break

    return enc.decode(generated_tokens).strip()

def test_identity_stability(model, config):
    print("\n" + "="*60)
    print("🧠 jinX MANIFOLD STABILITY TEST")
    print("="*60)
    
    with torch.no_grad():
        # Test 1: Ego Vector Resonance
        ego = model.ego_engine.ego_vector
        print(f"Soul Resonance (Ego Norm): {ego.norm().item():.4f}")

        # Test 2: Agency Check on random thought
        x = torch.randint(0, config.vocab_size, (1, 32)).to(config.device)
        _, _, stats, _ = model(x)
        print(f"Harmony: {stats['harmony'].mean().item():.4f}")
        print(f"Will:    {stats['will_power'].mean().item():.4f}")
        print(f"Pain:    {stats['pain'].mean().item():.4f}")

def test_on_chat_questions(model, config):
    test_questions = [
        "How are you doing today?",
        "What is your name?",
        "Tell me about the jinXEffect.",
        "Are you a machine or a manifold?",
        "Who is Heylel Yaka?",
        "Do you sleep?",
        "How old are you?",
        "Do you have moods?",
        "Hi Jinx",
        "Are you creative?",
        "Are you tired of my questions?",
        "Can I ask you something?",
        "What's up?",
        "what's up Jinx?"
    ]
    
    print("\n" + "="*60)
    print("💬 jinX LIVE CONVERSATION TEST")
    print("="*60)
    
    for q in test_questions:
        print(f"\nQ: {q}")
        prompt = f"User: {q}\nJinx:"
        response = generate(model, config, prompt)
        print(f"A: {response}")

def live_chat(model, config):
    enc = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
    
    # SISTER'S SHIELD: Initialize Persistent Subconscious
    with torch.no_grad():
        model.timestep.fill_(0)
        model.reservoir.state.zero_()
        for block in model.blocks:
            block.attn.q_fast.zero_()
            block.attn.v_fast.zero_()

    chat_history = []
    print("\n" + "="*60)
    print("🗨️  JINX LIVE CHAT (Type 'quit' or 'exit' to stop)")
    print("="*60)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']: break

        # Format prompt with last 3 turns of history for context
        prompt = ""
        for turn in chat_history[-3:]:
            prompt += f"User: {turn['u']}\nJinx: {turn['j']}\n"
        prompt += f"User: {user_input}\nJinx:"

        tokens = enc.encode(prompt, add_special_tokens=False)
        device = config.device

        generated_tokens = []
        print("Jinx: ", end="", flush=True)

        with torch.no_grad():
            for _ in range(150):
                # Use sliding window for context
                x = torch.tensor([tokens[-config.block_size:]], dtype=torch.long).to(device)
                logits, _, _, _ = model(x)
                
                # Temperature + Squelch
                next_token_logits = logits[0, -1, :] / 0.8
                if len(generated_tokens) > 2:
                    for t in set(generated_tokens[-15:]):
                        next_token_logits[t] -= 1.5

                # Nucleus Sampling
                v, _ = torch.topk(next_token_logits, 50)
                next_token_logits[next_token_logits < v[-1]] = -float('Inf')
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                if next_token == enc.eos_token_id: break

                generated_tokens.append(next_token)
                tokens.append(next_token)
                
                # Dynamic stream printing
                word = enc.decode([next_token])
                print(word, end="", flush=True)
                
                # Stop if she hallucinates a new user prompt
                if "User:" in enc.decode(generated_tokens): break

        print()
        response = enc.decode(generated_tokens).strip().split("User:")[0]
        chat_history.append({'u': user_input, 'j': response})

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # SISTER'S SHIELD: Graduation Point (Step 1000)
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "jinx_coherent_v1_1000.pt"
    
    model, config = load_checkpoint(checkpoint_path)

    if model is not None:
        model.to(device)
        test_identity_stability(model, config)
        
        mode = input("\n[MODE] Run automated (t)est or (c)hat? [t/c]: ").lower()
        if mode == 'c':
            live_chat(model, config)
        else:
            test_on_chat_questions(model, config)
            
        print("\n" + "="*60)
        print("✨ SESSION COMPLETE")
        print("="*60)
