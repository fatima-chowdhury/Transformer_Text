"""
generate.py
Loads trained TinyTransformer and generates text.
Prompts with ROMEO: and JULIET: and saves 200-token samples.
"""

import torch
from model import TinyTransformer

# ------------------------------
# Match training config
# ------------------------------
block_size = 128
n_layers = 4
n_heads = 4
n_embed = 256
dropout = 0.1

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Load vocab and model weights
# ------------------------------
stoi = torch.load("stoi.pt", map_location="cpu")
itos = torch.load("itos.pt", map_location="cpu")
vocab_size = len(stoi)

def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
def decode(t): return "".join(itos[int(i)] for i in t)

model = TinyTransformer(vocab_size, n_embed, n_layers, n_heads, block_size, dropout).to(device)
model.load_state_dict(torch.load("best_transformer.pt", map_location=device))
model.eval()

# ------------------------------
# Generate function
# ------------------------------
@torch.no_grad()
def sample_prompt(prompt="ROMEO:", tokens=200, temperature=0.9, top_k=50):
    idx = encode(prompt).unsqueeze(0).to(device)
    out = model.generate(idx, max_new_tokens=tokens, temperature=temperature, top_k=top_k)
    return decode(out[0])

# ------------------------------
# Run generation
# ------------------------------
romeo = sample_prompt("ROMEO:")
juliet = sample_prompt("JULIET:")

print("\n=== ROMEO SAMPLE ===\n", romeo[:500])
print("\n=== JULIET SAMPLE ===\n", juliet[:500])

with open("sample_ROMEO.txt", "w", encoding="utf-8") as f: f.write(romeo)
with open("sample_JULIET.txt", "w", encoding="utf-8") as f: f.write(juliet)
print("\nSaved: sample_ROMEO.txt, sample_JULIET.txt")
