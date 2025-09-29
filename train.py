"""
train.py
Trains the TinyTransformer on Tiny Shakespeare.
- Cross-entropy loss
- AdamW optimizer
- Logs train/val loss per epoch to CSV
- Saves best model and vocab
"""

import os, csv, requests, torch
from torch.utils.data import Dataset, DataLoader
from model import TinyTransformer

# ------------------------------
# Reproducibility and device
# ------------------------------
torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ------------------------------
# Configuration (according to homework specs)
# ------------------------------
batch_size = 64
block_size = 128
n_layers = 4
n_heads = 4
n_embed = 256
dropout = 0.1
lr = 3e-4
epochs = 5  # changing this with each training

# ------------------------------
# Dataset: Tiny Shakespeare
# ------------------------------
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("input.txt"):
    with open("input.txt", "wb") as f:
        f.write(requests.get(url, timeout=30).content)

text = open("input.txt", "r", encoding="utf-8").read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
def decode(t): return "".join(itos[int(i)] for i in t)

data = encode(text)
n = int(0.9*len(data))
train_data, val_data = data[:n], data[n:]

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    def __len__(self): return len(self.data) - self.block_size
    def __getitem__(self, idx):
        x = self.data[idx: idx+block_size]
        y = self.data[idx+1: idx+block_size+1]
        return x, y

def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)

train_loader = DataLoader(CharDataset(train_data, block_size), batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate)
val_loader   = DataLoader(CharDataset(val_data,   block_size), batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate)

# ------------------------------
# Model and using Adam optimizer
# ------------------------------
model = TinyTransformer(vocab_size, n_embed, n_layers, n_heads, block_size, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

@torch.no_grad()
def evaluate(loader):
    model.eval()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses)/len(losses)

# ------------------------------
# Training loop with CSV logging
# ------------------------------
best_val = float("inf")
with open("training_log.csv", "w", newline="") as fcsv:
    writer = csv.writer(fcsv)
    writer.writerow(["epoch", "train_loss", "val_loss"])
    for epoch in range(1, epochs+1):
        train_losses = []
        for i, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train = sum(train_losses)/len(train_losses)
        avg_val = evaluate(val_loader)
        print(f"Epoch {epoch} | Train {avg_train:.4f} | Val {avg_val:.4f}")
        writer.writerow([epoch, avg_train, avg_val])
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), "best_transformer.pt")
            torch.save(stoi, "stoi.pt")
            torch.save(itos, "itos.pt")
            print("✓ Saved best model")
