"""
WeibAct vs ReLU/GELU on enwiki8 character-level language modeling.

Small Transformer (6 layers, d=512) trained on first 90MB,
evaluated on last 5MB. Reports bits-per-character (BPC).

Usage:
    python weibact_enwiki8.py [--act relu|gelu|weibact] [--d_model 512] [--epochs 1]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import sys
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WeibAct(nn.Module):
    def __init__(self, width, M0_init=200.0):
        super().__init__()
        self.width = width
        self.log_M0 = nn.Parameter(torch.tensor(math.log(M0_init)))

    def forward(self, x):
        M0 = torch.exp(self.log_M0)
        u = (self.width / M0) ** (1.0 / 3.0)
        p = torch.clamp(2.0 * u / (1.0 + 3.0 * u), 0.05, 0.95)
        return torch.sign(x) * (1.0 - torch.exp(-(torch.abs(x) + 1e-8) ** p))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, act_fn):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            act_fn,
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x, mask=None):
        h = self.ln1(x)
        T = h.size(1)
        causal_mask = torch.triu(torch.ones(T, T, device=h.device), diagonal=1).bool()
        h, _ = self.attn(h, h, h, attn_mask=causal_mask)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x


class CharTransformer(nn.Module):
    def __init__(self, vocab, d_model, n_layers, n_heads, seq_len, act_name='gelu'):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.seq_len = seq_len

        def make_act():
            if act_name == 'relu':
                return nn.ReLU()
            elif act_name == 'gelu':
                return nn.GELU()
            elif act_name == 'weibact':
                return WeibAct(d_model * 4)
            return nn.GELU()

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, make_act())
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.embed(x) + self.pos(pos)
        for layer in self.layers:
            h = layer(h)
        h = self.ln_f(h)
        return self.head(h)


def load_enwiki8(path='data/enwik8', train_size=90_000_000):
    with open(path, 'rb') as f:
        data = f.read()
    # Use byte-level (256 vocab)
    train = torch.tensor(list(data[:train_size]), dtype=torch.long)
    test = torch.tensor(list(data[train_size:train_size + 5_000_000]), dtype=torch.long)
    return train, test, 256


def get_batch(data, seq_len, batch_size, device):
    n = len(data) - seq_len - 1
    idx = torch.randint(0, n, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in idx]).to(device)
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in idx]).to(device)
    return x, y


def evaluate(model, test_data, seq_len, batch_size, n_batches=50):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch(test_data, seq_len, batch_size, device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    return total_loss / total_tokens


def main():
    # Parse args
    act_name = 'gelu'
    d_model = 512
    n_epochs = 1
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--act' and i + 2 < len(sys.argv):
            act_name = sys.argv[i + 2]
        if arg == '--d_model' and i + 2 < len(sys.argv):
            d_model = int(sys.argv[i + 2])
        if arg == '--epochs' and i + 2 < len(sys.argv):
            n_epochs = int(sys.argv[i + 2])

    n_layers = 6
    n_heads = 8
    seq_len = 256
    batch_size = 32
    lr = 3e-4
    steps_per_epoch = 2000

    print(f"enwiki8 Benchmark: {act_name}, d={d_model}, layers={n_layers}, "
          f"heads={n_heads}, seq={seq_len}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load data
    train_data, test_data, vocab = load_enwiki8()
    print(f"Data: train={len(train_data):,} test={len(test_data):,} vocab={vocab}")

    # Build model
    model = CharTransformer(vocab, d_model, n_layers, n_heads, seq_len, act_name).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs * steps_per_epoch)

    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for step in range(steps_per_epoch):
            x, y = get_batch(train_data, seq_len, batch_size, device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if (step + 1) % 1000 == 0:
                avg = epoch_loss / (step + 1)
                bpc = avg / math.log(2)
                elapsed = time.time() - t0
                print(f"  step {step+1}/{steps_per_epoch}: loss={avg:.3f} bpc={bpc:.3f} "
                      f"({elapsed:.0f}s)", flush=True)

        # Evaluate
        test_loss = evaluate(model, test_data, seq_len, batch_size)
        test_bpc = test_loss / math.log(2)

        # WeibAct p values
        p_info = ""
        for m in model.modules():
            if isinstance(m, WeibAct):
                M0 = torch.exp(m.log_M0).item()
                u = (m.width / M0) ** (1 / 3)
                p = 2 * u / (1 + 3 * u)
                p_info = f" p={p:.3f} M0={M0:.0f}"
                break

        print(f"Epoch {epoch+1}: test_loss={test_loss:.3f} test_bpc={test_bpc:.3f} "
              f"({time.time()-t0:.0f}s){p_info}")

    print(f"\nFinal: {act_name} d={d_model} BPC={test_bpc:.4f} "
          f"params={n_params:,} time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
