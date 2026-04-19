"""
WeibAct: Holographic activation function from p(M) physics.

The activation shape adapts to layer width:
  small layers → p→0 (nearly linear, preserve information)
  large layers → p→2/3 (strongly nonlinear, compress to boundary)

This mirrors the holographic principle: large systems encode
information on their boundary ((d-1)/d fraction).

Usage:
    python weibact.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import numpy as np


class WeibAct(nn.Module):
    """Holographic activation: f(x) = 1 - exp(-|x|^p) * sign(x)

    p = 2u/(1+3u), u = (width/M0)^(1/3)
    width = number of neurons in this layer
    M0 = learnable transition scale
    """

    def __init__(self, width, M0_init=100.0):
        super().__init__()
        self.width = width
        self.log_M0 = nn.Parameter(torch.tensor(np.log(M0_init)))

    def forward(self, x):
        M0 = torch.exp(self.log_M0)
        u = (self.width / M0) ** (1.0 / 3.0)
        p = 2.0 * u / (1.0 + 3.0 * u)
        p = torch.clamp(p, 0.05, 0.95)
        # f(x) = sign(x) * (1 - exp(-|x|^p))
        ax = torch.abs(x) + 1e-8
        return torch.sign(x) * (1.0 - torch.exp(-ax ** p))


class WeibActFixed(nn.Module):
    """WeibAct with fixed p (no learnable M0)."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        ax = torch.abs(x) + 1e-8
        return torch.sign(x) * (1.0 - torch.exp(-ax ** self.p))


def make_model(activation, hidden=256):
    """Simple MLP for MNIST."""
    if activation == 'relu':
        act1, act2 = nn.ReLU(), nn.ReLU()
    elif activation == 'weibact':
        act1 = WeibAct(hidden)
        act2 = WeibAct(hidden)
    elif activation == 'weib_fixed':
        act1 = WeibActFixed(0.5)
        act2 = WeibActFixed(0.5)
    else:
        act1, act2 = nn.ReLU(), nn.ReLU()

    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, hidden),
        act1,
        nn.Linear(hidden, hidden),
        act2,
        nn.Linear(hidden, 10),
    )


def train_and_eval(model, train_loader, test_loader, epochs=5, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            out = model(batch_x)
            pred = out.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)

    return correct / total


def main():
    print("=" * 60)
    print("WeibAct: Holographic Activation Function Benchmark")
    print("=" * 60)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST('data/mnist', train=True, download=True,
                                transform=transform)
    test_data = datasets.MNIST('data/mnist', train=False, transform=transform)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)

    results = {}
    for name in ['relu', 'weib_fixed', 'weibact']:
        t0 = time.time()
        accs = []
        for trial in range(3):
            model = make_model(name, hidden=256)
            acc = train_and_eval(model, train_loader, test_loader, epochs=5)
            accs.append(acc)

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        elapsed = time.time() - t0
        results[name] = (mean_acc, std_acc)

        # Get p value for WeibAct
        p_info = ""
        if name == 'weibact':
            for m in model.modules():
                if isinstance(m, WeibAct):
                    M0 = torch.exp(m.log_M0).item()
                    u = (m.width / M0) ** (1/3)
                    p = 2*u/(1+3*u)
                    p_info = f" p={p:.3f} M0={M0:.1f}"
                    break

        print(f"  {name:15s}: {mean_acc:.4f} +/- {std_acc:.4f} ({elapsed:.0f}s){p_info}")

    print()
    print("=" * 60)
    best = max(results, key=lambda k: results[k][0])
    print(f"Best: {best} ({results[best][0]:.4f})")

    # Width scaling test
    print()
    print("Width scaling (WeibAct p adapts to layer width):")
    for width in [32, 64, 128, 256, 512]:
        model = make_model('weibact', hidden=width)
        acc = train_and_eval(model, train_loader, test_loader, epochs=5)
        for m in model.modules():
            if isinstance(m, WeibAct):
                M0 = torch.exp(m.log_M0).item()
                u = (m.width / M0) ** (1/3)
                p = 2*u/(1+3*u)
                print(f"  width={width:4d}: acc={acc:.4f} p={p:.3f} M0={M0:.1f}")
                break


if __name__ == "__main__":
    main()
