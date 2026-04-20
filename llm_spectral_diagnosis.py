"""
LLM Spectral Diagnosis: measure effective dimension of each Transformer layer.

Uses CDT-inspired spectral dimension to analyze attention patterns:
- Each layer's attention matrix → directed graph → random walk → d_spec
- d_spec(layer) profile reveals bottlenecks and redundant layers

Usage:
    python llm_spectral_diagnosis.py [--model gpt2]
"""
import torch
import numpy as np
import time
import sys


def attention_to_graph(attn_weights, threshold=0.01):
    """Convert attention matrix to adjacency list.

    attn_weights: (n_heads, seq_len, seq_len)
    Returns: adjacency list (averaged over heads, thresholded)
    """
    # Average over heads
    avg_attn = attn_weights.mean(dim=0).cpu().numpy()  # (seq, seq)
    seq_len = avg_attn.shape[0]

    # Threshold: keep edges with attention > threshold
    adj = [[] for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(seq_len):
            if avg_attn[i, j] > threshold:
                adj[i].append(j)
        if not adj[i]:  # ensure at least self-loop
            adj[i].append(i)

    return adj


def spectral_dimension_rw(adj, n_walks=5000, sigma_max=50):
    """Measure spectral dimension via random walk return probability."""
    N = len(adj)
    P = np.zeros(sigma_max + 1)

    for _ in range(n_walks):
        start = np.random.randint(N)
        pos = start
        for sigma in range(1, sigma_max + 1):
            neighbors = adj[pos]
            pos = neighbors[np.random.randint(len(neighbors))]
            if pos == start:
                P[sigma] += 1

    P /= n_walks

    # Compute d_spec at a few sigma values
    d_specs = {}
    for s in [5, 10, 20, 30]:
        w = max(2, s // 4)
        lo = [P[j] for j in range(max(1, s-w), s+1) if P[j] > 0]
        hi = [P[j] for j in range(s, min(sigma_max, s+w)+1) if P[j] > 0]
        if lo and hi:
            Plo, Phi = np.mean(lo), np.mean(hi)
            if Plo > 0 and Phi > 0:
                d = -2 * (np.log(Phi) - np.log(Plo)) / \
                    (np.log(s + w/2) - np.log(max(1, s - w/2)))
                if 0 < d < 20:
                    d_specs[s] = d

    return d_specs, P


def analyze_model(model_name='gpt2', text=None):
    """Analyze all layers of a Transformer model."""
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TRANSFORMERS_NO_TF'] = '1'
    from transformers import GPT2Tokenizer, GPT2Model

    print(f"Loading {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name, output_attentions=True)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if text is None:
        text = ("The relationship between dark matter and galaxy rotation curves "
                "has been a subject of intense debate in modern astrophysics. "
                "The radial acceleration relation discovered by McGaugh suggests "
                "a deep connection between baryonic and observed gravitational "
                "acceleration that challenges our understanding of gravity. "
                "Whether this is evidence for modified gravity or a property of "
                "dark matter halos remains an open question in cosmology.")

    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    seq_len = inputs['input_ids'].shape[1]

    print(f"Sequence length: {seq_len} tokens")
    print(f"Device: {device}")

    with torch.no_grad():
        outputs = model(**inputs)

    attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]

    print(f"Layers: {n_layers}, Heads: {n_heads}")
    print()

    # Analyze each layer
    print(f"{'Layer':>6} {'d_spec(5)':>10} {'d_spec(10)':>11} {'d_spec(20)':>11} "
          f"{'Entropy':>8} {'Sparsity':>9} {'Diagnosis':>12}")
    print("-" * 72)

    layer_dspecs = []
    layer_entropies = []

    for layer_idx in range(n_layers):
        attn = attentions[layer_idx][0]  # (heads, seq, seq)

        # Attention entropy (information content)
        avg_attn = attn.mean(dim=0)  # (seq, seq)
        # Normalize rows
        avg_attn = avg_attn / (avg_attn.sum(dim=-1, keepdim=True) + 1e-10)
        entropy = -(avg_attn * torch.log(avg_attn + 1e-10)).sum(dim=-1).mean().item()

        # Sparsity (fraction of attention < 0.01)
        sparsity = (avg_attn < 0.01).float().mean().item()

        # Spectral dimension
        adj = attention_to_graph(attn, threshold=0.01)
        d_specs, P = spectral_dimension_rw(adj, n_walks=3000, sigma_max=40)

        d5 = d_specs.get(5, float('nan'))
        d10 = d_specs.get(10, float('nan'))
        d20 = d_specs.get(20, float('nan'))

        # Diagnosis
        if d10 < 1.5:
            diagnosis = "BOTTLENECK"
        elif d10 > 5:
            diagnosis = "REDUNDANT?"
        elif entropy < 1.0:
            diagnosis = "TOO FOCUSED"
        elif sparsity > 0.95:
            diagnosis = "SPARSE"
        else:
            diagnosis = "healthy"

        layer_dspecs.append(d10 if not np.isnan(d10) else 0)
        layer_entropies.append(entropy)

        print(f"{layer_idx:>6} {d5:>10.2f} {d10:>11.2f} {d20:>11.2f} "
              f"{entropy:>8.2f} {sparsity:>9.3f} {diagnosis:>12}")

    # Summary
    print()
    print("=" * 72)
    print("Summary:")
    print(f"  d_spec range: {min(layer_dspecs):.2f} - {max(layer_dspecs):.2f}")
    print(f"  Entropy range: {min(layer_entropies):.2f} - {max(layer_entropies):.2f}")

    # Find candidates for pruning
    mean_d = np.mean(layer_dspecs)
    candidates = [i for i, d in enumerate(layer_dspecs)
                  if d > mean_d * 1.5 or d < mean_d * 0.5]
    if candidates:
        print(f"  Pruning candidates (anomalous d_spec): layers {candidates}")
    else:
        print(f"  No obvious pruning candidates (all layers similar)")

    # Dimensional flow
    if len(layer_dspecs) >= 4:
        early = np.mean(layer_dspecs[:n_layers//3])
        mid = np.mean(layer_dspecs[n_layers//3:2*n_layers//3])
        late = np.mean(layer_dspecs[2*n_layers//3:])
        print(f"  Dimensional flow: early={early:.2f} → mid={mid:.2f} → late={late:.2f}")
        if early < mid < late:
            print(f"  Pattern: EXPANDING (information diffuses outward)")
        elif early > mid > late:
            print(f"  Pattern: COMPRESSING (information concentrates)")
        else:
            print(f"  Pattern: NON-MONOTONIC")

    print("=" * 72)


if __name__ == "__main__":
    model_name = 'gpt2'
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--model' and i + 2 <= len(sys.argv):
            model_name = sys.argv[i + 2]

    t0 = time.time()
    analyze_model(model_name)
    print(f"\nTotal time: {time.time()-t0:.0f}s")
