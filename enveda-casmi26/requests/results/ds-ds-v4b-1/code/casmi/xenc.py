"""Cross-encoder reranker: joint transformer over (spectrum peak tokens, candidate atom tokens).

The spectrum side reuses a frozen FPNet encoder (its token states are projected to d); the
candidate side is a small graph transformer with shortest-path / bond attention bias. Joint
layers let peaks attend to atoms (and vice versa) so the score can reflect candidate-specific
peak explanations. Score = MLP(pooled) + w * (f.z of the FPNet) (residual on the bi-encoder).
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import fpnet, graphs

FEAT_SIZES = [graphs.N_ELEM, 2, 2, 7, 5, 6, 3, 4]   # elem, aromatic, in_ring, ring_size, n_h, degree, charge, hybrid


class BiasBlock(nn.Module):
    """Pre-LN transformer block with additive attention bias (B, H, T, T)."""
    def __init__(self, d, h, drop):
        super().__init__()
        self.h = h
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, bias):
        B, T, D = x.shape
        y = self.n1(x)
        q, k, v = self.qkv(y).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.ff(self.n2(x)))


class XEnc(nn.Module):
    def __init__(self, d_spec=512, d=256, layers=4, heads=8, drop=0.1):
        super().__init__()
        self.d = d; self.heads = heads; self.layers = layers; self.d_spec = d_spec
        self.proj = nn.Linear(d_spec, d)
        self.atom_emb = nn.ModuleList([nn.Embedding(n, d) for n in FEAT_SIZES])
        self.type_emb = nn.Embedding(2, d)
        self.spd_bias = nn.Embedding(graphs.MAX_SPD + 1, heads)
        self.bond_bias = nn.Embedding(graphs.N_BOND, heads)
        self.blocks = nn.ModuleList([BiasBlock(d, heads, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(3 * d + 2, 512), nn.GELU(), nn.Dropout(drop), nn.Linear(512, 1))
        self.fz_w = nn.Parameter(torch.tensor(1.0))

    def forward(self, spec_x, spec_pad, atoms, spd, bond, atom_pad, fz):
        """spec_x: (B, S, d_spec) frozen FPNet tokens (global first); spec_pad: (B, S) bool;
        atoms: (B, A, 8) long; spd/bond: (B, A, A) long; atom_pad: (B, A) bool; fz: (B,) float
        (FPNet f.z score, standardised within the candidate group). Returns scores (B,)."""
        B, S, _ = spec_x.shape; A = atoms.shape[1]
        xs = self.proj(spec_x) + self.type_emb.weight[0]
        xa = sum(emb(atoms[..., i]) for i, emb in enumerate(self.atom_emb)) + self.type_emb.weight[1]
        x = torch.cat([xs, xa], 1)
        T = S + A
        pad = torch.cat([spec_pad, atom_pad], 1)
        bias = torch.zeros(B, self.heads, T, T, device=x.device, dtype=x.dtype)
        ab = self.spd_bias(spd) + self.bond_bias(bond)            # (B, A, A, H)
        bias[:, :, S:, S:] = ab.permute(0, 3, 1, 2).to(x.dtype)
        bias = bias.masked_fill(pad[:, None, None, :], float('-inf'))
        for b in self.blocks:
            x = b(x, bias)
        x = self.norm(x)
        ms = (~spec_pad).float().unsqueeze(-1); ma = (~atom_pad).float().unsqueeze(-1)
        pool_s = (x[:, :S] * ms).sum(1) / ms.sum(1).clamp(min=1)
        pool_a = (x[:, S:] * ma).sum(1) / ma.sum(1).clamp(min=1)
        feat = torch.cat([x[:, 0], pool_s, pool_a, fz.unsqueeze(-1), (fz * fz).unsqueeze(-1)], -1)
        return self.head(feat).squeeze(-1) + self.fz_w * fz


def collate_graphs(items: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]], device):
    """items: list of (atoms (n,8), spd (n,n), bond (n,n)). Returns padded long tensors + pad mask."""
    B = len(items); A = max(1, max(len(a) for a, _, _ in items))
    atoms = np.zeros((B, A, graphs.N_ATOM_FEAT), np.int64); spd = np.full((B, A, A), graphs.MAX_SPD, np.int64)
    bond = np.zeros((B, A, A), np.int64); pad = np.ones((B, A), bool)
    for i, (a, s, b) in enumerate(items):
        n = len(a)
        atoms[i, :n] = a; spd[i, :n, :n] = s; bond[i, :n, :n] = b; pad[i, :n] = False
        if n == 0:
            pad[i, 0] = False
    T = lambda x: torch.as_tensor(x, device=device)
    return T(atoms), T(spd), T(bond), T(pad)


def save_ckpt(path, model, step, extra=None):
    d = dict(model=model.state_dict(), step=step, d=model.d, layers=model.layers, heads=model.heads, d_spec=model.d_spec)
    if extra:
        d.update(extra)
    torch.save(d, path)


def load_model(path, device='cuda'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = XEnc(d_spec=ck['d_spec'], d=ck['d'], layers=ck['layers'], heads=ck['heads'])
    m.load_state_dict(ck['model'])
    return m.to(device).eval()


class XScorer:
    """Scores (spectra, candidate graphs) at inference with a frozen FPNet + XEnc."""

    def __init__(self, fp_path, x_path, device='cuda'):
        self.device = device
        self.fp = fpnet.load_model(fp_path, device)
        self.x = load_model(x_path, device)

    @torch.no_grad()
    def score(self, spec_items: Sequence[dict], cand_graphs: Sequence, cand_fp: np.ndarray, bs_pairs=256) -> np.ndarray:
        """Returns (n_spec, n_cand) scores; cand_fp (n_cand, nbits) for f.z."""
        dev = self.device
        args = fpnet.collate(spec_items, dev)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
            sx, spad = self.fp.encode(*args)
            z = self.fp(*args)[0].float()
        fz = torch.as_tensor(cand_fp, device=dev, dtype=torch.float32) @ z.T        # (n_cand, n_spec)
        fzs = (fz - fz.mean(0, keepdim=True)) / (fz.std(0, keepdim=True) + 1e-6)
        gi = [g for g in cand_graphs]
        out = np.zeros((len(spec_items), len(cand_graphs)), np.float32)
        nS = len(spec_items); nC = len(cand_graphs)
        pairs = [(i, j) for i in range(nS) for j in range(nC)]
        for s in range(0, len(pairs), bs_pairs):
            chunk = pairs[s:s + bs_pairs]
            si = torch.as_tensor([p[0] for p in chunk], device=dev); cj = [p[1] for p in chunk]
            atoms, spd, bond, apad = collate_graphs([gi[j] for j in cj], dev)
            fzb = fzs[torch.as_tensor(cj, device=dev), si]
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
                sc = self.x(sx[si], spad[si], atoms, spd, bond, apad, fzb).float()
            for (i, j), v in zip(chunk, sc.cpu().numpy()):
                out[i, j] = v
        return out
