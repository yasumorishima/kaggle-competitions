"""Our fp2 (kernels/fp2: peak transformer, spectrum -> Morgan r2 2048-bit probabilities) as an extra re-scoring
term. Independent of the public FPNet (other architecture, trained by us with the enveda-np-examples and 400 random
enveda-180 molecules held out). score() returns {str(molecule_id): {smiles: log-likelihood}} for the given lists.
"""
import glob
import math

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch import nn

PEAKS, BITS = 64, 2048
ADDUCTS = ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
           "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"]
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=BITS)


class Sinus(nn.Module):
    def __init__(self, n=64, lo=1e-3, hi=2e3):
        super().__init__()
        self.register_buffer("w", 2 * math.pi / torch.logspace(math.log10(lo), math.log10(hi), n))

    def forward(self, x):
        a = x.unsqueeze(-1) * self.w
        return torch.cat([a.sin(), a.cos()], -1)


class FP2Net(nn.Module):
    def __init__(self, D=384):
        super().__init__()
        self.sin = Sinus()
        self.peak = nn.Sequential(nn.Linear(128 * 2 + 1, D), nn.GELU(), nn.Linear(D, D))
        self.prec = nn.Sequential(nn.Linear(128, D), nn.GELU(), nn.Linear(D, D))
        self.add = nn.Embedding(len(ADDUCTS) + 1, D)
        layer = nn.TransformerEncoderLayer(D, 8, 4 * D, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, 4)
        self.out = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1024), nn.GELU(), nn.Linear(1024, BITS))

    def forward(self, mz, it, prec, add):
        pad = mz <= 0
        tok = self.peak(torch.cat([self.sin(mz), self.sin((prec[:, None] - mz).clamp(min=0)), it[..., None]], -1))
        p = (self.prec(self.sin(prec)) + self.add(add))[:, None]
        x = torch.cat([p, tok], 1)
        mask = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
        return self.out(self.enc(x, src_key_padding_mask=mask)[:, 0])


def tokens(mz, it, prec):
    mz = np.asarray(mz, np.float32)
    it = np.asarray(it, np.float32)
    keep = (mz < prec + 1.0) & (it > 0)
    mz, it = mz[keep], it[keep]
    if len(mz) > PEAKS:
        top = np.argpartition(-it, PEAKS)[:PEAKS]
        mz, it = mz[top], it[top]
    M = np.zeros(PEAKS, np.float32)
    I = np.zeros(PEAKS, np.float32)
    M[:len(mz)] = mz
    if len(it):
        I[:len(it)] = np.log1p(1000.0 * it / max(it.max(), 1e-9))
    return M, I


def load(pattern="/kaggle/input/**/fp2.pt", only=None, device="cpu"):
    paths = sorted(glob.glob(pattern, recursive=True))
    if only:
        paths = [p for p in paths if any(o in p for o in only)]
    nets = []
    for p in paths:
        sd = torch.load(p, map_location="cpu")
        net = FP2Net(sd["add.weight"].shape[1])
        net.load_state_dict(sd)
        nets.append(net.eval().to(device))
    return paths, nets


def bits(smiles, cache):
    out = np.zeros((len(smiles), BITS), np.float32)
    ok = np.zeros(len(smiles), bool)
    for i, s in enumerate(smiles):
        v = cache.get(s)
        if v is None:
            m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            v = _gen.GetFingerprintAsNumPy(m).astype(np.float32) if m is not None else False
            cache[s] = v
        if v is not False:
            out[i] = v
            ok[i] = True
    return out, ok


def score(te, cand_lists, only=None, log=print, max_spec=16):
    """te: test DataFrame (molecule_id, ms2_mzs, ms2_normalized_intensities, precursor_mz, adduct);
    cand_lists: {molecule_id: [smiles]}. Probabilities are averaged over the molecule's spectra and the nets."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    paths, nets = load(only=only, device=dev)
    log("fp2 nets", paths)
    if not nets:
        return {}
    aidx = {a: i for i, a in enumerate(ADDUCTS)}
    out, cache = {}, {}
    for mid, g in te.groupby("molecule_id", sort=False):
        cl = cand_lists.get(mid) or cand_lists.get(str(mid))
        if not cl:
            continue
        g = g.iloc[:max_spec]
        tk = [tokens(r.ms2_mzs, r.ms2_normalized_intensities, float(r.precursor_mz)) for r in g.itertuples()]
        with torch.no_grad():
            args = (torch.from_numpy(np.stack([t[0] for t in tk])).to(dev),
                    torch.from_numpy(np.stack([t[1] for t in tk])).to(dev),
                    torch.from_numpy(g.precursor_mz.values.astype(np.float32)).to(dev),
                    torch.from_numpy(g.adduct.map(aidx).fillna(len(ADDUCTS)).values.astype(np.int64)).to(dev))
            p = torch.stack([torch.sigmoid(n(*args)) for n in nets]).mean(0).mean(0)
        p = p.clamp(1e-4, 1 - 1e-4).cpu().numpy().astype(np.float64)
        b, ok = bits(cl, cache)
        ll = b @ np.log(p) + (1 - b) @ np.log(1 - p)
        out[str(mid)] = {s: (float(v) if k else None) for s, v, k in zip(cl, ll, ok)}
    return out
