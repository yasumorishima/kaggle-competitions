"""CASMI26 dn1 on TPU (JAX training only; built by dn1tpu/build.py: 30 epochs, 1.7 h). dn1 (ours): formula-constrained de novo generation, spectrum + formula -> SMILES (GPU).

Why: the hidden class 3 is "not in PubChem"; our class-3 simulation showed that MMP generation from train
structures only reaches truths with a close parent (truth generated 0.286 with all parents, 0.053 when parents with
Tanimoto >= 0.5 to the truth are removed; the FP-guided multi-step walk 0.057). A generator that writes the
structure from the spectrum does not need a close parent.

Base (cited): the host's "CASMI denovo tutorial notebook" (inversion/casmi-denovo-tutorial-notebook): a peak
transformer encoder + autoregressive SMILES decoder, BPE tokens, unconstrained sampling.
Ours:
  1. atom-level SMILES tokens and decoding constrained by the molecular formula: a token that would exceed the
     formula's count of an element is masked, EOS only when every heavy atom is placed; then only candidates whose
     full formula (with H) equals the target are kept;
  2. the formula is also an encoder token (element counts), and the encoder is our fp2 peak transformer
     (exact m/z + neutral-loss sinusoids, precursor + adduct token);
  3. timsTOF (enveda-180) spectra first, as the hidden test is all timsTOF;
  4. a "novel" evaluation: every train structure with Tanimoto >= DN_NOVEL_T to a held-out NP structure is removed
     from training, the same condition as the MMP simulation at 0.5.
Held out and scored at the end (no submission): the enveda-np-examples structures (fp2's held-out np set).
Logs: valid / formula-correct rates, truth generated, MRR@25 of the generated list, best Tanimoto to the truth.
Writes dn1.pt (weights + vocab).
"""
import glob
import math
import os
import pickle
import re
import subprocess
import sys
import time

T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)


def first(pattern, default=None):
    hits = glob.glob(pattern, recursive=True)
    return os.path.dirname(hits[0]) if hits else default


COMP = os.environ.get("CASMI_COMP") or first("/kaggle/input/**/train.parquet")
COCO = os.environ.get("CASMI_COCO") or first("/kaggle/input/**/coco_meta.pkl")
WHEELS = first("/kaggle/input/**/rdkit-*.whl")
DN_WEIGHTS = glob.glob(os.environ.get("DN_WEIGHTS_GLOB", "/kaggle/input/**/dn1.pt"), recursive=True) \
    if os.environ.get("DN_INIT", "1") == "1" else []   # dn1tpu: present => continue training from it
GRAMMAR = os.environ.get("DN_GRAMMAR", "1") == "1"   # ours: parenthesis / ring-closure / bond constraints
if WHEELS:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-index",
                    *glob.glob(WHEELS + f"/*-cp{sys.version_info[0]}{sys.version_info[1]}-*.whl")], check=True)
log("comp", COMP, "coco", COCO, "wheels", WHEELS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402
from torch import nn  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec  # noqa: E402

RDLogger.DisableLog("rdApp.*")
DEV = "cpu"   # torch only builds the initial weights, writes dn1.pt and checks parity
EPOCHS = int(os.environ.get("DN_EPOCHS", "30"))
PER_STRUCT = 4
N_PEAKS = 64
FP_BITS = 2048
D = int(os.environ.get("DN_D", "384"))
SEED = 0
PPM = 10.0
TEST_ADDUCTS = ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
                "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"]
PROTON = 1.007276467
ADDUCTS = {
    "[M+H]+": (1, PROTON), "[M+NH4]+": (1, 18.033823), "[M-H2O+H]+": (1, PROTON - 18.010565),
    "[M-2H2O+H]+": (1, PROTON - 2 * 18.010565), "[M+Na]+": (1, 22.989218), "[M+K]+": (1, 38.963158),
    "[M-H]-": (1, -PROTON), "[M-H2O-H]-": (1, -PROTON - 18.010565),
    "[M+CH2O2-H]-": (1, 44.998201), "[M+Cl]-": (1, 34.969402),
}
MONO = {"C": 12.0, "H": 1.00782503207, "N": 14.0030740048, "O": 15.99491461956,
        "P": 30.97376163, "S": 31.97207100, "F": 18.99840322, "Cl": 34.96885268,
        "Br": 78.9183371, "I": 126.904473, "Si": 27.9769265325, "B": 11.0093054,
        "Se": 79.9165213, "Na": 22.9897692809, "K": 38.96370668}
_FORM = re.compile(r"([A-Z][a-z]?)(\d*)")


def neutral_mass(mz, adduct):
    a = ADDUCTS.get(adduct)
    return np.nan if a is None else (mz - a[1]) / a[0]


def formula_mass(f):
    if not isinstance(f, str) or any(c in f for c in "+-"):
        return np.nan
    m = 0.0
    for el, n in _FORM.findall(f):
        if el not in MONO:
            return np.nan
        m += MONO[el] * (int(n) if n else 1)
    return m


_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)


def fp_bits(smiles):
    out = np.zeros((len(smiles), FP_BITS // 8), np.uint8)
    ok = np.zeros(len(smiles), bool)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is not None:
            out[i] = np.packbits(_gen.GetFingerprintAsNumPy(m).astype(np.uint8))
            ok[i] = True
    return out, ok


def load_tokens(path, rows, prec):
    """Top-N_PEAKS peaks per row (by intensity, below the precursor + 1 Da) as (mz, logI)."""
    order = np.argsort(rows)
    rows_s = rows[order]
    MZ = np.zeros((len(rows), N_PEAKS), np.float32)
    IT = np.zeros((len(rows), N_PEAKS), np.float32)
    f = pq.ParquetFile(path)
    start = 0
    for g in range(f.num_row_groups):
        n = f.metadata.row_group(g).num_rows
        lo, hi = np.searchsorted(rows_s, [start, start + n])
        if hi > lo:
            t = f.read_row_group(g, columns=["ms2_mzs", "ms2_normalized_intensities"]).take(rows_s[lo:hi] - start)
            a, b = t.column(0).combine_chunks(), t.column(1).combine_chunks()
            for j, (x, y) in enumerate(zip(a, b)):
                k = order[lo + j]
                mz = x.values.to_numpy(zero_copy_only=False).astype(np.float32)
                it = y.values.to_numpy(zero_copy_only=False).astype(np.float32)
                keep = (mz < prec[k] + 1.0) & (it > 0)
                mz, it = mz[keep], it[keep]
                if len(mz) > N_PEAKS:
                    top = np.argpartition(-it, N_PEAKS)[:N_PEAKS]
                    mz, it = mz[top], it[top]
                MZ[k, :len(mz)] = mz
                IT[k, :len(mz)] = np.log1p(1000.0 * it / max(it.max(), 1e-9)) if len(it) else 0
            del t
        start += n
    return MZ, IT


class Sinus(nn.Module):
    def __init__(self, n=64, lo=1e-3, hi=2e3):
        super().__init__()
        self.register_buffer("w", 2 * math.pi / torch.logspace(math.log10(lo), math.log10(hi), n))

    def forward(self, x):
        a = x.unsqueeze(-1) * self.w
        return torch.cat([a.sin(), a.cos()], -1)



class FP2Net(nn.Module):
    """our fp2 (kernels/fp2): spectrum -> Morgan r2 2048-bit logits; same token format as load_tokens"""
    def __init__(self, D=384):
        super().__init__()
        self.sin = Sinus()
        self.peak = nn.Sequential(nn.Linear(128 * 2 + 1, D), nn.GELU(), nn.Linear(D, D))
        self.prec = nn.Sequential(nn.Linear(128, D), nn.GELU(), nn.Linear(D, D))
        self.add = nn.Embedding(10 + 1, D)
        layer = nn.TransformerEncoderLayer(D, 8, 4 * D, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, 4)
        self.out = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1024), nn.GELU(), nn.Linear(1024, 2048))

    def forward(self, mz, it, prec, add):
        pad = mz <= 0
        tok = self.peak(torch.cat([self.sin(mz), self.sin((prec[:, None] - mz).clamp(min=0)), it[..., None]], -1))
        p = (self.prec(self.sin(prec)) + self.add(add))[:, None]
        mask = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
        return self.out(self.enc(torch.cat([p, tok], 1), src_key_padding_mask=mask)[:, 0])


# ours: re-rank the generated list with fp2's predicted fingerprint (fp2 held out the same np-examples structures)
FP2_PATHS = sorted(glob.glob(os.environ.get("DN_FP2_GLOB", "/kaggle/input/**/fp2.pt"), recursive=True))


N_SAMPLES = int(os.environ.get("DN_SAMPLES", "48"))      # per spectrum at evaluation
MAX_TOK = 100
NOVEL_T = float(os.environ.get("DN_NOVEL_T", "0.5"))
ELEMS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Si", "B", "Se"]
FORM_ELEMS = ELEMS + ["H"]
_TOK = re.compile(r"(\[[^\]]+]|Br|Cl|Si|Se|N|O|S|P|F|I|B|C|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])")
_BR = re.compile(r"\[(\d*)([A-Z][a-z]?|[a-z][a-z]?)")
PAD, BOS, EOS = 0, 1, 2


def tokenize(s):
    t = _TOK.findall(s)
    return t if "".join(t) == s else None


def tok_elem(t):
    """element of an atom token (None for bonds, rings, branches); 'X' for an element we do not track"""
    if t.startswith("["):
        m = _BR.match(t)
        if not m:
            return "X"
        e = m.group(2)
        e = e[0].upper() + e[1:] if e.islower() else e
        if e.lower() in ("se", "as", "te"):
            e = e[0].upper() + e[1:].lower()
        return e if e in ELEMS else "X"
    if t in ("Br", "Cl", "Si", "Se"):
        return t
    if t in ("b", "c", "n", "o", "s", "p"):
        return t.upper()
    if t in ("B", "C", "N", "O", "S", "P", "F", "I"):
        return t
    return None


def form_counts(f):
    c = dict.fromkeys(FORM_ELEMS, 0)
    if not isinstance(f, str):
        return None
    for el, n in _FORM.findall(f):
        if el not in c:
            return None
        c[el] += int(n) if n else 1
    return np.array([c[e] for e in FORM_ELEMS], np.float32)


class Net(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.sin = Sinus()
        self.peak = nn.Sequential(nn.Linear(128 * 2 + 1, D), nn.GELU(), nn.Linear(D, D))
        self.prec = nn.Sequential(nn.Linear(128, D), nn.GELU(), nn.Linear(D, D))
        self.add = nn.Embedding(len(TEST_ADDUCTS) + 1, D)
        self.form = nn.Sequential(nn.Linear(len(FORM_ELEMS) * 2, D), nn.GELU(), nn.Linear(D, D))
        layer = nn.TransformerEncoderLayer(D, 8, 4 * D, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, 4)
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Embedding(MAX_TOK + 2, D)
        dl = nn.TransformerDecoderLayer(D, 8, 4 * D, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.dec = nn.TransformerDecoder(dl, 4)
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, V))

    def encode(self, mz, it, prec, add, fc):
        pad = mz <= 0
        tok = self.peak(torch.cat([self.sin(mz), self.sin((prec[:, None] - mz).clamp(min=0)), it[..., None]], -1))
        p = (self.prec(self.sin(prec)) + self.add(add))[:, None]
        f = self.form(torch.cat([torch.log1p(fc), (fc > 0).float()], -1))[:, None]
        x = torch.cat([p, f, tok], 1)
        mask = torch.cat([torch.zeros_like(pad[:, :2]), pad], 1)
        return self.enc(x, src_key_padding_mask=mask), mask

    def decode(self, mem, mmask, y):
        L = y.shape[1]
        h = self.emb(y) + self.pos(torch.arange(L, device=y.device))[None]
        causal = torch.triu(torch.full((L, L), float("-inf"), device=y.device), 1)
        return self.head(self.dec(h, mem, tgt_mask=causal, tgt_key_padding_mask=y == PAD, memory_key_padding_mask=mmask))


# ---- dn1's Net in JAX, on the torch state-dict names (norm-first encoder / decoder, 8 heads, GELU)
N_HEADS = 8
LSMOOTH = 0.05


def _lin(p, k, x):
    return x @ p[k + ".weight"].T + p[k + ".bias"]


def _ln(p, k, x):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + 1e-5) * p[k + ".weight"] + p[k + ".bias"]


def _gelu(x):
    return jax.nn.gelu(x, approximate=False)


def _sin(p, x):
    a = x[..., None] * p["sin.w"]
    return jnp.concatenate([jnp.sin(a), jnp.cos(a)], -1)


def _drop(x, key, rate=0.1):
    if key is None:
        return x
    return jnp.where(jax.random.bernoulli(key, 1 - rate, x.shape), x / (1 - rate), 0.0)


def _mha(p, k, xq, xkv, blocked, key):
    """torch nn.MultiheadAttention (batch_first); blocked: bool, broadcastable to (B, H, Tq, Tk)"""
    W, b = p[k + ".in_proj_weight"], p[k + ".in_proj_bias"]
    B, T, Dm = xq.shape
    S = xkv.shape[1]
    dh = Dm // N_HEADS
    q = (xq @ W[:Dm].T + b[:Dm]).reshape(B, T, N_HEADS, dh)
    kk = (xkv @ W[Dm:2 * Dm].T + b[Dm:2 * Dm]).reshape(B, S, N_HEADS, dh)
    v = (xkv @ W[2 * Dm:].T + b[2 * Dm:]).reshape(B, S, N_HEADS, dh)
    a = jnp.einsum("bthd,bshd->bhts", q, kk) / jnp.sqrt(dh)
    a = jax.nn.softmax(jnp.where(blocked, -1e9, a), -1)
    a = _drop(a, key)
    o = jnp.einsum("bhts,bshd->bthd", a, v).reshape(B, T, Dm)
    return _lin(p, k + ".out_proj", o)


def _nl(p, pre):
    return sum(1 for k in p if k.startswith(pre) and k.endswith(".norm1.weight"))


def jencode(p, mz, it, prec, add, fc, key=None):
    nl = _nl(p, "enc.layers.")
    keys = list(jax.random.split(key, 4 * nl)) if key is not None else [None] * (4 * nl)
    pad = mz <= 0
    tok = _lin(p, "peak.2", _gelu(_lin(p, "peak.0", jnp.concatenate(
        [_sin(p, mz), _sin(p, jnp.clip(prec[:, None] - mz, 0)), it[..., None]], -1))))
    pp = _lin(p, "prec.2", _gelu(_lin(p, "prec.0", _sin(p, prec)))) + p["add.weight"][add]
    f = _lin(p, "form.2", _gelu(_lin(p, "form.0", jnp.concatenate([jnp.log1p(fc), (fc > 0).astype(fc.dtype)], -1))))
    x = jnp.concatenate([pp[:, None], f[:, None], tok], 1)
    mask = jnp.concatenate([jnp.zeros_like(pad[:, :2]), pad], 1)
    blocked = mask[:, None, None, :]
    for l in range(nl):
        q = f"enc.layers.{l}."
        x = x + _drop(_mha(p, q + "self_attn", _ln(p, q + "norm1", x), _ln(p, q + "norm1", x), blocked, keys[4 * l]),
                      keys[4 * l + 1])
        h = _drop(_gelu(_lin(p, q + "linear1", _ln(p, q + "norm2", x))), keys[4 * l + 2])
        x = x + _drop(_lin(p, q + "linear2", h), keys[4 * l + 3])
    return x, mask


def jdecode(p, mem, mmask, y, key=None):
    nl = _nl(p, "dec.layers.")
    keys = list(jax.random.split(key, 6 * nl)) if key is not None else [None] * (6 * nl)
    L = y.shape[1]
    x = p["emb.weight"][y] + p["pos.weight"][:L][None]
    causal = jnp.triu(jnp.ones((L, L), bool), 1)
    self_blocked = causal[None, None] | (y == PAD)[:, None, None, :]
    mem_blocked = mmask[:, None, None, :]
    for l in range(nl):
        q = f"dec.layers.{l}."
        h = _ln(p, q + "norm1", x)
        x = x + _drop(_mha(p, q + "self_attn", h, h, self_blocked, keys[6 * l]), keys[6 * l + 1])
        x = x + _drop(_mha(p, q + "multihead_attn", _ln(p, q + "norm2", x), mem, mem_blocked, keys[6 * l + 2]),
                      keys[6 * l + 3])
        h = _drop(_gelu(_lin(p, q + "linear1", _ln(p, q + "norm3", x))), keys[6 * l + 4])
        x = x + _drop(_lin(p, q + "linear2", h), keys[6 * l + 5])
    return _lin(p, "head.1", _ln(p, "head.0", x))


def jlogits(p, mz, it, prec, add, fc, y):
    mem, mm = jencode(p, mz, it, prec, add, fc)
    return jdecode(p, mem, mm, y)


def jloss_sums(p, mz, it, prec, add, fc, y, key=None):
    """sum over non-PAD target tokens of torch's CrossEntropyLoss(label_smoothing) terms, of the plain NLL, count"""
    k1, k2 = jax.random.split(key) if key is not None else (None, None)
    mem, mm = jencode(p, mz, it, prec, add, fc, k1)
    lg = jdecode(p, mem, mm, y[:, :-1], k2).astype(jnp.float32)
    tgt = y[:, 1:]
    valid = (tgt != PAD).astype(jnp.float32)
    lsm = jax.nn.log_softmax(lg, -1)
    nll = -jnp.take_along_axis(lsm, tgt[..., None], -1)[..., 0]
    per = (1 - LSMOOTH) * nll + LSMOOTH * (-lsm.mean(-1))
    return (per * valid).sum(), (nll * valid).sum(), valid.sum()


def main():
    from rdkit import DataStructs
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    meta = pd.read_parquet(COMP + "/train.parquet",
                           columns=["ingest_lib", "inchikey14", "normalized_smiles", "molecular_formula",
                                    "precursor_mz", "adduct", "instrument_type"])
    meta["row"] = np.arange(len(meta))
    log("train", len(meta))
    rng = np.random.default_rng(2026)
    npx = set(meta.loc[meta.ingest_lib == "enveda-np-examples", "inchikey14"])
    held = set(npx)

    # ---- the "novel" condition: drop every train structure close to a held-out NP structure
    S_all = meta.drop_duplicates("inchikey14")[["inchikey14", "normalized_smiles", "molecular_formula"]]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    hfp = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in S_all[S_all.inchikey14.isin(npx)].normalized_smiles
           if Chem.MolFromSmiles(s) is not None]
    near = set()
    if NOVEL_T < 1.0:
        for ik, s in zip(S_all.inchikey14.values, S_all.normalized_smiles.values):
            if ik in npx:
                continue
            m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            if m is None:
                continue
            if max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m), hfp)) >= NOVEL_T:
                near.add(ik)
    log("novel condition: train structures within Tanimoto", NOVEL_T, "of a held-out NP structure removed:", len(near))

    tr = meta[meta.adduct.isin(TEST_ADDUCTS) & ~meta.inchikey14.isin(held | near)].copy()
    tr["pri"] = (tr.instrument_type == "timsTOF").astype(int)
    tr = tr.sample(frac=1, random_state=0).sort_values("pri", ascending=False, kind="stable")
    tr = tr.groupby("inchikey14").head(PER_STRUCT)
    # vocabulary and token ids per structure
    S = tr.drop_duplicates("inchikey14")
    toks, fcs = {}, {}
    from collections import Counter
    cnt = Counter()
    for ik, s, f in zip(S.inchikey14, S.normalized_smiles, S.molecular_formula):
        t = tokenize(s) if isinstance(s, str) else None
        fc = form_counts(f)
        if t is None or fc is None or len(t) > MAX_TOK or any(tok_elem(x) == "X" for x in t):
            continue
        toks[ik], fcs[ik] = t, fc
        cnt.update(t)
    vocab = ["<pad>", "<bos>", "<eos>"] + sorted(t for t, c in cnt.items() if c >= 20)
    vid = {t: i for i, t in enumerate(vocab)}
    ids = {}
    for ik, t in toks.items():
        if all(x in vid for x in t):
            ids[ik] = np.array([BOS] + [vid[x] for x in t] + [EOS], np.int64)
    CK = None
    if DN_WEIGHTS:                                        # dn1tpu: continue training, in the checkpoint's vocabulary
        CK = torch.load(DN_WEIGHTS[0], map_location="cpu")
        if CK.get("D", D) != D:
            globals()["D"] = CK["D"]
        log("init from", DN_WEIGHTS[0], "D", D, "vocab: checkpoint", len(CK["vocab"]), "data", len(vocab),
            "same" if list(CK["vocab"]) == list(vocab) else "DIFFERENT (checkpoint's is used)")
        vocab = list(CK["vocab"])
        vid = {t: i for i, t in enumerate(vocab)}
        ids = {}
        for ik, t in toks.items():
            if all(x in vid for x in t):
                ids[ik] = np.array([BOS] + [vid[x] for x in t] + [EOS], np.int64)
    tr = tr[tr.inchikey14.isin(ids)].reset_index(drop=True)
    if os.environ.get("DN_LIMIT"):
        tr = tr.sample(int(os.environ["DN_LIMIT"]), random_state=0).reset_index(drop=True)
    log("vocab", len(vocab), "train spectra", len(tr), "structures", tr.inchikey14.nunique())
    V = len(vocab)
    TE = np.zeros((V, len(ELEMS)), np.float32)                # heavy atoms each token adds
    for i, t in enumerate(vocab):
        e = tok_elem(t)
        if e in ELEMS:
            TE[i, ELEMS.index(e)] = 1

    ev_rows = meta[meta.inchikey14.isin(npx) & meta.adduct.isin(TEST_ADDUCTS)]
    ev_rows = ev_rows.sample(frac=1, random_state=0).groupby("inchikey14").head(4).reset_index(drop=True)
    if os.environ.get("DN_LIMIT"):
        ev_rows = ev_rows[ev_rows.inchikey14.isin(set(list(ev_rows.inchikey14.unique())[:12]))].reset_index(drop=True)
    elif os.environ.get("DN_EVAL_N"):                     # a CPU-sized subset of the held-out molecules
        ev_rows = ev_rows[ev_rows.inchikey14.isin(set(sorted(ev_rows.inchikey14.unique())[:int(os.environ["DN_EVAL_N"])]))].reset_index(drop=True)
    MZ, IT = load_tokens(COMP + "/train.parquet", tr.row.values, tr.precursor_mz.values.astype(np.float32))
    EMZ, EIT = load_tokens(COMP + "/train.parquet", ev_rows.row.values, ev_rows.precursor_mz.values.astype(np.float32))
    log("tokens loaded")
    aidx = {a: i for i, a in enumerate(TEST_ADDUCTS)}
    PR = tr.precursor_mz.values.astype(np.float32)
    AD = tr.adduct.map(aidx).fillna(len(TEST_ADDUCTS)).values.astype(np.int64)
    FC = np.stack([fcs[k] for k in tr.inchikey14.values])
    Yl = [ids[k] for k in tr.inchikey14.values]

    torch.manual_seed(SEED)
    model = Net(V).to(DEV)   # initial weights and names (torch default init)
    if CK is not None:
        model.load_state_dict(CK["state"])
    devs = jax.devices()
    nd = len(devs)
    log("jax devices", nd, devs[0].platform)
    mesh = Mesh(np.array(devs), ("b",))
    rep, shard = NamedSharding(mesh, PartitionSpec()), NamedSharding(mesh, PartitionSpec("b"))
    buffers = {"sin.w"}   # not trained
    sd = {k_: jnp.asarray(v.numpy()) for k_, v in model.state_dict().items()}
    train_p = jax.device_put({k_: v for k_, v in sd.items() if k_ not in buffers}, rep)
    fixed = jax.device_put({k_: v for k_, v in sd.items() if k_ in buffers}, rep)

    def save_ckpt(tag):
        state = {**jax.device_get(train_p), **jax.device_get(fixed)}
        model.load_state_dict({k_: torch.from_numpy(np.array(v)) for k_, v in state.items()})
        torch.save({"state": model.state_dict(), "vocab": vocab, "D": D}, "dn1.pt.tmp")
        os.replace("dn1.pt.tmp", "dn1.pt")
        log("saved dn1.pt", tag)
        return state

    bs = int(os.environ.get("DN_BS", "256"))
    assert bs % nd == 0, (bs, nd)
    LR = 4e-4
    steps = max(1, EPOCHS * (len(tr) // bs))
    warm = 0.05 * steps
    sched = optax.cosine_onecycle_schedule(steps, LR, pct_start=0.05, div_factor=25.0, final_div_factor=1e4)

    def b1_sched(c):                                      # torch OneCycleLR cycles Adam's beta1 0.95 -> 0.85 -> 0.95
        c = jnp.asarray(c, jnp.float32)
        up = 0.85 + 0.05 * (1 + jnp.cos(jnp.pi * jnp.clip(c / warm, 0, 1)))
        down = 0.95 + (0.85 - 0.95) / 2 * (1 + jnp.cos(jnp.pi * jnp.clip((c - warm) / max(steps - warm, 1), 0, 1)))
        return jnp.where(c < warm, up, down)

    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.inject_hyperparams(optax.adamw)(learning_rate=sched, b1=b1_sched, b2=0.999, eps=1e-8,
                                                            weight_decay=1e-4))
    opt_state = opt.init(train_p)

    @jax.jit
    def step(tp, ost, mz, it, pr, ad, fc, y, key):
        def lf(t):
            s, _, n_ = jloss_sums({**t, **fixed}, mz, it, pr, ad, fc, y, key)
            return s / jnp.maximum(n_, 1.0)
        loss, g = jax.value_and_grad(lf)(tp)
        upd, ost = opt.update(g, ost, tp)
        return optax.apply_updates(tp, upd), ost, loss

    jsums = jax.jit(lambda tp, mz, it, pr, ad, fc, y: jloss_sums({**tp, **fixed}, mz, it, pr, ad, fc, y))

    # padded token matrix; batches are cut to a few length buckets (extra PAD columns do not change the math)
    LMAX = MAX_TOK + 2
    Y = np.zeros((len(tr), LMAX), np.int32)
    YL = np.zeros(len(tr), np.int32)
    for r, j in enumerate(Yl):
        Y[r, :len(j)] = j
        YL[r] = len(j)
    BUCKETS = [b_ for b_ in (40, 56, 72, 88) if b_ < LMAX] + [LMAX]

    def bucket(n):
        return next(b_ for b_ in BUCKETS if b_ >= n)

    # held-out teacher-forced loss: the np-examples rows (their truths in the model's vocabulary)
    ev_ids, ev_ok = [], []
    EFC = np.zeros((len(ev_rows), len(FORM_ELEMS)), np.float32)
    for r, (s, f) in enumerate(zip(ev_rows.normalized_smiles.values, ev_rows.molecular_formula.values)):
        t = tokenize(s) if isinstance(s, str) else None
        fc = form_counts(f)
        ok = t is not None and fc is not None and len(t) <= MAX_TOK and all(x in vid for x in t)
        ev_ok.append(ok)
        ev_ids.append(np.array([BOS] + [vid[x] for x in t] + [EOS], np.int32) if ok else np.array([BOS, EOS], np.int32))
        if ok:
            EFC[r] = fc
    ev_ok = np.array(ev_ok, bool)
    EY = np.zeros((len(ev_rows), LMAX), np.int32)
    for r, j in enumerate(ev_ids):
        if ev_ok[r]:
            EY[r, :len(j)] = j
    EPR = ev_rows.precursor_mz.values.astype(np.float32)
    EAD = ev_rows.adduct.map(aidx).fillna(len(TEST_ADDUCTS)).values.astype(np.int64)
    log("held-out rows", int(ev_ok.sum()), "of", len(ev_rows), "molecules", ev_rows[ev_ok].inchikey14.nunique())
    put = lambda a: jax.device_put(a, shard)  # noqa: E731

    def heldout():
        rows = np.where(ev_ok)[0]
        if not len(rows):
            return float("nan"), float("nan")
        EB = bs
        tot = np.zeros(3)
        for i in range(0, len(rows), EB):
            b = rows[i:i + EB]
            idx = np.concatenate([b, np.full(EB - len(b), b[0])])
            y = EY[idx]
            y[len(b):] = PAD                          # padding rows add no tokens
            y = y[:, :bucket(int(max(len(ev_ids[j]) for j in b)))]
            s = jsums(train_p, put(EMZ[idx]), put(EIT[idx]), put(EPR[idx]), put(EAD[idx]), put(EFC[idx]), put(y))
            tot += np.array([float(v) for v in s])
        return tot[0] / tot[2], tot[1] / tot[2]

    budget = float(os.environ.get("DN_TRAIN_SECS", str(1.7 * 3600)))
    ck_every = float(os.environ.get("DN_CKPT_SECS", "1800"))
    l0, n0 = heldout()
    log(f"held-out before training: loss {l0:.4f} nll {n0:.4f}")
    log("train spectra", len(tr), "batch", bs, "steps", steps, "buckets", BUCKETS)
    base = jax.random.PRNGKey(SEED + (1000 if CK is not None else 0))
    k = 0
    t_ck = time.time()
    stop = False
    for ep in range(EPOCHS):
        perm = rng.permutation(len(tr))
        t_ep = time.time()
        losses = []
        for i in range(0, len(perm) - bs + 1, bs):
            b = perm[i:i + bs]
            y = Y[b, :bucket(int(YL[b].max()))]
            train_p, opt_state, loss = step(train_p, opt_state, put(MZ[b]), put(IT[b]), put(PR[b]), put(AD[b]),
                                            put(FC[b]), put(y), jax.random.fold_in(base, k))
            losses.append(loss)
            k += 1
            if k % 1000 == 0:
                log(f"ep {ep} step {k}/{steps} loss {float(jnp.stack(losses[-1000:]).mean()):.4f} "
                    f"({(time.time() - t_ep) / (i // bs + 1):.3f} s/step)")
            if time.time() - T0 > budget:
                stop = True
                break
            if time.time() - t_ck > ck_every:
                save_ckpt(f"ep {ep} step {k}")
                t_ck = time.time()
        tl = float(jnp.stack(losses).mean()) if losses else float("nan")
        hl, hn = heldout()
        log(f"epoch {ep} done, loss {tl:.4f} | held-out loss {hl:.4f} nll {hn:.4f} | "
            f"{time.time() - t_ep:.0f} s, {len(losses)} steps")
        state = save_ckpt(f"epoch {ep}")
        t_ck = time.time()
        if stop:
            log("training time budget reached")
            break
    if k == 0:
        state = save_ckpt("no steps")

    # parity: the saved torch weights vs the JAX forward, on a few train rows (dropout off)
    model.eval()
    n = min(16, len(tr))
    y = Y[:n, :int(YL[:n].max()) - 1]
    with torch.no_grad():
        mem, mm = model.encode(torch.from_numpy(MZ[:n]), torch.from_numpy(IT[:n]), torch.from_numpy(PR[:n]),
                               torch.from_numpy(AD[:n]), torch.from_numpy(FC[:n]))
        t_out = model.decode(mem, mm, torch.from_numpy(y).long()).numpy()
    with jax.default_matmul_precision("highest"):
        j_out = np.asarray(jax.jit(jlogits)(state, MZ[:n], IT[:n], PR[:n], AD[:n], FC[:n], y))
    live = y != PAD
    log("parity torch vs jax: max |logit diff|", float(np.abs(t_out - j_out)[live].max()),
        "logit scale", float(np.abs(t_out[live]).max()))
    log("done")

if __name__ == "__main__":
    main()
