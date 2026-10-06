"""CASMI26 FP2 on TPU (JAX; built by fp2tpu/build.py: seed 4, 16 epochs, d 384): spectrum -> Morgan fingerprint with a peak transformer (GPU).

Plan step 1 (enveda-casmi26/CLAUDE.md, "メダルへの道筋"): the fingerprint channel has to
reach the public FPNet's level (FP alone ~0.45+ on natural-product c2) before it can help.
The 1-Da-bin MLP scored 0.24 there. This kernel keeps exact m/z: every peak is a token
with sinusoidal embeddings of its m/z and of its neutral loss, plus its log intensity; the
precursor (mass + adduct) is one more token whose output is the fingerprint.

Held out from training and scored at the end, FP alone over the train+COCONUT pool
(+-10 ppm of the neutral mass), exactly as the ranking channel would use it:
  np   = enveda-np-examples structures (all their spectra, every library)  <- the LB-like c2
  rnd  = 400 random enveda-180-only structures
Writes fp2.pt (weights) and fp2_eval.txt.
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
WEIGHTS = glob.glob("/kaggle/input/**/fp2.pt", recursive=True)   # present => evaluate only (fp2dump)
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
DEV = "cpu"   # torch only builds the initial weights and checks parity
EPOCHS = int(os.environ.get("FP2_EPOCHS", "16"))
PER_STRUCT = 4
N_PEAKS = 64
FP_BITS = 2048
D = int(os.environ.get("FP2_D", "384"))
SEED = int(os.environ.get("FP2_SEED", "4"))   # 0 = fp2 v2; others = ensemble members (make_variant.py)
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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.sin = Sinus()
        self.peak = nn.Sequential(nn.Linear(128 * 2 + 1, D), nn.GELU(), nn.Linear(D, D))
        self.prec = nn.Sequential(nn.Linear(128, D), nn.GELU(), nn.Linear(D, D))
        self.add = nn.Embedding(len(TEST_ADDUCTS) + 1, D)
        layer = nn.TransformerEncoderLayer(D, 8, 4 * D, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, 4)
        self.out = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1024), nn.GELU(), nn.Linear(1024, FP_BITS))

    def forward(self, mz, it, prec, add):
        pad = mz <= 0
        tok = self.peak(torch.cat([self.sin(mz), self.sin((prec[:, None] - mz).clamp(min=0)), it[..., None]], -1))
        p = (self.prec(self.sin(prec)) + self.add(add))[:, None]
        x = torch.cat([p, tok], 1)
        mask = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
        return self.out(self.enc(x, src_key_padding_mask=mask)[:, 0])


# ---- the same network in JAX, on the torch state-dict names (norm-first encoder, 8 heads)
N_HEADS = 8


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


def jforward(p, mz, it, prec, add, key=None):
    n_layers = sum(1 for k in p if k.endswith("norm1.weight"))
    keys = list(jax.random.split(key, 4 * n_layers)) if key is not None else [None] * (4 * n_layers)
    pad = mz <= 0
    tok = _lin(p, "peak.2", _gelu(_lin(p, "peak.0", jnp.concatenate(
        [_sin(p, mz), _sin(p, jnp.clip(prec[:, None] - mz, 0)), it[..., None]], -1))))
    pp = _lin(p, "prec.2", _gelu(_lin(p, "prec.0", _sin(p, prec)))) + p["add.weight"][add]
    x = jnp.concatenate([pp[:, None], tok], 1)
    mask = jnp.concatenate([jnp.zeros_like(pad[:, :1]), pad], 1)
    B, T, Dm = x.shape
    dh = Dm // N_HEADS
    for l in range(n_layers):
        q = f"enc.layers.{l}."
        h = _ln(p, q + "norm1", x)
        qkv = h @ p[q + "self_attn.in_proj_weight"].T + p[q + "self_attn.in_proj_bias"]
        qq, kk, vv = [t.reshape(B, T, N_HEADS, dh) for t in jnp.split(qkv, 3, -1)]
        a = jnp.einsum("bthd,bshd->bhts", qq, kk) / jnp.sqrt(dh)
        a = jax.nn.softmax(jnp.where(mask[:, None, None, :], -1e9, a), -1)
        a = _drop(a, keys[4 * l])
        o = jnp.einsum("bhts,bshd->bthd", a, vv).reshape(B, T, Dm)
        x = x + _drop(_lin(p, q + "self_attn.out_proj", o), keys[4 * l + 1])
        h = _drop(_gelu(_lin(p, q + "linear1", _ln(p, q + "norm2", x))), keys[4 * l + 2])
        x = x + _drop(_lin(p, q + "linear2", h), keys[4 * l + 3])
    y = _gelu(_lin(p, "out.1", _ln(p, "out.0", x[:, 0])))
    return _lin(p, "out.3", y)


def main():
    meta = pd.read_parquet(COMP + "/train.parquet",
                           columns=["ingest_lib", "inchikey14", "normalized_smiles", "molecular_formula",
                                    "precursor_mz", "adduct", "instrument_type"])
    meta["row"] = np.arange(len(meta))
    log("train", len(meta))
    rng = np.random.default_rng(2026)
    npx = set(meta.loc[meta.ingest_lib == "enveda-np-examples", "inchikey14"])
    pub = set(meta.loc[~meta.ingest_lib.str.startswith("enveda"), "inchikey14"])
    ev = meta.loc[(meta.ingest_lib == "enveda-180") & meta.adduct.isin(TEST_ADDUCTS), "inchikey14"].unique()
    rnd = set(rng.choice([m for m in ev if m not in pub and m not in npx], 400, replace=False))
    held = npx | rnd
    if SEED:   # same held-out set, different spectra per structure, init and batch order
        rng = np.random.default_rng(2026 + SEED)
        torch.manual_seed(SEED)

    tr = meta[meta.adduct.isin(TEST_ADDUCTS) & ~meta.inchikey14.isin(held)].copy()
    tr["pri"] = (tr.instrument_type == "timsTOF").astype(int)
    tr = tr.sample(frac=1, random_state=SEED).sort_values("pri", ascending=False, kind="stable")
    tr = tr.groupby("inchikey14").head(PER_STRUCT)
    S = tr.drop_duplicates("inchikey14")
    bits, ok = fp_bits(S.normalized_smiles.values)
    fpb = dict(zip(S.inchikey14[ok], bits[ok]))
    tr = tr[tr.inchikey14.isin(fpb)].reset_index(drop=True)
    if os.environ.get("FP2_LIMIT"):                      # local smoke test only
        tr = tr.sample(int(os.environ["FP2_LIMIT"]), random_state=0).reset_index(drop=True)
    log("train spectra", len(tr), "structures", tr.inchikey14.nunique())

    ev_rows = meta[meta.inchikey14.isin(held) & meta.adduct.isin(TEST_ADDUCTS)]
    ev_rows = ev_rows[(ev_rows.ingest_lib == "enveda-np-examples") | ev_rows.inchikey14.isin(rnd)]
    ev_rows = ev_rows.sample(frac=1, random_state=0).groupby("inchikey14").head(16).reset_index(drop=True)
    if os.environ.get("FP2_LIMIT"):
        keep = set(list(ev_rows.inchikey14.unique())[:60])
        ev_rows = ev_rows[ev_rows.inchikey14.isin(keep)].reset_index(drop=True)

    MZ, IT = load_tokens(COMP + "/train.parquet", tr.row.values, tr.precursor_mz.values.astype(np.float32))
    EMZ, EIT = load_tokens(COMP + "/train.parquet", ev_rows.row.values, ev_rows.precursor_mz.values.astype(np.float32))
    log("tokens loaded")
    aidx = {a: i for i, a in enumerate(TEST_ADDUCTS)}
    PR = tr.precursor_mz.values.astype(np.float32)
    AD = tr.adduct.map(aidx).fillna(len(TEST_ADDUCTS)).values.astype(np.int64)
    Y = np.stack([fpb[k] for k in tr.inchikey14.values])

    model = Net().to(DEV)   # initial weights and names (torch default init, seeded above)
    devs = jax.devices()
    log("jax devices", len(devs), devs[0].platform)
    mesh = Mesh(np.array(devs), ("b",))
    rep, shard = NamedSharding(mesh, PartitionSpec()), NamedSharding(mesh, PartitionSpec("b"))
    params = jax.device_put({k: jnp.asarray(v.numpy()) for k, v in model.state_dict().items()}, rep)
    buffers = {"sin.w"}   # not trained
    bs = 512
    steps = EPOCHS * (len(tr) // bs)
    sched = optax.cosine_onecycle_schedule(steps, 5e-4, pct_start=0.05, div_factor=25.0, final_div_factor=1e4)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))
    train_p = {k: v for k, v in params.items() if k not in buffers}
    fixed = {k: v for k, v in params.items() if k in buffers}
    opt_state = opt.init(train_p)

    @jax.jit
    def step(tp, ost, mz, it, pr, ad, yb, key):
        def lossf(t):
            logits = jforward({**t, **fixed}, mz, it, pr, ad, key)
            y = jnp.unpackbits(yb, axis=1).astype(jnp.float32)
            return optax.sigmoid_binary_cross_entropy(logits, y).mean()
        loss, g = jax.value_and_grad(lossf)(tp)
        upd, ost = opt.update(g, ost, tp)
        return optax.apply_updates(tp, upd), ost, loss

    base = jax.random.PRNGKey(SEED)
    k = 0
    for ep in range(EPOCHS):
        perm = rng.permutation(len(tr))
        tot = 0.0
        for i in range(0, len(perm) - bs + 1, bs):
            b = np.sort(perm[i:i + bs])
            put = lambda a: jax.device_put(a, shard)  # noqa: E731
            train_p, opt_state, loss = step(train_p, opt_state, put(MZ[b]), put(IT[b]), put(PR[b]), put(AD[b]),
                                            put(Y[b]), jax.random.fold_in(base, k))
            k += 1
            if k % 1000 == 0:
                tot = float(loss)
                log(f"ep {ep} step {k}/{steps} loss {tot:.4f}")
        log(f"epoch {ep} done")
    params = {**jax.device_get(train_p), **jax.device_get(fixed)}
    model.load_state_dict({k_: torch.from_numpy(np.array(v)) for k_, v in params.items()})
    torch.save(model.state_dict(), "fp2.pt")
    model.eval()
    with torch.no_grad():
        n = min(64, len(tr))
        t_out = model(torch.from_numpy(MZ[:n]), torch.from_numpy(IT[:n]), torch.from_numpy(PR[:n]),
                      torch.from_numpy(AD[:n])).numpy()
    j_out = np.asarray(jax.jit(jforward)(params, MZ[:n], IT[:n], PR[:n], AD[:n]))
    log("parity torch vs jax: max |logit diff|", float(np.abs(t_out - j_out).max()), "logit scale", float(np.abs(t_out).max()))

    # ---- evaluation: FP alone over the train+COCONUT pool, as the ranking channel sees it
    model.eval()
    pool = meta.drop_duplicates("inchikey14")[["inchikey14", "normalized_smiles", "molecular_formula"]].copy()
    pool["fM"] = pool.molecular_formula.map(formula_mass)
    pool = pool.dropna(subset=["fM"]).rename(columns={"normalized_smiles": "smiles"})[["inchikey14", "smiles", "fM"]]
    if COCO:
        cm = pickle.load(open(COCO + "/coco_meta.pkl", "rb"))
        coco = pd.DataFrame({"inchikey14": cm["keys"], "smiles": cm["smiles"], "fM": np.load(COCO + "/coco_mass.npy")})
        pool = pd.concat([pool, coco[~coco.inchikey14.isin(set(pool.inchikey14))]], ignore_index=True)
    pool = pool.sort_values("fM").reset_index(drop=True)
    pm, pk = pool.fM.values, pool.inchikey14.values
    smi = dict(zip(pool.inchikey14, pool.smiles))
    ev_rows["M"] = [neutral_mass(m, a) for m, a in zip(ev_rows.precursor_mz, ev_rows.adduct)]
    EPR = ev_rows.precursor_mz.values.astype(np.float32)
    EAD = ev_rows.adduct.map(aidx).fillna(len(TEST_ADDUCTS)).values.astype(np.int64)
    jfwd = jax.jit(jforward)
    P = []
    for i in range(0, len(ev_rows), 1024):
        s = slice(i, i + 1024)
        n = len(EMZ[s])
        padn = lambda a: np.concatenate([a, np.zeros((1024 - n,) + a.shape[1:], a.dtype)])  # noqa: E731
        P.append(np.asarray(jax.nn.sigmoid(jfwd(params, padn(EMZ[s]), padn(EIT[s]), padn(EPR[s]), padn(EAD[s]))))[:n])
    P = np.concatenate(P)
    res = []
    for ik, g in ev_rows.groupby("inchikey14"):
        p = np.clip(P[g.index.values].mean(0), 1e-4, 1 - 1e-4)
        M = float(np.median(g.M))
        lo, hi = np.searchsorted(pm, [M * (1 - PPM * 1e-6), M * (1 + PPM * 1e-6)])
        cands = list(pk[lo:hi])
        cls = "np" if ik in npx else "rnd"
        if ik not in cands:
            res.append((cls, 0.0, len(cands), 0))
            continue
        b, okc = fp_bits([smi.get(c) for c in cands])
        b = np.unpackbits(b, axis=1).astype(np.float32)
        ll = b @ np.log(p) + (1 - b) @ np.log(1 - p)
        ll[~okc] = -1e9
        rank = 1 + int((ll > ll[cands.index(ik)]).sum())
        if WEIGHTS and cls == "np":   # candidate log-likelihoods for the local blend (class 4)
            print("FPLL " + ik + " " + " ".join(f"{c}:{v:.1f}" for c, v in zip(cands, ll) if v > -1e8), flush=True)
        res.append((cls, 1.0 / rank if rank <= 25 else 0.0, len(cands), 1))
    r = pd.DataFrame(res, columns=["cls", "rr", "ncand", "in_window"])
    out = r.groupby("cls").agg(mols=("rr", "size"), mrr=("rr", "mean"), in_window=("in_window", "mean"),
                               ncand=("ncand", "median")).round(3)
    log("FP-alone MRR@25 (held out)\n" + out.to_string())
    open("fp2_eval.txt", "w").write(out.to_string() + "\n")
    log("done")


if __name__ == "__main__":
    main()
