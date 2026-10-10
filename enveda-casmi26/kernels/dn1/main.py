"""CASMI26 dn1 (ours): formula-constrained de novo generation, spectrum + formula -> SMILES (GPU).

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

RDLogger.DisableLog("rdApp.*")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = int(os.environ.get("DN_EPOCHS", "8"))
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
    MZ, IT = load_tokens(COMP + "/train.parquet", tr.row.values, tr.precursor_mz.values.astype(np.float32))
    EMZ, EIT = load_tokens(COMP + "/train.parquet", ev_rows.row.values, ev_rows.precursor_mz.values.astype(np.float32))
    log("tokens loaded")
    aidx = {a: i for i, a in enumerate(TEST_ADDUCTS)}
    PR = tr.precursor_mz.values.astype(np.float32)
    AD = tr.adduct.map(aidx).fillna(len(TEST_ADDUCTS)).values.astype(np.int64)
    FC = np.stack([fcs[k] for k in tr.inchikey14.values])
    Yl = [ids[k] for k in tr.inchikey14.values]

    model = Net(V).to(DEV)
    bs = 256
    steps = max(1, EPOCHS * (len(tr) // bs))
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-4, total_steps=steps, pct_start=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=DEV == "cuda")
    lossf = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.05)
    k = 0
    budget = float(os.environ.get("DN_TRAIN_SECS", str(3.0 * 3600)))
    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(len(tr))
        tot, n = 0.0, 0
        for i in range(0, len(perm) - bs + 1, bs):
            b = perm[i:i + bs]
            Lm = max(len(Yl[j]) for j in b)
            y = np.zeros((len(b), Lm), np.int64)
            for r, j in enumerate(b):
                y[r, :len(Yl[j])] = Yl[j]
            y = torch.from_numpy(y).to(DEV)
            with torch.autocast(DEV, enabled=DEV == "cuda"):
                mem, mm = model.encode(torch.from_numpy(MZ[b]).to(DEV), torch.from_numpy(IT[b]).to(DEV),
                                       torch.from_numpy(PR[b]).to(DEV), torch.from_numpy(AD[b]).to(DEV),
                                       torch.from_numpy(FC[b]).to(DEV))
                logits = model.decode(mem, mm, y[:, :-1])
                loss = lossf(logits.float().reshape(-1, V), y[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if k < steps - 1:
                sched.step()
            tot += loss.item(); n += 1; k += 1
            if k % 1000 == 0:
                log(f"ep {ep} step {k}/{steps} loss {tot / n:.4f}")
            if time.time() - T0 > budget:
                break
        torch.save({"state": model.state_dict(), "vocab": vocab, "D": D}, "dn1.pt")
        log(f"epoch {ep} done, loss {tot / max(n, 1):.4f}")
        if time.time() - T0 > budget:
            log("training time budget reached")
            break

    # ---- evaluation on the held-out NP molecules (true formula; the hidden run would take it from the candidates)
    model.eval()
    TEt = torch.from_numpy(TE).to(DEV)
    truth = dict(zip(S_all.inchikey14, S_all.normalized_smiles))
    form = dict(zip(S_all.inchikey14, S_all.molecular_formula))
    EPR = ev_rows.precursor_mz.values.astype(np.float32)
    EAD = ev_rows.adduct.map(aidx).fillna(len(TEST_ADDUCTS)).values.astype(np.int64)
    res = []
    stats = Counter()
    for ik, g in ev_rows.groupby("inchikey14"):
        fc = form_counts(form.get(ik))
        if fc is None:
            continue
        rows = g.index.values
        R = len(rows) * N_SAMPLES
        rep = np.repeat(rows, N_SAMPLES)
        with torch.no_grad(), torch.autocast(DEV, enabled=DEV == "cuda"):
            mem, mm = model.encode(torch.from_numpy(EMZ[rep]).to(DEV), torch.from_numpy(EIT[rep]).to(DEV),
                                   torch.from_numpy(EPR[rep]).to(DEV), torch.from_numpy(EAD[rep]).to(DEV),
                                   torch.from_numpy(np.repeat(fc[None], R, 0)).to(DEV))
            remain = torch.from_numpy(np.repeat(fc[None, :len(ELEMS)], R, 0)).to(DEV)
            y = torch.full((R, 1), BOS, dtype=torch.long, device=DEV)
            done = torch.zeros(R, dtype=torch.bool, device=DEV)
            lp = torch.zeros(R, device=DEV)
            for t in range(MAX_TOK + 1):
                lg = model.decode(mem, mm, y)[:, -1].float()
                over = ((TEt[None] > remain[:, None]).any(-1))          # token would exceed the formula
                lg = lg.masked_fill(over, -1e9)
                lg[:, PAD] = -1e9; lg[:, BOS] = -1e9
                lg[:, EOS] = torch.where((remain > 0).any(-1), torch.full_like(lg[:, EOS], -1e9), lg[:, EOS])
                pr = torch.log_softmax(lg, -1)
                nxt = torch.multinomial(pr.exp(), 1).squeeze(1)
                nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
                lp = lp + torch.where(done, torch.zeros_like(lp), pr.gather(1, nxt[:, None]).squeeze(1))
                remain = remain - TEt[nxt] * (~done)[:, None]
                y = torch.cat([y, nxt[:, None]], 1)
                done = done | (nxt == EOS)
                if bool(done.all()):
                    break
        cands = {}
        tfp = gen.GetFingerprint(Chem.MolFromSmiles(truth[ik]))
        best_t = 0.0
        for si, (seq, l) in enumerate(zip(y.cpu().numpy(), lp.cpu().numpy())):
            stats["samples"] += 1
            s = "".join(vocab[x] for x in seq[1:] if x not in (PAD, BOS, EOS))
            if len(res) < 3 and si < 3:
                print("DN1 sample", ik, form.get(ik), "->", s, flush=True)
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            stats["valid"] += 1
            if CalcMolFormula(m) != form[ik]:
                continue
            stats["formula_ok"] += 1
            key = Chem.MolToInchiKey(m)[:14]
            c = cands.setdefault(key, [0, -1e9, m])
            c[0] += 1; c[1] = max(c[1], float(l))
        ranked = sorted(cands, key=lambda q: (-cands[q][0], -cands[q][1]))
        for q in ranked[:25]:
            best_t = max(best_t, DataStructs.TanimotoSimilarity(gen.GetFingerprint(cands[q][2]), tfp))
        rank = ranked.index(ik) + 1 if ik in cands else None
        res.append((1.0 / rank if rank and rank <= 25 else 0.0, ik in cands, len(cands), best_t))
    r = pd.DataFrame(res, columns=["rr", "generated", "n_unique", "best_tan25"])
    log(f"DN1 EVAL n={len(r)} | MRR@25 {r.rr.mean():.4f} top1 {(r.rr == 1).mean():.3f} hit@25 {(r.rr > 0).mean():.3f} "
        f"truth generated {r.generated.mean():.3f} | unique formula-correct per molecule {r.n_unique.median():.0f} | "
        f"best Tanimoto in top 25: median {r.best_tan25.median():.3f} q75 {r.best_tan25.quantile(0.75):.3f}")
    log("DN1 sample stats", dict(stats), f"valid {stats['valid'] / max(stats['samples'], 1):.3f} "
        f"formula_ok {stats['formula_ok'] / max(stats['samples'], 1):.3f}")
    log("done")


if __name__ == "__main__":
    main()
