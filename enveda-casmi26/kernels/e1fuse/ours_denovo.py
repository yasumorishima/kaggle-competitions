"""ours (yasunorim): formula-constrained de novo candidates (dn1) for class 3, added to the v1 engine's candidate set.

Model: kernels/dn1 (spectrum + molecular formula -> SMILES; peak-transformer encoder with a formula token, atom-level
SMILES decoder). Decode only: /kaggle/input/**/dn1.pt = {state, vocab, D}. Sampling is constrained by the formula: a
token that would exceed an element count is masked and EOS is allowed only when every heavy atom is placed; a sample
is kept only when its full formula (with H) equals the target formula. No grammar mask (dn1g: it hurt).

Per molecule (install wraps V.run):
  pass 1  the engine runs as before (base + MMP products, PubChem join rows); the public ranker orders the candidates;
  formula the distinct formulas of the best-ranked candidates: the top-1 candidate's formula, plus the next distinct
          formula when it appears within the top FORM_TOP2 (at most MAX_FORMS formulas). The pool window is +-10 ppm
          of the neutral mass, so the top candidates' formulas are the engine's own formula hypothesis (the ranker
          already weighed mass error, fragments, subformula and FP evidence);
  decode  up to MAX_SPEC spectra x S samples for the first formula (S // 2 for the second), dedupe by InChIKey14,
          drop structures already among the pass-1 candidates, order by fp2 log-likelihood (Morgan r2 2048 bits under
          the averaged fp2 probabilities; dn1r-2: truth at rank 1 for 7 of 11 generated truths), keep MAX_NEW;
  pass 2  the engine runs again with the kept structures INJECTED through pc_join (the route of the PubChem-join rows:
          pool-like rows without a library spectrum and without a parent), so every feature, the ranker, ICE / GLACIER,
          fp2 and the fusion see them like any other base candidate. The generator output of pass 1 (base + MMP) is
          cached and replayed in pass 2 (MMP is not recomputed).
Any error, an empty decode or the cumulative time cap (MAX_SECS) returns the pass-1 result unchanged.
"""
import glob
import math
import re
import time
import zlib

import numpy as np

PAD, BOS, EOS = 0, 1, 2
MAX_TOK = 100
ELEMS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Si", "B", "Se"]
FORM_ELEMS = ELEMS + ["H"]
TEST_ADDUCTS = ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
                "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"]
_FORM = re.compile(r"([A-Z][a-z]?)(\d*)")
_BR = re.compile(r"\[(\d*)([A-Z][a-z]?|[a-z][a-z]?)")

MAX_NEW = 20          # de novo rows per molecule
SAMPLES = 48          # samples per spectrum (first formula; S // 2 for the second)
MAX_SPEC = 4          # spectra used for decoding
MAX_FORMS = 2
FORM_TOP2 = 5         # a second formula only when it appears within the top FORM_TOP2 candidates
MAX_LIB = None        # skip molecules whose lib_max >= MAX_LIB (None: no gate)
MAX_SECS = 90 * 60    # cumulative de novo time (decode + pass 2); past it, pass 1 only
STATS = dict(molecules=0, skipped_lib=0, no_formula=0, no_new=0, samples=0, valid=0, formula_ok=0, unique=0,
             added=0, time_skipped=0, errors=0, secs=0.0, decode_secs=0.0, rerun_secs=0.0,
             diag_molecules=0, truth_formula_chosen=0, truth_generated=0, truth_in_added=0, truth_rank1=0)
ADDED = {}            # molecule id -> metric keys of the injected de novo rows
ADDED_SMI = {}        # molecule id -> (lib_max, SMILES of the injected rows in fp2 order) for the slot merge
DIAG = {}             # validation: molecule id -> truth SMILES
CUR = {}              # mid (set by the notebook), pass, generator cache
_M = {}


def tok_elem(t):
    """element of an atom token (None for bonds, rings, branches); 'X' for an element we do not track (== dn1)"""
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
    """element counts over FORM_ELEMS, None for a charged formula or an untracked element"""
    if not isinstance(f, str) or not f or any(c in f for c in "+-."):
        return None
    c = dict.fromkeys(FORM_ELEMS, 0)
    for el, n in _FORM.findall(f):
        if el not in c:
            return None
        c[el] += int(n) if n else 1
    v = np.array([c[e] for e in FORM_ELEMS], np.float32)
    return v if v[:len(ELEMS)].sum() > 0 else None


def _net_classes():
    import torch
    from torch import nn

    class Sinus(nn.Module):
        def __init__(self, n=64, lo=1e-3, hi=2e3):
            super().__init__()
            self.register_buffer("w", 2 * math.pi / torch.logspace(math.log10(lo), math.log10(hi), n))

        def forward(self, x):
            a = x.unsqueeze(-1) * self.w
            return torch.cat([a.sin(), a.cos()], -1)

    class Net(nn.Module):
        """== kernels/dn1/main.py Net (D from the checkpoint)"""
        def __init__(self, V, D):
            super().__init__()
            self.sin = Sinus()
            self.peak = nn.Sequential(nn.Linear(128 * 2 + 1, D), nn.GELU(), nn.Linear(D, D))
            self.prec = nn.Sequential(nn.Linear(128, D), nn.GELU(), nn.Linear(D, D))
            self.add = nn.Embedding(len(TEST_ADDUCTS) + 1, D)
            self.form = nn.Sequential(nn.Linear(len(FORM_ELEMS) * 2, D), nn.GELU(), nn.Linear(D, D))
            layer = nn.TransformerEncoderLayer(D, 8, 4 * D, 0.1, batch_first=True, norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, 4, enable_nested_tensor=False)
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
    return Net


def load(pattern="/kaggle/input/**/dn1.pt", fp2_only=None, device=None, log=print, fp2_pattern="/kaggle/input/**/fp2.pt"):
    """dn1 weights (first match of pattern) + our fp2 nets (ours_fp2.load). Raises when dn1.pt is missing."""
    import torch
    import ours_fp2
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError("dn1.pt not found: " + pattern)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(paths[0], map_location="cpu")
    vocab = list(ck["vocab"])
    D = int(ck.get("D") or ck["state"]["add.weight"].shape[1])
    net = _net_classes()(len(vocab), D)
    net.load_state_dict(ck["state"])
    TE = np.zeros((len(vocab), len(ELEMS)), np.float32)        # heavy atoms each token adds
    for i, t in enumerate(vocab):
        e = tok_elem(t)
        if e in ELEMS:
            TE[i, ELEMS.index(e)] = 1
    fpp, fpn = ours_fp2.load(fp2_pattern, only=fp2_only, device=dev)
    _M.update(net=net.eval().to(dev), vocab=vocab, TE=torch.from_numpy(TE).to(dev), dev=dev, fp2=fpn, path=paths[0])
    log(f"dn1 de novo: {paths[0]} (all matches {paths}) vocab {len(vocab)} D {D} device {dev} | fp2 nets {fpp}")
    return _M


def _tokens(spectra, k):
    import ours_fp2
    aidx = {a: i for i, a in enumerate(TEST_ADDUCTS)}
    sp = [s for s in spectra if len(s.get("mz", ())) > 0][:k]
    if not sp:
        return None
    tk = [ours_fp2.tokens(s["mz"], s["it"], float(s["prec"])) for s in sp]
    return (np.stack([t[0] for t in tk]), np.stack([t[1] for t in tk]),
            np.array([float(s["prec"]) for s in sp], np.float32),
            np.array([aidx.get(s.get("adduct"), len(TEST_ADDUCTS)) for s in sp], np.int64))


def sample(spectra, fc, n_per_spec, seed=0):
    """[(smiles, log p)] of len(spectra[:MAX_SPEC]) x n_per_spec formula-constrained samples (grammar off, == dn1)"""
    import torch
    net, TE, dev, vocab = _M["net"], _M["TE"], _M["dev"], _M["vocab"]
    tk = _tokens(spectra, MAX_SPEC)
    if tk is None or n_per_spec <= 0:
        return []
    mz, it, pr, ad = tk
    rep = np.repeat(np.arange(len(mz)), n_per_spec)
    R = len(rep)
    g = torch.Generator(device=dev)
    g.manual_seed(int(seed) % (2 ** 31))
    with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
        mem, mm = net.encode(torch.from_numpy(mz[rep]).to(dev), torch.from_numpy(it[rep]).to(dev),
                             torch.from_numpy(pr[rep]).to(dev), torch.from_numpy(ad[rep]).to(dev),
                             torch.from_numpy(np.repeat(fc[None], R, 0)).to(dev))
        remain = torch.from_numpy(np.repeat(fc[None, :len(ELEMS)], R, 0)).to(dev)
        y = torch.full((R, 1), BOS, dtype=torch.long, device=dev)
        done = torch.zeros(R, dtype=torch.bool, device=dev)
        lp = torch.zeros(R, device=dev)
        for _t in range(MAX_TOK + 1):
            lg = net.decode(mem, mm, y)[:, -1].float()
            over = (TE[None] > remain[:, None]).any(-1)            # token would exceed the formula
            lg = lg.masked_fill(over, -1e9)
            lg[:, PAD] = -1e9
            lg[:, BOS] = -1e9
            lg[:, EOS] = torch.where((remain > 0).any(-1), torch.full_like(lg[:, EOS], -1e9), lg[:, EOS])
            stuck = (lg <= -1e8).all(-1)                           # no legal token left: end it (discarded later)
            lg[:, EOS] = torch.where(stuck, torch.zeros_like(lg[:, EOS]), lg[:, EOS])
            p = torch.log_softmax(lg, -1)
            nxt = torch.multinomial(p.exp(), 1, generator=g).squeeze(1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            lp = lp + torch.where(done, torch.zeros_like(lp), p.gather(1, nxt[:, None]).squeeze(1))
            remain = remain - TE[nxt] * (~done)[:, None]
            y = torch.cat([y, nxt[:, None]], 1)
            done = done | (nxt == EOS)
            if bool(done.all()):
                break
    out = []
    for seq, l in zip(y.cpu().numpy(), lp.cpu().numpy()):
        out.append(("".join(vocab[x] for x in seq[1:] if x not in (PAD, BOS, EOS)), float(l)))
    return out


def fp2_probs(spectra, max_spec=16):
    """averaged fp2 bit probabilities (2048,) of the molecule, None without fp2 nets"""
    import torch
    nets, dev = _M.get("fp2") or [], _M["dev"]
    tk = _tokens(spectra, max_spec)
    if not nets or tk is None:
        return None
    mz, it, pr, ad = tk
    with torch.no_grad():
        a = (torch.from_numpy(mz).to(dev), torch.from_numpy(it).to(dev), torch.from_numpy(pr).to(dev),
             torch.from_numpy(ad).to(dev))
        p = torch.stack([torch.sigmoid(n(*a).float()) for n in nets]).mean(0).mean(0)
    return p.clamp(1e-4, 1 - 1e-4).cpu().numpy().astype(np.float64)


def choose_formulas(formulas, score, keys=None, max_forms=None, top2=None):
    """distinct formulas of the best-ranked candidates: the top-1 formula, then the next distinct formula if it appears
    within the first `top2` (key-deduplicated) candidates. Returns [(formula, counts)]."""
    max_forms = MAX_FORMS if max_forms is None else max_forms
    top2 = FORM_TOP2 if top2 is None else top2
    order = np.argsort(-np.asarray(score, np.float64), kind="stable")
    out, seen_k, seen_f, r = [], set(), set(), 0
    for i in order:
        k = keys[i] if keys is not None else i
        if k in seen_k:
            continue
        seen_k.add(k)
        r += 1
        f = formulas[i]
        if f in seen_f:
            continue
        seen_f.add(f)
        fc = form_counts(f)
        if fc is not None and (not out or r <= top2):
            out.append((f, fc))
        if len(out) >= max_forms or (out and r >= top2):
            break
    return out


def propose(spectra, forms, exclude_ik14=(), seed=0):
    """de novo structures for the chosen formulas: {ik14: [smiles, count, best log p, mol, formula index]} ranked by
    fp2 log-likelihood (count, then log p, without fp2). Returns (ranked list of entries, fp2 used)."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    cands = {}
    for fi, (_f, fc) in enumerate(forms):
        n = SAMPLES if fi == 0 else max(1, SAMPLES // 2)
        for s, l in sample(spectra, fc, n, seed=seed + 7919 * fi):
            STATS["samples"] += 1
            m = Chem.MolFromSmiles(s) if s else None
            if m is None:
                continue
            STATS["valid"] += 1
            fc2 = form_counts(CalcMolFormula(m))
            if fc2 is None or not np.array_equal(fc2, fc):
                continue
            STATS["formula_ok"] += 1
            try:
                ik = Chem.MolToInchiKey(m)[:14]
            except Exception:
                continue
            if not ik or ik in exclude_ik14:
                continue
            c = cands.get(ik)
            if c is None:
                cands[ik] = [Chem.MolToSmiles(m), 1, l, m, fi]
            else:
                c[1] += 1
                c[2] = max(c[2], l)
    STATS["unique"] += len(cands)
    if not cands:
        return [], False
    keys = list(cands)
    p = fp2_probs(spectra)
    if p is not None:
        if "g2" not in _M:
            _M["g2"] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        B = np.stack([np.asarray(_M["g2"].GetFingerprintAsNumPy(cands[k][3]), np.float64) for k in keys])
        ll = B @ np.log(p) + (1 - B) @ np.log(1 - p)
        order = np.argsort(-ll, kind="stable")
        return [(keys[i], cands[keys[i]], float(ll[i])) for i in order], True
    keys.sort(key=lambda k: (-cands[k][1], -cands[k][2]))
    return [(k, cands[k], float("nan")) for k in keys], False


def install(E, V, score_fn, log=print):
    """Wrap E.generate (cache for pass 2; call BEFORE pc_join.install wraps it, after ours_mmp.install) and V.run
    (pass 1 -> de novo -> pass 2 with the injected rows). score_fn(C, X, F, names) = the public ranker's scores."""
    import pc_join
    from casmi import chem, frag as fragmod
    from rdkit import Chem
    gen0 = E.generate

    def generate(analogs, target, exclude_keys):
        tk = round(float(target), 6)
        if CUR.get("pass") == 2 and tk in CUR.get("gen_cache", {}):
            return list(CUR["gen_cache"][tk])               # pass 2: the same base + MMP products, not recomputed
        r = gen0(analogs, target, exclude_keys)
        if CUR.get("pass") == 1:
            CUR.setdefault("gen_cache", {})[tk] = list(r)
        return r
    E.generate = generate
    if not getattr(E, "_pc_installed", False):
        pc_join.install(E, V, "first")
    run0 = V.run

    def run(spectra, target, *a, **kw):
        CUR["pass"] = 1
        CUR["gen_cache"] = {}
        try:
            out = run0(spectra, target, *a, **kw)
        finally:
            CUR["pass"] = 0
        mid = CUR.get("mid")
        ADDED[mid] = []
        if STATS["secs"] > MAX_SECS:
            STATS["time_skipped"] += 1
            return out
        t0 = time.time()
        prev_inj = list(getattr(E, "_pc_inject", []) or [])
        prev_last = list(getattr(E, "_pc_last", []) or [])
        try:
            out2 = second(out, spectra, target, a, kw, mid, prev_inj)
            if out2 is None:
                ADDED[mid] = []
                return out
            return out2
        except Exception as e:                          # never lose the base candidates
            STATS["errors"] += 1
            STATS["error"] = repr(e)[:300]
            ADDED[mid] = []
            E._pc_last = prev_last                      # injected_mask / injected_info describe the pass-1 result
            return out
        finally:
            pc_join.set_injection(E, prev_inj)
            CUR["pass"] = 0
            CUR["gen_cache"] = {}
            STATS["secs"] += time.time() - t0

    def second(out, spectra, target, a, kw, mid, prev_inj):
        C, X, F, names, info = out
        if len(C.pid) == 0:
            return None
        lib_max = float(info.get("lib_max", 0.0)) if isinstance(info, dict) else 0.0
        if MAX_LIB is not None and lib_max >= MAX_LIB:
            STATS["skipped_lib"] += 1
            return None
        STATS["molecules"] += 1
        score = score_fn(C, X, F, names)
        forms = choose_formulas(list(C.formula), score, list(C.key))
        if not forms:
            STATS["no_formula"] += 1
            return None
        t1 = time.time()
        have = set(C.key) | {r["key"] for r in prev_inj}
        seed = zlib.crc32(str(mid).encode()) % 100000           # reproducible per molecule
        ranked, _used_fp2 = propose(spectra, forms, seed=seed)
        recs = []
        for ik, c, _ll in ranked:
            if len(recs) >= MAX_NEW:
                break
            r = pc_join.candidate_record(c[0], None, chem, fragmod, E.pool.bits)
            if r is None or r["key"] in have:
                continue
            have.add(r["key"])
            r["dn"] = True
            r["dn_ik14"] = ik
            recs.append(r)
        STATS["decode_secs"] += time.time() - t1
        if mid in DIAG:                                 # validation bookkeeping
            _diag(mid, forms, ranked, recs, chem, Chem)
        if not recs:
            STATS["no_new"] += 1
            return None
        t2 = time.time()
        pc_join.set_injection(E, prev_inj + recs)
        CUR["pass"] = 2                                 # generate() replays the pass-1 products
        out2 = run0(spectra, target, *a, **kw)
        STATS["rerun_secs"] += time.time() - t2
        ADDED[mid] = [r["key"] for r in recs]
        ADDED_SMI[mid] = (lib_max, [r["smiles"] for r in recs])
        STATS["added"] += len(recs)
        return out2

    V.run = run
    log(f"de novo installed: max new {MAX_NEW}, samples/spectrum {SAMPLES} x {MAX_SPEC} spectra, formulas {MAX_FORMS} "
        f"(2nd within top {FORM_TOP2}), lib gate {MAX_LIB}, time cap {MAX_SECS / 60:.0f} min")


def _diag(mid, forms, ranked, recs, chem, Chem):
    t = DIAG[mid]
    m = Chem.MolFromSmiles(t)
    if m is None:
        return
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    STATS["diag_molecules"] += 1
    tik = Chem.MolToInchiKey(m)[:14]
    tkey = chem.score_key(chem.standardize_smiles(t) or t)
    tfc = form_counts(CalcMolFormula(m))
    STATS["truth_formula_chosen"] += int(tfc is not None and any(np.array_equal(tfc, fc) for _f, fc in forms))
    gi = next((i for i, (ik, _c, _l) in enumerate(ranked) if ik == tik), None)
    STATS["truth_generated"] += int(gi is not None)
    STATS["truth_rank1"] += int(gi == 0)
    STATS["truth_in_added"] += int(any(r["key"] == tkey for r in recs))


def summary(log=print):
    s = STATS
    n = max(s["samples"], 1)
    d = max(s["diag_molecules"], 1)
    log(f"DENOVO stats denovo_mols {s['molecules']} denovo_added {s['added']} ({s['added'] / max(s['molecules'], 1):.1f}/mol) "
        f"| samples {s['samples']} valid {s['valid'] / n:.3f} formula_ok {s['formula_ok'] / n:.3f} unique {s['unique']} "
        f"| skipped: lib {s['skipped_lib']} no formula {s['no_formula']} nothing new {s['no_new']} time {s['time_skipped']} "
        f"errors {s['errors']} {s.get('error', '')} | secs total {s['secs']:.0f} decode {s['decode_secs']:.0f} "
        f"pass-2 {s['rerun_secs']:.0f}")
    if s["diag_molecules"]:
        log(f"DENOVO validation n={s['diag_molecules']} | truth formula chosen {s['truth_formula_chosen'] / d:.3f} | "
            f"truth generated {s['truth_generated'] / d:.3f} | truth fp2-rank 1 {s['truth_rank1'] / d:.3f} | "
            f"denovo_truth_in_added {s['truth_in_added'] / d:.3f}")
