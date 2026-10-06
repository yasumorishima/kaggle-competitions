"""Build kernels/b5/main.py = b1's kernel + fp2 inference + the LightGBM re-ranker.

Kaggle uploads only the code file, so the re-ranker (lgb_model.txt, written by
eval/rerank.py) is embedded as a string. fp2's weights come from the fp2 kernel's
output (kernel_sources). Run after either input changes:

    python enveda-casmi26/kernels/b5/build.py [MODE]

MODE (default lgb) picks how the analog and fp2 channels are fused outside the gate:
  lgb     the LightGBM re-ranker (class-4 CV 0.65)
  rrfK    reciprocal-rank fusion 1/(K+rank_analog) + 1/(K+rank_fp2) (local 0.590 at K=5)
  expT_W  analog + W * exp((ll - max ll) / T) (local 0.579 at T=25, W=0.1)
  expT_W_rel  the same with W scaled by the molecule's best analog score (all-class local 0.587 at T=50, W=0.5)
Options appended with "+":
  ens     average fp2's bit probabilities over every fp2.pt found (fp2 v2 + fp2all; c4 local +0.002)
  gX      library gate X instead of 0.8
  tpF     scale the fused score by F for train structures that have library spectra but no match
          at the gate (hidden class 2/3 answers never carry public spectra; eval/blend_trainpen.py:
          F 0.7 -> c1 .935 c2 .860 c4 .628 vs F 1 .936 .820 .568)
  tpF_L   the same, only when the candidate's best own-spectrum match is below L
  popMU   add MU * log1p(PubMed links) of the candidate's InChIKey14 (own dataset from PubChem FTP,
          datasets/pubchem-pop; local c1 .937 c2 .844 c4 .700 at MU 0.05 vs .938 .841 .585 without)
  laA_L   add A * lib to candidates whose library match is in [L, gate) (H3: on the LB, matches
          between 0.5 and the gate are right more often than in the local bench)
  enR.R.R put the best Enamine-tier candidates (own dataset from PubChem's Enamine deposit; 92% of
          enveda-180 is there) at fixed 1-based ranks R, only when no library match reaches the gate.
          Enamine candidates in the window that are not in the pool are scored like the pool (analog +
          fp2), eval/enamine_slot.py
  emF     instead of slots, the top 10 Enamine candidates join the ranking with F * their score
          (eval/enamine_mix.py: F 1 -> c1 .931 c2 .819 answer-outside-pool .522 c4 .575;
          slots 2.4.6 -> .935 .835 .273 .563)
  ewT_W   Enamine ranking: analog + W * exp((ll - max ll) / T) (default T 50, W 0.2)
  sens    average fp2 v2 with the seed members (fp2s1, fp2s2; fp2all stays out: it cost 0.009 on the LB)
  ppmX    candidate window +-X ppm instead of 10 (enveda-180 errors: 99% < 4.4 ppm, max 7.1;
          eval/ppm_window.py: 5 ppm -> c2 +0.004, c1 unchanged)
e.g. exp50_0.2+ens+g0.75, exp50_0.2+tp0.7, exp50_0.2+tp0.7_0.5+en5.10.15.20.25
"""
import os
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "exp50_0.2+tp0.7_0.5"   # LB best 0.299 (10-03; tp0.5_0.5 0.297, exp50_0.2 alone 0.294)
FUSE, *OPTS = MODE.split("+")
ENS = "all" if "ens" in OPTS else "seeds" if "sens" in OPTS else ""
PPM = next((float(o[3:]) for o in OPTS if o.startswith("ppm")), None)
GATE = next((o[1:] for o in OPTS if o.startswith("g")), None)
TP = next((o[2:].split("_") for o in OPTS if o.startswith("tp")), None)
TP_F, TP_L = (float(TP[0]), float(TP[1]) if len(TP) > 1 else 9.0) if TP else (1.0, 9.0)
POP = next((float(o[3:]) for o in OPTS if o.startswith("pop")), 0.0)
LA = next((o[2:].split("_") for o in OPTS if o.startswith("la")), None)
LA_A, LA_L = (float(LA[0]), float(LA[1])) if LA else (0.0, 9.0)
EN = next(([int(x) for x in o[2:].split(".")] for o in OPTS if o.startswith("en") and o != "ens"), [])
EM = next((float(o[2:]) for o in OPTS if o.startswith("em")), 0.0)
EW = next((o[2:].split("_") for o in OPTS if o.startswith("ew")), None)
EN_T, EN_W = (float(EW[0]), float(EW[1])) if EW else (50.0, 0.2)

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "..", "b1", "main.py"), encoding="utf-8").read()
model = open(os.path.join(HERE, "lgb_model.txt"), encoding="utf-8").read()

src = src.replace('"""CASMI26 B1: mass-window candidates', '"""CASMI26 B5 (b3 + fp2 + LightGBM re-ranker). Built from b1 by b5/build.py.\n\nB1: mass-window candidates', 1)

if PPM is not None:
    assert "PPM, PPM_WIDE = 10.0, 30.0" in src
    src = src.replace("PPM, PPM_WIDE = 10.0, 30.0", f"PPM, PPM_WIDE = {PPM}, 30.0", 1)

src = src.replace("from rdkit.Chem import rdFingerprintGenerator  # noqa: E402\n", '''from rdkit.Chem import rdFingerprintGenerator  # noqa: E402
import lightgbm as lgb  # noqa: E402
import math  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
''', 1)

FP2 = '''

# ---- fp2: peak transformer spectrum -> Morgan r2 2048 bits (kernels/fp2) ----
FUSE = "''' + FUSE + '''"
TP_F, TP_L = ''' + repr(TP_F) + ", " + repr(TP_L) + '''  # demotion of train structures whose spectra do not match
LA_A, LA_L = ''' + repr(LA_A) + ", " + repr(LA_L) + '''  # boost of library matches in [LA_L, gate)
EN_SLOTS = ''' + repr(EN) + '''  # 1-based ranks for Enamine-tier candidates (empty = off)
EN_MERGE = ''' + repr(EM) + '''  # > 0: the top 10 Enamine candidates compete with the pool on EN_MERGE * score
EN_T, EN_W = ''' + repr(EN_T) + ", " + repr(EN_W) + '''  # Enamine ranking: analog + EN_W * exp((ll - max ll) / EN_T)
_en_path = glob.glob("/kaggle/input/**/enamine_tier.parquet", recursive=True)
POP_MU = ''' + repr(POP) + '''  # popularity prior weight (log1p PubMed links per InChIKey14)
_pop_path = glob.glob("/kaggle/input/**/pubchem_pop.parquet", recursive=True)
POP = dict(zip(*pd.read_parquet(_pop_path[0], columns=["inchikey14", "n_pmid"]).values.T)) if POP_MU and _pop_path else {}
FP2_ENS = ''' + repr(ENS) + '''  # "": fp2 v2 only, "seeds": v2 + fp2s*, "all": also fp2all
# fp2 v2 first (the LB-checked model), then the seed members, fp2all last
FP2_WEIGHTS = sorted(glob.glob("/kaggle/input/**/fp2.pt", recursive=True),
                     key=lambda p: (0 if "fp2-peak-transformer" in p else 2 if "fp2all" in p else 1, p))
if FP2_ENS == "seeds":
    FP2_WEIGHTS = [p for p in FP2_WEIGHTS if "fp2all" not in p]
FP2_D, FP2_PEAKS, FP2_BITS = 384, 64, 2048
FP2_ADDUCTS = ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
               "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"]
_gen2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP2_BITS)


class Sinus(nn.Module):
    def __init__(self, n=64, lo=1e-3, hi=2e3):
        super().__init__()
        self.register_buffer("w", 2 * math.pi / torch.logspace(math.log10(lo), math.log10(hi), n))

    def forward(self, x):
        a = x.unsqueeze(-1) * self.w
        return torch.cat([a.sin(), a.cos()], -1)


class FP2Net(nn.Module):
    def __init__(self):
        super().__init__()
        D = FP2_D
        self.sin = Sinus()
        self.peak = nn.Sequential(nn.Linear(128 * 2 + 1, D), nn.GELU(), nn.Linear(D, D))
        self.prec = nn.Sequential(nn.Linear(128, D), nn.GELU(), nn.Linear(D, D))
        self.add = nn.Embedding(len(FP2_ADDUCTS) + 1, D)
        layer = nn.TransformerEncoderLayer(D, 8, 4 * D, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, 4)
        self.out = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1024), nn.GELU(), nn.Linear(1024, FP2_BITS))

    def forward(self, mz, it, prec, add):
        pad = mz <= 0
        tok = self.peak(torch.cat([self.sin(mz), self.sin((prec[:, None] - mz).clamp(min=0)), it[..., None]], -1))
        p = (self.prec(self.sin(prec)) + self.add(add))[:, None]
        x = torch.cat([p, tok], 1)
        mask = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
        return self.out(self.enc(x, src_key_padding_mask=mask)[:, 0])


def fp2_tokens(mz, it, prec):
    mz = np.asarray(mz, np.float32)
    it = np.asarray(it, np.float32)
    keep = (mz < prec + 1.0) & (it > 0)
    mz, it = mz[keep], it[keep]
    if len(mz) > FP2_PEAKS:
        top = np.argpartition(-it, FP2_PEAKS)[:FP2_PEAKS]
        mz, it = mz[top], it[top]
    M = np.zeros(FP2_PEAKS, np.float32)
    I = np.zeros(FP2_PEAKS, np.float32)
    M[:len(mz)] = mz
    if len(it):
        I[:len(it)] = np.log1p(1000.0 * it / max(it.max(), 1e-9))
    return M, I


def fp2_bits(smiles):
    out = np.zeros((len(smiles), FP2_BITS), np.float32)
    ok = np.zeros(len(smiles), bool)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is not None:
            out[i] = _gen2.GetFingerprintAsNumPy(m)
            ok[i] = True
    return out, ok


# ---- re-ranker (eval/rerank.py): class-4 CV 0.65-0.66 vs analog alone 0.616 ----
RR_FEATS = ["ana3", "ana1", "ana3_rel", "ana_rank", "ll", "ll_rank", "n"]


def rr_features(M, ll):
    Ms = np.sort(M, 0)
    ana3, ana1 = Ms[-3:].sum(0), Ms[-1]
    ll = np.asarray(ll, float)
    good = ll > -1e8
    ll = np.where(good, ll, (ll[good].min() - 50) if good.any() else 0.0)
    return pd.DataFrame({"ana3": ana3, "ana1": ana1, "ana3_rel": ana3 - ana3.max(),
                         "ana_rank": np.argsort(np.argsort(-ana3)), "ll": ll - ll.max(),
                         "ll_rank": np.argsort(np.argsort(-ll)), "n": M.shape[1]})[RR_FEATS]


LGB_MODEL = r"""''' + model + '''"""
'''
anchor = "\n\ndef main():"
assert anchor in src
src = src.replace(anchor, FP2 + anchor, 1)

src = src.replace('''    out = []
    for mid, g in test.groupby("molecule_id"):''', '''    fp2s = []
    for path in FP2_WEIGHTS[:len(FP2_WEIGHTS) if FP2_ENS else 1]:
        net = FP2Net()
        net.load_state_dict(torch.load(path, map_location="cpu"))
        fp2s.append(net.eval())
    fp2 = fp2s[0] if fp2s else None
    ranker = lgb.Booster(model_str=LGB_MODEL)
    log("fp2", FP2_WEIGHTS[:len(fp2s)], "gate", LIB_GATE, "re-ranker trees", ranker.num_trees())
    aidx = {a: i for i, a in enumerate(FP2_ADDUCTS)}

    out = []
    for mid, g in test.groupby("molecule_id"):''', 1)

old_score = '''        score = {c: (lib[c] + 1.0 if lib[c] >= LIB_GATE else 0.0) + ana[c] for c in cands}'''
new_score = '''        score = {c: (lib[c] + 1.0 if lib[c] >= LIB_GATE else 0.0) + ana[c] for c in cands}
        if fp2 is not None and cands:
            toks = [fp2_tokens(s.ms2_mzs, s.ms2_normalized_intensities, float(s.precursor_mz)) for _, s in g.iterrows()]
            with torch.no_grad():
                p = torch.stack([torch.sigmoid(net(torch.from_numpy(np.stack([t[0] for t in toks])),
                                                   torch.from_numpy(np.stack([t[1] for t in toks])),
                                                   torch.from_numpy(g.precursor_mz.values.astype(np.float32)),
                                                   torch.from_numpy(g.adduct.map(aidx).fillna(len(FP2_ADDUCTS)).values.astype(np.int64))))
                                 for net in fp2s]).mean(0)
            p = p.mean(0).clamp(1e-4, 1 - 1e-4).numpy()
            bits, okb = fp2_bits([smi.get(c) for c in cands])
            ll = bits @ np.log(p) + (1 - bits) @ np.log(1 - p)
            ll[~okb] = -1e9
            if FUSE == "lgb":
                pred = ranker.predict(rr_features(M[:, :len(cands)], ll))
            else:
                a = np.array([ana[c] for c in cands])
                if FUSE.startswith("rrf"):
                    k = float(FUSE[3:])
                    ra = np.argsort(np.argsort(-a, kind="stable"))
                    rf = np.argsort(np.argsort(-ll, kind="stable"))
                    pred = 1 / (k + ra) + 1 / (k + rf)
                else:
                    parts = FUSE[3:].split("_")
                    t, w = float(parts[0]), float(parts[1])
                    if parts[2:] == ["rel"]:
                        w *= max(float(a.max()), 1e-9)
                    pred = a + w * np.exp((ll - ll.max()) / t)
            if TP_F != 1.0:
                dem = np.array([bool(lib_by_ik.get(c)) and lib[c] < LIB_GATE and lib[c] < TP_L for c in cands])
                pred = np.where(dem, TP_F * pred, pred)
            if POP:
                pred = pred + POP_MU * np.log1p(np.array([float(POP.get(c, 0)) for c in cands]))
            if LA_A:
                lv = np.array([lib[c] for c in cands])
                pred = pred + np.where((lv >= LA_L) & (lv < LIB_GATE), LA_A * lv, 0.0)
            # the library gate keeps its LB-checked role; the re-ranker orders everything else
            score = {c: (1000.0 + lib[c] if lib[c] >= LIB_GATE else 0.0) + float(pr) for c, pr in zip(cands, pred)}'''
assert old_score in src
src = src.replace(old_score, new_score, 1)

# ---- Enamine tier: extra candidates at fixed ranks (only when no library match reaches the gate) ----
old = '    log("pool", len(S))\n'
assert old in src
src = src.replace(old, old + '''    em = None
    if (EN_SLOTS or EN_MERGE) and _en_path:
        E = pd.read_parquet(_en_path[0], columns=["inchikey14", "smiles", "fM"])
        E = E[~E.inchikey14.isin(set(S.inchikey14))].sort_values("fM")
        em, eik, esm = E.fM.values, E.inchikey14.values, E.smiles.values
        del E
        log("enamine tier (not in pool)", len(em), "slots", EN_SLOTS)
''', 1)
old = '''    for mid, g in test.groupby("molecule_id"):
        cands = cand[mid]
'''
assert old in src
src = src.replace(old, old + "        etop = []\n", 1)
old = "            # the library gate keeps its LB-checked role"
assert old in src
src = src.replace(old, '''            if em is not None and np.isfinite(tm.M[mid]):
                elo, ehi = np.searchsorted(em, [tm.M[mid] * (1 - PPM * 1e-6), tm.M[mid] * (1 + PPM * 1e-6)])
                if ehi > elo:
                    ek, es = eik[elo:ehi], esm[elo:ehi]
                    efp = [fp(x) for x in es]
                    eok = [i for i, f in enumerate(efp) if f is not None]
                    EM = np.zeros((max(len(items), 1), len(es)))
                    for k, (aw, a) in enumerate(items):
                        if eok:
                            EM[k, eok] = (aw ** POW) * np.array(DataStructs.BulkTanimotoSimilarity(getfp(a), [efp[i] for i in eok]))
                    eb, ebok = fp2_bits(list(es))
                    ell = eb @ np.log(p) + (1 - eb) @ np.log(1 - p)
                    ell[~ebok] = -1e9
                    epred = np.sort(EM, 0)[-TOP_K:].sum(0) + EN_W * np.exp((ell - ell.max()) / EN_T)
                    epred[[f is None for f in efp]] = -1e9
                    top = np.argsort(-epred, kind="stable")[:10 if EN_MERGE else len(EN_SLOTS)]
                    etop = [(ek[i], es[i], float(epred[i])) for i in top if epred[i] > -1e8]
''' + old, 1)
old = "        ranked = sorted(cands, key=lambda c: -score[c])[:25]\n"
assert old in src
src = src.replace(old, old + '''        if etop and not any(lib[c] >= LIB_GATE for c in cands):
            for k_, s_, _ in etop:
                smi[k_] = s_
            if EN_MERGE:
                both = [(score[c], c) for c in cands] + [(EN_MERGE * v_, k_) for k_, _, v_ in etop]
                ranked = [c for _, c in sorted(both, key=lambda x: -x[0])[:25]]
            else:
                for r_, (k_, _, _) in zip(EN_SLOTS, etop):
                    ranked.insert(r_ - 1, k_)
                ranked = ranked[:25]
''', 1)

if GATE is not None:
    old = 'LIB_GATE = float(os.environ.get("CASMI_LIB_GATE", "0.8"))'
    assert old in src
    src = src.replace(old, 'LIB_GATE = float(os.environ.get("CASMI_LIB_GATE", "' + GATE + '"))', 1)

open(os.path.join(HERE, "main.py"), "w", encoding="utf-8").write(src)
print("wrote", os.path.join(HERE, "main.py"), MODE, len(src), "chars")
