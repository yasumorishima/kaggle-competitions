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
e.g. exp50_0.2+ens+g0.75, exp50_0.2+tp0.7
"""
import os
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "exp50_0.2"   # LB best 0.294 (09-29)
FUSE, *OPTS = MODE.split("+")
ENS = "ens" in OPTS
GATE = next((o[1:] for o in OPTS if o.startswith("g")), None)
TP = next((o[2:].split("_") for o in OPTS if o.startswith("tp")), None)
TP_F, TP_L = (float(TP[0]), float(TP[1]) if len(TP) > 1 else 9.0) if TP else (1.0, 9.0)

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "..", "b1", "main.py"), encoding="utf-8").read()
model = open(os.path.join(HERE, "lgb_model.txt"), encoding="utf-8").read()

src = src.replace('"""CASMI26 B1: mass-window candidates', '"""CASMI26 B5 (b3 + fp2 + LightGBM re-ranker). Built from b1 by b5/build.py.\n\nB1: mass-window candidates', 1)

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
FP2_ENS = ''' + str(ENS) + '''
# fp2 v2 first (the LB-checked model); fp2all (class 1-3 held out) joins only with FP2_ENS
FP2_WEIGHTS = sorted(glob.glob("/kaggle/input/**/fp2.pt", recursive=True), key=lambda p: "fp2all" in p)
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
            # the library gate keeps its LB-checked role; the re-ranker orders everything else
            score = {c: (1000.0 + lib[c] if lib[c] >= LIB_GATE else 0.0) + float(pr) for c, pr in zip(cands, pred)}'''
assert old_score in src
src = src.replace(old_score, new_score, 1)

if GATE is not None:
    old = 'LIB_GATE = float(os.environ.get("CASMI_LIB_GATE", "0.8"))'
    assert old in src
    src = src.replace(old, 'LIB_GATE = float(os.environ.get("CASMI_LIB_GATE", "' + GATE + '"))', 1)

open(os.path.join(HERE, "main.py"), "w", encoding="utf-8").write(src)
print("wrote", os.path.join(HERE, "main.py"), MODE, len(src), "chars")
