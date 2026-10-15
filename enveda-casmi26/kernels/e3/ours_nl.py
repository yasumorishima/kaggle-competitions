"""ours_nl: neutral-loss substructure evidence (ours).

A peak at (precursor m/z - L) with L a diagnostic neutral loss (hexose 162.0528, glucuronic acid 176.0321, SO3 79.9568,
H3PO4 97.9769, CO2 43.9898, NH3 17.0265, CH3 radical 15.0235, ...) says the molecule carries the matching group.
Each rule has a SMARTS for the group and two rates measured by us on train (enveda-np-examples, gnps, mona,
massbank, riken, msdial; up to 3 spectra per molecule, peaks >= 5% of the base peak, tolerance max(5 mDa, 10 ppm)):
P(loss seen | group present) and P(loss seen | group absent). A candidate's score is the naive-Bayes log likelihood
of the molecule's observed / not observed losses given whether the candidate has each group. Isomers that differ in
the group (glycoside vs aglycone-C-isomer, acid vs ester, sulfate vs sulfonate ...) get separated; candidates that
agree on every group get the same score (the order is then unchanged). The log likelihood enters raw (not z-scored).
Losses are adduct-invariant (precursor m/z - fragment m/z), so every polarity and adduct is used.
With USE_IONS, diagnostic product ions (phosphocholine 184.073, benzoyl 105.033, caffeoyl 163.039, galloyl 153.018,
immonium ions, HSO4- 96.960, PO3- 78.959, ...) join the evidence the same way (rates measured on the same train set).
"""
import math

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
REL_MIN, TOL_DA, TOL_PPM = 0.05, 0.005, 10.0
RULES = {  # name: (loss Da, SMARTS of the group, P(seen | has), P(seen | not))
    "hexose": (162.05282, '[OX2][C;R1]1[O;R1][C;R1]([CH2][OX2])[C;R1]([OX2])[C;R1]([OX2])[C;R1]1[OX2]', 0.1375, 0.00385),
    "deoxyhexose": (146.05791, '[OX2][C;R1]1[O;R1][C;R1]([CH3])[C;R1]([OX2])[C;R1]([OX2])[C;R1]1[OX2]', 0.0709, 0.00496),
    "pentose": (132.04226, '[OX2][C;R1]1[O;R1][CH2;R1][C;R1]([OX2])[C;R1]([OX2])[C;R1]1[OX2]', 0.0641, 0.00807),
    "glucuronide": (176.03209, '[OX2][C;R1]1[O;R1][C;R1](C(=O)[OX2])[C;R1]([OX2])[C;R1]([OX2])[C;R1]1[OX2]', 0.1188, 0.00151),
    "co2_acid": (43.98983, '[CX3](=O)[OX2H1,OX1-]', 0.0582, 0.00658),
    "so3": (79.95681, 'S(=O)(=O)[OX2,OX1-]', 0.0753, 0.00040),
    "nh3_amine": (17.02655, '[NX3;H2][CX4]', 0.2337, 0.02231),
    "tma": (59.07350, '[NX4+](C)(C)(C)C', 0.0743, 0.00600),
    "h3po4": (97.97690, 'P(=O)(O)O', 0.1980, 0.00040),
    "ch3_methoxy": (15.02348, '[CH3][OX2][c]', 0.0671, 0.00676),
    "hexose_x2": (324.10565, '[OX2][C;R1]1[O;R1][C;R1]([CH2][OX2])[C;R1]([OX2])[C;R1]([OX2])[C;R1]1[OX2].[OX2][C;R1]1[O;R1][C;R1]([CH2][OX2])[C;R1]([OX2])[C;R1]([OX2])[C;R1]1[OX2]', 0.0633, 0.00172),
    "galloyl": (152.01096, 'c1c(O)c(O)c(O)cc1C(=O)O', 0.0277, 0.00185),
    "caffeoyl": (162.03169, 'O=C(C=Cc1ccc(O)c(O)c1)O', 0.0246, 0.00289),
    "coumaroyl": (146.03678, 'O=C(C=Cc1ccc(O)cc1)O', 0.0142, 0.00231),
    "malonyl": (86.00039, 'O=C(CC(=O)O)O', 0.0250, 0.00236),
    "acetyl": (42.01057, '[CH3]C(=O)[OX2][CX4]', 0.0931, 0.01093)
}
ION_RULES = {  # name: (product ion m/z, SMARTS of the group, P(seen | has), P(seen | not)); same train set and peak rule
    "phosphocholine": (184.07332, 'C[N+](C)(C)CCOP(=O)([OX2,OX1-])[OX2]', 0.2986, 0.00610),
    "choline": (104.10699, 'C[N+](C)(C)CC[OX2]', 0.0675, 0.00061),
    "coumaroyl_ion": (147.04406, '[#6](=O)[CH]=[CH]c1ccc([OX2])cc1', 0.2695, 0.01461),
    "caffeoyl_ion": (163.03897, '[#6](=O)[CH]=[CH]c1ccc([OX2])c([OX2])c1', 0.2865, 0.00841),
    "feruloyl_ion": (177.05462, '[#6](=O)[CH]=[CH]c1ccc([OX2])c(O[CH3])c1', 0.2371, 0.00910),
    "galloyl_ion": (153.01824, '[#6](=O)c1cc([OX2])c([OX2])c([OX2])c1', 0.1938, 0.00476),
    "benzoyl_ion": (105.03349, '[CX3](=O)c1[cH][cH][cH][cH][cH]1', 0.5700, 0.01103),
    "tropylium": (91.05423, '[CH2]c1[cH][cH][cH][cH][cH]1', 0.3752, 0.09990),
    "hydroxybenzoyl_ion": (121.02841, '[CX3](=O)c1ccc([OH])cc1', 0.0856, 0.01400),
    "indole_ion": (130.06513, 'c1ccc2c(c1)cc[nX3]2', 0.1737, 0.02390),
    "leu_immonium": (86.09643, '[CH3][CH]([CH3])C[CH](N)C(=O)', 0.3005, 0.02186),
    "phe_immonium": (120.08078, 'c1ccccc1C[CH](N)C(=O)', 0.2580, 0.02011),
    "tyr_immonium": (136.07569, 'c1cc(O)ccc1C[CH](N)C(=O)', 0.2996, 0.00943),
    "hso4": (96.96010, '[OX2,OX1-]S(=O)(=O)[OX2]', 0.1312, 0.00066),
    "po3": (78.95905, 'P(=O)([OX2,OX1-])[OX2,OX1-]', 0.0453, 0.00056),
    "glycerophosphate": (152.99583, 'OCC(O)COP(=O)(O)O', 0.0271, 0.00073),
    "quinic_ion": (191.05611, 'OC1CC(O)(C(=O)O)CC(O)C1O', 0.1250, 0.00325)
}
USE_IONS = False   # set by the notebook (nlion=1): diagnostic product ions join the evidence
_PAT = {k: Chem.MolFromSmarts(v[1]) for k, v in {**RULES, **ION_RULES}.items()}
_LL = {k: ((math.log(v[2]), math.log(1 - v[2])), (math.log(max(v[3], 1e-5)), math.log(1 - v[3])))
       for k, v in {**RULES, **ION_RULES}.items()}


def observed(spectra):
    """spectra: [(peaks Nx2 [mz, intensity], precursor_mz)] -> {rule: seen in any spectrum}."""
    seen = {k: False for k in RULES}
    if USE_IONS:
        seen.update({k: False for k in ION_RULES})
    for pk, prec in spectra:
        pk = np.asarray(pk, float)
        if not len(pk) or prec is None or not np.isfinite(prec):
            continue
        it = pk[:, 1]
        if it.max() <= 0:
            continue
        loss = prec - pk[it >= REL_MIN * it.max(), 0]
        tol = max(TOL_DA, prec * TOL_PPM * 1e-6)
        for k, v in RULES.items():
            if not seen[k] and np.any(np.abs(loss - v[0]) <= tol):
                seen[k] = True
        if USE_IONS:
            ions = pk[it >= REL_MIN * it.max(), 0]
            for k, v in ION_RULES.items():
                if not seen[k] and np.any(np.abs(ions - v[0]) <= tol):
                    seen[k] = True
    return seen


def groups(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return None
    return {k: m.HasSubstructMatch(p) for k, p in _PAT.items()}


def score(smi, seen):
    """Naive-Bayes log likelihood of the observed losses given the candidate's groups (None: unparsable)."""
    g = groups(smi)
    if g is None:
        return None
    s = 0.0
    for k, obs in seen.items():
        has_ll, not_ll = _LL[k]
        s += (has_ll if g[k] else not_ll)[0 if obs else 1]
    return s


def score_list(smis, seen):
    sc = [score(s, seen) for s in smis]
    ok = [x for x in sc if x is not None]
    fill = min(ok) if ok else 0.0
    return [fill if x is None else x for x in sc]


def rerank(smis, nl, lam):
    """Order of indices by z(-log rank) + lam * nl, nl the raw log likelihood (not z-scored: a weak rule, such as an
    unseen loss, must stay a small term; a seen hexose loss separates has / has-not by 3.6 nats)."""
    r = -np.log(np.arange(1, len(smis) + 1, dtype=np.float64))
    zr = (r - r.mean()) / r.std() if r.std() > 0 else np.zeros_like(r)
    f = np.asarray(nl, np.float64)
    sc = zr + lam * (f - f.max())
    return list(np.argsort(-sc, kind="stable"))
