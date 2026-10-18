"""frag2 support: intrinsic substructure CAPABILITIES of a structure for diagnostic neutral losses (SMILES only ->
identical for train / COCONUT / generated structures; no provenance). Precomputed for the full pool by res_frag2.py
(res/frag2/pool_cap.npy, uint8 (n_pool, len(CAPS)), by pid); generated candidates call caps_of_smiles (same function).

Diagnostic losses (LOSSES): (name, neutral mass, required capabilities (any of), specific?) + pairwise sums of the
sugar / acyl losses (both units required; the same unit twice needs count >= 2)."""
from __future__ import annotations

import numpy as np

# capability name -> SMARTS (count = number of unique matches); '|' separates alternatives (counts summed)
CAP_SMARTS = [
    ('oh_aliph', '[OX2H1][CX4]'),
    ('cooh', '[CX3](=O)[OX2H1,OX1-]'),
    ('phenol', '[OX2H1]c'),
    ('nh2', '[#7;H2,H3;!$([#7]-[#7,#8]);!$([#7]=*)]'),
    ('carbonyl', '[#6][CX3](=[OX1])'),
    ('ester', '[#6][CX3](=O)[OX2][#6]'),
    ('acetyl', '[CH3][CX3](=O)[OX2,NX3]'),
    ('ome', '[CH3][OX2][#6]'),
    ('ome_ar', '[CH3][OX2]c'),
    ('nme', '[CH3][NX3,NX4+]'),
    ('hex', '[O,N,n][C;R]1[O;R][C;R]([CH2][OX2])[C;R]([OX2])[C;R]([OX2])[C;R]1[OX2]'
            '|[O,N,n][C;R]1([CH2][OX2])[O;R][C;R]([CH2][OX2])[C;R]([OX2])[C;R]1[OX2]'),
    ('dhex', '[O,N,n][C;R]1[O;R][C;R]([CH3])[C;R]([OX2])[C;R]([OX2])[C;R]1[OX2]'),
    ('pent', '[O,N,n][C;R]1[O;R][CH2;R][C;R]([OX2])[C;R]([OX2])[C;R]1[OX2]'
             '|[O,N,n][CH1;R]1[O;R][C;R]([CH2][OX2])[C;R]([OX2])[C;R]1[OX2]'),
    ('glca', '[O,N,n][C;R]1[O;R][C;R]([CX3](=O)[OX2,OX1-])[C;R]([OX2])[C;R]([OX2])[C;R]1[OX2]'),
    ('cglyc', 'c[C;R]1[O;R][C;R][C;R]([OX2])[C;R]([OX2])[C;R]1[OX2]|c[C;R]1[O;R][C;R][C;R]([OX2])[C;R]1[OX2]'),
    ('malonyl', '[OX2][CX3](=O)[CH2][CX3](=O)[OX2H1,OX1-]'),
    ('coum', '[OX2,NX3][CX3](=O)[CX3]=[CX3]c1[cH][cH]c([OX2H1])[cH][cH]1'),
    ('caff', '[OX2,NX3][CX3](=O)[CX3]=[CX3]c1[cH][cH]c([OX2H1])c([OX2H1])[cH]1'),
    ('ferul', '[OX2,NX3][CX3](=O)[CX3]=[CX3]c1[cH][cH]c([OX2H1])c([OX2][CH3])[cH]1'),
    ('galloyl', '[OX2][CX3](=O)c1cc([OX2H1])c([OX2H1])c([OX2H1])c1'),
    ('sulf', '[SX4](=O)(=O)[OX2H1,OX1-]'),
    ('phos', '[PX4](=O)[OX2H1,OX1-]'),
]
CAPS = [c for c, _ in CAP_SMARTS]
CI = {c: i for i, c in enumerate(CAPS)}

# (name, neutral mass, required capabilities (any of), specific)
LOSSES = [
    ('H2O', 18.010565, ('oh_aliph', 'cooh'), False),
    ('CO', 27.994915, ('carbonyl', 'phenol'), False),
    ('CO2', 43.989829, ('cooh', 'ester'), False),
    ('CH2O2', 46.005479, ('cooh',), False),
    ('CH3', 15.023475, ('ome_ar', 'nme'), False),
    ('CH4O', 32.026215, ('ome',), False),
    ('NH3', 17.026549, ('nh2',), True),
    ('CH5N', 31.042199, ('nme', 'nh2'), True),
    ('C2H7N', 45.057849, ('nme',), True),
    ('C3H9N', 59.073499, ('nme',), True),
    ('C2H2O', 42.010565, ('acetyl',), True),
    ('C2H4O2', 60.021129, ('acetyl', 'cglyc', 'hex', 'pent'), True),
    ('C3H6O3', 90.031694, ('cglyc',), True),
    ('C4H8O4', 120.042259, ('cglyc',), True),
    ('hex', 162.052824, ('hex',), True),
    ('dhex', 146.057909, ('dhex',), True),
    ('pent', 132.042259, ('pent',), True),
    ('glca', 176.032088, ('glca',), True),
    ('malonyl', 86.000394, ('malonyl',), True),
    ('coum', 146.036779, ('coum',), True),
    ('caff', 162.031694, ('caff',), True),
    ('ferul', 176.047344, ('ferul',), True),
    ('galloyl', 152.010959, ('galloyl',), True),
    ('SO3', 79.956815, ('sulf',), True),
    ('H2SO4', 97.967380, ('sulf',), True),
    ('HPO3', 79.966331, ('phos',), True),
    ('H3PO4', 97.976896, ('phos',), True),
]
# Pairwise losses: only chemically common two-unit losses (sugar + sugar, acylated hexose). A scan of 500 sim queries
# showed that the full cross product (acyl + acyl, deoxyhexose + acyl, ...) is dominated by chance matches of large
# losses (observed, supported by neither truth nor decoys) -> noise; curated list below.
PAIRS = [('hex', 'hex'), ('hex', 'dhex'), ('hex', 'pent'), ('hex', 'glca'), ('dhex', 'dhex'), ('dhex', 'pent'),
         ('pent', 'pent'), ('glca', 'glca'), ('hex', 'malonyl'), ('hex', 'coum'), ('hex', 'caff'), ('hex', 'ferul'),
         ('hex', 'galloyl'), ('hex', 'C2H2O')]
# capabilities whose diagnostic loss is expected when present ("expected but not observed" count)
SPEC_CAPS = ['nh2', 'nme', 'acetyl', 'cglyc', 'hex', 'dhex', 'pent', 'glca', 'malonyl', 'coum', 'caff', 'ferul',
             'galloyl', 'sulf', 'phos']


def loss_table():
    """-> names, masses (D,), specific (D,) bool, req: list of (kind, caps_a, caps_b) with kind 'any' | 'pair'."""
    names, masses, spec, req = [], [], [], []
    by = {l[0]: l for l in LOSSES}
    for n, m, caps, sp in LOSSES:
        names.append(n); masses.append(m); spec.append(sp); req.append(('any', tuple(CI[c] for c in caps), None))
    for ua, ub in PAIRS:
        la, lb = by[ua], by[ub]
        names.append(f'{la[0]}+{lb[0]}'); masses.append(la[1] + lb[1]); spec.append(True)
        req.append(('pair', tuple(CI[c] for c in la[2]), tuple(CI[c] for c in lb[2])))
    return names, np.array(masses), np.array(spec, bool), req


def support_matrix(caps: np.ndarray, req) -> np.ndarray:
    """caps (n, C) counts -> (n, D) bool: the candidate has the substructure(s) a diagnostic loss requires."""
    caps = np.asarray(caps, np.int32)
    out = np.zeros((caps.shape[0], len(req)), bool)
    for d, (kind, a, b) in enumerate(req):
        ca = caps[:, list(a)].sum(1)
        if kind == 'any':
            out[:, d] = ca > 0
        elif a == b:
            out[:, d] = ca >= 2
        else:
            out[:, d] = (ca > 0) & (caps[:, list(b)].sum(1) > 0)
    return out


def loss_caps(req):
    """(D, C) bool: capabilities a loss involves (for the 'expected but not observed' logic)."""
    M = np.zeros((len(req), len(CAPS)), bool)
    for d, (kind, a, b) in enumerate(req):
        M[d, list(a)] = True
        if b is not None:
            M[d, list(b)] = True
    return M


_P = {}


def _pats():
    if not _P:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        _P['Chem'] = Chem
        _P['pats'] = [[Chem.MolFromSmarts(s) for s in sm.split('|')] for _, sm in CAP_SMARTS]
        assert all(p is not None for ps in _P['pats'] for p in ps)
    return _P


def caps_of_smiles(smiles: str) -> np.ndarray:
    """uint8 (len(CAPS),) capability counts (capped at 255); zeros when RDKit fails."""
    out = np.zeros(len(CAPS), np.uint8)
    try:
        p = _pats(); mol = p['Chem'].MolFromSmiles(smiles)
        if mol is None:
            return out
        for j, ps in enumerate(p['pats']):
            out[j] = min(255, sum(len(mol.GetSubstructMatches(q, maxMatches=255)) for q in ps))
    except Exception:
        return np.zeros(len(CAPS), np.uint8)
    return out
