"""Chemistry helpers: adducts, neutral mass, formulas, score keys, fingerprints.

Score key = tautomer-canonical InChIKey14 exactly as the competition metric computes it
(RDKit 2026.03.3, ``TautomerEnumerator().Canonicalize`` -> ``MolToInchiKey`` -> first 14 chars).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

# Monoisotopic masses of elements (NIST).
MASS: Dict[str, float] = dict(
    C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, P=30.97376163,
    S=31.97207100, F=18.99840322, Cl=34.96885268, Br=78.9183371, I=126.904473,
    Na=22.9897692809, K=38.96370668, Si=27.9769265325, B=11.0093054, Se=79.9165213,
    Li=7.01600455, Mg=23.985041700, Ca=39.96259098, Fe=55.9349375, Cu=62.9295975,
    Zn=63.9291422, As=74.9215965, Al=26.98153863, Co=58.933195, Mn=54.9380451,
    Ni=57.9353429, Cr=51.9405075, Sn=119.9021947, Hg=201.970643, Pt=194.9647911,
    Au=196.9665687, Ag=106.905097, Ti=47.9479463, V=50.9439595, Ge=73.9211778,
    Te=129.9062244, Sb=120.9038157, Bi=208.9803987, Ba=137.9052472, Sr=87.9056121,
    Rb=84.911789738, Cs=132.905451933, Be=9.0121822, Ga=68.9255736, W=183.9509312,
    Mo=97.9054082, Pb=207.9766521, Tl=204.9744275, Cd=113.9033585, Zr=89.9047044,
    Pd=105.903486, Rh=102.905504, Ru=101.9043493, Ir=192.9629264, Os=191.9614807,
    Re=186.9557531, Nb=92.9063781, Y=88.9058483, Sc=44.9559119, La=138.9063533,
    Ce=139.9054387, Nd=141.9077233, Eu=152.9212303, Gd=157.9241039, Tb=158.9253468,
    Dy=163.9291748, Ho=164.9303221, Er=165.9302931, Tm=168.9342133, Yb=173.9388621,
    Lu=174.9407718, Th=232.0380553, U=238.0507882, Hf=179.94655, Ta=180.9479958,
    In=114.903878, Xe=131.9041535, Kr=83.911507, Ar=39.9623831225, Ne=19.9924401754,
    He=4.00260325415, D=2.0141017778,
)
ELECTRON = 0.00054857990
H2O = 2 * MASS['H'] + MASS['O']
NH3 = MASS['N'] + 3 * MASS['H']
PROTON = MASS['H'] - ELECTRON

# adduct -> (n_M, charge_abs, delta, polarity)  where m/z*|z| = n_M * M + delta
# delta already contains the electron correction.
def _ad(n, z, formula_delta, sign):
    """formula_delta: mass of atoms gained (positive) / lost (negative); sign=+1 positive mode."""
    return (n, z, formula_delta - sign * z * ELECTRON, sign)

_H = MASS['H']; _C = MASS['C']; _N = MASS['N']; _O = MASS['O']
ADDUCTS: Dict[str, Tuple[int, int, float, int]] = {
    "[M+H]+": _ad(1, 1, _H, +1),
    "[M+NH4]+": _ad(1, 1, _N + 4 * _H, +1),
    "[M+Na]+": _ad(1, 1, MASS['Na'], +1),
    "[M+K]+": _ad(1, 1, MASS['K'], +1),
    "[M+Li]+": _ad(1, 1, MASS['Li'], +1),
    "[M-H2O+H]+": _ad(1, 1, _H - H2O, +1),
    "[M-2H2O+H]+": _ad(1, 1, _H - 2 * H2O, +1),
    "[M-H2O]+": _ad(1, 1, -H2O, +1),
    "[M]+": _ad(1, 1, 0.0, +1),
    "[M+2H]2+": _ad(1, 2, 2 * _H, +1),
    "[M+3H]3+": _ad(1, 3, 3 * _H, +1),
    "[M+H+Na]2+": _ad(1, 2, _H + MASS['Na'], +1),
    "[M+HNa]2+": _ad(1, 2, _H + MASS['Na'], +1),
    "[M+2Na]2+": _ad(1, 2, 2 * MASS['Na'], +1),
    "[M+2Na-H]+": _ad(1, 1, 2 * MASS['Na'] - _H, +1),
    "[M+Ca]2+": _ad(1, 2, MASS['Ca'], +1),
    "[M+CH3OH+H]+": _ad(1, 1, _H + _C + 4 * _H + _O, +1),
    "[M+CH3CN+H]+": _ad(1, 1, _H + 2 * _C + 3 * _H + _N, +1),
    "[M+CH3CN+Na]+": _ad(1, 1, MASS['Na'] + 2 * _C + 3 * _H + _N, +1),
    "[M+C2H4N]+": _ad(1, 1, 2 * _C + 4 * _H + _N, +1),
    "[M+CH5O]+": _ad(1, 1, _C + 5 * _H + _O, +1),
    "[M-H2N]+": _ad(1, 1, -(_N + 2 * _H), +1),
    "[M-NH3+H]+": _ad(1, 1, _H - NH3, +1),
    "[M-H5O3]+": _ad(1, 1, -(5 * _H + 3 * _O), +1),
    "[M-CH3]+": _ad(1, 1, -(_C + 3 * _H), +1),
    "[M+H-CH3]+": _ad(1, 1, _H - (_C + 3 * _H), +1),
    "[M-H]-": _ad(1, 1, -_H, -1),
    "[M-2H]-": _ad(1, 1, -2 * _H, -1),
    "[M-2H]2-": _ad(1, 2, -2 * _H, -1),
    "[M-H2]2-": _ad(1, 2, -2 * _H, -1),
    "[M]-": _ad(1, 1, 0.0, -1),
    "[M-H2O-H]-": _ad(1, 1, -_H - H2O, -1),
    "[M+CH2O2-H]-": _ad(1, 1, _C + 2 * _H + 2 * _O - _H, -1),
    "[M+HCOO]-": _ad(1, 1, _C + 2 * _H + 2 * _O - _H, -1),
    "[M+C2H4O2-H]-": _ad(1, 1, 2 * _C + 4 * _H + 2 * _O - _H, -1),
    "[M+CH3COO]-": _ad(1, 1, 2 * _C + 4 * _H + 2 * _O - _H, -1),
    "[M+Cl]-": _ad(1, 1, MASS['Cl'], -1),
    "[M+Br]-": _ad(1, 1, MASS['Br'], -1),
    "[M+Na-2H]-": _ad(1, 1, MASS['Na'] - 2 * _H, -1),
    "[M+K-2H]-": _ad(1, 1, MASS['K'] - 2 * _H, -1),
    "[M+Na-H2]-": _ad(1, 1, MASS['Na'] - 2 * _H, -1),
    "[M+K-H2]-": _ad(1, 1, MASS['K'] - 2 * _H, -1),
    "[M-CH3]-": _ad(1, 1, -(_C + 3 * _H), -1),
    "[M-C3H6NO2]-": _ad(1, 1, -(3 * _C + 6 * _H + _N + 2 * _O), -1),
    "[M+C2F3O2]-": _ad(1, 1, 2 * _C + 3 * MASS['F'] + 2 * _O, -1),
    "[M+CF3COO]-": _ad(1, 1, 2 * _C + 3 * MASS['F'] + 2 * _O, -1),
    "[2M+H]+": _ad(2, 1, _H, +1),
    "[2M+Na]+": _ad(2, 1, MASS['Na'], +1),
    "[2M+K]+": _ad(2, 1, MASS['K'], +1),
    "[2M+NH4]+": _ad(2, 1, _N + 4 * _H, +1),
    "[2M+Ca]2+": _ad(2, 2, MASS['Ca'], +1),
    "[2M-H]-": _ad(2, 1, -_H, -1),
    "[2M+CH2O2-H]-": _ad(2, 1, _C + 2 * _H + 2 * _O - _H, -1),
    "[2M+C2H4O2-H]-": _ad(2, 1, 2 * _C + 4 * _H + 2 * _O - _H, -1),
    "[2M+Na-2H]-": _ad(2, 1, MASS['Na'] - 2 * _H, -1),
    "[2M+Cl]-": _ad(2, 1, MASS['Cl'], -1),
    "[3M+H]+": _ad(3, 1, _H, +1),
    "[3M+Na]+": _ad(3, 1, MASS['Na'], +1),
    "[3M-H]-": _ad(3, 1, -_H, -1),
}
# The ten adducts the hidden test set uses.
TEST_ADDUCTS = ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
                "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"]
ADDUCT_LIST = TEST_ADDUCTS + [a for a in ADDUCTS if a not in TEST_ADDUCTS] + ["<unk>"]
ADDUCT_IX = {a: i for i, a in enumerate(ADDUCT_LIST)}


def adduct_index(a: str) -> int:
    return ADDUCT_IX.get(a, ADDUCT_IX["<unk>"])


def neutral_mass(mz, adduct) -> np.ndarray:
    """Neutral monoisotopic mass of M from measured precursor m/z and adduct string(s)."""
    mz = np.asarray(mz, dtype=np.float64)
    ad = np.asarray(adduct, dtype=object)
    out = np.full(len(mz), np.nan)
    for a, (n, z, d, _) in ADDUCTS.items():
        m = ad == a
        if m.any():
            out[m] = (mz[m] * z - d) / n
    return out


def precursor_mz_from_mass(mass: float, adduct: str) -> float:
    n, z, d, _ = ADDUCTS[adduct]
    return (n * mass + d) / z


def adduct_delta(adduct: str) -> Tuple[int, int, float, int]:
    return ADDUCTS[adduct]


_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(f: str) -> Optional[Dict[str, int]]:
    """'C10H12N2O' -> {'C':10,'H':12,'N':2,'O':1}. Returns None on garbage."""
    if not isinstance(f, str) or not f:
        return None
    f = f.replace('+', '').replace('-', '')
    pos = 0
    out: Dict[str, int] = {}
    for m in _FORMULA_RE.finditer(f):
        if m.start() != pos:
            return None
        pos = m.end()
        el, cnt = m.group(1), m.group(2)
        if el not in MASS:
            return None
        out[el] = out.get(el, 0) + (int(cnt) if cnt else 1)
    if pos != len(f):
        return None
    return out


def formula_mass(counts: Dict[str, int]) -> float:
    return float(sum(MASS[e] * n for e, n in counts.items()))


def formula_string(counts: Dict[str, int]) -> str:
    """Hill order string."""
    keys = list(counts.keys())
    order = []
    if 'C' in counts:
        order.append('C')
        if 'H' in counts:
            order.append('H')
        order += sorted(k for k in keys if k not in ('C', 'H'))
    else:
        order = sorted(keys)
    return ''.join(f"{k}{counts[k] if counts[k] != 1 else ''}" for k in order if counts[k] > 0)


# ----------------------------------------------------------------------------------------------
# RDKit-dependent helpers (imported lazily so numba-only workers don't need RDKit)
# ----------------------------------------------------------------------------------------------
_rd = {}


def _rdkit():
    if not _rd:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator, MACCSkeys
        from rdkit.Chem.MolStandardize import rdMolStandardize
        RDLogger.DisableLog('rdApp.*')
        _rd['Chem'] = Chem
        _rd['Descriptors'] = Descriptors
        _rd['rdMD'] = rdMolDescriptors
        _rd['gen'] = rdFingerprintGenerator
        _rd['MACCS'] = MACCSkeys
        _rd['taut'] = rdMolStandardize.TautomerEnumerator()
        _rd['lfc'] = rdMolStandardize.LargestFragmentChooser()
        _rd['uncharger'] = rdMolStandardize.Uncharger()
    return _rd


def score_key(smiles: str) -> Optional[str]:
    """Tautomer-canonical InChIKey14 exactly like the metric notebook."""
    r = _rdkit()
    try:
        mol = r['Chem'].MolFromSmiles(smiles)
        if mol is None:
            return None
        cm = r['taut'].Canonicalize(mol)
        ik = r['Chem'].MolToInchiKey(cm)
        return ik[:14] if ik else None
    except Exception:
        return None


def raw_key(smiles: str) -> Optional[str]:
    """Plain InChIKey14 (no tautomer canonicalization)."""
    r = _rdkit()
    try:
        mol = r['Chem'].MolFromSmiles(smiles)
        if mol is None:
            return None
        ik = r['Chem'].MolToInchiKey(mol)
        return ik[:14] if ik else None
    except Exception:
        return None


def standardize_smiles(smiles: str, keep_charged: bool = False) -> Optional[str]:
    """Largest fragment, uncharged (if possible), canonical SMILES without stereo.
    Returns None if the molecule cannot be parsed or stays charged (unless keep_charged)."""
    r = _rdkit()
    Chem = r['Chem']
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = r['lfc'].choose(mol)
        mol = r['uncharger'].uncharge(mol)
        if not keep_charged and Chem.GetFormalCharge(mol) != 0:
            return None
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def mol_props(smiles: str):
    """(formula, exact_mass, n_heavy) or None."""
    r = _rdkit()
    try:
        mol = r['Chem'].MolFromSmiles(smiles)
        if mol is None:
            return None
        f = r['rdMD'].CalcMolFormula(mol)
        m = r['Descriptors'].ExactMolWt(mol)
        return f, float(m), mol.GetNumHeavyAtoms()
    except Exception:
        return None


# Fingerprint layout (raw, before informative-bit selection)
FP_BLOCKS = [("ecfp4", 4096), ("ecfp6", 4096), ("fcfp4", 2048), ("rdk6", 2048),
             ("ap", 2048), ("tt", 2048), ("maccs", 167)]
FP_RAW_BITS = sum(n for _, n in FP_BLOCKS)


def _fp_gens():
    r = _rdkit()
    if 'gens' not in r:
        g = r['gen']
        r['gens'] = dict(
            ecfp4=g.GetMorganGenerator(radius=2, fpSize=4096),
            ecfp6=g.GetMorganGenerator(radius=3, fpSize=4096),
            fcfp4=g.GetMorganGenerator(radius=2, fpSize=2048,
                                       atomInvariantsGenerator=g.GetMorganFeatureAtomInvGen()),
            rdk6=g.GetRDKitFPGenerator(fpSize=2048, maxPath=6),
            ap=g.GetAtomPairGenerator(fpSize=2048),
            tt=g.GetTopologicalTorsionGenerator(fpSize=2048),
        )
    return r['gens']


def raw_fingerprint(smiles: str) -> Optional[np.ndarray]:
    """uint8 vector of FP_RAW_BITS (0/1) or None."""
    r = _rdkit()
    gens = _fp_gens()
    try:
        mol = r['Chem'].MolFromSmiles(smiles)
        if mol is None:
            return None
        parts = [gens[name].GetFingerprintAsNumPy(mol).astype(np.uint8)
                 for name, _ in FP_BLOCKS if name != 'maccs']
        parts.append(np.array(list(r['MACCS'].GenMACCSKeys(mol)), dtype=np.uint8))
        return np.concatenate(parts)
    except Exception:
        return None


def struct_record(smiles: str, bits: Optional[np.ndarray] = None):
    """All per-structure fields used downstream. Returns dict or None."""
    p = mol_props(smiles)
    if p is None:
        return None
    fp = raw_fingerprint(smiles)
    if fp is None:
        return None
    if bits is not None:
        fp = fp[bits]
    return dict(formula=p[0], mass=p[1], n_heavy=p[2], fp=np.packbits(fp))


def tanimoto_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A: (n,bits) uint8/float, B: (m,bits) -> (n,m) Tanimoto."""
    A = A.astype(np.float32); B = B.astype(np.float32)
    inter = A @ B.T
    sa = A.sum(1)[:, None]; sb = B.sum(1)[None, :]
    return inter / np.maximum(sa + sb - inter, 1e-9)
