"""Class-3 candidate generation: biosynthetically plausible derivatives of strong analogs.

Given an analog structure A (from the analog channel) and the mass difference d = target - mass(A),
enumerate single/double transformations whose exact-mass change matches d within tolerance, apply
them at every applicable site (RDKit reaction SMARTS), and return unique (score key) products.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import chem

# name -> (reaction SMARTS, formula delta as dict)
_TRANSFORMS: Dict[str, Tuple[str, Dict[str, int]]] = {
    # additions
    'hydroxyl_arom': ('[c;H1:1]>>[c:1]O', dict(O=1)),
    'hydroxyl_aliph': ('[C;H2,H3;!$(C=O):1]>>[C:1]O', dict(O=1)),
    'hydroxyl_ch': ('[C;H1;!$(C=O):1]>>[C:1]O', dict(O=1)),
    'methyl_O': ('[O;H1;!$(OC=O):1]>>[O:1]C', dict(C=1, H=2)),
    'methyl_N': ('[N;H1,H2:1]>>[N:1]C', dict(C=1, H=2)),
    'methyl_c': ('[c;H1:1]>>[c:1]C', dict(C=1, H=2)),
    'methoxy_arom': ('[c;H1:1]>>[c:1]OC', dict(C=1, H=2, O=1)),
    'acetyl_O': ('[O;H1;!$(OC=O):1]>>[O:1]C(C)=O', dict(C=2, H=2, O=1)),
    'acetyl_N': ('[N;H1,H2;!$(NC=O):1]>>[N:1]C(C)=O', dict(C=2, H=2, O=1)),
    'malonyl_O': ('[O;H1;!$(OC=O):1]>>[O:1]C(=O)CC(=O)O', dict(C=3, H=2, O=3)),
    'hexose_O': ('[O;H1;!$(OC=O):1]>>[O:1]C1OC(CO)C(O)C(O)C1O', dict(C=6, H=10, O=5)),
    'pentose_O': ('[O;H1;!$(OC=O):1]>>[O:1]C1OCC(O)C(O)C1O', dict(C=5, H=8, O=4)),
    'deoxyhexose_O': ('[O;H1;!$(OC=O):1]>>[O:1]C1OC(C)C(O)C(O)C1O', dict(C=6, H=10, O=4)),
    'glucuronyl_O': ('[O;H1;!$(OC=O):1]>>[O:1]C1OC(C(=O)O)C(O)C(O)C1O', dict(C=6, H=8, O=6)),
    'hexose_C': ('[c;H1:1]>>[c:1]C1OC(CO)C(O)C(O)C1O', dict(C=6, H=10, O=5)),
    'sulfate_O': ('[O;H1;!$(OC=O):1]>>[O:1]S(=O)(=O)O', dict(S=1, O=3)),
    'prenyl_c': ('[c;H1:1]>>[c:1]CC=C(C)C', dict(C=5, H=8)),
    'prenyl_O': ('[O;H1;!$(OC=O):1]>>[O:1]CC=C(C)C', dict(C=5, H=8)),
    'ethyl_O': ('[O;H1;!$(OC=O):1]>>[O:1]CC', dict(C=2, H=4)),
    'hydrogenate_CC': ('[C:1]=[C:2]>>[C:1][C:2]', dict(H=2)),
    'hydrogenate_CO': ('[C:1]=[O:2]>>[C:1][O:2]', dict(H=2)),
    'dehydrogenate': ('[C;H1,H2:1][C;H1,H2:2]>>[C:1]=[C:2]', dict(H=-2)),
    'oxidize_CHOH': ('[C;H1:1][O;H1:2]>>[C:1]=[O:2]', dict(H=-2)),
    'carboxyl_c': ('[c;H1:1]>>[c:1]C(=O)O', dict(C=1, O=2)),
    'carboxyl_CH3': ('[C;H3:1]>>[C:1](=O)O', dict(O=2)),
    'amine_c': ('[c;H1:1]>>[c:1]N', dict(N=1, H=1)),
    'chloro_c': ('[c;H1:1]>>[c:1]Cl', dict(Cl=1, H=-1)),
    'coumaroyl_O': ('[O;H1;!$(OC=O):1]>>[O:1]C(=O)C=Cc1ccc(O)cc1', dict(C=9, H=6, O=2)),
    'feruloyl_O': ('[O;H1;!$(OC=O):1]>>[O:1]C(=O)C=Cc1ccc(O)c(OC)c1', dict(C=10, H=8, O=3)),
    'caffeoyl_O': ('[O;H1;!$(OC=O):1]>>[O:1]C(=O)C=Cc1ccc(O)c(O)c1', dict(C=9, H=6, O=3)),
    'galloyl_O': ('[O;H1;!$(OC=O):1]>>[O:1]C(=O)c1cc(O)c(O)c(O)c1', dict(C=7, H=4, O=4)),
    'benzoyl_O': ('[O;H1;!$(OC=O):1]>>[O:1]C(=O)c1ccccc1', dict(C=7, H=4, O=1)),
    # removals
    'demethyl_O': ('[O:1][CH3]>>[O:1]', dict(C=-1, H=-2)),
    'demethyl_N': ('[N:1][CH3]>>[N:1]', dict(C=-1, H=-2)),
    'dehydroxyl_arom': ('[c:1][OH]>>[c:1]', dict(O=-1)),
    'dehydroxyl_aliph': ('[C;!$(C=O):1][OH]>>[C:1]', dict(O=-1)),
    'deacetyl_O': ('[O:1]C(=O)[CH3]>>[O:1]', dict(C=-2, H=-2, O=-1)),
    'dehexose': ('[O:1]C1OC(CO)C(O)C(O)C1O>>[O:1]', dict(C=-6, H=-10, O=-5)),
    'dehexose2': ('[O:1]C1OC(CO)C(O)C(O)C1O>>[O:1]', dict(C=-6, H=-10, O=-5)),
    'depentose': ('[O:1]C1OCC(O)C(O)C1O>>[O:1]', dict(C=-5, H=-8, O=-4)),
    'dedeoxyhexose': ('[O:1]C1OC(C)C(O)C(O)C1O>>[O:1]', dict(C=-6, H=-10, O=-4)),
    'demalonyl': ('[O:1]C(=O)CC(=O)O>>[O:1]', dict(C=-3, H=-2, O=-3)),
    'desulfate': ('[O:1]S(=O)(=O)O>>[O:1]', dict(S=-1, O=-3)),
    'deprenyl': ('[c:1]CC=C(C)C>>[c:1]', dict(C=-5, H=-8)),
    'decarboxyl': ('[C,c:1]C(=O)[OH]>>[C,c:1]', dict(C=-1, O=-2)),
    'demethoxy': ('[c:1]O[CH3]>>[c:1]', dict(C=-1, H=-2, O=-1)),
}


def _delta_mass(d: Dict[str, int]) -> float:
    return float(sum(chem.MASS[e] * n for e, n in d.items()))


_RXN = {}


def _rxn(name):
    if name not in _RXN:
        from rdkit.Chem import AllChem
        _RXN[name] = AllChem.ReactionFromSmarts(_TRANSFORMS[name][0])
    return _RXN[name]


def _combos(max_steps=2):
    names = list(_TRANSFORMS)
    out = []
    for n in names:
        out.append(((n,), _delta_mass(_TRANSFORMS[n][1])))
    if max_steps >= 2:
        for a, b in itertools.combinations_with_replacement(names, 2):
            d = dict(_TRANSFORMS[a][1])
            for e, v in _TRANSFORMS[b][1].items():
                d[e] = d.get(e, 0) + v
            out.append(((a, b), _delta_mass(d)))
    return out


_COMBOS = None


def matching_transforms(delta: float, tol: float = 0.003, max_steps: int = 2):
    global _COMBOS
    if _COMBOS is None:
        _COMBOS = _combos(2)
    return [c for c, m in _COMBOS if abs(m - delta) <= tol and len(c) <= max_steps]


def apply_transform(mol, name, max_products=64):
    from rdkit import Chem
    rxn = _rxn(name)
    out = []
    try:
        prods = rxn.RunReactants((mol,))
    except Exception:
        return out
    seen = set()
    for p in prods:
        m = p[0]
        try:
            Chem.SanitizeMol(m)
            s = Chem.MolToSmiles(m)
        except Exception:
            continue
        if s in seen:
            continue
        seen.add(s); out.append(m)
        if len(out) >= max_products:
            break
    return out


def derive(smiles: str, delta: float, tol: float = 0.003, max_steps: int = 2, max_per_analog: int = 300,
           allow_isomerisation: bool = True) -> List[str]:
    """Return derivative SMILES of `smiles` whose mass differs by ~delta.
    With delta ~ 0 and allow_isomerisation, apply neutral (remove+add) pairs of the same delta."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    combos = matching_transforms(delta, tol, max_steps)
    if abs(delta) < tol and not allow_isomerisation:
        combos = []
    out = {}
    budget = 800          # total reaction applications per parent (keeps pathological cases bounded)
    for combo in combos:
        mols = [mol]
        for depth, name in enumerate(combo):
            nxt = []
            for m in mols[:24]:
                nxt.extend(apply_transform(m, name, max_products=32 if depth == 0 else 16))
                budget -= 1
                if budget <= 0:
                    break
            mols = nxt
            if not mols or budget <= 0:
                break
        for m in mols:
            s = Chem.MolToSmiles(m)
            if s != smiles and s not in out:
                out[s] = combo
        if len(out) >= max_per_analog or budget <= 0:
            break
    return list(out.keys())[:max_per_analog]
