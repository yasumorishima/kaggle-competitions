"""Shared pieces for the local CASMI26 harness: paths, masses, adducts, the metric.

The data lives outside the checkout (3 GB); point CASMI_DATA at the folder that
holds train.parquet / test.parquet.
"""
import os
import re

import numpy as np

DATA = os.environ.get("CASMI_DATA", os.path.expanduser("~/casmi_data"))

PROTON = 1.007276467
ELECTRON = 0.000548580
MONO = {"C": 12.0, "H": 1.00782503207, "N": 14.0030740048, "O": 15.99491461956,
        "P": 30.97376163, "S": 31.97207100, "F": 18.99840322, "Cl": 34.96885268,
        "Br": 78.9183371, "I": 126.904473, "Si": 27.9769265325, "B": 11.0093054,
        "Se": 79.9165213, "Na": 22.9897692809, "K": 38.96370668}

# Neutral mass M from precursor m/z: M = (mz - shift) / mult, for the ten test adducts
# plus the common extra ones in train.
ADDUCTS = {
    "[M+H]+": (1, PROTON), "[M+NH4]+": (1, 18.033823), "[M-H2O+H]+": (1, PROTON - 18.010565),
    "[M-2H2O+H]+": (1, PROTON - 2 * 18.010565), "[M+Na]+": (1, 22.989218), "[M+K]+": (1, 38.963158),
    "[M-H]-": (1, -PROTON), "[M-H2O-H]-": (1, -PROTON - 18.010565),
    "[M+CH2O2-H]-": (1, 44.998201), "[M+Cl]-": (1, 34.969402),
    "[2M+H]+": (2, PROTON), "[2M+Na]+": (2, 22.989218), "[2M-H]-": (2, -PROTON),
    "[M]+": (1, -ELECTRON),
}


def neutral_mass(mz, adduct):
    a = ADDUCTS.get(adduct)
    if a is None:
        return np.nan
    mult, shift = a
    return (mz - shift) / mult


_FORM = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_mass(f):
    if not isinstance(f, str) or any(c in f for c in "+-"):
        return np.nan
    m = 0.0
    for el, n in _FORM.findall(f):
        if el not in MONO:
            return np.nan
        m += MONO[el] * (int(n) if n else 1)
    return m


def ik14(smiles):
    """The metric's key: RDKit tautomer-canonical InChIKey, first block."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        m = rdMolStandardize.TautomerEnumerator().Canonicalize(m)
    except Exception:
        pass
    k = Chem.MolToInchiKey(m)
    return k.split("-")[0] if k else None


def mrr25(ranked_keys, truth_key):
    """ranked_keys: list of InChIKey14 in submitted order; the first match counts."""
    for i, k in enumerate(ranked_keys[:25]):
        if k == truth_key:
            return 1.0 / (i + 1)
    return 0.0
