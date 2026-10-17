"""Molecular graph tokens for the cross-encoder: atom features + shortest-path-distance matrix."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

ELEM = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5, 'Cl': 6, 'Br': 7, 'I': 8}
N_ELEM = 10
MAX_ATOMS = 96
N_ATOM_FEAT = 8      # elem, aromatic, in_ring, ring_size, n_h, degree, charge, hybrid
MAX_SPD = 9          # distances clipped to 0..8; 9 = unreachable / pad
N_BOND = 5           # 0 none, 1 single, 2 double, 3 triple, 4 aromatic


def mol_tokens(smiles: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Returns (atom_feats (n,8) uint8, spd (n,n) uint8, bond (n,n) uint8) or None."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    n = m.GetNumAtoms()
    if n == 0 or n > MAX_ATOMS:
        return None
    ri = m.GetRingInfo()
    hyb = {Chem.HybridizationType.SP: 1, Chem.HybridizationType.SP2: 2, Chem.HybridizationType.SP3: 3}
    A = np.zeros((n, N_ATOM_FEAT), np.uint8)
    for a in m.GetAtoms():
        i = a.GetIdx()
        rs = 0
        for sz in (3, 4, 5, 6, 7, 8):
            if ri.IsAtomInRingOfSize(i, sz):
                rs = sz - 2; break
        A[i] = (ELEM.get(a.GetSymbol(), 9), int(a.GetIsAromatic()), int(a.IsInRing()), rs,
                min(a.GetTotalNumHs(), 4), min(a.GetDegree(), 5), int(np.clip(a.GetFormalCharge(), -1, 1)) + 1,
                hyb.get(a.GetHybridization(), 0))
    B = np.zeros((n, n), np.uint8)
    bt = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        B[i, j] = B[j, i] = bt.get(b.GetBondType(), 1)
    D = np.asarray(Chem.GetDistanceMatrix(m))
    D = np.where(np.isfinite(D) & (D < 1e6), D, MAX_SPD).astype(np.float64)
    S = np.clip(D, 0, MAX_SPD).astype(np.uint8)
    return A, S, B


class GraphStore:
    """CSR storage of tokens for many molecules (built by scripts/05c)."""

    def __init__(self, prefix: str):
        self.off = np.load(f'{prefix}_off.npy')          # atom offsets (n_mol+1)
        self.atoms = np.load(f'{prefix}_atoms.npy')      # (total_atoms, 8)
        self.moff = np.load(f'{prefix}_moff.npy')        # matrix offsets (n_mol+1) into flat spd/bond
        self.spd = np.load(f'{prefix}_spd.npy')          # flat n*n per molecule
        self.bond = np.load(f'{prefix}_bond.npy')

    def get(self, i: int):
        a, b = self.off[i], self.off[i + 1]
        n = b - a
        if n == 0:
            return None
        ma, mb = self.moff[i], self.moff[i + 1]
        return self.atoms[a:b], self.spd[ma:mb].reshape(n, n), self.bond[ma:mb].reshape(n, n)

    def n_atoms(self, i: int) -> int:
        return int(self.off[i + 1] - self.off[i])
