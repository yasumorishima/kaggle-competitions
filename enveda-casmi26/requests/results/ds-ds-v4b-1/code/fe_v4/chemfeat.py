"""Intrinsic per-structure chemistry records shared by fe_v4 families (computed from SMILES only, identical for train and
COCONUT entries -> no provenance). Pool / train-structure tables are precomputed by res_common.py; generated candidates
call struct_record(smiles) on the fly (same function).

Record fields (float32 vector, see REC_FIELDS):
  scaf_h / gscaf_h   Murcko scaffold / generic (all-C, single-bond) scaffold hash as two float32 halves (hi, lo 24 bits)
  n_ring, n_arom_ring, n_scaf_heavy, frac_scaf  scaffold size descriptors
  charge_pos, charge_neg, quat_n, basic_amine, amide_n, arom_n, cooh, so3h, po3h, phenol_oh, alc_oh, n_N, n_O, n_S, n_P,
  n_hal, logp, tpsa, hbd, hba, n_rotb, fsp3, glycoside, ester, methoxy, n_heavy
"""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

REC_FIELDS = ['scaf_hi', 'scaf_lo', 'gscaf_hi', 'gscaf_lo', 'n_ring', 'n_arom_ring', 'n_scaf_heavy', 'frac_scaf',
              'charge_pos', 'charge_neg', 'quat_n', 'basic_amine', 'amide_n', 'arom_n', 'cooh', 'so3h', 'po3h',
              'phenol_oh', 'alc_oh', 'n_N', 'n_O', 'n_S', 'n_P', 'n_hal', 'logp', 'tpsa', 'hbd', 'hba', 'n_rotb',
              'fsp3', 'glycoside', 'ester', 'methoxy', 'n_heavy']
RI = {f: i for i, f in enumerate(REC_FIELDS)}

_SMARTS = {
    'quat_n': '[NX4+;!$([N+][O-])]',
    'basic_amine': '[NX3;H2,H1,H0;!$(NC=[O,S,N]);!$(N-a);!$(N-[SX4]);!$(N#*);!$(N=*);!$([N]-[N,O])]',
    'amide_n': '[NX3][CX3]=[OX1]',
    'arom_n': '[n]',
    'cooh': '[CX3](=O)[OX2H1,OX1-]',
    'so3h': '[SX4](=O)(=O)[OX2H1,OX1-]',
    'po3h': '[PX4](=O)([OX2H1,OX1-])',
    'phenol_oh': '[OX2H1]c',
    'alc_oh': '[OX2H1][CX4]',
    'glycoside': '[OX2;R][CX4;R]([OX2,NX3,#6;!R])[CX4;R][CX4;R]([OX2H1,OX2])',
    'ester': '[#6][CX3](=O)[OX2][#6]',
    'methoxy': '[CH3][OX2][#6]',
}
_P = {}


def _rd():
    if not _P:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski
        from rdkit.Chem.Scaffolds import MurckoScaffold
        RDLogger.DisableLog('rdApp.*')
        _P.update(Chem=Chem, D=Descriptors, Crippen=Crippen, R=rdMolDescriptors, Lip=Lipinski, M=MurckoScaffold,
                  pat={k: Chem.MolFromSmarts(v) for k, v in _SMARTS.items()})
    return _P


def _h48(s: str):
    """48-bit hash of a string as two exact float32 halves (24 bits each)."""
    h = int.from_bytes(hashlib.blake2b(s.encode(), digest_size=6).digest(), 'little')
    return float(h >> 24), float(h & 0xFFFFFF)


def struct_record(smiles: str) -> np.ndarray:
    """Chemistry record of one structure (zeros + NaN-free defaults when RDKit fails)."""
    out = np.zeros(len(REC_FIELDS), np.float32)
    try:
        p = _rd(); Chem = p['Chem']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return out
        try:
            sc = p['M'].GetScaffoldForMol(mol)
            ss = Chem.MolToSmiles(sc) if sc is not None else ''
            gs = Chem.MolToSmiles(p['M'].MakeScaffoldGeneric(sc)) if sc is not None and sc.GetNumAtoms() else ''
            nsh = sc.GetNumHeavyAtoms() if sc is not None else 0
        except Exception:
            ss, gs, nsh = '', '', 0
        out[RI['scaf_hi']], out[RI['scaf_lo']] = _h48('S:' + ss)
        out[RI['gscaf_hi']], out[RI['gscaf_lo']] = _h48('G:' + gs)
        ri = mol.GetRingInfo()
        out[RI['n_ring']] = ri.NumRings()
        out[RI['n_arom_ring']] = p['R'].CalcNumAromaticRings(mol)
        nh = mol.GetNumHeavyAtoms()
        out[RI['n_heavy']] = nh; out[RI['n_scaf_heavy']] = nsh; out[RI['frac_scaf']] = nsh / max(nh, 1)
        chg = [a.GetFormalCharge() for a in mol.GetAtoms()]
        out[RI['charge_pos']] = sum(c for c in chg if c > 0); out[RI['charge_neg']] = -sum(c for c in chg if c < 0)
        for k, pat in p['pat'].items():
            out[RI[k]] = len(mol.GetSubstructMatches(pat))
        cnt = {}
        for a in mol.GetAtoms():
            cnt[a.GetSymbol()] = cnt.get(a.GetSymbol(), 0) + 1
        out[RI['n_N']] = cnt.get('N', 0); out[RI['n_O']] = cnt.get('O', 0); out[RI['n_S']] = cnt.get('S', 0)
        out[RI['n_P']] = cnt.get('P', 0)
        out[RI['n_hal']] = sum(cnt.get(e, 0) for e in ('F', 'Cl', 'Br', 'I'))
        out[RI['logp']] = p['Crippen'].MolLogP(mol); out[RI['tpsa']] = p['R'].CalcTPSA(mol)
        out[RI['hbd']] = p['R'].CalcNumHBD(mol); out[RI['hba']] = p['R'].CalcNumHBA(mol)
        out[RI['n_rotb']] = p['R'].CalcNumRotatableBonds(mol); out[RI['fsp3']] = p['R'].CalcFractionCSP3(mol)
    except Exception:
        return np.zeros(len(REC_FIELDS), np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def scaf_key(rec: np.ndarray, generic=False) -> int:
    """Exact integer scaffold hash from a record."""
    i = RI['gscaf_hi'] if generic else RI['scaf_hi']
    return (int(rec[..., i]) << 24) | int(rec[..., i + 1]) if np.ndim(rec) == 1 else \
        (rec[:, i].astype(np.int64) << 24) | rec[:, i + 1].astype(np.int64)


class Tables:
    """Precomputed records: pool (by pid) and train structures (by sid); struct_record() fallback for others."""

    def __init__(self, path: Optional[str] = None):
        import os
        path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res', 'common')
        self.pool = np.load(f'{path}/pool_rec.npy', mmap_mode='r')
        self.train = np.load(f'{path}/train_rec.npy', mmap_mode='r')
        self._gen = {}

    def cand(self, v) -> np.ndarray:
        """(n, len(REC_FIELDS)) records of a QueryView's candidates."""
        out = np.zeros((v.n, len(REC_FIELDS)), np.float32)
        pi = np.flatnonzero(v.pid >= 0)
        if len(pi):
            out[pi] = self.pool[v.pid[pi]]
        for i in np.flatnonzero(v.pid < 0):
            s = v.smiles[i]
            if s not in self._gen:
                if len(self._gen) > 200000:
                    self._gen.clear()
                self._gen[s] = struct_record(s)
            out[i] = self._gen[s]
        return out

    def analogs(self, v) -> np.ndarray:
        return np.asarray(self.train[v.an_sid]) if len(v.an_sid) else np.zeros((0, len(REC_FIELDS)), np.float32)
