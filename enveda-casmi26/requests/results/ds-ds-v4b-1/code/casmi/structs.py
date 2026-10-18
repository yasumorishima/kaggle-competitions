"""Parallel structure processing: score keys, formulas, masses, raw fingerprints."""
from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from typing import List, Optional

import numpy as np
import pandas as pd

from . import chem


def _work(args):
    i, smi, do_std = args
    if do_std:
        smi = chem.standardize_smiles(smi)
        if smi is None:
            return i, None
    p = chem.mol_props(smi)
    if p is None:
        return i, None
    fp = chem.raw_fingerprint(smi)
    if fp is None:
        return i, None
    sk = chem.score_key(smi)
    rk = chem.raw_key(smi)
    return i, dict(smiles=smi, formula=p[0], mass=p[1], n_heavy=p[2], key=sk, rawkey=rk,
                   fp=np.packbits(fp))


def process_smiles(smiles: List[str], workers: int = 20, standardize: bool = False,
                   chunksize: int = 64, log_every: int = 20000, desc: str = "structs"):
    """Returns (DataFrame with smiles/formula/mass/n_heavy/key/rawkey, packed fp matrix uint8).
    Rows that failed are dropped; DataFrame keeps column 'orig_index'."""
    n = len(smiles)
    t0 = time.time()
    recs = [None] * n
    tasks = [(i, s, standardize) for i, s in enumerate(smiles)]
    with Pool(workers) as pool:
        for k, (i, r) in enumerate(pool.imap_unordered(_work, tasks, chunksize=chunksize)):
            recs[i] = r
            if (k + 1) % log_every == 0:
                el = time.time() - t0
                print(f"  [{desc}] {k+1}/{n} {el:.0f}s ({(k+1)/el:.0f}/s)", flush=True)
    ok = [i for i in range(n) if recs[i] is not None]
    df = pd.DataFrame([{k: v for k, v in recs[i].items() if k != 'fp'} for i in ok])
    df['orig_index'] = ok
    fp = np.stack([recs[i]['fp'] for i in ok]) if ok else np.zeros((0, chem.FP_RAW_BITS // 8), np.uint8)
    print(f"  [{desc}] done {len(ok)}/{n} ok in {time.time()-t0:.0f}s", flush=True)
    return df, fp


def select_bits(fp_packed: np.ndarray, lo: float = 0.005, hi: float = 0.995,
                sample: int = 200000, seed: int = 0) -> np.ndarray:
    """Indices of bits whose frequency lies in [lo, hi] over a sample of structures."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(fp_packed), size=min(sample, len(fp_packed)), replace=False)
    freq = np.zeros(chem.FP_RAW_BITS, np.float64)
    for s in range(0, len(idx), 20000):
        blk = np.unpackbits(fp_packed[idx[s:s + 20000]], axis=1)[:, :chem.FP_RAW_BITS]
        freq += blk.sum(0)
    freq /= len(idx)
    keep = np.where((freq >= lo) & (freq <= hi))[0]
    return keep.astype(np.int32)


def unpack_bits(fp_packed: np.ndarray, bits: Optional[np.ndarray] = None) -> np.ndarray:
    u = np.unpackbits(fp_packed, axis=1)[:, :chem.FP_RAW_BITS]
    return u if bits is None else u[:, bits]
