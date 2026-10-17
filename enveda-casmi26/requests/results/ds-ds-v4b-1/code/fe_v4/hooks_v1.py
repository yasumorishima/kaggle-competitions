"""Recording hooks for the v1 engine (kaggle/ds_models/code/casmi = the engine of the v3.6 / 0.358 model): verbatim copies
of v1 Engine.generate and v1 derive.derive that additionally record
  E._last_an   analog list of the last run            E._last_gen  accepted generated products (+ raw smiles, combo)
  E._derlog    derived products NOT accepted: (parent_sid, parent_sim, combo, 'pool'|'gen', 'key', score key)
The v1 candidate list and its 94 features are unchanged (checked bit-exact against an unhooked engine in parity_v1.py).
v1 semantics: products are de-duplicated by score key only (no smiles set) and the key check precedes the mass check.
"""
import numpy as np
from casmi import chem, derive, frag as fragmod

_COMBO = {}


def derive_rec(smiles, delta, tol=0.003, max_steps=2, max_per_analog=300, allow_isomerisation=True):
    """== v1 derive.derive + (parent smiles, product smiles) -> first matching combo."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    combos = derive.matching_transforms(delta, tol, max_steps)
    if abs(delta) < tol and not allow_isomerisation:
        combos = []
    out = {}
    budget = 800
    for combo in combos:
        mols = [mol]
        for depth, name in enumerate(combo):
            nxt = []
            for m in mols[:24]:
                nxt.extend(derive.apply_transform(m, name, max_products=32 if depth == 0 else 16))
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
    res = list(out.keys())[:max_per_analog]
    for s in res:
        _COMBO[(smiles, s)] = out[s]
    return res


def install(E):
    E._orig_derive = derive.derive
    orig_analogs = E.analogs

    def analogs_cap(*a, **k):
        r = orig_analogs(*a, **k); E._last_an = r; return r
    E.analogs = analogs_cap

    def generate_rec(analogs, target, exclude_keys):
        """== v1 Engine.generate (verbatim) + recording."""
        cfg = E.cfg; L = E.L
        out = []; seen = set(exclude_keys)
        win_keys = set(exclude_keys)
        parents = [a for a in analogs if a[1] >= cfg.gen_min_sim][:cfg.gen_n_analog]
        tol = max(cfg.gen_tol_da, target * cfg.ppm_win * 1e-6)
        for rank, (sid, sim, shift, _) in enumerate(parents):
            psmi = L.struct_smiles[sid]
            pfp = np.unpackbits(E.train_fp[sid])[:E.pool.nbits].astype(np.float32)
            delta = target - float(L.struct_mass[sid])
            try:
                prods = derive_rec(psmi, delta, tol=tol, max_steps=2, max_per_analog=cfg.gen_max_per_parent)
            except Exception:
                prods = []
            for s in prods:
                std = chem.standardize_smiles(s)
                if std is None:
                    continue
                k = chem.score_key(std)
                if k is None:
                    continue
                if k in seen:
                    E._derlog.append((int(sid), float(sim), _COMBO.get((psmi, s)), 'pool' if k in win_keys else 'gen', 'key', k))
                    continue
                p = chem.mol_props(std)
                if p is None or abs(p[1] - target) > tol:
                    continue
                fp = chem.raw_fingerprint(std)
                if fp is None:
                    continue
                fp = fp[E.pool.bits].astype(np.float32)
                inter = float((fp * pfp).sum()); tan = inter / (fp.sum() + pfp.sum() - inter + 1e-9)
                seen.add(k)
                out.append(dict(smiles=std, key=k, formula=p[0], mass=p[1], n_heavy=p[2], fp=fp,
                                frags=fragmod.fragments_for_smiles(std), sim=sim, steps=1, tan=tan, parent=sid,
                                raw=s, combo=_COMBO.get((psmi, s))))
                if len(out) >= cfg.gen_max_total:
                    E._last_gen = out
                    return out
        E._last_gen = out
        return out
    E.generate = generate_rec


def reset(E):
    _COMBO.clear()
    E._last_an = None; E._last_gen = []; E._derlog = []
