"""ours_lfrag (yasunorim): learned peak credits for the base's MetFrag-lite fragment scorer (frag_rescore).

frag_rescore credits a peak 1.0 when its cheapest explanation is the precursor or a 1-bond cleavage and 0.6 for a
2-bond cleavage, whatever hydrogen shift dh (-2..+3) it needs, on linear intensity. Here each peak falls in one bucket
(cleavage level 0/1 vs 2) x (dh of its cheapest explanation: fewest cleavages, then smallest |dh|), on linear and on
sqrt intensity, and the candidate score is the learned linear combination of the bucket shares (mean over spectra).
Weights: listwise softmax over 20 same-formula train isomers, fitted on 4,432 train molecules (3,000 enveda-180 +
1,432 gnps / mona / massbank / riken / msdial / pluskal), L2 0.01 towards the base credits. Held out: enveda-180
500 molecules MRR 0.723 -> 0.748, enveda-np-examples 171 molecules 0.567 -> 0.584 (truth among 20 isomers).
"""
import numpy as np

W = np.array([0.78009, 1.313512, 1.016127, 1.228675, 0.670753, 0.58122, 0.632298, 0.4081, 0.905728, 0.542433, 0.588673, 0.17127, -0.22641, 1.224142, 0.419948, 0.73033, -0.094456, -0.409913, 0.293189, 0.068881, 0.430604, 0.013667, -0.279784, -0.495254, 0.0])


def install(fr):
    """Replace fr.score_molecule (fr = the imported frag_rescore module) by the learned scorer; same signature."""
    C = fr.resolve_cfg(None)
    HS = C['h_shifts']
    NH = len(HS)
    assert len(W) == 4 * NH + 1, (len(W), NH)

    def spec_feats(lv, s):
        mz = np.asarray(s.get('mz', ()), np.float64).ravel()
        it = np.asarray(s.get('it', ()), np.float64).ravel()
        n = min(len(mz), len(it))
        mz, it = mz[:n], it[:n]
        ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)
        mz, it = mz[ok], it[ok]
        if len(mz) == 0:
            return None
        car = fr._carriers(s.get('adduct'), s.get('mode'))
        shifts = np.array([c + dh * fr.H for c in car for dh in HS])
        q = mz[None, :] - shifts[:, None]
        best_l = np.full(len(mz), 9)
        best_d = np.zeros(len(mz), int)
        best_a = np.full(len(mz), 99)
        for L, arr in enumerate(lv[:3]):
            if len(arr) == 0:
                continue
            a = np.searchsorted(arr, q - C['tol'], 'left')
            b = np.searchsorted(arr, q + C['tol'], 'right')
            hit = b > a
            lvl = 0 if L <= 1 else 1
            for si in range(len(shifts)):
                di = si % NH
                ad = abs(HS[di])
                m = hit[si] & ((lvl < best_l) | ((lvl == best_l) & (ad < best_a)))
                best_l = np.where(m, lvl, best_l)
                best_d = np.where(m, di, best_d)
                best_a = np.where(m, ad, best_a)
        f = np.zeros(4 * NH + 1)
        hk = best_l < 9
        for w_i, wt in enumerate((it / it.sum(), np.sqrt(it) / np.sqrt(it).sum())):
            np.add.at(f, w_i * 2 * NH + best_l[hk] * NH + best_d[hk], wt[hk])
        f[-1] = 1.0
        return f

    def score_molecule(spectra, cand_smiles, cfg=None, cache=None):
        c = fr.resolve_cfg(cfg)
        cache = {} if cache is None else cache
        out = []
        for smi in cand_smiles:
            lv = fr._levels_cached(smi, c, cache)
            if lv is None:
                out.append(None)
                continue
            fs = [x for x in (spec_feats(lv, s) for s in (spectra or [])) if x is not None]
            out.append(float(np.mean(fs, 0) @ W) if fs else 0.0)
        return out

    fr.score_molecule_base = fr.score_molecule
    fr.score_molecule = score_molecule
    return fr
