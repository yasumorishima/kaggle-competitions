"""Identity-disjoint and scaffold-disjoint validation folds for the CASMI 2026 training file.

The competition scores the first 14 InChIKey characters after RDKit tautomer canonicalisation. A split is leak-free
only if every spectrum of one *scored identity* stays in one fold, including structures whose stored `inchikey14`
differs from the scorer key (tautomer aliases). This script builds those groups and two 5-fold assignments.

usage: python build_folds.py <train.parquet> <out_dir> [n_workers]
"""
import hashlib, json, os, sys, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SEED, K = 20260930, 5


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def _chem(smiles_block):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog('rdApp.*')
    te = rdMolStandardize.TautomerEnumerator()
    out = []
    for s in smiles_block:
        m = Chem.MolFromSmiles(s)
        if m is None:
            out.append(('', '', '', np.nan, 0)); continue
        try:
            t = te.Canonicalize(m)
            key = Chem.MolToInchiKey(t)[:14]
        except Exception:
            t, key = m, ''
        try:
            sc = MurckoScaffold.GetScaffoldForMol(t)
            scaf = Chem.MolToSmiles(sc, isomericSmiles=False) if sc.GetNumAtoms() else ''
        except Exception:
            scaf = ''
        out.append((key, scaf, rdMolDescriptors.CalcMolFormula(m), Descriptors.ExactMolWt(m), m.GetNumHeavyAtoms()))
    return out


class UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]; x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def groups_from(*label_arrays):
    """Connected components of rows linked by sharing any label (empty labels link nothing)."""
    n = len(label_arrays[0]); uf = UF(n)
    for lab in label_arrays:
        first = {}
        for i, v in enumerate(lab):
            if v == '' or v is None:
                continue
            j = first.setdefault(v, i)
            if j != i:
                uf.union(i, j)
    root = np.array([uf.find(i) for i in range(n)])
    _, gid = np.unique(root, return_inverse=True)
    return gid


def assign(group_id, weight, strata, rng):
    """Greedy balanced assignment of whole groups to K folds inside each stratum."""
    g = pd.DataFrame({'g': group_id, 'w': weight, 's': strata}).groupby('g').agg(w=('w', 'sum'), s=('s', 'max'))
    fold = np.zeros(len(g), np.int8)
    for _, idx in g.groupby('s').indices.items():
        idx = idx[rng.permutation(len(idx))]
        idx = idx[np.argsort(-g.w.values[idx], kind='stable')]
        load = np.zeros(K)
        for i in idx:
            f = int(np.argmin(load)); fold[i] = f; load[f] += g.w.values[i]
    return pd.Series(fold, index=g.index).reindex(group_id).values


def main():
    train, out = sys.argv[1], sys.argv[2]
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    os.makedirs(out, exist_ok=True)
    h = hashlib.sha256()
    with open(train, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 24), b''):
            h.update(blk)
    cols = ['ingest_lib', 'normalized_smiles', 'inchikey14', 'ionization_mode', 'instrument_type']
    df = pq.read_table(train, columns=cols).to_pandas()
    log('train rows', len(df), 'sha256', h.hexdigest())
    df['tims'] = df.ingest_lib.fillna('').str.startswith('enveda')
    st = (df.groupby(['normalized_smiles', 'inchikey14'], sort=False)
            .agg(n_spectra=('ingest_lib', 'size'), n_enveda_spectra=('tims', 'sum'),
                 n_pos=('ionization_mode', lambda x: int((x == 'positive').sum())),
                 libs=('ingest_lib', lambda x: ';'.join(sorted(set(x.dropna())))))
            .reset_index())
    log('distinct (smiles, inchikey14) rows', len(st))
    import multiprocessing as mp
    smi = st.normalized_smiles.tolist()
    blocks = [smi[i:i + 500] for i in range(0, len(smi), 500)]
    with mp.get_context('fork').Pool(nw) as pool:
        res = [r for blk in pool.imap(_chem, blocks, chunksize=2) for r in blk]
    st['scorer_key'] = [r[0] for r in res]; st['scaffold'] = [r[1] for r in res]
    st['formula'] = [r[2] for r in res]; st['mono_mass'] = [r[3] for r in res]; st['n_heavy'] = [r[4] for r in res]
    st['scorer_key'] = np.where(st.scorer_key == '', st.inchikey14, st.scorer_key)
    st['alias'] = st.scorer_key != st.inchikey14
    st['identity_group'] = groups_from(st.scorer_key.values, st.inchikey14.values)
    # acyclic molecules have no scaffold: they stay in their own identity group
    st['scaffold_group'] = groups_from(st.scorer_key.values, st.inchikey14.values, st.scaffold.values)
    rng = np.random.default_rng(SEED)
    strata = (st.groupby('identity_group').n_enveda_spectra.transform('sum') > 0).astype(int).values
    st['fold_identity'] = assign(st.identity_group.values, np.ones(len(st)), strata, rng)
    strata_s = (st.groupby('scaffold_group').n_enveda_spectra.transform('sum') > 0).astype(int).values
    st['fold_scaffold'] = assign(st.scaffold_group.values, np.ones(len(st)), strata_s, rng)

    # ---- checks: no identity or stored key crosses folds
    for col, fold in (('scorer_key', 'fold_identity'), ('inchikey14', 'fold_identity'), ('scorer_key', 'fold_scaffold'),
                      ('inchikey14', 'fold_scaffold'), ('scaffold', 'fold_scaffold')):
        x = st[st[col] != ''] if col == 'scaffold' else st
        assert x.groupby(col)[fold].nunique().max() == 1, (col, fold)
    st = st.rename(columns={'normalized_smiles': 'smiles'})
    st = st[['inchikey14', 'scorer_key', 'alias', 'smiles', 'formula', 'mono_mass', 'n_heavy', 'scaffold', 'identity_group',
             'scaffold_group', 'fold_identity', 'fold_scaffold', 'n_spectra', 'n_enveda_spectra', 'n_pos', 'libs']]
    st.to_parquet(os.path.join(out, 'structures.parquet'), index=False)
    key = st.drop_duplicates('inchikey14').set_index('inchikey14')
    sp = pd.DataFrame({'row': np.arange(len(df), dtype=np.int32), 'inchikey14': df.inchikey14.values,
                       'identity_group': key.identity_group.reindex(df.inchikey14).values.astype(np.int32),
                       'fold_identity': key.fold_identity.reindex(df.inchikey14).values.astype(np.int8),
                       'fold_scaffold': key.fold_scaffold.reindex(df.inchikey14).values.astype(np.int8)})
    sp.to_parquet(os.path.join(out, 'spectra_folds.parquet'), index=False)
    summ = dict(train_sha256=h.hexdigest(), train_rows=int(len(df)), structures=int(len(st)),
                scorer_identities=int(st.scorer_key.nunique()), stored_keys=int(st.inchikey14.nunique()),
                identity_groups=int(st.identity_group.nunique()), scaffold_groups=int(st.scaffold_group.nunique()),
                alias_rows=int(st.alias.sum()), seed=SEED, folds=K,
                rdkit=__import__('rdkit').__version__,
                fold_identity={str(f): dict(structures=int((st.fold_identity == f).sum()), spectra=int((sp.fold_identity == f).sum()),
                                            enveda_structures=int(((st.fold_identity == f) & (st.n_enveda_spectra > 0)).sum()))
                               for f in range(K)},
                fold_scaffold={str(f): dict(structures=int((st.fold_scaffold == f).sum()), spectra=int((sp.fold_scaffold == f).sum()),
                                            enveda_structures=int(((st.fold_scaffold == f) & (st.n_enveda_spectra > 0)).sum()))
                               for f in range(K)},
                largest_scaffold_group=int(st.scaffold_group.value_counts().iloc[0]))
    json.dump(summ, open(os.path.join(out, 'SUMMARY.json'), 'w'), indent=1)
    log(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
