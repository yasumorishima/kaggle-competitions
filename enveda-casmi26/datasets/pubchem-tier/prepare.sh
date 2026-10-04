#!/bin/sh
# Build pubchem_tier.parquet: a PubChem candidate tier for structures the spectral library and COCONUT miss
# (hidden-test class 2). Kept: compounds with >= MIN_PMID PubMed links (literature-known), neutral, single
# fragment, CHNOPS + halogens, monoisotopic mass 100-1500. One row per InChIKey first block (best-linked CID).
# Source: PubChem FTP Compound/Extras (public): CID-PMID, CID-SMILES, CID-InChI-Key.
# Columns: inchikey14, smiles, fM (RDKit exact mass), n_pmid.
set -e
cd "$(dirname "$0")"
MIN_PMID=3
sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc /opt/hostedtoolcache/CodeQL || true
df -h . | tail -1
B=https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras
for f in CID-PMID CID-SMILES CID-InChI-Key; do curl -sSfL --retry 5 -o /tmp/$f.gz $B/$f.gz; done
ls -la /tmp/*.gz
pip install -q rdkit pyarrow pandas
MIN_PMID=$MIN_PMID python3 - <<'PY'
import gzip, collections, os, time
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt
RDLogger.DisableLog("rdApp.*")
t = time.time()
cnt = collections.Counter()
with gzip.open("/tmp/CID-PMID.gz", "rt") as f:
    for line in f:
        cnt[int(line[:line.index("\t")])] += 1
keep = {c: n for c, n in cnt.items() if n >= int(os.environ["MIN_PMID"])}
del cnt
print("cids kept", len(keep), round(time.time() - t), flush=True)
block = {}
with gzip.open("/tmp/CID-InChI-Key.gz", "rt") as f:
    for line in f:
        cid = int(line[:line.index("\t")])
        if cid in keep:
            block[cid] = line.rstrip("\n").rsplit("\t", 1)[-1][:14]
best = {}   # block -> (n_pmid, cid)
for cid, k in block.items():
    if keep[cid] > best.get(k, (0, 0))[0]:
        best[k] = (keep[cid], cid)
want = {cid: k for k, (n, cid) in best.items()}
print("blocks", len(best), round(time.time() - t), flush=True)
OK = set("C H N O P S F Cl Br I".split())
rows = []
with gzip.open("/tmp/CID-SMILES.gz", "rt") as f:
    for line in f:
        cid_s, smi = line.rstrip("\n").split("\t", 1)
        k = want.get(int(cid_s))
        if k is None or "." in smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None or Chem.GetFormalCharge(m) != 0 or any(a.GetSymbol() not in OK for a in m.GetAtoms()) or not any(a.GetSymbol() == "C" for a in m.GetAtoms()):
            continue
        w = ExactMolWt(m)
        if 100 <= w <= 1500:
            rows.append((k, smi, w, best[k][0]))
df = pd.DataFrame(rows, columns=["inchikey14", "smiles", "fM", "n_pmid"]).sort_values("fM", kind="stable")
df.to_parquet("pubchem_tier.parquet", index=False)
print(len(df), round(time.time() - t), flush=True)
print(df.describe(), flush=True)
PY
rm -f /tmp/*.gz
ls -la
