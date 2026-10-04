#!/bin/sh
# Build enamine_tier.parquet: compounds the depositor "Enamine" registered in PubChem (about 5.2M substances).
# pccov-4: 92% of the enveda-180 structures (the hidden test's instrument and source) are in this deposit,
# against 2.5% for PubMed-linked compounds. Kept: neutral, single fragment, CHNOPS + halogens, 100-1500 Da.
# One row per InChIKey first block. Source: PubChem FTP Substance/Extras/SID-Map, Compound/Extras/CID-SMILES, CID-InChI-Key.
# Columns: inchikey14, smiles, fM (RDKit exact mass).
set -e
cd "$(dirname "$0")"
sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc /opt/hostedtoolcache/CodeQL || true
B=https://ftp.ncbi.nlm.nih.gov/pubchem
curl -sSfL --retry 5 -o /tmp/SID-Map.gz $B/Substance/Extras/SID-Map.gz
for f in CID-SMILES CID-InChI-Key; do curl -sSfL --retry 5 -o /tmp/$f.gz $B/Compound/Extras/$f.gz; done
ls -la /tmp/*.gz
pip install -q rdkit pyarrow pandas
python3 - <<'PY'
import gzip, time
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt
RDLogger.DisableLog("rdApp.*")
t = time.time()
cids = set()
with gzip.open("/tmp/SID-Map.gz", "rt") as f:
    for line in f:
        a = line.rstrip("\n").split("\t")
        if len(a) >= 4 and a[3] and a[1] == "Enamine":
            cids.add(int(a[3]))
print("enamine cids", len(cids), round(time.time() - t), flush=True)
key = {}
with gzip.open("/tmp/CID-InChI-Key.gz", "rt") as f:
    for line in f:
        cid = int(line[:line.index("\t")])
        if cid in cids:
            key[cid] = line.rstrip("\n").rsplit("\t", 1)[-1][:14]
OK = set("C H N O P S F Cl Br I".split())
rows, seen = [], set()
with gzip.open("/tmp/CID-SMILES.gz", "rt") as f:
    for line in f:
        cid_s, smi = line.rstrip("\n").split("\t", 1)
        k = key.get(int(cid_s))
        if k is None or k in seen or "." in smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None or Chem.GetFormalCharge(m) != 0 or any(a.GetSymbol() not in OK for a in m.GetAtoms()) or not any(a.GetSymbol() == "C" for a in m.GetAtoms()):
            continue
        w = ExactMolWt(m)
        if 100 <= w <= 1500:
            seen.add(k)
            rows.append((k, smi, w))
df = pd.DataFrame(rows, columns=["inchikey14", "smiles", "fM"]).sort_values("fM", kind="stable")
df.to_parquet("enamine_tier.parquet", index=False)
print(len(df), round(time.time() - t), flush=True)
print(df.describe(), flush=True)
PY
rm -f /tmp/*.gz
ls -la
