#!/bin/sh
# Build coco_meta.pkl + coco_mass.npy (same layout the b5 kernel reads) from the official COCONUT dump
# (coconut.naturalproducts.net, CC BY 4.0), instead of another user's Kaggle dataset.
# coco_meta.pkl = {"keys": InChIKey first block (14), "smiles", "nbits": 0}; coco_mass.npy = neutral monoisotopic mass,
# one row per InChIKey14, mass 100-1500, sorted by mass.
set -e
cd "$(dirname "$0")"
pip install -q rdkit pandas pyarrow
P=https://coconut.naturalproducts.net/download
curl -sSfL --retry 5 -A "Mozilla/5.0" -o /tmp/page.html "$P" || true
grep -oE 'https?://[^"'"'"' <>]+\.(zip|csv\.gz|csv)' /tmp/page.html | sort -u | tee /tmp/links.txt || true
# the full dump is coconut_csv-MM-YYYY.zip (not _lite, not the MORTAR fragment lists); newest first
URL=$(grep -E '/coconut_csv-[0-9]{2}-[0-9]{4}\.zip$' /tmp/links.txt | sort -r | head -1)
if [ -z "$URL" ]; then   # fallback: the COCONUT dumps deposited on Zenodo
  curl -sSfL --retry 5 "https://zenodo.org/api/records?q=COCONUT%20natural%20products%20csv&sort=mostrecent&size=10" -o /tmp/z.json
  URL=$(python3 -c "
import json
for r in json.load(open('/tmp/z.json'))['hits']['hits']:
    for f in r.get('files', []):
        k = f['key'].lower()
        if 'coconut' in k and ('csv' in k):
            print(f['links']['self']); raise SystemExit")
fi
echo "source: $URL"
curl -sSfL --retry 5 -o /tmp/coconut.dl "$URL"
ls -la /tmp/coconut.dl
mkdir -p /tmp/coco && cd /tmp/coco
case "$URL" in
  *.zip) unzip -o -q /tmp/coconut.dl ;;
  *.gz) gunzip -c /tmp/coconut.dl > coconut.csv ;;
  *) mv /tmp/coconut.dl coconut.csv ;;
esac
ls -la
cd - >/dev/null
python3 - <<'PY'
import glob, pickle, time
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt
RDLogger.DisableLog("rdApp.*")
t = time.time()
f = max(glob.glob("/tmp/coco/**/*.csv", recursive=True), key=lambda p: __import__("os").path.getsize(p))
df = pd.read_csv(f, low_memory=False)
assert len(df) > 100000, "not the full COCONUT dump"
print(f, df.shape, list(df.columns)[:40], flush=True)
col = next(c for c in ["canonical_smiles", "smiles", "SMILES", "isomeric_smiles"] if c in df.columns)
best = {}
for s in df[col].dropna().astype(str):
    m = Chem.MolFromSmiles(s)
    if m is None:
        continue
    k = Chem.MolToInchiKey(m)[:14]
    if len(k) == 14 and k not in best:
        best[k] = (Chem.MolToSmiles(m), ExactMolWt(m))
print("blocks", len(best), round(time.time() - t), flush=True)
keys = np.array(list(best), dtype=object)            # object dtype: fixed-width unicode made a 3 GB pickle
smi = np.array([v[0] for v in best.values()], dtype=object)
mass = np.array([v[1] for v in best.values()])
o = np.argsort(mass, kind="stable")
o = o[(mass[o] >= 100) & (mass[o] <= 1500)]
pickle.dump({"keys": keys[o], "smiles": smi[o], "nbits": 0}, open("coco_meta.pkl", "wb"))
np.save("coco_mass.npy", mass[o])
print(pd.Series(mass).describe(), flush=True)
PY
rm -rf /tmp/coco /tmp/coconut.dl
ls -la
