#!/bin/sh
# Measure which part of PubChem holds the enveda-180 structures (same instrument and source as the hidden test),
# to choose a candidate tier with a high in-window rate. Writes coverage.txt (numbers only; no competition data kept).
# Sources: competition train.parquet (two columns; Kaggle API, read on the runner only) and PubChem FTP Compound/Extras.
set -e
cd "$(dirname "$0")"
sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc /opt/hostedtoolcache/CodeQL || true
mkdir -p /tmp/comp
curl -sSfL --retry 5 -o /tmp/comp/train.parquet "https://www.kaggle.com/api/v1/competitions/data/download/enveda-CASMI26-molecule-id-mass-spectra/train.parquet"
ls -la /tmp/comp
B=https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras
for f in CID-InChI-Key CID-PMID CID-Patent CID-SID; do curl -sSfL --retry 5 -o /tmp/$f.gz $B/$f.gz; done
ls -la /tmp/*.gz
pip install -q pyarrow pandas
python3 - <<'PY' | tee coverage.txt
import gzip, collections, time
import pandas as pd
t = time.time()
m = pd.read_parquet("/tmp/comp/train.parquet", columns=["inchikey14", "ingest_lib"])
E = set(m.inchikey14[m.ingest_lib == "enveda-180"])
O = set(m.inchikey14[m.ingest_lib != "enveda-180"])
print("enveda-180 structures", len(E), "other-library structures", len(O), flush=True)
cid_e = {}
nrow = 0
with gzip.open("/tmp/CID-InChI-Key.gz", "rt") as f:
    for line in f:
        a = line.rstrip("\n").split("\t")
        k = a[-1][:14]
        nrow += 1
        if k in E:
            cid_e.setdefault(k, []).append(int(a[0]))
print("pubchem compounds", nrow, "enveda-180 in pubchem", round(len(cid_e) / len(E), 4), round(time.time() - t), flush=True)
cids = {c for v in cid_e.values() for c in v}
def linked(fn):
    s = set()
    with gzip.open(fn, "rt") as f:
        for line in f:
            c = int(line[:line.index("\t")])
            if c in cids:
                s.add(c)
    return s
for name in ["CID-PMID", "CID-Patent"]:
    s = linked(f"/tmp/{name}.gz")
    print(name, "enveda-180 with a link", round(sum(any(c in s for c in v) for v in cid_e.values()) / len(E), 4), flush=True)
nsid = collections.Counter()
with gzip.open("/tmp/CID-SID.gz", "rt") as f:
    for line in f:
        c = int(line[:line.index("\t")])
        if c in cids:
            nsid[c] += 1
best = [max(nsid.get(c, 0) for c in v) for v in cid_e.values()]
print("SIDs per enveda-180 compound: quantiles", pd.Series(best).quantile([.1, .25, .5, .75, .9]).to_dict(), flush=True)
mincid = pd.Series([min(v) for v in cid_e.values()])
print("smallest CID quantiles", mincid.quantile([.1, .25, .5, .75, .9]).to_dict(), flush=True)
PY
rm -rf /tmp/*.gz /tmp/comp
ls -la
