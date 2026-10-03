#!/bin/sh
# Build pubchem_pop.parquet: PubChem literature popularity per InChIKey first block (14 chars).
# Source: PubChem FTP Compound/Extras (public): CID-PMID (cid, pmid, type) and CID-InChI-Key (cid, inchi, key).
# Columns: inchikey14, n_pmid (PubMed links of its best-linked CID), n_cid (CIDs with links sharing the block).
set -e
cd "$(dirname "$0")"
sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc /opt/hostedtoolcache/CodeQL || true
df -h . | tail -1
B=https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras
curl -sSfL --retry 5 -o /tmp/pmid.gz $B/CID-PMID.gz
curl -sSfL --retry 5 -o /tmp/ikey.gz $B/CID-InChI-Key.gz
ls -la /tmp/pmid.gz /tmp/ikey.gz
pip install -q pyarrow pandas
python3 - <<'PY'
import gzip, collections, time
import pandas as pd
t = time.time()
cnt = collections.Counter()           # cid -> PubMed links (lines in CID-PMID)
with gzip.open("/tmp/pmid.gz", "rt") as f:
    for line in f:
        cnt[int(line[:line.index("\t")])] += 1
print("cids with pmids", len(cnt), round(time.time() - t), flush=True)
best, ncid = {}, collections.Counter()  # block -> max links over its CIDs (the parent compound dominates)
with gzip.open("/tmp/ikey.gz", "rt") as f:
    for line in f:
        cid = int(line[:line.index("\t")])
        c = cnt.get(cid)
        if c:
            k = line.rstrip("\n").rsplit("\t", 1)[-1][:14]
            if c > best.get(k, 0):
                best[k] = c
            ncid[k] += 1
print("blocks", len(best), round(time.time() - t), flush=True)
df = pd.DataFrame({"inchikey14": list(best), "n_pmid": list(best.values()), "n_cid": [ncid[k] for k in best]})
df.to_parquet("pubchem_pop.parquet", index=False)
print(df.describe(), flush=True)
PY
rm -f /tmp/pmid.gz /tmp/ikey.gz prepare.sh.bak
ls -la
