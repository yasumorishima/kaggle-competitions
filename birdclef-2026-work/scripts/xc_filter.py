"""Filter XC recordings: Aves + Q A|B + non-ND license + per-species cap 500.
Output: xc_filtered.csv (one row per recording with metadata for DL planning)."""
import json
from pathlib import Path
import pandas as pd

CACHE = Path("C:/Users/fw_ya/Desktop/Claude_code/kaggle-competitions/birdclef-2026-work/xc_cache")
TAX = Path("C:/Users/fw_ya/Desktop/Claude_code/kaggle-competitions/birdclef-2026-embed/out/taxonomy.csv")
OUT = Path("C:/Users/fw_ya/Desktop/Claude_code/kaggle-competitions/birdclef-2026-work/xc_filtered.csv")

PER_SPECIES_CAP = 500
ALLOWED_Q = {"A", "B"}
DENY_LICENSE_SUBSTR = ["by-nc-nd"]  # ND = No Derivatives — exclude

tax = pd.read_csv(TAX).set_index("primary_label")

rows = []
for jf in sorted(CACHE.glob("*.json")):
    label = jf.stem
    if label not in tax.index:
        continue
    if tax.loc[label, "class_name"] != "Aves":
        continue
    data = json.loads(jf.read_text(encoding="utf-8"))
    recs = data.get("recordings", [])

    cand = []
    for r in recs:
        if r.get("q") not in ALLOWED_Q:
            continue
        lic = r.get("lic", "")
        if any(s in lic for s in DENY_LICENSE_SUBSTR):
            continue
        length = r.get("length", "0:00")
        try:
            parts = [int(x) for x in length.split(":")]
            sec = parts[0] * 60 + parts[1] if len(parts) == 2 else (parts[0]*3600 + parts[1]*60 + parts[2] if len(parts) == 3 else 0)
        except Exception:
            sec = 0
        cand.append({
            "label": label,
            "scientific": tax.loc[label, "scientific_name"],
            "xc_id": r.get("id"),
            "file_url": r.get("file"),
            "file_name": r.get("file-name"),
            "q": r.get("q"),
            "length_sec": sec,
            "country": r.get("cnt"),
            "lat": r.get("lat"),
            "lon": r.get("lon"),
            "lic": r.get("lic"),
            "type": r.get("type"),
            "also": ";".join(r.get("also", []) or []),
            "smp": r.get("smp"),
        })

    cand.sort(key=lambda x: (x["q"], -x["length_sec"]))
    if len(cand) > PER_SPECIES_CAP:
        cand = cand[:PER_SPECIES_CAP]
    rows.extend(cand)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

print(f"Total recordings selected: {len(df):,}")
print(f"Total species: {df['label'].nunique()}")
print(f"Total duration: {df['length_sec'].sum()/3600:.1f} hours ({df['length_sec'].sum()/86400:.2f} days)")
print(f"Q=A: {(df['q']=='A').sum():,}  Q=B: {(df['q']=='B').sum():,}")
print()
print("Per-species count distribution:")
cnt = df.groupby("label").size()
print(f"  mean={cnt.mean():.1f}  median={cnt.median():.0f}  min={cnt.min()}  max={cnt.max()}")
print(f"  Species with 1 recording: {(cnt==1).sum()}")
print(f"  Species at cap ({PER_SPECIES_CAP}): {(cnt==PER_SPECIES_CAP).sum()}")
print()
print("Size estimate (assuming avg 1.5 MB/min @ mp3 128kbps):")
mins = df["length_sec"].sum() / 60
mb = mins * 1.5
print(f"  ~{mb/1024:.1f} GB")
print()
print(f"Saved: {OUT}")
