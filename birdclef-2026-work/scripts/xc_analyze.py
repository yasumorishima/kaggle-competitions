"""Aggregate XC metadata JSON per species → distribution, rare species, licenses, duration."""
import json
from pathlib import Path
import pandas as pd
from collections import Counter

CACHE = Path("C:/Users/fw_ya/Desktop/Claude_code/kaggle-competitions/birdclef-2026-work/xc_cache")
TAX = Path("C:/Users/fw_ya/Desktop/Claude_code/kaggle-competitions/birdclef-2026-embed/out/taxonomy.csv")

tax = pd.read_csv(TAX).set_index("primary_label")

rows = []
license_ctr = Counter()
quality_ctr = Counter()
country_ctr = Counter()
dur_total_sec = 0

for jf in sorted(CACHE.glob("*.json")):
    label = jf.stem
    data = json.loads(jf.read_text(encoding="utf-8"))
    n = int(data.get("numRecordings", 0))
    scientific = tax.loc[label, "scientific_name"] if label in tax.index else "?"
    cls = tax.loc[label, "class_name"] if label in tax.index else "?"
    recs = data.get("recordings", [])
    sp_dur = 0
    for r in recs:
        license_ctr[r.get("lic", "?")] += 1
        quality_ctr[r.get("q", "?")] += 1
        country_ctr[r.get("cnt", "?")] += 1
        length = r.get("length", "0:00")
        try:
            parts = [int(x) for x in length.split(":")]
            if len(parts) == 2:
                sp_dur += parts[0] * 60 + parts[1]
            elif len(parts) == 3:
                sp_dur += parts[0] * 3600 + parts[1] * 60 + parts[2]
        except Exception:
            pass
    dur_total_sec += sp_dur
    rows.append({
        "label": label,
        "scientific": scientific,
        "class": cls,
        "n_recordings": n,
        "duration_hr": sp_dur / 3600.0,
    })

df = pd.DataFrame(rows)
df = df.sort_values("n_recordings", ascending=False).reset_index(drop=True)

print("===== Overall =====")
print(f"Species: {len(df)}")
print(f"Total recordings: {df['n_recordings'].sum():,}")
print(f"Total duration: {dur_total_sec/3600:.1f} hours ({dur_total_sec/86400:.1f} days)")
print()

print("===== Per-class distribution =====")
print(df.groupby("class")["n_recordings"].agg(["count", "sum", "mean", "median", "min", "max"]))
print()

print("===== Rare species (< 20 recordings) =====")
rare = df[df["n_recordings"] < 20].sort_values("class")
print(f"Count: {len(rare)}")
print(rare.to_string(index=False))
print()

print("===== Very rare (0 recordings) =====")
zero = df[df["n_recordings"] == 0]
print(f"Count: {len(zero)}")
print(zero[["label", "scientific", "class"]].to_string(index=False))
print()

print("===== License distribution =====")
for lic, cnt in license_ctr.most_common(10):
    print(f"  {cnt:6d}  {lic}")
print()

print("===== Quality distribution =====")
for q, cnt in quality_ctr.most_common():
    print(f"  {cnt:6d}  q={q!r}")
print()

print("===== Top-10 source countries =====")
for c, cnt in country_ctr.most_common(10):
    print(f"  {cnt:6d}  {c}")
print()

print("===== Top-20 species (most recordings) =====")
print(df.head(20).to_string(index=False))
