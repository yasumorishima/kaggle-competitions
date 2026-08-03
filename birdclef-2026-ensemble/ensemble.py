# %% [markdown]
# # BirdCLEF+ 2026 — Ensemble
#
# Logit blend of:
# - yasunorim/birdclef-2026-perch-v2-repro (our Perch v2 fork, LB 0.908)
# - yaroslavkholmirzayev/protossm-v18-maximum-ensemble-artifact (Perch+ProtoSSM v5+ResSSM pipeline, V18 is improvement over V16 claimed 0.924)
#
# Both kernels publish submission.csv in competition format (verified).

# %%
import numpy as np
import pandas as pd
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
COMP = INPUT_ROOT / "competitions" / "birdclef-2026"

# Locate chained kernel outputs (Kaggle mounts them under notebooks/<owner>/<slug>/)
def find_submission(slug_fragment: str) -> Path:
    candidates = sorted((INPUT_ROOT).rglob("submission.csv"))
    for c in candidates:
        if slug_fragment in str(c):
            return c
    raise FileNotFoundError(f"No submission.csv for fragment: {slug_fragment}")

p1 = find_submission("perch-v2-repro")
p2 = find_submission("protossm-v18")
print("perch-v2-repro:", p1)
print("protossm v18:", p2)

df1 = pd.read_csv(p1)
df2 = pd.read_csv(p2)
print("df1:", df1.shape, "df2:", df2.shape)

# Align row order by row_id
df1 = df1.set_index("row_id")
df2 = df2.set_index("row_id")
common_rows = df1.index.intersection(df2.index)
assert len(common_rows) == len(df1) == len(df2), f"row_id mismatch: {len(common_rows)} vs {len(df1)}/{len(df2)}"

# Align column order
species = [c for c in df1.columns if c != "row_id"]
assert set(species) == set(c for c in df2.columns if c != "row_id"), "species column mismatch"
df2 = df2[species]
df1 = df1[species]

# Logit blend (clip to avoid log(0))
EPS = 1e-6
def to_logit(p):
    p = np.clip(p.values.astype(np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))

logit1 = to_logit(df1)
logit2 = to_logit(df2)

# Weight protossm-v18 more — V18 > V16 (V16 claimed 0.924) is higher than our Perch v2 repro (LB 0.908)
W1, W2 = 0.4, 0.6  # (perch-v2-repro, protossm-v18)
blended_logit = W1 * logit1 + W2 * logit2
blended = 1.0 / (1.0 + np.exp(-blended_logit))

out = pd.DataFrame(blended, index=df1.index, columns=species)
out = out.reset_index()

# Reorder to sample_submission.csv order
sample = pd.read_csv(COMP / "sample_submission.csv")
expected_ids = set(sample["row_id"])
our_ids = set(out["row_id"])
missing = expected_ids - our_ids
if missing:
    print(f"WARNING: {len(missing)} missing row_ids — filling with zeros")
    missing_df = pd.DataFrame({"row_id": list(missing)})
    for sp in species:
        missing_df[sp] = 0.0
    out = pd.concat([out, missing_df], ignore_index=True)

extra = our_ids - expected_ids
if extra:
    print(f"Dropping {len(extra)} extra row_ids")
    out = out[out["row_id"].isin(expected_ids)]

out = out.set_index("row_id").loc[sample["row_id"]].reset_index()
out.to_csv("submission.csv", index=False)
print(f"Submission saved: {out.shape}")
print(out.head())
