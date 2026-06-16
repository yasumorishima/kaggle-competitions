"""
Generate the ROGII TUNED public baseline notebook (faithful reproduction).

Source = romantamrazov's public "Sub-9 Solution v2 (Tuned LGB+CB)" notebook
(_source_super.ipynb), the strongest *public-image, self-contained* pipeline in
the competition: it builds ALL features from the raw competition data (7 beam
configs + 2 particle filters + multi-scale NCC + FormationPlaneKNN / DenseANCC
spatial imputers + WLS b_well + formation consensus + 4-family tw_diff), then
trains 3x LightGBM (num_leaves=255, 8000 iter, diverse lr/seed) + CatBoost
(depth 7, 8000 iter) and a positive Ridge stack, with alpha/tau/w_pf grid + SG
post-processing. Public best public-notebook score ~ sub-9 (target < 9.0).

This is STEP 1 (gate: reproduce public best). My self-best is 9.905, which is
*below* the public ceiling (~8.0, pilkwang clean / sub-9 tuned). Reproducing
this self-contained tuned pipeline re-bases me at ~sub-9. STEP 2 will inject a
global formation structural-surface model + the TCN on top of this baseline.

Patches applied here are MINIMAL and non-behavioural (faithful repro):
  1. SMOKE gating cell (well/iter/fold subset) for a疎通 smoke run first.
  2. CatBoost devices "0:1" (T4x2 in the source) -> "0" (single T4 on my
     machine_shape). LightGBM stays device_type="gpu".
Nothing else in the modelling path is touched.

FLIP SMOKE to False, regenerate, commit, push for the full run.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / "_source_super.ipynb"
OUT = BASE / "rogii-tuned-pubdwt.ipynb"

# Flip to False (regenerate + commit + push) for the full run.
SMOKE = True


def src_str(cell):
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def set_src(cell, text):
    cell["source"] = text


def find_cell(cells, needle, cell_type="code"):
    for i, c in enumerate(cells):
        if c["cell_type"] != cell_type:
            continue
        if needle in src_str(c):
            return i
    raise RuntimeError(f"cell containing {needle!r} not found")


def code_cell(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


SMOKE_SRC = '''\
# === SMOKE gating (baked-in; Kaggle injects no env). FLIP via generate_notebook.py. ===
SMOKE = __SMOKE__
if SMOKE:
    N_SPLITS = 2
    for _c in LGB_CONFIGS:
        _c["n_estimators"] = 60
    CB_PARAMS["iterations"] = 60
    CB_PARAMS["od_wait"] = 30
print(f"SMOKE={SMOKE}  N_SPLITS={N_SPLITS}")
'''


def main():
    nb = json.load(open(SRC, encoding="utf-8"))
    cells = nb["cells"]
    report = []

    # 1. SMOKE cell right after the config cell (defines LGB_CONFIGS/CB_PARAMS/N_SPLITS)
    i_cfg = find_cell(cells, "CB_PARAMS=dict(")
    cfg = src_str(cells[i_cfg])
    assert "LGB_CONFIGS=[" in cfg and "N_SPLITS=" in cfg
    cells.insert(i_cfg + 1, code_cell(SMOKE_SRC.replace("__SMOKE__", "True" if SMOKE else "False")))
    report.append(f"inserted SMOKE cell at {i_cfg + 1}")

    # 2. SMOKE subset of train wells (imputers + train build) -- defined in imputer cell
    i_imp = find_cell(cells, "hw_paths=sorted(TRAIN_DIR.glob('*__horizontal_well.csv'))")
    s = src_str(cells[i_imp])
    anchor = "hw_paths=sorted(TRAIN_DIR.glob('*__horizontal_well.csv'))\n"
    assert anchor in s
    s2 = s.replace(anchor, anchor + "if SMOKE: hw_paths=hw_paths[:14]\n")
    assert s2 != s
    set_src(cells[i_imp], s2)
    report.append(f"cell {i_imp}: SMOKE subset hw_paths[:14]")

    # 3. SMOKE subset of test wells
    i_bld = find_cell(cells, "test_paths=sorted(TEST_DIR.glob('*__horizontal_well.csv'))")
    s = src_str(cells[i_bld])
    anchor = "test_paths=sorted(TEST_DIR.glob('*__horizontal_well.csv'))\n"
    assert anchor in s
    s2 = s.replace(anchor, anchor + "if SMOKE: test_paths=test_paths[:6]\n")
    assert s2 != s
    set_src(cells[i_bld], s2)
    report.append(f"cell {i_bld}: SMOKE subset test_paths[:6]")

    # 4. CatBoost single-GPU + bootstrap fix.
    #    - source is T4x2 -> my machine_shape is single T4 (devices "0:1" -> "0").
    #    - source sets subsample=0.75 but no bootstrap_type; CatBoost's default GPU
    #      bootstrap is Bayesian, which REJECTS 'subsample' (hard error). Add
    #      bootstrap_type="Bernoulli" so the author's intended row-subsampling
    #      actually applies (faithful to intent, fixes the crash).
    nd = 0
    for c in cells:
        if c["cell_type"] != "code":
            continue
        cs = src_str(c)
        if 'devices="0:1"' in cs:
            set_src(c, cs.replace('devices="0:1"', 'devices="0", bootstrap_type="Bernoulli"')); nd += 1
    assert nd >= 1, "expected a CatBoost devices=\"0:1\" to patch"
    report.append(f"patched {nd} CatBoost devices -> single-GPU + bootstrap_type=Bernoulli")

    nb.setdefault("nbformat", 4)
    nb.setdefault("nbformat_minor", 5)
    json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # round-trip + sanity
    rt = json.load(open(OUT, encoding="utf-8"))
    full = "\n".join(src_str(c) for c in rt["cells"] if c["cell_type"] == "code")
    assert 'devices="0:1"' not in full
    assert 'devices="0"' in full
    assert 'bootstrap_type="Bernoulli"' in full
    assert "if SMOKE: hw_paths=hw_paths[:14]" in full
    assert "if SMOKE: test_paths=test_paths[:6]" in full
    assert ("SMOKE = True" if SMOKE else "SMOKE = False") in full
    assert "ridge=Ridge(alpha=1.,fit_intercept=False,positive=True)" in full
    print("=== PATCH REPORT ===")
    for r in report:
        print(" -", r)
    print(f"=== {OUT.name}: {len(rt['cells'])} cells (SMOKE={SMOKE}) | round-trip + asserts OK ===")


if __name__ == "__main__":
    main()
