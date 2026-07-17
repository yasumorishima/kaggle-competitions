"""Kernel B generator: sp45 + TCN + BLEND-OOF, mounting kernel A's fleongg-OOF stash.

WHY THIS EXISTS
---------------
The single-kernel BLEND-OOF diagnostic OOM-killed at 6.5h (fleongg retrained on all 773
wells with sp45's stack resident) and, even OOM-fixed, would blow Kaggle's 12h wall. The
work was split in two:

  kernel A (rogii-wellbore-fle-stash) : runs ONLY fleongg at 773-well fidelity and dumps
      `_BLEND_OOF_STASH` (id/true/last/sub1/lp/fle) to fle_stash.npz.
  kernel B (this)                     : runs sp45 (surface+dipbeam+GBT+TCN+Ridge) + the
      read-only BLEND-OOF diagnostic, MOUNTS kernel A's fle_stash.npz, and never touches
      fleongg. Both fit in time and memory.

WHAT IT REUSES
--------------
It imports the live dualpipe generator (rogii-wellbore-dualpipe/generate_notebook.py) purely
to reuse its verified cell-source blocks (SMOKE / SURFACE / DIPBEAM / GBT-CPU-fallback / TCN /
DIAG / BLEND-OOF) and helpers, so kernel B stays byte-in-sync with the banked sp45 pipeline.
Importing that module runs nothing (its main() is __main__-guarded). This generator NEVER
writes to the dualpipe dir; it reads that dir's pristine base notebook read-only.

DIFFERENCES FROM THE DUALPIPE GENERATOR
---------------------------------------
  * fleongg cells (base code cells 23-34) and the final-blend / override / gold cells
    (35-38) are DROPPED -- kernel B is a read-only diagnostic, not a submission.
  * GEO-OOF is DROPPED (it is the only diagnostic that needs fleongg's lik_pf; already
    concluded -> DenseANCC rejected, quality-conditioning refuted).
  * BLEND-OOF's stash read `_ST = globals().get('_BLEND_OOF_STASH')` is patched to fall
    back to loading kernel A's mounted fle_stash.npz.
  * A tiny faithful `PP` stub is injected (kernel B does not run fleongg, but BLEND-OOF's
    trailing [BLEND-OOF-FLE] w_sub1 sweep reads PP.{w_sub1,sg_win,sg_poly}). Literals copied
    verbatim from fleongg's `class PP` (base code cell 32).

SMOKE=True  -> 8 wells / 24x350 PF / 2-epoch TCN / capped GBTs for a fast plumbing check
             (validates the npz mount + stash load + FINAL print). Numbers are unreliable.
SMOKE=False -> 40 wells / 128x500 production-fidelity PF / full TCN -> the real signal.
"""
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
DPDIR = BASE.parent / "rogii-wellbore-dualpipe"
SRC = DPDIR / "rogii-dualpipe.base.ipynb"      # pristine base (read-only)
OUT = BASE / "rogii-bo-blend.ipynb"

# Flip to False, regenerate, commit, push for the full-fidelity run.
SMOKE = True

# --- import the dualpipe generator to reuse its verified cell-source blocks + helpers ----
_spec = importlib.util.spec_from_file_location("dpgen", DPDIR / "generate_notebook.py")
dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dp)   # safe: dp.main() is __main__-guarded, import has no side effects

src_str, set_src, find_cell, code_cell = dp.src_str, dp.set_src, dp.find_cell, dp.code_cell


# PP stub: kernel B does not run fleongg, but BLEND-OOF's [BLEND-OOF-FLE] sweep reads
# PP.{w_sub1,sg_win,sg_poly}. Literals verbatim from fleongg `class PP` (base cell 32).
PP_STUB_SRC = '''\
# === PP stub (kernel B runs no fleongg; [BLEND-OOF-FLE] reads PP.w_sub1/sg_win/sg_poly) ===
# Literals copied verbatim from fleongg's `class PP` in the base notebook (code cell 32).
class PP:
    alpha = 1.0
    tau = 85.0
    w_pf = 0.0
    w_sub1 = 0.60
    sub2_scale = "scale_5"
    sg_win = 61
    sg_poly = 3
print("[PP-STUB] w_sub1=", PP.w_sub1, "sg_win=", PP.sg_win, "sg_poly=", PP.sg_poly, flush=True)
'''

# BLEND-OOF stash read: kernel A's global is absent here (no fleongg), so fall back to the
# mounted fle_stash.npz that kernel A persisted.
_STASH_ANCHOR = "    _ST = globals().get('_BLEND_OOF_STASH')\n"
_STASH_PATCH = (
    "    _ST = globals().get('_BLEND_OOF_STASH')\n"
    "    if _ST is None:                       # kernel B: load kernel A's mounted stash\n"
    "        import numpy as _np_ld, glob as _glob_ld\n"
    "        _cand = _glob_ld.glob('/kaggle/input/**/fle_stash.npz', recursive=True)\n"
    "        if _cand:\n"
    "            _z = _np_ld.load(_cand[0], allow_pickle=False)\n"
    "            _ST = {_k: _z[_k] for _k in _z.files}\n"
    "            print('[BLEND-OOF] mounted stash', _cand[0], 'rows=', len(_ST['id']), flush=True)\n"
    "        else:\n"
    "            print('[BLEND-OOF] NO mounted fle_stash.npz -> FINAL/sweep skipped', flush=True)\n"
)


def main():
    nb = json.load(open(SRC, encoding="utf-8"))
    cells = nb["cells"]
    report = []

    # --- 1. SMOKE cell right after the sp45 CFG (find_cell returns the FIRST "class CFG:",
    #        which is sp45's at code cell 3, not fleongg's at 23). ------------------------
    i_cfg = find_cell(cells, "class CFG:")
    cfg = src_str(cells[i_cfg])
    assert "n_splits = 5" in cfg and "GroupKFold(n_splits=n_splits)" in cfg, "not the sp45 CFG"
    cells.insert(i_cfg + 1, code_cell(dp.SMOKE_SRC.replace("__SMOKE__", "True" if SMOKE else "False")
                                      .replace("__FLE_RETRAIN__", "False")))
    report.append(f"inserted SMOKE cell at {i_cfg + 1} (after sp45 CFG)")

    # --- 1b. SURFACE then DIPBEAM after the feature-build cell -------------------------
    i_feat = find_cell(cells, "features = [c for c in train_df.columns if c not in {'well','id','target'}]")
    fc = src_str(cells[i_feat])
    assert "X = train_df[features]" in fc and "X_test = test_df[features]" in fc, "unexpected feature-build cell"
    cells.insert(i_feat + 1, code_cell(dp.SURF_SRC))
    cells.insert(i_feat + 2, code_cell(dp.DIPBEAM_SRC))
    report.append(f"inserted SURFACE at {i_feat + 1} and DIPBEAM at {i_feat + 2}")

    # --- 1c. FORCE GBT retrain (surface/dipbeam changed the feature set) ----------------
    nr = 0
    for c in cells:
        if c["cell_type"] != "code":
            continue
        cs = src_str(c)
        anc = "if (CFG.artifacts_path / save_path).exists():"
        if anc in cs and "and not FORCE_RETRAIN" not in cs:
            set_src(c, cs.replace(anc, "if (CFG.artifacts_path / save_path).exists() and not FORCE_RETRAIN:"))
            nr += 1
    assert nr == 2, f"expected to gate 2 GBT load branches, gated {nr}"
    report.append(f"gated {nr} GBT load branches on `and not FORCE_RETRAIN`")

    # --- 1d. sp45 GBT CPU/quota fallback right after the param dicts --------------------
    i_par = find_cell(cells, "lgb_params = [")
    par_src = src_str(cells[i_par])
    assert "cb_params = [" in par_src and "ridge_params = {" in par_src, "unexpected sp45 param cell"
    cells.insert(i_par + 1, code_cell(dp.GBT_CPU_FALLBACK_SRC))
    report.append(f"inserted sp45 GBT CPU-fallback cell at {i_par + 1}")

    # --- 2. TCN member cell right before `oof_preds = pd.DataFrame(oof_preds)` ----------
    i_df = find_cell(cells, "oof_preds = pd.DataFrame(oof_preds)")
    cells.insert(i_df, code_cell(dp.TCN_SRC))
    report.append(f"inserted TCN cell at {i_df}")

    # --- 3. DIAG cell right after the Ridge stack --------------------------------------
    i_rd = find_cell(cells, "ridge_oof_preds = ridge_trainer.oof_preds")
    cells.insert(i_rd + 1, code_cell(dp.DIAG_SRC))
    report.append(f"inserted DIAG cell at {i_rd + 1}")

    # --- 4. TRIM: drop the fleongg bridge cell and everything after it (fleongg 23-34 +
    #        final-blend/override/gold 35-38). All sp45 cells + our inserts are BEFORE the
    #        bridge, so they survive. --------------------------------------------------
    i_bridge = find_cell(cells, "fleongg pretrained inference section")
    dropped = len(cells) - i_bridge
    cells[:] = cells[:i_bridge]
    report.append(f"trimmed {dropped} cells from the fleongg bridge onward (kept {len(cells)})")

    # --- 5. PP stub, then the BLEND-OOF diagnostic (stash-read patched to mount npz) -----
    cells.append(code_cell(PP_STUB_SRC))
    report.append(f"appended PP stub at {len(cells) - 1}")

    bo = dp.BLEND_OOF_SRC
    assert bo.count(_STASH_ANCHOR) == 1, "BLEND-OOF stash anchor not found exactly once"
    bo = bo.replace(_STASH_ANCHOR, _STASH_PATCH)
    cells.append(code_cell(bo))
    report.append(f"appended BLEND-OOF diagnostic (npz-mount stash loader) at {len(cells) - 1} (last)")

    nb["cells"] = cells
    nb.setdefault("nbformat", 4)
    nb.setdefault("nbformat_minor", 5)
    json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # === round-trip + asserts =========================================================
    rt = json.load(open(OUT, encoding="utf-8"))
    codes = [c for c in rt["cells"] if c["cell_type"] == "code"]
    full = "\n".join(src_str(c) for c in codes)

    # sp45 backbone intact
    assert "class _FormSurf:" in full and "tvtS_" in full, "SURFACE cell missing"
    assert "def _beam_dip_jit(" in full and "tvt_dipbeam_d" in full, "DIPBEAM cell missing"
    assert "class _TCN(nn.Module)" in full, "TCN class missing"
    assert 'oof_preds["tcn"] = _tcn_oof' in full, "TCN dict assignment missing"
    assert full.index('oof_preds["tcn"] = _tcn_oof') < full.index("oof_preds = pd.DataFrame(oof_preds)"), \
        "TCN dict assignment must precede DataFrame conversion"
    assert "[DIAG] Ridge-A OOF" in full, "DIAG cell missing"
    assert full.index("ridge_oof_preds = ridge_trainer.oof_preds") < full.index("[DIAG] Ridge-A OOF"), \
        "DIAG must come after the Ridge stack"
    assert "[GBT-DEVICE]" in full, "GBT CPU-fallback cell missing"
    assert full.count("and not FORCE_RETRAIN") == 2, "both GBT load branches must be gated"
    assert 'pd.read_csv(CFG.artifacts_path / "data" / "train.csv"' in full, "train.csv fast-load harmed"
    assert "SMOKE = True" in full or "SMOKE = False" in full, "SMOKE flag cell missing"

    # fleongg + downstream fully removed
    for banned in ("sub, cv_final = main()", "def make_prediction", "def build_likpf",
                   "_SELECTED_SP45_WEIGHT = 0.55", "Gold visible-prefix calibration",
                   "_ov_tvt_from_contacts", "[GEO-OOF]", "fleongg pretrained inference section"):
        assert banned not in full, f"fleongg/downstream leaked into kernel B: {banned!r}"
    assert "def main(" not in full, "no fleongg main() should survive"

    # PP stub present with the three constants BLEND-OOF-FLE reads
    assert "[PP-STUB]" in full and "w_sub1 = 0.60" in full and "sg_win = 61" in full \
        and "sg_poly = 3" in full, "PP stub / constants missing"

    # BLEND-OOF present, last, read-only, production-fidelity, mounts the npz stash
    bo_cell = src_str(codes[-1])
    assert "[BLEND-OOF]" in bo_cell, "BLEND-OOF must be the last cell"
    assert "to_csv(" not in bo_cell, "BLEND-OOF must be read-only (no csv writes)"
    assert "n_particles=_BO_PART, n_seeds=_BO_SEEDS" in bo_cell, "BLEND-OOF must run at prod fidelity"
    assert "glob('/kaggle/input/**/fle_stash.npz', recursive=True)" in bo_cell, "npz stash loader missing"
    assert "mounted stash" in bo_cell, "stash mount print missing"
    assert "[BLEND-OOF-FLE]" in bo_cell, "fleongg w_sub1 sweep missing"
    assert "FINAL@default 0.70/0.55/0.15" in bo_cell, "FINAL@default view missing"
    # the stash-read fallback sits before the fleongg-dependent [BLEND-OOF-FLE] section
    assert bo_cell.index("mounted stash") < bo_cell.index("[BLEND-OOF-FLE]"), \
        "stash must load before the FLE sweep uses it"

    print("=== PATCH REPORT (kernel B) ===")
    for r in report:
        print(" -", r)
    print(f"=== {OUT.name}: {len(rt['cells'])} cells (SMOKE={SMOKE}) | round-trip + asserts OK ===")


if __name__ == "__main__":
    main()
