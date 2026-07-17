"""Kernel C generator: kernel B (sp45 + TCN + BLEND-OOF, mounts kernel A's fle stash) PLUS a
read-only PF-motion-constant sweep on sub2 (the selector particle filter).

WHY
---
full-B proved blend-weight tuning is exhausted (no knob beats the 0.70/0.55/0.15 default; every
pooled gain overfits the drift wells). full-B also showed WHERE the gold gap lives: on the drift
wells that dominate the metric, the selector-PF (sub2) blows up (RMSE 19-23) while GBT/TCN stay
~5. That is the flatten-to-horizontal drift of the PF motion model. Its six motion constants
(MOM/VN/PN/RP/RR/RESAMP, hardcoded in `run_particle_filter`) plus the init spreads and GR-sigma
floor have NEVER been validated against held-out truth. This kernel measures them.

DESIGN (Opus-reviewed, see kaggle-rogii.md "v2 設計レビュー確定")
-----------------------------------------------------------------
1. kwargs-ify `run_particle_filter` + both `run_pf_lik_ensemble*` (defaults == original literals,
   so the production path is byte-unchanged when unswept).
2. In BLEND-OOF, after the baseline, recompute sub2 per PF-constant CANDIDATE (holding s1/fle/tcn
   and the default weights), rebuild FINAL, score vs the true toe. beam/variant/lk are
   PF-independent -> cached from the baseline pass; only the PF is recomputed per candidate.
   CRN is automatic: `run_pf_lik_ensemble_scales` reseeds 0..n_seeds identically for every
   candidate, so RMSE_cand - RMSE_default is a *paired* delta (variance largely cancels).
3. Guards (Opus Q4): per-candidate pooled / median / win-lose / both even-odd folds, PLUS
   leave-one-well-out sign-stability of the pooled delta (kills single-drift-well artifacts) and a
   drift-vs-clean split (a constant must not help drift wells while quietly hurting clean ones).
4. This is a SCREEN with fle frozen in the mounted stash (Opus Q2): fle does NOT respond to the
   constant here, so a winner must later be CERTIFIED with fle unfrozen (a kernel-A re-run at the
   new constant). Do not deploy off this screen alone. Read-only: no submission is written.

Candidates target the flatten-bias mechanism: MOM -> 1.0 (remove decay-to-horizontal), co-swept
with VN; plus the GR-sigma floor (let GR pull position back onto structure). 7 candidates incl. a
`default_check` sanity (must reproduce the baseline sub2 exactly via CRN).

SMOKE=True -> 8 wells / 24x350 / 3 candidates for a fast plumbing check.
SMOKE=False -> 40 wells / 128x500 / 7 candidates; ~41min per candidate (full-B measured T=2442s).
"""
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
DPDIR = BASE.parent / "rogii-wellbore-dualpipe"
BODIR = BASE.parent / "rogii-wellbore-bo-blend"
SRC = DPDIR / "rogii-dualpipe.base.ipynb"      # pristine base (read-only)
OUT = BASE / "rogii-pf-sweep.ipynb"

SMOKE = False

# import the dualpipe generator (SRC blocks + helpers) and kernel B generator (its stash-loader
# patch + PP stub). Both are __main__-guarded, so importing runs nothing.
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
dp = _load("dpgen", DPDIR / "generate_notebook.py")
kb = _load("kbgen", BODIR / "generate_notebook.py")
src_str, set_src, find_cell, code_cell = dp.src_str, dp.set_src, dp.find_cell, dp.code_cell


# --- PF kwargs-ification: exact string patches on sp45 cell 4 (run_particle_filter is unique to
#     sp45; fleongg uses _pf_lik_allseeds/lik_pf, and kernel C trims fleongg anyway). -----------
PF_PATCHES = [
    ("def run_particle_filter(hw, tw, n_particles=500, seed=42):",
     "def run_particle_filter(hw, tw, n_particles=500, seed=42, "
     "MOM=0.998, VN=0.002, PN=0.005, RP=0.1, RR=0.001, RESAMP=0.5, POS0=4.5, RATE0=0.01, GS_LO=10.0):"),
    ("    MOM = 0.998; VN = 0.002; PN = 0.005; RP = 0.1; RR = 0.001; RESAMP = 0.5",
     "    # motion constants are kwargs now (v2 PF sweep); defaults == the original literals"),
    ("    pos  = ls + 4.5 * rng.standard_normal(N)  # sp45 patch (sel15 vb best)",
     "    pos  = ls + POS0 * rng.standard_normal(N)  # sp45 patch (sel15 vb best)"),
    ("    rate = ir + 0.01 * rng.standard_normal(N)",
     "    rate = ir + RATE0 * rng.standard_normal(N)"),
    ("    gs = float(np.clip(np.nanstd(kn['GR'].fillna(0).values - tw_at_k), 10., 60.))",
     "    gs = float(np.clip(np.nanstd(kn['GR'].fillna(0).values - tw_at_k), GS_LO, 60.))"),
    ("def run_pf_lik_ensemble(hw, tw, n_particles=500, n_seeds=128, scale=5.0):",
     "def run_pf_lik_ensemble(hw, tw, n_particles=500, n_seeds=128, scale=5.0, **pf_kw):"),
    ("def run_pf_lik_ensemble_scales(hw, tw, scales=SELECTOR_SCALES, n_particles=500, n_seeds=128):",
     "def run_pf_lik_ensemble_scales(hw, tw, scales=SELECTOR_SCALES, n_particles=500, n_seeds=128, **pf_kw):"),
    # both ensembles call run_particle_filter with the identical line -> thread pf_kw through both
    ("        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s)",
     "        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s, **pf_kw)"),
]

# --- BLEND-OOF surgery 1: cache the PF-independent per-well objects so the sweep can recompute
#     only the PF. Injected into the `_bo_wells.append(dict(...))` at the end of the baseline loop. -
_CACHE_ANCHOR = "                ps=float(_lr['MD']), end=float(_hw['MD'].iloc[-1]), nk=int(len(_kn))))"
_CACHE_PATCH = (
    "                ps=float(_lr['MD']), end=float(_hw['MD'].iloc[-1]), nk=int(len(_kn)),\n"
    "                hw=_hw, tw=_tw, beam=_beam, variant=_variant, lk=float(_lk), ti=_ti, mk=_mk))"
)

# --- BLEND-OOF surgery 2: the PF-constant sweep block, injected right before [BLEND-OOF-FLE]. ---
_SWEEP_ANCHOR = "    # --- [BLEND-OOF-FLE] fleongg w_sub1 sweep on ALL train wells (OOF-only, no physics) ---"
_SWEEP_BLOCK = r'''    # --- [PF-SWEEP] recompute sub2 (selector-PF) per PF-motion-constant candidate ------------
    #     Hold s1/fle/tcn and the default weights (0.70/0.55/0.15); score FINAL vs held-out truth.
    #     beam/variant/lk are PF-independent -> cached in the well dict; only the PF is recomputed.
    #     CRN: run_pf_lik_ensemble_scales reseeds 0..n_seeds identically per candidate == paired delta.
    #     SCREEN ONLY: fle is frozen (mounted) -> a winner must be re-certified with fle unfrozen.
    if _bo_wells and _bo_fle:
        _PF_CANDS = ([('default_check', {}), ('MOM1.0', dict(MOM=1.0)), ('GS_LO5', dict(GS_LO=5.0))]
                     if SMOKE else
                     [('default_check',   {}),
                      ('MOM0.995',        dict(MOM=0.995)),
                      ('MOM0.999',        dict(MOM=0.999)),
                      ('MOM1.0',          dict(MOM=1.0)),
                      ('MOM1.0_VN0.004',  dict(MOM=1.0, VN=0.004)),
                      ('GS_LO5',          dict(GS_LO=5.0)),
                      ('GS_LO7',          dict(GS_LO=7.0))])
        _pfsc = tuple(globals().get('SELECTOR_SCALES', (3.0, 5.0, 8.0, 12.0)))
        _pf_fold = _bnp.arange(len(_bo_wells)) % 2
        _rarr = _bnp.asarray(_ref, dtype=float)             # baseline FINAL@default per-well RMSE
        _pf_drift = _rarr >= 9.0                            # drift wells = high baseline FINAL RMSE
        import time as _pf_time
        # cache baseline FINAL per-well residual-sums for the fast leave-one-out recompute
        _base_ss = _bnp.asarray([float(((_bo_final_pred(_w, 0.70, 0.55, 0.15) - _w['true']) ** 2).sum())
                                 for _w in _bo_wells], dtype=float)
        _base_n = _bnp.asarray([len(_w['true']) for _w in _bo_wells], dtype=float)
        print(f'[PF-SWEEP] {len(_PF_CANDS)} candidates x {len(_bo_wells)} wells @ {_BO_SEEDS}x{_BO_PART} '
              f'(baseline pooled={_refp:.4f})', flush=True)
        try:
            import resource as _pf_res
            print(f'[PF-SWEEP] maxRSS={_pf_res.getrusage(_pf_res.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB '
                  f'before sweep', flush=True)
        except Exception:
            pass
        for _cname, _cand in _PF_CANDS:
            _pf_t0 = _pf_time.time(); _nfb = 0
            _cfin = []
            for _w in _bo_wells:
                try:
                    _pf2 = run_pf_lik_ensemble_scales(_w['hw'], _w['tw'], scales=_pfsc,
                                                      n_particles=_BO_PART, n_seeds=_BO_SEEDS, **_cand)
                    _sel2 = _bnp.asarray(apply_selector_variant(_w['variant'], _pf2, _w['beam'], _w['lk']),
                                         dtype=float)
                    _wc = dict(_w); _wc['s2'] = _sel2[_w['ti']][_w['mk']]
                    _cfin.append(_bnp.asarray(_bo_final_pred(_wc, 0.70, 0.55, 0.15), dtype=float))
                except Exception as _pe:
                    _nfb += 1
                    _cfin.append(_bnp.asarray(_bo_final_pred(_w, 0.70, 0.55, 0.15), dtype=float))
            _perr = _bnp.empty(len(_bo_wells)); _cand_ss = _bnp.empty(len(_bo_wells)); _ss = 0.0; _ntot = 0
            for _wi, _w in enumerate(_bo_wells):
                _r = _cfin[_wi] - _w['true']
                _cand_ss[_wi] = float((_r ** 2).sum())
                _perr[_wi] = float(_bnp.sqrt(_bnp.mean(_r ** 2))); _ss += _cand_ss[_wi]; _ntot += len(_r)
            _cp = float(_bnp.sqrt(_ss / max(_ntot, 1)))
            _win = int((_perr < _rarr - 1e-9).sum()); _lose = int((_perr > _rarr + 1e-9).sum())
            _cm, _cmd, _ = _bo_stats(_perr); _rm, _rmd, _ = _bo_stats(_rarr)
            _mA = float(_bnp.mean(_perr[_pf_fold == 0])); _mB = float(_bnp.mean(_perr[_pf_fold == 1]))
            _rA = float(_bnp.mean(_rarr[_pf_fold == 0])); _rB = float(_bnp.mean(_rarr[_pf_fold == 1]))
            _full_d = _refp - _cp                            # >0 == candidate better (lower pooled)
            # leave-one-well-out sign stability: drop each well, require the pooled-delta sign to hold
            _tot_ss_c = _cand_ss.sum(); _tot_ss_b = _base_ss.sum(); _tot_n = _base_n.sum()
            _loo_ok = True
            for _j in range(len(_bo_wells)):
                _nn = _tot_n - _base_n[_j]
                if _nn <= 0:
                    continue
                _dj = (_bnp.sqrt((_tot_ss_b - _base_ss[_j]) / _nn)
                       - _bnp.sqrt((_tot_ss_c - _cand_ss[_j]) / _nn))
                if (_full_d > 0) != (_dj > 0):
                    _loo_ok = False; break
            _dc = float(_bnp.sqrt((_perr[_pf_drift] ** 2).mean())) if _pf_drift.any() else float('nan')
            _rdc = float(_bnp.sqrt((_rarr[_pf_drift] ** 2).mean())) if _pf_drift.any() else float('nan')
            _cc = float(_bnp.sqrt((_perr[~_pf_drift] ** 2).mean())) if (~_pf_drift).any() else float('nan')
            _rcc = float(_bnp.sqrt((_rarr[~_pf_drift] ** 2).mean())) if (~_pf_drift).any() else float('nan')
            print(f'[PF-SWEEP] {_cname:16s} pooled={_cp:7.4f}(d={_full_d:+.4f}) '
                  f'median={_cmd:7.4f}(def {_rmd:.4f}) win/lose={_win}/{_lose} '
                  f'foldA={_mA:.3f}/{_rA:.3f} foldB={_mB:.3f}/{_rB:.3f} '
                  f'LOOsign={"OK" if _loo_ok else "FLIP"} '
                  f'drift={_dc:.3f}/{_rdc:.3f} clean={_cc:.3f}/{_rcc:.3f} '
                  f'fb={_nfb} ({_pf_time.time()-_pf_t0:.0f}s)', flush=True)

'''


def main():
    nb = json.load(open(SRC, encoding="utf-8"))
    cells = nb["cells"]
    report = []

    # 1. SMOKE cell after sp45 CFG (find_cell returns the first "class CFG:" = sp45's).
    i_cfg = find_cell(cells, "class CFG:")
    assert "n_splits = 5" in src_str(cells[i_cfg]), "not the sp45 CFG"
    cells.insert(i_cfg + 1, code_cell(dp.SMOKE_SRC.replace("__SMOKE__", "True" if SMOKE else "False")
                                      .replace("__FLE_RETRAIN__", "False")))
    report.append(f"inserted SMOKE cell at {i_cfg + 1}")

    # 1b. SURFACE + DIPBEAM after the feature-build cell.
    i_feat = find_cell(cells, "features = [c for c in train_df.columns if c not in {'well','id','target'}]")
    assert "X = train_df[features]" in src_str(cells[i_feat]), "unexpected feature-build cell"
    cells.insert(i_feat + 1, code_cell(dp.SURF_SRC))
    cells.insert(i_feat + 2, code_cell(dp.DIPBEAM_SRC))
    report.append(f"inserted SURFACE/DIPBEAM at {i_feat + 1}/{i_feat + 2}")

    # 1c. FORCE GBT retrain (surface/dipbeam changed the feature set).
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
    report.append("gated 2 GBT load branches on FORCE_RETRAIN")

    # 1d. sp45 GBT CPU/quota fallback after the param dicts.
    i_par = find_cell(cells, "lgb_params = [")
    assert "cb_params = [" in src_str(cells[i_par]), "unexpected sp45 param cell"
    cells.insert(i_par + 1, code_cell(dp.GBT_CPU_FALLBACK_SRC))
    report.append(f"inserted GBT CPU-fallback at {i_par + 1}")

    # 1e. PF kwargs-ification on sp45 cell 4 (run_particle_filter + both ensembles).
    #     Applied across the whole notebook source; the anchors are unique to sp45 (fleongg uses
    #     _pf_lik_allseeds/lik_pf), and fleongg is trimmed below anyway.
    for c in cells:
        if c["cell_type"] != "code":
            continue
        cs = src_str(c)
        changed = cs
        for old, new in PF_PATCHES:
            changed = changed.replace(old, new)
        if changed != cs:
            set_src(c, changed)
    joined_all = "\n".join(src_str(c) for c in cells if c["cell_type"] == "code")
    for old, new in PF_PATCHES:
        assert new in joined_all, f"PF patch did not apply: {new[:60]!r}"
        assert old not in joined_all, f"PF pre-patch text survived: {old[:60]!r}"
    report.append(f"applied {len(PF_PATCHES)} PF kwargs patches to run_particle_filter + ensembles")

    # 2. TCN member before the dict->DataFrame conversion.
    i_df = find_cell(cells, "oof_preds = pd.DataFrame(oof_preds)")
    cells.insert(i_df, code_cell(dp.TCN_SRC))
    report.append(f"inserted TCN at {i_df}")

    # 3. DIAG after the Ridge stack.
    i_rd = find_cell(cells, "ridge_oof_preds = ridge_trainer.oof_preds")
    cells.insert(i_rd + 1, code_cell(dp.DIAG_SRC))
    report.append(f"inserted DIAG at {i_rd + 1}")

    # 4. TRIM fleongg bridge onward (keep sp45 + our inserts).
    i_bridge = find_cell(cells, "fleongg pretrained inference section")
    dropped = len(cells) - i_bridge
    cells[:] = cells[:i_bridge]
    report.append(f"trimmed {dropped} cells from fleongg bridge (kept {len(cells)})")

    # 5. PP stub (kernel B's, verbatim).
    cells.append(code_cell(kb.PP_STUB_SRC))
    report.append("appended PP stub")

    # 6. BLEND-OOF: kernel B's npz-mount stash loader + cache PF-independent per-well objects +
    #    the PF-constant sweep block.
    bo = dp.BLEND_OOF_SRC
    assert bo.count(kb._STASH_ANCHOR) == 1, "stash anchor not found once"
    bo = bo.replace(kb._STASH_ANCHOR, kb._STASH_PATCH)
    assert bo.count(_CACHE_ANCHOR) == 1, "well-cache anchor not found once"
    bo = bo.replace(_CACHE_ANCHOR, _CACHE_PATCH)
    assert bo.count(_SWEEP_ANCHOR) == 1, "sweep anchor ([BLEND-OOF-FLE]) not found once"
    bo = bo.replace(_SWEEP_ANCHOR, _SWEEP_BLOCK + _SWEEP_ANCHOR)
    cells.append(code_cell(bo))
    report.append(f"appended BLEND-OOF + PF-sweep at {len(cells) - 1} (last)")

    nb["cells"] = cells
    nb.setdefault("nbformat", 4)
    nb.setdefault("nbformat_minor", 5)
    json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # === round-trip + asserts ===
    rt = json.load(open(OUT, encoding="utf-8"))
    codes = [c for c in rt["cells"] if c["cell_type"] == "code"]
    full = "\n".join(src_str(c) for c in codes)

    # sp45 backbone + kernel-B pieces intact
    assert "class _FormSurf:" in full and "def _beam_dip_jit(" in full, "SURFACE/DIPBEAM missing"
    assert "class _TCN(nn.Module)" in full and 'oof_preds["tcn"] = _tcn_oof' in full, "TCN missing"
    assert "[DIAG] Ridge-A OOF" in full, "DIAG missing"
    assert "[GBT-DEVICE]" in full and full.count("and not FORCE_RETRAIN") == 2, "GBT fallback/gate missing"
    assert "[PP-STUB]" in full and "w_sub1 = 0.60" in full, "PP stub missing"
    assert "glob('/kaggle/input/**/fle_stash.npz', recursive=True)" in full, "npz stash loader missing"

    # fleongg + downstream gone
    for banned in ("sub, cv_final = main()", "def make_prediction", "def build_likpf",
                   "_SELECTED_SP45_WEIGHT = 0.55", "Gold visible-prefix calibration", "[GEO-OOF]"):
        assert banned not in full, f"fleongg/downstream leaked: {banned!r}"

    # PF kwargs-ification
    assert "def run_particle_filter(hw, tw, n_particles=500, seed=42, MOM=0.998" in full, "PF signature not kwargs-ified"
    assert "MOM = 0.998; VN = 0.002; PN = 0.005" not in full, "PF hardcode line survived"
    assert "n_seeds=128, **pf_kw)" in full, "ensemble_scales not **pf_kw"
    assert "seed=s, **pf_kw)" in full, "run_particle_filter call not threading pf_kw"
    assert "GS_LO, 60." in full and "POS0 * rng" in full and "RATE0 * rng" in full, "init/gs kwargs not wired"

    # PF sweep block, read-only, with all guards, before [BLEND-OOF-FLE]
    bo_cell = src_str(codes[-1])
    assert "[PF-SWEEP]" in bo_cell, "PF-sweep block missing"
    assert "to_csv(" not in bo_cell, "BLEND-OOF/PF-sweep must be read-only"
    assert "run_pf_lik_ensemble_scales(_w['hw'], _w['tw']" in bo_cell, "sweep must recompute PF from cached hw/tw"
    assert "hw=_hw, tw=_tw, beam=_beam, variant=_variant" in bo_cell, "well-cache not injected"
    assert "default_check" in bo_cell, "default_check sanity candidate missing"
    assert "LOOsign" in bo_cell and "_loo_ok" in bo_cell, "leave-one-well-out guard missing"
    assert "drift=" in bo_cell and "clean=" in bo_cell, "drift/clean split missing"
    assert bo_cell.index("[PF-SWEEP]") < bo_cell.index("[BLEND-OOF-FLE]"), "PF-sweep must precede FLE sweep"
    assert bo_cell.index("mounted stash") < bo_cell.index("[PF-SWEEP]"), "stash must load before PF-sweep"

    print("=== PATCH REPORT (kernel C / PF-sweep) ===")
    for r in report:
        print(" -", r)
    print(f"=== {OUT.name}: {len(rt['cells'])} cells (SMOKE={SMOKE}) | round-trip + asserts OK ===")


if __name__ == "__main__":
    main()
