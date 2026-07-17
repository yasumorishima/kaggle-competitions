"""Kernel A generator: fleongg-only notebook that persists the BLEND-OOF stash.

WHY THIS EXISTS
---------------
The single-kernel BLEND-OOF diagnostic died at 6.5h with DeadKernelError (OOM) while
fleongg retrained on all 773 wells with sp45's stack still resident, and even with the
OOM fixed the remaining work would not fit Kaggle's 12h wall.

Static analysis (ast free-variable scan) proved the two pipelines are independent:
  base.ipynb code cells 0-22  = sp45     -> 0 free variables
  base.ipynb code cells 23-34 = fleongg  -> 0 free variables
so fleongg runs standalone. This kernel (A) runs ONLY fleongg at full 773-well fidelity
and dumps `_BLEND_OOF_STASH` to fle_stash.npz. Kernel B then mounts that output and runs
sp45 + TCN + BLEND-OOF without ever retraining fleongg -> both fit time and memory.

The stash arrays are exactly what BLEND-OOF consumes: id/true/last/sub1/lp/fle.

This generator NEVER touches rogii-wellbore-dualpipe/generate_notebook.py (the live
submit path). It reads that dir's pristine base notebook read-only.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE.parent / "rogii-wellbore-dualpipe" / "rogii-dualpipe.base.ipynb"
OUT = BASE / "rogii-fle-stash.ipynb"

FLEONGG_FIRST = 23   # inclusive, code-cell index in base.ipynb
FLEONGG_LAST = 34    # inclusive


def src_str(cell):
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def set_src(cell, text):
    cell["source"] = text


def main():
    nb = json.load(open(SRC, encoding="utf-8"))
    all_cells = nb["cells"]
    code_idx = [i for i, c in enumerate(all_cells) if c["cell_type"] == "code"]
    assert len(code_idx) >= FLEONGG_LAST + 1, f"expected >={FLEONGG_LAST+1} code cells, got {len(code_idx)}"

    keep = [all_cells[code_idx[k]] for k in range(FLEONGG_FIRST, FLEONGG_LAST + 1)]
    report = [f"kept fleongg code cells {FLEONGG_FIRST}-{FLEONGG_LAST} ({len(keep)} cells)"]

    # --- sanity: the kept range must be the fleongg pipeline, not something else -------
    head = src_str(keep[0])
    assert "fleongg pretrained inference section" not in head, \
        "cell 22 (sp45 submission save + fleongg header) must NOT be in the kept range"
    joined = "\n".join(src_str(c) for c in keep)
    for anchor in ("def build_likpf", "class PP:", "def make_prediction", "sub, cv_final = main()"):
        assert joined.count(anchor) >= 1, f"fleongg anchor missing from kept cells: {anchor}"
    assert "ridge_oof_preds" not in joined, "sp45 symbol leaked into the fleongg-only range"
    report.append("verified fleongg anchors present and no sp45 symbols in range")

    # --- patch fleongg main(): stash + unconditional retrain ---------------------------
    # Mirrors generate_notebook.py step 7, minus the SMOKE well cap: kernel A exists
    # precisely to get production 773-well fidelity (the 60-well SMOKE fleongg is weaker
    # than production, which biases every blend weight measured against it).
    i_fm = next(i for i, c in enumerate(keep) if "sub, cv_final = main()" in src_str(c))
    fsrc = src_str(keep[i_fm])

    anc_cv = ('        cv_final = rmse(train_df["last_known_tvt"].values + y, '
              'make_prediction(train_df, meta_oof, None))')
    assert fsrc.count(anc_cv) == 1, "fleongg cv_final anchor not found exactly once"
    stash_src = (
        '        _fle_oof_tvt = make_prediction(train_df, meta_oof, None)\n'
        '        cv_final = rmse(train_df["last_known_tvt"].values + y, _fle_oof_tvt)\n'
        '        _pfd = train_df["pf_ancc"].values.astype(float) - train_df["last_known_tvt"].values.astype(float)\n'
        "        globals()['_BLEND_OOF_STASH'] = {\n"
        '            "id":   train_df["id"].astype(str).to_numpy(),\n'
        '            "true": (train_df["last_known_tvt"].values.astype(float) + y.astype(float)),\n'
        '            "last": train_df["last_known_tvt"].values.astype(float),\n'
        '            "sub1": (PP.alpha * warmup(train_df["md_since"].values.astype(float), PP.tau)\n'
        '                     * (meta_oof * (1 - PP.w_pf) + _pfd * PP.w_pf)).astype(float),\n'
        '            "lp":   (train_df["likpf_" + PP.sub2_scale].values.astype(float)\n'
        '                     - train_df["last_known_tvt"].values.astype(float)),\n'
        '            "fle":  _bnp_stash.asarray(_fle_oof_tvt, dtype=float),\n'
        '        }\n'
    )
    fsrc = fsrc.replace(anc_cv, stash_src.rstrip("\n"))

    # Force the from-scratch TRAIN branch: the mounted boosters were fit on ALL train
    # wells, so INFERENCE mode would yield in-sample (leaky) train predictions and would
    # never populate the stash at all.
    anc_fm = "    models_dir = _find_models()"
    assert fsrc.count(anc_fm) == 1, "fleongg models_dir anchor not found exactly once"
    fsrc = fsrc.replace(anc_fm, "    models_dir = None   # kernel A: always retrain (stash needs honest OOF)")

    assert fsrc.startswith("def _find_models():"), "unexpected fleongg main cell head"
    fsrc = ("import numpy as _bnp_stash\n"
            "_BLEND_OOF_STASH = None   # populated in main()'s TRAIN branch only\n" + fsrc)
    set_src(keep[i_fm], fsrc)
    report.append(f"patched fleongg main cell (kept idx {i_fm}): stash + forced retrain, no SMOKE cap")

    # --- append the dump cell ---------------------------------------------------------
    dump = (
        "# === kernel A output: persist the BLEND-OOF stash for kernel B ===\n"
        "# Fails LOUDLY: a silent skip here would cost another multi-hour GPU run.\n"
        "import numpy as _dnp\n"
        "_ST = globals().get('_BLEND_OOF_STASH')\n"
        "assert _ST is not None, 'stash is None -> fleongg TRAIN branch never ran (INFERENCE mode?)'\n"
        "for _k in ('id', 'true', 'last', 'sub1', 'lp', 'fle'):\n"
        "    assert _k in _ST, f'stash missing key: {_k}'\n"
        "_n = len(_ST['id'])\n"
        "assert _n > 0, 'stash is empty'\n"
        "for _k, _v in _ST.items():\n"
        "    assert len(_v) == _n, f'stash length mismatch on {_k}: {len(_v)} != {_n}'\n"
        "_fin = {_k: float(_dnp.isfinite(_dnp.asarray(_v, dtype=float)).mean())\n"
        "        for _k, _v in _ST.items() if _k != 'id'}\n"
        "print('[STASH] rows=', _n, ' wells=', len({str(_i)[:8] for _i in _ST['id']}), flush=True)\n"
        "print('[STASH] finite-fraction:', {_k: round(_v, 6) for _k, _v in _fin.items()}, flush=True)\n"
        "print('[STASH] cv_final=', cv_final, flush=True)\n"
        "_dnp.savez_compressed('/kaggle/working/fle_stash.npz',\n"
        "                      id=_dnp.asarray(_ST['id'], dtype=object).astype('U'),\n"
        "                      true=_dnp.asarray(_ST['true'], dtype=_dnp.float64),\n"
        "                      last=_dnp.asarray(_ST['last'], dtype=_dnp.float64),\n"
        "                      sub1=_dnp.asarray(_ST['sub1'], dtype=_dnp.float64),\n"
        "                      lp=_dnp.asarray(_ST['lp'], dtype=_dnp.float64),\n"
        "                      fle=_dnp.asarray(_ST['fle'], dtype=_dnp.float64))\n"
        "import os as _dos\n"
        "print('[STASH] wrote fle_stash.npz',\n"
        "      _dos.path.getsize('/kaggle/working/fle_stash.npz'), 'bytes', flush=True)\n"
        "# round-trip verify: kernel B must be able to read exactly what we think we wrote\n"
        "_chk = _dnp.load('/kaggle/working/fle_stash.npz', allow_pickle=False)\n"
        "assert len(_chk['id']) == _n and len(_chk['fle']) == _n, 'npz round-trip length mismatch'\n"
        "print('[STASH] round-trip OK', flush=True)\n"
    )
    keep.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                 "outputs": [], "source": dump})
    report.append("appended stash dump cell (asserts + finite-fraction + round-trip verify)")

    nb["cells"] = keep
    json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"wrote {OUT.name} ({len(keep)} cells)")
    for r in report:
        print(" -", r)


if __name__ == "__main__":
    main()
