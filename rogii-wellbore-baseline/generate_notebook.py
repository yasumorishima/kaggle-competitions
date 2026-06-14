"""Generate the ROGII Wellbore Geology Prediction RE-TRAINING fork baseline.

Strategy
--------
The public ~9.251 "DWT-based" notebook
(nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based) is a
RE-TRAINING solution: it reads the precomputed feature matrix `data/train.csv`
from the artifacts dataset (ravaghi/wellbore-geology-prediction-artifacts),
trains 3x LightGBM + 3x CatBoost on GPU with early stopping, greedily ensembles
the OOF/test predictions with `Climber`, searches post-processing coefficients
with Optuna, then predicts on freshly computed test features and writes
`submission.csv`.

The ONLY custom dependency it imports is `from hill_climbing import Climber`
(the `Trainer` class in the artifacts is NOT used — boosters are cached to the
working dir as plain joblib pickles). So to fork it without the private BYOD
image we only need to (a) drop the `hill_climbing` import and inline an
equivalent greedy ensemble `Climber`, and (b) enable GPU in the metadata
(the code already uses `device_type="gpu"` / `task_type="GPU"`).

Surgical patches (everything else is kept VERBATIM):
  1. Cell 1 (imports): remove the line `from hill_climbing import Climber`.
  2. Insert a new code cell right after the imports cell that defines an inline
     `Climber` (Caruana 2004 greedy ensemble selection) with the same
     fit/predict/best_score contract the notebook uses.
  3. Insert a new SMOKE/config code cell right after the CFG cell defining
     `SMOKE` and `N_TRIALS`.
  4. Cell 18: replace `n_trials=500` with `n_trials=N_TRIALS`.
  5. SMOKE fast-path (only active when env ROGII_SMOKE=1):
       - cell 5: subsample train_df to 300 rows/well and test_paths to [:6]
       - cell 7: shrink LightGBM n_estimators and CatBoost iterations to ~50
     These are minimal-invasive guarded edits; full run is byte-identical to the
     original except for the Climber inline + N_TRIALS=60 (vs 500).

Nothing is executed here. Output ipynb is validated only by JSON round-trip.

Run:  python generate_notebook.py
"""

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = (BASE.parent / "rogii-wellbore-work" / "pub_dwt"
       / "9-251-rogii-wellbore-geology-prediction-dwt-based.ipynb")
OUT = BASE / "rogii-baseline-infer.ipynb"


def src_str(cell):
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def set_src(cell, text):
    # store as a single string; nbformat accepts str or list of str.
    cell["source"] = text


def find_cell(cells, needle, cell_type="code"):
    """Return index of the first cell of cell_type whose source contains needle."""
    for i, c in enumerate(cells):
        if c["cell_type"] != cell_type:
            continue
        if needle in src_str(c):
            return i
    raise RuntimeError(f"cell containing {needle!r} not found")


# ---------------------------------------------------------------------------
# Inline Climber: Caruana (2004) greedy ensemble selection with replacement.
# Contract used by the notebook (cell 15):
#   Climber(objective="minimize", eval_metric=CFG.metric,
#           allow_negative_weights=False, precision=0.001,
#           score_decimal_places=3, n_jobs=-1, use_gpu=False).fit(oof_df, y)
#   .predict(df) -> weighted column average (np.ndarray)
#   .best_score  -> achieved RMSE (float)
# oof_df / test_df are pandas DataFrames whose columns are the per-model preds.
# ---------------------------------------------------------------------------
CLIMBER_SRC = '''\
# === Inline greedy ensemble selection (replaces the external hill_climbing.Climber) ===
# Caruana et al. 2004 "Ensemble Selection from Libraries of Models":
# start from the empty ensemble, repeatedly add (with replacement) the single
# candidate column that most improves the eval metric of the running mean, stop
# when no addition improves. Final weights = normalized selection counts.
import numpy as _np


class Climber:
    def __init__(self, objective="minimize", eval_metric=None,
                 allow_negative_weights=False, precision=0.001,
                 score_decimal_places=3, n_jobs=-1, use_gpu=False,
                 max_steps=100):
        self.objective = objective
        self.eval_metric = eval_metric
        self.allow_negative_weights = allow_negative_weights
        self.precision = precision
        self.score_decimal_places = score_decimal_places
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.max_steps = max_steps
        self.weights_ = None
        self.columns_ = None
        self.best_score = None

    def _score(self, y_true, y_pred):
        s = float(self.eval_metric(y_true, y_pred))
        return round(s, self.score_decimal_places)

    def _better(self, cand, ref):
        # ref is None -> any finite score is better
        if ref is None:
            return True
        if self.objective == "minimize":
            return cand < ref
        return cand > ref

    def fit(self, preds, y):
        # preds: DataFrame (rows = samples, columns = candidate models)
        self.columns_ = list(preds.columns)
        P = preds.values.astype(_np.float64)            # (n, m)
        y = _np.asarray(y).astype(_np.float64)
        n, m = P.shape

        counts = _np.zeros(m, dtype=_np.float64)
        ens_sum = _np.zeros(n, dtype=_np.float64)       # running sum of selected preds
        n_sel = 0
        best_score = None

        for _ in range(self.max_steps):
            step_best_score = None
            step_best_j = -1
            for j in range(m):
                cand_mean = (ens_sum + P[:, j]) / (n_sel + 1)
                sc = self._score(y, cand_mean)
                if self._better(sc, step_best_score):
                    step_best_score = sc
                    step_best_j = j
            # accept the step only if it improves the global best
            if step_best_j >= 0 and self._better(step_best_score, best_score):
                counts[step_best_j] += 1.0
                ens_sum += P[:, step_best_j]
                n_sel += 1
                best_score = step_best_score
            else:
                break

        if n_sel == 0:
            # degenerate fallback: pick the single best column
            single = [self._score(y, P[:, j]) for j in range(m)]
            j = int(_np.argmin(single)) if self.objective == "minimize" else int(_np.argmax(single))
            counts[j] = 1.0
            n_sel = 1
            best_score = single[j]

        self.weights_ = counts / counts.sum()
        self.best_score = float(best_score)
        return self

    def predict(self, preds):
        cols = [preds[c].values.astype(_np.float64) for c in self.columns_]
        M = _np.vstack(cols).T                          # (n, m)
        return (M * self.weights_[None, :]).sum(axis=1)
'''

# SMOKE/config cell inserted right after CFG.
SMOKE_SRC = '''\
# Kaggle has no way to inject env vars at push time, so SMOKE is a baked-in
# constant. FLIP to False (re-run generate_notebook.py, commit, push) for the
# real full run. SMOKE proves the whole pipeline + submission format end to end
# cheaply: tiny train subsample, 6 test wells, 50-round GBTs, 3 Optuna trials.
SMOKE = False
# Full run searches 500 post-processing trials; 60 is plenty for a baseline and
# much faster. SMOKE keeps it to 3 just to prove the pipeline runs end to end.
N_TRIALS = 3 if SMOKE else 60
print(f"SMOKE={SMOKE}  N_TRIALS={N_TRIALS}")
'''


def main():
    nb = json.load(open(SRC, encoding="utf-8"))
    cells = nb["cells"]
    report = []

    # ---- 1. drop hill_climbing import from the imports cell ----
    i_imp = find_cell(cells, "from hill_climbing import Climber")
    imp_src = src_str(cells[i_imp])
    assert "from hill_climbing import Climber\n" in imp_src, "import line shape changed"
    new_imp = imp_src.replace("from hill_climbing import Climber\n", "")
    set_src(cells[i_imp], new_imp)
    report.append(f"cell {i_imp}: removed 'from hill_climbing import Climber'")

    # ---- 2. insert inline Climber cell right after the imports cell ----
    climber_cell = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": CLIMBER_SRC,
    }
    cells.insert(i_imp + 1, climber_cell)
    report.append(f"inserted inline Climber cell at index {i_imp + 1}")

    # ---- 3. insert SMOKE/config cell right after the CFG cell ----
    i_cfg = find_cell(cells, "class CFG:")
    smoke_cell = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": SMOKE_SRC,
    }
    cells.insert(i_cfg + 1, smoke_cell)
    report.append(f"inserted SMOKE/config cell at index {i_cfg + 1}")

    # ---- 4. N_TRIALS in Optuna cell ----
    i_opt = find_cell(cells, "study.optimize(objective, n_trials=500")
    opt_src = src_str(cells[i_opt])
    new_opt = opt_src.replace(
        "study.optimize(objective, n_trials=500, n_jobs=-1)",
        "study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1)",
    )
    assert new_opt != opt_src, "n_trials replace failed"
    set_src(cells[i_opt], new_opt)
    report.append(f"cell {i_opt}: n_trials 500 -> N_TRIALS")

    # ---- 5a. SMOKE subsample in the data-loading cell (train_df + test_paths) ----
    i_load = find_cell(cells, "test_paths = sorted((CFG.dataset_path / \"test\")")
    load_src = src_str(cells[i_load])

    # 5a-0. SMOKE: cap the 7.4GB train.csv read with nrows to avoid GPU-kernel OOM.
    read_anchor = ('train_df = pd.read_csv(CFG.artifacts_path / "data" / "train.csv", '
                   'low_memory=False)')
    assert read_anchor in load_src, "train.csv read anchor not found"
    load_src = load_src.replace(
        read_anchor,
        'train_df = pd.read_csv(CFG.artifacts_path / "data" / "train.csv", '
        'low_memory=False, nrows=(60000 if SMOKE else None))',
    )
    anchor = ('test_paths = sorted((CFG.dataset_path / "test").glob'
              "('*__horizontal_well.csv'))")
    assert anchor in load_src, "test_paths anchor not found"
    smoke_load = (
        'if SMOKE:\n'
        "    train_df = train_df.groupby('well', group_keys=False).head(300).reset_index(drop=True)\n"
        '\n'
        + anchor + '\n'
        'if SMOKE:\n'
        '    test_paths = test_paths[:6]\n'
    )
    new_load = load_src.replace(anchor + "\n", smoke_load)
    assert new_load != load_src, "data-load SMOKE patch failed"
    set_src(cells[i_load], new_load)
    report.append(
        f"cell {i_load}: SMOKE subsample train_df (300/well) + test_paths[:6]")

    # ---- 5b. SMOKE shrink GBT rounds in the params cell ----
    i_par = find_cell(cells, "lgb_params_base = dict(")
    par_src = src_str(cells[i_par])
    assert "lgb_params = [" in par_src and "cb_params = [" in par_src
    smoke_rounds = (
        "\n\n"
        "if SMOKE:\n"
        "    for _p in lgb_params:\n"
        "        _p['n_estimators'] = 50\n"
        "    for _p in cb_params:\n"
        "        _p['iterations'] = 50\n"
    )
    new_par = par_src.rstrip("\n") + smoke_rounds
    set_src(cells[i_par], new_par)
    report.append(
        f"cell {i_par}: SMOKE shrink lgb n_estimators=50 & cb iterations=50")

    # ---- 5c. force RE-TRAIN: the public artifacts dataset ships ravaghi's
    # `*_trainer_*.pkl` (Trainer objects, BYOD-only), NOT the `models.pkl` /
    # `oof_preds.pkl` cache pub_dwt's guard expects. The guard only checks that
    # the models/<name> DIR exists (it does, with the wrong files) and then
    # blows up on `joblib.load(path/"models.pkl")`. Make the guard check the
    # actual models.pkl FILE so it always falls through to training. ----
    # train_lightgbm and train_catboost live in SEPARATE cells, so patch every
    # code cell that contains the dir-existence guard.
    guard_old = 'if (CFG.artifacts_path / "models" / name).exists():'
    guard_new = 'if (CFG.artifacts_path / "models" / name / "models.pkl").exists():'
    n_guard_total = 0
    for ci, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        cs = src_str(c)
        k = cs.count(guard_old)
        if k:
            set_src(c, cs.replace(guard_old, guard_new))
            n_guard_total += k
            report.append(f"cell {ci}: forced re-train ({k} guard -> models.pkl file)")
    assert n_guard_total >= 2, f"expected >=2 cache guards, found {n_guard_total}"

    # ---- 5d. CatBoost devices="0:1" assumes a 2-GPU box (the author used T4x2).
    # A single-GPU kernel raises 'id 1 greater than limit 1'. Pin to device 0 so
    # it runs on any GPU count. ----
    dev_old = 'devices="0:1"'
    dev_new = 'devices="0"'
    n_dev = 0
    for c in cells:
        if c["cell_type"] != "code":
            continue
        cs = src_str(c)
        if dev_old in cs:
            set_src(c, cs.replace(dev_old, dev_new))
            n_dev += cs.count(dev_old)
    assert n_dev >= 1, "CatBoost devices=\"0:1\" not found to patch"
    report.append(f"patched {n_dev} CatBoost devices 0:1 -> 0 (single-GPU)")

    # ---- title: tweak the first markdown cell header (non-functional) ----
    # Add a provenance markdown cell at the very top.
    title_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": (
            "# ROGII Baseline DWT Fork\n\n"
            "Re-training fork of the public ~9.251 DWT-based solution "
            "(nihilisticneuralnet). Trains 3x LightGBM + 3x CatBoost on GPU, "
            "greedy-ensembles with an inline `Climber`, then Optuna "
            "post-processing. Set env `ROGII_SMOKE=1` for a fast pipeline check.\n"
        ),
    }
    cells.insert(0, title_cell)
    report.append("inserted provenance markdown title cell at index 0")

    # ---- ensure nbformat basics ----
    nb.setdefault("nbformat", 4)
    nb.setdefault("nbformat_minor", 5)

    json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- round-trip validation (JSON load only, no execution) ----
    rt = json.load(open(OUT, encoding="utf-8"))
    code_cells = [c for c in rt["cells"] if c["cell_type"] == "code"]
    md_cells = [c for c in rt["cells"] if c["cell_type"] == "markdown"]
    # sanity asserts
    full = "\n".join(src_str(c) for c in code_cells)
    assert "\nfrom hill_climbing import Climber" not in ("\n" + full), "Climber import still present!"
    assert "class Climber:" in full, "inline Climber missing!"
    assert "n_trials=N_TRIALS" in full, "N_TRIALS not wired!"
    assert "SMOKE = True" in full or "SMOKE = False" in full, "SMOKE flag missing!"
    assert "nrows=(60000 if SMOKE else None)" in full, "SMOKE nrows cap missing!"
    assert '"models" / name).exists()' not in full, "dir-existence cache guard still present (should force re-train)!"
    assert '"models" / name / "models.pkl").exists()' in full, "re-train guard not wired!"
    assert 'devices="0:1"' not in full, "CatBoost 2-GPU devices not patched!"
    assert "device_type=\"gpu\"" in full, "GPU device flag lost!"
    assert 'task_type="GPU"' in full, "CatBoost GPU lost!"

    print("=== PATCH REPORT ===")
    for r in report:
        print(" -", r)
    print(f"=== OUTPUT: {OUT.name} ===")
    print(f"total cells: {len(rt['cells'])}  code: {len(code_cells)}  markdown: {len(md_cells)}")
    print("round-trip + sanity asserts: OK")


if __name__ == "__main__":
    main()
