# Kaggle Competitions

Kaggle Notebooks Expert. 14 Bronze Notebook Medals + active competition participation.

**Note:** Notebook Medals are earned through community votes on shared notebooks - NOT competition ranking medals.

---

## ☁️ GitHub-Driven Workflow (GitHub Actions + Kaggle API)

Manage all notebook code in GitHub — version control, diff, CI, and auto-deploy to Kaggle via `git push`.

```
Edit in VSCode / any editor → git push → GitHub Actions / RPi5 → kaggle kernels push → Auto-submit via API
```

### Why GitHub Instead of Kaggle's Browser Editor?

- **Version control**: Full git history, branching, diff for every change
- **Editor freedom**: Use VSCode, Vim, or any editor instead of Kaggle's browser UI
- **CI/CD**: GitHub Actions automates deployment to Kaggle — no manual upload
- **Secrets management**: API keys stored in GitHub Secrets, not local files
- **Multi-device**: Push from any machine with git

### How It Works

1. Edit `.ipynb` in your preferred editor
2. `git push` to this repository
3. Trigger: `gh workflow run kaggle-push.yml -f notebook_dir=<dir>`
4. GitHub Actions runs [`kaggle kernels push`](.github/workflows/kaggle-push.yml) to upload the notebook
5. Auto-submit via Kaggle API (`competition_submit` with kernel output download)

### RPi5 Self-Hosted Runner

For long-running notebooks that exceed GitHub Actions' 90-min timeout, RPi5 runs `kaggle-submit.sh` with:
- No timeout limit (6h polling)
- GPU → CPU auto-fallback (any GPU failure triggers CPU retry, not just quota errors)
- Auto-submission via `kaggle.api.competition_submit()` (download output CSV → submit)
- Score polling → W&B recording → Discord notification

### Key Findings

- **`enable_internet: false`** is required for code competition submissions — Internet ON prevents the notebook from being eligible
- **`competition_sources`** mounts data at `/kaggle/input/competitions/<slug>/` (not `/kaggle/input/<slug>/`)
- **Submission differs by competition type (verified 2026-06 on an active comp):** for **code competitions** the API path is blocked both ways — `CreateCodeSubmission` returns **403** and file submission (`kaggle competitions submit -f`) returns **400** (the upload is rejected and leaves no entry). Code-competition submission must go through the notebook **"Submit to Competition"** UI. File submission (`submit -f`) works only for **regular** competitions (downloadable test + `sample_submission`, not flagged "code competition"), e.g. S6E6 / BirdCLEF.

**Blog post:** [DEV.to](https://dev.to/yasumorishima/kaggle-code-competitions-without-a-local-gpu-github-actions-kaggle-api-cloud-workflow-m3)

---

## 🔬 Experiment Management (EXP + child-exp)

### Credit & Origin

The experiment management methodology (EXP + child-exp structure, role division, EXP_SUMMARY.md, CLAUDE_COMP.md) is inspired by [chimanさんの記事](https://zenn.dev/chiman/articles/b233cc808d6af3) — a write-up on winning a Kaggle gold medal using Claude Code / Codex.

The following parts are our own design, built to work in a GPU-less local environment (Celeron N4500 / 4GB RAM):

| Component | Origin |
|---|---|
| EXP + child-exp directory structure | chimanさんの記事 |
| Role division (ideas = human, implementation = AI) | chimanさんの記事 |
| EXP_SUMMARY.md (experiment history as AI guardrail) | chimanさんの記事 |
| CLAUDE_COMP.md (competition-specific AI guardrails) | chimanさんの記事 |
| RPi5 xrdp remote desktop (session persists on disconnect) | 独自設計 |
| xdotool keepalive (systemd, 30min interval) | 独自設計 |
| Google Drive for Desktop ↔ Colab sync | 独自設計 |
| Colab file monitor notebook (auto-detect & run) | 独自設計 |

### Architecture

```
[Local PC]                  [RPi5 (xrdp)]               [Google Colab (Free)]
Claude Code                 Remote Desktop session       Experiment Runner v4
  ↓ Write config/code        ↓ RDP for Colab setup        ↓ Auto-run train.py
  ↓                          ↓ Session persists on DC      ↓
Google Drive (for Desktop) ←――――――――――――――――――→ Google Drive (mount)
  EXP/config/child-exp005.yaml                          Detect new config → execute
  EXP/output/child-exp005/result.json                   Save results to Drive
```

- **RPi5 xrdp** provides remote desktop for Colab setup; session persists after disconnect
- **xdotool keepalive** (systemd service) sends keystrokes every 30 min to prevent Colab idle timeout
- **Claude Code** writes experiment configs to Google Drive
- **Colab** auto-detects new configs and runs `train.py`
- **[colab-mcp](https://github.com/googlecolab/colab-mcp)** enables direct Colab GPU interaction from Claude Code via MCP. PC起動中かつColabノートブックをブラウザで開いている間のみ利用可能（WebSocket接続のため）
- **Kaggle kernels** used only for final submission

### Directory Structure

```
<comp-slug>/
├── CLAUDE_COMP.md              # Competition-specific guardrails for AI
├── EXP_SUMMARY.md              # Experiment history (AI memory)
├── docs/Idea_Research/         # Hypothesis memos, Deep Research results
├── EXP/
│   ├── EXP001/
│   │   ├── train.py
│   │   ├── config/
│   │   │   ├── child-exp000.yaml  # Baseline
│   │   │   ├── child-exp001.yaml  # + feature engineering
│   │   │   └── child-exp002.yaml  # + loss change
│   │   └── output/
│   │       ├── child-exp000/
│   │       │   ├── oof.csv
│   │       │   └── result.json
│   │       └── ...
│   └── EXP002/                 # New EXP when pipeline changes significantly
└── submit/                     # Final Kaggle kernel for submission
```

### Role Division

| Human | AI (Claude Code) |
|---|---|
| Hypotheses & ideas | Implementation (train.py, features) |
| CV design decisions | OOF error analysis & visualization |
| Interpret results → next action | Config generation & experiment tracking |
| Domain knowledge | Notebook/Discussion summarization |

### Key Principles

- **Never ask AI "improve the score"** — provide specific hypotheses
- **EXP_SUMMARY.md is a guardrail**, not a strategy generator
- **1 competition, 50+ experiments** — depth over breadth
- **OOF analysis by AI, next action by human**

### Google Drive ↔ GitHub Actions Integration

Experiment results on Google Drive are synced to GitHub Actions for W&B recording and Kaggle/SIGNATE/DrivenData submission.

```
Google Drive (EXP results)
  ↓ Google Drive API (Service Account)
GitHub Actions
  ├── EXP W&B Sync      → Record cv_score, config to W&B
  └── EXP to Kaggle     → Fetch best model → Push to Kaggle kernel
```

#### Setup (for your own fork)

1. Create a Google Cloud project (free) and enable the **Google Drive API**
2. Create a **Service Account** → download JSON key
3. Share your Drive experiment folder with the service account email (Viewer permission)
4. Add GitHub Secrets to your repository:
   - `GOOGLE_SERVICE_ACCOUNT_KEY`: Service account JSON key contents
   - `DRIVE_SHARED_FOLDER_ID`: Google Drive folder ID (from the folder URL)

No billing required — Google Drive API is free within standard quotas.

#### Usage

```bash
# Sync experiment results to W&B
gh workflow run "EXP W&B Sync" \
  -f comp=s6e3-churn -f exp=EXP001 -f memo="sync all results"

# Submit best experiment to Kaggle
gh workflow run "EXP to Kaggle Submit" \
  -f comp=s6e3-churn -f exp=EXP001 -f child=child-exp005 \
  -f notebook_dir=playground-series-s6e3-work \
  -f kernel_id=yasunorim/s6e3-churn-optuna-stacking-work \
  -f memo="best config submit"
```

---

## 🏆 Competition Results

### Santa 2026 - The Elf Calculator (Active)

**Competition:** [Santa 2026](https://www.kaggle.com/competitions/santa-2026) | **Deadline:** 2027-01-28 | **Prize:** $50,000 | **Metric:** total instruction count over 1,000 puzzles (lower is better)

Each puzzle asks for the shortest program, on 48 registers with `+ - * /`, that produces every target number from the starting values. Scoring is exact, so one instruction saved is one point. Code is in [`santa-2026/`](santa-2026). The programs themselves live only in a private Kaggle dataset; this repository holds the search code and the scores.

**How it runs.** A cycle in GitHub Actions (`santa-kaggle.yml`) does five things: it merges the previous pool with the outputs of the search shards and with the public program sets (each program is replayed by our own referee), updates the private dataset, submits when the pool beats the best submission (at most 5 a day), and pushes new shards. The search runs on 5 Kaggle CPU kernels and 16 public GitHub runners (`santa-gha-search.yml`). The runners' results come back encrypted with a key that only the private dataset holds. Our runner (`search/runner.py`) schedules the work: puzzles that just got shorter go first, and equal-length variants are kept as new starting points.

**Where it stands (2026-11-25).**
- Best submission **128,037**: **67 instructions below the merged public frontier** (128,104 at that time).
- Merging the public sets alone only ever tied the frontier, and about 175 teams shared that tie.
- Our runs of the public engines (Wang Jakie's graph search, Atakan Aldemir's optimizers; Apache-2.0) saved **0 instructions in about 900 core-hours**. The programs had already been polished by every team running the same engines.
- What worked was **triwalk**, goodpjw2008's annealer over the *set of values* a program computes (Apache-2.0, vendored unchanged with its NOTICE). In one GitHub run (16 runners × 320 min) it shortened 96 of 12,500 jobs (85 puzzles, 104 instructions). The public engines shortened 0 of 3,159 jobs in the same run.
- The search plan is now triwalk only: 60 s runs on settled puzzles, 40 s runs on puzzles that just changed.

### Generalist Gaming Agent (Active)

**Competition:** [Generalist Gaming Agent](https://www.kaggle.com/competitions/generalist-gaming-agent) | **Deadline:** 2027-03-01 | **Prize:** $50,000 | **Metric:** mean normalized score over unseen browser games

A single agent plays games it has never seen, from JPEG frames only, with mouse and keyboard, while the game keeps running at 20 fps. Code is in [`gga/`](gga). Kernels are pushed and submitted from a request file (`gga/requests/kaggle.json` → `gga-kaggle.yml`).

**Where it stands (2026-11-25).**
- Best **0.046** (v2) from a rule-based agent with no model. It reads the game description, presses menu buttons, reopens a stalled screen and favours actions that change the screen.
- v4 added clicks aimed at rare colours and scored **0.041**, so those clicks were removed.
- The plan is to add a slow LLM layer (what to do) over a fast rule layer (how to react), run on the free Kaggle TPU with JAX.
- The TPU image was probed: Python 3.12, jax 0.11, protobuf 7. Missing wheels (grpcio-tools 1.85.1, polars, pyarrow, pydantic, opentelemetry) are installed from a private dataset at kernel start.
- The first TPU submission has been queued for more than 11 hours.

### enveda CASMI26 - Molecule Identification from MS/MS (Active)

**Competition:** [enveda-CASMI26](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra) | **Deadline:** 2026-12-14 | **Prize:** $50,000 | **Metric:** MRR@25 on InChIKey (first 14 characters)

Given MS/MS spectra of a molecule, rank up to 25 candidate structures. Hidden test molecules fall into three classes: spectra already in a public library, structure in PubChem/COCONUT but no public spectrum, and novel structures (about half the test, untouched by every public notebook). Submissions are offline notebooks, so the repository drives them through GitHub Actions from a request file (`enveda-casmi26/requests/kaggle.json` → kernel push → submit → result file back on the branch). Code and the running log are in [`enveda-casmi26/`](enveda-casmi26) and [`enveda-casmi26/CLAUDE.md`](enveda-casmi26/CLAUDE.md).

**Where it stands (2026-09-27).** Best public score **0.283** (b3). The candidate pool is train structures plus COCONUT within ±10 ppm of the neutral mass; each candidate is scored by analog propagation — spectral neighbours found by FlashEntropy hybrid search, weighted by similarity and multiplied by the candidate's Tanimoto similarity to each neighbour — plus a library hit only when it is near-identical (≥ 0.8). b3 sums each candidate's three best analogs instead of taking the single best (Morgan radius-3 counts, 400 analogs): 0.275 → 0.283.

**The local validation had to be rebuilt before it could judge anything.** Holding out random molecules from the timsTOF library gave a class-2 score near 0.80, while the leaderboard implies about 0.47. Removing every analog closer than Tanimoto 0.5 to the truth still left 0.66, so near-duplicates were not the cause; the kind of molecule was. Using the 250 natural products shipped with the data as class 2 (all their spectra removed from the library, structures kept in the pool) gives **0.526** for the submitted method — on the leaderboard's scale. One lesson came with it: that set agrees with the leaderboard on class-2 changes but not on the library gate (0.95 looked +0.044 locally and lost 0.004 on the board), so gate settings stay where the board put them. A spectrum-to-fingerprint MLP and a MetFrag-style fragment-explanation score both lowered the natural-product score when blended in, and are parked until they are stronger on their own.

### Kaggriculture (Ended 2026-09-30)

**Competition:** [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) | **Deadline:** 2026-09-30 | **Prize:** $50,000 | **Metric:** ladder rating

Two players run neighbouring farms for 30 days of 24 turns, buying land, hiring hands, sowing, tending livestock and selling into a shared market. No GPU is involved, so a whole season simulates in 3–4 seconds and the bottleneck is experiment design rather than compute. Code lives in [`kaggriculture/`](kaggriculture); every episode runs in GitHub Actions (`Kaggriculture Eval` / `Analyze` / `Sweep` / `Record` / `Optimize` / `Schedule`), never locally.

**Where it stands (2026-09-27).** The two submissions carried into the final tournament are `agents/v70_sched.py` and `agents/v69_sched.py`. Each was submitted only after beating the previous submission head to head: v69 beats v68 by **+1,720 ± 799** over 192 paired games (win rate 0.63) and held **+1,641** on a fresh seed band; v70 (v69 keeping at least ten wheat in stock) beats v69 by **+841 ± 586** and held **+1,330**. Both tie the public router over 48 games. v69 is itself two changes that each read as a tie alone — selling fertilizer from day 5 and selling wool whole — and beat v68 only together, so stacking small ties at 192 games is now part of the method. Two further sweeps on v70 (wheat, fertilizer, feed, milk and rush settings) found nothing better: the existing knobs are at a local optimum. Router checks run in the cloud container against the public router file; twin sweeps and submissions run in GitHub Actions from request files.

**The score is a rating, not money.** Downloading the leaderboard shows the field spans −129 to 3,172, so an agent's own final cash says nothing about where it stands; only paired head-to-head play does.

Earlier updates (2026-09-16 and 2026-09-24), the rating analysis (2026-09-17), what the environment pays for, the measurement tools in `kaggriculture/sim/` and the full list of findings are kept verbatim in [`docs/history.md`](docs/history.md#kaggriculture).

### ROGII - Wellbore Geology Prediction (Closed 2026-08-05)

**Competition:** [ROGII - Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction) | **Deadline:** 2026-08-05 | **Prize:** $50,000 | **Metric:** RMSE

Predict True Vertical Thickness (TVT) along horizontal wellbores to automate geosteering. Per-well data (`*__horizontal_well.csv` logs with `[MD, X, Y, Z, GR, TVT_input]` + `*__typewell.csv` reference stratigraphy); predict the unrevealed "toe" rows (`TVT_input` is NaN, ~3,800 of ~5,300 rows per well) as a **delta from the last known TVT**. **Code competition** — submission is by notebook "Submit to Competition", not file upload.

| Approach | Score (RMSE, lower better) |
|---|---|
| Leaderboard top | 4.859 (public LB, 2026-07-16) |
| Medal lines (5,063 teams, 2026-07-16) | gold 6.118 / silver 7.043 / bronze 7.084 (public LB) |
| **Dual-pipeline fork — sp45 + fleongg blend + guarded override (best submitted)** | **7.311 (public LB — rank 480 / 3,589 when submitted 2026-06; 1,406 / 5,063 as of 2026-07-16)** |
| Public best (DWT/DTW-based clone group, *claimed*) | ~9.25 (public LB) |
| Champion — surface + dip-beam aligner (this repo) | 9.978 (leak-free GroupKFold CV; not LB-submitted) |
| GBT + TCN ensemble (this repo) | 9.905 (public LB) |
| Re-training fork baseline — GBT only | 10.224 (public LB) |

The development log (approaches tried, the 2026-06-25 strategy pivot and the `kaggle kernels push` ops trap) is kept verbatim in [`docs/history.md`](docs/history.md#rogii---wellbore-geology-prediction).

---

### Playground Series S6E6 - Stellar Classification (Finished 2026-06-30)

**Competition:** [Playground Series S6E6](https://www.kaggle.com/competitions/playground-series-s6e6) | **Deadline:** 2026-06-30 (finished) | **Final:** public 0.95944 / **private 0.95939**

3-class classification of astronomical objects (GALAXY 65% / QSO 20% / STAR 14%, SDSS-style). 577k train rows, 10 features (2 categorical). Label submission (`[id, class]`).

| Approach | Public LB |
|---|---|
| **Two-stage pseudo-label distillation (153k confident test rows join fold-train)** | **0.95944** |
| Full ensemble (LGB + XGB + CatBoost, 5-fold ES) + pairwise-diff features + per-class probability weights | 0.95884 |
| Full ensemble + per-class probability weights (no diff features) | 0.95866 |
| + original SDSS17 dataset in fold-train (external data) | 0.95741 (rejected) |
| LGB-only 3-fold smoke baseline | 0.95466 |

- **Metric finding:** the LB metric is macro-F1 and OOF macro-F1 matches it within ~0.001, so per-class probability weights tuned on OOF by coordinate ascent are a proven lever (+0.004 LB over plain argmax; balanced class weights alone hurt argmax but win after weight tuning).
- **Feature lever:** pairwise differences of all numeric columns (generalized color indices), validated by an A/B smoke (+0.00093 OOF) before spending a full run (+0.0002 LB).
- **External-data lesson:** appending the original SDSS17 dataset to the training side of each fold RAISED OOF (+0.0002, near-duplicates of validation rows leak in) but DROPPED LB (-0.0014). OOF deltas only predict LB deltas while the training distribution stays unchanged.
- **Pseudo-label lever (proven):** test rows predicted with 0.995-plus confidence join the fold-train side in a second stage: +0.0006 LB while STAGE2 OOF only rose +0.00025 - test-distribution alignment gains do not show in OOF.
- **Levers exhausted (champion holds at 0.95944):** pseudo round 2 (LB 0.95907), MLP/NN diversity blend (NN OOF far below GBT, every blend weight worse), physics interaction features (OOF -0.0003), and an Optuna-tuned LGB ported into the full ensemble (STAGE2 OOF 0.95790 < champion 0.95796) were each tested and rejected - GBT ceiling confirmed. The proxy HPO gain (+0.00068 on a 3-fold LGB-only proxy) did not survive the 5-fold LGB+XGB+CatBoost ensemble. OOF / test-probability .npy artifacts stay persisted (kernel_sources) for any future stacking.

- **Final result (private revealed):** the champion scored **private 0.95939** vs public 0.95944 — a gap of only 0.00005, and it was our best submission on both splits, so Kaggle's default "best public" auto-selection was the correct pick. The metric finding held end to end: with macro-F1 and no distribution shift, OOF ≈ public ≈ private, which is why every OOF-validated lever above transferred (and why the external-data lever, the one that *did* change the training distribution, was the only one whose OOF gain inverted on the LB).
- **Pipeline:** `playground-series-s6e6/generate_notebook.py` → `kaggle-push.yml` (GitHub Actions) → Kaggle kernel. A `SMOKE` flag validates the plumbing (LGB-only, 3-fold) before the full ensemble (1 seed x 5 folds).
- **Gotcha confirmed:** `competition_sources` mounts data at `/kaggle/input/competitions/<slug>/` (the notebook auto-discovers the dir via `os.walk`).

---

### BirdCLEF+ 2026 - Acoustic Species Identification (Ended)

**Competition:** [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) | **Deadline:** 2026-06-03 (ended)

Identify 234 species (birds, insects, amphibians, reptiles) from audio recordings in the Pantanal, Brazil. Evaluated with macro ROC AUC. **Final: private 0.92187 / public 0.92685** (improved-ensemble fork, 2026-04-18); no further submissions after 2026-04-23.

| Approach | LB |
|---|---|
| improved-ensemble fork (Perch v2 + ProtoSSM v5 + ResidualSSM + TTA + rank-aware + delta smooth) | **0.926** |
| Perch v2 + Bayesian prior + LogReg probe (fork) | 0.908 |
| public-blend-v6 fork (lb862 + lb872 blend) | 0.890 |
| eca_nfnet_l0 mel baseline (single fold0, CV 0.969) | 0.768 |
| BEATs-SED + Attention Pooling (archived) | 0.745 |

- **Strategy notes from 2026-04-23 and the external-data pipeline:** kept verbatim in [`docs/history.md`](docs/history.md#birdclef-2026).
- **Prior-years solution survey:** [PRIOR_YEARS_SOLUTION_SURVEY.md](birdclef-2026-work/docs/PRIOR_YEARS_SOLUTION_SURVEY.md) — 2024 3rd (jfpuget/TheoViel) + 2025 2nd (VSydorskyy) + 2025 5th (myso1987) writeups distilled into actionable gap list.
- **Competition rules confirmed:** External data allowed under Section 2.6 (publicly available + equally accessible). Xeno-Canto, iNaturalist, past BirdCLEF data all eligible. No pre-deadline disclosure thread obligation in 2026 rules.
- **Notebooks:**
  - `birdclef-2026-improved-ensemble-fork` (CPU): yuriygreben claimed LB 0.929 fork — LB **0.926** (current best, reproducibility variance -0.003)
  - `birdclef-2026-perch-v2-repro` (CPU): Perch v2 fork — LB 0.908
  - `birdclef-2026-public-blend-v6-fork` (CPU): public blend reproduction — LB 0.890 (below our base, archived)
- **Archived:** BEATs (0.745), nfnet_l0 from-scratch (0.768) — ruled out before reproduction base was reached; from-scratch architectures stay blocked, proven-stacking techniques are active.

---

### Deep Past Challenge - Akkadian to English Translation (Ended)

**Competition:** [Deep Past Initiative Machine Translation](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)

Ancient cuneiform (Akkadian) transliteration → English translation task. Evaluated with BLEU + chrF++.

**Notebook:** [Deep Past Cloud Workflow + TF-IDF Baseline](https://www.kaggle.com/code/yasunorim/deep-past-cloud-workflow-tfidf-baseline) *(public)* 🥉

| Approach | Public Score |
|---|---|
| TF-IDF char n-gram nearest neighbor | 5.6 |

- **Approach:** Character n-gram TF-IDF (2-5), cosine similarity nearest neighbor
- Pushed via GitHub Actions cloud workflow (see above)

---

### S6E2 - Predicting Heart Disease (Ended)

**Competition:** [Playground Series S6E2](https://www.kaggle.com/competitions/playground-series-s6e2) | **Deadline:** 2026-02-28

Binary classification (Presence / Absence) with AUC-ROC evaluation.

**Notebook:** [S6E2 Heart Disease - EDA & Ensemble](https://www.kaggle.com/code/yasunorim/s6e2-heart-disease-eda-ensemble-wandb)

| Model | CV AUC |
|---|---|
| **Ensemble (avg)** | **0.95528** |
| CatBoost | 0.95524 |
| LightGBM | 0.95515 |
| XGBoost | 0.95513 |

- **LB Score:** 0.95337
- **Approach:** LightGBM + XGBoost + CatBoost (GPU), 5-fold Stratified CV, 6 interaction features
- **Blog:** [Zenn](https://zenn.dev/shogaku/articles/kaggle-s6e2-github-wandb-gpu-workflow)

**Tech Stack:** LightGBM, XGBoost, CatBoost, W&B, GPU

---

<details>
<summary><h2>🥉 Bronze Medal Notebooks (14)</h2></summary>

### 1. CAFA 6 - Protein Function Prediction

**Notebook:** [Baseline with Regularization](https://www.kaggle.com/code/yasunorim/baseline-with-regularization)

Multi-label classification of protein functions using Gene Ontology (GO) terms.

**Approach:**
- TF-IDF k-mer features (3-grams) from amino acid sequences
- MLP with regularization (Dropout 0.5, Weight Decay, Early Stopping, BatchNorm)
- 1500 GO terms across 3 aspects (Biological Process, Molecular Function, Cellular Component)
- GO hierarchy propagation

**Tech Stack:** PyTorch, scikit-learn, pandas, numpy

---

### 2. NFL Big Data Bowl 2026 - Prediction

**Notebook:** [Geometric Rules Baseline - 2.921 RMSE (No ML)](https://www.kaggle.com/code/yasunorim/geometric-rules-baseline-2-921-rmse-no-ml)

Sports analytics using NFL player tracking data.

**Approach:**
- Physics-based geometric rules (no ML)
- Targeted receivers → direct path to ball landing point
- Defensive coverage → distance-based offset from receivers

**Performance:** RMSE 2.921 yards, <5 seconds execution

**Tech Stack:** Python, pandas, polars, numpy

---

### 3. PhysioNet - Digitization of ECG Images

**Notebook:** [PhysioNet ECG Baseline](https://www.kaggle.com/code/yasunorim/physionet-ecg-baseline)

Submission format guide for ECG image digitization challenge.

**Key Contributions:**
- Correct submission format documentation
- Common mistakes and how to avoid them
- Working baseline with verified format

**Tech Stack:** Python, pandas, numpy

---

### 4. Diabetes Prediction (S5E12) - EDA & Baseline

**Notebook:** [Diabetes Prediction - EDA & Baseline](https://www.kaggle.com/code/yasunorim/diabetes-prediction-eda-baseline-s5e12)

Comprehensive EDA and LightGBM baseline. CV AUC 0.72687.

**Tech Stack:** Python, pandas, LightGBM, scikit-learn, matplotlib, seaborn

---

### 5. Diabetes Prediction (S5E12) - Rank-Based Ensemble

**Notebook:** [Diabetes Prediction - Rank-Based Ensemble](https://www.kaggle.com/code/yasunorim/diabetes-prediction-rank-based-ensemble)

Rank-based blending with dual LightGBM models. Blended OOF AUC 0.72716.

**Tech Stack:** Python, pandas, LightGBM, scikit-learn

---

### 6. MLB Statcast - Senga's Ghost Fork (2023-2025)

**Notebook:** [Senga Ghost Fork Analysis](https://www.kaggle.com/code/yasunorim/senga-ghost-fork-analysis-2023-2025)

Statcast data analysis of Kodai Senga's forkball ("Ghost Fork") across 3 seasons. Movement comparison, release point analysis (FF vs FO), batter splits, and performance by batting order.

**Tech Stack:** Python, pybaseball, DuckDB, matplotlib, seaborn

---

### 7. MLB Statcast - Kikuchi's Slider Revolution (2019-2025)

**Notebook:** [Kikuchi Slider Revolution](https://www.kaggle.com/code/yasunorim/kikuchi-slider-revolution-2019-2025)

Statcast data analysis of Yusei Kikuchi's pitching evolution from Mariners to Blue Jays to Astros. Pitch mix changes, slider usage trends, and movement analysis across 7 seasons.

**Tech Stack:** Python, pybaseball, DuckDB, matplotlib, seaborn

---

### 8. MLB Bat Tracking - Japanese MLB Batters (2024-2025)

**Notebook:** [Bat Tracking: Japanese MLB Batters (2024-2025)](https://www.kaggle.com/code/yasunorim/bat-tracking-japanese-mlb-batters-2024-2025)

MLB bat speed and swing metrics analysis for Japanese MLB batters using Baseball Savant bat tracking data.

**Tech Stack:** Python, pandas, matplotlib, seaborn

---

### 9. March Machine Learning Mania 2026 - Baseline

**Notebook:** [March Machine Learning Mania 2026 Baseline](https://www.kaggle.com/code/yasunorim/march-machine-learning-mania-2026-baseline)

NCAA basketball tournament prediction using historical game data.

**Approach:**
- LightGBM + Logistic Regression ensemble
- Feature engineering from seed differences and historical win rates
- Brier score optimization

**Tech Stack:** Python, LightGBM, scikit-learn, pandas

---

### 10. WBC 2026 Scouting - MLB Statcast Spray Charts

**Notebook:** [MLB Statcast Spray Charts for WBC 2026 Players](https://www.kaggle.com/code/yasunorim/mlb-statcast-spray-charts-for-wbc-2026-players)

Spray charts and pitch zone charts for WBC 2026 players using Baseball Savant Statcast data and baseball-field-viz.

**Approach:**
- Spray charts by batter (hit direction + distance)
- Pitch zone charts by pitcher (location heatmaps)
- Visualization using baseball-field-viz (self-published PyPI package)

**Tech Stack:** Python, pybaseball, baseball-field-viz, matplotlib

---

### 11. Deep Past Challenge - Cloud Workflow + TF-IDF Baseline

**Notebook:** [Deep Past Cloud Workflow + TF-IDF Baseline](https://www.kaggle.com/code/yasunorim/deep-past-cloud-workflow-tfidf-baseline)

Akkadian cuneiform transliteration → English translation baseline using TF-IDF character n-grams. Demonstrates GitHub Actions cloud workflow for Kaggle code competitions.

**Approach:**
- Character n-gram TF-IDF (2-5), cosine similarity nearest neighbor
- Fully managed via GitHub Actions (`git push` → Kaggle)

**Tech Stack:** Python, scikit-learn, pandas

---

### 12. Titanic - Japanese Optuna Test

**Notebook:** [Titanic Japanese Optuna Test](https://www.kaggle.com/code/yasunorim/titanic-japanese-optuna-test)

Titanic survival prediction with Optuna hyperparameter optimization. Japanese-language notebook demonstrating automated tuning workflow.

**Tech Stack:** Python, Optuna, LightGBM, scikit-learn, pandas

---

### 13. Matplotlib & Seaborn 日本語化テンプレート

**Notebook:** [【日本語化】Matplotlib & Seaborn 文字化け解消テンプレート](https://www.kaggle.com/code/yasunorim/matplotlib-seaborn)

Kaggle環境でMatplotlibとSeabornの日本語フォント文字化けを解消するテンプレートノートブック。

**Tech Stack:** Python, matplotlib, seaborn

</details>

---

<details>
<summary><h2>📓 Study Notes (5)</h2></summary>

Located in [`study-notes/`](./study-notes/).

| # | Competition | Notebook | Blog |
|---|---|---|---|
| 1 | Titanic | [Kaggle](https://www.kaggle.com/code/yasunorim/a-journey-to-0-789-with-feature-engine-optuna) | [解説](./study-notes/01-feature-engine-optuna.md) |
| 2 | House Prices | [Kaggle](https://www.kaggle.com/code/yasunorim/japanese-stacking-feature-engineering-guide) | [解説](./study-notes/02-stacking-feature-engineering.md) |
| 3 | Spaceship Titanic | [Kaggle](https://www.kaggle.com/code/yasunorim/japanese-spaceship-titanic) | [解説](./study-notes/03-spaceship-titanic.md) |
| 4 | Commodity Prediction | [Kaggle](https://www.kaggle.com/code/yasunorim/forward-looking-target-fix) | [解説](./study-notes/04-forward-looking-target-fix.md) |
| 5 | LLM Classification | [Kaggle](https://www.kaggle.com/code/yasunorim/japanese-llm-classification) | [解説](./study-notes/05-llm-classification.md) |

</details>

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **ML Libraries** | LightGBM, XGBoost, CatBoost, PyTorch, scikit-learn |
| **Data Processing** | pandas, numpy, polars |
| **Visualization** | matplotlib, seaborn |
| **Experiment Tracking** | Weights & Biases |
| **CI/CD** | GitHub Actions, Kaggle API |
| **Development** | Claude Code, Jupyter Notebook |

---

**Kaggle:** [@yasunorim](https://www.kaggle.com/yasunorim)

*Built with Claude Code*