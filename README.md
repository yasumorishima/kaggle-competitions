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

### ROGII - Wellbore Geology Prediction (Active)

**Competition:** [ROGII - Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction) | **Deadline:** 2026-08-05 | **Prize:** $50,000 | **Metric:** RMSE

Predict True Vertical Thickness (TVT) along horizontal wellbores to automate geosteering. Per-well data (`*__horizontal_well.csv` logs with `[MD, X, Y, Z, GR, TVT_input]` + `*__typewell.csv` reference stratigraphy); predict the unrevealed "toe" rows (`TVT_input` is NaN, ~3,800 of ~5,300 rows per well) as a **delta from the last known TVT**. **Code competition** — submission is by notebook "Submit to Competition", not file upload.

| Approach | Score (RMSE, lower better) |
|---|---|
| Leaderboard top | ~5.35 (public LB) |
| Medal lines (3,589 teams) | gold 6.696 / silver 7.212 / bronze 7.256 (public LB) |
| **Dual-pipeline fork — sp45 + fleongg blend + guarded override (best submitted)** | **7.311 (public LB, rank 480 / 3,589, top 13.4%)** |
| Public best (DWT/DTW-based clone group, *claimed*) | ~9.25 (public LB) |
| Champion — surface + dip-beam aligner (this repo) | 9.978 (leak-free GroupKFold CV; not LB-submitted) |
| GBT + TCN ensemble (this repo) | 9.905 (public LB) |
| Re-training fork baseline — GBT only | 10.224 (public LB) |

- **Anti-fork wall:** the public ~9.25 notebooks depend on a private BYOD image (`gcr.io/kaggle-private-byod`), ravaghi's `Trainer` pickles, and a custom `hill_climbing` module — a blind fork breaks. Only the public-image `9.251 DWT-based` notebook is forkable, and its sole custom dependency is `from hill_climbing import Climber`.
- **Fork-runnable baseline (Strategy A):** surgically transform the public source via `rogii-wellbore-baseline/generate_notebook.py` — (1) drop the `hill_climbing` import and inline an equivalent Caruana greedy-ensemble `Climber`; (2) the public artifacts dataset ships `*_trainer_*.pkl` (Trainer/BYOD) not the `models.pkl` cache the notebook's guard expects, so force a re-train from the precomputed `train.csv` features; (3) pin CatBoost to a single GPU (`devices="0:1"` → `"0"`). A `SMOKE` flag validated the full pipeline before the full run (3 LGB + 3 CatBoost → Climber → Optuna post-proc → Savitzky-Golay smoothing → valid `submission.csv`, 14,151 rows, no NaN).
- **Reproduction gap diagnosed (10.224 vs claimed 9.25):** pulling the submitted kernel and diffing it cell-by-cell against the original shows **no substantive difference** — `lgb_params`/`cb_params` and the entire post-processing are byte-identical (the only deltas are the inlined `Climber`, the single-GPU pin, and a smaller Optuna trial count that converges to the same value). The re-trained models are **per-fold on par with or better than** the author's saved models (our LGB fold-0 RMSE 9.49 vs the author's 9.64, recovered by loading the `*_trainer_*.pkl` with a stub `Trainer` class). Conclusion: **10.224 is the honest from-scratch reproduction of this GBT approach**; the title-claimed 9.251 only arises from the notebook's *load* branch (pre-trained `models.pkl`/`oof_preds.pkl`), which the public artifacts do not contain — so it is not cleanly reproducible from public materials.
- **Submission finding (verified on an active comp):** `kaggle competitions submit -f` returns **400** on this code competition — file submission is structurally disabled; submission must go through the notebook UI. Scoring this code competition takes ~70–90 min (notebook re-run on the hidden test set), not seconds.
- **Sequence-model lever worked — new best 9.905 via a GBT + TCN ensemble:** a flat per-row GBT ignores the row-to-row continuity of the toe trajectory (it only patches it post-hoc with Savitzky-Golay smoothing). A **TCN** (dilated residual 1D-CNN) that convolves along the ordered toe rows of each well captures it directly. Fed the *same* ~195 engineered features as the GBT, the TCN reached CV toe RMSE **10.60** standalone (`rogii-wellbore-tcn-hybrid/`) after stabilizing batch-size-1 training (**GroupNorm** instead of BatchNorm, gradient clipping, cosine LR) — 3 of 5 folds beat the GBT. It is then injected as a **7th ensemble member** into the proven GBT pipeline (`rogii-wellbore-ensemble/`): the TCN's OOF / test predictions are added as a `tcn` column to the `oof_preds`/`test_preds` dicts right before the hill-climb, so the existing greedy `Climber` weights all seven members with no other change (`allow_negative_weights=False` ⇒ a useless member gets weight 0, never worse than the GBT alone). Because the sequence and flat models have partly decorrelated errors, the blend improved the LB from **10.224 → 9.905** (verified). GPU note: the Kaggle P100 (sm_60) is incompatible with the current Torch build, so the kernel pins `machine_shape: NvidiaTeslaT4` (sm_75).
- **Rank reality check (fresh LB, 3,465 teams):** the 9.905 public LB is rank **1,657 / 3,465 (top 48%)** — mid-pack. Medal lines: **gold 6.72 (rank 17) / silver 7.229 (rank 174) / bronze 7.29 (rank 347)**; leaders 5.35–5.7. The whole public-clone lineage tops out ~9–10, so it sits far outside the medal zone — closing a **−3.2 RMSE gap to gold is an order of magnitude beyond any ±0.1–0.3 incremental tune** (target fixed first, gap back-calculated). A structural change of formulation, not another feature, is the only path.
- **Evaluation structure (verified) — optimize CV, not public LB:** the 3 visible test wells are *byte-identical* to 3 training wells (same `MD`/`GR`; the visible-prefix `TVT_input` equals the train `TVT`), so the public-LB sample is train-overlapping and the hidden toe answers literally exist under `train/`. But this is a notebook-rerun code competition with 3,127 teams and **no ~0 scores**, so the scored/private set is unseen (non-overlapping) wells. We therefore optimize **GroupKFold(by well) CV** — the honest unseen-well proxy — and treat the public LB as overlap-inflated (no leak exploitation; private decides medals).
- **Global formation structural-surface features (CV win):** the public pipelines reconstruct the train-only formation-top columns spatially via a per-well-median **local KNN plane**, which discards within-well dip. We replace that collapse with a per-formation absolute-depth surface `S_f(X,Y)` = **global 2nd-order WLS trend + local residual IDW**, fit from **per-row** samples across all wells (recovers dip), datum-separated, and **leave-one-well-out fold-safe** (per-well normal-equation downdate + self-excluded residuals — features valid under any GroupKFold split). Injected as new `tvtS_*` delta members (`rogii-wellbore-surface/`), this improved the self-contained tuned base's GroupKFold OOF from **10.32 → 10.19**, confirming the structural lever generalizes to unseen wells.
- **Surface + TCN stack — CV 10.19 → 10.06 (`rogii-wellbore-medal/`):** the TCN sequence model added as a Ridge-stack member on top of the surface base improved GroupKFold OOF absolute RMSE from 10.19 to **10.06**, and took the **largest stack weight (0.431 of 5 members)** — the sequence model carries unseen-well signal the flat GBTs miss.
- **Aligner core — the public beam is geometry-blind (the structural lever):** the public DTW/beam correlator (`_beam_jit`) walks the toe gamma-ray against the typewell with a symmetric index-move penalty `mc·|d|` and **never uses the well trajectory (`Z`, `MD`) at all** — it is pure GR matching. But `TVT = −Z + S_f(X,Y) + b`, so the expected per-step change is `ΔTVT = −ΔZ + dip·Δ(horizontal)`, all known or heel-estimable. A **dip-aware transition** centers each move on that geometric expectation instead of on zero. This is why the public-clone lineage plateaus ~9–10: it throws away the geometry.
- **Verified go/no-go (773 wells, leak-free):** reconstructing a held-out pseudo-toe (last 30% of each well's *known* zone; dip estimated from the heel 70% only, mirroring heel→toe deployment) the dip-aware beam cuts mean per-well RMSE from **6.933 → 5.753 (−1.18)** versus the geometry-blind beam at *matched search width* (the fair baseline is the same routine with zero expected-drift, which reproduced the public beam's 6.933 exactly — a clean apples-to-apples check). Geometry-only (no GR) scores 9.68, so GR and dip-geometry are complementary.
- **Step lesson — dip belongs in the transition, not as a feature:** handing the GBT a raw dip column first *hurt* CV (10.19 → 10.23) and was reverted — a tree cannot reproduce the along-trajectory integration that turns dip into TVT; the gain only materializes inside the aligner's path search.
- **Champion — dip-beam as an intra-well stack member (`rogii-wellbore-surface/`, CV 9.978):** the dip-aware beam's toe TVT, emitted as a feature (`tvt_dipbeam_d`) computed from each well's *own* known-zone dip, typewell, and trajectory only (**fold-safe by construction, no cross-well leakage**), improved the surface base's GroupKFold OOF absolute RMSE from **10.19 → 9.978 (−0.215)** — the single largest lever and the current champion. Confirmed that dip helps *only inside the aligner transition*: the same signal handed to the GBT as a raw feature hurt CV. The dip-beam also makes the TCN redundant (`medal+dipbeam` 10.058 > surface+dipbeam 9.978), so the TCN is dropped from the champion.
- **GBT-feature levers exhausted (10 experiments, all ±0.1–0.3, none beat 9.978):** GR-alignment tuning, surface-gradient dip (`surfdip` 10.107), ANCC cross-well kriging (`krig` 10.110), multi-scale wavelet GR texture (`wav` 10.266), and track-aligner replacement (10.171) each stayed inside the 9–10 band. **The GBT-feature core is mined out**; every axis that converts an inversion signal into a tree feature collapses back to the public-clone ceiling.
- **Pivot — inversion core (the formulation gap):** deep research (ROGII's own production patent US 11,480,045 + geosteering state-space/SMC literature) located the 7.x-vs-9–10 divide not in features but in **formulation**: the public forks do per-row DWT-feature → GBT *regression*, while the principled approach is **forward-model inversion** — build a synthetic log from the typewell under candidate (dip × typewell × thickness) and select the geology that best explains the observed horizontal GR. Caveat (verified): **no public artifact was confirmed to reach 7.x** (best public inversion-flavored notebook is still ~9.96), so there is no copyable winning recipe — the inversion core is a principled, high-risk research bet inferred from domain SOTA. A self-contained **go/no-go diagnostic** (`rogii-wellbore-invcore/`) gates each candidate core against the champion's dip-aware beam on a long pseudo-toe before any full build.
- **Learned-inversion core — NO-GO (decisive):** a from-scratch GR→stratigraphy inversion net (1D-CNN + typewell cross-attention, residual-on-geometry), trained directly on real known-zone pairs with good convergence, scored **14.8 vs the dip-aware beam's 5.58** and lost even to geometry-only (9.58), winning on just 22% of held-out wells. The DP-global-optimal beam is not beaten by a small learned aligner. The cheap diagnostic saved weeks of a full MTP build.
- **Particle-filter core (online dip) — NO-GO:** a distance-direction particle filter (state = TVT + OU-bounded local dip, GR likelihood vs typewell, confidence-gated blend) targets the *fixed-heel-dip* degradation. On a representative 40-well sample it was a **coin-flip** (blend 23.1 vs beam 16.9, wins 40%); an 8-well smoke that looked like a win (+1.8) was a lucky-sample artifact. Online-dip-from-GR does not robustly beat the fixed-dip beam.
- **Stretch lever — real but not extrapolable (key finding):** an **oracle** test (inject the *true* per-step typewell-index advance into the beam) collapses the long-toe RMSE **16.9 → 11.1 (−5.8, helps 68% of wells)** — by far the largest lever signal seen, proving typewell↔horizontal bed-thickness *stretch* is the dominant long-toe error. **But it does not extrapolate from the heel:** injecting the leak-free heel-mean stretch (mean ratio ≈ 0.985 ≈ 1) made it *worse* (18.7 vs 16.9). The heel matches geometry by construction (dip is fit there); the toe's structural deviation is not foreshadowed by the heel.
- **Conclusion (2026-06-23) & next lever:** across learned-inversion, particle-filter, and stretch, **the toe's structural deviation is not predictable from within-well data (heel + own typewell + GR)** — the within-well alignment frame is exhausted at ~9–10. The remaining gold-direction is **cross-well**: constrain an unseen well's toe TVT from neighboring *training* wells' known tops at similar (X,Y) — a stronger 3D spatial formulation than the per-formation WLS+IDW surface (−0.13) and ANCC kriging (±0.3) already tried.

#### Strategy pivot (2026-06-25) — public LB is an override mirage; private is decided by the base blend

- **The public frontier is a dual-pipeline fork, and we matched it (7.311).** The strongest public lineage (`fle3n-rogii-v5` / `rogii-dual-pipeline` / `rogii-lb-7-159/201`) is a two-pipeline blend — **Pipeline A "ridge-sp45"** (selector physics + LightGBM/CatBoost/Ridge stack + projection) and **Pipeline B "fleongg"** (likelihood-PF + GBM stack) — combined `0.55·A + 0.45·B`, then a **guarded contact override** on the few test wells that duplicate `train/`. Forking it (`rogii-wellbore-dualpipe/`) reached **public LB 7.311 (rank 480 / 3,589, top 13.4%)** — a ~2.6 jump over the prior 9.905, and within 0.06 of the public bronze line.
- **But the override does not transfer to the private leaderboard — the public notebooks say so themselves.** The 7.159 frontier notebook's own results summary states the guarded override is a *"public-LB-only gain"* and the gold-prefix calibration overlay *"can push the public LB toward ~7.2–7.3 but is a leakage path that does not transfer to the private leaderboard."* The visible test wells are train-overlapping; the override reconstructs them near-exactly (≈0.01 ft), which inflates only the public score. The honest blend number (no override) is ~7.5–7.6 on public, ~9.2 on unseen-well GroupKFold CV. **So on the private/medal leaderboard every override-using fork collapses back to its base blend — the medal goes to whoever has the best genuine base, and the field is clustered near the ~9.2 lineage ceiling.** The gold lever is therefore to lower the *base* CV; a small base gain can move private rank where the field is dense.
- **TCN as a decorrelated Ridge-stack member — verified base gain (−0.213 GroupKFold OOF).** Injecting the sequence-model TCN into Pipeline A's `oof_preds`/`test_preds` dicts (the proven 10.224→9.905 pattern) lets the positive-constrained Ridge stack weight it: pipeline-A Ridge OOF dropped **10.4197 → 10.2068** on GroupKFold (the unseen-well / private proxy). GPU note: the kernel pins `machine_shape: NvidiaTeslaT4` (the assigned P100 sm_60 is Torch-incompatible → CPU fallback otherwise). A re-generatable patcher (`generate_notebook.py`, `SMOKE` flag, byte-identical-override round-trip asserts) drives the injection.
- **Blend-level fixed-weight 3-way — rejected by validation.** Riding the standalone TCN over the blend (`(1−w)·base + w·tcn`) has no faithful blend-level OOF; its only signal — true-RMSE on the train-overlapping public wells — is **in-sample** (the GBTs memorize those wells, a CV~10 sequence model cannot match), so it favors the wrong thing and worsens in-sample. Kept dormant (`w_tcn = 0`); the Ridge-weighted A-injection (which *is* validated on the unseen-well proxy) is shipped instead.
- **Next lever (in progress) — stack the repo's verified surface signals into the dual-pipeline base.** The two features that beat the pub_dwt base in our own `rogii-wellbore-surface` champion — the **global formation structural surface** (`tvtS_*`, −0.123) and the **dip-aware beam** (`tvt_dipbeam_d`, −0.215) — are injected into Pipeline A's feature build, forcing a GBT re-train (`FORCE_RETRAIN`) on the augmented feature set, validated by the same Ridge-OOF diagnostic against the 10.4197 baseline. Goal: a maximally-stacked genuine base (surface + dip-beam + TCN) below the ~9.2 lineage ceiling, since private rank is decided by base quality.

---

### Playground Series S6E6 - Stellar Classification (Active)

**Competition:** [Playground Series S6E6](https://www.kaggle.com/competitions/playground-series-s6e6) | **Deadline:** 2026-06-30

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

- **Current strategy (2026-04-23):** Reproduction base reached (0.926 ≒ public max 0.929 claimed). Moving to **proven-stacking phase**: external Xeno-Canto data (pipeline active), multi-backbone ensemble, class balancing — all recurrent techniques across BirdCLEF 2024/2025 top solutions. Note: `unlabeled_soundscapes/` not provided in 2026 (unlike 2024/2025) → pseudo-label distillation replaced by external data.
- **External data pipeline (active):** Xeno-Canto (XC) v3 API → 11,563 Aves recordings filtered (Q A|B, non-ND license, 159 species, 138.9h total). Dataset: [`yasunorim/xc-birdclef-2026-target-urls`](https://www.kaggle.com/datasets/yasunorim/xc-birdclef-2026-target-urls). Embedding kernel: [`yasunorim/xc-perch-v2-embed-birdclef-2026`](https://www.kaggle.com/code/yasunorim/xc-perch-v2-embed-birdclef-2026) — outputs Perch v2 embeddings + logits (1536-dim, 234-class) for ProtoSSM ingestion. Integration design: [XC_INTEGRATION_DESIGN.md](birdclef-2026-work/docs/XC_INTEGRATION_DESIGN.md). **Lesson:** Perch v2 is a CPU-only SavedModel — must set `CUDA_VISIBLE_DEVICES=""` before `import tensorflow` on GPU-enabled Kaggle machines to avoid `InvalidArgumentError`.
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