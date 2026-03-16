# Kaggle Competitions

Kaggle Notebooks Expert. 12 Bronze Notebook Medals + active competition participation.

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
- **`CreateCodeSubmission` API returns 403** — but file-based submission (`competition_submit` with output CSV download) works as a reliable alternative

**Blog post:** [DEV.to](https://dev.to/yasumorishima/kaggle-code-competitions-without-a-local-gpu-github-actions-kaggle-api-cloud-workflow-m3)

---

## 🔬 Experiment Management (EXP + child-exp)

EXP + child-exp experiment management is inspired by [chimanさんの記事](https://zenn.dev/chiman/articles/b233cc808d6af3) (Kaggle gold medal with Claude Code/Codex). The architecture (RPi5 Colab keepalive, Google Drive sync, file monitor automation) is our own design for a GPU-less local environment.

### Architecture

```
[Local PC]                  [RPi5]                     [Google Colab (Free)]
Claude Code                 Chromium + wtype keepalive  File Monitor Notebook
  ↓ Write config/code        ↓ Session keepalive         ↓ Auto-run train.py
  ↓                          ↓ 30min Page Down/Up        ↓
Google Drive (for Desktop) ←――――――――――――――――――→ Google Drive (mount)
  EXP/config/child-exp005.yaml                          Detect new config → execute
  EXP/output/child-exp005/result.json                   Save results to Drive
```

- **RPi5 Chromium + wtype** keeps free Colab sessions alive (prevents 90-min idle timeout via systemd service)
- **Claude Code** writes experiment configs to Google Drive
- **Colab** auto-detects new configs and runs `train.py`
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

### RPi5 Colab Session Recovery

After RPi5 reboot, restore the Colab session via SSH:

```bash
ssh yasu@100.77.198.48 "DISPLAY=:0 WAYLAND_DISPLAY=wayland-1 \
  XDG_RUNTIME_DIR=/run/user/\$(id -u) \
  chromium-browser --disable-gpu --disable-software-rasterizer \
  --disable-extensions --disable-dev-shm-usage \
  'https://colab.research.google.com/drive/11whi9Hyc7JPdWDolkQRHdQNLyXIkqnXi' &"
```

- Google login cookies persist in Chromium profile — no re-login needed
- `colab-keepalive.service` (wtype) auto-starts on boot
- `hdmi_force_hotplug=1` in `/boot/firmware/config.txt` enables display output without a monitor
- Keyring dialog may appear on first launch — requires Enter on RPi5 monitor (or disable keyring)

---

## 🏆 Competition Results

### Deep Past Challenge - Akkadian to English Translation (Active)

**Competition:** [Deep Past Initiative Machine Translation](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)

Ancient cuneiform (Akkadian) transliteration → English translation task. Evaluated with BLEU + chrF++.

**Notebook:** [Deep Past Cloud Workflow + TF-IDF Baseline](https://www.kaggle.com/code/yasunorim/deep-past-cloud-workflow-tfidf-baseline) *(public)* 🥉

| Approach | Public Score |
|---|---|
| TF-IDF char n-gram nearest neighbor | 5.6 |

- **Approach:** Character n-gram TF-IDF (2-5), cosine similarity nearest neighbor
- Pushed via GitHub Actions cloud workflow (see above)

---

### S6E2 - Predicting Heart Disease (Active)

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
<summary><h2>🥉 Bronze Medal Notebooks (12)</h2></summary>

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
