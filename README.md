# Kaggle Competitions

My Kaggle notebook contributions - 4 Bronze Medals earned with AI-assisted development.
(Kaggleノートブック投稿 - AI支援開発で獲得した4つのブロンズメダル)

## Development Approach

All notebooks were developed using **Claude Code** (AI-assisted development tool by Anthropic).
(すべてのノートブックは **Claude Code**（Anthropic社のAI支援開発ツール）を使用して開発しました)

**Important Note:** These are **Notebook Medals**, earned through community votes on shared notebooks - NOT competition ranking medals.
(重要: これらは**ノートブックメダル**であり、共有ノートブックへのコミュニティ投票により獲得したものです。コンペティション順位によるメダルではありません)

---

## 🥉 Bronze Medal Notebooks (4)

### 1. NFL Big Data Bowl 2026 - Prediction

**Notebook:** [Geometric Rules Baseline - 2.921 RMSE (No ML)](https://www.kaggle.com/code/yasunorim/geometric-rules-baseline-2-921-rmse-no-ml)

Sports analytics using NFL player tracking data.
(NFLプレイヤートラッキングデータを用いたスポーツ分析)

**Approach:**
- Physics-based geometric rules (物理ベースの幾何学的ルール)
- Targeted receivers → direct path to ball landing point
- Defensive coverage → distance-based offset from receivers
- No machine learning required (機械学習不要)

**Performance:**
- **RMSE:** 2.921 yards
- **Execution Time:** <5 seconds

**Tech Stack:** Python, pandas, polars, numpy

**Key Learning:** Domain knowledge and simple geometric rules can outperform complex ML models in specific contexts.
(重要な学び: 特定の状況では、ドメイン知識とシンプルな幾何学的ルールが複雑なMLモデルを上回ることがある)

---

### 2. PhysioNet - Digitization of ECG Images

**Notebook:** [PhysioNet ECG Baseline](https://www.kaggle.com/code/yasunorim/physionet-ecg-baseline)

Complete submission format guide for ECG image digitization challenge.
(ECG画像デジタル化チャレンジの完全な提出フォーマットガイド)

**Key Contributions:**
- Correct submission format documentation (正しい提出フォーマットの文書化)
- Common mistakes and how to avoid them (よくあるミスとその回避方法)
- Working baseline with verified format (検証済みフォーマットの動作するベースライン)

**Format Learnings:**
- Submission file: Must be `.csv` (NOT `.parquet`)
- ID format: `{ecg_id}_{sample_index}_{lead}` (order matters!)
- Column names: `['id', 'value']` (NOT 'voltage')

**Tech Stack:** Python, pandas, numpy

**Key Learning:** Always read sample_submission file first - format errors waste precious submission attempts.
(重要な学び: 必ず最初にsample_submissionファイルを読む - フォーマットエラーは貴重な提出回数を無駄にする)

---

### 3. Diabetes Prediction Challenge (S5E12) - EDA & Baseline

**Notebook:** [Diabetes Prediction - EDA & Baseline (S5E12)](https://www.kaggle.com/code/yasunorim/diabetes-prediction-eda-baseline-s5e12)

Comprehensive exploratory data analysis and LightGBM baseline.
(包括的な探索的データ分析とLightGBMベースライン)

**Key Contributions:**
- Debug-first approach with detailed data inspection (詳細なデータ検査によるデバッグファースト手法)
- Step-by-step EDA visualization (段階的なEDA可視化)
- Proper 5-fold cross-validation setup (適切な5-fold交差検証の設定)

**Performance:**
- **CV AUC:** 0.72687 ± 0.00082
- **5-Fold scores:** [0.72768, 0.72542, 0.72662, 0.72711, 0.72754]

**Tech Stack:** Python, pandas, LightGBM, scikit-learn, matplotlib, seaborn

---

### 4. Diabetes Prediction Challenge (S5E12) - Rank-Based Ensemble

**Notebook:** [Diabetes Prediction - Rank-Based Ensemble](https://www.kaggle.com/code/yasunorim/diabetes-prediction-rank-based-ensemble)

Advanced ensemble technique using rank-based blending.
(ランクベースブレンディングを使用した高度なアンサンブル手法)

**Approach:**
- Dual LightGBM models with different random seeds (異なるランダムシードを使用した2つのLightGBMモデル)
- Rank-based blending using `.rank(pct=True)` (`.rank(pct=True)`を使用したランクベースブレンディング)
- Weighted averaging (main=1.0, diversity=0.5) (重み付け平均)

**Key Insight:**
- AUC is a rank-based metric (AUCはランクベースの指標)
- Rank averaging directly optimizes ranking quality (ランク平均化はランキング品質を直接最適化)
- Standardizes predictions across models (モデル間の予測を標準化)

**Performance:**
- **Blended OOF AUC:** 0.72716 (improvement over single model)

**Tech Stack:** Python, pandas, LightGBM, scikit-learn

---

## 📚 Key Learnings (主要な学び)

1. **Format First** (まずフォーマット)
   - Always verify submission format before complex modeling
   - Read sample_submission carefully
   - Test with simple baseline first

2. **Domain Knowledge Matters** (ドメイン知識が重要)
   - Simple physics-based rules can beat ML
   - Understanding the problem > model complexity

3. **Ensemble Techniques** (アンサンブル手法)
   - Rank-based blending for AUC optimization
   - Diversity through different random seeds

4. **AI-Assisted Development** (AI支援開発)
   - Claude Code accelerates notebook development
   - Focus on problem understanding, let AI handle boilerplate

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Languages** | Python |
| **ML Libraries** | LightGBM, scikit-learn, XGBoost |
| **Data Processing** | pandas, numpy, polars |
| **Visualization** | matplotlib, seaborn |
| **Development** | Claude Code, Jupyter Notebook |

---

## 📫 Profile

**Kaggle:** [@yasunorim](https://www.kaggle.com/yasunorim)

---

> 💡 *4 Bronze Medals earned through AI-human collaboration - proving that effective tool usage is a valuable skill*
>
> (AI×人間のコラボレーションで獲得した4つのブロンズメダル - 効果的なツール活用が価値あるスキルであることの証明)

---

*Built with Claude Code*
