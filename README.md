# Kaggle Competitions

My Kaggle competition submissions and learning journey.
(Kaggleコンペティションへの提出と学習の記録)

## 🏆 Competitions

### MITSUI&CO. Commodity Prediction Challenge (2025)

**Status:** Evaluation period (Final results: Jan 2026)
(ステータス: 評価期間中 - 最終結果: 2026年1月)

Commodity futures price prediction with forward-running evaluation.
(先行評価による商品先物価格予測)

**Key Learning:**
- Fixed critical forward-looking vs backward-looking target bug
  - (重大なバグ修正: 前方参照と後方参照のターゲット設定ミス)
- Simple mean reversion model outperformed complex approaches
  - (シンプルな平均回帰モデルが複雑なアプローチを上回った)
- Understanding problem > Model complexity in financial time series
  - (金融時系列では問題理解がモデルの複雑さより重要)

**Tech Stack:** Python, pandas, polars, numpy

**Code:** [Kaggle Notebook](https://www.kaggle.com/code/yasunorim/forward-looking-target-fix)

---

### NFL Big Data Bowl 2026

**Status:** 🥉 **Bronze Medal (Notebook)**

Sports analytics using NFL player tracking data.
(NFLプレイヤートラッキングデータを用いたスポーツ分析)

**Achievement:** Community-recognized notebook (not competition ranking)
(実績: コミュニティに評価されたノートブック - コンペティション順位ではありません)

**Approach:** (アプローチ)
- Physics-based geometric rules (物理ベースの幾何学的ルール)
- Targeted receivers → direct path to ball landing point
  - (ターゲットレシーバー → ボール着地点への直線経路)
- Defensive coverage → distance-based offset from receivers
  - (ディフェンスカバレッジ → レシーバーからの距離ベースオフセット)
- Linear interpolation for trajectory prediction
  - (軌道予測のための線形補間)
- No machine learning required (機械学習不要)

**Performance:**
- **RMSE:** 2.921 yards
- **Execution Time:** <5 seconds
- **Public Leaderboard Score:** Competitive baseline

**Tech Stack:** Python, pandas, polars, numpy

**Code:** [Geometric Rules Baseline - 2.921 RMSE](https://www.kaggle.com/code/yasunorim/geometric-rules-baseline-2-921-rmse-no-ml)

**Key Learning:** Domain knowledge and simple geometric rules can outperform complex ML models in specific contexts.
(重要な学び: 特定の状況では、ドメイン知識とシンプルな幾何学的ルールが複雑なMLモデルを上回ることがある)

**Note:** This medal was earned through notebook sharing and community votes, not competition placement.
(注: このメダルはノートブック共有とコミュニティ投票により獲得したもので、コンペティション順位によるものではありません)

**Applied to Work:** Competition experience contributed to processing time prediction system development (R²=0.579).
(実務への応用: コンペでの経験を、社内の処理時間予測システムの開発に応用 - R²=0.579)

---

## 📚 Key Learnings (主要な学び)

1. **Problem Understanding First** (まず問題理解)
   - Read documentation thoroughly (ドキュメントを徹底的に読む)
   - Understand evaluation metrics deeply (評価指標を深く理解する)
   - Start with simple baselines (シンプルなベースラインから始める)

2. **Financial Time Series** (金融時系列)
   - Forward-looking vs backward-looking (前方参照 vs 後方参照)
   - Mean reversion properties (平均回帰の性質)
   - Avoid overfitting (過学習を避ける)

3. **Debugging** (デバッグ)
   - Negative correlation = fundamental issue (負の相関 = 根本的な問題)
   - Score meaning > Score value (スコアの意味 > スコアの値)

## 🛠️ Common Tech Stack

- **Languages:** Python
- **Libraries:** pandas, numpy, scikit-learn, XGBoost, LightGBM
- **Tools:** Jupyter Notebook, Kaggle Notebooks

## 📫 Profile

Kaggle: [@yasunorim](https://www.kaggle.com/yasunorim)

---

> 💡 *Learning through competition - Problem understanding often matters more than model complexity*
> 
> (コンペティションを通じた学習 - モデルの複雑さよりも問題理解が重要)
