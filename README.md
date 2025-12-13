# Kaggle Competitions

My Kaggle competition submissions and learning journey.

## 🏆 Competitions

### MITSUI&CO. Commodity Prediction Challenge (2025)
**Status:** Evaluation period (Final results: Jan 2026)

Commodity futures price prediction with forward-running evaluation.

**Key Learning:**
- Fixed critical forward-looking vs backward-looking target bug
- Simple mean reversion model outperformed complex approaches
- Understanding problem > Model complexity in financial time series

**Tech Stack:** Python, pandas, polars, numpy

**Code:** [Kaggle Notebook](https://www.kaggle.com/code/yasunorim/forward-looking-target-fix)

---

### NFL Big Data Bowl 2026
**Status:** 🥉 **Bronze Medal (Notebook)**

Sports analytics using NFL player tracking data.

**Achievement:** Community-recognized notebook (not competition ranking)

**Approach:**
- Physics-based geometric rules
- Targeted receivers → direct path to ball landing point
- Defensive coverage → distance-based offset from receivers
- Linear interpolation for trajectory prediction
- No machine learning required

**Performance:**
- **RMSE:** 2.921 yards
- **Execution Time:** <5 seconds
- **Public Leaderboard Score:** Competitive baseline

**Tech Stack:** Python, pandas, polars, numpy

**Code:** [Geometric Rules Baseline - 2.921 RMSE](https://www.kaggle.com/code/yasunorim/geometric-rules-baseline-2-921-rmse-no-ml)

**Key Learning:** Domain knowledge and simple geometric rules can outperform complex ML models in specific contexts.

**Note:** This medal was earned through notebook sharing and community votes, not competition placement.

---

## 📚 Key Learnings

1. **Problem Understanding First**
   - Read documentation thoroughly
   - Understand evaluation metrics deeply
   - Start with simple baselines

2. **Financial Time Series**
   - Forward-looking vs backward-looking
   - Mean reversion properties
   - Avoid overfitting

3. **Debugging**
   - Negative correlation = fundamental issue
   - Score meaning > Score value

## 🛠️ Common Tech Stack

- **Languages:** Python
- **Libraries:** pandas, numpy, scikit-learn, XGBoost, LightGBM
- **Tools:** Jupyter Notebook, Kaggle Notebooks

## 📫 Profile

Kaggle: [@yasunorim](https://www.kaggle.com/yasunorim)
