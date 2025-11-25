# MITSUI&CO. Commodity Prediction Challenge

Financial time series forecasting competition for commodity futures prices.

## 📋 Competition Overview

- **Host:** MITSUI&CO. (三井物産) × Preferred Networks
- **Platform:** Kaggle
- **Prize:** $100,000
- **Evaluation Period:** Sep 30, 2025 - Jan 16, 2026
- **Type:** Forward-running evaluation with real market data

## 🎯 Task

Predict log returns for 424 commodity pair spreads across 4 markets:
- LME (London Metal Exchange)
- JPX (Japan Exchange Group)
- US Markets
- FX (Foreign Exchange)

**Evaluation Metric:** Sharpe Ratio of Spearman Rank Correlation

## 🐛 Critical Bug Fixed

### Problem
Initial submission scored **-0.058** (negative correlation)

### Root Cause
**Backward-looking target calculation** instead of forward-looking:

```python
# ❌ Wrong (Backward-looking)
target[t] = log(price[t] / price[t-lag])

# ✅ Correct (Forward-looking)
target[t] = log(price[t+lag+1] / price[t+1])
```

### Impact
Fundamental misunderstanding of what to predict led to inverse correlation.

## 💡 Solution

### Simple Mean Reversion Model

```python
def predict(target_id, historical_stats, recent_context):
    # 1. Historical mean
    base = historical_mean[target_id]
    
    # 2. Mean reversion
    if recent_values:
        momentum = -mean(recent_values) * 0.1
    
    # 3. Realistic noise
    noise = normal(0, historical_std * 0.8)
    
    return clip(base + momentum + noise, bounds)
```

**Why Simple?**
- Financial markets are near-random short-term
- Overfitting is fatal in forward-running evaluation
- Robustness > Complexity

## 📊 Results

- **Before Fix:** -0.058 (negative correlation)
- **After Fix:** Positive score, evaluation ongoing
- **Status:** Currently in evaluation period

## 🎓 Key Learnings

### 1. Problem Understanding > Model Complexity
Spent time on complex models before understanding the basic task definition.

### 2. Time Axis is Critical in Finance
Forward vs backward-looking distinction is fundamental in financial prediction.

### 3. Read the Documentation
The correct target definition was in the documentation all along.

### 4. Negative Score = Red Flag
Don't ignore suspicious scores. -0.058 meant something was fundamentally wrong.

### 5. Simple Baselines First
Should have started with:
```python
prediction = historical_mean + random_noise
```

## 🛠️ Tech Stack

- **Language:** Python 3
- **Core Libraries:** pandas, polars, numpy
- **Environment:** Kaggle Notebooks

## 📁 Files

```
mitsui-commodity-prediction/
├── forward-looking-fix.ipynb    # Main submission
├── analysis.ipynb                # Data exploration
└── README.md                     # This file
```

## 🔗 Links

- **Competition:** [Kaggle Competition Page](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge)
- **Code:** [Forward-Looking Target Fix](https://www.kaggle.com/code/yasunorim/forward-looking-target-fix)

## 📝 Notes

This was my first financial time series competition. The experience taught me:
- Importance of understanding domain-specific concepts
- Value of simple, robust approaches in noisy data
- How to debug from score interpretation

## 📅 Timeline

- **Sep 29, 2025:** Submission deadline
- **Sep 30 - Jan 16, 2026:** Forward-running evaluation period
- **Jan 16, 2026:** Final results

---

*Evaluation ongoing. Final results pending.*
