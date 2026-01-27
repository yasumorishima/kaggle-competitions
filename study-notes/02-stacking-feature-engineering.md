# Kaggle「住宅価格予測」- 特徴量エンジニアリングとスタッキング

## 結果サマリー

| 項目 | 値 |
|------|-----|
| コンペ | House Prices - Advanced Regression Techniques |
| 最良単一モデル RMSLE | 0.1080 |
| スタッキング RMSLE | 0.1065（1.39%改善） |
| 作成した特徴量数 | 20種類以上 |
| 特徴量選択 | 約330列 → 約150列（重要度ベース） |
| 使用モデル | Ridge, Lasso, ElasticNet, XGBoost, LightGBM, GradientBoosting |

---

## 1. ライブラリと環境設定

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib  # 日本語グラフ対応

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import skew
from scipy.special import boxcox1p
import xgboost as xgb
import lightgbm as lgb
```

## 2. データの読み込み

```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f'訓練データ: {train.shape}')  # (1460, 81)
print(f'テストデータ: {test.shape}')  # (1459, 80)
```

訓練データは1460件、81個の特徴量。目的変数は`SalePrice`（販売価格）。

## 3. 探索的データ分析（EDA）

### 3.1 外れ値の発見

```python
plt.scatter(train['GrLivArea'], train['SalePrice'])
plt.xlabel('地上居住面積')
plt.ylabel('販売価格')
plt.title('外れ値の確認')
plt.show()
```

面積4000平方フィート以上で価格30万ドル未満の物件が異常値として検出された。

### 3.2 目的変数の分布

```python
sns.histplot(train['SalePrice'], kde=True)
plt.title('販売価格の分布')
plt.show()

sns.histplot(np.log1p(train['SalePrice']), kde=True)
plt.title('販売価格の分布（対数変換後）')
plt.show()
```

- 元データの歪度: 1.88（右に偏った分布）
- 対数変換後の歪度: 0.12（正規分布に近い）
- RMSLEで評価される問題では対数変換が必須

### 3.3 相関分析

```python
correlations = train.corr()['SalePrice'].sort_values(ascending=False)
print(correlations.head(11))
```

相関の高い特徴量:
- `OverallQual`（全体的な品質）: 0.79
- `GrLivArea`（地上居住面積）: 0.71
- `GarageCars`（ガレージ収容台数）: 0.64

## 4. 外れ値の除去

```python
# 面積は大きいのに価格が異常に低い物件
train = train.drop(train[(train['GrLivArea'] > 4000) &
                        (train['SalePrice'] < 300000)].index)

# 地下室面積が極端に大きい物件
train = train.drop(train[train['TotalBsmtSF'] > 3000].index)

# 土地面積が極端に大きい物件
train = train.drop(train[train['LotArea'] > 100000].index)
```

外れ値はデータ入力ミスや特殊な状況であることが多く、モデルが正しいパターンを学習する妨げになる。

## 5. 欠損値処理

### 5.1 訓練データとテストデータの結合

```python
ntrain = train.shape[0]
y_train = train['SalePrice'].values
all_data = pd.concat([train.drop('SalePrice', axis=1), test], axis=0)
```

### 5.2 4つの欠損値処理パターン

**パターン1: 'None'で埋める** — 「ない」ことが意味を持つ特徴量（プール、ガレージ、地下室など）

```python
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
             'GarageType', 'BsmtQual', 'BsmtCond', ...]
for col in none_cols:
    all_data[col] = all_data[col].fillna('None')
```

**パターン2: 0で埋める** — 数値で「ない」を表現できる特徴量

```python
zero_cols = ['GarageArea', 'GarageCars', 'TotalBsmtSF', 'MasVnrArea', ...]
for col in zero_cols:
    all_data[col] = all_data[col].fillna(0)
```

**パターン3: 最頻値で埋める** — カテゴリ変数

```python
mode_cols = ['MSZoning', 'Electrical', 'KitchenQual', ...]
for col in mode_cols:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
```

**パターン4: グループ別中央値** — 地域性のある特徴量

```python
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))
```

## 6. 特徴量エンジニアリング

### 6.1 集約特徴量（合計・平均）

```python
# 総床面積（地下 + 1階 + 2階）
all_data['TotalSF'] = (all_data['TotalBsmtSF'] +
                       all_data['1stFlrSF'] +
                       all_data['2ndFlrSF'])

# 総バスルーム数（フルバス + ハーフバス×0.5）
all_data['TotalBath'] = (all_data['FullBath'] +
                         0.5 * all_data['HalfBath'] +
                         all_data['BsmtFullBath'] +
                         0.5 * all_data['BsmtHalfBath'])

# 総ポーチ面積
all_data['TotalPorchSF'] = (all_data['OpenPorchSF'] +
                            all_data['EnclosedPorch'] +
                            all_data['ScreenPorch'] +
                            all_data['WoodDeckSF'])
```

### 6.2 時系列特徴量

```python
all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['YearsSinceRemod'] = all_data['YrSold'] - all_data['YearRemodAdd']
all_data['GarageAge'] = all_data['YrSold'] - all_data['GarageYrBlt']
```

### 6.3 バイナリ特徴量（あり/なし）

```python
all_data['IsNew'] = (all_data['YearBuilt'] == all_data['YrSold']).astype(int)
all_data['HasRemod'] = (all_data['YearBuilt'] != all_data['YearRemodAdd']).astype(int)
all_data['Has2ndFloor'] = (all_data['2ndFlrSF'] > 0).astype(int)
all_data['HasBsmt'] = (all_data['TotalBsmtSF'] > 0).astype(int)
all_data['HasGarage'] = (all_data['GarageArea'] > 0).astype(int)
all_data['HasPool'] = (all_data['PoolArea'] > 0).astype(int)
all_data['HasFireplace'] = (all_data['Fireplaces'] > 0).astype(int)
```

### 6.4 交互作用項

2つの特徴量を掛け合わせることで、相乗効果を表現する。

```python
# 品質 × 総床面積（高品質で広い家は特に高価）
all_data['OverallQual_TotalSF'] = all_data['OverallQual'] * all_data['TotalSF']

# 品質 × 居住面積
all_data['OverallQual_GrLivArea'] = all_data['OverallQual'] * all_data['GrLivArea']

# 総合品質（品質 + 状態）
all_data['TotalQual'] = all_data['OverallQual'] + all_data['OverallCond']
```

### 6.5 比率特徴量

```python
all_data['Bsmt_Ratio'] = all_data['TotalBsmtSF'] / (all_data['TotalSF'] + 1)
all_data['Garage_Ratio'] = all_data['GarageArea'] / (all_data['TotalSF'] + 1)
all_data['AreaPerRoom'] = all_data['GrLivArea'] / (all_data['TotRmsAbvGrd'] + 1)
```

### 6.6 カテゴリのグループ化

地域を価格帯別にグループ化し、カテゴリ数を減らして過学習を防ぐ。

```python
neighborhood_price = train.groupby('Neighborhood')['SalePrice'].median()

def categorize_neighborhood(neighborhood):
    price = neighborhood_price[neighborhood]
    if price < neighborhood_price.quantile(0.33):
        return 'Low'
    elif price < neighborhood_price.quantile(0.67):
        return 'Medium'
    else:
        return 'High'

all_data['NeighborhoodGroup'] = all_data['Neighborhood'].apply(categorize_neighborhood)
```

## 7. 特徴量変換

### 7.1 目的変数の対数変換

```python
y_train = np.log1p(y_train)
```

### 7.2 Box-Cox変換

歪んだ分布を持つ特徴量を正規分布に近づける。

```python
numeric_feats = all_data.select_dtypes(include=[np.number]).columns
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))

skewed_features = skewed_feats[abs(skewed_feats) > 0.75].index

lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
```

### 7.3 One-Hot Encoding

```python
all_data = pd.get_dummies(all_data)
print(f'エンコーディング後: {all_data.shape[1]}列')  # 約330列
```

## 8. 特徴量選択

### 8.1 LightGBMで重要度を計算

```python
X_train = all_data[:ntrain]
X_test = all_data[ntrain:]

lgb_selector = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
lgb_selector.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': lgb_selector.feature_importances_
}).sort_values('importance', ascending=False)
```

### 8.2 重要度の低い特徴量を除去

```python
threshold = 5
important_features = feature_importance[
    feature_importance['importance'] > threshold
]['feature'].tolist()

print(f'元の特徴量数: {X_train.shape[1]}')      # 約330列
print(f'選択後: {len(important_features)}')      # 約150列

X_train = X_train[important_features]
X_test = X_test[important_features]
```

約半分の特徴量を除去しても精度は下がらない。重要度の低い特徴量はノイズとなり、CVスコアとLBスコアのギャップ（過学習）を生む。

## 9. モデル構築

### 9.1 クロスバリデーション関数

```python
def rmsle_cv(model, X, y, n_folds=5):
    kf = KFold(n_folds, shuffle=True, random_state=42)
    rmse = np.sqrt(-cross_val_score(
        model, X, y,
        scoring='neg_mean_squared_error',
        cv=kf
    ))
    return rmse
```

### 9.2 線形モデル（正則化付き）

```python
ridge = Ridge(alpha=15.0, random_state=42)
lasso = Lasso(alpha=0.0005, random_state=42, max_iter=10000)
elastic = ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=42, max_iter=10000)
```

### 9.3 勾配ブースティングモデル

```python
# XGBoost
xgboost = xgb.XGBRegressor(
    n_estimators=3000, learning_rate=0.01,
    max_depth=3, min_child_weight=3, gamma=0.1,
    subsample=0.6, colsample_bytree=0.6,
    reg_alpha=0.0001, reg_lambda=2, random_state=42
)

# LightGBM
lightgbm = lgb.LGBMRegressor(
    n_estimators=3000, learning_rate=0.01,
    max_depth=3, num_leaves=8, min_child_samples=30,
    subsample=0.6, colsample_bytree=0.6,
    reg_alpha=0.2, reg_lambda=0.2, random_state=42
)

# Gradient Boosting
gb = GradientBoostingRegressor(
    n_estimators=3000, learning_rate=0.01,
    max_depth=3, min_samples_split=10, min_samples_leaf=8,
    subsample=0.7, random_state=42
)
```

### 9.4 過学習対策のポイント

- `max_depth=3`: 木の深さを浅くし、複雑なパターンの学習を抑制
- `subsample=0.6`: データの60%のみで学習し、ランダム性を持たせる
- `reg_lambda=2`: L2正則化で重みの肥大化を防止

## 10. スタッキングアンサンブル

### 10.1 仕組み

```
[訓練データ]
    ↓
[ベースモデル1, 2, 3, 4, 5, 6]（5-fold CVで予測）
    ↓
[Out-of-fold予測] → メタモデルの入力
    ↓
[メタモデル（Ridge）] → 最終予測
```

### 10.2 実装

```python
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = ...  # ベースモデルの予測を集約
        return self.meta_model_.predict(meta_features)

stacked_model = StackingAveragedModels(
    base_models=[ridge, lasso, elastic, xgboost, lightgbm, gb],
    meta_model=Ridge(alpha=10.0)
)
```

### 10.3 スタッキングの効果

| モデル | RMSLE |
|--------|-------|
| 最良単一モデル | 0.1080 |
| スタッキング | 0.1065 |
| 改善率 | 1.39% |

## 11. 特徴量重要度

```python
feature_imp = pd.DataFrame({
    'feature': X_train.columns,
    'importance': lightgbm.feature_importances_
}).sort_values('importance', ascending=False)
```

重要な特徴量 Top 5:

1. `OverallQual` — 全体的な品質
2. `GrLivArea` — 地上居住面積
3. `TotalSF` — 総床面積（作成した特徴量）
4. `GarageCars` — ガレージ収容台数
5. `OverallQual_TotalSF` — 品質×総床面積（作成した交互作用項）

自作の特徴量が上位にランクインしており、特徴量エンジニアリングの効果が確認できた。

## 12. 予測と提出

### 12.1 重み付きアンサンブル

CVスコアが良いモデルに高い重みを設定する。

```python
ensemble_pred = (
    0.50 * pred_stacked +      # スタッキング（最も重視）
    0.20 * pred_lasso +
    0.15 * pred_elastic +
    0.10 * pred_gb +
    0.05 * pred_ridge
)

final_predictions = np.expm1(ensemble_pred)
```

### 12.2 提出ファイル作成

```python
submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': final_predictions
})
submission.to_csv('submission.csv', index=False)
```

## 学んだ重要なポイント

### データクリーニング
- 外れ値は慎重に判断して除去
- 欠損値は特徴量の性質に応じて処理（None / 0 / 最頻値 / グループ別中央値）

### 特徴量エンジニアリング
- ドメイン知識（不動産の常識）を活用
- 集約、時系列、交互作用、比率など多様な視点で作成
- 20個以上の新特徴量を作成し、多くが重要度上位にランクイン

### 過学習対策
- 特徴量選択: 約半分の特徴量を除去してノイズ削減
- 正則化: パラメータ調整で複雑さを制御
- クロスバリデーション: 真の性能を測定
- CVとLBのギャップを過学習の指標として活用

### アンサンブル学習
- 多様なモデルを組み合わせて安定した予測を実現
- スタッキングは単純な平均より効果的
- 重み付けはCVスコアに基づいて調整

## 参考リソース

- [Kaggle: House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- [scikit-learn 公式ドキュメント](https://scikit-learn.org/)
- [XGBoost 公式ガイド](https://xgboost.readthedocs.io/)
- [LightGBM 公式ガイド](https://lightgbm.readthedocs.io/)
