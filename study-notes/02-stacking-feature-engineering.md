Kaggle「住宅価格予測」で学ぶ機械学習の基礎 - 初心者のための完全ガイド

Notebook

はじめに

こんにちは！今回はKaggleの有名なコンペティション「House Prices - Advanced Regression Techniques」に挑戦した学習記録をまとめました。

このコンペは回帰問題の入門として最適で、データクリーニングから特徴量エンジニアリング、モデル構築まで、実務で使える技術を一通り学べます。

この記事で学べること

✅ 外れ値の検出と適切な処理方法✅ 欠損値を特徴量の性質に応じて処理する方法✅ ドメイン知識を活用した20種類以上の特徴量作成✅ 過学習を防ぐための具体的なテクニック✅ 6つのモデルを使ったアンサンブル学習の実装

1. 使用するライブラリと環境設定

まずは必要なライブラリをインストールします。特に日本語でグラフを表示するためにjapanize-matplotlibが重要です。

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


重要ポイント💡

日本語フォントの設定は順序が大切です。sns.set(font='IPAexGothic')を実行することで、Seabornのグラフでも日本語が正しく表示されます。

2. データの読み込みと基本的な理解

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f'訓練データ: {train.shape}')  # (1460, 81)
print(f'テストデータ: {test.shape}')  # (1459, 80)


訓練データには1460件の住宅データがあり、81個の特徴量（変数）が含まれています。目的変数はSalePrice（販売価格）です。

3. 探索的データ分析（EDA）- データを「見る」

3.1 外れ値の発見

データを可視化することで、異常な値を持つデータポイント（外れ値）を発見できます。

# 地上居住面積 vs 販売価格の散布図
plt.scatter(train['GrLivArea'], train['SalePrice'])
plt.xlabel('地上居住面積')
plt.ylabel('販売価格')
plt.title('外れ値の確認')
plt.show()

（赤い点で外れ値を強調したグラフ）

このグラフから、面積が4000平方フィート以上なのに価格が30万ドル未満の物件が見つかりました。これは明らかに異常値です。

3.2 目的変数の分布確認

# 販売価格の分布
sns.histplot(train['SalePrice'], kde=True)
plt.title('販売価格の分布')
plt.show()

# 対数変換後
sns.histplot(np.log1p(train['SalePrice']), kde=True)
plt.title('販売価格の分布（対数変換後）')
plt.show()

学んだこと📚

元のデータは右に偏った分布（歪度: 1.88）

対数変換により正規分布に近づく（歪度: 0.12）

RMSLEで評価される問題では対数変換が必須

3.3 相関分析

# 販売価格と相関の高い特徴量 Top 10
correlations = train.corr()['SalePrice'].sort_values(ascending=False)
print(correlations.head(11))

最も相関が高いのは：

OverallQual（全体的な品質）: 0.79

GrLivArea（地上居住面積）: 0.71

GarageCars（ガレージ収容台数）: 0.64

4. 外れ値の除去 - モデルの精度を高める第一歩

print(f'外れ値除去前: {train.shape[0]}件')  # 1460件

# 1. 面積は大きいのに価格が異常に低い物件
train = train.drop(train[(train['GrLivArea'] > 4000) & 
                        (train['SalePrice'] < 300000)].index)

# 2. 地下室面積が極端に大きい物件
train = train.drop(train[train['TotalBsmtSF'] > 3000].index)

# 3. 土地面積が極端に大きい物件
train = train.drop(train[train['LotArea'] > 100000].index)

print(f'外れ値除去後: {train.shape[0]}件')  # 約1454件


なぜ外れ値を除去するのか？🤔

外れ値はデータ入力ミスや特殊な状況であることが多く、モデルが正しいパターンを学習する妨げになります。慎重に判断して除去することで、予測精度が向上します。

5. 欠損値処理 - 特徴量の性質を理解する

欠損値処理は「単純に平均値で埋める」だけではありません。特徴量の意味を考えて適切な方法を選ぶことが重要です。

5.1 訓練データとテストデータの結合

# 特徴量エンジニアリングを一貫して行うため結合
ntrain = train.shape[0]
y_train = train['SalePrice'].values
all_data = pd.concat([train.drop('SalePrice', axis=1), test], axis=0)


5.2 4つの欠損値処理パターン

パターン1: 'None'で埋める

「ない」ことが意味を持つ特徴量（プール、ガレージ、地下室など）

none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
             'GarageType', 'BsmtQual', 'BsmtCond', ...]
for col in none_cols:
    all_data[col] = all_data[col].fillna('None')


パターン2: 0で埋める

数値で「ない」を表現できる特徴量

zero_cols = ['GarageArea', 'GarageCars', 'TotalBsmtSF', 'MasVnrArea', ...]
for col in zero_cols:
    all_data[col] = all_data[col].fillna(0)


パターン3: 最頻値で埋める

カテゴリ変数

mode_cols = ['MSZoning', 'Electrical', 'KitchenQual', ...]
for col in mode_cols:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])


パターン4: グループ別中央値

地域性のある特徴量（道路までの距離など）

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))


学んだこと📚

プールがない家のPoolQC（プールの品質）は「None」とする

ガレージがない家のGarageArea（ガレージ面積）は「0」とする

近隣の値を参考にすることで、より現実的な値を補完できる

6. 特徴量エンジニアリング - 予測精度を上げる最大の武器

特徴量エンジニアリングとは、既存のデータから新しい有用な特徴量を作り出す作業です。これがモデルの性能を大きく左右します。

6.1 集約特徴量（合計・平均）

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


6.2 時系列特徴量

# 家の築年数
all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']

# リフォームからの年数
all_data['YearsSinceRemod'] = all_data['YrSold'] - all_data['YearRemodAdd']

# ガレージの年齢
all_data['GarageAge'] = all_data['YrSold'] - all_data['GarageYrBlt']


6.3 バイナリ特徴量（あり/なし）

# 新築かどうか
all_data['IsNew'] = (all_data['YearBuilt'] == all_data['YrSold']).astype(int)

# リフォームしたかどうか
all_data['HasRemod'] = (all_data['YearBuilt'] != all_data['YearRemodAdd']).astype(int)

# 2階があるか
all_data['Has2ndFloor'] = (all_data['2ndFlrSF'] > 0).astype(int)

# 地下室、ガレージ、プール、暖炉があるか
all_data['HasBsmt'] = (all_data['TotalBsmtSF'] > 0).astype(int)
all_data['HasGarage'] = (all_data['GarageArea'] > 0).astype(int)
all_data['HasPool'] = (all_data['PoolArea'] > 0).astype(int)
all_data['HasFireplace'] = (all_data['Fireplaces'] > 0).astype(int)


6.4 交互作用項（最重要！）

2つの特徴量を掛け合わせることで、相乗効果を表現します。

# 品質 × 総床面積（高品質で広い家は特に高価）
all_data['OverallQual_TotalSF'] = all_data['OverallQual'] * all_data['TotalSF']

# 品質 × 居住面積
all_data['OverallQual_GrLivArea'] = all_data['OverallQual'] * all_data['GrLivArea']

# 総合品質（品質 + 状態）
all_data['TotalQual'] = all_data['OverallQual'] + all_data['OverallCond']


6.5 比率特徴量

# 地下室の割合（地下室面積 / 総床面積）
all_data['Bsmt_Ratio'] = all_data['TotalBsmtSF'] / (all_data['TotalSF'] + 1)

# ガレージの割合
all_data['Garage_Ratio'] = all_data['GarageArea'] / (all_data['TotalSF'] + 1)

# 1部屋あたりの面積
all_data['AreaPerRoom'] = all_data['GrLivArea'] / (all_data['TotRmsAbvGrd'] + 1)


6.6 カテゴリのグループ化

地域を価格帯別にグループ化することで、カテゴリ数を減らし過学習を防ぎます。

# 各地域の価格中央値を計算
neighborhood_price = train.groupby('Neighborhood')['SalePrice'].median()

# 3つのグループに分類（Low, Medium, High）
def categorize_neighborhood(neighborhood):
    price = neighborhood_price[neighborhood]
    if price < neighborhood_price.quantile(0.33):
        return 'Low'
    elif price < neighborhood_price.quantile(0.67):
        return 'Medium'
    else:
        return 'High'

all_data['NeighborhoodGroup'] = all_data['Neighborhood'].apply(categorize_neighborhood)


学んだこと📚

ドメイン知識が重要：不動産の常識を活用

交互作用項は強力：ただし過学習のリスクもある

新しい視点：合計、比率、時系列など多様な角度で特徴量を作成

7. 特徴量変換 - データの分布を正規化する

7.1 目的変数の対数変換

# RMSLEで評価されるため、対数変換は必須
y_train = np.log1p(y_train)


7.2 Box-Cox変換

歪んだ分布を持つ特徴量を正規分布に近づけます。

# 歪度（skewness）を計算
numeric_feats = all_data.select_dtypes(include=[np.number]).columns
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))

# 歪度が0.75以上の特徴量にBox-Cox変換を適用
skewed_features = skewed_feats[abs(skewed_feats) > 0.75].index

lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)


7.3 One-Hot Encoding

カテゴリ変数を数値に変換します。

all_data = pd.get_dummies(all_data)
print(f'エンコーディング後: {all_data.shape[1]}列')  # 約330列


学んだこと📚

対数変換により歪度が0に近づく（正規分布に近くなる）

正規分布に近いデータは、モデルが学習しやすい

One-Hot Encodingで「NeighborhoodがStoneBr」→「Neighborhood_StoneBr=1」に変換

8. 特徴量選択 - 過学習を防ぐ重要なステップ

特徴量が多すぎると、モデルが訓練データに過適合（過学習）してしまいます。

8.1 LightGBMで重要度を計算

# データを分割
X_train = all_data[:ntrain]
X_test = all_data[ntrain:]

# 特徴量重要度の計算
lgb_selector = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
lgb_selector.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': lgb_selector.feature_importances_
}).sort_values('importance', ascending=False)


8.2 重要度の低い特徴量を除去

# 重要度が5以上の特徴量のみを使用
threshold = 5
important_features = feature_importance[
    feature_importance['importance'] > threshold
]['feature'].tolist()

print(f'元の特徴量数: {X_train.shape[1]}')      # 約330列
print(f'選択後: {len(important_features)}')      # 約150列
print(f'除去: {X_train.shape[1] - len(important_features)}')  # 約180列

X_train = X_train[important_features]
X_test = X_test[important_features]


学んだこと📚

約半分の特徴量を除去しても精度は下がらない

重要度の低い特徴量は「ノイズ」となり精度を下げる

CVスコアとLBスコアのギャップが縮小（過学習が減少）

9. モデル構築 - 6つの異なるアプローチ

9.1 クロスバリデーション関数

def rmsle_cv(model, X, y, n_folds=5):
    """5-fold CVでRMSLEを計算"""
    kf = KFold(n_folds, shuffle=True, random_state=42)
    rmse = np.sqrt(-cross_val_score(
        model, X, y, 
        scoring='neg_mean_squared_error', 
        cv=kf
    ))
    return rmse


9.2 線形モデル（正則化付き）

# Ridge回帰（L2正則化）
ridge = Ridge(alpha=15.0, random_state=42)
ridge_scores = rmsle_cv(ridge, X_train, y_train)

# Lasso回帰（L1正則化 + 特徴量選択）
lasso = Lasso(alpha=0.0005, random_state=42, max_iter=10000)
lasso_scores = rmsle_cv(lasso, X_train, y_train)

# ElasticNet（L1 + L2正則化）
elastic = ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=42, max_iter=10000)
elastic_scores = rmsle_cv(elastic, X_train, y_train)


9.3 勾配ブースティングモデル（過学習対策強化版）

# XGBoost
xgboost = xgb.XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=3,           # 木の深さを制限
    min_child_weight=3,    # ノード分割を厳しく
    gamma=0.1,             # ゲイン閾値
    subsample=0.6,         # サンプリング率
    colsample_bytree=0.6,  # 特徴量サンプリング
    reg_alpha=0.0001,      # L1正則化
    reg_lambda=2,          # L2正則化
    random_state=42
)
xgb_scores = rmsle_cv(xgboost, X_train, y_train)

# LightGBM
lightgbm = lgb.LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=3,
    num_leaves=8,          # 葉の数を制限
    min_child_samples=30,  # 最小サンプル数
    subsample=0.6,
    colsample_bytree=0.6,
    reg_alpha=0.2,         # L1正則化
    reg_lambda=0.2,        # L2正則化
    random_state=42
)
lgb_scores = rmsle_cv(lightgbm, X_train, y_train)

# Gradient Boosting
gb = GradientBoostingRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=8,
    subsample=0.7,
    random_state=42
)
gb_scores = rmsle_cv(gb, X_train, y_train)


9.4 モデル性能の比較

model_scores = pd.DataFrame({
    'モデル': ['Ridge', 'Lasso', 'ElasticNet', 'XGBoost', 'LightGBM', 'GradientBoosting'],
    '平均RMSLE': [...],
    '標準偏差': [...]
}).sort_values('平均RMSLE')


過学習を防ぐパラメータ設定のポイント💡

XGBoost/LightGBM

max_depth=3: 木の深さを浅くする → 複雑なパターンを学習しすぎない

subsample=0.6: データの60%だけ使って学習 → ランダム性を持たせる

reg_lambda=2: L2正則化を強化 → 重みが大きくなりすぎるのを防ぐ

Ridge/Lasso/ElasticNet

alpha: 大きいほど正則化が強い → 過学習を防ぐ

学んだこと📚

多様なモデルを試すことで、データに最適なモデルを見つける

正則化パラメータの調整が過学習対策の鍵

クロスバリデーションで真の性能を測定

10. スタッキングアンサンブル - 複数モデルの力を合わせる

スタッキングとは、複数のベースモデルの予測を組み合わせて、メタモデルが最終予測を行う手法です。

10.1 スタッキングの仕組み

[訓練データ]
    ↓
[ベースモデル1, 2, 3, 4, 5, 6]（5-fold CVで予測）
    ↓
[Out-of-fold予測] → これをメタモデルの入力とする
    ↓
[メタモデル（Ridge）] → 最終予測


10.2 実装

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    """スタッキングアンサンブル実装"""
    
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    def fit(self, X, y):
        # Out-of-fold予測の生成
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
        
        # メタモデルの学習
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    def predict(self, X):
        meta_features = ...  # ベースモデルの予測を集約
        return self.meta_model_.predict(meta_features)

# スタッキングモデルの構築
stacked_model = StackingAveragedModels(
    base_models=[ridge, lasso, elastic, xgboost, lightgbm, gb],
    meta_model=Ridge(alpha=10.0)
)

stacking_scores = rmsle_cv(stacked_model, X_train, y_train)


10.3 スタッキングの効果

最良の単一モデル: 0.1080
スタッキング: 0.1065
改善率: 1.39%


学んだこと📚

スタッキングは単一モデルより安定した予測が可能

各モデルの強みを活かし、弱点を補完できる

Out-of-fold予測により過学習を防ぐ

11. 特徴量重要度の確認

# LightGBMの特徴量重要度を可視化
feature_imp = pd.DataFrame({
    'feature': X_train.columns,
    'importance': lightgbm.feature_importances_
}).sort_values('importance', ascending=False)

# 上位20をプロット
plt.barh(range(20), feature_imp.head(20)['importance'])
plt.yticks(range(20), feature_imp.head(20)['feature'])
plt.title('LightGBM - 特徴量重要度 Top 20')
plt.show()

重要な特徴量 Top 5

OverallQual - 全体的な品質

GrLivArea - 地上居住面積

TotalSF - 総床面積（作成した特徴量！）

GarageCars - ガレージ収容台数

OverallQual_TotalSF - 品質×総床面積（作成した交互作用項！）

自分で作成した特徴量が上位にランクインしています！特徴量エンジニアリングの効果が確認できました。

12. 予測と提出

12.1 重み付きアンサンブル

CVスコアが良いモデルに高い重みを設定します。

# 各モデルで予測
pred_ridge = ridge.predict(X_test)
pred_lasso = lasso.predict(X_test)
pred_elastic = elastic.predict(X_test)
pred_xgb = xgboost.predict(X_test)
pred_lgb = lightgbm.predict(X_test)
pred_gb = gb.predict(X_test)
pred_stacked = stacked_model.predict(X_test)

# 重み付きアンサンブル
ensemble_pred = (
    0.50 * pred_stacked +      # スタッキング（最も重視）
    0.20 * pred_lasso +        # CVスコア良好
    0.15 * pred_elastic +      # CVスコア良好
    0.10 * pred_gb +           # CVスコア良好
    0.05 * pred_ridge          # 安定性のため
)

# 対数変換を元に戻す
final_predictions = np.expm1(ensemble_pred)


12.2 提出ファイル作成

submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': final_predictions
})
submission.to_csv('submission.csv', index=False)


まとめ - 学んだ重要なポイント

1. データクリーニングの重要性

✅ 外れ値は慎重に判断して除去✅ 欠損値は特徴量の性質に応じて処理

2. 特徴量エンジニアリングが性能の鍵

✅ ドメイン知識（不動産の常識）を活用✅ 集約、時系列、交互作用、比率など多様な視点で作成✅ 20個以上の新特徴量を作成し、多くが重要度上位に

3. 過学習との戦い

✅ 特徴量選択: 約半分の特徴量を除去してノイズ削減✅ 正則化: パラメータ調整で複雑さを制御✅ クロスバリデーション: 真の性能を測定✅ CVとLBのギャップ: 過学習の指標

4. アンサンブル学習の力

✅ 多様なモデルを組み合わせて安定した予測✅ スタッキングは単純な平均より効果的✅ 重み付けはCVスコアに基づいて調整

5. 実務で使えるベストプラクティス

✅ 再現性の確保: random_stateを固定✅ 段階的な改善: 一度に多くを変えない✅ 可視化: グラフでデータを理解✅ コメント: 理由を記録して後から見返せるように

さらなる改善のアイデア

中級者向け

ハイパーパラメータのグリッドサーチ（GridSearchCV）

特徴量選択の閾値を最適化

アンサンブル重みの自動最適化

上級者向け

SHAP値による特徴量の詳細分析

2段階スタッキング

ニューラルネットワークの追加

Optunaによる高度なハイパーパラメータ最適化

おわりに

このコンペを通じて、機械学習の実践的なワークフローを一通り学ぶことができました。

最も重要なのは「データを理解すること」　です。可視化や相関分析を通じてデータの特性を把握し、ドメイン知識を活用することで、効果的な特徴量を作成できます。

また、過学習対策は実務でも非常に重要です。特徴量選択、正則化、クロスバリデーションなど、様々なテクニックを組み合わせて使いこなせるようになりましょう。

皆さんもぜひKaggleに挑戦してみてください！

参考リソース

Kaggle: House Prices Competition

scikit-learn 公式ドキュメント

XGBoost 公式ガイド

LightGBM 公式ガイド

