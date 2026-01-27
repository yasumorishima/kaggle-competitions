# Kaggle「Spaceship Titanic」で0.805を達成した方法

## サマリー

| 項目 | 内容 |
|------|------|
| タスク | 乗客が異次元に転送されたかの二項分類 |
| 最高スコア | **0.80523**（LightGBM） |
| 最大の勝因 | PassengerIdからのグループ情報活用（+0.01〜0.02） |
| 主な教訓 | 複雑さよりも本質的な特徴量が重要 |

---

## 1. コンペティションの概要

西暦2912年、宇宙船タイタニック号が時空の異常に遭遇し、約半数の乗客が異次元に転送された。各乗客が転送されたか（Transported）を予測する二項分類問題。

- 訓練データ: 8,693人
- テストデータ: 4,277人
- 特徴量: 13個（PassengerId、年齢、支出額、船室番号など）

## 2. データの理解

### 2.1 データの基本構造

```python
train_df.head()
```

**重要な発見**:

- **PassengerId**は「gggg_pp」形式（gggg: グループID、pp: グループ内番号）
  - 例: 0001_01、0001_02 → 同じグループ
- **Cabin**は「Deck/Num/Side」形式
  - 例: B/0/P → デッキB、0番、Port側
- **Name**から姓と名を分離できる

### 2.2 欠損値の確認

```python
missing_train = train_df.isnull().sum() / len(train_df) * 100
```

各列で2-3%程度の欠損値。単純に中央値や最頻値で埋めるのではなく、グループ情報を使って埋めると精度が上がる。

### 2.3 カテゴリカル変数の分析

- CryoSleep=Trueの人は転送率が非常に高い
- HomePlanetによって転送率が異なる
- VIPステータスも影響あり

### 2.4 数値変数の分析

- 多くの人が各施設で$0しか使っていない（CryoSleepの人は支出が0になる）
- 転送された人は支出が少ない傾向

## 3. グループ情報の活用（最重要）

PassengerIdから抽出したGroupIdが最も重要な特徴。同じグループの人は一緒に旅行しているため、同じHomePlanet・Destination・Cabinエリア、似た年齢層を持つ。

### 3.1 グループ特徴量の作成

```python
def create_group_features(train_df, test_df):
    # 訓練とテストを結合（重要！）
    all_data = pd.concat([train_df, test_df], axis=0)

    # グループサイズ
    group_size = all_data.groupby('GroupId').size()
    train_df['GroupSize'] = train_df['GroupId'].map(group_size)

    # グループ内平均年齢
    group_age_mean = all_data.groupby('GroupId')['Age'].mean()
    train_df['Group_Age_Mean'] = train_df['GroupId'].map(group_age_mean)

    # グループ内総支出
    group_spending = all_data.groupby('GroupId')['TotalSpending'].sum()
    train_df['Group_TotalSpending'] = train_df['GroupId'].map(group_spending)

    return train_df, test_df
```

**ポイント**: 訓練データとテストデータを結合してからグループ統計を計算する。

### 3.2 グループ情報で欠損値を埋める

```python
# HomePlanetの欠損値を埋める
# ステップ1: グループ内最頻値で埋める
df['HomePlanet'] = df['HomePlanet'].fillna(df['Group_HomePlanet'])

# ステップ2: まだ残っている欠損値を全体の最頻値で埋める
df['HomePlanet'] = df['HomePlanet'].fillna(df['HomePlanet'].mode()[0])
```

この方法でスコアが0.01〜0.02向上した。

## 4. 特徴量エンジニアリング

### 4.1 基本的な特徴量

```python
# PassengerIdを分解
df['GroupId'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
df['GroupNum'] = df['PassengerId'].apply(lambda x: int(x.split('_')[1]))

# Cabinを分解
df['Cabin_Deck'] = df['Cabin'].apply(lambda x: x.split('/')[0] if pd.notna(x) else np.nan)
df['Cabin_Num'] = df['Cabin'].apply(lambda x: int(x.split('/')[1]) if pd.notna(x) else np.nan)
df['Cabin_Side'] = df['Cabin'].apply(lambda x: x.split('/')[2] if pd.notna(x) else np.nan)

# 総支出額
spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df['TotalSpending'] = df[spending_cols].sum(axis=1)
```

### 4.2 高度な特徴量

```python
# CryoSleepと支出の矛盾フラグ
df['CryoSleep_Spending_Conflict'] = (
    ((df['CryoSleep'] == True) & (df['TotalSpending'] > 0)) |
    ((df['CryoSleep'] == False) & (df['TotalSpending'] == 0))
).astype(int)

# 利用施設数
for col in spending_cols:
    df[f'{col}_Used'] = (df[col] > 0).astype(int)
df['NumFacilitiesUsed'] = df[[f'{col}_Used' for col in spending_cols]].sum(axis=1)

# 支出の標準偏差
df['SpendingStd'] = df[spending_cols].std(axis=1).fillna(0)
```

### 4.3 相関分析

支出関連の変数同士に相関があるが、それぞれ独自の情報も持っている。

## 5. モデルの構築

### 5.1 複数モデルの比較

```python
# LightGBM（基本パラメータ）
lgb_model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# クロスバリデーション
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(lgb_model, X, y, cv=skf, scoring='accuracy')
print(f"CV Score: {scores.mean():.4f}")
```

### 5.2 結果

| モデル | CVスコア |
|--------|----------|
| LightGBM | 0.8087 |
| XGBoost | 0.8062 |
| Gradient Boosting | 0.8050 |
| Random Forest | 0.8012 |
| Logistic Regression | 0.7902 |

LightGBMが最も安定して高性能。

### 5.3 特徴量の重要度（トップ10）

1. SpendingStd - 支出の標準偏差
2. MaxSpendingCategory - 最も多く使った施設
3. TotalSpending - 総支出額
4. SpendingPerAge - 年齢あたりの支出
5. NumFacilitiesUsed - 利用施設数
6. HasSpending - 支出があるか
7. FoodCourt - フードコート支出
8. Cabin_Num - 部屋番号
9. Spa - スパ支出
10. ShoppingMall - ショッピングモール支出

支出関連の特徴量が上位を占めている。

### 5.4 アンサンブル

```python
# 重み付きアンサンブル（CVスコアに基づく）
weights = np.array([rf_mean, xgb_mean, lgb_mean, gb_mean, lr_mean])
weights = weights / weights.sum()

ensemble_pred = (
    pred_rf.astype(int) * weights[0] +
    pred_xgb.astype(int) * weights[1] +
    pred_lgb.astype(int) * weights[2] +
    pred_gb.astype(int) * weights[3] +
    pred_lr.astype(int) * weights[4]
)

ensemble_pred_final = (ensemble_pred >= 0.5)
```

## 6. 成功のポイント

### グループ情報の徹底活用（効果: +0.01〜0.02）

PassengerIdから抽出したGroupIdを使って、グループ内統計（平均年齢、総支出など）の作成、グループ内最頻値での欠損値補完、グループサイズの特徴量化を行った。これが最大の勝因。

### CryoSleepと支出の関係

CryoSleep=Trueの人は支出が0という関係を利用して、欠損値の推定精度向上、矛盾フラグの作成、特徴量の相互作用を実現。

### シンプルさの勝利

- 38特徴量: 0.80336
- 50特徴量: 0.80523
- 差はわずか +0.00187

複雑な特徴量より、本質的な特徴量が重要。

## 7. 失敗から学んだこと

### Optunaの罠

Optunaで最適化した結果、CVスコア0.8123と高くなったが、Kaggleスコアは0.80523。訓練データに過度に最適化され汎化性能が低下した（過学習）。基本パラメータで十分。

### スタッキングの期待外れ

スコア0.79798（最低スコア）。理論上は強力だが、計算時間がかかり過学習リスクが高く、単純なLightGBMに負けた。複雑 ≠ 高性能。

### 提出ファイルのフォーマット

Transported列を整数型で保存するとスコア0.00になる。

```python
# 間違い
'Transported': pred  # 0/1の整数

# 正解
'Transported': pred.astype(bool)  # True/Falseのbool型
```

Kaggleはbool型を期待している。

### 複雑な特徴量の限界

Age²、Age³、支出の偏り指標、過度な相互作用などは効果薄。ドメイン知識に基づく特徴量の方が機械的な特徴量より有効。

## 8. スコアの推移

```
開始時:
├─ 基本的な提出: 0.70台（推定）

グループ情報活用:
├─ 初回LightGBM: 0.80336

最適化:
├─ LightGBM最適化: 0.80523  ← 最高スコア
├─ Ensemble最適化: 0.80500
├─ XGBoost最適化: 0.80266
└─ スタッキング: 0.79798
```

## 参考リンク

- [Kaggle: Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)

---

最終更新: 2025年11月
