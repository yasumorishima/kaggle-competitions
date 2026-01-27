🚀 Kaggle「Spaceship Titanic」で0.805を達成した方法【初心者向け完全ガイド】

Notebook

はじめに

この記事では、初心者の私が学んだことを、同じように学習中の方や未来の自分のために記録します。



目次

コンペティションの概要

データの理解

最重要：グループ情報の活用

特徴量エンジニアリング

モデルの構築

成功のポイント

失敗から学んだこと

まとめ

1. コンペティションの概要

問題設定

西暦2912年、宇宙船タイタニック号が時空の異常に遭遇し、約半数の乗客が異次元に転送されてしまいました。

タスク: 各乗客が異次元に転送されたか（Transported）を予測する二項分類問題

データの特徴

訓練データ: 8,693人

テストデータ: 4,277人

特徴量: 13個（PassengerId、年齢、支出額、船室番号など）

2. データの理解

2.1 データの基本構造

まず、データがどういう形式なのか理解することが最重要です。

train_df.head()


重要な発見

PassengerIdは「gggg_pp」形式

gggg: グループID

pp: グループ内番号

例: 0001_01、0001_02 → 同じグループ

Cabinは「Deck/Num/Side」形式

例: B/0/P → デッキB、0番、Port側

Nameから姓と名を分離できる

これらの情報を分解して使うことが成功の鍵でした！

2.2 欠損値の確認

# 欠損値の割合
missing_train = train_df.isnull().sum() / len(train_df) * 100


結果: 各列で2-3%程度の欠損値

💡 学んだこと: 単純に中央値や最頻値で埋めるのではなく、グループ情報を使って埋めると精度が上がる！

2.3 カテゴリカル変数の分析

重要な発見

CryoSleep=Trueの人は転送率が非常に高い

HomePlanetによって転送率が異なる

VIPステータスも影響あり

2.4 数値変数の分析

重要な発見

多くの人が各施設で$0しか使っていない　 → CryoSleepの人は支出が0になる

転送された人は支出が少ない傾向



3. 最重要：グループ情報の活用

なぜグループ情報が重要なのか？

PassengerIdから抽出したGroupIdが最も重要な特徴でした。

理由: 同じグループの人は一緒に旅行しているため：

同じHomePlanetから来ている

同じDestinationに向かっている

似た年齢層

同じCabinエリア

3.1 グループ特徴量の作成

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


💡 ポイント: 訓練データとテストデータを結合してからグループ統計を計算する！

3.2 グループ情報で欠損値を埋める

# HomePlanetの欠損値を埋める
# ステップ1: グループ内最頻値で埋める
df['HomePlanet'] = df['HomePlanet'].fillna(df['Group_HomePlanet'])

# ステップ2: まだ残っている欠損値を全体の最頻値で埋める
df['HomePlanet'] = df['HomePlanet'].fillna(df['HomePlanet'].mode()[0])


この方法でスコアが0.01〜0.02向上しました！

4. 特徴量エンジニアリング

4.1 基本的な特徴量

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


4.2 高度な特徴量

ドメイン知識を活用

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


4.3 相関分析

💡 発見: 支出関連の変数同士に相関があるが、それぞれ独自の情報も持っている

5. モデルの構築

5.1 複数モデルの比較

試したモデル：

Random Forest

XGBoost

LightGBM ⭐最高性能

Gradient Boosting

Logistic Regression

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


5.2 結果

モデル CVスコア 

LightGBM 0.8087XGBoost 0.8062Gradient Boosting 0.8050Random Forest 0.8012Logistic Regression 0.7902

💡 学んだこと: LightGBMが最も安定して高性能！

5.3 特徴量の重要度

トップ10の重要な特徴量

SpendingStd - 支出の標準偏差

MaxSpendingCategory - 最も多く使った施設

TotalSpending - 総支出額

SpendingPerAge - 年齢あたりの支出

NumFacilitiesUsed - 利用施設数

HasSpending - 支出があるか

FoodCourt - フードコート支出

Cabin_Num - 部屋番号

Spa - スパ支出

ShoppingMall - ショッピングモール支出

共通点: 支出関連の特徴量が多い！

5.4 アンサンブル

複数のモデルを組み合わせることで、より安定した予測ができます。

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


6. 成功のポイント

✅ 1. グループ情報の徹底活用

効果: スコア +0.01〜0.02

PassengerIdから抽出したGroupIdを使って：

グループ内統計（平均年齢、総支出など）

グループ内最頻値で欠損値を埋める

グループサイズ

これが最大の勝因！

✅ 2. CryoSleepと支出の関係

発見: CryoSleep=Trueの人は支出が0

この関係を使って：

欠損値の推定精度向上

矛盾フラグの作成

特徴量の相互作用

✅ 3. LightGBMの安定性

理由:

ハイパーパラメータに鈍感

過学習しにくい

XGBoostより安定

基本パラメータで十分な性能が出ました。

✅ 4. シンプルさの勝利

結果:

38特徴量: 0.80336

50特徴量: 0.80523

差はわずか +0.00187

💡 教訓: 複雑な特徴量より、本質的な特徴量が重要

7. 失敗から学んだこと

❌ 1. Optunaの罠

問題

ハイパーパラメータ自動最適化（Optuna）を使って最適化した結果：

CVスコア: 0.8123（高い！）

実際のKaggleスコア: 0.80523

差分: -0.007

原因

訓練データに過度に最適化されて、汎化性能が落ちた（過学習）

教訓

基本パラメータで十分。過度な最適化は逆効果。

❌ 2. スタッキングの期待外れ

結果

スコア: 0.79798（最低スコア）

理論上は強力だが：

計算時間がかかる

過学習リスクが高い

単純なLightGBMに負けた

教訓

複雑 ≠ 高性能

❌ 3. 提出ファイルのフォーマット

トラブル

Transported列を整数型で保存 → スコア 0.00

解決策

# ❌ 間違い
'Transported': pred  # 0/1の整数

# ✅ 正解
'Transported': pred.astype(bool)  # True/Falseのbool型


💡 重要: Kaggleはbool型を期待している！

❌ 4. 複雑な特徴量の限界

試したが効果薄だったもの

Age²、Age³（非線形性を捉える）

支出の偏り指標

過度な相互作用

教訓

ドメイン知識に基づく特徴量 > 機械的な特徴量



8. まとめ

🏆 成功の3本柱

グループ情報 - PassengerIdの徹底活用

シンプルさ - 本質的な特徴量に集中

LightGBM - 安定した高性能モデル

📈 スコアの推移

開始時:
├─ 基本的な提出: 0.70台（推定）
│
グループ情報活用:
├─ 初回LightGBM: 0.80336 ✅
│
最適化:
├─ LightGBM最適化: 0.80523 ⭐ ← 最高スコア
├─ Ensemble最適化: 0.80500
├─ XGBoost最適化: 0.80266
└─ スタッキング: 0.79798 ❌


💡 学んだ最大の教訓

「複雑さよりも本質」

50個の特徴量より、38個の良質な特徴量 Optunaより、基本パラメータ スタッキングより、シンプルなLightGBM

🎓 初心者へのアドバイス

データを理解する

まずグラフで可視化

欠損値の確認

ターゲットとの関係を見る

ドメイン知識を活用する

データの意味を考える

CryoSleepと支出の関係のような発見が重要

シンプルから始める

複雑なモデルより、まず基本

過学習に注意

グループ情報は宝の山

IDから情報を抽出できないか考える

集約統計が強力

クロスバリデーションで評価

訓練データだけで評価しない

CVスコアと実際のスコアの差に注意

参考リンク

Kaggle: Spaceship Titanic

おわりに

初めてのKaggleコンペで、試行錯誤しながら0.805を達成できました。

この記事が、同じように学習中の方や、未来の自分の復習に役立てば嬉しいです。

最も大切なこと:

データを理解すること

シンプルに考えること

失敗から学ぶこと

Happy Kaggling! 🚀✨

最終更新: 2025年11月