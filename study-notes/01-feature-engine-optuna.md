# 特徴量エンジニアリング × Optuna × アンサンブル学習で0.78到達

Kaggle Titanicコンペティションにおける特徴量エンジニアリング、Optunaによるハイパーパラメータ最適化、アンサンブル学習の実践記録。

## サマリー

| 指標 | スコア |
|------|--------|
| CV（交差検証） | 0.8384 |
| LB（リーダーボード） | 0.78299 |
| 使用モデル | Random Forest + Gradient Boosting + LightGBM（ソフト投票） |

**CVとLBの差（約0.05）がオーバーフィッティングの証拠。** これが本記事の最大のテーマ。

---

## ライブラリの準備

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from lightgbm import LGBMClassifier
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
```

## データの読み込み

```python
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
all_data = pd.concat([train_df.drop('Survived', axis=1), test_df], sort=False).reset_index(drop=True)
```

- `train.csv`: 答え付き学習データ（891人）
- `test.csv`: 予測対象データ（418人）
- `all_data`: 特徴量作成を一括で行うために結合

## EDA（探索的データ分析）

### 欠損値

| カラム | 欠損数 |
|--------|--------|
| Age | 177 |
| Cabin | 687 |
| Embarked | 2 |

### 生存率の可視化

```python
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

sns.countplot(ax=axes[0, 0], x='Survived', data=train_df)
axes[0, 0].set_title('生存者数')

sns.countplot(ax=axes[0, 1], x='Sex', hue='Survived', data=train_df)
axes[0, 1].set_title('性別ごとの生存率')

sns.countplot(ax=axes[1, 0], x='Pclass', hue='Survived', data=train_df)
axes[1, 0].set_title('客室クラスごとの生存率')

sns.histplot(ax=axes[1, 1], data=train_df, x='Age', hue='Survived', kde=True)
axes[1, 1].set_title('年齢ごとの生存率')

plt.tight_layout()
plt.show()
```

### EDAの知見

- **性別**: 女性の生存率が圧倒的に高い（「女性と子供が優先」）
- **客室クラス**: 1等客室の生存率が高い
- **年齢**: 子供の生存率が高い傾向

## 特徴量エンジニアリング

### A. Title（敬称）の抽出

```python
all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

all_data['Title'] = all_data['Title'].replace(['Mlle', 'Ms'], 'Miss')
all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')

rare_titles = all_data['Title'].value_counts()[all_data['Title'].value_counts() < 10].index
all_data['Title'] = all_data['Title'].replace(rare_titles, 'Rare')
```

Titleは「性別」「年齢」「結婚状況」を含む強力な特徴量。例: `Mr.` = 成人男性、`Master.` = 少年。

### B. FamilySize（家族の人数）

```python
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
all_data['IsAlone'] = (all_data['FamilySize'] == 1).astype(int)
```

一人旅の乗客は生存率が低い傾向がある。

### C. Deck（デッキ階）

```python
all_data['Deck'] = all_data['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'U')
```

客室番号の先頭文字（A, B, C...）が船の階層を表す。高い階ほど脱出しやすい。`U`（Unknown）は客室番号不明。

### D. Ticket_Frequency（同一チケット番号の人数）

```python
ticket_counts = all_data['Ticket'].value_counts()
all_data['Ticket_Frequency'] = all_data['Ticket'].map(ticket_counts)
```

**今回の最も効果的な特徴量。** 同じチケット番号を持つ人はグループ（家族・友人）である可能性が高く、SibSp/Parchに含まれない関係者も捕捉できる。この特徴だけでスコアが大きく向上した。

### E. 欠損値の補完

```python
# 年齢：同じTitle & Pclassの中央値で補完
all_data['Age'] = all_data.groupby(['Title', 'Pclass'])['Age'].transform(
    lambda x: x.fillna(x.median())
)

# 料金：全体の中央値
all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].median())

# 乗船港：最頻値（S）
all_data['Embarked'] = all_data['Embarked'].fillna(all_data['Embarked'].mode()[0])
```

Ageの補完にTitle+Pclassを使う理由: `Mr.`と`Master.`では年齢分布が大きく異なるため。

### F. Fare_log（料金の対数変換）

```python
all_data['Fare_log'] = np.log1p(all_data['Fare'])
```

料金の分布は偏りが大きい（7ドル〜500ドル）ため、対数変換で正規分布に近づける。

### G. ビニング（カテゴリ化）

```python
all_data['AgeBin'] = pd.qcut(all_data['Age'], 5, labels=False, duplicates='drop')
all_data['FareBin'] = pd.qcut(all_data['Fare_log'], 4, labels=False, duplicates='drop')
```

連続値をグループに分けることで、モデルがパターンを学習しやすくなる。

## データの準備

```python
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Fare',
             'Fare_log', 'SibSp', 'Parch', 'FamilySize']
all_data_clean = all_data.drop(drop_cols, axis=1)

all_data_encoded = pd.get_dummies(
    all_data_clean,
    columns=['Title', 'Deck', 'Embarked', 'Sex'],
    drop_first=True
)

X_train = all_data_encoded[:len(train_df)]
X_test = all_data_encoded[len(train_df):]
y_train = train_df['Survived']
```

`drop_first=True`: 冗長な列を削除（例: male=1ならfemale=0は自明）。

## Optunaでハイパーパラメータ最適化

LightGBMのパラメータをOptunaで自動探索。50回の試行でCV Score 0.8406を達成。

```python
def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42
    }

    model = LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

## アンサンブル学習

3つのモデルをソフト投票（確率の平均）で組み合わせる。

```python
rf = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=42
)

gb = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.01, max_depth=4, random_state=42
)

lgbm_best = LGBMClassifier(**study.best_params, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('lgbm', lgbm_best)],
    voting='soft'
)
```

| モデル | 特徴 |
|--------|------|
| Random Forest | 多数の決定木で多数決。安定性が高い |
| Gradient Boosting | 弱いモデルを逐次改善。精度が高い |
| LightGBM | 高速・高精度。Optunaで最適化済み |

## 評価と予測

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
final_cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=cv, scoring='accuracy')

voting_clf.fit(X_train, y_train)
predictions = voting_clf.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions
})
submission.to_csv('submission_final.csv', index=False)
```

## オーバーフィッティング分析

### CVスコアとLBスコアの乖離

| スコア | 値 |
|--------|-----|
| CV | 0.8384 |
| LB | 0.78299 |
| **差** | **約0.05** |

この差はモデルが訓練データに過剰適応していることを示す。

### 過学習の原因

1. **特徴量が多すぎた**: 20個の特徴量は891人のデータに対して多い
2. **モデルが複雑すぎた**: 3モデルのアンサンブル + 高度なチューニング
3. **CVスコアへの過剰最適化**: CVスコアを上げることに集中しすぎた

### 対策

- **特徴量の削減**: 重要度の低い特徴は削除
- **正則化**: モデルの複雑さにペナルティを与える
- **シンプルなモデル**: 複雑すぎるモデルは避ける

## 学んだこと

1. **特徴量エンジニアリング**: `Ticket_Frequency`が最大の武器。創造的な特徴作成が精度を左右する
2. **Optuna**: 手動チューニングより効率的。50回の試行で最適解を発見
3. **アンサンブル**: 複数モデルの組み合わせで安定性向上。`voting='soft'`で柔軟な判定
4. **過学習（最重要）**: CVスコアだけを信じてはいけない。汎化性能が本当の実力
