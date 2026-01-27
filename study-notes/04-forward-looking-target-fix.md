# Forward-Looking Targetの落とし穴と修正 — 三井物産コモディティ予測コンペ

## サマリー

- **問題**: ターゲット計算がbackward-looking（過去参照）になっており、スコアが **-0.058**（負の相関）
- **原因**: `log(price[t] / price[t-lag])` と過去を参照していた。正しくは `log(price[t+lag+1] / price[t+1])` で未来を予測する
- **修正後**: スコアが正の値へ改善
- **教訓**: 複雑なモデルより正しい問題理解が最重要

---

## コンペの概要

- **課題**: 424種類のコモディティペア（例：銅と亜鉛の価格差）の対数リターンを予測
- **評価指標**: スピアマン順位相関のシャープレシオ（予測の安定性を重視）
- **特殊性**: 提出締切後、実際の市場データで3.5ヶ月間評価される「フォワードランニング評価」

## 直面した問題：スコアが -0.058

最初の提出でスコアが -0.058 という負の値になった。

```
スピアマン相関: -0.058
→ 予測が実際と逆相関している状態
```

単に精度が低いのではなく、「予測の方向性が間違っている」ことを意味する。

## バグの正体：時間軸の誤解

問題はターゲット（予測すべき値）の計算方法にあった。

### ❌ 間違った実装（Backward-looking）

```python
def generate_log_returns_wrong(data, lag):
    """
    間違い：過去を見ている
    target[t] = log(price[t] / price[t-lag])
    """
    log_returns = pd.Series(np.nan, index=data.index)

    for t in range(lag, len(data)):
        # 時刻tで、過去のlag期間前との比較
        log_returns.iloc[t] = np.log(data.iloc[t] / data.iloc[t - lag])

    return log_returns
```

この実装では、「時刻tでの予測値」が「時刻tから過去を見た対数リターン」になっている。時刻tの時点で過去のデータは既に分かっているため、ターゲットとして意味がない。

### ✅ 正しい実装（Forward-looking）

```python
def generate_log_returns_corrected(data, lag):
    """
    正解：未来を予測する
    target[t] = log(price[t+lag+1] / price[t+1])
    """
    log_returns = pd.Series(np.nan, index=data.index)

    # CRITICAL FIX: Forward-looking calculation
    for t in range(len(data) - lag - 1):
        # 時刻tで、未来のlag期間後を予測
        log_returns.iloc[t] = np.log(data.iloc[t + lag + 1] / data.iloc[t + 1])

    return log_returns
```

「時刻tでの予測値」は「時刻t+1からt+lag+1までの対数リターン」であるべき。未来の価格変動を予測するのが本来のタスク。

### 図解：時間軸の違い

```
時系列データ: [price[0], price[1], price[2], price[3], price[4], ...]

❌ Backward-looking (間違い):
時刻t=3で予測 → log(price[3] / price[1])  # lag=2の場合
                 ↑ 過去を見ている

✅ Forward-looking (正解):
時刻t=1で予測 → log(price[4] / price[2])  # lag=2の場合
                 ↑ 未来を予測している
```

## ミスの要因

1. **金融時系列の経験不足** — 一般的な時系列予測と金融予測の違いを理解していなかった。「対数リターン」の定義を曖昧に理解していた
2. **ドキュメントの読み込み不足** — Data Descriptionやサンプルコードを十分に確認しなかった
3. **検証の甘さ** — 「コードが動いている = 正しい」と思い込み、スコアの意味（負の相関）を深く考えなかった

## 修正後のアプローチ

ターゲット計算を修正した上で、シンプルなモデルを実装した。

```python
def predict_target_corrected(target_id, current_features, lag_context=None):
    """
    正しいターゲット理解に基づくシンプルな予測
    """
    # 1. 過去の統計情報を取得
    stats = global_target_stats[target_id]
    base_mean = stats['mean']
    base_std = stats['std']

    # 2. 平均回帰モデル
    if lag_context and len(lag_context) > 0:
        recent_values = extract_recent_values(lag_context, target_id)
        if recent_values:
            recent_mean = np.mean(recent_values)
            # 平均回帰：最近の値が平均から乖離していたら、戻る方向に予測
            momentum = -recent_mean * 0.1

    # 3. ランダムノイズ
    noise = np.random.normal(0, base_std * 0.8)

    # 4. 予測値を計算
    prediction = base_mean + momentum + noise

    # 5. 現実的な範囲にクリップ
    prediction = np.clip(prediction,
                        base_mean - base_std * 3,
                        base_mean + base_std * 3)

    return prediction
```

### シンプルなモデルを選択した理由

- **効率的市場仮説**: 短期的な価格変動はほぼランダム
- **過学習の危険性**: 複雑なモデルは訓練データに過適合しやすい
- **ロバストネス重視**: シンプルなモデルの方が未知のデータに強い
- 評価指標の「シャープレシオ」（平均÷標準偏差）も予測の安定性を重視している

## 実装のポイント

### 1. 正しいターゲット計算

```python
for t in range(len(data) - lag - 1):  # ← 未来のデータが必要
    target[t] = log(price[t + lag + 1] / price[t + 1])
```

### 2. 平均回帰の実装

```python
recent_mean = np.mean(recent_values)
momentum = -recent_mean * 0.1  # 平均から離れたら戻る方向に予測
```

### 3. 適切なノイズモデリング

```python
# 対数リターンは正規分布に近い
noise = np.random.normal(0, historical_std * 0.8)
prediction = base_mean + momentum + noise
```

### 4. 現実的な範囲制限

```python
# 対数リターンは通常±10%以内
max_bound = historical_std * 3
prediction = np.clip(prediction, mean - max_bound, mean + max_bound)
```

## 結果

```
修正前: -0.058 (負の相関)
修正後: 正の値へ改善
```

## 学んだこと

### 1. 問題理解が最重要

複雑なモデルを作る前に、何を予測しようとしているのかを正確に理解する必要がある。

- ターゲットの定義（forward-looking）
- 評価指標の意味（シャープレシオ）
- データの時間構造（lagの意味）

### 2. ベースラインの重要性

いきなり複雑なモデル（LightGBM、LSTMなど）に飛びつくのではなく、まずシンプルなベースラインを作るべき。

```python
# 最もシンプルなベースライン
prediction = historical_mean + random_noise
```

正しいターゲット理解があれば、これでも意味のある予測になる。

### 3. スコアの意味を理解する

スコアが -0.058 だった時、「なぜ負の相関なのか？」と深く考えるべきだった。負の相関は以下を示唆する：

- ターゲットの定義ミス
- 特徴量の時間軸ミス
- データリークの逆パターン

### 4. 金融時系列の特殊性

- **非定常性**: 統計的性質が時間とともに変化
- **ボラティリティクラスタリング**: 変動が大きい時期と小さい時期が塊になる
- **予測不可能性**: 短期的にはほぼランダムウォーク

### 5. ドキュメントを読む

コンペのルール、データの説明、評価方法を最初に徹底的に読むことが重要。特に以下を早期に理解すべきだった：

- `target_pairs.csv` の構造
- lagパラメータの意味
- フォワードランニング評価の仕組み

## 金融コンペ用チェックリスト

- [ ] ターゲットの時間軸を確認（forward or backward?）
- [ ] lagパラメータの意味を理解
- [ ] 最もシンプルなベースラインから開始
- [ ] 評価指標の意味を深く理解
- [ ] データリークがないか確認
- [ ] 金融特有の性質（平均回帰、ボラティリティ）を考慮

## 参考資料

- [Kaggle: MITSUI&CO. Commodity Prediction Challenge](https://www.kaggle.com/competitions/mitsui-and-co-commodity-prediction)
- [修正版コード（Kaggle Notebook）](https://www.kaggle.com/)
