キーワードやクリエイターで検索


メニュー
 投稿

見出し画像
【機械学習 第5回】三井物産コンペ で勉強

1
shogaku
shogaku
2025年11月26日 08:03

Kaggle金融コンペで学んだ：Forward-Looking Targetの落とし穴と修正
Notebook

Forward-Looking Target Fix
Explore and run machine learning code with Kaggle Notebooks |
www.kaggle.com

目次
Kaggle金融コンペで学んだ：Forward-Looking Targetの落とし穴と修正
はじめに
コンペの概要
直面した問題：スコアが-0.058
バグの正体：時間軸の誤解
❌ 間違った実装（Backward-looking）
✅ 正しい実装（Forward-looking）
図解：時間軸の違い
なぜこのミスをしたのか
修正後のアプローチ

すべて表示
はじめに
Kaggleで開催されていた「MITSUI&CO. Commodity Prediction Challenge」に参加しました。これは三井物産が主催する、コモディティ（商品先物）の価格予測コンペです。

金融系のコンペは初めてで、様々な学びがありました。特に、初心者が陥りやすい「時間軸の誤解」という致命的なバグに遭遇し、それを修正するまでのプロセスが非常に勉強になったので記録として残します。

コンペの概要
課題：424種類のコモディティペア（例：銅と亜鉛の価格差）の対数リターンを予測

評価指標：スピアマン順位相関のシャープレシオ（予測の安定性を重視）

特殊性：提出締切後、実際の市場データで3.5ヶ月間評価される「フォワードランニング評価」

直面した問題：スコアが-0.058
最初の提出で、スコアが　-0.058　という負の値になりました。

スピアマン相関: -0.058
→ 予測が実際と逆相関している状態

copy
これは単に精度が低いというレベルではなく、「予測の方向性が間違っている」ことを意味します。モデルは動いているのに、何かが根本的におかしい。

バグの正体：時間軸の誤解
問題はターゲット（予測すべき値）の計算方法にありました。

❌ 間違った実装（Backward-looking）
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

copy
この実装では、「時刻tでの予測値」が「時刻tから過去を見た対数リターン」になっています。これはターゲットとして意味がありません。なぜなら、時刻tの時点で過去のデータは既に分かっているからです。

✅ 正しい実装（Forward-looking）
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

copy
正しくは、「時刻tでの予測値」は「時刻t+1からt+lag+1までの対数リターン」であるべきでした。つまり、未来の価格変動を予測するのが本来のタスクです。

図解：時間軸の違い
時系列データ: [price[0], price[1], price[2], price[3], price[4], ...]

❌ Backward-looking (間違い):
時刻t=3で予測 → log(price[3] / price[1])  # lag=2の場合
                 ↑ 過去を見ている

✅ Forward-looking (正解):
時刻t=1で予測 → log(price[4] / price[2])  # lag=2の場合
                 ↑ 未来を予測している

copy
なぜこのミスをしたのか
振り返ると、以下の要因がありました：

金融時系列の経験不足

一般的な時系列予測（天気予報など）と金融予測の違いを理解していなかった

「対数リターン」の定義を曖昧に理解していた

ドキュメントの読み込み不足

コンペのData Descriptionに書いてあったはずだが、流し読みしていた

サンプルコードを参照すべきだった

検証の甘さ

「コードが動いている = 正しい」と思い込んだ

スコアの意味（負の相関）を深く考えなかった

修正後のアプローチ
ターゲット計算を修正した上で、シンプルなモデルを実装しました。

基本設計
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

copy
なぜシンプルなモデルなのか
金融時系列予測では：

効率的市場仮説：短期的な価格変動はほぼランダム

過学習の危険性：複雑なモデルは訓練データに過適合しやすい

ロバストネス重視：シンプルなモデルの方が未知のデータに強い

コンペの評価指標も「シャープレシオ」（平均÷標準偏差）で、予測の安定性を重視しています。これは実際のトレーディングでも重要な考え方です。

学んだこと
1. 問題理解が最重要
複雑なモデルを作る前に、何を予測しようとしているのかを正確に理解する必要があります。今回は：

ターゲットの定義（forward-looking）

評価指標の意味（シャープレシオ）

データの時間構造（lag の意味）

これらを最初に徹底的に理解すべきでした。

2. ベースラインの重要性
いきなり複雑なモデル（LightGBM、LSTM など）に飛びつくのではなく、まずシンプルなベースラインを作るべきでした。今回のシンプルなアプローチは：

# 最もシンプルなベースライン
prediction = historical_mean + random_noise

copy
これでも、正しいターゲット理解があれば意味のある予測になります。

3. スコアの意味を理解する
スコアが　-0.058　だった時、「精度が低い」と思うだけでなく、「なぜ負の相関なのか？」と深く考えるべきでした。負の相関は：

ターゲットの定義ミス

特徴量の時間軸ミス

データリークの逆パターン

などを示唆します。

4. 金融時系列の特殊性
金融データは：

非定常性：統計的性質が時間とともに変化

ボラティリティクラスタリング：変動が大きい時期と小さい時期が塊になる

予測不可能性：短期的にはほぼランダムウォーク

これらの特性を理解した上でモデリングする必要があります。

5. ドキュメントを読む
当たり前ですが、コンペのルール、データの説明、評価方法を最初に徹底的に読むことが重要です。今回は：

target_pairs.csvの構造

lagパラメータの意味

フォワードランニング評価の仕組み

これらをもっと早く理解していれば、無駄な時間を省けました。

実装のポイント
修正版の実装で意識したポイント：

1. 正しいターゲット計算
# 必ずこのパターンで実装する
for t in range(len(data) - lag - 1):  # ← 未来のデータが必要
    target[t] = log(price[t + lag + 1] / price[t + 1])

copy
2. 平均回帰の実装
# 金融では「平均回帰」が重要な性質
recent_mean = np.mean(recent_values)
momentum = -recent_mean * 0.1  # 平均から離れたら戻る方向に予測

copy
3. 適切なノイズモデリング
# 対数リターンは正規分布に近い
noise = np.random.normal(0, historical_std * 0.8)
prediction = base_mean + momentum + noise

copy
4. 現実的な範囲制限
# 対数リターンは通常±10%以内
max_bound = historical_std * 3
prediction = np.clip(prediction, 
                    mean - max_bound, 
                    mean + max_bound)

copy
結果
バグ修正後、スコアは大幅に改善しました：

修正前: -0.058 (負の相関)
修正後: 正の値へ改善

copy
現在は評価期間中（2026年1月16日まで）で、実際の市場データで継続的に評価されています。

今後に向けて
次に金融系コンペに参加する時のチェックリスト：

[ ] ターゲットの時間軸を確認（forward or backward?）

[ ] lagパラメータの意味を理解

[ ] 最もシンプルなベースラインから開始

[ ] 評価指標の意味を深く理解

[ ] データリークがないか確認

[ ] 金融特有の性質（平均回帰、ボラティリティ）を考慮

まとめ
初めての金融コンペで、基本的だが重要なバグを経験しました。

最大の学び：

複雑なモデルより、正しい問題理解が重要

金融では「予測可能性の限界」を認識することが大切

シンプルなアプローチでも、原理的に正しければ十分戦える

この経験は、今後のコンペや実務でも活かせると思います。同じようなバグで悩んでいる方の参考になれば幸いです。

参考資料
Kaggle: MITSUI&CO. Commodity Prediction Challenge

修正版のコードはKaggle Notebookで公開しています

この記事は学習記録として書きました。間違いや改善点があればご指摘ください。

#kaggle

1




shogaku
shogaku
GAS、VBA、Pythonを使って業務改善とかしています。備忘録や勉強メモを記載しています。

1


noteプレミアム
note pro
よくある質問・noteの使い方
プライバシー
クリエイターへのお問い合わせ
フィードバック
ご利用規約
通常ポイント利用特約
加盟店規約
資⾦決済法に基づく表⽰
特商法表記
投資情報の免責事項
【機械学習 第5回】三井物産コンペ で勉強｜shogaku