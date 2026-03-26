# CLAUDE_COMP.md — BirdCLEF+ 2026

## コンペ概要
- **テーマ**: パンタナール（ブラジル）の音響種識別（鳥・昆虫・爬虫類・両生類）
- **評価指標**: macro ROC AUC（全234種）
- **データ**: 35,549件 .ogg音声 + 59本の60秒サウンドスケープ（ラベル付き）
- **提出**: row_id × 234種の確率値、5秒チャンク単位
- **締切**: 2026-06-03
- **制約**: Code Competition（Kaggle Notebook、internet off、9h GPU / 12h CPU）

## 234種の内訳
- Aves（鳥）: 大多数
- Amphibia（両生類）: カエル類
- Insecta（昆虫）: セミ・コオロギ等
- Reptilia（爬虫類）: カイマン等

## 評価関数の正しい実装
```python
from sklearn.metrics import roc_auc_score
# 各クラスで正例がないクラスはスキップ
auc = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')
# ただしクラスが全て0のものは除外して計算
```

## 戦略（差別化の柱）
1. **BEATs + SED Attention Pooling** — クリップ分類ではなくイベント検出。鳥の短鳴き/虫の持続音に自動適応
2. **Multi-resolution Spectrogram** — 高時間分解能（鳥向け）+ 高周波数分解能（虫/蛙向け）の2ストリーム
3. **Perch embedding融合** — frozen Perch embeddingをclassifier headに連結（linear probeの上位互換）
4. **時刻・サイト条件付き** — hour/site embeddingをモデルに入力
5. **最終段**: Perch linear probe + BEATs SED + CNN のweighted blend

## 禁止事項
- Perch linear probeだけで満足しない（全員がやっている）
- train_audioのrating=0のデータを無条件で使わない（ノイズが多い）
- 5秒チャンクの境界をまたぐ鳴き声を無視しない（overlap推論を検討）
- CV↑LB↓になった施策を繰り返さない（下記リストに追記）
- ローカルPCで音声データをDL・処理しない（全てクラウド）

## CV↑LB↓ リスト（失敗施策）
（まだなし — 実験が進んだら追記）

## よくあるバグ
- taxonomy.csvの種順序とsubmissionのカラム順序の不一致
- 60秒未満の音声ファイルのパディング忘れ
- Perchのサンプルレート(32kHz)とCNNのサンプルレートの不一致
- BCEWithLogitsLossでsigmoidを二重適用

## データソース注意
- train_audio: iNaturalist + Xeno-Canto混在。collectionカラムで区別
- rating: XCデータのみ。iNatは0.0。ratingでフィルタする場合はiNatを除外しない
- secondary_labels: 多くは空。マルチラベル学習時に注意
