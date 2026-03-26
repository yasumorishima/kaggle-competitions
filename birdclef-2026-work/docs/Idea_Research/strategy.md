# BirdCLEF+ 2026 戦略メモ

## 現状の上位アプローチ（2026-03-26調査）

### Perch v2 系（CPU、0.91付近）
- Google Perch v2 frozen → embedding抽出 → PCA → LogisticRegression
- ベイズ事前分布（サイト×時刻）融合
- テンポラルスムージング（虫/蛙クラス）
- **全員が同じ**。差がつかない。

### PyTorch Baseline（GPU）
- EfficientNetV2-B0 + mel spectrogram(256x256)
- Mixup + StratifiedKFold
- 単体ではPerch系に劣る

## 我々の差別化

### Phase 1: BEATs-SED（EXP001）
- **BEATs**: AudioSet 2M事前学習。Perchは鳥専用だが、BEATsは汎用音声。昆虫/両生類への汎化が期待
- **Attention Pooling SED**: クリップ全体を分類するのではなく、鳴いているフレームにattention。短い鳥の鳴き声でも見逃さない
- **Focal Loss**: 234種で著しいクラス不均衡。focal lossで少数種の学習効率を上げる
- **Differential LR**: backbone凍結 + 上位層だけfine-tune（低リソースでも効く）

### Phase 2: Multi-resolution + Perch融合（EXP002予定）
- Perch embedding(1536d) + BEATs embedding(768d) を連結
- 2つの解像度のmel spectrogramから特徴抽出
- メタデータ（時刻、サイト）をembeddingとして注入

### Phase 3: アンサンブル + 後処理
- Perch linear probe + BEATs-SED + CNN のweighted blend
- テンポラルスムージング（0.910 notebookの手法を取り込み改良）
- クラス別閾値最適化

## BEATs準備
- チェックポイント: `BEATs_iter3_plus_AS2M.pt`（~370MB）
- ソース: https://github.com/microsoft/unilm/tree/master/beats
- Kaggle Datasetとしてアップロード必要（ローカルDL禁止→GH Actions経由）
- コード: `hubfor/microsoft-beats-model` にBEATs.py等あり
