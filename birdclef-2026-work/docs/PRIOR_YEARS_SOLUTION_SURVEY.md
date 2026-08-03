# BirdCLEF 2024 / 2025 上位解法サーベイと 2026 現行 pipeline とのギャップ

日付: 2026-04-19
目的: アカデミックに勝ち筋を整理。小手先でない改善候補の列挙。
出典: 各チームの GitHub README（一次情報）

## 自現状
- 2026-04-18 自LB 0.926 (yuriygreben/birdclef-2026-improved-ensemble fork)
- 公開最高 claimed LB 0.929（同著者）
- **未超過**。feedback_kaggle.md により独自手法は禁止状態（超えたら解禁）。
- 本ドキュメントは「超えたあとに何をやるか」の勝ち筋整理。

## 現在の 0.929 pipeline 構成（yuriygreben）
- **Backbone**: Google Perch v2（frozen, 1536-dim embeddings）
- **Sequence**: ProtoSSM v5（4-layer, d_model=320, bidirectional selective SSM + cross-attn 8heads）
- **Head**: prototypical cosine + gated Perch distillation + MLP probe
- **2nd pass**: ResidualSSM（d_model=128, 2-layer, 重み 0.35）
- **Post-processing**: TTA (5 offsets) + per-taxon temperature + file-level scale + rank-aware scaling (power=0.4) + delta shift smoothing (α=0.20) + per-class thresholds (OOF-optimized)
- **Metadata fusion**: Bayesian site/hour prior tables
- **Loss**: focal BCE (γ=2.5) + label smoothing 0.03
- **Training**: 80 epoch, cosine warm restart, SWA from epoch 52, 5-fold GroupKFold
- **Augment**: mixup + cutmix（テンポラル埋め込み系列に）

## BirdCLEF 2024 3rd place (jfpuget/TheoViel/Henkel, Private 0.69)
URL: https://github.com/jfpuget/birdclef-2024 / https://github.com/TheoViel/kaggle_birdclef2024

### 勝ち筋
1. **unlabeled soundscapes への pseudo-label distillation**（最大の貢献）
   - 1st level models を train で学習 → unlabeled soundscapes を 5 秒クリップ切り出して予測
   - 高信頼予測を train に追加 → 2nd level models を再学習
2. **外部データ**: Xeno Canto + 過去 BirdCLEF (2021/2022/2023) + Birdsong Recognition 2020。種ごと 500 件キャップ、最新のものを残す
3. **多様な backbone ensemble**: efficientnet, mobilenet, tinynet, mnasnet, mixnet, EfficientVit b0/b1/m3
4. **2nd level はスピード重視**: EfficientVit-B0 + mnasnet-100 で diversity、ONNX 化で 5fold 40 分推論
5. **augmentation**: time shift 1 秒 + mixup（ただし **max-of-labels**、平均ではない）
6. **入力**: 224x224 log mel spectrogram

## BirdCLEF 2025 2nd place (VSydorskyy/Fernando, Public 0.925 / Private 0.928)
URL: https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place
Paper: https://ceur-ws.org/Vol-4038/paper_256.pdf

### 勝ち筋
1. **Domain shift 対策**: transfer learning + semi-supervised distillation
2. **大規模事前学習**: Xeno-Canto + iNaturalist + CSA の鳥類音声で backbone を事前学習してから 2025 競技データで fine-tune
3. **Pseudo label iter** (iter=2〜3): `PseudoF2PT05MT01P04I3` 等の notation から、threshold ベースの iterative pseudo-labeling
4. **2 backbone ensemble**: `tf_efficientnetv2_s_in21k` + `eca_nfnet_l0`
5. **Class balancing**: SqrtBalancing（eca）/ EqualBalancing（ebs）/ MinorOverSampleV1（少数クラス over-sample）
6. **Rare species 対応**: `AddRareBirdsNoLeak`（label leak せず rare 種を追加）
7. **Loss**: FocalBCELoss + label smoothing 0.05
8. **推論**: OpenVINO fp16（ONNX より CPU 速い）
9. **入力**: 5 秒 clip

## BirdCLEF 2025 5th place (myso1987)
URL: https://github.com/myso1987/BirdCLEF-2025-5th-place-solution

### 勝ち筋
1. **SED (Sound Event Detection) framework**: フレーム単位予測 + attention pooling → clip レベル
2. **3-stage training**:
   - Stage 1: 生 label で学習
   - Stage 2: 1st model で pseudo-label → 再学習
   - Stage 3: 2nd model で再 pseudo-label → 再学習
3. **多様な backbone**: EfficientNet B0 / B3 / EfficientNetV2 B3 / EfficientNetV2 S
4. **入力**: 30 秒 or 60 秒 crop（5 秒ではない）
5. **Over-sampling**: 20 件未満の種は oversampling
6. **推論**: OpenVINO（PyTorch → OpenVINO 変換）

## ギャップ分析：現 pipeline に欠けている「実績ある手法」

| # | 手法 | 採用状況 (2024 3rd / 2025 2nd / 2025 5th) | 現 0.929 pipeline | 推定効果 |
|---|---|:-:|:-:|---|
| 1 | **unlabeled soundscapes への pseudo-label distillation** | ◯/◯/◯ | ✗ | +0.01〜0.03（最大の勝ち筋） |
| 2 | **外部データ** (Xeno-Canto, 過去 BirdCLEF) | ◯/◯/-- | ✗ (Perch 事前学習のみ) | +0.005〜0.02、rare class 救済 |
| 3 | **CNN/SED on log mel spec** (画像モデル) | ◯/◯/◯ | ✗ (Perch embedding のみ) | diversity +0.005〜0.015 |
| 4 | **多 backbone ensemble** | ◯/◯/◯ | ✗ (単一 Perch+ProtoSSM) | +0.005〜0.015 |
| 5 | **Mixup max-of-labels** | ◯/--/-- | △ (mixup + cutmix あるが max 運用か要確認) | +0.002〜0.008 |
| 6 | **Class balancing (SqrtBalancing / minor oversample)** | --/◯/◯ | ✗ | macro AUC で +0.005〜0.01 |
| 7 | **事前学習 backbone**（同系種の XC / iNaturalist） | --/◯/-- | △ (Perch 自体が事前学習済み) | +0.005〜0.015（domain gap 狭める） |
| 8 | **OpenVINO fp16 推論** | --/◯/◯ | ✗ (ONNX 想定) | 推論速度のみ、LB には直接効かない |

## 優先順位（公開最高超えた後の実装順）

1. **Pseudo-label distillation on unlabeled_soundscapes**（全上位採用、単独で +0.01〜0.03）
   - 実装: 現 0.926 モデルで unlabeled_soundscapes を 5 秒クリップ化 → 予測 → 高信頼を学習データに追加 → ProtoSSM 再学習
   - 1 iter で効果出る、2-3 iter で収束
2. **CNN/SED 別 path を ensemble 追加**（diversity が決定的）
   - 例: EfficientVit-B0 on log mel spec を別訓練 → Perch+ProtoSSM と logit 平均
3. **外部データ追加**（競技規則確認: BirdCLEF+ 2026 は external data OK）
   - Xeno-Canto から 234 種の追加サンプル（特に rare class）
4. **Class balancing の見直し**（macro AUC のため）

## 直近の次 action（未超過状態では）

feedback_kaggle.md 厳格運用で上記はまだ提案できない。次の合法 action:

**A**. 公開 Notebook 群で claimed LB > 0.929 のものを再サーベイ
   - 596 vote の `dingjiarun/pantanal-distill-birdclef2026-onnx` V17 は "(0.924)" 表記だが実スコア確認必要
   - V18 系 (yaroslavkholmirzayev 等) も確認
**B**. 現 fork と別系統 Notebook の単純 ensemble で 0.929 超え狙い
   - ただし ensemble 相手は「claimed LB > 0.9」のみ、低スコア混ぜ禁止

## 参考 URL
- BirdCLEF 2024 3rd place (jfpuget): https://github.com/jfpuget/birdclef-2024
- BirdCLEF 2024 3rd place (TheoViel): https://github.com/TheoViel/kaggle_birdclef2024
- BirdCLEF 2024 discussion: https://www.kaggle.com/competitions/birdclef-2024/discussion/511905
- BirdCLEF 2025 2nd place: https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place
- BirdCLEF 2025 2nd place Kaggle: https://www.kaggle.com/competitions/birdclef-2025/discussion/583699
- BirdCLEF 2025 5th place: https://github.com/myso1987/BirdCLEF-2025-5th-place-solution
- BirdCLEF 2025 5th place Kaggle: https://www.kaggle.com/competitions/birdclef-2025/discussion/583312
- VSydorskyy paper (domain shift): https://ceur-ws.org/Vol-4038/paper_256.pdf
