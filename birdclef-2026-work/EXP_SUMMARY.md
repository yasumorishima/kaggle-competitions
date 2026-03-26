# EXP_SUMMARY.md — BirdCLEF+ 2026

## 実験履歴

| EXP | child | 変更内容 | CV(AUC) | LB | 結果 | 所感 |
|-----|-------|---------|---------|-----|------|------|
| 001 | 000 | BEATs-SED baseline (5s chunk, attention pooling, focal loss) | — | — | — | 初回実験 |

## 公開Notebookベンチマーク（参考）
| Notebook | Score | アプローチ |
|---|---|---|
| 0.910 Score | 0.910 | Perch v2 + Bayesian prior + LogReg probe + temporal smoothing |
| Perch v2 starter | ~0.908 | Perch v2 + PCA + LogReg probe |
| PyTorch Baseline | — | EfficientNetV2-B0 + mel spectrogram + mixup |
| Blend V6 | — | 複数モデルblend |

## アーキテクチャ比較
| モデル | 強み | 弱み |
|---|---|---|
| Perch v2 (frozen) | 鳥14,795種の事前知識、CPU動作 | 昆虫/両生類に弱い、fine-tune不可 |
| BEATs (fine-tune) | AudioSet汎用、SED対応 | GPU必須、学習コスト高 |
| EfficientNet + mel | シンプル、高速 | 音声特有の知識なし |
| **BEATs + Perch融合** | **両方の強みを統合** | **実装複雑** |
