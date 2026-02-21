# Stanford RNA 3D Folding 2 — セッションノート

## コンペ概要

- **URL**: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2
- **タスク**: RNA配列から3D構造を予測（C1'原子のx,y,z座標）
- **評価指標**: TM-score（0〜1、高いほど良い。0.45以上で正しい折り畳みとみなす）
- **提出形式**: 1配列につき5構造予測、各残基の座標を出力
- **締切**: 2026-03-25
- **賞金**: $100,000
- **インターネット**: 無効（enable_internet: false 必須）
- **参加者数**: 1045チーム

## データ構造

```
/kaggle/input/competitions/stanford-rna-3d-folding-2/
├── test_sequences.csv       # 28配列（target_id, sequence, ...）
├── sample_submission.csv    # 9762行（ID形式: 8ZNQ_1, 8ZNQ_2, ...）
├── train_sequences.csv
├── train_labels.csv
├── validation_sequences.csv
├── validation_labels.csv
└── MSA/                     # Multiple Sequence Alignment FASTAファイル群
```

### test_sequences.csv カラム
```
target_id, sequence, temporal_cutoff, description, stoichiometry,
all_sequences, ligand_ids, ligand_SMILES
```

⚠️ **重要**: `sequence` カラムを使う。`all_sequences` はFASTA形式（`>`ヘッダー付き）なので使わない。

### sample_submission.csv フォーマット
```
ID,          resname, resid, x_1,y_1,z_1, x_2,y_2,z_2, ..., x_5,y_5,z_5
8ZNQ_1,      A,       1,     0,  0,  0,   0,  0,  0,   ...,  0,  0,  0
8ZNQ_2,      C,       2,     ...
```

- **ID**: `{target_id}_{resid}` 形式（`target_id` だけではNG）
- **行数**: 9762行（28配列の合計残基数）

## Notebook情報

- **Kaggle**: https://www.kaggle.com/code/yasunorim/stanford-rna-3d-folding-2-baseline
- **GitHub**: https://github.com/yasumorishima/kaggle-competitions/tree/main/stanford-rna-3d-folding-2
- **kernel-metadata.json id**: `yasunorim/stanford-rna-3d-folding-2-baseline`

## バージョン履歴

| Version | 内容 | 結果 |
|---|---|---|
| v1 | 初回push | - |
| v2 | タイトル変更（kaggle-wandb-sync明記） | - |
| v3 | GitHub/PyPIリンク追加 | Submission Scoring Error（`all_sequences`誤検出） |
| v4 | `sequence`列を優先選択に修正 | Submission Scoring Error（IDが`8ZNQ`のみ、行数9000） |
| v5 | `sample_submission.csv`テンプレート方式に変更 | TM-score 0.103（baseline-v1） |
| v6 | 5構造にσ=0.5Åのノイズ追加（improved-v1） | TM-score 0.104 |
| v7 | **Template Matching v1**: 訓練データの3D座標をテンプレートとして使用 | 確認待ち |

## トラブルシュート記録

### 1. `all_sequences` 列の誤検出
- **症状**: `resname` が `>`, `8`, `Z` などFASTA文字になる
- **原因**: 列検出ロジックが `all_sequences`（FASTA形式）を選んでいた
- **修正**: `sequence` を完全一致で最優先、`all_sequences` は除外

### 2. IDフォーマット・行数不一致
- **症状**: Submission Scoring Error
- **原因1**: IDが `8ZNQ`（target_idのみ）→ 正しくは `8ZNQ_1`（`{target_id}_{resid}`形式）
- **原因2**: 行数が 9000 ≠ 9762（`sequence`列の長さ ≠ 実際の残基数）
- **修正**: `sample_submission.csv` をテンプレートとして読み込み、座標だけ上書き

```python
# NG: test_df から自前で行を作る → IDフォーマット・行数が合わない
# OK: sample_submission をテンプレートに使う
submission = sample_sub.copy()
submission['_target'] = submission['ID'].str.rsplit('_', n=1).str[0]
for target_id, group in submission.groupby('_target', sort=False):
    coords = helix_coords(len(group))
    ...
submission = submission.drop(columns=['_target'])
```

### 3. kernel titleとidのスラッグ不一致（警告のみ）
- タイトルに `|` や長い説明を入れると警告が出るが、既存カーネルへの更新は通る
- 新規カーネル作成時はタイトルがidに完全一致するシンプルなものにしてからpush、その後タイトルを変更する

## W&B設定

- **プロジェクト名**: `stanford-rna-3d-folding-2`
- **entity**: `fw_yasu11-personal`
- **run一覧**: https://wandb.ai/fw_yasu11-personal/stanford-rna-3d-folding-2

### Notebookでの設定（importより前に必須）
```python
import os
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_PROJECT'] = 'stanford-rna-3d-folding-2'
import wandb  # ← この順序が重要
```

### sync手順
```bash
# 実行完了後（--skip-pushでpushをスキップ）
PYTHONUTF8=1 kaggle-wandb-sync run stanford-rna-3d-folding-2/ --skip-push
```

## ベースライン実装メモ

### A-form RNA ヘリックス幾何学
```python
def helix_coords(seq_len, rise=2.81, radius=9.0, twist_deg=32.7):
    """A-form RNA ヘリックスのC1'座標を生成"""
    twist_rad = np.radians(twist_deg)
    indices = np.arange(seq_len)
    x = radius * np.cos(indices * twist_rad)
    y = radius * np.sin(indices * twist_rad)
    z = indices * rise
    return np.stack([x, y, z], axis=1)
```

パラメータ根拠:
- rise: 2.81 Å/残基（A-form RNA標準値）
- radius: 9.0 Å（糖-リン酸骨格の回転半径）
- twist: 32.7°/残基（A-form RNA標準値）

## 次のステップ

### v5確認後
1. Submit → TM-scoreスコア確認
2. `kaggle-wandb-sync run --skip-push` でW&B sync

### improved-v1（比較実験）
- **内容**: 5構造に微小ノイズ追加（構造多様性UP）
- **期待効果**: TM-scoreのわずかな改善（5構造が同一だとbest-of-5の意味がない）
- **実装**:
  ```python
  for s in range(1, 6):
      noise = np.random.normal(0, 0.5, (seq_len, 3))  # σ=0.5Å
      submission.loc[idx, f'x_{s}'] = (coords[:, 0] + noise[:, 0]).round(3)
      submission.loc[idx, f'y_{s}'] = (coords[:, 1] + noise[:, 1]).round(3)
      submission.loc[idx, f'z_{s}'] = (coords[:, 2] + noise[:, 2]).round(3)
  ```
- **W&B run_name**: `improved-v1`
- **比較**: W&Bで baseline-v1 vs improved-v1 を並べて比較

## Template Matching v1 実装メモ（v7）

### アプローチ
- 訓練データの3D座標（train_labels.csv）をテンプレートとして使用
- テスト配列ごとに、配列長が最も近い訓練構造を5件選出
- 長さが異なる場合はscipy.interpolate.interp1dで線形補間してリサイズ
- 5件のテンプレートがそのまま5構造予測になる

### ヘリックスとの違い
- ヘリックス: 全残基を1本のらせんに配置（RNAの二次構造を無視）
- テンプレート: 実際のRNA構造のステム・ループ・ヘアピン等のパターンを保持

### NaN処理
- train_labelsの一部構造はNaN座標を含む
- NaN率50%超 → スキップ（構造として使えない）
- NaN率50%以下 → np.interpで欠損値を線形補間して補完

### W&B config
```python
config = {
    'approach': 'template_matching',
    'n_structures': 5,
    'n_templates': 5,
    'interpolation': 'linear',
    'n_train_structures': <動的>,
}
```

### 改善の方向性（中長期）
1. **Nussinov二次構造予測**（v3計画済み）: Watson-Crick塩基対に基づく二次構造予測 → ステム/ループの幾何学的配置
2. 配列類似度ベースのテンプレート選択（長さだけでなくBLAST的なアラインメント）
3. 事前学習済みモデル（RhoFold+, trRosettaRNA等）をKaggle Datasetとして追加
