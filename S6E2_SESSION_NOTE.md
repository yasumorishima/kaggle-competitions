# S6E2 Heart Disease - セッションメモ

## コンペ情報
- **URL**: https://www.kaggle.com/competitions/playground-series-s6e2
- **タスク**: 心臓病予測（二値分類: Presence / Absence）
- **評価指標**: AUC-ROC
- **締切**: 2026-02-28
- **データ**: Train 630,000行 / Test 270,000行

## データ構造（実際のカラム名）
```
id                      int64
Age                     int64
Sex                     int64
Chest pain type         int64
BP                      int64
Cholesterol             int64
FBS over 120            int64
EKG results             int64
Max HR                  int64
Exercise angina         int64
ST depression           float64
Slope of ST             int64
Number of vessels fluro int64
Thallium                int64
Heart Disease           object (Presence/Absence)
```
- 全特徴量がint/float（カテゴリカル型なし）
- ターゲットは `Heart Disease`（スペースあり、object型）
- 欠損値なし

## ノートブック
- **Kaggle URL**: https://www.kaggle.com/code/yasunorim/s6e2-heart-disease-eda-ensemble-wandb
- **ローカル**: `s6e2-heart-disease-baseline.ipynb`
- **現在**: v10（GPU版、W&Bデバッグログ付き、wandb upgrade追加）
- **構成**: EDA + LightGBM + XGBoost + CatBoost (GPU) + 5-fold CV + W&B + アンサンブル

## スコア
| Version | Status | CV AUC | LB Score |
|---|---|---|---|
| v1 | Error | - | - |
| v2 | Error | - | - |
| v3 | Error（XGBoost gpu_hist非対応） | LGB 0.95515 | - |
| **v4** | **Complete** | **0.95528** | **0.95337（616位）** |
| v5〜v10 | W&Bデバッグ用（スコア同等） | 0.95528 | - |

### v4 モデル別CV AUC
| Model | CV AUC |
|---|---|
| Ensemble (avg) | 0.95528 |
| CatBoost | 0.95524 |
| LightGBM | 0.95515 |
| XGBoost | 0.95513 |

## GPU設定
- `kernel-metadata.json`: `"enable_gpu": "true"`
- LightGBM: `'device': 'gpu'`
- XGBoost: `'tree_method': 'hist'`, `'device': 'cuda'`（※XGBoost 2.0+では`gpu_hist`廃止）
- CatBoost: `'task_type': 'GPU'`

## Feature Engineering（現在）
- 6つの交互作用特徴量: Age×MaxHR, Age×STdep, STdep×Slope, BP×Chol, MaxHR/Age, Vessels×Thal

## W&B
- プロジェクト: `kaggle-s6e2-heart-disease`
- ダッシュボード: https://wandb.ai/fw_yasu11-personal/kaggle-s6e2-heart-disease
- エンティティ: `fw_yasu11-personal`（W&Bが自動生成）
- Kaggle Secretsに `WANDB_API_KEY` 設定必要（Add-ons > Secrets）
- **`!pip install -q --upgrade wandb`** が必要（Kaggle標準のwandbは古く、新形式キー `wandb_v1_...` 非対応）
- v1〜v8: 接続エラー（Secrets未設定/キー形式不一致）
- v10: Web UIでSecret ON + wandb upgrade済み → **W&B接続を次回確認**

### W&B トラブルシューティング経緯
1. v1〜v4: Secrets未設定 → `Connection error trying to communicate with service.`
2. v8: Secret設定済みだがキー36文字 → `API key must be 40 characters long, yours was 36`（Key IDをコピーしていた）
3. v10: wandb upgrade + 新キー作成（`wandb_v1_...`形式）→ Web UIで実行済み、**結果未確認**

### Kaggle Secrets の挙動（要検証）
- `kaggle kernels push` するとSecretsの紐付けがリセットされる（推測、次回検証）
- CLIからSecrets設定する方法はない（Kaggle公式: セキュリティ上の理由で非対応）
- **運用**: CLI push → Web UIでSecret ON → Run。以降Web UIからRunすればSecret保持される（推測）

## トラブルシューティング
1. **v1エラー**: カラム名不一致（`HeartDisease`→実際は`Heart Disease`）
2. **v2エラー**: v1と同じ（EDAセルに`HeartDisease`残り）
3. **v3エラー**: XGBoost `gpu_hist`非対応（XGBoost 2.0+では`hist`+`device:'cuda'`に変更）
4. **v4成功**: 全修正適用、3モデル完走
5. **v8**: W&B Key ID（識別子）とAPIキー値を混同 → 新キー作成で解決
6. **v10**: wandb upgrade追加（新形式キー対応）

## ブログ記事
- **Zenn/Qiita記事11本目**: 「Kaggle S6E2参加記：GitHub連携 + W&B + GPU 3モデルアンサンブルのワークフロー」公開済み（2/7）

## W&B運用方針
- **普段はW&Bなしで実行**（フォールバック設計で問題なく動く）
- **W&Bが必要なとき（Optunaチューニング等）はユーザーが指示する**
- W&B使用時の手順: CLI push → Web UIでSecret ON → Run
- `kaggle kernels push` するとSecretsがリセットされる（確認済み）
- W&B接続確認済み（v10、4 run記録成功）

## 次のステップ
- ~~W&B接続確認~~: 完了（v10で4 run記録成功）
- ~~Secrets保持検証~~: 完了（CLIpushでリセットされる。毎回Web UIで再設定が必要）
- Kaggle Notebooks Expert取得済み → プロフィール書き換え相談
- スコア改善: Optuna tuning, multi-seed, rank ensemble, 元データ追加
- Notebookメダル狙い: upvoteされるよう内容充実
