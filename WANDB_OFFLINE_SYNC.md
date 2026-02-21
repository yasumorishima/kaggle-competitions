# W&B Offline Sync パイプライン - 実装ノウハウ

## 概要

`WANDB_MODE=offline` でKaggle上にW&Bログを保存し、GitHub Actions経由で
`kaggle kernels output` → `wandb sync` してW&Bクラウドへアップロードするパイプライン。

**検証完了**: 2026-02-21
**W&B run**: https://wandb.ai/fw_yasu11-personal/march-mania-2026-test/runs/dj59aupa
**テスト用ファイル**: `march-machine-learning-mania-2026-wandb-test/`
**ワークフロー**: `.github/workflows/kaggle-wandb-sync.yml`

---

## ハマりポイント（全て解決済み）

### 1. `kaggle kernels status` の出力形式

**誤解**: key-value形式（`status: queued`）だと思っていた
**実際**: 文章形式

```
yasunorim/march-mania-2026-wandb-test has status "KernelWorkerStatus.QUEUED"
yasunorim/march-mania-2026-wandb-test has status "KernelWorkerStatus.RUNNING"
yasunorim/march-mania-2026-wandb-test has status "KernelWorkerStatus.COMPLETE"
```

**正しいパース方法**:
```bash
STATUS=$(kaggle kernels status "$KERNEL_ID" 2>&1 | sed 's/.*has status "\([^"]*\)".*/\1/')
# → "KernelWorkerStatus.COMPLETE" などが取得できる
```

**NG例**（最初に書いたコード）:
```bash
# NG: key-value想定で grep → 何も取れない
STATUS=$(kaggle kernels status "$KERNEL_ID" | grep "^status:" | awk '{print $2}')
# NG: tail -1 → URL行を拾う（さらに古いコード）
STATUS=$(kaggle kernels status "$KERNEL_ID" | grep -v "^ref|^---" | tail -1 | awk '{print $(NF-1)}')
```

### 2. `|| true` が必須

`kaggle kernels status` はカーネルが存在しない場合などに非ゼロを返す。
GitHub Actions の `bash -e` モードでは即時スクリプト終了してしまう。

```bash
RAW=$(kaggle kernels status "$KERNEL_ID" 2>&1) || true
# ↑ || true がないと、エラー時に echo すら実行されずに終了する
```

### 3. 409 Conflict on push

前のrunで起動したKaggleカーネルがまだ実行中の場合、再pushすると409になる。
→ push前にステータスをチェックして、COMPLETE/ERROR/CANCELになるまで待つ。

```bash
for i in $(seq 1 20); do
  RAW=$(kaggle kernels status "$KERNEL_ID" 2>&1) || true
  STATUS=$(echo "$RAW" | sed 's/.*has status "\([^"]*\)".*/\1/')
  if [ -z "$STATUS" ] || echo "$STATUS" | grep -qiE "COMPLETE|ERROR|CANCEL"; then
    break
  fi
  sleep 30
done
kaggle kernels push -p "$NOTEBOOK_DIR"
```

### 4. タイトルのslug不一致

kernel-metadata.json の `title` が `id` のslugに一致しないと警告が出る。
`&` はslugで除去されるので注意:

```json
// NG: "W&B" → slug上は "wb" になり "wandb" と不一致
"title": "March Mania 2026 W&B Offline Test",
"id": "yasunorim/march-mania-2026-wandb-test"

// OK: "WandB" → slug "wandb" と一致
"title": "March Mania 2026 WandB Test",
"id": "yasunorim/march-mania-2026-wandb-test"
```

### 5. `kaggle kernels output` でwandbディレクトリが取得できるか

**結果: 取得できる。** ディレクトリ構造:

```
./kaggle_output/
├── march-mania-2026-wandb-test.log          # Kaggle実行ログ
└── wandb/
    └── offline-run-20260221_104123-dj59aupa/
        ├── files/
        │   └── requirements.txt
        ├── logs/
        │   ├── debug-internal.log
        │   └── debug.log
        └── run-dj59aupa.wandb               # ← これがsync対象
```

### 6. `wandb sync` の対象ディレクトリ

`offline-run-*` ディレクトリを指定して sync する。`--sync-all` は不要（むしろ不安定）。

```bash
WANDB_DIRS=$(find ./kaggle_output -type d -name "offline-run-*" 2>/dev/null)
for dir in $WANDB_DIRS; do
  wandb sync "$dir"
done
```

---

## Notebookの書き方

### WANDB_MODE=offline はimport前に設定する

```python
# NG: import後に設定しても効かないことがある
import wandb
os.environ['WANDB_MODE'] = 'offline'  # ← NG

# OK: import前に設定
import os
os.environ['WANDB_MODE'] = 'offline'  # ← OK
import wandb
```

### データパス自動検出（competition_sources）

competition_sourcesのマウント先は環境によって異なる:

```python
SLUG = 'march-machine-learning-mania-2026'
CANDIDATES = [
    Path(f'/kaggle/input/competitions/{SLUG}'),  # 新形式
    Path(f'/kaggle/input/{SLUG}'),               # 旧形式
]
DATA_DIR = next((p for p in CANDIDATES if (p / 'MTeams.csv').exists()), None)
if DATA_DIR is None:
    raise FileNotFoundError(f'Not found in: {CANDIDATES}')
```

---

## ワークフロー全体像

```yaml
# .github/workflows/kaggle-wandb-sync.yml の構成

Inputs:
  - notebook_dir: e.g. "march-machine-learning-mania-2026-wandb-test"
  - kernel_id:    e.g. "yasunorim/march-mania-2026-wandb-test"

Secrets:
  - KAGGLE_API_TOKEN  # 既存
  - WANDB_API_KEY     # 新規: fw_yasu11アカウントのAPIキー

Steps:
1. checkout
2. pip install kaggle wandb
3. Wait if kernel running, then push   ← 409対策
4. Poll until KernelWorkerStatus.COMPLETE  ← sedでパース
5. kaggle kernels output → ./kaggle_output/
6. find ./kaggle_output -type f  (デバッグ表示)
7. find offline-run-* → wandb sync  → W&Bクラウド
```

---

## 実行時間の目安

| フェーズ | 時間 |
|---|---|
| GitHub Actions セットアップ（checkout/python/pip）| 約30秒 |
| Kaggleカーネル実行（シンプルなLRモデル）| QUEUED 30秒 + RUNNING 30秒 = 約1分 |
| kaggle kernels output（ダウンロード）| 数秒 |
| wandb sync | 数秒 |
| **合計** | **約2分** |

LightGBMなど重いモデルでも基本的に同じフロー（実行時間が延びるだけ）。

---

## v2実装時のチェックリスト

- [ ] このワークフロー（`kaggle-wandb-sync.yml`）をベースにする
- [ ] Notebookに `os.environ['WANDB_MODE'] = 'offline'` を import 前に追加
- [ ] kernel-metadata.jsonのタイトルとidのslugを一致させる
- [ ] WANDB_API_KEY は GitHub Secrets に登録済み（`yasumorishima/kaggle-competitions`）
- [ ] W&Bでプロジェクト名を確認: `fw_yasu11-personal/march-mania-2026-test`
