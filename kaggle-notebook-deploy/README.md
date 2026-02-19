# kaggle-notebook-deploy

`git push` するだけで Kaggle Notebook をデプロイする CLI ツール。

Kaggle Notebook のコード管理を GitHub に集約し、GitHub Actions 経由で自動デプロイするワークフローをセットアップします。

## ワークフロー

```
ノートブック編集 → git push → GitHub Actions → Kaggle にアップロード → ブラウザで Submit
```

## インストール

```bash
pip install kaggle-notebook-deploy
```

## クイックスタート

### 1. リポジトリのセットアップ

```bash
# GitHub Actions ワークフローと .gitignore を生成
kaggle-notebook-deploy init-repo
```

生成されるファイル:
- `.github/workflows/kaggle-push.yml` - Kaggle プッシュ用ワークフロー
- `scripts/setup-credentials.sh` - 認証情報セットアップスクリプト
- `.gitignore` への Kaggle 関連パターン追記

### 2. GitHub Secrets の設定

```bash
gh secret set KAGGLE_USERNAME
gh secret set KAGGLE_KEY
```

### 3. コンペ用ディレクトリの作成

```bash
# 基本
kaggle-notebook-deploy init titanic

# GPU有効・公開Notebook
kaggle-notebook-deploy init deep-past-initiative-machine-translation --gpu --public
```

生成されるファイル:
- `<slug>/kernel-metadata.json` - Kaggle カーネルメタデータ
- `<slug>/<slug>-baseline.ipynb` - ベースライン Notebook

### 4. 開発・デプロイ

```bash
# Notebook を編集
vim titanic/titanic-baseline.ipynb

# バリデーション
kaggle-notebook-deploy validate titanic

# ローカルからプッシュ
kaggle-notebook-deploy push titanic

# または GitHub Actions 経由
git add titanic/ && git commit -m "Add titanic baseline" && git push
gh workflow run kaggle-push.yml -f notebook_dir=titanic
```

## コマンド一覧

### `kaggle-notebook-deploy init <competition-slug>`

コンペ用ディレクトリを雛形から生成します。

| オプション | 説明 |
|---|---|
| `-u, --username` | Kaggle ユーザー名（省略時は ~/.kaggle/kaggle.json から取得） |
| `-t, --title` | Notebook タイトル（省略時はスラッグから自動生成） |
| `--gpu` | GPU 有効化 |
| `--internet` | インターネット有効化（コードコンペでは非推奨） |
| `--public` | 公開 Notebook として作成 |

### `kaggle-notebook-deploy init-repo`

GitHub Actions ワークフローと関連ファイルをセットアップします。

| オプション | 説明 |
|---|---|
| `-f, --force` | 既存ファイルを上書き |

### `kaggle-notebook-deploy validate [directory]`

`kernel-metadata.json` のバリデーションを行います。

チェック項目:
- 必須フィールドの存在
- `id` のフォーマット（`username/slug`）
- `code_file` の存在
- `language`, `kernel_type` の有効値
- `enable_internet` と `competition_sources` の整合性

### `kaggle-notebook-deploy push [directory]`

Kaggle に Notebook をプッシュします（内部で `kaggle kernels push` を実行）。

| オプション | 説明 |
|---|---|
| `--skip-validate` | バリデーションをスキップ |
| `--dry-run` | 実行せずにコマンドを表示 |

## 注意事項

### コードコンペでの制約

- `enable_internet: false` が必須（true だと提出不可）
- API での Submit は制限あり（手動ブラウザ Submit が必要）
- `kaggle kernels push` は Kaggle Secrets の紐付けをリセットするため、W&B 等を使う場合は Web UI で再設定が必要

### データパスの違い

| ソース | マウントパス |
|---|---|
| `competition_sources` | `/kaggle/input/competitions/<slug>/` |
| `dataset_sources` | `/kaggle/input/<slug>/` |

## ライセンス

MIT
