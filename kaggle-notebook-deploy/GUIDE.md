# kaggle-notebook-deploy: git pushするだけでKaggle Notebookをデプロイ

## はじめに

KaggleのNotebookをブラウザ上のエディタで直接編集していませんか？

ブラウザエディタでは、Gitによるバージョン管理ができない、差分の確認がしづらい、使い慣れたエディタ（VSCode等）が使えない、といった不便があります。

**kaggle-notebook-deploy** は、このワークフローをコマンド一発でセットアップするCLIツールです。

```
pip install kaggle-notebook-deploy
```

GitHub Actionsを使い、**`git push`するだけでKaggleにNotebookを自動デプロイ**できます。

## 完成するワークフロー

```
ノートブック編集 → git push → GitHub Actions → Kaggleにアップロード → ブラウザでSubmit
```

1. ローカルまたは任意のデバイスで `.ipynb` を編集
2. `git add && git commit && git push`
3. `gh workflow run kaggle-push.yml -f notebook_dir=<コンペ名>`
4. GitHub ActionsがKaggle APIでノートブックをアップロード
5. ブラウザでKaggleのNotebook画面を開き「Submit to Competition」をクリック

## セットアップ（5分で完了）

### Step 1: kaggle-notebook-deploy をインストール

```bash
pip install kaggle-notebook-deploy
```

### Step 2: Kaggle APIの認証情報を準備

[Kaggle Account Settings](https://www.kaggle.com/settings) から API Token を取得し、`~/.kaggle/kaggle.json` に配置します。

```bash
# すでに kaggle CLI を使っている場合はスキップ
mkdir -p ~/.kaggle
echo '{"username":"your-username","key":"your-api-key"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### Step 3: リポジトリの初期化

```bash
# 新しいリポジトリを作成（既存リポでもOK）
mkdir my-kaggle && cd my-kaggle && git init

# GitHub Actions ワークフローをセットアップ
kaggle-notebook-deploy init-repo
```

以下のファイルが自動生成されます：

```
my-kaggle/
├── .github/workflows/
│   └── kaggle-push.yml          # Kaggleプッシュ用ワークフロー
├── scripts/
│   └── setup-credentials.sh     # CI/新デバイス用の認証セットアップ
└── .gitignore                   # データファイル等を除外
```

### Step 4: GitHub Secretsの設定

GitHubにpushした後、リポジトリのSecretsにKaggle認証情報を登録します。

```bash
# GitHub CLI を使う場合
gh secret set KAGGLE_USERNAME
gh secret set KAGGLE_KEY
```

またはGitHubのWeb UI: **Settings > Secrets and variables > Actions** から設定。

## 使い方

### コンペに参加する

```bash
# コンペ用ディレクトリを作成（slugはKaggle URLの末尾部分）
kaggle-notebook-deploy init titanic
```

```
titanic/
├── kernel-metadata.json         # Kaggleに必要なメタデータ
└── titanic-baseline.ipynb       # ベースラインNotebook
```

`kernel-metadata.json` は以下の内容で自動生成されます：

```json
{
  "id": "your-username/titanic-baseline",
  "title": "Titanic Baseline",
  "code_file": "titanic-baseline.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "false",
  "enable_internet": "false",
  "competition_sources": ["titanic"]
}
```

### よく使うオプション

```bash
# GPU有効（XGBoost, LightGBM等のGPU学習に）
kaggle-notebook-deploy init my-competition --gpu

# 公開Notebook（メダル狙いのポートフォリオ用）
kaggle-notebook-deploy init my-competition --public

# タイトル指定
kaggle-notebook-deploy init my-competition --title "My Awesome Approach"

# ユーザー名を明示指定
kaggle-notebook-deploy init my-competition --username yasunorim
```

### デプロイ前のバリデーション

```bash
kaggle-notebook-deploy validate titanic
```

チェック内容：
- 必須フィールド（id, title, code_file 等）の存在
- `id` が `username/slug` 形式か
- `code_file` で指定したファイルが実際に存在するか
- `language`, `kernel_type` の値が有効か
- コードコンペで `enable_internet: true` になっていないか（警告）

### Kaggleにデプロイ

```bash
# ローカルから直接プッシュ（バリデーション付き）
kaggle-notebook-deploy push titanic

# バリデーションのみスキップ
kaggle-notebook-deploy push titanic --skip-validate

# 実行せずにコマンドを確認
kaggle-notebook-deploy push titanic --dry-run
```

### GitHub Actions経由でデプロイ

```bash
git add titanic/ && git commit -m "Update titanic baseline" && git push
gh workflow run kaggle-push.yml -f notebook_dir=titanic
```

GitHub Actionsのログで実行状況を確認できます：

```bash
gh run list --workflow=kaggle-push.yml
gh run view <run-id> --log
```

### Notebookの実行状況を確認

```bash
kaggle kernels status your-username/titanic-baseline
```

ステータス: `QUEUED` → `RUNNING` → `COMPLETE` or `ERROR`

## 実践例：Deep Past Challenge

実際に [Deep Past Challenge](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)（Akkadian→English翻訳コンペ）で運用した例です。

```bash
# コンペ用ディレクトリを作成
kaggle-notebook-deploy init deep-past-initiative-machine-translation \
  --title "Deep Past Akkadian Baseline"

# Notebookを編集（TF-IDFベースライン等）
vim deep-past-initiative-machine-translation/deep-past-initiative-machine-translation-baseline.ipynb

# バリデーション → デプロイ
kaggle-notebook-deploy validate deep-past-initiative-machine-translation
kaggle-notebook-deploy push deep-past-initiative-machine-translation
```

## ハマりポイントと対策

### 1. コードコンペでは `enable_internet: false` が必須

コードコンペ（Code Competition）では、提出するNotebookのインターネット接続が禁止されています。`enable_internet: true` だと提出ボタンが押せません。

`kaggle-notebook-deploy init` はデフォルトで `false` に設定するので、`--internet` フラグを付けなければ安全です。

### 2. データパスの罠

| ソース種別 | kernel-metadata.json のキー | マウントパス |
|---|---|---|
| コンペデータ | `competition_sources` | `/kaggle/input/competitions/<slug>/` |
| データセット | `dataset_sources` | `/kaggle/input/<slug>/` |

**`competition_sources` には `competitions/` サブディレクトリが入ります。** これは頻出のハマりポイントです。

Notebook内でデバッグするには：

```python
from pathlib import Path
for item in sorted(Path('/kaggle/input').iterdir()):
    print(f'  {item.name}/')
    for sub in sorted(item.iterdir()):
        print(f'    {sub.name} ({sub.stat().st_size:,} bytes)')
```

### 3. API経由のSubmitはできない

Kaggle APIの `CreateCodeSubmission` は公開トークンでは `kernelSessions.get` 権限がなく403になります（2026-02時点）。

そのため、**最後のSubmitだけはブラウザ操作が必要**です。これはKaggle側の制限であり、kaggle-notebook-deployのスコープ外です。

### 4. `kaggle kernels push` はSecretsをリセットする

Kaggle Notebookの「Secrets」（W&B APIキー等）は、`kaggle kernels push` のたびに紐付けが外れます。

**対策**: CLIでpush → ブラウザのWeb UIでSecretsを再度有効化 → Run

### 5. コンペルール同意を忘れずに

`competition_sources` でコンペデータをマウントするには、ブラウザでコンペルールに同意（Accept）する必要があります。同意前だとデータがマウントされず、FileNotFoundErrorになります。

## ディレクトリ構成の推奨パターン

```
my-kaggle/
├── .github/workflows/
│   └── kaggle-push.yml            # kaggle-notebook-deploy init-repo で生成
├── scripts/
│   └── setup-credentials.sh       # kaggle-notebook-deploy init-repo で生成
├── titanic/                       # kaggle-notebook-deploy init titanic で生成
│   ├── kernel-metadata.json
│   └── titanic-baseline.ipynb
├── house-prices/                  # kaggle-notebook-deploy init house-prices で生成
│   ├── kernel-metadata.json
│   └── house-prices-baseline.ipynb
├── .gitignore
└── README.md
```

コンペごとにディレクトリを分け、それぞれに `kernel-metadata.json` と Notebook を配置します。GitHub Actions のワークフローは1つで全コンペに対応（`notebook_dir` パラメータで切り替え）。

## コマンドリファレンス

### `kaggle-notebook-deploy init <competition-slug>`

コンペ用ディレクトリを雛形から生成。

| オプション | 説明 | デフォルト |
|---|---|---|
| `-u, --username` | Kaggleユーザー名 | `~/.kaggle/kaggle.json` から取得 |
| `-t, --title` | Notebookタイトル | slugから自動生成 |
| `--gpu` | GPU有効化 | off |
| `--internet` | インターネット有効化 | off（コードコンペ向け） |
| `--public` | 公開Notebook | off（非公開） |

### `kaggle-notebook-deploy init-repo`

リポジトリにGitHub Actionsワークフローをセットアップ。

| オプション | 説明 |
|---|---|
| `-f, --force` | 既存ファイルを上書き |

### `kaggle-notebook-deploy validate [directory]`

`kernel-metadata.json` のバリデーション。エラーがあれば終了コード1で終了。

### `kaggle-notebook-deploy push [directory]`

Kaggle にNotebookをプッシュ（内部で `kaggle kernels push` を実行）。

| オプション | 説明 |
|---|---|
| `--skip-validate` | バリデーションをスキップ |
| `--dry-run` | 実行せずにコマンドを表示 |

## なぜ完全自動化できないのか

理想は `git push` だけで提出まで完了することですが、以下のKaggle側の制限により、最後のSubmitは手動です：

1. **API制限**: `CreateCodeSubmission` APIは公開トークンでは権限不足（403）
2. **Secrets問題**: `kaggle kernels push` でSecretsの紐付けがリセットされる
3. **ルール同意**: コンペ参加にはブラウザでの同意が必要

それでも、**コード管理・差分確認・デプロイの大部分を自動化**できるだけで、開発体験は大きく向上します。

## まとめ

```bash
pip install kaggle-notebook-deploy          # インストール
kaggle-notebook-deploy init-repo            # リポジトリセットアップ
kaggle-notebook-deploy init titanic         # コンペ参加
# ... Notebookを編集 ...
kaggle-notebook-deploy push titanic         # デプロイ
```

GitHub のエコシステム（バージョン管理、Actions、Secrets、Pull Request）をそのままKaggleコンペに活用できます。ブラウザエディタからの脱却、ぜひ試してみてください。
