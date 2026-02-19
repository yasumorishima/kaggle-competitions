"""kaggle-deploy init-repo: リポジトリにGitHub Actionsワークフローをセットアップする."""

from pathlib import Path

import click


WORKFLOW_TEMPLATE = """\
name: Push Notebook to Kaggle

on:
  workflow_dispatch:
    inputs:
      notebook_dir:
        description: "Notebookが格納されたディレクトリ名"
        required: true
        type: string

jobs:
  push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install Kaggle CLI
        run: pip install kaggle

      - name: Push notebook to Kaggle
        env:
          KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
          KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
        run: kaggle kernels push -p ${{ inputs.notebook_dir }}
"""

GITIGNORE_ADDITIONS = """\
# === Kaggle Deploy ===
# Data files
*.csv
*.parquet
*.h5
*.hdf5
*.pkl
*.pickle
*.feather
*.arrow

# Model files
*.joblib
*.bin
*.onnx
*.pt
*.pth
*.safetensors

# Archive files
*.zip
*.tar.gz
*.7z

# Jupyter checkpoints
.ipynb_checkpoints/

# Submissions
submission*.csv

# Credentials (NEVER commit these)
.kaggle/
kaggle.json

# Virtual environments
.venv/
venv/
"""

CREDENTIALS_SCRIPT = """\
#!/bin/bash
# Kaggle API認証情報のセットアップ
# 新しいデバイスやCI環境で実行してください

set -e

# Kaggle
if [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
    mkdir -p ~/.kaggle
    cat > ~/.kaggle/kaggle.json << EOF
{"username": "$KAGGLE_USERNAME", "key": "$KAGGLE_KEY"}
EOF
    chmod 600 ~/.kaggle/kaggle.json
    echo "Kaggle credentials configured."
elif [ -f ~/.kaggle/kaggle.json ]; then
    echo "Kaggle credentials already exist."
else
    echo "Warning: KAGGLE_USERNAME and KAGGLE_KEY not set. Skipping Kaggle setup."
fi
"""


@click.command("init-repo")
@click.option("--force", "-f", is_flag=True, default=False, help="既存ファイルを上書きする")
def init_repo(force):
    """リポジトリにGitHub Actionsワークフローと関連ファイルをセットアップする.

    カレントディレクトリに以下を生成します:
    - .github/workflows/kaggle-push.yml
    - scripts/setup-credentials.sh
    - .gitignore への追記
    """
    created = []

    # GitHub Actions workflow
    workflow_dir = Path(".github/workflows")
    workflow_path = workflow_dir / "kaggle-push.yml"

    if workflow_path.exists() and not force:
        click.echo(f"  Skip: {workflow_path} (既に存在。--force で上書き)")
    else:
        workflow_dir.mkdir(parents=True, exist_ok=True)
        workflow_path.write_text(WORKFLOW_TEMPLATE)
        created.append(str(workflow_path))
        click.echo(f"  {workflow_path}")

    # Credentials script
    scripts_dir = Path("scripts")
    cred_path = scripts_dir / "setup-credentials.sh"

    if cred_path.exists() and not force:
        click.echo(f"  Skip: {cred_path} (既に存在。--force で上書き)")
    else:
        scripts_dir.mkdir(parents=True, exist_ok=True)
        cred_path.write_text(CREDENTIALS_SCRIPT)
        cred_path.chmod(0o755)
        created.append(str(cred_path))
        click.echo(f"  {cred_path}")

    # .gitignore
    gitignore_path = Path(".gitignore")
    marker = "# === Kaggle Deploy ==="

    if gitignore_path.exists():
        existing = gitignore_path.read_text()
        if marker in existing and not force:
            click.echo(f"  Skip: {gitignore_path} (Kaggle Deployセクション追加済み)")
        else:
            if marker not in existing:
                with open(gitignore_path, "a") as f:
                    f.write("\n" + GITIGNORE_ADDITIONS)
                created.append(str(gitignore_path) + " (追記)")
                click.echo(f"  {gitignore_path} (追記)")
    else:
        gitignore_path.write_text(GITIGNORE_ADDITIONS)
        created.append(str(gitignore_path))
        click.echo(f"  {gitignore_path}")

    # サマリ
    click.echo("")
    if created:
        click.echo(f"{len(created)}個のファイルをセットアップしました。")
    else:
        click.echo("全てのファイルが既に存在しています。")

    click.echo("")
    click.echo("次のステップ:")
    click.echo("  1. GitHub Secretsを設定:")
    click.echo("     gh secret set KAGGLE_USERNAME")
    click.echo("     gh secret set KAGGLE_KEY")
    click.echo("  2. コンペ用ディレクトリを作成:")
    click.echo("     kaggle-deploy init <competition-slug>")
