"""kaggle-deploy push: KaggleにNotebookをプッシュする."""

import json
import subprocess
import sys
from pathlib import Path

import click


@click.command()
@click.argument("directory", default=".")
@click.option("--skip-validate", is_flag=True, default=False, help="バリデーションをスキップ")
@click.option("--dry-run", is_flag=True, default=False, help="実行せずにコマンドを表示")
def push(directory, skip_validate, dry_run):
    """KaggleにNotebookをプッシュする.

    DIRECTORY はkernel-metadata.jsonを含むディレクトリです（デフォルト: カレントディレクトリ）。

    内部で `kaggle kernels push -p <directory>` を実行します。
    事前に `kaggle` CLIのインストールと認証情報の設定が必要です。
    """
    dir_path = Path(directory)
    metadata_path = dir_path / "kernel-metadata.json"

    if not metadata_path.exists():
        click.echo(f"Error: {metadata_path} が見つかりません。", err=True)
        raise SystemExit(1)

    # バリデーション
    if not skip_validate:
        from kaggle_deploy.commands.validate import validate as validate_cmd

        ctx = click.Context(validate_cmd, info_name="validate")
        try:
            ctx.invoke(validate_cmd, directory=directory)
        except SystemExit as e:
            if e.code != 0:
                click.echo("")
                click.echo("バリデーションエラーがあります。--skip-validate で無視できます。", err=True)
                raise SystemExit(1)

    # メタデータ表示
    with open(metadata_path) as f:
        metadata = json.load(f)

    click.echo("")
    click.echo("Push対象:")
    click.echo(f"  Kernel: {metadata['id']}")
    click.echo(f"  File:   {metadata['code_file']}")
    click.echo(f"  GPU:    {metadata['enable_gpu']}")
    click.echo(f"  Private: {metadata['is_private']}")

    cmd = ["kaggle", "kernels", "push", "-p", str(dir_path)]

    if dry_run:
        click.echo("")
        click.echo(f"Dry run: {' '.join(cmd)}")
        return

    click.echo("")
    click.echo("Pushing to Kaggle...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        click.echo("Error: kaggle コマンドが見つかりません。", err=True)
        click.echo("  pip install kaggle でインストールしてください。", err=True)
        raise SystemExit(1)

    if result.stdout:
        click.echo(result.stdout.rstrip())
    if result.stderr:
        click.echo(result.stderr.rstrip(), err=True)

    if result.returncode != 0:
        raise SystemExit(result.returncode)

    click.echo("")
    click.echo("次のステップ:")
    click.echo("  ブラウザでKaggle Notebook画面を開き「Submit to Competition」をクリック")

    # ステータス確認コマンドの案内
    kernel_id = metadata["id"]
    click.echo(f"  kaggle kernels status {kernel_id}")
