"""kaggle-deploy init: コンペ用ディレクトリを雛形から生成する."""

import json
import os
from pathlib import Path
from string import Template

import click


KERNEL_METADATA_TEMPLATE = {
    "id": "{username}/{slug}",
    "title": "{title}",
    "code_file": "{slug}.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "false",
    "enable_tpu": "false",
    "enable_internet": "false",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}

NOTEBOOK_TEMPLATE = Template("""{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# $title\\n",
    "\\n",
    "Competition: $competition"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "\\n",
    "# データの読み込み\\n",
    "# コンペデータ: /kaggle/input/competitions/<slug>/\\n",
    "# データセット: /kaggle/input/<slug>/"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}""")


def _get_kaggle_username() -> str:
    """Kaggle APIの認証情報からユーザー名を取得する."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        with open(kaggle_json) as f:
            data = json.load(f)
            return data.get("username", "your-username")

    # 環境変数からのフォールバック
    return os.environ.get("KAGGLE_USERNAME", "your-username")


@click.command()
@click.argument("competition_slug")
@click.option("--username", "-u", default=None, help="Kaggleユーザー名（省略時は~/.kaggle/kaggle.jsonから取得）")
@click.option("--title", "-t", default=None, help="Notebookのタイトル（省略時はslugから生成）")
@click.option("--gpu", is_flag=True, default=False, help="GPU有効化")
@click.option("--internet", is_flag=True, default=False, help="インターネット有効化（コードコンペでは非推奨）")
@click.option("--public", is_flag=True, default=False, help="公開Notebook（デフォルトは非公開）")
def init(competition_slug, username, title, gpu, internet, public):
    """コンペ用ディレクトリを雛形から生成する.

    COMPETITION_SLUG はKaggleコンペのスラッグ（URLの末尾部分）です。
    例: kaggle-deploy init titanic
    """
    if username is None:
        username = _get_kaggle_username()

    if title is None:
        title = competition_slug.replace("-", " ").title() + " Baseline"

    slug = competition_slug + "-baseline"

    # ディレクトリ作成
    dir_path = Path(competition_slug)
    if dir_path.exists():
        click.echo(f"Error: ディレクトリ '{competition_slug}' は既に存在します。", err=True)
        raise SystemExit(1)

    dir_path.mkdir(parents=True)

    # kernel-metadata.json 生成
    metadata = KERNEL_METADATA_TEMPLATE.copy()
    metadata["id"] = f"{username}/{slug}"
    metadata["title"] = title
    metadata["code_file"] = f"{slug}.ipynb"
    metadata["is_private"] = "false" if public else "true"
    metadata["enable_gpu"] = "true" if gpu else "false"
    metadata["enable_internet"] = "true" if internet else "false"
    metadata["competition_sources"] = [competition_slug]

    metadata_path = dir_path / "kernel-metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")

    click.echo(f"  {metadata_path}")

    # Notebook 生成
    notebook_content = NOTEBOOK_TEMPLATE.substitute(
        title=title,
        competition=competition_slug,
    )
    notebook_path = dir_path / f"{slug}.ipynb"
    with open(notebook_path, "w") as f:
        f.write(notebook_content)

    click.echo(f"  {notebook_path}")

    # サマリ
    click.echo("")
    click.echo(f"'{competition_slug}/' を作成しました。")
    click.echo("")
    click.echo("次のステップ:")
    click.echo(f"  1. {notebook_path} を編集")
    click.echo(f"  2. git add {competition_slug}/ && git commit && git push")
    click.echo(f"  3. gh workflow run kaggle-push.yml -f notebook_dir={competition_slug}")

    if internet:
        click.echo("")
        click.echo("Warning: enable_internet=true はコードコンペでは提出不可になります。")
