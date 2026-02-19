"""CLI entry point for kaggle-notebook-deploy."""

import click

from kaggle_notebook_deploy import __version__
from kaggle_notebook_deploy.commands.init import init
from kaggle_notebook_deploy.commands.init_repo import init_repo
from kaggle_notebook_deploy.commands.validate import validate
from kaggle_notebook_deploy.commands.push import push


@click.group()
@click.version_option(version=__version__)
def main():
    """git pushするだけでKaggle NotebookをデプロイするCLIツール

    Kaggle NotebookのコードをGitHubで管理し、GitHub Actions経由で
    自動デプロイするワークフローをセットアップします。
    """
    pass


main.add_command(init)
main.add_command(init_repo)
main.add_command(validate)
main.add_command(push)
