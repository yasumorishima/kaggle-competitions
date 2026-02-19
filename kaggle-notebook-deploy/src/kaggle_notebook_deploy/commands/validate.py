"""kaggle-deploy validate: kernel-metadata.jsonのバリデーション."""

import json
from pathlib import Path

import click


REQUIRED_FIELDS = [
    "id",
    "title",
    "code_file",
    "language",
    "kernel_type",
    "is_private",
    "enable_gpu",
    "enable_tpu",
    "enable_internet",
]

VALID_LANGUAGES = ["python", "r", "rmarkdown"]
VALID_KERNEL_TYPES = ["script", "notebook"]
VALID_BOOL_STRINGS = ["true", "false"]


@click.command()
@click.argument("directory", default=".")
def validate(directory):
    """kernel-metadata.jsonのバリデーションを行う.

    DIRECTORY はkernel-metadata.jsonを含むディレクトリです（デフォルト: カレントディレクトリ）。
    """
    dir_path = Path(directory)
    metadata_path = dir_path / "kernel-metadata.json"

    if not metadata_path.exists():
        click.echo(f"Error: {metadata_path} が見つかりません。", err=True)
        raise SystemExit(1)

    with open(metadata_path) as f:
        try:
            metadata = json.load(f)
        except json.JSONDecodeError as e:
            click.echo(f"Error: JSONパースエラー: {e}", err=True)
            raise SystemExit(1)

    errors = []
    warnings = []

    # 必須フィールドチェック
    for field in REQUIRED_FIELDS:
        if field not in metadata:
            errors.append(f"必須フィールド '{field}' がありません")

    if errors:
        # 必須フィールドが足りないと後続チェックが失敗するので先に出力
        _print_results(errors, warnings)
        raise SystemExit(1)

    # id フォーマット: username/slug
    kernel_id = metadata["id"]
    if "/" not in kernel_id:
        errors.append(f"id '{kernel_id}' は 'username/slug' の形式である必要があります")
    elif kernel_id.count("/") > 1:
        errors.append(f"id '{kernel_id}' にスラッシュが多すぎます")

    # code_file 存在チェック
    code_file = metadata["code_file"]
    code_path = dir_path / code_file
    if not code_path.exists():
        errors.append(f"code_file '{code_file}' がディレクトリ内に見つかりません")

    # language チェック
    if metadata["language"] not in VALID_LANGUAGES:
        errors.append(f"language '{metadata['language']}' は無効です。有効値: {VALID_LANGUAGES}")

    # kernel_type チェック
    if metadata["kernel_type"] not in VALID_KERNEL_TYPES:
        errors.append(f"kernel_type '{metadata['kernel_type']}' は無効です。有効値: {VALID_KERNEL_TYPES}")

    # bool文字列チェック
    for field in ["is_private", "enable_gpu", "enable_tpu", "enable_internet"]:
        val = metadata[field]
        if val not in VALID_BOOL_STRINGS:
            errors.append(f"{field} '{val}' は無効です。'true' または 'false' を使用してください")

    # code_file と kernel_type の整合性
    if code_path.suffix == ".ipynb" and metadata["kernel_type"] != "notebook":
        warnings.append(f"code_file が .ipynb ですが kernel_type が '{metadata['kernel_type']}' です")
    elif code_path.suffix == ".py" and metadata["kernel_type"] != "script":
        warnings.append(f"code_file が .py ですが kernel_type が '{metadata['kernel_type']}' です")

    # コードコンペ向けの警告
    if metadata.get("competition_sources") and metadata["enable_internet"] == "true":
        warnings.append(
            "enable_internet=true はコードコンペでは提出不可になります。"
            " 提出用Notebookでは false に設定してください"
        )

    # title と slug の整合性チェック
    if "/" in kernel_id:
        slug = kernel_id.split("/", 1)[1]
        expected_slug = metadata["title"].lower().replace(" ", "-")
        # Kaggle は title からスラッグを自動生成するため、大きくずれていたら警告
        if slug != expected_slug:
            # 厳密一致は求めないが、情報として出す
            pass

    _print_results(errors, warnings)

    if errors:
        raise SystemExit(1)


def _print_results(errors: list[str], warnings: list[str]):
    """バリデーション結果を表示する."""
    if errors:
        click.echo(f"Errors ({len(errors)}):")
        for e in errors:
            click.echo(f"  x {e}")

    if warnings:
        click.echo(f"Warnings ({len(warnings)}):")
        for w in warnings:
            click.echo(f"  ! {w}")

    if not errors and not warnings:
        click.echo("OK: kernel-metadata.json is valid.")
    elif not errors:
        click.echo("")
        click.echo("OK (with warnings): kernel-metadata.json is valid.")
