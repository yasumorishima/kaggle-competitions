"""kaggle-notebook-deploy CLI tests."""

import json
import os

from click.testing import CliRunner

from kaggle_notebook_deploy.cli import main


runner = CliRunner()


def test_version():
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_help():
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "init" in result.output
    assert "init-repo" in result.output
    assert "validate" in result.output
    assert "push" in result.output


class TestInit:
    def test_basic(self, tmp_path):
        os.chdir(tmp_path)
        result = runner.invoke(main, ["init", "titanic", "-u", "testuser"])
        assert result.exit_code == 0
        assert "titanic/" in result.output

        # kernel-metadata.json
        metadata_path = tmp_path / "titanic" / "kernel-metadata.json"
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text())
        assert metadata["id"] == "testuser/titanic-baseline"
        assert metadata["competition_sources"] == ["titanic"]
        assert metadata["is_private"] == "true"
        assert metadata["enable_internet"] == "false"

        # notebook
        notebook_path = tmp_path / "titanic" / "titanic-baseline.ipynb"
        assert notebook_path.exists()
        nb = json.loads(notebook_path.read_text())
        assert nb["nbformat"] == 4

    def test_gpu_public(self, tmp_path):
        os.chdir(tmp_path)
        result = runner.invoke(main, ["init", "my-comp", "-u", "user1", "--gpu", "--public"])
        assert result.exit_code == 0

        metadata = json.loads((tmp_path / "my-comp" / "kernel-metadata.json").read_text())
        assert metadata["enable_gpu"] == "true"
        assert metadata["is_private"] == "false"

    def test_custom_title(self, tmp_path):
        os.chdir(tmp_path)
        result = runner.invoke(main, ["init", "test-comp", "-u", "user1", "-t", "My Custom Title"])
        assert result.exit_code == 0

        metadata = json.loads((tmp_path / "test-comp" / "kernel-metadata.json").read_text())
        assert metadata["title"] == "My Custom Title"

    def test_already_exists(self, tmp_path):
        os.chdir(tmp_path)
        (tmp_path / "existing").mkdir()
        result = runner.invoke(main, ["init", "existing", "-u", "user1"])
        assert result.exit_code == 1
        assert "既に存在" in result.output

    def test_internet_warning(self, tmp_path):
        os.chdir(tmp_path)
        result = runner.invoke(main, ["init", "test-inet", "-u", "user1", "--internet"])
        assert result.exit_code == 0
        assert "enable_internet=true" in result.output


class TestInitRepo:
    def test_basic(self, tmp_path):
        os.chdir(tmp_path)
        result = runner.invoke(main, ["init-repo"])
        assert result.exit_code == 0

        assert (tmp_path / ".github" / "workflows" / "kaggle-push.yml").exists()
        assert (tmp_path / "scripts" / "setup-credentials.sh").exists()
        assert (tmp_path / ".gitignore").exists()

        # workflow content
        workflow = (tmp_path / ".github" / "workflows" / "kaggle-push.yml").read_text()
        assert "kaggle kernels push" in workflow
        assert "KAGGLE_USERNAME" in workflow

    def test_skip_existing(self, tmp_path):
        os.chdir(tmp_path)
        # first run
        runner.invoke(main, ["init-repo"])
        # second run
        result = runner.invoke(main, ["init-repo"])
        assert "Skip" in result.output
        assert "既に存在" in result.output

    def test_force(self, tmp_path):
        os.chdir(tmp_path)
        runner.invoke(main, ["init-repo"])
        result = runner.invoke(main, ["init-repo", "--force"])
        assert "Skip" not in result.output

    def test_append_gitignore(self, tmp_path):
        os.chdir(tmp_path)
        (tmp_path / ".gitignore").write_text("# existing rules\nnode_modules/\n")
        result = runner.invoke(main, ["init-repo"])
        assert result.exit_code == 0

        content = (tmp_path / ".gitignore").read_text()
        assert "node_modules/" in content
        assert "# === Kaggle Deploy ===" in content


class TestValidate:
    def _make_valid_dir(self, tmp_path):
        """Create a valid competition directory for testing."""
        comp_dir = tmp_path / "test-comp"
        comp_dir.mkdir()
        metadata = {
            "id": "testuser/test-comp-baseline",
            "title": "Test Comp Baseline",
            "code_file": "test-comp-baseline.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "false",
            "enable_internet": "false",
            "competition_sources": ["test-comp"],
        }
        (comp_dir / "kernel-metadata.json").write_text(json.dumps(metadata))
        (comp_dir / "test-comp-baseline.ipynb").write_text("{}")
        return comp_dir

    def test_valid(self, tmp_path):
        comp_dir = self._make_valid_dir(tmp_path)
        result = runner.invoke(main, ["validate", str(comp_dir)])
        assert result.exit_code == 0
        assert "OK" in result.output

    def test_missing_metadata(self, tmp_path):
        result = runner.invoke(main, ["validate", str(tmp_path)])
        assert result.exit_code == 1
        assert "見つかりません" in result.output

    def test_missing_code_file(self, tmp_path):
        comp_dir = tmp_path / "bad"
        comp_dir.mkdir()
        metadata = {
            "id": "user/slug",
            "title": "Title",
            "code_file": "nonexistent.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "false",
            "enable_internet": "false",
        }
        (comp_dir / "kernel-metadata.json").write_text(json.dumps(metadata))
        result = runner.invoke(main, ["validate", str(comp_dir)])
        assert result.exit_code == 1
        assert "見つかりません" in result.output

    def test_bad_id_format(self, tmp_path):
        comp_dir = tmp_path / "bad-id"
        comp_dir.mkdir()
        metadata = {
            "id": "no-slash-here",
            "title": "Title",
            "code_file": "test.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "false",
            "enable_internet": "false",
        }
        (comp_dir / "kernel-metadata.json").write_text(json.dumps(metadata))
        (comp_dir / "test.ipynb").write_text("{}")
        result = runner.invoke(main, ["validate", str(comp_dir)])
        assert result.exit_code == 1
        assert "username/slug" in result.output

    def test_internet_warning(self, tmp_path):
        comp_dir = tmp_path / "inet"
        comp_dir.mkdir()
        metadata = {
            "id": "user/slug",
            "title": "Title",
            "code_file": "test.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "false",
            "enable_internet": "true",
            "competition_sources": ["some-comp"],
        }
        (comp_dir / "kernel-metadata.json").write_text(json.dumps(metadata))
        (comp_dir / "test.ipynb").write_text("{}")
        result = runner.invoke(main, ["validate", str(comp_dir)])
        assert result.exit_code == 0  # warning, not error
        assert "enable_internet=true" in result.output


class TestPush:
    def test_dry_run(self, tmp_path):
        comp_dir = tmp_path / "dry"
        comp_dir.mkdir()
        metadata = {
            "id": "user/dry-baseline",
            "title": "Dry Baseline",
            "code_file": "dry-baseline.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "false",
            "enable_internet": "false",
        }
        (comp_dir / "kernel-metadata.json").write_text(json.dumps(metadata))
        (comp_dir / "dry-baseline.ipynb").write_text("{}")

        result = runner.invoke(main, ["push", str(comp_dir), "--dry-run"])
        assert result.exit_code == 0
        assert "Dry run" in result.output
        assert "kaggle kernels push" in result.output

    def test_missing_dir(self):
        result = runner.invoke(main, ["push", "/nonexistent/path"])
        assert result.exit_code == 1
