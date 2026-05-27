# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from vllm.transformers_utils.repo_utils import (
    any_pattern_in_repo_files,
    get_model_path,
    is_mistral_model_repo,
    list_filtered_repo_files,
)


@pytest.mark.parametrize(
    "allow_patterns,expected_relative_files",
    [
        (
            ["*.json", "correct*.txt"],
            ["json_file.json", "subfolder/correct.txt", "correct_2.txt"],
        ),
    ],
)
def test_list_filtered_repo_files(
    allow_patterns: list[str], expected_relative_files: list[str]
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Prep folder and files
        path_tmp_dir = Path(tmp_dir)
        subfolder = path_tmp_dir / "subfolder"
        subfolder.mkdir()
        (path_tmp_dir / "json_file.json").touch()
        (path_tmp_dir / "correct_2.txt").touch()
        (path_tmp_dir / "incorrect.txt").touch()
        (path_tmp_dir / "incorrect.jpeg").touch()
        (subfolder / "correct.txt").touch()
        (subfolder / "incorrect_sub.txt").touch()

        def _glob_path() -> list[str]:
            return [
                str(file.relative_to(path_tmp_dir))
                for file in path_tmp_dir.glob("**/*")
                if file.is_file()
            ]

        # Patch list_repo_files called by fn
        with patch(
            "vllm.transformers_utils.repo_utils.list_repo_files",
            MagicMock(return_value=_glob_path()),
        ) as mock_list_repo_files:
            out_files = sorted(
                list_filtered_repo_files(
                    tmp_dir, allow_patterns, "revision", "model", "token"
                )
            )
        assert out_files == sorted(expected_relative_files)
        assert mock_list_repo_files.call_count == 1
        assert mock_list_repo_files.call_args_list[0] == call(
            repo_id=tmp_dir,
            revision="revision",
            repo_type="model",
            token="token",
        )


@pytest.mark.parametrize(
    ("allow_patterns", "expected_bool"),
    [
        (["*.json", "correct*.txt"], True),
        (
            ["*.jpeg"],
            True,
        ),
        (
            ["not_found.jpeg"],
            False,
        ),
    ],
)
def test_one_filtered_repo_files(allow_patterns: list[str], expected_bool: bool):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Prep folder and files
        path_tmp_dir = Path(tmp_dir)
        subfolder = path_tmp_dir / "subfolder"
        subfolder.mkdir()
        (path_tmp_dir / "incorrect.jpeg").touch()
        (subfolder / "correct.txt").touch()

        def _glob_path() -> list[str]:
            return [
                str(file.relative_to(path_tmp_dir))
                for file in path_tmp_dir.glob("**/*")
                if file.is_file()
            ]

        # Patch list_repo_files called by fn
        with patch(
            "vllm.transformers_utils.repo_utils.list_repo_files",
            MagicMock(return_value=_glob_path()),
        ) as mock_list_repo_files:
            assert (
                any_pattern_in_repo_files(
                    tmp_dir, allow_patterns, "revision", "model", "token"
                )
            ) is expected_bool
        assert mock_list_repo_files.call_count == 1
        assert mock_list_repo_files.call_args_list[0] == call(
            repo_id=tmp_dir,
            revision="revision",
            repo_type="model",
            token="token",
        )


@pytest.mark.parametrize(
    ("files", "expected_bool"),
    [
        (["consolidated.safetensors", "incorrect.txt"], True),
        (["consolidated-1.safetensors", "incorrect.txt"], True),
        (
            ["consolidated-1.json"],
            False,
        ),
    ],
)
def test_is_mistral_model_repo(files: list[str], expected_bool: bool):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Prep folder and files
        path_tmp_dir = Path(tmp_dir)
        for file in files:
            (path_tmp_dir / file).touch()

        def _glob_path() -> list[str]:
            return [
                str(file.relative_to(path_tmp_dir))
                for file in path_tmp_dir.glob("**/*")
                if file.is_file()
            ]

        # Patch list_repo_files called by fn
        with patch(
            "vllm.transformers_utils.repo_utils.list_repo_files",
            MagicMock(return_value=_glob_path()),
        ) as mock_list_repo_files:
            assert (
                is_mistral_model_repo(tmp_dir, "revision", "model", "token")
                is expected_bool
            )
        assert mock_list_repo_files.call_count == 1
        assert mock_list_repo_files.call_args_list[0] == call(
            repo_id=tmp_dir,
            revision="revision",
            repo_type="model",
            token="token",
        )


# ---------------------------------------------------------------------------
# get_model_path: VLLM_MODEL_REDIRECT_PATH integration
# ---------------------------------------------------------------------------


def test_get_model_path_resolves_redirect_to_local_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A redirected repo id resolves to a local directory and skips
    snapshot_download — the offline-staging case under HF_HUB_OFFLINE=1.
    """
    local_model_dir = tmp_path / "tiny-random-llama"
    local_model_dir.mkdir()

    redirect_json = tmp_path / "aliases.json"
    redirect_json.write_text(
        '{"hmellor/tiny-random-LlamaForCausalLM": '
        f'"{local_model_dir}"}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("VLLM_MODEL_REDIRECT_PATH", str(redirect_json))

    # Defeat the @cache on maybe_model_redirect so the test is hermetic
    # across parametrisations / repeated invocations in the same session.
    from vllm.transformers_utils.utils import maybe_model_redirect

    maybe_model_redirect.cache_clear()

    sentinel_called = {"snapshot": False}

    def boom(*_args, **_kwargs):
        sentinel_called["snapshot"] = True
        raise AssertionError(
            "snapshot_download must not be called when redirect resolves to "
            "an existing local directory"
        )

    with patch("huggingface_hub.snapshot_download", side_effect=boom):
        result = get_model_path("hmellor/tiny-random-LlamaForCausalLM")

    assert Path(result) == local_model_dir
    assert not sentinel_called["snapshot"]


def test_get_model_path_passes_through_when_no_redirect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Without VLLM_MODEL_REDIRECT_PATH, behaviour is unchanged: existing
    local paths round-trip and unredirected repo ids hit snapshot_download.
    """
    monkeypatch.delenv("VLLM_MODEL_REDIRECT_PATH", raising=False)

    from vllm.transformers_utils.utils import maybe_model_redirect

    maybe_model_redirect.cache_clear()

    local_dir = tmp_path / "already_local"
    local_dir.mkdir()
    assert Path(get_model_path(str(local_dir))) == local_dir

    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_OFFLINE", True, raising=False
    )

    sentinel = MagicMock(return_value="/cache/snap/path")
    with patch("huggingface_hub.snapshot_download", sentinel):
        out = get_model_path("some-org/some-repo", revision="main")

    assert out == "/cache/snap/path"
    sentinel.assert_called_once_with(
        repo_id="some-org/some-repo",
        local_files_only=True,
        revision="main",
    )

# Trigger AI review
