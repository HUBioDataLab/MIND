"""create_chunk_metadata must fail (exception, non-zero exit) when a chunk fails (B-8)."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from chunk_fixtures import write_chunk
from data_loading.create_chunk_metadata import process_chunk_directory

REPO = Path(__file__).resolve().parents[1]


def test_metadata_directory_run_fails_when_a_chunk_fails(tmp_path):
    good = write_chunk(tmp_path, "good", with_metadata=False)
    directory = tmp_path / "chunks"
    directory.mkdir()
    shutil.copy(good, directory / "good.pt")
    (directory / "corrupt.pt").write_bytes(b"not a torch file")
    with pytest.raises(RuntimeError, match="corrupt.pt"):
        process_chunk_directory(str(directory))
    # The healthy chunk still gets its metadata before the run reports the failure.
    assert (directory / "good_metadata.json").exists()


def test_metadata_command_exits_non_zero_when_a_chunk_fails(tmp_path):
    directory = tmp_path / "chunks"
    directory.mkdir()
    (directory / "corrupt.pt").write_bytes(b"not a torch file")
    result = subprocess.run(
        [sys.executable, "-m", "data_loading.create_chunk_metadata", "--chunk-dir", str(directory)],
        cwd=REPO, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "corrupt.pt" in result.stderr


def test_metadata_directory_without_chunk_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no .pt"):
        process_chunk_directory(str(tmp_path))
