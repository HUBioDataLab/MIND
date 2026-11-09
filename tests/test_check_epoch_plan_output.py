"""`make check-epoch-plan` writes its full output to a log and prints one summary line."""

import os
import stat
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from core import check_epoch_plan

REPO = Path(__file__).resolve().parents[1]
ROWS = [
    {"epoch": 0, "natural_batches": 90, "actual_total_atoms": 900, "mean_atoms_per_batch": 10.0},
    {"epoch": 1, "natural_batches": 95, "actual_total_atoms": 950, "mean_atoms_per_batch": 10.0},
]


def _run_main(capsys, fixed):
    argv = [
        "check_epoch_plan",
        "--config-yaml-path",
        str(REPO / "core/pretraining_config_protein_only.yaml"),
        "--epochs",
        "2",
    ]
    with mock.patch.object(sys, "argv", argv), mock.patch.object(
        check_epoch_plan, "natural_plan_counts", return_value=(fixed, ROWS)
    ):
        code = check_epoch_plan.main()
    return code, capsys.readouterr()


def test_ok_run_prints_one_summary_line(capsys):
    code, captured = _run_main(capsys, fixed=100)
    assert code == 0
    lines = [l for l in captured.out.splitlines() if l.startswith("PLAN_SUMMARY: ")]
    assert len(lines) == 1
    assert "epoch_num_batches=100" in lines[0]
    assert "natural_max=95" in lines[0]
    assert "margin=5" in lines[0]
    assert "status=OK" in lines[0]


def test_failing_run_reports_fail_summary_and_exit_code(capsys):
    code, captured = _run_main(capsys, fixed=90)
    assert code == 1
    assert "FAIL: natural plan needs up to 95" in captured.err
    lines = [l for l in captured.out.splitlines() if l.startswith("PLAN_SUMMARY: ")]
    assert len(lines) == 1 and "status=FAIL" in lines[0]


def _stub_python(tmp_path, body):
    """A fake `python` first on PATH so `make` never runs the real, slow check."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python"
    stub.write_text("#!/bin/bash\n" + body)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def _make(tmp_path, bin_dir):
    log_dir = tmp_path / "logs"
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    result = subprocess.run(
        ["make", "check-epoch-plan", "CONDA_RUN=", f"LOG_MONTH_DIR={log_dir}", "CONFIG=f"],
        cwd=REPO, env=env, capture_output=True, text=True,
    )
    return result, log_dir


def test_target_logs_full_output_and_prints_summary_only(tmp_path):
    body = (
        'for i in 1 2 3 4 5; do echo "{ \\"detail\\": $i }"; done\n'
        'echo "PLAN_SUMMARY: config=x epoch_num_batches=100 natural_max=95 margin=5 status=OK"\n'
    )
    result, log_dir = _make(tmp_path, _stub_python(tmp_path, body))
    assert result.returncode == 0, result.stderr
    out_lines = [l for l in result.stdout.splitlines() if l.strip()]
    assert not any("detail" in l for l in out_lines)
    assert sum(l.startswith("PLAN_SUMMARY: ") for l in out_lines) == 1
    logs = list(log_dir.glob("check_epoch_plan_*.log"))
    assert len(logs) == 1
    assert str(logs[0]) in result.stdout
    assert "detail" in logs[0].read_text()


def test_target_failure_keeps_exit_code_and_shows_error(tmp_path):
    body = 'echo "Traceback: boom in the sampler" >&2\nexit 3\n'
    result, log_dir = _make(tmp_path, _stub_python(tmp_path, body))
    assert result.returncode != 0
    assert "boom in the sampler" in result.stderr
    logs = list(log_dir.glob("check_epoch_plan_*.log"))
    assert len(logs) == 1 and str(logs[0]) in (result.stdout + result.stderr)
