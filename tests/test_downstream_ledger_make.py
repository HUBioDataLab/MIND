"""Make targets of the results ledger: dry-run commands and required-variable guards only.

``make -n`` prints recipes without running them; the guard tests run only the first ``@test`` line, which
fails before any Python starts. Nothing is written.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _make(*args, env_root=None):
    env = {k: v for k, v in os.environ.items() if k != "MIND_RESULTS_ROOT"}
    if env_root:
        env["MIND_RESULTS_ROOT"] = env_root
    result = subprocess.run(["make", "--no-print-directory", *args], cwd=REPO, env=env, capture_output=True, text=True)
    result.stdout = " ".join(result.stdout.replace("\\\n", " ").split())  # join recipe continuation lines
    return result


@pytest.mark.parametrize("target, args, expected", [
    ("ds-ledger-add", ["OUTPUTS_ROOT=/o", "EVAL_DIRS=/e1 /e2"],
     'python -m downstream.ledger add --outputs-root "/o" --eval-dir "/e1" --eval-dir "/e2"'),
    ("ds-ledger-annotate", ["SELECT=task=ec data_revision=r1", "STATUS=superseded", "REASON=old set", "REPLACED_BY=r2"],
     'python -m downstream.ledger annotate --select "task=ec data_revision=r1" --status superseded --reason "old set" '
     '--replaced-by "r2"'),
    ("ds-ledger-report", ["MODELS=m1 m2"], "python -m downstream.ledger_report report --models m1 m2"),
    ("ds-summary", [], "python -m downstream.ledger_report summary"),
    ("ds-missing", ["MODELS=m1"], "python -m downstream.ledger_report missing --models m1"),
    ("ds-compare", ["A=m1", "B=m2", "TASKS=ec lba", "BOOTSTRAP=200"],
     'python -m downstream.ledger_compare --a "m1" --b "m2" --bootstrap 200 --tasks ec lba'),
    ("ds-ledger-verify", [], "python -m downstream.ledger verify"),
    ("ds-ledger-recompute", ["MODEL=m1", "CONTROL=m2", "DATA_REVISION=r2", "DRY_RUN=1"],
     'python -m downstream.ledger recompute-macro-aupr --task ec --model "m1" --control "m2" --data-revision "r2" '
     '--dry-run'),
    ("ds-ledger-recompute", ["TASK=go_mf", "MODEL=m1", "CONTROL=m2", "DATA_REVISION=r2", "POOLING=ca"],
     'python -m downstream.ledger recompute-macro-aupr --task go_mf --model "m1" --control "m2" --data-revision "r2" '
     '--pooling ca'),
])
def test_dry_run_commands(target, args, expected):
    result = _make("-n", target, *args)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
    assert 'MIND_RESULTS_ROOT="' in result.stdout and 'CUDA_VISIBLE_DEVICES=""' in result.stdout


def test_results_root_is_overridable_by_variable_and_environment():
    assert 'MIND_RESULTS_ROOT="/tmp/x"' in _make("-n", "ds-ledger-verify", "MIND_RESULTS_ROOT=/tmp/x").stdout
    assert 'MIND_RESULTS_ROOT="/tmp/y"' in _make("-n", "ds-ledger-verify", env_root="/tmp/y").stdout


@pytest.mark.parametrize("target, args, message", [
    ("ds-ledger-add", [], "OUTPUTS_ROOT"),
    ("ds-ledger-annotate", ["STATUS=superseded", "REASON=x"], "SELECT"),
    ("ds-compare", ["A=m1"], "set A and B"),
    ("ds-ledger-recompute", ["MODEL=m1", "CONTROL=m2"], "DATA_REVISION"),
])
def test_missing_variables_stop_before_python(target, args, message):
    result = _make(target, *args)
    assert result.returncode != 0 and message in result.stdout + result.stderr
    assert "downstream.ledger" not in result.stdout


def test_ds_report_is_unchanged():
    assert "python -m downstream.report --task ec" in _make("-n", "ds-report", "TASK=ec").stdout
