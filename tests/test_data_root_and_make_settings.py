"""MIND_DATA_ROOT expansion in configs and the machine-independent Makefile defaults (``make -n`` only, nothing runs)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

from core.pretraining_config import load_config_yaml
from data_loading import data_root as dr
from downstream.tasks import MOLNET_TASKS
from downstream.tasks.ec import dataset as ec_dataset

REPO = Path(__file__).resolve().parents[1]
# Settings a caller may have in its environment; MAKEFLAGS would pass an outer `make VAR=...` to the inner make.
LOCAL_SETTINGS = ("MIND_CONDA_ENV", "CONDA_ROOT", "MIND_DATA_ROOT", "MIND_MODELS_ROOT", "MIND_HF_HOME",
                  "MIND_RESULTS_ROOT", "MAKEFLAGS", "MFLAGS", "MAKELEVEL")
no_local_mk = pytest.mark.skipif((REPO / "local.mk").exists(), reason="a local.mk overrides the defaults")


def _make(*args, makefiles=(), extra_env=None):
    env = {k: v for k, v in os.environ.items() if k not in LOCAL_SETTINGS}
    env["XDG_CONFIG_HOME"] = "/nonexistent"  # keep a per-user local.mk out of the Makefile defaults under test
    env.update(extra_env or {})
    files = [arg for path in makefiles for arg in ("-f", str(path))]
    result = subprocess.run(["make", "--no-print-directory", *files, *args], cwd=REPO, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return " ".join(result.stdout.replace("\\\n", " ").split())


# -- data root -------------------------------------------------------------------------------------------

def test_expansion_uses_the_environment_and_defaults_to_repo_datasets(monkeypatch):
    monkeypatch.setenv("MIND_DATA_ROOT", "/data/x")
    assert dr.expand_data_root({"a": ["${MIND_DATA_ROOT}/p", 3], "b": "plain"}) == {"a": ["/data/x/p", 3], "b": "plain"}
    monkeypatch.setenv("MIND_DATA_ROOT", "")
    assert dr.expand_data_root("${MIND_DATA_ROOT}/p") == f"{REPO / 'datasets'}/p"
    monkeypatch.delenv("MIND_DATA_ROOT")
    assert dr.data_root() == REPO / "datasets"


def test_unknown_variable_is_refused(monkeypatch):
    monkeypatch.setenv("MIND_DATA_ROOT", "/data/x")
    with pytest.raises(ValueError, match="only"):
        dr.expand_data_root({"chunk_dir": "${MIND_DATA_DIR}/proteins"})


def test_pretraining_and_task_configs_resolve_under_the_data_root(monkeypatch):
    monkeypatch.setenv("MIND_DATA_ROOT", "/data/x")
    config = load_config_yaml(REPO / "core" / "pretraining_config_multidomain_1m_coconut.yaml")
    split = config["multi_domain"]["datasets"]["pdb"]["splits"]["train"]
    assert split["chunk_dir"] == "/data/x/proteins/processed_14atom_1m_edgefree_train"
    assert config["dataset_download_dir"].startswith("/data/x/")
    assert ec_dataset.load_task_config()["data_root"] == "/data/x/downstream/ec/protein"


def test_no_tracked_config_or_makefile_names_a_home_directory():
    tracked = subprocess.run(["git", "ls-files", "Makefile", "*.yaml"], cwd=REPO, capture_output=True, text=True,
                             check=True).stdout.split()
    assert tracked
    offenders = [path for path in tracked if "/home/" in (REPO / path).read_text()]
    assert offenders == []
    for path in tracked:
        if path.endswith(".yaml"):
            yaml.safe_load((REPO / path).read_text())


# -- Makefile settings ------------------------------------------------------------------------------------

@no_local_mk
def test_defaults_are_repo_relative_and_use_the_calling_environment():
    text = _make("-n", "ds-ledger-verify")
    assert f'MIND_RESULTS_ROOT="{REPO}/results"' in text
    assert "HF_HOME" not in text and "conda activate" not in text
    text = _make("-n", "ds-embed", "TASK=ec", "MODEL=esm2_t6_8M_UR50D", "GPU=1")
    assert f"--hf-cache-dir {REPO}/models/huggingface/hub" in text
    text = _make("-n", "ds-embed", "TASK=ec", "MODEL=esm2_t6_8M_UR50D", "GPU=1", "MIND_HF_HOME=/hf")
    assert 'export HF_HOME="/hf"' in text and "--hf-cache-dir /hf/hub" in text


def test_conda_environment_and_roots_come_from_local_mk(tmp_path):
    (tmp_path / "local.mk").write_text("MIND_CONDA_ENV ?= /envs/mind\nCONDA_ROOT ?= /opt/conda\n"
                                       "MIND_RESULTS_ROOT ?= /results/mind\nMIND_DATA_ROOT ?= /data/mind\n")
    if (REPO / "local.mk").exists():
        pytest.skip("a local.mk in the repository takes precedence over -I")
    text = _make("-I", str(tmp_path), "-n", "ds-ledger-verify")
    assert 'source "/opt/conda/etc/profile.d/conda.sh" && conda activate "/envs/mind"' in text
    assert 'MIND_RESULTS_ROOT="/results/mind"' in text
    text = _make("-I", str(tmp_path), "-n", "ds-ledger-verify", extra_env={"MIND_RESULTS_ROOT": "/env/results"})
    assert 'MIND_RESULTS_ROOT="/env/results"' in text


def test_data_and_model_roots_are_exported_to_every_recipe(tmp_path):
    probe = tmp_path / "probe.mk"
    probe.write_text("print-roots:\n\t@echo data=$$MIND_DATA_ROOT models=$$MIND_MODELS_ROOT\n")
    assert _make("print-roots", "MIND_DATA_ROOT=/d", "MIND_MODELS_ROOT=/m",
                 makefiles=(REPO / "Makefile", probe)) == "data=/d models=/m"
    assert _make("print-roots", makefiles=(REPO / "Makefile", probe),
                 extra_env={"MIND_DATA_ROOT": "/env/d"}).startswith("data=/env/d ")


def test_molnet_alias_expands_to_the_registered_moleculenet_tasks():
    tasks = ",".join(MOLNET_TASKS)
    assert f"--tasks {tasks} " in _make("-n", "ds-run", "MODEL=m", "GPU=0", "PLAN=1", "TASKS=molnet")
    assert f"--tasks {tasks},ec " in _make("-n", "ds-run", "MODEL=m", "GPU=0", "PLAN=1", "TASKS=molnet,ec")
    assert "--tasks all " in _make("-n", "ds-run", "MODEL=m", "GPU=0", "PLAN=1")
    assert "--tasks ec,lba " in _make("-n", "ds-run", "MODEL=m", "GPU=0", "PLAN=1", "TASKS=ec,lba")


def test_moleculenet_download_target_and_help():
    text = _make("-n", "ds-download", "TASK=mn_bbbp")
    assert "python -u -m downstream.tasks.molnet.prepare download --task mn_bbbp" in text
    help_text = _make("help")
    for name in ("ds-run", "ds-status", "ds-validate", "TASKS=all|molnet", "ds-download TASK=mn_bbbp"):
        assert name in help_text, name
