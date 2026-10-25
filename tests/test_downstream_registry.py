"""Model snapshots, aliases and the downstream results layout."""

import json
import os

import pytest
import torch

from downstream.common import registry


def _fake_checkpoint(run_dir):
    run_dir.mkdir(parents=True)
    path = run_dir / "best.ckpt"
    torch.save({
        "epoch": 3, "global_step": 120, "pytorch-lightning_version": "2.5.2", "state_dict": {},
        "hyper_parameters": {"config": {"multi_domain": {"datasets": {
            "pdb": {"enabled": True}, "rna": {"enabled": True}, "unimol": {"enabled": False}}}}},
        "callbacks": {"ModelCheckpoint{'monitor': 'val_total_loss'}": {
            "monitor": "val_total_loss", "best_model_score": torch.tensor(0.75),
            "current_score": torch.tensor(0.8), "best_model_path": str(path)}},
    }, path)
    meta = run_dir / "wandb" / "run-1" / "files"
    meta.mkdir(parents=True)
    (meta / "wandb-metadata.json").write_text(json.dumps(
        {"git": {"commit": "abc123"}, "args": ["--config-yaml-path", "x.yaml"], "program": "-m core.train_pretrain"}))
    return path


def test_snapshot_copies_hashes_and_never_touches_the_source(tmp_path):
    src = _fake_checkpoint(tmp_path / "run")
    before = (src.stat().st_size, src.stat().st_mtime_ns, registry.sha256_file(src))
    dest = registry.snapshot_model(src, "md_e3_20261019", root=tmp_path / "out")
    assert (src.stat().st_size, src.stat().st_mtime_ns, registry.sha256_file(src)) == before
    card = json.loads((dest / "card.json").read_text())
    assert card["sha256"] == before[2] == registry.sha256_file(dest / "model.ckpt")
    assert card["epoch"] == 3 and card["global_step"] == 120
    assert card["domains"] == ["pdb", "rna"]
    assert card["val_metrics"]["best_model_score"] == pytest.approx(0.75)
    assert card["training_git_commits"] == ["abc123"]
    assert card["source_path"] == str(src.resolve())
    assert not os.access(dest / "model.ckpt", os.W_OK) or os.geteuid() == 0
    assert (dest / "hparams.json").is_file()
    with pytest.raises(FileExistsError):
        registry.snapshot_model(src, "md_e3_20261019", root=tmp_path / "out")


def test_alias_resolution_and_hash_verification(tmp_path):
    root = tmp_path / "out"
    src = _fake_checkpoint(tmp_path / "run")
    registry.snapshot_model(src, "snap_a", root=root)
    registry.snapshot_model(src, "snap_b", root=root)
    registry.set_alias("current_best", "snap_a", root=root)
    resolved = registry.resolve_model("current_best", root=root)
    assert resolved.model_id == "snap_a" and resolved.requested == "current_best"
    with pytest.raises(FileExistsError, match="repoint"):
        registry.set_alias("current_best", "snap_b", root=root)
    registry.set_alias("current_best", "snap_b", root=root, repoint=True)
    assert registry.resolve_model("current_best", root=root).model_id == "snap_b"
    with pytest.raises(FileExistsError, match="real directory"):
        registry.set_alias("snap_a", "snap_b", root=root, repoint=True)
    ckpt = root / "models" / "snap_a" / "model.ckpt"
    os.chmod(ckpt, 0o644)
    ckpt.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="sha256"):
        registry.resolve_model("snap_a", root=root)
    with pytest.raises(FileNotFoundError):
        registry.resolve_model("missing", root=root)


def test_results_layout_and_refusals(tmp_path):
    path = registry.results_dir("ec", "snap_a", "linear_probe", 2, pooling="mean_atoms", root=tmp_path)
    assert path == tmp_path / "downstream" / "ec" / "snap_a" / "linear_probe.mean_atoms" / "seed2"
    created = registry.new_results_dir("ec", "snap_a", "linear_probe", 2, pooling="mean_atoms", root=tmp_path)
    (created / "metrics.json").write_text("{}")
    with pytest.raises(FileExistsError):
        registry.new_results_dir("ec", "snap_a", "linear_probe", 2, pooling="mean_atoms", root=tmp_path)
    assert registry.embeddings_path("ec", "snap_a.random_init", "test", root=tmp_path).name == "test.npz"
    with pytest.raises(ValueError, match="protocol"):
        registry.results_dir("ec", "snap_a", "zero_shot", 0, pooling="mean_atoms", root=tmp_path)
    with pytest.raises(ValueError, match="invalid model id"):
        registry.results_dir("ec", "../escape", "finetune", 0, pooling="mean_atoms", root=tmp_path)
    assert registry.pretrain_run_dir("multidomain", "run1", root=tmp_path) == tmp_path / "pretrain" / "multidomain" / "run1"
    with pytest.raises(ValueError, match="pretrain group"):
        registry.pretrain_run_dir("protein", "run1", root=tmp_path)


def test_registry_cli(tmp_path, capsys):
    src = _fake_checkpoint(tmp_path / "run")
    root = str(tmp_path / "out")
    assert registry.main(["--outputs-root", root, "snapshot", str(src), "snap_cli"]) == 0
    assert registry.main(["--outputs-root", root, "alias", "current_best", "snap_cli"]) == 0
    assert registry.resolve_model("current_best", root=tmp_path / "out").model_id == "snap_cli"
    assert '"sha256"' in capsys.readouterr().out
