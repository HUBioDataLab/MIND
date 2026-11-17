"""Synthetic downstream output trees for the results-ledger tests.

Shapes follow the files ``downstream/probe.py``, ``downstream/finetune.py``, ``downstream/embed.py`` and
``downstream/common/registry.py`` write (metrics.json, config.json with per-split embedding records,
predictions.npz, models/<name>/card.json); no real data is read.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

SHA_A = "a" * 64
SHA_B = "b" * 64
EMBED_GIT = "1111111111111111111111111111111111111111"
RUN_GIT = "2222222222222222222222222222222222222222"
PROBE_SETTINGS = {"pooling": "mean_atoms", "seeds": [0, 1, 2], "epochs": 300, "patience": 30, "batch_size": 256,
                  "lr": 0.001, "weight_decay": 0.0001, "mlp_hidden": 512, "mlp_dropout": 0.2}


def card(outputs: Path, name: str, sha: str, *, best: bool = True, epoch: int = 98) -> Path:
    run = "/runs/model_a_run"
    source = f"{run}/best-checkpoint-epoch={epoch}-val_total_loss=0.7374.ckpt" if best else f"{run}/last.ckpt"
    directory = outputs / "models" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "card.json").write_text(json.dumps({
        "name": name, "created_utc": "2026-10-31T18:00:46.927337+00:00", "source_path": source,
        "source_size": 1, "source_mtime_ns": 1, "sha256": sha, "epoch": epoch, "global_step": 633600,
        "pytorch_lightning_version": "2.5.2", "domains": ["pdb", "rna"],
        "val_metrics": {"monitor": "val_total_loss", "best_model_score": 0.7374, "current_score": 0.7374,
                        "best_model_path": f"{run}/best-checkpoint-epoch={epoch}-val_total_loss=0.7374.ckpt"},
        "config_path": None, "config_sha256": None, "snapshot_repo_commit": "bbbbbbb+dirty",
        "wandb_runs": {"x": {"git": {"commit": "aaaaaaa"}, "args": ["--seed", "42"], "program": "-m core.train_pretrain"}},
        "training_git_commits": ["aaaaaaa"], "notes": ""}))
    return directory / "card.json"


def ec_block(fmax: float, *, macro: bool = True, n: int = 4) -> dict:
    block = {"fmax": fmax, "fmax_threshold": 0.1, "fmax_precision": fmax + 0.01, "fmax_recall": fmax - 0.01,
             "pair_aupr": fmax - 0.05, "top1_accuracy": fmax + 0.02, "subset_accuracy@0.5": 0.1, "num_proteins": n}
    if macro:
        block.update(macro_aupr=fmax - 0.02, macro_aupr_num_terms=3, macro_aupr_num_terms_skipped=0)
    return block


def ec_metrics(model_id: str, seed: int, fmax: float, *, protocol="mlp_probe", pooling="mean_atoms",
               macro=True) -> dict:
    return {"task": "ec", "kind": "multilabel", "model_id": model_id, "protocol": protocol, "pooling": pooling,
            "seed": seed, "best_epoch": 10, "split_tag": None, "representation": None,
            "val": ec_block(fmax - 0.03, macro=macro),
            "test": {"all": ec_block(fmax, macro=macro), "cutoff_30": ec_block(fmax - 0.1, macro=macro, n=2),
                     "cutoff_95": ec_block(fmax, macro=macro)}}


def fold_metrics(model_id: str, seed: int, accuracies=(0.2, 0.3, 0.7)) -> dict:
    def block(acc, n):
        return {"accuracy": acc, "top5_accuracy": acc + 0.2, "num_proteins": n, "num_classes_present": 3,
                "balanced_accuracy": acc - 0.1}
    return {"task": "fold", "kind": "multiclass", "model_id": model_id, "protocol": "mlp_probe",
            "pooling": "mean_atoms", "seed": seed, "best_epoch": 3, "split_tag": None, "representation": None,
            "val": block(0.25, 6),
            "test": {"test_fold": block(accuracies[0], 5), "test_superfamily": block(accuracies[1], 5),
                     "test_family": block(accuracies[2], 5)}}


def lba_block(rmse: float, *, pearson=0.5, reason=None) -> dict:
    return {"rmse": rmse, "mae": rmse - 0.3, "pearson_r": pearson, "spearman_rho": pearson,
            "num_samples": 6, "target_mean": 6.2, "target_std": 1.7, "correlation_undefined_reason": reason}


def lba_metrics(model_id: str, seed: int, rmse: float, *, split_tag="identity30") -> dict:
    return {"task": "lba", "kind": "regression", "model_id": model_id,
            "protocol": f"mlp_probe.{split_tag}.joint", "pooling": "concat_pocket_ligand", "seed": seed,
            "best_epoch": 3, "split_tag": None, "representation": None,
            "val": lba_block(rmse + 0.1), "test": {"all": lba_block(rmse)}}


def molnet_cls_metrics(model_id: str, seed: int, roc: float) -> dict:
    def block(value):
        return {"roc_auc": value, "prc_auc": value - 0.2, "per_task": [value], "per_task_prc_auc": [value - 0.2],
                "num_tasks": 1, "num_tasks_evaluated": 1, "num_tasks_skipped": 0, "skipped_task_indices": []}
    return {"task": "mn_bace", "kind": "multilabel_masked", "model_id": model_id, "protocol": "mlp_probe",
            "pooling": "mean_atoms", "seed": seed, "best_epoch": 5, "split_tag": None, "representation": None,
            "val": block(roc - 0.05),
            "test": {"all": block(roc), "subsets": {
                "flat_conformers": {"molecules": 3, "undefined": "every task is single-class"},
                "rare_elements": {"molecules": 4, **block(roc - 0.1)}}}}


def molnet_reg_metrics(model_id: str, seed: int, rmse: float) -> dict:
    def block(value):
        return {"rmse": value, "mae": value - 0.2, "rmse_per_target": [value], "mae_per_target": [value - 0.2],
                "num_targets": 1, "num_labeled_per_target": [6]}
    return {"task": "mn_esol", "kind": "multitarget_regression", "model_id": model_id, "protocol": "mlp_probe",
            "pooling": "mean_atoms", "seed": seed, "best_epoch": 5, "split_tag": None, "representation": None,
            "val": block(rmse + 0.1), "test": {"all": block(rmse), "subsets": {}},
            "target_standardization": {"mean": [0.0], "std": [1.0], "primary_metric": "rmse"}}


FINGERPRINT = {"algorithm": "Morgan fingerprint (ECFP4)", "radius": 2, "n_bits": 2048, "variant": "bits",
               "rdkit_version": "2022.09.5", "include_chirality": False, "use_bond_types": True}


def embedding_record(task: str, model_id: str, split: str, *, sha=SHA_A, revision="rev_v2", git=EMBED_GIT,
                     random_init=False, init_seed=0, hf=False, chains=4, data_sha=None, fingerprint=None) -> dict:
    snapshot = model_id.split(".random_init")[0]
    record = {"task": task, "started_utc": "2026-11-09T10:07:01.342092+00:00", "git": git, "device": "cuda:0",
              "seed": 0, "deterministic_algorithms": True, "split": split, "model_id": model_id,
              "processed": f"/data/downstream/{task}/processed/{revision}/{split}.pt",
              "processed_sha256": data_sha or f"{revision}-{split}".ljust(64, "0")[:64],
              "chains": chains, "poolings": ["mean_atoms"], "dim": 8, "seconds": 1.0}
    if fingerprint is not None:
        record.update(model_id=model_id, fingerprint=fingerprint, device="cpu")
        for key in ("seed", "deterministic_algorithms"):
            del record[key]
    elif hf:
        record.update(model=f"facebook/{model_id}", hf_revision="c731040fcd8d73dceaa04b0a8e6329b345b0f5df")
    else:
        record.update(model_ref=snapshot, snapshot=snapshot, sha256=sha, random_init=random_init, init_seed=init_seed)
    return record


def write_probe(outputs: Path, metrics: dict, *, splits=("train", "valid", "test"), run_git=RUN_GIT,
                predictions=None, settings=None, feature_scaling=None, **record_kwargs) -> Path:
    """One probe seed directory; ``run_git=None`` imitates the probe code before the run-trace patch."""
    task, model_id = metrics["task"], metrics["model_id"]
    out = (outputs / "downstream" / task / model_id / f"{metrics['protocol']}.{metrics['pooling']}"
           / f"seed{metrics['seed']}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    config = {"args": {"task": task, "model_id": model_id, "head": metrics["protocol"].split("_")[0],
                       "pooling": None, "seeds": None,
                       **({} if feature_scaling is None else {"feature_scaling": feature_scaling})},
              "probe": copy.deepcopy(settings or PROBE_SETTINGS),
              "embeddings": {split: embedding_record(task, model_id, split, **record_kwargs) for split in splits}}
    if run_git is not None:
        config["run"] = {"git": run_git, "started_utc": "2026-11-09T12:00:00.000000+00:00",
                         "argv": ["downstream/probe.py"]}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    if predictions is not None:
        np.savez_compressed(out / "predictions.npz", **predictions)
    return out


def write_finetune(outputs: Path, metrics: dict, *, sha=SHA_A, processed=True, random_init=False) -> Path:
    task, model_id = metrics["task"], metrics["model_id"]
    out = (outputs / "downstream" / task / model_id / f"{metrics['protocol']}.{metrics['pooling']}"
           / f"seed{metrics['seed']}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    config = {"args": {"task": task, "model": model_id.split(".random_init")[0], "random_init": str(random_init),
                       "seeds": "[0, 1, 2]"},
              "finetune": {"pooling": metrics["pooling"], "head": "mlp", "seeds": [0], "epochs": 50},
              "snapshot": model_id.split(".random_init")[0], "sha256": sha, "git": RUN_GIT, "eval_seed": 0}
    if processed:
        config["processed"] = {split: {"path": f"/data/downstream/{task}/processed/rev_v2/{split}.pt",
                                       "sha256": f"rev_v2-{split}".ljust(64, "0")[:64]}
                               for split in ("train", "valid", "test")}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    return out


def write_embeddings_only(outputs: Path, task: str, model_id: str, *, sha=SHA_A) -> None:
    directory = outputs / "downstream" / task / model_id / "embeddings"
    directory.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        (directory / f"{split}.json").write_text(json.dumps(embedding_record(task, model_id, split, sha=sha)))
        np.savez(directory / f"{split}.npz", chains=np.array(["x"]))


def ec_predictions(rng: np.random.Generator, *, n=40, terms=5, quality=0.5, order=None) -> dict:
    """Scores from ``rng``; labels are fixed (one test set shared by every seed and model)."""
    labels = np.zeros((n, terms), np.float32)
    labels[np.arange(n), np.random.default_rng(12345).integers(0, terms, n)] = 1
    scores = np.clip(quality * labels + (1 - quality) * rng.random((n, terms)), 0, 1).astype(np.float32)
    chains = np.array([f"C{i}" for i in range(n)])
    if order is not None:
        labels, scores, chains = labels[order], scores[order], chains[order]
    return {"test_chains": chains, "test_scores": scores, "test_labels": labels, "valid_scores": scores[:2]}
