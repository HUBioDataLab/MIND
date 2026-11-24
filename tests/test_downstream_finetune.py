"""``downstream.finetune`` on CPU: every task kind, the probe's metric code path, random init, ledger ingest.

The MIND encoder needs CUDA (flash_varlen attention), so its forward pass is replaced by a small trainable stand-in
(element embedding + coordinate projection) that reads the same batch fields; everything else is the production
code: runtime graph building, batching by atoms, pooling, the head, the loss, validation-based epoch selection,
``downstream.probe.evaluate_task`` and the results files. MoleculeNet runs on synthetic Uni-Mol LMDB data processed by
the real pipeline; the protein tasks use a synthetic processed-split stand-in (chains made of small graphs, a chain
may have two windows) because building real protein samples needs structure files.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from core.virtual_slot_graph import build_geometry_neutral_graph_view
from downstream import finetune, ledger, probe
from downstream.common import complex_graph as cg
from downstream.common import encoder as mind_encoder
from downstream.tasks.molnet import common as molnet
from ledger_helpers import card
from test_downstream_molnet import make_molecule, tiny_model_config, write_family

SETTINGS = {"pooling": "mean_atoms", "head": "mlp", "seeds": [0], "epochs": 3, "patience": None,
            "max_atoms_per_batch": 40, "max_samples_per_batch": None, "lr_encoder": 1e-3,
            "lr_encoder_random_init": 5e-3, "lr_head": 1e-2, "weight_decay": 0.0, "grad_clip_norm": 0.5,
            "mlp_hidden": 16, "mlp_dropout": 0.0}
ELEMENTS = ("C", "N", "O", "S", "Cl")


class StandInEncoder(nn.Module):
    """Trainable stand-in for ``PretrainingESAModel``: node states from the element id and the coordinates."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.element = nn.Embedding(130, 8)
        self.coordinates = nn.Linear(3, 8)


def stand_in_encode(model, batch):
    """Same contract as ``downstream.common.encoder.encode``: MLM fields set, ``active_node_mask`` from the real
    geometry-neutral graph view (as ``PretrainingESAModel.forward`` sets it), pooling mask of unmasked real atoms."""
    batch = mind_encoder.prepare_encoder_batch(batch)
    batch.active_node_mask = build_geometry_neutral_graph_view(batch, model.config).active_node_mask
    types = batch.original_types.long()
    assert int(types.max()) < 130
    nodes = torch.tanh(model.element(types) + model.coordinates(batch.pos.float()))
    return nodes, batch.active_node_mask & ~batch.mlm_mask


@pytest.fixture
def encoder_calls(monkeypatch):
    calls = []
    config = tiny_model_config()

    def load(ckpt_path, device, *, random_init=False, init_seed=0, domains=("protein",)):
        calls.append({"ckpt": ckpt_path, "random_init": random_init, "init_seed": init_seed})
        torch.manual_seed(1000 + init_seed)
        return StandInEncoder(config).to(device)

    monkeypatch.setattr(mind_encoder, "load_mind_encoder", load)
    monkeypatch.setattr(mind_encoder, "encode", stand_in_encode)
    return calls


@pytest.fixture
def outputs(tmp_path):
    root = tmp_path / "outputs"
    directory = root / "models" / "snap"
    directory.mkdir(parents=True)
    (directory / "model.ckpt").write_bytes(b"stand-in checkpoint")
    card(root, "snap", hashlib.sha256(b"stand-in checkpoint").hexdigest())
    return root


def finetune_args(task, outputs, *, data_root=None, random_init=False, seeds=(0,), **overrides):
    argv = ["--task", task, "--model", "snap", "--device", "cpu", "--outputs-root", str(outputs),
            "--seeds", *map(str, seeds)] + (["--data-root", str(data_root)] if data_root else [])
    argv += ["--random-init"] if random_init else []
    for name, value in overrides.items():
        argv += [f"--{name.replace('_', '-')}", str(value)]
    parser = finetune.build_parser()
    args = parser.parse_args(argv)
    finetune.check_args(parser, args)
    return args


# ---------------------------------------------------------------------------------------------------------
# Synthetic protein-task stand-in (EC / GO / Fold / Reaction / LBA layouts)
# ---------------------------------------------------------------------------------------------------------

class StandInSplit:
    """Processed-split stand-in with the attributes fine-tuning reads; samples are small graphs (windows)."""

    def __init__(self, path: Path, *, representation: str = "joint"):
        payload = torch.load(path, weights_only=False)
        self.path, self.representation = Path(path), representation
        self.chains, self.labels = payload["chains"], payload["labels"]
        self.sample_chain = torch.tensor(payload["sample_chain"])
        self.sample_residues = torch.tensor(payload["sample_residues"])
        self._graphs = payload["graphs"]

    def __len__(self):
        return len(self._graphs)

    def runtime(self, index, model_config, *, seed):
        elements, positions = self._graphs[index]
        return cg.ligand_runtime_sample(cg.ligand_from_rows(elements, positions), f"{self.path.stem}_{index}",
                                        model_config, seed=seed)

    def iter_runtime(self, model_config, *, seed):
        for index in range(len(self)):
            yield index, self.runtime(index, model_config, seed=seed)

    def chain_embeddings(self, sample_embeddings):
        weights = self.sample_residues.numpy().astype(np.float64)
        out = np.zeros((len(self.chains), sample_embeddings.shape[1]))
        totals = np.zeros(len(self.chains))
        np.add.at(out, self.sample_chain.numpy(), sample_embeddings * weights[:, None])
        np.add.at(totals, self.sample_chain.numpy(), weights)
        return (out / totals[:, None]).astype(np.float32)


def _labels(kind, rng, count, width):
    if kind == "multiclass":
        return np.arange(count) % width  # every class in every split
    if kind == "regression":
        return rng.normal(6.0, 1.5, (count, 1)).astype(np.float32)
    labels = np.zeros((count, width), np.float32)
    labels[np.arange(count), np.arange(count) % width] = 1
    labels[::3, (np.arange(count)[::3] + 1) % width] = 1  # some chains carry two terms
    return labels


def _write_split(path, kind, rng, *, count, width):
    graphs, sample_chain, residues = [], [], []
    for chain in range(count):
        for window in range(2 if chain % 4 == 0 else 1):  # every fourth chain is cropped into two windows
            atoms = int(rng.integers(3, 7))
            graphs.append(([ELEMENTS[(chain + k) % len(ELEMENTS)] for k in range(atoms)],
                           rng.normal(0.0, 1.5, (atoms, 3)).astype(np.float32)))
            sample_chain.append(chain)
            residues.append(int(rng.integers(5, 20)))
    torch.save({"chains": [f"{path.stem}_c{i}" for i in range(count)], "labels": _labels(kind, rng, count, width),
                "graphs": graphs, "sample_chain": sample_chain, "sample_residues": residues}, path)


def stand_in_task(monkeypatch, tmp_path, task, kind, *, width=4, extra=None):
    """A task module like ``downstream.tasks.load_task`` returns, over synthetic processed splits."""
    rng = np.random.default_rng(7)
    root = tmp_path / "data" / task
    root.mkdir(parents=True)
    if task == "lba":
        splits = ["train", "val", "test"]
    elif kind == "multiclass":
        splits = ["train", "valid", *(["test_fold", "test_family", "test_superfamily"] if task == "fold" else ["test"])]
    else:
        splits = ["train", "valid", "test"]
    for split in splits:
        _write_split(root / f"{split}.pt", kind, rng, count=12 if split == "train" else 8, width=width)
    config = {"task": task, "splits": splits, "embedding": {"seed": 0, "poolings": ["mean_atoms", "mean_residues"]},
              "finetune": dict(SETTINGS), **(extra or {})}
    if kind == "multiclass":
        config.update(train_split="train", valid_split="valid", test_splits=splits[2:], label_count=width)
    if task == "lba":
        config.update(primary_split_tag="identity30", representation="joint")
    test_chains = torch.load(root / f"{splits[2]}.pt", weights_only=False)["chains"]
    subsets = {30: np.arange(len(test_chains)) % 2 == 0, 95: np.ones(len(test_chains), bool)}
    module = SimpleNamespace(
        KIND=kind, TASK_YAML=root / "task.yaml",
        load_task_config=lambda path, *, data_root=None: copy.deepcopy(config),
        processed_dir=lambda cfg, *tag: root, ProcessedSplit=StandInSplit,
        test_subsets=lambda cfg, chains: {k: v.copy() for k, v in subsets.items()})
    monkeypatch.setattr(finetune, "load_task", lambda name: module)
    return config, root, subsets


def _probe_metrics_from_predictions(out, kind, config, *, train_labels, val_labels, test_labels, subsets):
    """What ``downstream.probe.evaluate_task`` reports for the saved predictions of a fine-tune run."""
    with np.load(out / "predictions.npz") as data:
        names = list(test_labels)
        scores = {names[0]: data["test_scores"], **{n: data[f"{n}_scores"] for n in names[1:]}}
        assert np.array_equal(data["test_labels"], test_labels[names[0]], equal_nan=True)
        return probe.evaluate_task(kind, config, train_labels=train_labels, val_labels=val_labels,
                                   val_scores=data["valid_scores"], test_labels=test_labels, test_scores=scores,
                                   subsets=subsets)


def _split_labels(root, names):
    return {name: np.asarray(torch.load(root / f"{name}.pt", weights_only=False)["labels"]) for name in names}


@pytest.mark.parametrize("task, kind, extra", [
    ("ec", "multilabel", {}),
    ("go_mf", "multilabel", {"ic_binned_aupr": {"ic_reference": "train_frequency", "ic_thresholds": [0.5, 1.0]}}),
    ("fold", "multiclass", {}),
    ("reaction", "multiclass", {}),
    ("lba", "regression", {}),
])
def test_protein_task_kinds_use_the_probe_metrics_and_reach_the_ledger(monkeypatch, tmp_path, outputs, encoder_calls,
                                                                        task, kind, extra):
    config, root, subsets = stand_in_task(monkeypatch, tmp_path, task, kind, extra=extra)
    out, = finetune.run(finetune_args(task, outputs, **({"split_tag": "identity30"} if task == "lba" else {})))
    protocol = "finetune.identity30.joint" if task == "lba" else "finetune"
    assert out == outputs / "downstream" / task / "snap" / f"{protocol}.mean_atoms" / "seed0"
    metrics = json.loads((out / "metrics.json").read_text())
    assert (metrics["kind"], metrics["protocol"], metrics["model_id"]) == (kind, protocol, "snap")
    names = config["splits"]
    labels = _split_labels(root, names)
    tests = {name: labels[name] for name in names[2:]}
    used_subsets = subsets if kind == "multilabel" else {}
    val, test, _ = _probe_metrics_from_predictions(out, kind, config, train_labels=labels["train"],
                                                   val_labels=labels[names[1]], test_labels=tests,
                                                   subsets=used_subsets)
    assert metrics["test"] == json.loads(json.dumps(test)) and metrics["val"] == json.loads(json.dumps(val))
    if kind == "multilabel":
        assert sorted(metrics["test"]) == ["all", "cutoff_30", "cutoff_95"]
    if task == "go_mf":
        assert "ic_binned_macro_aupr_num_terms" in metrics["test"]["all"]  # GO's IC-binned AUPR, as in the probe
    if kind == "multiclass":
        assert list(metrics["test"]) == names[2:] and "weighted_f1" in metrics["test"][names[2]]
    log = (out / "run.log").read_text()
    assert sum(line.startswith("epoch ") for line in log.splitlines()) == SETTINGS["epochs"]  # patience: null
    saved = json.loads((out / "config.json").read_text())
    assert saved["finetune"]["encoder_init"] == "pretrained" and saved["finetune"]["lr_encoder"] == 1e-3
    assert sorted(saved["processed"]) == sorted(names)
    rows, model = ledger.flatten_run(out / "metrics.json", ledger.load_catalog(), outputs)
    assert rows and {r["protocol_family"] for r in rows} == {"finetune"} and model["model_key"] == "snap"
    assert {r["protocol_id"] for r in rows} == {protocol} and {r["model_kind"] for r in rows} == {"mind"}
    spec = ledger.load_catalog()["tasks"][task]
    present = {(r["eval_split"], r["subset"], r["metric"]) for r in rows}
    required = {(c["eval_split"], c["subset"], c["metric"]) for c in spec["required"]
                if c["subset"] in ("all", "cutoff_30", "cutoff_95")}  # the stand-in has the 30 and 95 % cutoffs only
    assert required <= present, required - present


def test_random_init_arm_trains_its_own_model_id_with_its_own_encoder_lr(monkeypatch, tmp_path, outputs, encoder_calls):
    stand_in_task(monkeypatch, tmp_path, "ec", "multilabel")
    outs = finetune.run(finetune_args("ec", outputs, random_init=True, seeds=(1, 2)))
    assert [o.parent.parent.name for o in outs] == ["snap.random_init"] * 2
    assert [(c["random_init"], c["init_seed"]) for c in encoder_calls] == [(True, 1), (True, 2)]
    saved = json.loads((outs[0] / "config.json").read_text())
    assert saved["finetune"]["encoder_init"] == "random" and saved["finetune"]["lr_encoder"] == 5e-3
    assert saved["args"]["random_init"] == "True"
    rows, model = ledger.flatten_run(outs[1] / "metrics.json", ledger.load_catalog(), outputs)
    assert {r["model_kind"] for r in rows} == {"mind_random_init"} and {r["init_seed"] for r in rows} == {2}
    assert model is None
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        finetune.run(finetune_args("ec", outputs, random_init=True, seeds=(1,)))


def test_cli_overrides_win_and_are_recorded(monkeypatch, tmp_path, outputs, encoder_calls):
    stand_in_task(monkeypatch, tmp_path, "fold", "multiclass")
    out, = finetune.run(finetune_args("fold", outputs, random_init=True, epochs=2, lr_encoder=0.02, patience=5))
    saved = json.loads((out / "config.json").read_text())["finetune"]
    assert (saved["epochs"], saved["lr_encoder"], saved["patience"]) == (2, 0.02, 5)
    assert sum(line.startswith("epoch ") for line in (out / "run.log").read_text().splitlines()) == 2


def test_hyperparameter_overrides_win_and_are_recorded(monkeypatch, tmp_path, outputs, encoder_calls):
    stand_in_task(monkeypatch, tmp_path, "fold", "multiclass")
    out, = finetune.run(finetune_args("fold", outputs, epochs=1, lr_head=0.02, weight_decay=0.05, mlp_dropout=0.3,
                                      pooling="mean_residues"))
    saved = json.loads((out / "config.json").read_text())["finetune"]
    assert (saved["lr_head"], saved["weight_decay"], saved["mlp_dropout"], saved["pooling"]) == (
        0.02, 0.05, 0.3, "mean_residues")
    assert saved["lr_encoder"] == SETTINGS["lr_encoder"]  # untouched settings keep the task.yaml value
    assert out.parent.name.endswith("mean_residues")  # the pooling is part of the results directory name
    assert json.loads((out / "metrics.json").read_text())["pooling"] == "mean_residues"


def test_override_values_are_validated():
    base = {"lr_encoder": None, "epochs": None, "patience": None}
    for name, value in (("lr_head", 0), ("lr_head", -1e-3), ("weight_decay", -0.1), ("mlp_dropout", 1.0),
                        ("mlp_dropout", -0.1), ("pooling", "max")):
        with pytest.raises(ValueError, match=name):
            finetune.resolve_settings(SETTINGS, random_init=False, overrides={**base, name: value})
    ok = finetune.resolve_settings(SETTINGS, random_init=False,
                                   overrides={"weight_decay": 0.0, "mlp_dropout": 0.0, "lr_head": 1e-4})
    assert (ok["weight_decay"], ok["mlp_dropout"], ok["lr_head"]) == (0.0, 0.0, 1e-4)
    with pytest.raises(ValueError, match="not available for this task"):
        finetune.resolve_settings(SETTINGS, random_init=False, overrides={"pooling": "ca"},
                                  allowed_poolings=["mean_atoms"])
    with pytest.raises(ValueError, match="not one of"):
        finetune.resolve_settings(SETTINGS, random_init=False, overrides={"warmup": 3})


def test_override_cli_refusals(capsys):
    parser = finetune.build_parser()
    for argv, text in ((["--lr-head", "0"], "--lr-head must be positive"),
                       (["--weight-decay", "-1"], "--weight-decay must be >= 0"),
                       (["--mlp-dropout", "1"], "--mlp-dropout must be in [0, 1)")):
        args = parser.parse_args(["--task", "ec", "--model", "x", *argv])
        with pytest.raises(SystemExit):
            finetune.check_args(parser, args)
        assert text in capsys.readouterr().err
    with pytest.raises(SystemExit):  # an unknown pooling is refused by the parser itself
        parser.parse_args(["--task", "ec", "--model", "x", "--pooling", "max"])
    assert "--pooling" in capsys.readouterr().err


def test_ds_run_treats_overrides_as_other_settings():
    from downstream.finetune_settings import same_settings

    plain = finetune.resolve_settings(SETTINGS, random_init=False)
    for name, value in (("lr_head", 3e-4), ("weight_decay", 1e-2), ("mlp_dropout", 0.5), ("pooling", "mean_residues")):
        changed = finetune.resolve_settings(SETTINGS, random_init=False, overrides={name: value})
        assert not same_settings(changed, plain)
    assert same_settings(plain, finetune.resolve_settings(SETTINGS, random_init=False, overrides={"lr_head": None}))


# ---------------------------------------------------------------------------------------------------------
# MoleculeNet: synthetic Uni-Mol data through the real processing and the real processed-split loader
# ---------------------------------------------------------------------------------------------------------

ATOMS = (["C", "C", "O", "H"], ["N", "C", "C", "C"], ["Cl", "C", "C", "H"], ["C", "C", "S"], ["C", "O", "H"],
         ["C", "C", "N"], ["C", "O", "O"], ["C", "C", "C", "O"])


def _molnet_data(tmp_path, dataset, targets):
    def records(ids):
        return [make_molecule(ATOMS[i % len(ATOMS)], targets(i), ori_index=i) for i in ids]
    config = write_family(tmp_path, dataset, train=records(range(0, 10)), valid=records(range(10, 14)),
                          test=records(range(14, 18)))
    assert molnet.build_processed(config, splits=["train", "valid", "test"], config_path=molnet.TASK_YAML) == 0
    return Path(config["data_root"])


@pytest.mark.parametrize("task, dataset, targets", [
    # two targets, missing labels (-1 = NaN) in both; each split keeps both classes for target 0
    ("mn_clintox", "clintox", lambda i: (float(i % 2), -1.0 if i % 3 == 0 else float((i // 2) % 2))),
    ("mn_esol", "esol", lambda i: (-0.4 * i - 1.0,)),
])
def test_molnet_kinds_fine_tune_on_processed_data(tmp_path, outputs, encoder_calls, task, dataset, targets):
    data_root = _molnet_data(tmp_path, dataset, targets)
    out, = finetune.run(finetune_args(task, outputs, data_root=data_root, epochs=2))
    metrics = json.loads((out / "metrics.json").read_text())
    config = molnet.load_task_config(data_root=data_root, dataset=dataset)
    splits = {s: molnet.MolnetProcessedSplit(molnet.processed_dir(config) / f"{s}.pt") for s in config["splits"]}
    kind = config["kind"]
    subsets = molnet.test_subsets(config, splits["test"].chains)
    val, test, _ = _probe_metrics_from_predictions(out, kind, config, train_labels=splits["train"].labels,
                                                   val_labels=splits["valid"].labels,
                                                   test_labels={"test": splits["test"].labels}, subsets=subsets)
    assert metrics["test"] == json.loads(json.dumps(test)) and metrics["val"] == json.loads(json.dumps(val))
    assert sorted(metrics["test"]) == ["all", "subsets"]
    if kind == "multilabel_masked":
        assert np.isnan(splits["train"].labels).any()  # the masked loss saw missing labels
        assert {"roc_auc", "prc_auc"} <= set(metrics["test"]["all"])
    else:
        mean, std = probe.target_scale(splits["train"].labels)
        assert metrics["target_standardization"] == {"mean": mean.tolist(), "std": std.tolist(), "primary_metric": "rmse"}
        assert {"rmse", "mae"} <= set(metrics["test"]["all"])
    rows, _ = ledger.flatten_run(out / "metrics.json", ledger.load_catalog(), outputs)
    primary = [r for r in rows if r["is_primary"]]
    assert len(primary) == 1 and primary[0]["protocol_id"] == "finetune" and primary[0]["value"] is not None


# ---------------------------------------------------------------------------------------------------------
# Settings and CLI
# ---------------------------------------------------------------------------------------------------------

def test_settings_resolution_and_refusals():
    pretrained = finetune.resolve_settings(SETTINGS, random_init=False, overrides={"epochs": None})
    random = finetune.resolve_settings(SETTINGS, random_init=True, overrides={"epochs": 7})
    assert (pretrained["lr_encoder"], pretrained["encoder_init"]) == (1e-3, "pretrained")
    assert (random["lr_encoder"], random["epochs"], random["encoder_init"]) == (5e-3, 7, "random")
    with pytest.raises(ValueError, match="lacks"):
        finetune.resolve_settings({k: v for k, v in SETTINGS.items() if k != "lr_encoder_random_init"},
                                  random_init=False, overrides={})
    with pytest.raises(ValueError, match="patience must be null or positive"):
        finetune.resolve_settings({**SETTINGS, "patience": 0}, random_init=False, overrides={})
    with pytest.raises(ValueError, match="finetune.head"):
        finetune.resolve_settings({**SETTINGS, "head": "cnn"}, random_init=False, overrides={})
    for name in ("max_samples_per_batch", "grad_clip_norm"):
        with pytest.raises(ValueError, match=f"{name} must be null or positive"):
            finetune.resolve_settings({**SETTINGS, name: 0}, random_init=False, overrides={})


@pytest.mark.parametrize("task", ["ec", "go_bp", "fold", "reaction", "lba", "mn_tox21", "mn_qm7"])
def test_every_task_config_has_a_complete_finetune_block(task):
    from downstream.tasks import load_task

    module = load_task(task)
    block = module.load_task_config(module.TASK_YAML)["finetune"]
    settings = finetune.resolve_settings(block, random_init=False, overrides={})
    assert settings["lr_encoder"] < settings["lr_head"] and settings["lr_encoder_random_init"] < settings["lr_head"]
    assert settings["patience"] is None and settings["seeds"] == [0, 1, 2]


def test_main_needs_cuda_and_one_device(capsys):
    for task in ("fold", "mn_esol", "mn_tox21"):  # every kind parses; only the device is refused
        with pytest.raises(SystemExit) as exc:
            finetune.main(["--task", task, "--model", "x", "--device", "cpu"])
        assert exc.value.code == 2 and "needs CUDA" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        finetune.main(["--task", "ec", "--model", "x", "--split-tag", "identity30"])
    with pytest.raises(SystemExit):
        finetune.main(["--task", "ec", "--model", "x", "--seeds", "0", "0"])


# ---------------------------------------------------------------------------------------------------------
# Epoch selection, optimizer, head standardization, target back-transformation, batching, clipping
# ---------------------------------------------------------------------------------------------------------

def _initial_stand_in(seed=0):
    """The encoder the ``encoder_calls`` loader returns for ``init_seed=seed`` (same seeded construction)."""
    torch.manual_seed(1000 + seed)
    return StandInEncoder(tiny_model_config())


def _independent_features(model, split):
    """Mean-pooled node states per molecule, one graph at a time (no batching, no production pooling code)."""
    rows = []
    with torch.no_grad():
        for index in range(len(split)):
            graph = mind_encoder.collate([split.runtime(index, model.config, seed=0)])
            nodes, mask = stand_in_encode(model, graph)
            rows.append(nodes[mask].mean(dim=0).numpy())
    return np.stack(rows).astype(np.float64)


def test_best_epoch_is_restored_and_outputs_match_an_independent_computation(tmp_path, outputs, encoder_calls,
                                                                             monkeypatch):
    data_root = _molnet_data(tmp_path, "esol", lambda i: (-0.4 * i - 1.0,))
    scores = iter([0.1, 0.9, 0.2, 0.3])  # validation (higher is better): epoch 1 is the best, not the last
    monkeypatch.setattr(finetune, "validation_score", lambda *a, **k: next(scores))
    states, real_train_epoch = [], finetune.train_epoch

    def recording_train_epoch(model, *args, **kwargs):
        loss = real_train_epoch(model, *args, **kwargs)
        states.append({k: v.clone() for k, v in model.state_dict().items()})
        return loss
    monkeypatch.setattr(finetune, "train_epoch", recording_train_epoch)
    out, = finetune.run(finetune_args("mn_esol", outputs, data_root=data_root, epochs=4))
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["best_epoch"] == 1 and "best epoch is the last" not in (out / "run.log").read_text()
    encoder = torch.load(out / "encoder.pt")
    initial = _initial_stand_in().state_dict()
    assert all(torch.equal(encoder[k], states[1][k]) for k in encoder)  # the best epoch's weights, not the last
    assert not torch.equal(states[1]["element.weight"], states[3]["element.weight"])
    assert not torch.equal(encoder["element.weight"], initial["element.weight"])  # the encoder was trained

    config = molnet.load_task_config(data_root=data_root, dataset="esol")
    split = {s: molnet.MolnetProcessedSplit(molnet.processed_dir(config) / f"{s}.pt") for s in ("train", "test")}
    saved = torch.load(out / "head.pt", weights_only=False)
    train_features = _independent_features(_initial_stand_in(), split["train"])  # statistics of the initial encoder
    np.testing.assert_allclose(saved["feature_mean"], train_features.mean(axis=0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(saved["feature_std"], np.maximum(train_features.std(axis=0), 1e-6), rtol=1e-4, atol=1e-6)
    labels = split["train"].labels[:, 0].astype(np.float64)
    assert metrics["target_standardization"]["mean"][0] == pytest.approx(labels.mean(), rel=1e-6)
    assert metrics["target_standardization"]["std"][0] == pytest.approx(labels.std(), rel=1e-6)

    restored = StandInEncoder(tiny_model_config())
    restored.load_state_dict(encoder)
    hidden = config["finetune"]["mlp_hidden"]  # the real mn_esol settings (mlp head)
    head = nn.Sequential(nn.Linear(8, hidden), nn.ReLU(), nn.Dropout(0.0), nn.Linear(hidden, 1))
    head.load_state_dict({k[len("head."):]: v for k, v in saved["state_dict"].items() if k.startswith("head.")})
    head.eval()
    features = _independent_features(restored, split["test"])
    standardized = (features - saved["feature_mean"]) / saved["feature_std"]
    with torch.no_grad():
        raw = head(torch.from_numpy(standardized.astype(np.float32))).numpy()
    expected = raw * labels.std() + labels.mean()  # back to the original units, numpy only
    with np.load(out / "predictions.npz") as predictions:
        np.testing.assert_allclose(predictions["test_scores"], expected, rtol=1e-4, atol=1e-4)


def test_optimizer_covers_every_encoder_and_head_parameter_with_its_own_lr():
    model, head = _initial_stand_in(), nn.Linear(8, 2)
    optimizer = finetune.build_optimizer(model, head, SETTINGS)
    encoder_group, head_group = optimizer.param_groups
    assert {id(p) for p in encoder_group["params"]} == {id(p) for p in model.parameters()}
    assert {id(p) for p in head_group["params"]} == {id(p) for p in head.parameters()}
    assert (encoder_group["lr"], head_group["lr"]) == (SETTINGS["lr_encoder"], SETTINGS["lr_head"])
    standardized = finetune.StandardizedHead(head, np.zeros(8), np.ones(8))
    assert len(list(standardized.parameters())) == 2 and {n for n, _ in standardized.named_buffers()} == {
        "feature_mean", "feature_std"}  # the statistics are buffers, never optimized


def test_random_init_builds_the_architecture_without_the_checkpoint_weights(tmp_path):
    from core.pretraining_model import PretrainingESAModel

    config = dict(vars(tiny_model_config()))
    trained = PretrainingESAModel(config=dict(config))
    state = {k: torch.full_like(v, 0.125) for k, v in trained.state_dict().items()}
    path = tmp_path / "tiny.ckpt"
    torch.save({"hyper_parameters": {"config": config}, "state_dict": state}, path)
    first = mind_encoder.load_mind_encoder(str(path), "cpu", random_init=True, init_seed=0).state_dict()
    again = mind_encoder.load_mind_encoder(str(path), "cpu", random_init=True, init_seed=0).state_dict()
    other = mind_encoder.load_mind_encoder(str(path), "cpu", random_init=True, init_seed=1).state_dict()
    weights = [k for k, v in first.items() if v.is_floating_point() and v.numel() > 1 and k in state]
    assert weights and not any(torch.all(first[k] == 0.125) for k in weights)  # nothing comes from the checkpoint
    assert all(torch.equal(first[k], again[k]) for k in first)  # seeded
    assert any(not torch.equal(first[k], other[k]) for k in weights)


def _esol_split(tmp_path):
    data_root = _molnet_data(tmp_path, "esol", lambda i: (-0.4 * i - 1.0,))
    config = molnet.load_task_config(data_root=data_root, dataset="esol")
    return molnet.MolnetProcessedSplit(molnet.processed_dir(config) / "train.pt")


@pytest.mark.parametrize("cap, max_atoms, expected", [(4, 15000, [4, 4, 2]), (None, 15000, [10]), (None, 7, None)])
def test_batches_respect_the_molecule_cap_and_the_node_budget(tmp_path, monkeypatch, encoder_calls, cap, max_atoms,
                                                              expected):
    split = _esol_split(tmp_path)
    model, head = _initial_stand_in(), nn.Linear(8, 1)
    sizes, real_collate = [], mind_encoder.collate

    def recording_collate(samples):
        sizes.append([s.num_nodes for s in samples])
        return real_collate(samples)
    monkeypatch.setattr(mind_encoder, "collate", recording_collate)
    settings = {**SETTINGS, "max_samples_per_batch": cap, "max_atoms_per_batch": max_atoms}
    finetune.train_epoch(model, head, split, finetune.build_optimizer(model, head, settings), "cpu", epoch_seed=0,
                         settings=settings, loss_fn=probe.build_loss("multitarget_regression", settings),
                         kind="multitarget_regression", targets=torch.zeros(len(split), 1))
    assert sum(len(batch) for batch in sizes) == len(split)
    if expected is not None:
        assert [len(batch) for batch in sizes] == expected
    else:  # node budget: a batch exceeds it only when one molecule alone does
        assert all(sum(batch) <= max_atoms or len(batch) == 1 for batch in sizes) and len(sizes) > 1


@pytest.mark.parametrize("clip", [0.5, None])
def test_gradients_are_clipped_to_the_configured_norm(tmp_path, monkeypatch, encoder_calls, clip):
    split = _esol_split(tmp_path)
    model, head = _initial_stand_in(), nn.Linear(8, 1)
    calls, real_clip = [], torch.nn.utils.clip_grad_norm_

    def recording_clip(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        real_clip(parameters, max_norm, *args, **kwargs)
        calls.append((len(parameters), max_norm, float(torch.norm(torch.stack([p.grad.norm() for p in parameters
                                                                                  if p.grad is not None])))))
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)
    settings = {**SETTINGS, "grad_clip_norm": clip, "max_samples_per_batch": 4, "lr_encoder": 1.0, "lr_head": 1.0}
    finetune.train_epoch(model, head, split, finetune.build_optimizer(model, head, settings), "cpu", epoch_seed=0,
                         settings=settings, loss_fn=probe.build_loss("multitarget_regression", settings),
                         kind="multitarget_regression", targets=torch.full((len(split), 1), 100.0))
    if clip is None:
        assert not calls
    else:
        expected_parameters = len(list(model.parameters())) + len(list(head.parameters()))
        assert len(calls) == 3 and all(n == expected_parameters and limit == 0.5 for n, limit, _ in calls)
        assert all(norm <= 0.5 + 1e-5 for _, _, norm in calls)


def test_molnet_finetune_overrides_and_costs_come_from_task_yaml(tmp_path):
    from downstream.tasks import load_task

    def block(task):
        module = load_task(task)
        return module.load_task_config(module.TASK_YAML)
    assert block("mn_pcba")["finetune"]["epochs"] == 20 and block("mn_hiv")["finetune"]["epochs"] == 50
    assert block("mn_muv")["finetune"]["epochs"] == 50 and block("mn_esol")["finetune"]["max_samples_per_batch"] == 32
    assert block("mn_pcba")["finetune_seconds_per_epoch"] > block("mn_esol")["finetune_seconds_per_epoch"] > 0
    import yaml
    values = yaml.safe_load(molnet.TASK_YAML.read_text())
    values["finetune_overrides"] = {"pcba": {"epoch": 20}}
    bad = tmp_path / "task.yaml"
    bad.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="unknown finetune keys"):
        molnet.with_finetune_entries(molnet.load_task_config(bad, dataset="pcba"), bad)
