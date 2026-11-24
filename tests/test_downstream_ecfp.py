"""ECFP4 fingerprint baseline: featurizer, head training on a toy set, ledger ingestion and reports.

Synthetic molecules and synthetic processed MoleculeNet files in temporary directories; the ledger of a test is
always a temporary results root. No real data, no GPU.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from downstream import ledger, ledger_report, probe
from downstream.baselines import ecfp
from downstream.common.registry import embeddings_path, results_dir
from downstream.tasks import MOLNET_TASKS, SMILES_TASKS
from downstream.tasks.molnet import common
from downstream.baselines import ecfp_tuned
from ledger_helpers import (
    FINGERPRINT, PROBE_SETTINGS, SHA_A, card, molnet_cls_metrics, molnet_reg_metrics, write_probe,
)
from test_downstream_molnet import make_molecule, write_family

REPO = Path(__file__).resolve().parents[1]
SMILES = ["CCO", "CCN", "CC(=O)O", "c1ccccc1", "CCCCCl", "CC(C)N", "OCCO", "c1ccncc1"]


def _pool(count=96):
    """Deterministic alkyl chains with one optional branch: some contain nitrogen, some do not."""
    branches = ["", "N", "O", "Cl", "C(=O)O", "c1ccccc1", "c1ccncc1", "S"]
    pool = []
    for i in range(count):
        branch = branches[(i // 6) % len(branches)]
        pool.append("C" * (2 + i % 6) + (f"({branch})C" if branch else "") + "C" * (i % 3))
    return pool


# -- featurizer ---------------------------------------------------------------------------------------------

def test_bits_are_deterministic_and_equal_the_rdkit_morgan_bit_vector():
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    spec = ecfp.FingerprintSpec()
    first = ecfp.fingerprint_matrix(SMILES, SMILES, spec)
    np.testing.assert_array_equal(first, ecfp.fingerprint_matrix(SMILES, SMILES, spec))
    assert first.shape == (len(SMILES), 2048) and first.dtype == np.uint8 and set(np.unique(first)) <= {0, 1}
    for row, smiles in zip(first, SMILES):  # independent oracle: the classic API used for the hand-made baseline
        reference = np.zeros(2048, np.uint8)
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=2048),
                                        reference)
        np.testing.assert_array_equal(row, reference)
    assert len({tuple(np.flatnonzero(row)) for row in first}) == len(SMILES)  # different molecules differ


def test_fingerprint_does_not_depend_on_the_smiles_spelling():
    a, b = ecfp.fingerprint_matrix(["OCC", "C1=CC=CC=C1"], ["a", "b"], ecfp.FingerprintSpec())
    c, d = ecfp.fingerprint_matrix(["CCO", "c1ccccc1"], ["a", "b"], ecfp.FingerprintSpec())
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)


def test_counts_extend_the_bits_and_the_size_is_a_parameter():
    bits = ecfp.fingerprint_matrix(SMILES, SMILES, ecfp.FingerprintSpec("bits", 1024))
    counts = ecfp.fingerprint_matrix(SMILES, SMILES, ecfp.FingerprintSpec("counts", 1024))
    assert bits.shape == counts.shape == (len(SMILES), 1024) and counts.dtype == np.uint16
    np.testing.assert_array_equal(counts > 0, bits > 0)
    assert counts.max() > 1  # at least one repeated environment (the six ring atoms of benzene)


def test_pooling_name_and_description_identify_the_feature():
    assert ecfp.FingerprintSpec().pooling == "ecfp4_bits2048"
    assert ecfp.FingerprintSpec("counts", 1024).pooling == "ecfp4_counts1024"
    description = ecfp.FingerprintSpec().describe()
    assert description["radius"] == 2 and description["n_bits"] == 2048 and description["variant"] == "bits"
    import rdkit
    assert description["rdkit_version"] == rdkit.__version__
    for bad in (("floats", 2048), ("bits", 4), ("bits", 2048.0)):
        with pytest.raises(ValueError):
            ecfp.FingerprintSpec(*bad)


def test_an_unparsable_smiles_raises_with_its_id_instead_of_becoming_a_zero_row():
    with pytest.raises(ValueError, match=r"cannot parse 1 of 3.*mol_1"):
        ecfp.fingerprint_matrix(["CCO", "not a smiles", "CCN"], ["mol_0", "mol_1", "mol_2"], ecfp.FingerprintSpec())
    with pytest.raises(ValueError, match="3 SMILES but 2 ids"):
        ecfp.fingerprint_matrix(["C", "C", "C"], ["a", "b"], ecfp.FingerprintSpec())


def test_only_smiles_tasks_are_accepted_and_all_expands_to_moleculenet_plus_npass():
    assert ecfp.expand_tasks("molnet") == list(MOLNET_TASKS) and len(MOLNET_TASKS) == 14
    assert ecfp.expand_tasks("all") == list(SMILES_TASKS) and len(SMILES_TASKS) == 22
    assert ecfp.expand_tasks("mn_bbbp,mn_esol,mn_bbbp") == ["mn_bbbp", "mn_esol"]
    with pytest.raises(ValueError, match="MoleculeNet and NPASS tasks only"):
        ecfp.expand_tasks("mn_bbbp,ec")


# -- heads on a toy set (the production training function, fingerprint features) ------------------------------

SETTINGS = {"epochs": 80, "patience": 20, "batch_size": 32, "lr": 0.01, "weight_decay": 1e-4, "mlp_hidden": 32,
            "mlp_dropout": 0.0}


def _toy(kind):
    smiles = _pool()
    x = torch.from_numpy(ecfp.fingerprint_matrix(smiles, smiles, ecfp.FingerprintSpec()).astype(np.float32))
    from rdkit import Chem

    molecules = [Chem.MolFromSmiles(s) for s in smiles]
    if kind == "multilabel_masked":
        nitrogen = Chem.MolFromSmarts("[#7]")
        y = torch.tensor([[1.0 if m.HasSubstructMatch(nitrogen) else 0.0] for m in molecules])
    else:
        y = torch.tensor([[float(m.GetNumHeavyAtoms())] for m in molecules])
    return x, y


@pytest.mark.parametrize("head", ["linear", "mlp"])
def test_heads_learn_a_substructure_label_from_the_fingerprint(head):
    x, y = _toy("multilabel_masked")
    assert 0 < float(y.mean()) < 1
    train, valid = slice(0, 64), slice(64, 96)
    model, best = probe.train_probe(x[train], y[train], x[valid], y[valid], head_kind=head, settings=SETTINGS, seed=0,
                                    log=lambda message: None, kind="multilabel_masked")
    assert best["score"] > 0.9  # validation ROC-AUC
    again, best_again = probe.train_probe(x[train], y[train], x[valid], y[valid], head_kind=head, settings=SETTINGS,
                                          seed=0, log=lambda message: None, kind="multilabel_masked")
    assert best_again["score"] == best["score"] and best_again["epoch"] == best["epoch"]  # seeded: reproducible


def test_regression_head_beats_the_train_mean_on_heavy_atom_counts():
    x, y = _toy("multitarget_regression")
    train, valid = slice(0, 64), slice(64, 96)
    scale = probe.target_scale(y[train].numpy())
    model, best = probe.train_probe(x[train], y[train], x[valid], y[valid], head_kind="linear", settings=SETTINGS,
                                    seed=0, log=lambda message: None, kind="multitarget_regression", scale=scale,
                                    metric="rmse")
    mean_rmse = float(np.sqrt(((y[valid].numpy() - y[train].numpy().mean()) ** 2).mean()))
    assert -best["score"] < 0.8 * mean_rmse


# -- end to end on synthetic processed MoleculeNet files ------------------------------------------------------

@pytest.fixture
def esol_data(tmp_path, monkeypatch):
    """Processed ``mn_esol`` splits with real SMILES under a temporary data root (``MIND_DATA_ROOT``)."""
    data = tmp_path / "data"
    monkeypatch.setenv("MIND_DATA_ROOT", str(data))
    atoms = [["C", "C", "O"], ["C", "C", "N"], ["C", "C", "O", "O"], ["C"] * 6, ["Cl", "C", "C", "C", "C"],
             ["C", "C", "C", "N"], ["C", "C", "O", "O"], ["C"] * 5 + ["N"]]
    records = [make_molecule(atoms[i], (0.7 * i - 2.0,), ori_index=i, smi=SMILES[i]) for i in range(8)]
    config = write_family(data / "downstream", train=records[:5], valid=records[5:7], test=records[7:])
    assert common.build_processed(config, splits=["train", "valid", "test"], config_path=common.TASK_YAML) == 0
    return data


def _run(tmp_path, *extra):
    return ecfp.main(["--tasks", "mn_esol", "--heads", "linear", "--seeds", "0", "1", "--threads", "1", "--models", "probe",
                      "--outputs-root", str(tmp_path / "outputs"), "--results-root", str(tmp_path / "results"),
                      "--log-root", str(tmp_path / "logs"), *extra])


def test_runner_writes_embeddings_probes_and_ledger_rows_and_skips_what_is_done(tmp_path, esol_data, capsys):
    outputs, results = tmp_path / "outputs", tmp_path / "results"
    assert _run(tmp_path, "--plan") == 0
    assert not outputs.exists() and not results.exists()  # the plan writes nothing
    results.mkdir()
    (results / "comparisons.yaml").write_text("summary_models: [ecfp4]\n")  # the report models, as in a real results root
    assert _run(tmp_path) == 0
    assert (results / "reports" / "tasks" / "mn_esol.md").is_file() and (results / "reports" / "summary.md").is_file()

    record = json.loads(embeddings_path("mn_esol", "ecfp4", "train", root=outputs).with_suffix(".json").read_text())
    assert record["model_id"] == "ecfp4" and record["device"] == "cpu" and record["poolings"] == ["ecfp4_bits2048"]
    assert record["fingerprint"]["radius"] == 2 and record["fingerprint"]["n_bits"] == 2048
    assert record["fingerprint"]["rdkit_version"] and record["processed_sha256"] and record["chains"] == 5
    with np.load(embeddings_path("mn_esol", "ecfp4", "train", root=outputs)) as data:
        assert data["emb_ecfp4_bits2048"].shape == (5, 2048) and data["labels"].shape == (5, 1)
        assert data["chains"].tolist() == [f"esol_{i}" for i in range(5)]

    config = json.loads((results_dir("mn_esol", "ecfp4", "linear_probe", 0, pooling="ecfp4_bits2048", root=outputs)
                         / "config.json").read_text())
    assert config["args"]["feature_scaling"] == "none" and config["args"]["pooling"] == "ecfp4_bits2048"

    rows = ledger.read_records(results, "measurements")
    assert {r["model_key"] for r in rows} == {"ecfp4"} and {r["model_kind"] for r in rows} == {"fingerprint_baseline"}
    assert {r["run_seed"] for r in rows} == {0, 1} and {r["protocol_id"] for r in rows} == {"linear_probe"}
    assert {r["pooling"] for r in rows} == {"ecfp4_bits2048"} and {r["data_revision"] for r in rows} == {"v1"}
    assert {r["model_sha256"] for r in rows} == {None} and {r["hf_revision"] for r in rows} == {None}
    primary, = [r for r in rows if r["is_primary"] and r["run_seed"] == 0]
    assert primary["metric"] == "rmse" and primary["eval_split"] == "test" and primary["value"] > 0
    assert primary["input_feature"]["rdkit_version"] and primary["input_feature"]["feature_scaling"] == "none"
    assert primary["missing_provenance"] == [] and primary["code_git_run"] and primary["expected_seeds"] == [0, 1]
    assert {r["metric"] for r in rows if r["subset"] == "all" and r["eval_split"] == "test"} == {"rmse", "mae"}
    assert ledger.verify(results)["lines"]["measurements"] == len(rows)

    capsys.readouterr()
    assert _run(tmp_path) == 0  # second run: nothing left to do, nothing added
    assert "all probes done" in capsys.readouterr().out
    assert len(ledger.read_records(results, "measurements")) == len(rows)
    last = ledger.read_records(results, "ingests")[-1]
    assert last["rows_added"] == 0 and last["files_skipped_same_sha"] == 2


def test_a_rerun_with_another_fingerprint_definition_is_refused_and_the_probe_guards_the_scaling(tmp_path, esol_data):
    assert _run(tmp_path, "--no-ledger") == 0
    assert _run(tmp_path, "--n-bits", "1024", "--no-ledger") == 2  # same outputs root: ecfp4 embeddings exist as 2048
    with pytest.raises(SystemExit):  # fingerprint features must not be standardized with train statistics
        probe.main(["--task", "mn_esol", "--model-id", "ecfp4", "--head", "linear", "--pooling", "ecfp4_bits2048",
                    "--seeds", "5", "--outputs-root", str(tmp_path / "outputs")])
    assert not (tmp_path / "results").exists()


def test_a_failing_task_is_reported_and_the_exit_code_is_nonzero(tmp_path, esol_data, capsys):
    split = esol_data / "downstream" / "unimol_molnet" / "mn_esol" / "processed" / "v1" / "train.pt"
    payload = torch.load(split, weights_only=False)
    payload["smiles"][0] = "not a smiles"
    torch.save(payload, split)
    assert _run(tmp_path, "--no-ledger") == 1
    out = capsys.readouterr().out
    assert "ERROR fingerprints mn_esol" in out and "esol_0" in out and "not run: the fingerprints of this task failed" in out
    assert not embeddings_path("mn_esol", "ecfp4", "valid", root=tmp_path / "outputs").exists()


# -- ledger kinds, catalog and reports ------------------------------------------------------------------------

def _fingerprint_probe(outputs, metrics, **kwargs):
    metrics = dict(metrics, pooling="ecfp4_bits2048")
    return write_probe(outputs, metrics, fingerprint=FINGERPRINT, feature_scaling="none", **kwargs)


def _tree(tmp_path):
    outputs = tmp_path / "outputs"
    card(outputs, "model_a", SHA_A)
    for seed in (0, 1, 2):
        write_probe(outputs, molnet_reg_metrics("model_a", seed, 1.0 + 0.01 * seed))
        _fingerprint_probe(outputs, molnet_reg_metrics("ecfp4", seed, 1.8 + 0.01 * seed))
    tuned = dict(molnet_reg_metrics("ecfp4_tuned", 0, 1.7), protocol="sklearn_tuned")  # the tuned fit has one seed
    _fingerprint_probe(outputs, tuned, settings=dict(PROBE_SETTINGS, estimator="Ridge", seeds=[0]))
    return outputs


def test_fingerprint_rows_have_their_own_kind_without_a_checkpoint(tmp_path):
    root, outputs = tmp_path / "results", _tree(tmp_path)
    ledger.ingest(root, outputs_root=outputs, command="test")
    tuned = [r for r in ledger.read_records(root, "measurements") if r["model_key"] == "ecfp4_tuned"]
    assert {r["model_kind"] for r in tuned} == {"fingerprint_baseline"} and {r["run_seed"] for r in tuned} == {0}
    assert {r["protocol_id"] for r in tuned} == {"sklearn_tuned"} and {r["head"] for r in tuned} == {"sklearn_tuned"}
    assert {tuple(r["expected_seeds"]) for r in tuned} == {(0,)}
    rows = [r for r in ledger.read_records(root, "measurements") if r["model_key"] == "ecfp4"]
    assert rows and {r["model_kind"] for r in rows} == {"fingerprint_baseline"}
    assert {r["model_sha256"] for r in rows} == {None} and {r["selection_rule"] for r in rows} == {None}
    assert {r["input_feature"]["n_bits"] for r in rows} == {2048}
    assert {r["pooling"] for r in rows} == {"ecfp4_bits2048"}
    mind = [r for r in ledger.read_records(root, "measurements") if r["model_key"] == "model_a"]
    assert {r["input_feature"] for r in mind} == {None}
    assert {m["model_key"] for m in ledger.read_records(root, "models")} == {"model_a"}  # no card for the baseline
    assert {r["protocol_settings_sha256"] for r in rows} == {r["protocol_settings_sha256"] for r in mind}


def test_mixing_feature_scaling_and_feature_kind_is_refused_at_ingest(tmp_path):
    outputs = tmp_path / "outputs"
    write_probe(outputs, dict(molnet_reg_metrics("ecfp4", 0, 1.8), pooling="ecfp4_bits2048"), fingerprint=FINGERPRINT)
    with pytest.raises(ValueError, match="must run with feature_scaling 'none'"):
        ledger.ingest(tmp_path / "r1", outputs_root=outputs, command="test")
    other = tmp_path / "other"
    write_probe(other, molnet_reg_metrics("model_a", 0, 1.0), feature_scaling="none")
    with pytest.raises(ValueError, match="only valid for a fingerprint baseline"):
        ledger.ingest(tmp_path / "r2", outputs_root=other, command="test")


def test_the_baseline_cannot_take_the_name_of_a_checkpoint(tmp_path):
    outputs = tmp_path / "outputs"
    card(outputs, "ecfp4", SHA_A)
    write_probe(outputs, molnet_reg_metrics("ecfp4", 0, 1.0))
    _fingerprint_probe(outputs, molnet_reg_metrics("ecfp4", 1, 1.8))
    with pytest.raises(ledger.LedgerError, match="one model name must name one checkpoint"):
        ledger.ingest(tmp_path / "results", outputs_root=outputs, command="test")


def _state(tmp_path, outputs):
    root = tmp_path / "results"
    ledger.ingest(root, outputs_root=outputs, command="test")
    return root, ledger_report.load_state(root)


def test_baseline_sits_in_the_encoder_setting_and_is_footnoted(tmp_path):
    root, state = _state(tmp_path, _tree(tmp_path))
    sections = [s for s in ledger_report.sections(state, "mn_esol") if s["protocol_id"] == "mlp_probe"]
    assert len(sections) == 1 and sections[0]["models"] == ["ecfp4", "model_a"]  # one comparable setting
    own, = [s for s in ledger_report.sections(state, "mn_esol") if s["protocol_id"] == "sklearn_tuned"]
    assert own["models"] == ["ecfp4_tuned"] and own["pooling"] == "mean_atoms"  # its own protocol, its own setting
    assert sections[0]["pooling"] == "mean_atoms"
    assert {r["input_pooling"] for r in sections[0]["rows"] if r["model_key"] == "ecfp4"} == {"ecfp4_bits2048"}
    markdown, csv_rows = ledger_report.task_report(state, "mn_esol")
    assert "ecfp4 §" in markdown and "§ ecfp4: input is a fingerprint" in markdown and "RDKit" in markdown
    assert "feature scaling none" in markdown
    assert {r["model_key"] for r in csv_rows} == {"ecfp4", "model_a", "ecfp4_tuned"}
    stored = [r for r in ledger.read_records(root, "measurements") if r["model_key"] == "ecfp4"]
    assert {r["pooling"] for r in stored} == {"ecfp4_bits2048"}  # the ledger itself keeps the real pooling


def test_baseline_is_not_missing_on_other_families_and_names_its_command_on_molnet(tmp_path):
    _, state = _state(tmp_path, _tree(tmp_path))
    models = ["model_a", "ecfp4", "ecfp4_tuned"]
    missing = ledger_report.missing_cells(state, models)
    assert not [m for m in missing if m["model_key"].startswith("ecfp4") and not m["task"].startswith(("mn_", "npass_"))]
    for model, variant, protocol in (("ecfp4", "probe", "mlp_probe"), ("ecfp4_tuned", "tuned", "sklearn_tuned")):
        found = [m for m in missing if m["model_key"] == model]
        assert found and {m["task"] for m in found} <= set(SMILES_TASKS) - {"mn_esol"}
        assert {m["protocol_id"] for m in found} == {protocol}  # the tuned baseline is required in its own protocol
        assert all(m["hint"] == f"make ds-ecfp TASKS={m['task']} VARIANT={variant}" for m in found)
    assert [m for m in missing if m["model_key"] == "model_a" and m["task"] == "ec"]  # MIND models still need EC
    summary = ledger_report.summary_page(state, models)
    ec_line, = [line for line in summary.splitlines() if line.startswith("| ec ")]
    assert "n/a (no baseline for this family)" in ec_line
    esol_line, = [line for line in summary.splitlines() if line.startswith("| mn_esol ")]
    assert "1.8" in esol_line and "1.7" in esol_line and "n/a" not in esol_line
    assert "`ecfp4_tuned` is read with its own protocol `sklearn_tuned`" in summary
    applicable = [t for t in ledger_report.summary_tasks(state["catalog"])
                  if ledger_report.applicable(state["catalog"], t, "ecfp4")]
    assert applicable == list(SMILES_TASKS)


def test_catalog_checks_the_fingerprint_baseline_entries(tmp_path):
    values = yaml.safe_load(ledger.CATALOG_PATH.read_text())
    assert values["fingerprint_baselines"]["ecfp4"]["families"] == ["molecule_property"]
    values["fingerprint_baselines"]["ecfp4"]["families"] = ["small_molecules"]
    bad = tmp_path / "catalog.yaml"
    bad.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="fingerprint_baselines.ecfp4"):
        ledger.load_catalog(bad)
    values["fingerprint_baselines"]["ecfp4"] = {"families": ["molecule_property"]}
    bad.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="table_pooling"):
        ledger.load_catalog(bad)


def test_a_fingerprint_model_the_catalog_does_not_list_is_refused_by_the_report(tmp_path):
    outputs = tmp_path / "outputs"
    _fingerprint_probe(outputs, molnet_reg_metrics("ecfp6", 0, 1.8))
    root = tmp_path / "results"
    ledger.ingest(root, outputs_root=outputs, command="test")
    with pytest.raises(ValueError, match="fingerprint_baselines.ecfp6"):
        ledger_report.load_state(root)


# -- the tuned scikit-learn baseline -------------------------------------------------------------------------

def test_grids_and_settings_follow_the_validation_tuned_recipe():
    assert ecfp_tuned.grid_for("multilabel_masked", 29999) == (0.01, 0.1, 1.0)
    assert ecfp_tuned.grid_for("multilabel_masked", 30000) == (0.05, 0.5)
    assert ecfp_tuned.grid_for("multitarget_regression", 10) == (0.1, 1.0, 10.0, 100.0, 1000.0)
    with pytest.raises(ValueError, match="MoleculeNet kinds only"):
        ecfp_tuned.grid_for("regression", 10)
    settings = ecfp_tuned.settings_for("multilabel_masked", (0.1,), "roc_auc")
    assert settings["estimator"] == "LogisticRegression" and settings["train_subsampling"] is None
    assert settings["seeds"] == [0] and settings["feature_scaling"] == "none" and settings["class_weight"] is None


def _toy_sparse(kind):
    from scipy import sparse

    x, y = _toy(kind)
    return sparse.csr_matrix(x.numpy()), y.numpy()


def test_candidate_fits_equal_scikit_learn_fitted_by_hand_and_a_single_class_target_raises():
    from sklearn.linear_model import LogisticRegression, Ridge

    x, y = _toy_sparse("multilabel_masked")
    (scores,) = ecfp_tuned.fit_candidate("multilabel_masked", 0.1, x[:64], y[:64], [x[64:]], threads=1)
    reference = LogisticRegression(C=0.1, solver="liblinear", max_iter=2000).fit(x[:64], y[:64, 0]).predict_proba(x[64:])[:, 1]
    np.testing.assert_allclose(scores[:, 0], reference, atol=1e-6)
    assert scores.dtype == np.float32 and scores.min() >= 0 and scores.max() <= 1
    with pytest.raises(ValueError, match="single class"):
        ecfp_tuned.fit_candidate("multilabel_masked", 0.1, x[:64], np.zeros((64, 1), np.float32), [x[64:]], threads=1)
    x, y = _toy_sparse("multitarget_regression")
    (scores,) = ecfp_tuned.fit_candidate("multitarget_regression", 10.0, x[:64], y[:64], [x[64:]], threads=1)
    np.testing.assert_allclose(scores[:, 0], Ridge(alpha=10.0).fit(x[:64], y[:64, 0].astype(np.float64)).predict(x[64:]),
                               atol=1e-4)


def test_missing_labels_are_left_out_of_each_targets_fit():
    x, y = _toy_sparse("multilabel_masked")
    y = np.concatenate([y, 1 - y], axis=1)
    y[:10, 1] = np.nan  # second target: ten unlabelled train rows
    (scores,) = ecfp_tuned.fit_candidate("multilabel_masked", 1.0, x[:64], y[:64], [x[64:]], threads=1)
    (alone,) = ecfp_tuned.fit_candidate("multilabel_masked", 1.0, x[10:64], y[10:64, 1:], [x[64:]], threads=1)
    np.testing.assert_allclose(scores[:, 1], alone[:, 0], atol=1e-6)


def _tuned(tmp_path, *extra, task="mn_esol"):
    return ecfp.main(["--tasks", task, "--models", "tuned", "--threads", "1", "--outputs-root", str(tmp_path / "outputs"),
                      "--results-root", str(tmp_path / "results"), "--log-root", str(tmp_path / "logs"), *extra])


def test_tuned_run_selects_on_validation_scores_test_once_and_enters_the_ledger_with_one_seed(tmp_path, esol_data):
    outputs, results = tmp_path / "outputs", tmp_path / "results"
    results.mkdir()
    (results / "comparisons.yaml").write_text("summary_models: [ecfp4_tuned]\n")
    assert _tuned(tmp_path) == 0
    out = results_dir("mn_esol", "ecfp4_tuned", "sklearn_tuned", 0, pooling="ecfp4_bits2048", root=outputs)
    assert sorted(p.name for p in out.parent.iterdir()) == ["seed0"]  # a deterministic fit: exactly one seed
    assert {p.name for p in out.iterdir()} == {"metrics.json", "predictions.npz", "config.json", "run.log"}
    metrics = json.loads((out / "metrics.json").read_text())
    probe_keys = {"task", "kind", "model_id", "protocol", "pooling", "seed", "best_epoch", "split_tag", "representation",
                  "val", "test"}
    assert set(metrics) == probe_keys and metrics["best_epoch"] is None and metrics["protocol"] == "sklearn_tuned"
    assert set(metrics["test"]) == {"all", "subsets"} and set(metrics["test"]["all"]) >= {"rmse", "mae"}
    config = json.loads((out / "config.json").read_text())
    selected = config["selected"]["alpha"]
    assert selected in ecfp_tuned.REGRESSION_GRID and config["args"]["seeds"] == [0]
    assert config["args"]["feature_scaling"] == "none" and config["probe"]["train_subsampling"] is None
    assert config["libraries"]["scikit-learn"] and {r["model_id"] for r in config["embeddings"].values()} == {"ecfp4_tuned"}
    scores = [float(line.split()[-1]) for line in config["selected"]["validation_scores"]]
    assert ecfp_tuned.REGRESSION_GRID[int(np.argmin(scores))] == selected  # lowest validation RMSE, first of ties
    with np.load(out / "predictions.npz") as saved:
        assert set(saved.files) == {"test_chains", "test_scores", "test_labels", "valid_scores"}
        recomputed = float(np.sqrt(np.mean((saved["test_scores"] - saved["test_labels"]) ** 2)))
    assert recomputed == pytest.approx(metrics["test"]["all"]["rmse"], rel=1e-6)

    rows = [r for r in ledger.read_records(results, "measurements") if r["model_key"] == "ecfp4_tuned"]
    assert {r["run_seed"] for r in rows} == {0} and {r["expected_seeds"][0] for r in rows} == {0}
    assert {len(r["expected_seeds"]) for r in rows} == {1}
    assert {r["protocol_id"] for r in rows} == {"sklearn_tuned"} and {r["model_kind"] for r in rows} == {"fingerprint_baseline"}
    assert {r["pooling"] for r in rows} == {"ecfp4_bits2048"} and all(r["missing_provenance"] == [] for r in rows)
    cell = ledger_report.lookup(ledger_report.load_state(results), "mn_esol", "ecfp4_tuned", "sklearn_tuned",
                                {"eval_split": "test", "subset": "all", "metric": "rmse"})
    assert cell["category"] is None and "n_seeds=1" in cell["text"] and "std undefined" in cell["text"]
    assert _tuned(tmp_path) == 0  # nothing left to do: no second run, no new rows
    assert len(ledger.read_records(results, "measurements")) == len(rows)


def test_the_test_labels_never_enter_the_selection(tmp_path, esol_data):
    assert _tuned(tmp_path, "--no-ledger") == 0
    outputs = tmp_path / "outputs"
    first = ecfp_tuned.run_task("mn_esol", ecfp.FingerprintSpec(), outputs, threads=1, write=False)
    path = embeddings_path("mn_esol", "ecfp4", "test", root=outputs)
    with np.load(path) as data:
        arrays = dict(data)
    arrays["labels"] = arrays["labels"] + 100.0
    np.savez_compressed(path, **arrays)
    second = ecfp_tuned.run_task("mn_esol", ecfp.FingerprintSpec(), outputs, threads=1, write=False)
    assert second["selected"] == first["selected"] and second["val"] == first["val"]
    assert second["test"]["all"]["rmse"] > first["test"]["all"]["rmse"] + 50  # only the test score moved


@pytest.fixture
def bbbp_data(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("MIND_DATA_ROOT", str(data))
    smiles = ["CCO", "CCN", "CC(=O)O", "c1ccccc1", "CCCCCl", "CC(C)N", "OCCO", "c1ccncc1", "CCCN", "CCC", "CCOC", "NCCN",
              "CC(=O)N", "CCCC", "CCCCO", "CNC"]
    atoms = [["C", "C", "O"], ["C", "C", "N"], ["C", "C", "O", "O"], ["C"] * 6, ["Cl", "C", "C", "C", "C"],
             ["C", "C", "C", "N"], ["C", "O", "O"], ["C"] * 5 + ["N"]]
    label = [0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
    records = [make_molecule(atoms[i % 8], (label[i],), ori_index=i, smi=smiles[i]) for i in range(16)]
    config = write_family(data / "downstream", "bbbp", train=records[:8], valid=records[8:12], test=records[12:])
    assert common.build_processed(config, splits=["train", "valid", "test"], config_path=common.TASK_YAML) == 0
    return data


def test_tuned_classification_picks_c_on_validation_roc_auc(tmp_path, bbbp_data):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "comparisons.yaml").write_text("summary_models: [ecfp4_tuned]\n")
    assert _tuned(tmp_path, task="mn_bbbp") == 0
    out = results_dir("mn_bbbp", "ecfp4_tuned", "sklearn_tuned", 0, pooling="ecfp4_bits2048", root=tmp_path / "outputs")
    config = json.loads((out / "config.json").read_text())
    scores = [float(line.split()[-1]) for line in config["selected"]["validation_scores"]]
    grid = config["selected"]["grid"]
    assert grid == [0.01, 0.1, 1.0] and config["selected"]["C"] == grid[int(np.argmax(scores))]
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["kind"] == "multilabel_masked" and 0 <= metrics["test"]["all"]["roc_auc"] <= 1
    assert metrics["val"]["roc_auc"] == pytest.approx(max(scores), abs=1e-6)
    with np.load(out / "predictions.npz") as saved:
        assert saved["test_scores"].min() >= 0 and saved["test_scores"].max() <= 1
    rows = ledger.read_records(tmp_path / "results", "measurements")
    primary, = [r for r in rows if r["is_primary"]]
    assert primary["metric"] == "roc_auc" and primary["model_key"] == "ecfp4_tuned" and primary["run_seed"] == 0


# -- Make target ----------------------------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
def test_make_target_runs_detached_on_cpu_and_passes_the_options():
    env = {k: v for k, v in os.environ.items() if k != "MIND_RESULTS_ROOT"}
    run = lambda *args: " ".join(subprocess.run(  # noqa: E731
        ["make", "--no-print-directory", "-n", "ds-ecfp", *args], cwd=REPO, env=env, capture_output=True,
        text=True, check=True).stdout.replace("\\\n", " ").split())
    detached = run("TASKS=mn_bbbp,mn_esol", "HEADS=mlp", "SEEDS=0 1", "PROBE_JOBS=2", "OUTPUTS_ROOT=/o", "MIND_RESULTS_ROOT=/r",
                   "VARIANT=tuned")
    assert "setsid nohup" in detached and "EXIT_CODE=$code" in detached and 'CUDA_VISIBLE_DEVICES=""' in detached
    assert ("python -u -m downstream.baselines.ecfp --tasks mn_bbbp,mn_esol --heads mlp --seeds 0 1 "
            "--models tuned --jobs 2 --threads 8") in detached
    assert '--outputs-root "/o"' in detached and 'MIND_RESULTS_ROOT="/r"' in detached and "--plan" not in detached
    everything = run("TASKS=molnet", "MIND_RESULTS_ROOT=/r", "NO_LEDGER=1")
    assert "--tasks " + ",".join(MOLNET_TASKS) in everything and "--no-ledger" in everything
    planned = run("PLAN=1", "MIND_RESULTS_ROOT=/r")
    assert "setsid" not in planned and "--tasks all" in planned and planned.endswith("--plan")
    assert "--models both" in planned  # both baselines by default
