"""Results ledger core: flattening, append-only storage, integrity and identity checks.

Synthetic output trees only (``ledger_helpers``); each test uses its own temporary results root.
"""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

from downstream import ledger
from ledger_helpers import (
    EMBED_GIT, RUN_GIT, SHA_A, SHA_B, card, ec_metrics, fold_metrics, lba_block, lba_metrics,
    molnet_cls_metrics, molnet_reg_metrics, write_embeddings_only, write_finetune, write_probe,
)


def _rows(root, **where):
    rows = ledger.read_records(root, "measurements")
    return [r for r in rows if all(r.get(k) == v for k, v in where.items())]


def _tree(tmp_path):
    outputs = tmp_path / "outputs"
    card(outputs, "model_a", SHA_A)
    for seed in (0, 1, 2):
        write_probe(outputs, ec_metrics("model_a", seed, 0.50 + 0.01 * seed))
        write_probe(outputs, fold_metrics("model_a", seed), splits=("train", "valid", "test_fold", "test_superfamily",
                                                                   "test_family"))
        write_probe(outputs, lba_metrics("model_a", seed, 1.5), splits=("train", "val", "test"))
        write_probe(outputs, molnet_cls_metrics("model_a", seed, 0.8))
        write_probe(outputs, molnet_reg_metrics("model_a", seed, 1.0))
    return outputs


# -- T9: the results root never defaults silently ---------------------------------------------------------

def test_results_root_requires_the_environment_variable(monkeypatch, tmp_path):
    monkeypatch.delenv(ledger.ENV_VAR, raising=False)
    with pytest.raises(ledger.LedgerError, match=ledger.ENV_VAR):
        ledger.results_root()
    monkeypatch.setenv(ledger.ENV_VAR, str(tmp_path / "r"))
    assert ledger.results_root() == tmp_path / "r"
    assert ledger.results_root(tmp_path / "x") == tmp_path / "x"


# -- T1: every kind flattens into one row per (seed, eval split, subset, metric) -----------------------------

def test_ingest_flattens_every_kind_with_provenance(tmp_path):
    root, outputs = tmp_path / "results", _tree(tmp_path)
    summary = ledger.ingest(root, outputs_root=outputs, command="test")
    assert summary["files_seen"] == 15 and summary["files_skipped_same_sha"] == 0
    row, = _rows(root, task="ec", eval_split="test", subset="cutoff_30", metric="fmax", run_seed=1)
    assert row["value"] == pytest.approx(0.41)
    assert row["model_key"] == "model_a" and row["model_sha256"] == SHA_A and row["model_kind"] == "mind"
    assert row["protocol_family"] == "frozen_probe" and row["head"] == "mlp" and row["protocol_id"] == "mlp_probe"
    assert row["pooling"] == "mean_atoms" and row["data_revision"] == "rev_v2"
    assert row["eval_processed_sha256"].startswith("rev_v2-test")
    assert row["n_samples"] == 2 and row["direction"] == "higher" and row["unit"] == "fraction"
    assert row["code_git_embed"] == [EMBED_GIT] and row["code_git_run"] == RUN_GIT
    assert row["train_epoch"] == 98 and "best-checkpoint" in row["selection_rule"]
    assert row["schema_version"] == ledger.SCHEMA_VERSION and row["expected_seeds"] == [0, 1, 2]
    primary, = _rows(root, task="ec", eval_split="test", subset="all", metric="fmax", run_seed=0)
    assert primary["is_primary"] and primary["is_catalog_metric"]
    # LBA: 4 catalog metrics on val and test per seed; split tag lives in the protocol id
    assert len(_rows(root, task="lba", run_seed=0)) == 8
    assert {r["protocol_id"] for r in _rows(root, task="lba")} == {"mlp_probe.identity30.joint"}
    assert _rows(root, task="lba", metric="rmse", eval_split="test", run_seed=0)[0]["unit"] == "pK"
    # Fold: the Avg. over the three test sets is derived per seed and says so
    avg, = _rows(root, task="fold", eval_split="test_avg", metric="accuracy", run_seed=2)
    assert avg["value"] == pytest.approx(0.4) and avg["derived_from"].startswith("mean of")
    # MoleculeNet: undefined subsets keep their reason, list fields never become metrics
    flat = _rows(root, task="mn_bace", subset="flat_conformers", run_seed=0)
    assert {r["metric"] for r in flat} == {"roc_auc", "prc_auc"}
    assert all(r["value"] is None and r["undefined_reason"] == "every task is single-class" for r in flat)
    assert not _rows(root, metric="per_task")
    assert len(_rows(root, task="mn_esol", run_seed=0)) == 4


def test_finetune_rows_record_their_protocol_and_missing_data_revision(tmp_path):
    root, outputs = tmp_path / "results", tmp_path / "outputs"
    card(outputs, "model_a", SHA_A)
    write_finetune(outputs, ec_metrics("model_a", 0, 0.6, protocol="finetune"))
    write_finetune(outputs, ec_metrics("model_a.random_init", 1, 0.4, protocol="finetune"), processed=False,
                   random_init=True)
    ledger.ingest(root, outputs_root=outputs, command="test")
    tuned, = _rows(root, model_key="model_a", metric="fmax", eval_split="test", subset="all")
    assert tuned["protocol_family"] == "finetune" and tuned["data_revision"] == "rev_v2"
    assert tuned["code_git_run"] == RUN_GIT and "data_revision" not in tuned["missing_provenance"]
    control, = _rows(root, model_key="model_a.random_init", metric="fmax", eval_split="test", subset="all")
    assert control["model_kind"] == "mind_random_init" and control["init_seed"] == 1
    assert control["data_revision"] is None and "data_revision" in control["missing_provenance"]


# -- T2: re-ingesting the same files adds nothing -------------------------------------------------------------

def test_second_ingest_adds_no_rows_and_logs_both_runs(tmp_path):
    root, outputs = tmp_path / "results", _tree(tmp_path)
    first = ledger.ingest(root, outputs_root=outputs, command="test")
    second = ledger.ingest(root, outputs_root=outputs, command="test")
    assert first["rows_added"] > 0 and second["rows_added"] == 0 and second["files_skipped_same_sha"] == 15
    assert len(ledger.read_records(root, "ingests")) == 2
    assert len(ledger.read_records(root, "models")) == 1


# -- T3: history cannot be edited behind the ledger's back ---------------------------------------------------

def test_edited_history_stops_appends_and_verify(tmp_path):
    root, outputs = tmp_path / "results", _tree(tmp_path)
    ledger.ingest(root, outputs_root=outputs, command="test")
    path = root / "ledger" / "measurements.jsonl"
    text = path.read_text()
    path.write_text(text.replace('"value": 0.5', '"value": 0.9', 1))
    with pytest.raises(ledger.LedgerError, match="measurements.jsonl"):
        ledger.verify(root)
    with pytest.raises(ledger.LedgerError, match="measurements.jsonl"):
        ledger.annotate(root, {"task": "ec"}, "caveat", "x")


def test_lines_appended_outside_the_ledger_are_refused(tmp_path):
    root, outputs = tmp_path / "results", _tree(tmp_path)
    ledger.ingest(root, outputs_root=outputs, command="test")
    with open(root / "ledger" / "annotations.jsonl", "a") as handle:
        handle.write('{"status": "invalid"}\n')
    with pytest.raises(ledger.LedgerError, match="annotations.jsonl"):
        ledger.verify(root)


# -- T4: one model name, one checkpoint ---------------------------------------------------------------------

def test_same_model_name_with_another_checkpoint_is_refused(tmp_path):
    root = tmp_path / "results"
    first, second = tmp_path / "o1", tmp_path / "o2"
    card(first, "model_a", SHA_A)
    write_probe(first, ec_metrics("model_a", 0, 0.5))
    card(second, "model_a", SHA_B)
    write_probe(second, ec_metrics("model_a", 0, 0.5), sha=SHA_B)
    ledger.ingest(root, outputs_root=first, command="test")
    before = len(ledger.read_records(root, "measurements"))
    with pytest.raises(ledger.LedgerError, match="model_a.*sha256"):
        ledger.ingest(root, outputs_root=second, command="test")
    assert len(ledger.read_records(root, "measurements")) == before


def test_card_and_embedding_checkpoints_must_agree(tmp_path):
    outputs = tmp_path / "outputs"
    card(outputs, "model_a", SHA_B)
    write_probe(outputs, ec_metrics("model_a", 0, 0.5), sha=SHA_A)
    with pytest.raises(ledger.LedgerError, match="card"):
        ledger.ingest(tmp_path / "results", outputs_root=outputs, command="test")


def test_random_init_control_shares_the_template_sha_without_conflict(tmp_path):
    root, outputs = tmp_path / "results", tmp_path / "outputs"
    card(outputs, "model_a", SHA_A)
    write_probe(outputs, ec_metrics("model_a", 0, 0.5))
    write_probe(outputs, ec_metrics("model_a.random_init", 0, 0.3), random_init=True)
    ledger.ingest(root, outputs_root=outputs, command="test")
    kinds = {(r["model_key"], r["model_kind"]) for r in ledger.read_records(root, "measurements")}
    assert kinds == {("model_a", "mind"), ("model_a.random_init", "mind_random_init")}


# -- T5 / T6: malformed inputs fail loudly ----------------------------------------------------------------

def test_missing_value_without_a_reason_is_an_error(tmp_path):
    outputs = tmp_path / "outputs"
    card(outputs, "model_a", SHA_A)
    metrics = ec_metrics("model_a", 0, 0.5)
    metrics["test"]["all"]["pair_aupr"] = None
    path = write_probe(outputs, metrics) / "metrics.json"
    with pytest.raises(ValueError, match=f"pair_aupr.*{path}"):
        ledger.ingest(tmp_path / "results", outputs_root=outputs, command="test")


def test_undefined_correlation_keeps_its_reason(tmp_path):
    root, outputs = tmp_path / "results", tmp_path / "outputs"
    card(outputs, "model_a", SHA_A)
    metrics = lba_metrics("model_a", 0, 1.5)
    metrics["test"]["all"] = lba_block(1.5, pearson=None, reason="constant_prediction")
    write_probe(outputs, metrics, splits=("train", "val", "test"))
    ledger.ingest(root, outputs_root=outputs, command="test")
    row, = _rows(root, eval_split="test", metric="pearson_r")
    assert row["value"] is None and row["undefined_reason"] == "constant_prediction"
    assert _rows(root, eval_split="test", metric="rmse")[0]["value"] == 1.5


def test_unknown_kind_and_unknown_field_are_errors(tmp_path):
    outputs = tmp_path / "outputs"
    card(outputs, "model_a", SHA_A)
    metrics = ec_metrics("model_a", 0, 0.5)
    metrics["kind"] = "ranking"
    write_probe(outputs, metrics)
    with pytest.raises(ValueError, match="kind 'ranking'"):
        ledger.ingest(tmp_path / "r1", outputs_root=outputs, command="test")
    metrics = ec_metrics("model_a", 0, 0.5)
    metrics["test"]["all"]["histogram"] = {"a": 1}
    write_probe(outputs, metrics)
    with pytest.raises(ValueError, match="histogram"):
        ledger.ingest(tmp_path / "r2", outputs_root=outputs, command="test")


def test_legacy_file_without_kind_takes_the_catalog_kind_visibly(tmp_path):
    """Older EC probes wrote no ``kind``; the catalog kind is used and the gap is listed."""
    root, outputs = tmp_path / "results", tmp_path / "outputs"
    card(outputs, "model_a", SHA_A)
    metrics = ec_metrics("model_a", 0, 0.5)
    del metrics["kind"]
    write_probe(outputs, metrics)
    ledger.ingest(root, outputs_root=outputs, command="test")
    row = _rows(root, metric="fmax", eval_split="test", subset="all")[0]
    assert row["kind"] == "multilabel" and "kind" in row["missing_provenance"]


def test_metrics_outside_the_catalog_are_kept_and_marked(tmp_path):
    root, outputs = tmp_path / "results", tmp_path / "outputs"
    card(outputs, "model_a", SHA_A)
    write_probe(outputs, ec_metrics("model_a", 0, 0.5))
    ledger.ingest(root, outputs_root=outputs, command="test")
    extra, = _rows(root, eval_split="test", subset="all", metric="subset_accuracy@0.5")
    assert extra["is_catalog_metric"] is False and extra["value"] == 0.1


# -- T7: concurrent writers never interleave or lose lines ---------------------------------------------------

def _append_many(root, worker):
    for i in range(40):
        ledger.append_records(root, "ingests", [{"worker": worker, "i": i, "pad": "x" * 2000}])


def test_parallel_appends_keep_the_ledger_consistent(tmp_path):
    root = tmp_path / "results"
    context = multiprocessing.get_context("fork")
    workers = [context.Process(target=_append_many, args=(root, w)) for w in range(3)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
        assert worker.exitcode == 0
    records = ledger.read_records(root, "ingests")
    assert len(records) == 120 and len({(r["worker"], r["i"]) for r in records}) == 120
    ledger.verify(root)


# -- T8: the run's own commit is never borrowed from the embedding step ---------------------------------------

def test_missing_probe_commit_is_listed_not_copied(tmp_path):
    root, outputs = tmp_path / "results", tmp_path / "outputs"
    card(outputs, "model_a", SHA_A)
    write_probe(outputs, ec_metrics("model_a", 0, 0.5), run_git=None)
    ledger.ingest(root, outputs_root=outputs, command="test")
    row = _rows(root, metric="fmax", eval_split="test", subset="all")[0]
    assert row["code_git_run"] is None and "run_git" in row["missing_provenance"]
    assert row["code_git_embed"] == [EMBED_GIT]
    assert "train_config_sha256" in row["missing_provenance"]  # cards carry no training config hash


# -- models, selection rule, baselines ---------------------------------------------------------------------

def test_model_card_selection_rule_comes_from_the_card(tmp_path):
    root, outputs = tmp_path / "results", tmp_path / "outputs"
    card(outputs, "model_a", SHA_A)
    card(outputs, "model_a_last", SHA_B, best=False, epoch=28)
    write_probe(outputs, ec_metrics("model_a", 0, 0.5))
    write_probe(outputs, ec_metrics("model_a_last", 0, 0.5), sha=SHA_B)
    ledger.ingest(root, outputs_root=outputs, command="test")
    models = {m["model_key"]: m for m in ledger.read_records(root, "models")}
    assert models["model_a"]["selection_rule"].startswith("best-checkpoint by val_total_loss")
    assert models["model_a_last"]["selection_rule"].startswith("not the ModelCheckpoint best")
    assert models["model_a"]["sha256"] == SHA_A and models["model_a"]["epoch"] == 98


def test_hf_baseline_is_identified_by_revision(tmp_path):
    root, outputs = tmp_path / "results", tmp_path / "outputs"
    write_probe(outputs, ec_metrics("esm2_t6_8M_UR50D", 0, 0.7, pooling="mean_residues"), hf=True)
    ledger.ingest(root, outputs_root=outputs, command="test")
    row = _rows(root, metric="fmax", eval_split="test", subset="all")[0]
    assert row["model_kind"] == "hf_baseline" and row["model_sha256"] is None
    assert row["hf_revision"] == "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"


def test_embedding_only_sets_are_recorded_for_the_missing_report(tmp_path):
    root, outputs = tmp_path / "results", tmp_path / "outputs"
    card(outputs, "model_a", SHA_A)
    write_embeddings_only(outputs, "go_cc", "model_a")
    summary = ledger.ingest(root, outputs_root=outputs, command="test")
    assert summary["rows_added"] == 0
    ingest, = ledger.read_records(root, "ingests")
    assert {"task": "go_cc", "model_key": "model_a", "splits": ["test", "train", "valid"]} in ingest["embedding_sets"]


# -- pretraining evaluations ---------------------------------------------------------------------------------

def _eval_dir(path, *, counterfactual=None, hiding=None):
    path.mkdir(parents=True)
    provenance = {"checkpoint": {"path": "/m/model.ckpt", "sha256": SHA_A, "epoch": 98},
                  "git": {"commit": RUN_GIT, "dirty": True}, "split": "test", "seed": 0,
                  "masking_protocol": {"mlm_mask_ratio": 0.15}}
    if hiding is not None:
        provenance["unimol_masked_geometry_hiding"] = hiding
    summary = {"n_masked": 100, "top1_acc_real": 0.9, "top3_acc_real": 0.99, "ce_real": 0.2, "ece_real": 0.01,
               "top1_acc_virtual": None}
    metrics = {"provenance": provenance, "plan": {}, "losses": {}, "mask_digest": {},
               "domain_mlm": {"protein": {"batches": 3, "summary": summary},
                              "unimol": {"batches": 3, "summary": summary}}}
    if counterfactual:
        provenance["counterfactual"] = {"mode": counterfactual, "domains": ["unimol"]}
        metrics["counterfactual"] = {"mode": counterfactual}
    (path / "metrics.json").write_text(json.dumps(metrics))
    return path


def test_pretraining_and_counterfactual_evals_are_separate_tasks(tmp_path):
    root, outputs = tmp_path / "results", tmp_path / "outputs"
    card(outputs, "model_a", SHA_A)
    plain = _eval_dir(tmp_path / "eval" / "model_a_test")
    noise = _eval_dir(tmp_path / "eval" / "cf" / "masked_pos_noise", counterfactual="masked_pos_noise")
    ledger.ingest(root, outputs_root=outputs, eval_dirs=[plain, noise], command="test")
    top1, = _rows(root, task="pretrain_eval", subset="protein", metric="top1_acc_real")
    assert top1["value"] == 0.9 and top1["model_sha256"] == SHA_A and top1["code_git_run"] == RUN_GIT + "+dirty"
    cf = _rows(root, task="counterfactual", subset="unimol", metric="top1_acc_real")
    assert len(cf) == 1 and cf[0]["protocol_id"] == "masked_pos_noise"
    assert top1["protocol_settings_sha256"] != cf[0]["protocol_settings_sha256"]


def test_eval_settings_carry_the_effective_geometry_hiding(tmp_path):
    root, outputs = tmp_path / "results", tmp_path / "outputs"
    card(outputs, "model_a", SHA_A)
    on = {"enabled": True, "bond_threshold_angstrom": 2.1}
    off = {"enabled": False, "bond_threshold_angstrom": None}
    dirs = {
        "legacy": _eval_dir(tmp_path / "e" / "legacy", counterfactual="none"),  # metrics.json written before the record
        "forced_on": _eval_dir(tmp_path / "e" / "forced_on", counterfactual="none",
                               hiding={"source": "on", "checkpoint": off, "effective": on}),
        "trained_on": _eval_dir(tmp_path / "e" / "trained_on", counterfactual="none",
                                hiding={"source": "checkpoint", "checkpoint": on, "effective": on}),
        "off": _eval_dir(tmp_path / "e" / "off", counterfactual="none",
                         hiding={"source": "checkpoint", "checkpoint": off, "effective": off}),
    }
    ledger.ingest(root, outputs_root=outputs, eval_dirs=list(dirs.values()), command="test")
    sha = {}
    for row in _rows(root, task="counterfactual", subset="unimol", metric="top1_acc_real"):
        name = Path(row["source_metrics_path"]).parent.name
        sha[name] = row["protocol_settings_sha256"]
    assert set(sha) == set(dirs)
    # the same effective condition shares one settings hash, whatever the source of the flag
    assert sha["forced_on"] == sha["trained_on"]
    assert sha["off"] != sha["forced_on"]
    # a legacy file keeps the hash it had before the record existed
    legacy_settings = {"masking_protocol": {"mlm_mask_ratio": 0.15}, "counterfactual": {"mode": "none", "domains": ["unimol"]},
                       "domains": None}
    assert sha["legacy"] == ledger.sha256_json(legacy_settings)


# -- annotations -------------------------------------------------------------------------------------------

def test_annotations_select_rows_and_the_latest_status_wins(tmp_path):
    root, outputs = tmp_path / "results", _tree(tmp_path)
    ledger.ingest(root, outputs_root=outputs, command="test")
    ledger.annotate(root, {"task": "ec", "data_revision": "rev_v2"}, "superseded", "test set changed",
                    replaced_by="rev_v3")
    ledger.annotate(root, {"task": "ec", "run_seed": 0}, "caveat", "seed 0 odd")
    rows = ledger.apply_annotations(ledger.read_records(root, "measurements"),
                                    ledger.read_records(root, "annotations"))
    ec0 = [r for r in rows if r["task"] == "ec" and r["run_seed"] == 0]
    assert all(r["status"] == "superseded" and r["caveats"] == ["seed 0 odd"] for r in ec0)
    assert all(r["status"] == "valid" for r in rows if r["task"] == "lba")
    ledger.annotate(root, {"task": "ec"}, "valid", "restored")
    rows = ledger.apply_annotations(ledger.read_records(root, "measurements"),
                                    ledger.read_records(root, "annotations"))
    assert all(r["status"] == "valid" for r in rows if r["task"] == "ec")


@pytest.mark.parametrize("selector, status, message", [
    ({"task": "nope"}, "superseded", "matches no measurement"),
    ({"tsk": "ec"}, "superseded", "unknown selector field"),
    ({"task": "ec"}, "deleted", "status"),
    ({}, "superseded", "empty selector"),
])
def test_bad_annotations_are_refused(tmp_path, selector, status, message):
    root, outputs = tmp_path / "results", _tree(tmp_path)
    ledger.ingest(root, outputs_root=outputs, command="test")
    with pytest.raises(ledger.LedgerError, match=message):
        ledger.annotate(root, selector, status, "reason")
    assert ledger.read_records(root, "annotations") == []
