"""``ledger recompute-macro-aupr``: a missing macro AUPR recomputed from a run's saved predictions.npz.

Synthetic trees only (``ledger_helpers``). The target model's metrics.json lacks macro AUPR; the control model's
has it. Every safety check must stop the command before anything is written.
"""

from __future__ import annotations

import numpy as np
import pytest

from downstream import ledger, ledger_compare, ledger_report
from downstream.common.metrics import evaluate, label_aupr
from ledger_helpers import SHA_A, SHA_B, card, ec_metrics, ec_predictions, write_probe

MACRO_KEYS = ("macro_aupr", "macro_aupr_num_terms", "macro_aupr_num_terms_skipped")


def _run(outputs, model, seed, sha, *, macro, macro_offset=0.0):
    predictions = ec_predictions(np.random.default_rng(seed + (0 if macro else 10)), quality=0.5)
    block = evaluate(predictions["test_labels"], predictions["test_scores"])
    if not macro:
        block = {k: v for k, v in block.items() if k not in MACRO_KEYS}
    else:
        block["macro_aupr"] += macro_offset
    metrics = ec_metrics(model, seed, 0.5)
    metrics["val"], metrics["test"] = block, {"all": block}
    return write_probe(outputs, metrics, sha=sha, predictions=predictions)


def _setup(tmp_path, control_offset=0.0):
    root, outputs = tmp_path / "results", tmp_path / "outputs"
    card(outputs, "model_a", SHA_A)
    card(outputs, "po", SHA_B)
    dirs = {seed: _run(outputs, "model_a", seed, SHA_A, macro=False) for seed in (0, 1, 2)}
    for seed in (0, 1, 2):
        _run(outputs, "po", seed, SHA_B, macro=True, macro_offset=control_offset)
    ledger.ingest(root, outputs_root=outputs, command="test")
    return root, dirs


def _recompute(root, *extra):
    return ledger.main(["--results-root", str(root), "recompute-macro-aupr", "--task", "ec", "--model", "model_a",
                        "--control", "po", "--data-revision", "rev_v2", *extra])


def _macro_rows(root, model="model_a"):
    return [r for r in ledger.read_records(root, "measurements") if r["model_key"] == model
            and r["metric"] == "macro_aupr" and (r["eval_split"], r["subset"]) == ("test", "all")]


def test_dry_run_writes_nothing(tmp_path, capsys):
    root, _ = _setup(tmp_path)
    before = ledger.verify(root)
    assert _recompute(root, "--dry-run") == 0
    out = capsys.readouterr().out
    assert "DRY-RUN" in out and out.count("control po seed") == 3
    assert ledger.verify(root) == before and not _macro_rows(root)


def test_recompute_adds_one_row_per_seed_with_its_source(tmp_path):
    root, dirs = _setup(tmp_path)
    assert _recompute(root) == 0
    rows = sorted(_macro_rows(root), key=lambda r: r["run_seed"])
    assert [r["run_seed"] for r in rows] == [0, 1, 2]
    for row in rows:
        with np.load(dirs[row["run_seed"]] / "predictions.npz") as data:
            assert row["value"] == pytest.approx(label_aupr(data["test_labels"], data["test_scores"])["macro_aupr"])
        assert "predictions_sha256=" in row["recomputed_from"] and row["derived_from"] is None
        assert row["is_primary"] is False and row["unit"] == "fraction"
    ingests = ledger.read_records(root, "ingests")
    assert ingests[-1]["rows_added"] == 3 and ingests[-1]["ingest_id"] == rows[0]["ingest_id"]
    # S1 + S2: the new rows join the setting of their source files and compare without an error
    state = ledger_report.load_state(root)
    assert len(ledger_report.sections(state, "ec")) == 1
    pair = next(p for p in ledger_compare.compare(state, "po", "model_a", n_boot=20)["pairs"]
                if (p["metric"], p["eval_split"], p["subset"]) == ("macro_aupr", "test", "all"))
    assert pair["mean_b"] == pytest.approx(np.mean([r["value"] for r in rows]))


def test_second_run_is_refused(tmp_path):
    root, _ = _setup(tmp_path)
    _recompute(root)
    with pytest.raises(ledger.LedgerError, match="already has a macro_aupr row"):
        _recompute(root)


def test_changed_metrics_file_is_refused(tmp_path):
    root, dirs = _setup(tmp_path)
    path = dirs[1] / "metrics.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ledger.LedgerError, match="changed since it was ingested"):
        _recompute(root, "--dry-run")


def test_predictions_that_do_not_reproduce_fmax_are_refused(tmp_path):
    root, dirs = _setup(tmp_path)
    npz = dirs[2] / "predictions.npz"
    with np.load(npz) as data:
        arrays = dict(data)
    arrays["test_scores"] = arrays["test_scores"][::-1].copy()
    np.savez_compressed(npz, **arrays)
    with pytest.raises(ledger.LedgerError, match="does not reproduce"):
        _recompute(root, "--dry-run")
    assert not _macro_rows(root)


def test_control_that_does_not_reproduce_its_stored_value_is_refused(tmp_path):
    root, _ = _setup(tmp_path, control_offset=0.01)
    with pytest.raises(ledger.LedgerError, match="control po seed 0"):
        _recompute(root, "--dry-run")


def test_only_multilabel_tasks_and_a_known_setting(tmp_path):
    root, _ = _setup(tmp_path)
    with pytest.raises(ledger.LedgerError, match="multilabel"):
        ledger.main(["--results-root", str(root), "recompute-macro-aupr", "--task", "lba", "--model", "model_a",
                     "--control", "po", "--data-revision", "rev_v2", "--dry-run"])
    with pytest.raises(ledger.LedgerError, match="no ec test/all fmax row"):
        ledger.main(["--results-root", str(root), "recompute-macro-aupr", "--task", "ec", "--model", "model_a",
                     "--control", "po", "--data-revision", "other_rev", "--dry-run"])
