"""Reports generated from the results ledger: task tables, timeline, summary page, missing cells."""

from __future__ import annotations

import csv
import re

import pytest

from downstream import ledger, ledger_report
from ledger_helpers import SHA_A, SHA_B, card, ec_metrics, write_embeddings_only, write_probe

LITERATURE_HEADER = ("task,paper,model,metric,eval_split,subset,value,std,unit,protocol,n_seeds,split_note,pdf,page,"
                     "table\n")


def _ingest(tmp_path, build):
    root, outputs = tmp_path / "results", tmp_path / "outputs"
    card(outputs, "model_a", SHA_A)
    card(outputs, "po", SHA_B)
    build(outputs)
    ledger.ingest(root, outputs_root=outputs, command="test")
    return root


def _table_cells(markdown: str) -> list:
    cells = []
    for line in markdown.splitlines():
        if line.startswith("|") and not set(line) <= set("|-: "):
            cells += [cell.strip() for cell in line.strip().strip("|").split("|")]
    return cells


# -- different pooling or data revision never share a column ------------------------------------------

def test_pooling_and_data_revision_split_sections(tmp_path):
    def build(outputs):
        for seed in (0, 1, 2):
            write_probe(outputs, ec_metrics("model_a", seed, 0.50))
            write_probe(outputs, ec_metrics("model_a", seed, 0.40, pooling="ca"))
            write_probe(outputs, ec_metrics("po", seed, 0.45), sha=SHA_B, revision="rev_v1")
    state = ledger_report.load_state(_ingest(tmp_path, build))
    sections = ledger_report.sections(state, "ec")
    keys = {(s["pooling"], s["data_revision"], tuple(s["models"])) for s in sections}
    assert keys == {("mean_atoms", "rev_v2", ("model_a",)), ("ca", "rev_v2", ("model_a",)), ("mean_atoms", "rev_v1", ("po",))}
    markdown, _ = ledger_report.task_report(state, "ec")
    assert markdown.count("### Setting") == 3


def test_frozen_probe_and_finetune_are_separate_blocks(tmp_path):
    from ledger_helpers import write_finetune

    def build(outputs):
        write_probe(outputs, ec_metrics("model_a", 0, 0.50))
        write_finetune(outputs, ec_metrics("model_a", 0, 0.60, protocol="finetune"))
    state = ledger_report.load_state(_ingest(tmp_path, build))
    families = [s["protocol_family"] for s in ledger_report.sections(state, "ec")]
    assert sorted(families) == ["finetune", "frozen_probe"]


# -- every missing cell says why; no cell is empty -----------------------------------------------------

def test_missing_cells_carry_a_category_and_no_cell_is_empty(tmp_path):
    def build(outputs):
        for seed in (0, 1):  # 2 of 3 expected seeds
            write_probe(outputs, ec_metrics("model_a", seed, 0.50, macro=False))
    state = ledger_report.load_state(_ingest(tmp_path, build))
    markdown, rows = ledger_report.task_report(state, "ec")
    cells = _table_cells(markdown)
    assert cells and all(cells), "an empty table cell was rendered"
    assert "MISSING:seeds 2/3" in markdown
    assert "MISSING:metric-not-in-file" in markdown  # macro AUPR absent from the files
    assert any(r["cell"] == "MISSING:metric-not-in-file" and r["metric"] == "coverage" for r in rows)  # coded, not in the file
    assert any(r["cell"] == "MISSING:metric-not-in-file" and r["metric"] == "macro_aupr" for r in rows)


def test_undefined_values_render_with_their_reason(tmp_path):
    def build(outputs):
        metrics = ec_metrics("model_a", 0, 0.5)
        metrics["test"]["cutoff_30"] = None
        write_probe(outputs, metrics)
    state = ledger_report.load_state(_ingest(tmp_path, build))
    markdown, _ = ledger_report.task_report(state, "ec")
    assert "UNDEFINED:empty subset (no evaluated sample)" in markdown


def test_missing_list_categories(tmp_path):
    def build(outputs):
        for seed in (0, 1, 2):
            write_probe(outputs, ec_metrics("model_a", seed, 0.50))
            write_probe(outputs, ec_metrics("po", seed, 0.45), sha=SHA_B, revision="rev_v1")
        write_embeddings_only(outputs, "go_mf", "model_a")
    root = _ingest(tmp_path, build)
    ledger.annotate(root, {"model_key": "po"}, "superseded", "old test set")
    state = ledger_report.load_state(root)
    missing = ledger_report.missing_cells(state, ["model_a", "po", "ghost"])
    by = {(m["task"], m["model_key"], m["metric"], m["subset"]): m["category"] for m in missing}
    assert by[("ec", "model_a", "coverage", "all")] == "MISSING:metric-not-in-file"  # code exists, files predate it
    assert ("ec", "model_a", "fmax", "all") not in by
    assert by[("ec", "po", "fmax", "all")] == "OLD-REVISION-ONLY"
    assert by[("go_mf", "model_a", "fmax", "all")] == "MISSING:embedding-only"
    assert by[("go_mf", "po", "fmax", "all")] == "MISSING:not-run"  # po is in the ledger, never run on go_mf
    assert by[("go_mf", "ghost", "fmax", "all")] == "MISSING:model-not-in-ledger"  # listed, never skipped silently
    assert by[("ec", "ghost", "fmax", "all")] == "MISSING:model-not-in-ledger"
    hint = next(m["hint"] for m in missing if m["task"] == "go_mf" and m["model_key"] == "model_a")
    assert "make ds-probe TASK=go_mf MODEL_ID=model_a HEAD=mlp" in hint
    assert all(m["category"] for m in missing)


# -- literature is its own column, in its own unit -------------------------------------------------------

def test_literature_column_keeps_unit_and_protocol(tmp_path):
    root = _ingest(tmp_path, lambda outputs: write_probe(outputs, ec_metrics("model_a", 0, 0.5)))
    state = ledger_report.load_state(root)
    markdown, _ = ledger_report.task_report(state, "ec")
    assert "no literature file" in markdown
    (root / "literature").mkdir()
    (root / "literature" / "literature.csv").write_text(
        LITERATURE_HEADER + "ec,SaProt,SaProt,fmax,test,all,88.2,,percent,full fine-tune,1,,saprot.pdf,7,T2\n"
        "ec,GearNet,GearNet-Edge,fmax,test,all,81.0,,percent,full fine-tune,1,,gearnet.pdf,6,T2\n")
    markdown, _ = ledger_report.task_report(ledger_report.load_state(root), "ec")
    assert "88.2 percent (SaProt, SaProt, full fine-tune" in markdown
    header = next(line for line in markdown.splitlines() if line.startswith("| eval_split"))
    assert header.rstrip(" |").endswith("literature (best; separate protocol)")


def test_malformed_literature_rows_are_refused(tmp_path):
    root = _ingest(tmp_path, lambda outputs: write_probe(outputs, ec_metrics("model_a", 0, 0.5)))
    (root / "literature").mkdir()
    (root / "literature" / "literature.csv").write_text(
        LITERATURE_HEADER + "ec,SaProt,SaProt,fmax,test,all,88.2,,,full fine-tune,1,,saprot.pdf,7,T2\n")
    with pytest.raises(ValueError, match="unit"):
        ledger_report.load_state(root)


# -- superseded rows leave the main table but stay in the timeline -------------------------------------

def test_superseded_rows_only_in_timeline(tmp_path):
    def build(outputs):
        write_probe(outputs, ec_metrics("model_a", 0, 0.50))
        write_probe(outputs, ec_metrics("po", 0, 0.45), sha=SHA_B, revision="rev_v1")
    root = _ingest(tmp_path, build)
    ledger.annotate(root, {"data_revision": "rev_v1"}, "superseded", "46 chains lost", replaced_by="rev_v2")
    state = ledger_report.load_state(root)
    markdown, _ = ledger_report.task_report(state, "ec")
    assert "po" not in _table_cells(markdown)
    timeline, timeline_rows = ledger_report.timeline_report(state, "ec")
    assert "po" in timeline and "superseded" in timeline
    assert {r["status"] for r in timeline_rows if r["model_key"] == "po"} == {"superseded"}


def test_recomputed_value_shows_newest_and_footnotes_the_old_one(tmp_path):
    root, outputs = tmp_path / "results", tmp_path / "outputs"
    card(outputs, "model_a", SHA_A)
    write_probe(outputs, ec_metrics("model_a", 0, 0.50))
    ledger.ingest(root, outputs_root=outputs, command="test")
    other = tmp_path / "outputs2"
    card(other, "model_a", SHA_A)
    write_probe(other, ec_metrics("model_a", 0, 0.52))
    ledger.ingest(root, outputs_root=other, command="test")
    markdown, _ = ledger_report.task_report(ledger_report.load_state(root), "ec")
    assert "0.520" in markdown and "previous value 0.500" in markdown


# -- summary page counts match a hand count -----------------------------------------------------------

def test_summary_page_counts(tmp_path):
    root = _ingest(tmp_path, lambda outputs: [write_probe(outputs, ec_metrics("model_a", s, 0.5)) for s in (0, 1, 2)])
    state = ledger_report.load_state(root)
    page = ledger_report.summary_page(state, ["model_a"])
    catalog = state["catalog"]
    tasks = [t for t, spec in catalog["tasks"].items() if spec["family"] not in ("pretraining", "diagnostic")]
    required = sum(len(catalog["tasks"][t]["required"]) for t in tasks)
    coded = sum(1 for t in tasks for c in catalog["tasks"][t]["required"] if c.get("code", "present") == "present")
    expected = sum(len(catalog["tasks"][t]["required_protocols"]) * sum(
        1 for c in catalog["tasks"][t]["required"] if c.get("code", "present") == "present") for t in tasks)
    assert f"{required} required metric cells" in page and f"computes {coded} of them" in page
    assert f"{required - coded} NO-CODE" in page
    # EC: test/all fmax, cutoff_30 fmax, cutoff_95 fmax, pair_aupr, macro_aupr present (5 of 8 coded cells)
    assert f"expected {expected} cells, filled 5, missing {expected - 5}" in page
    ec_line = next(line for line in page.splitlines() if line.startswith("| ec "))
    assert "0.500 ± 0.000" in ec_line
    assert "pretraining-seed noise was not measured" in page


# -- every report names the ledger state it came from --------------------------------------------------

def test_report_header_names_ledger_hash(tmp_path):
    root = _ingest(tmp_path, lambda outputs: write_probe(outputs, ec_metrics("model_a", 0, 0.5)))
    state = ledger_report.load_state(root)
    sha = ledger.verify(root)["measurements_sha256"]
    markdown, _ = ledger_report.task_report(state, "ec")
    assert sha in markdown and f"{len(state['rows'])} rows" in markdown
    assert "std: ddof=1 (sample std)" in markdown
    written = ledger_report.write_reports(root, models=["model_a"], state=state)
    assert (root / "reports" / "tasks" / "ec.md").read_text() == markdown
    assert (root / "reports" / "summary.md") in written
    with open(root / "reports" / "missing_cells.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all(r["category"] for r in rows)
    assert re.search(r"sha256 [0-9a-f]{64}", (root / "reports" / "summary.md").read_text())


def test_summary_models_default_then_comparisons_yaml_then_cli(tmp_path):
    root = _ingest(tmp_path, lambda outputs: write_probe(outputs, ec_metrics("model_a", 0, 0.5)))
    state = ledger_report.load_state(root)
    assert ledger_report.report_models(root, None, state["models"]) == ["model_a"]  # default: the ledger's models
    with pytest.raises(ValueError, match="no models in the ledger"):
        ledger_report.report_models(root, None, [])
    (root / "comparisons.yaml").write_text("summary_models: [model_a, model_b]\n")
    assert ledger_report.report_models(root, None, state["models"]) == ["model_a", "model_b"]
    assert ledger_report.report_models(root, ["po"], state["models"]) == ["po"]
    page = ledger_report.summary_page(state, ["model_a", "model_b"])
    assert "`model_b`" in page and "MISSING:model-not-in-ledger" in page


def test_std_is_the_sample_std_and_undefined_for_one_seed(tmp_path):
    def build(outputs):
        write_probe(outputs, ec_metrics("model_a", 0, 0.50))
        write_probe(outputs, ec_metrics("model_a", 1, 0.52))
        write_probe(outputs, ec_metrics("po", 0, 0.45), sha=SHA_B)
    markdown, rows = ledger_report.task_report(ledger_report.load_state(_ingest(tmp_path, build)), "ec")
    cell = next(r for r in rows if r["model_key"] == "model_a" and r["metric"] == "fmax" and r["subset"] == "all")
    assert cell["std"] == pytest.approx(0.0141421, abs=1e-6)  # ddof=1; ddof=0 would give 0.010
    single = next(r for r in rows if r["model_key"] == "po" and r["metric"] == "fmax" and r["subset"] == "all")
    assert single["std"] is None and "std undefined (n=1)" in single["cell"]


# -- S1: a recomputed row (new ingest) stays in the setting of the file it was computed from ----------------

def test_recomputed_row_stays_in_the_same_setting(tmp_path):
    import uuid

    root = _ingest(tmp_path, lambda outputs: [write_probe(outputs, ec_metrics("model_a", s, 0.5, macro=False))
                                              for s in (0, 1, 2)])
    templates = [r for r in ledger.read_records(root, "measurements")
                 if (r["eval_split"], r["subset"], r["metric"]) == ("test", "all", "fmax")]
    new_ingest = str(uuid.uuid4())
    ledger.append_records(root, "measurements", [
        dict(row, metric="macro_aupr", value=0.4, row_id=str(uuid.uuid4()), ingest_id=new_ingest,
             ingested_utc=ledger.now_utc(), recomputed_from="predictions.npz") for row in templates])
    sections = ledger_report.sections(ledger_report.load_state(root), "ec")
    assert len(sections) == 1
    assert {r["metric"] for r in sections[0]["rows"] if r["subset"] == "all" and r["eval_split"] == "test"} >= {
        "fmax", "macro_aupr"}
