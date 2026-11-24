"""Per-seed data splits (NPASS): the seed-to-tag rule, the ledger setting over the tags, ds-run planning and wiring.

The ledger and report tests use synthetic result trees (``ledger_helpers``); the planning tests replace the task
description by a stand-in, so no real data and no GPU is involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from downstream import ledger, ledger_compare, ledger_report, run_all
from downstream.common import registry, split_replicates
from ledger_helpers import SHA_A, card, molnet_reg_metrics, write_embeddings_only, write_probe

TASK = "npass_ptp1b"
CONFIG = {"task": TASK, "split_seeds": [0, 1, 2], "split_tags": ["seed0", "seed1", "seed2"]}


# ---------------------------------------------------------------------------
# The seed <-> tag rule
# ---------------------------------------------------------------------------

def test_tags_follow_the_seed_and_other_tags_are_refused():
    assert [split_replicates.tag_for_seed(s) for s in (0, 1, 12)] == ["seed0", "seed1", "seed12"]
    assert split_replicates.seed_for_tag("seed12") == 12
    for bad in ("seed", "seed01", "seed-1", "identity30", "Seed1", "seed1.5", ""):
        with pytest.raises(ValueError, match="not of the form"):
            split_replicates.seed_for_tag(bad)
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="non-negative integer"):
            split_replicates.tag_for_seed(bad)
    assert split_replicates.seeds_of_tag("seed1", [0, 1, 2]) == [1] and split_replicates.seeds_of_tag("seed5", [0, 1]) == []


def test_a_run_gets_the_tag_of_its_seed_and_never_another_seeds_split():
    assert split_replicates.tag_for_run(CONFIG, 1) == "seed1"
    assert split_replicates.tag_for_run(CONFIG, 1, explicit="seed1") == "seed1"
    with pytest.raises(ValueError, match="share the seed"):
        split_replicates.tag_for_run(CONFIG, 1, explicit="seed0")
    with pytest.raises(ValueError, match="seed 7 has no split"):
        split_replicates.tag_for_run(CONFIG, 7)
    assert split_replicates.tag_for_run({"task": "ec"}, 0) is None
    with pytest.raises(ValueError, match="no per-seed splits"):
        split_replicates.tag_for_run({"task": "ec"}, 0, explicit="seed0")


# ---------------------------------------------------------------------------
# Ledger: the tags are the repeats of one cell
# ---------------------------------------------------------------------------

def npass_metrics(model_id, seed, rmse):
    metrics = molnet_reg_metrics(model_id, seed, rmse)
    return {**metrics, "task": TASK, "split_tag": f"seed{seed}"}


def write_seed(outputs, model_id, seed, rmse, *, data_sha=None, pooling="mean_atoms", **kwargs):
    """One probe seed on the split of its own tag (processed/<tag>/<revision>/ as ``downstream.tasks.npass`` writes)."""
    tag = f"seed{seed}"
    metrics = {**npass_metrics(model_id, seed, rmse), "pooling": pooling}
    return write_probe(outputs, metrics, revision=f"{tag}/v1", data_sha=data_sha or (tag * 20)[:64], **kwargs)


def ingest(tmp_path, build):
    root, outputs = tmp_path / "results", tmp_path / "outputs"
    card(outputs, "model_a", SHA_A)
    build(outputs)
    ledger.ingest(root, outputs_root=outputs, command="test")
    return root, outputs


def test_the_catalog_marks_every_npass_task_as_split_replicates_without_tags_in_the_protocol():
    catalog = ledger.load_catalog()
    from downstream.tasks import NPASS_TASKS

    for name in NPASS_TASKS:
        assert catalog["tasks"][name]["split_replicates"] is True
        assert catalog["tasks"][name]["required_protocols"] == ["mlp_probe"]
    assert not any(spec.get("split_replicates") for name, spec in catalog["tasks"].items() if name not in NPASS_TASKS)


def test_the_catalog_refuses_a_tagged_protocol_on_a_per_seed_split_task(tmp_path):
    import yaml

    values = yaml.safe_load(ledger.CATALOG_PATH.read_text())
    values["tasks"][TASK] = dict(values["tasks"][TASK], required_protocols=["mlp_probe.seed0"])
    bad = tmp_path / "catalog.yaml"
    bad.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="protocols carry no split tag"):
        ledger.load_catalog(bad)
    values["tasks"][TASK] = dict(values["tasks"][TASK], required_protocols=["mlp_probe"], split_replicates="yes")
    bad.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="true or false"):
        ledger.load_catalog(bad)


def test_three_tags_are_one_setting_with_three_seeds_and_each_row_keeps_its_tag(tmp_path):
    def build(outputs):
        for seed, rmse in ((0, 1.0), (1, 1.2), (2, 1.1)):
            write_seed(outputs, "model_a", seed, rmse)
    root, _ = ingest(tmp_path, build)
    rows = ledger.read_records(root, "measurements")
    assert {(r["run_seed"], r["split_tag"]) for r in rows} == {(0, "seed0"), (1, "seed1"), (2, "seed2")}
    assert {tuple(r["expected_seeds"]) for r in rows} == {(0, 1, 2)} and {r["protocol_id"] for r in rows} == {"mlp_probe"}
    assert {r["data_revision"] for r in rows} == {"v1"}                    # the revision, not the tag directory
    state = ledger_report.load_state(root)
    sections = ledger_report.sections(state, TASK)
    assert len(sections) == 1 and sections[0]["split_tag"] == ledger_report.REPLICATE_SPLIT_TAG
    assert {r["run_split_tag"] for r in sections[0]["rows"]} == {"seed0", "seed1", "seed2"}
    primary = ledger.load_catalog()["tasks"][TASK]["primary"]
    found = ledger_report.lookup(state, TASK, "model_a", "mlp_probe", primary)
    assert found["category"] is None and "n_seeds=3" in found["text"] and "1.100 ± 0.100" in found["text"]
    assert "MISSING" not in found["text"]
    markdown, _ = ledger_report.task_report(state, TASK)
    assert markdown.count("### Setting") == 1 and "per-seed splits" in markdown


def test_a_missing_tag_is_a_missing_seed_not_a_complete_cell(tmp_path):
    def build(outputs):
        write_seed(outputs, "model_a", 0, 1.0)
        write_seed(outputs, "model_a", 2, 1.1)
    root, _ = ingest(tmp_path, build)
    state = ledger_report.load_state(root)
    found = ledger_report.lookup(state, TASK, "model_a", "mlp_probe", ledger.load_catalog()["tasks"][TASK]["primary"])
    assert found["category"] == "MISSING:seeds 2/3" and "MISSING:seeds 2/3" in found["text"]
    missing = ledger_report.missing_cells(state, ["model_a"], [TASK])
    assert missing and all(m["category"] == "MISSING:seeds 2/3" for m in missing)
    assert f"make ds-run MODEL=model_a TASKS={TASK}" in missing[0]["hint"] and "SPLIT_TAG" not in missing[0]["hint"]


def test_flattening_refuses_a_row_on_another_seeds_split_or_with_a_tagged_protocol(tmp_path):
    outputs = tmp_path / "outputs"
    card(outputs, "model_a", SHA_A)
    catalog = ledger.load_catalog()
    wrong = write_probe(outputs, {**npass_metrics("model_a", 1, 1.0), "split_tag": "seed0"}, revision="seed0/v1")
    with pytest.raises(ValueError, match="seed 1 must carry split_tag 'seed1'"):
        ledger.flatten_run(wrong / "metrics.json", catalog, outputs)
    absent = write_probe(outputs, {**npass_metrics("model_a", 2, 1.0), "split_tag": None}, revision="seed2/v1")
    with pytest.raises(ValueError, match="split_tag None"):
        ledger.flatten_run(absent / "metrics.json", catalog, outputs)
    tagged = write_probe(outputs, {**npass_metrics("model_a", 0, 1.0), "protocol": "mlp_probe.seed0.joint"},
                         revision="seed0/v1")
    with pytest.raises(ValueError, match="protocol without qualifiers"):
        ledger.flatten_run(tagged / "metrics.json", catalog, outputs)
    unlisted = write_probe(outputs, npass_metrics("model_a", 5, 1.0), revision="seed5/v1")
    with pytest.raises(ValueError, match="must contain seed 5"):
        ledger.flatten_run(unlisted / "metrics.json", catalog, outputs)


def test_two_processed_files_for_one_tag_stop_the_report(tmp_path):
    def build(outputs):
        write_seed(outputs, "model_a", 0, 1.0)
        write_seed(outputs, "model_a", 0, 1.0, data_sha="f" * 64, pooling="ca")      # another file for the same tag
    root, _ = ingest(tmp_path, build)
    with pytest.raises(ValueError, match="two different processed files"):
        ledger_report.load_state(root)


def test_compare_names_why_no_paired_bootstrap_exists_for_per_seed_splits(tmp_path):
    def build(outputs):
        card(outputs, "model_b", "b" * 64)
        for seed, (a, b) in enumerate(((1.0, 0.8), (1.2, 0.9), (1.1, 1.0))):
            write_seed(outputs, "model_a", seed, a)
            write_seed(outputs, "model_b", seed, b, sha="b" * 64)
    root, _ = ingest(tmp_path, build)
    result = ledger_compare.compare(ledger_report.load_state(root), "model_a", "model_b", tasks=[TASK], n_boot=10)
    pair, = [p for p in result["pairs"] if p["metric"] == "rmse" and p["eval_split"] == "test"]
    assert pair["n_seeds_a"] == pair["n_seeds_b"] == 3 and pair["common_seeds"] == [0, 1, 2]
    assert pair["ci_low"] is None and "differ between seeds" in pair["ci_reason"]
    assert pair["decision"].startswith("no decision") and pair["diff_b_better"] == pytest.approx(0.2, abs=1e-6)


# ---------------------------------------------------------------------------
# ds-run: a unit per tag, the seed of the tag, protocols without tags
# ---------------------------------------------------------------------------

def test_expand_tasks_gives_one_unit_per_requested_seed_for_a_per_seed_split_task():
    catalog = ledger.load_catalog()
    units, explicit = run_all.expand_tasks(f"{TASK},ec", catalog, [0, 2])
    assert explicit and [u.key for u in units] == [f"{TASK}.seed0", f"{TASK}.seed2", "ec"]
    unit = units[0]
    assert unit.prefix == "seed0." and unit.cli() == ["--split-tag", "seed0"]
    assert unit.protocol("mlp") == "mlp_probe" and unit.protocol(run_all.FINETUNE) == "finetune"
    lba = [u for u in run_all.expand_tasks("lba", catalog)[0]][0]
    assert lba.prefix == "identity30.joint." and lba.protocol("mlp") == "mlp_probe.identity30.joint"
    assert len(run_all.expand_tasks("npass_ptp1b,npass_cox1", catalog, [0, 1, 2])[0]) == 6
    with pytest.raises(run_all.PlanError, match="one split per seed"):
        run_all.expand_tasks(TASK, catalog)


def describe_stand_in(unit, root=Path("/data")):
    """What ``run_all.describe_task`` returns for a unit, with the real task.yaml fine-tune block."""
    from downstream.tasks import load_task

    task = load_task(unit.task)
    finetune = task.load_task_config(task.TASK_YAML)["finetune"]
    directory = root / unit.task / "processed" / (unit.split_tag or "") / "v1"
    return {"modality": "small_molecule", "splits": ["train", "valid", "test"], "pooling": "mean_atoms",
            "finetune_pooling": "mean_atoms", "replicated": True, "finetune": finetune, "finetune_seconds": 10.0,
            "processed": {s: directory / f"{s}.pt" for s in ("train", "valid", "test")}}


def test_each_tags_unit_trains_only_its_own_seed_and_reads_its_own_embeddings(tmp_path):
    catalog = ledger.load_catalog()
    units, explicit = run_all.expand_tasks(TASK, catalog, [0, 1, 2])
    model = run_all.ModelPlan("snap", "snap", "a" * 64, ["unimol"], "snapshot exists")
    outputs = tmp_path / "outputs"
    plan = run_all.plan_units(units, explicit, model, heads=["mlp"], seeds=[0, 1, 2], outputs_root=outputs,
                              describe=describe_stand_in)
    assert [(u.split_tag, head, todo) for u, head, todo in plan["probe"]] == [
        ("seed0", "mlp", [0]), ("seed1", "mlp", [1]), ("seed2", "mlp", [2])]
    assert [f[0] for f in plan["failed"]] == [f"embed {TASK}.seed{k}" for k in range(3)]     # no processed data here
    # one tag's seed already done: only the other tags remain, and its embedding is not redone
    done = registry.results_dir(TASK, "snap", "mlp_probe", 1, pooling="mean_atoms", root=outputs)
    done.mkdir(parents=True)
    (done / "metrics.json").write_text("{}")
    plan = run_all.plan_units(units, explicit, model, heads=["mlp"], seeds=[0, 1, 2], outputs_root=outputs,
                              describe=describe_stand_in)
    assert [(u.split_tag, todo) for u, _, todo in plan["probe"]] == [("seed0", [0]), ("seed2", [2])]
    assert [u.split_tag for u, _ in plan["probe_done"]] == ["seed1"]


def test_jobs_pass_the_tag_to_embed_probe_and_finetune_and_check_the_tags_outputs(tmp_path):
    catalog = ledger.load_catalog()
    units, explicit = run_all.expand_tasks(TASK, catalog, [0, 1])
    model = run_all.ModelPlan("snap", "snap", "a" * 64, ["unimol"], "snapshot exists")
    outputs, data = tmp_path / "outputs", tmp_path / "data"
    for unit in units:                                   # processed splits exist for both tags
        for path in describe_stand_in(unit, data)["processed"].values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x")
    plan = run_all.plan_units(units, explicit, model, heads=["mlp", "finetune"], seeds=[0, 1], outputs_root=outputs,
                              describe=lambda unit: describe_stand_in(unit, data))
    assert not plan["failed"]
    commands = {key: ["cmd", key] for key in ("embed", "probe", "finetune")}
    embeds, probes, finetunes = run_all.build_jobs(plan, model, commands=commands, outputs_root=outputs, threads=1,
                                                   random_init=False)
    assert len(embeds) == len(probes) == len(finetunes) == 2

    def after(argv, flag):
        return argv[argv.index(flag) + 1]

    for job, seed in zip(embeds, (0, 1)):
        argv = job.argv("0")
        assert after(argv, "--split-tag") == f"seed{seed}" and "--representation" not in argv
        assert after(argv, "--task") == TASK and argv[argv.index("--splits") + 1:] == ["train", "valid", "test"]
    for job, seed in zip(probes, (0, 1)):
        argv = job.argv("cpu0")
        assert after(argv, "--split-tag") == f"seed{seed}" and after(argv, "--seeds") == str(seed)
    for job, seed in zip(finetunes, (0, 1)):
        argv = job.argv("0")
        assert after(argv, "--split-tag") == f"seed{seed}" and after(argv, "--seeds") == str(seed)
    assert "metrics.json missing for seeds [0]" in probes[0].check()
    done = registry.results_dir(TASK, "snap", "mlp_probe", 0, pooling="mean_atoms", root=outputs)
    done.mkdir(parents=True)
    (done / "metrics.json").write_text("{}")
    assert probes[0].check() is None and "metrics.json missing for seeds [1]" in probes[1].check()


def test_a_seed_without_a_processed_split_is_a_failed_config_not_a_silent_skip():
    units = [run_all.Unit(TASK, "seed7")]
    model = run_all.ModelPlan("snap", "snap", "a" * 64, ["unimol"], "snapshot exists")
    plan = run_all.plan_units(units, True, model, heads=["mlp"], seeds=[7], outputs_root=Path("/nonexistent"))
    assert plan["failed"] and "no split" in plan["failed"][0][1] and plan["units"] == []


def test_the_run_summary_has_one_row_per_task_and_protocol_for_the_tags(tmp_path):
    def build(outputs):
        for seed in (0, 1, 2):
            write_seed(outputs, "model_a", seed, 1.0 + 0.1 * seed)
    root, _ = ingest(tmp_path, build)
    units, _ = run_all.expand_tasks(TASK, ledger.load_catalog(), [0, 1, 2])
    model = run_all.ModelPlan("model_a", "model_a", SHA_A, ["unimol"], "snapshot exists")
    plan = {"units": units, "domain_skips": []}
    text, _ = run_all.summary(root, model, plan, ["mlp"], [(f"probe_mlp {TASK}.seed1", "boom")])
    rows = [line for line in text.splitlines() if line.startswith(f"| {TASK}")]
    assert len(rows) == 1 and "n_seeds=3" in rows[0] and "mlp_probe" in rows[0]


# ---------------------------------------------------------------------------
# embed / probe / finetune command lines
# ---------------------------------------------------------------------------

def test_embed_probe_and_finetune_refuse_a_tag_where_none_applies_or_one_is_missing(capsys, monkeypatch, tmp_path):
    from downstream import embed, finetune, probe

    monkeypatch.setenv("MIND_DATA_ROOT", str(tmp_path))
    with pytest.raises(SystemExit):
        embed.main(["--task", "ec", "--model", "x", "--split-tag", "seed0"])
    assert "--split-tag applies to the lba task and to tasks with one split per seed only" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        embed.main(["--task", TASK, "--model", "x"])                      # the tag of an embedding set must be named
    assert "--split-tag must be one of ['seed0', 'seed1', 'seed2']" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        embed.main(["--task", TASK, "--model", "x", "--split-tag", "identity30"])
    with pytest.raises(SystemExit):
        embed.main(["--task", TASK, "--model", "x", "--split-tag", "seed0", "--representation", "joint"])
    capsys.readouterr()
    with pytest.raises(SystemExit):
        probe.main(["--task", "ec", "--model-id", "x", "--head", "mlp", "--split-tag", "seed0"])
    with pytest.raises(SystemExit):
        probe.main(["--task", TASK, "--model-id", "x", "--head", "mlp", "--seeds", "1", "--split-tag", "seed0"])
    assert "share the seed" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        probe.main(["--task", TASK, "--model-id", "x", "--head", "mlp", "--seeds", "3"])
    assert "seed 3 has no split" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        finetune.main(["--task", "mn_esol", "--model", "x", "--split-tag", "seed0"])


# ---------------------------------------------------------------------------
# ds-prepare: one processed directory per tag, each verified against its own record
# ---------------------------------------------------------------------------

def test_prepare_plan_and_verification_cover_every_tag_directory(tmp_path, monkeypatch):
    from downstream import prepare_all
    from downstream.tasks.npass import common
    from test_downstream_npass import fixture_rows, tiny_config, write_csv

    monkeypatch.setenv("MIND_DATA_ROOT", str(tmp_path / "data"))
    info = prepare_all.inspect_task(TASK)
    assert [d.parent.name for d in info.processed] == ["seed0", "seed1", "seed2"] and info.splits == ("train", "valid", "test")
    assert prepare_all.processed_state(info.processed, info.splits)[0] == prepare_all.PENDING
    config = tiny_config(tmp_path)
    write_csv(common.csv_path(config), fixture_rows())
    assert common.build_processed(config, splits=list(config["splits"]), tags=["seed0"], config_path=common.TASK_YAML) == 0
    dirs = [common.processed_dir(config, tag) for tag in config["split_tags"]]
    state, detail = prepare_all.processed_state(dirs, config["splits"])
    assert state == prepare_all.PROBLEM and "partly built" in detail            # one tag of three: never silently accepted
    assert common.build_processed(config, splits=list(config["splits"]), tags=["seed1", "seed2"],
                                  config_path=common.TASK_YAML) == 0
    assert prepare_all.processed_state(dirs, config["splits"])[0] == prepare_all.DONE
    lines = prepare_all.verify_processed(dirs, common.task_paths(config)["provenance"])
    assert len(lines) == 3 and all("recorded file(s) match their sha256" in line for line in lines)
    (dirs[1] / "qc_test.tsv").write_text("changed\n")                          # a change in one tag is found in that tag
    with pytest.raises(RuntimeError, match="seed1"):
        prepare_all.verify_processed(dirs, common.task_paths(config)["provenance"])
