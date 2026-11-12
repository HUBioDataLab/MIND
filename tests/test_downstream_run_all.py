"""``downstream.run_all`` (``make ds-run``): plan, skips, failures, log names, GPU slots, model identity.

No GPU and no real embed/probe: the embed and probe commands are a fake script (written per test) that takes the
same arguments and writes the files ``downstream.embed`` / ``downstream.probe`` would (shapes from
``ledger_helpers``). The ledger step and the summary are the real modules. Task configs are stubbed by
``describe`` (one test reads the real task.yaml files, without data).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from downstream import ledger, run_all
from downstream.common import registry
from ledger_helpers import card

REPO = Path(__file__).resolve().parents[1]
FAKE = r'''
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
from downstream.common import registry
import ledger_helpers as H

kind = sys.argv[1]
p = argparse.ArgumentParser()
for name in ("--threads", "--outputs-root", "--task", "--model", "--model-id", "--device", "--head", "--split-tag",
             "--representation"):
    p.add_argument(name)
p.add_argument("--random-init", action="store_true")
p.add_argument("--splits", nargs="+")
p.add_argument("--seeds", nargs="+", type=int)
a = p.parse_args(sys.argv[2:])
start = time.time()
print(f"fake {kind} pid {os.getpid()}", flush=True)
time.sleep(float(os.environ.get("FAKE_SLEEP", "0.2")))
event = {"kind": kind, "task": a.task, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "pid": os.getpid(),
         "start": start, "splits": a.splits, "seeds": a.seeds, "split_tag": a.split_tag}
if a.task in os.environ.get("FAKE_FAIL", "").split(","):
    print(f"boom {a.task}", flush=True)
    sys.exit(3)
root, sha = Path(a.outputs_root), os.environ["FAKE_SHA"]
prefix = f"{a.split_tag}.{a.representation}." if a.split_tag else ""
if kind == "embed" and a.task not in os.environ.get("FAKE_SILENT", "").split(","):
    model_id = a.model + (".random_init" if a.random_init else "")
    for split in a.splits:
        out = registry.embeddings_path(a.task, model_id, prefix + split, root=root)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, chains=np.array(["x"]))
        out.with_suffix(".json").write_text(json.dumps(H.embedding_record(a.task, model_id, split, sha=sha)))
if kind == "probe":
    for seed in a.seeds:
        if a.task == "ec":
            metrics, splits = H.ec_metrics(a.model_id, seed, 0.5, protocol=f"{a.head}_probe"), ("train", "valid", "test")
        elif a.task == "lba":
            metrics, splits = H.lba_metrics(a.model_id, seed, 1.5, split_tag=a.split_tag), ("train", "val", "test")
            metrics["protocol"] = f"{a.head}_probe.{a.split_tag}.joint"
        else:
            metrics, splits = H.molnet_cls_metrics(a.model_id, seed, 0.8), ("train", "valid", "test")
            metrics.update(task=a.task, protocol=f"{a.head}_probe")
        H.write_probe(root, metrics, splits=splits, sha=sha)
event["end"] = time.time()
with open(os.environ["FAKE_EVENTS"], "a") as handle:
    handle.write(json.dumps(event) + "\n")
'''
LOG_NAME = re.compile(r"^(embed|probe_linear|probe_mlp|ledger_add|ledger_report)_([A-Za-z0-9_.]+)_(md4)_(\d{8}_\d{6}_\d{3})_(\d+)\.log$")


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Outputs root with snapshot md4 (all four domains), processed split files, fake commands, results root."""
    outputs = tmp_path / "outputs"
    models = outputs / "models" / "md4"
    models.mkdir(parents=True)
    (models / "model.ckpt").write_bytes(b"fake checkpoint")
    sha = hashlib.sha256(b"fake checkpoint").hexdigest()
    card(outputs, "md4", sha)
    _set_domains(outputs, "md4", ["metabolite", "pdb", "rna", "unimol"])
    fake = tmp_path / "fake_job.py"
    fake.write_text(FAKE)
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("FAKE_SHA", sha)
    monkeypatch.setenv("FAKE_EVENTS", str(events))
    monkeypatch.setenv("PYTHONPATH", f"{REPO}{os.pathsep}{REPO / 'tests'}")
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    commands = {"embed": [sys.executable, str(fake), "embed"], "probe": [sys.executable, str(fake), "probe"],
                "ledger": [sys.executable, "-m", "downstream.ledger"]}
    data = tmp_path / "data"

    def describe(unit):
        modality = ("small_molecule" if unit.task.startswith("mn_") else "complex" if unit.task == "lba"
                    else "protein")
        splits = ["train", "val", "test"] if unit.task == "lba" else ["train", "valid", "test"]
        pooling = "concat_pocket_ligand" if unit.task == "lba" else "mean_atoms"
        return {"modality": modality, "splits": splits, "pooling": pooling,
                "processed": {s: data / unit.key / f"{s}.pt" for s in splits}}

    def processed(*keys):
        for key in keys:
            for split in (["train", "val", "test"] if key.startswith("lba") else ["train", "valid", "test"]):
                (data / key).mkdir(parents=True, exist_ok=True)
                (data / key / f"{split}.pt").write_bytes(b"")

    class World:
        pass
    w = World()
    w.tmp, w.outputs, w.sha, w.commands, w.describe, w.events = tmp_path, outputs, sha, commands, describe, events
    w.results, w.logs, w.processed = tmp_path / "results", tmp_path / "logs", processed

    def main(*args, **kwargs):
        argv = ["--outputs-root", str(outputs), "--results-root", str(w.results), "--log-root", str(w.logs),
                "--poll", "0.02", "--status-every", "0.1", *args]
        return run_all.main(argv, commands=commands, describe=describe, **kwargs)
    w.main = main
    return w


def _set_domains(outputs, name, domains):
    path = outputs / "models" / name / "card.json"
    data = json.loads(path.read_text())
    data["domains"] = domains
    path.write_text(json.dumps(data))


def _events(world):
    return [json.loads(line) for line in world.events.read_text().splitlines()] if world.events.exists() else []


def _write_embedding(world, task, model_id, split, *, json_too=True, seconds=None):
    out = registry.embeddings_path(task, model_id, split, root=world.outputs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"npz")
    if json_too:
        out.with_suffix(".json").write_text(json.dumps({"seconds": seconds} if seconds is not None else {}))


# -- plan --------------------------------------------------------------------------------------------------

def test_plan_lists_runs_skips_and_known_timings_and_writes_nothing(world, capsys):
    world.processed("ec", "mn_bace", "lba.identity30.joint", "lba.identity60.joint")
    for split in ("train", "valid", "test"):
        _write_embedding(world, "ec", "md4", split)  # complete: skipped
        _write_embedding(world, "mn_bace", "other_model", split, seconds=12.5)  # another model's recorded time
    for seed in (0, 1, 2):  # ec mlp probe complete: skipped; ec linear runs
        d = registry.results_dir("ec", "md4", "mlp_probe", seed, pooling="mean_atoms", root=world.outputs)
        d.mkdir(parents=True)
        (d / "metrics.json").write_text("{}")
    before = sorted(p for p in world.tmp.rglob("*"))
    assert world.main("--model", "md4", "--tasks", "ec,mn_bace,lba", "--gpus", "2,3", "--plan") == 0
    out = capsys.readouterr().out
    assert "ATLANACAK  embed ec: zaten tamam" in out
    assert "ÇALIŞACAK  embed mn_bace: splits train valid test" in out
    assert "other_model: 37.5 s over train/valid/test" in out
    assert "ÇALIŞACAK  embed lba.identity30.joint" in out and "ÇALIŞACAK  embed lba.identity60.joint" in out
    assert "ATLANACAK  probe ec mlp: zaten tamam" in out and "ÇALIŞACAK  probe ec linear: seeds 0 1 2" in out
    assert "PLAN: hiçbir şey koşulmadı" in out
    assert sorted(p for p in world.tmp.rglob("*")) == before and not world.logs.exists() and not _events(world)


def test_partial_embedding_runs_only_missing_splits_and_half_written_one_is_an_error(world, capsys):
    world.processed("ec", "mn_bace")
    _write_embedding(world, "ec", "md4", "train")
    _write_embedding(world, "mn_bace", "md4", "train", json_too=False)
    assert world.main("--model", "md4", "--tasks", "ec,mn_bace", "--gpus", "2", "--plan") == 1
    out = capsys.readouterr().out
    assert "ÇALIŞACAK  embed ec: splits valid test" in out
    assert re.search(r"HATA  embed mn_bace: half-written embedding .*train\.npz", out)


def test_missing_processed_data_is_reported_per_task(world, capsys):
    world.processed("ec")
    assert world.main("--model", "md4", "--tasks", "ec,mn_bace", "--gpus", "2", "--plan") == 1
    assert re.search(r"HATA  embed mn_bace: no processed data: .*mn_bace/train\.pt", capsys.readouterr().out)


def test_real_task_configs_describe_every_unit_of_all():
    units, explicit = run_all.expand_tasks("all", ledger.load_catalog())
    assert not explicit and len(units) == 21  # ec, 3 GO, fold, 2 LBA split tags, 14 MoleculeNet
    assert [u.key for u in units if u.task == "lba"] == ["lba.identity30.joint", "lba.identity60.joint"]
    for unit in units:
        info = run_all.describe_task(unit)
        assert info["modality"] in run_all.MODALITY_DOMAINS and info["splits"]
        assert set(info["processed"]) == set(info["splits"])
    with pytest.raises(run_all.PlanError, match="unknown task"):
        run_all.expand_tasks("ec,nope", ledger.load_catalog())


# -- domains ------------------------------------------------------------------------------------------------

def test_domain_mismatch_is_skipped_under_all_and_refused_when_named(world, capsys):
    _set_domains(world.outputs, "md4", ["pdb"])
    assert world.main("--model", "md4", "--tasks", "all", "--gpus", "2", "--plan") == 1  # no processed data
    out = capsys.readouterr().out
    assert out.count("ATLANDI  mn_") == 14 and "small_molecule ['unimol'] ister" in out
    assert "lba.identity30.joint: ligand atoms, but md4 saw no small-molecule domain" in out
    assert world.main("--model", "md4", "--tasks", "ec,mn_esol", "--gpus", "2", "--plan") == 2
    assert "mn_esol: alan uyumsuz" in capsys.readouterr().err


# -- model identity -----------------------------------------------------------------------------------------

def _tiny_ckpt(path: Path, epoch=3) -> Path:
    import torch
    config = {"multi_domain": {"datasets": {"pdb": {"enabled": True}, "unimol": {"enabled": True}}},
              "unimol_masked_geometry_hiding": {"enabled": True, "bond_threshold_angstrom": 2.1}}
    torch.save({"hyper_parameters": {"config": config}, "epoch": epoch, "global_step": 10}, path)
    return path


def test_checkpoint_path_snapshot_and_name_conflicts(world, capsys):
    world.processed("ec")
    ckpt = _tiny_ckpt(world.tmp / "best.ckpt")
    # same name as an existing snapshot with another checkpoint: refused before anything runs
    assert world.main("--model", str(ckpt), "--name", "md4", "--tasks", "ec", "--gpus", "2", "--plan") == 2
    assert "one name must name one checkpoint" in capsys.readouterr().err
    assert world.main("--model", str(ckpt), "--tasks", "ec", "--gpus", "2", "--plan") == 2
    assert "needs NAME" in capsys.readouterr().err
    # the ledger already knows the name with another checkpoint
    ledger.append_records(world.results, "models", [{"model_key": "g1", "sha256": "c" * 64}])
    assert world.main("--model", str(ckpt), "--name", "g1", "--tasks", "ec", "--gpus", "2", "--plan") == 2
    assert "results ledger has 'g1'" in capsys.readouterr().err
    # a new name: the plan says it will snapshot and writes nothing; the run creates it with the same sha256
    assert world.main("--model", str(ckpt), "--name", "g1new", "--tasks", "ec", "--gpus", "2", "--plan") == 0
    out = capsys.readouterr().out
    assert "create snapshot" in out and "unimol_masked_geometry_hiding = {'enabled': True" in out
    assert not (world.outputs / "models" / "g1new").exists()
    world.main("--model", str(ckpt), "--name", "g1new", "--tasks", "ec", "--gpus", "2", "--heads", "mlp")
    card_data = json.loads((world.outputs / "models" / "g1new" / "card.json").read_text())
    assert card_data["sha256"] == hashlib.sha256(ckpt.read_bytes()).hexdigest()
    # the same checkpoint and name again: the snapshot is reused, not an error
    capsys.readouterr()
    assert world.main("--model", str(ckpt), "--name", "g1new", "--tasks", "ec", "--gpus", "2", "--plan") == 0
    assert "snapshot exists (same checkpoint)" in capsys.readouterr().out


@pytest.mark.parametrize("args, message", [
    (["--model", "esm2_t6_8M_UR50D"], "HF baseline"),
    (["--model", "md4", "--name", "other"], "only for a checkpoint path"),
    (["--model", "nosuch"], "no snapshot or alias"),
    (["--model", "md4", "--heads", "mlp,cnn"], "HEADS="),
    (["--model", "md4", "--gpus", "auto"], "ds-run does not pick GPUs"),
])
def test_bad_arguments_stop_before_anything(world, capsys, args, message):
    world.processed("ec")
    assert world.main(*args, "--tasks", "ec") == 2
    assert message in capsys.readouterr().err and not world.logs.exists()


# -- execution ----------------------------------------------------------------------------------------------

def test_failures_are_collected_others_continue_and_exit_is_nonzero(world, capsys, monkeypatch):
    world.processed("ec", "mn_bace", "lba.identity30.joint", "lba.identity60.joint")
    monkeypatch.setenv("FAKE_FAIL", "mn_bace")
    code = world.main("--model", "md4", "--tasks", "ec,mn_bace,lba", "--gpus", "5,6", "--per-gpu", "2",
                      "--heads", "linear,mlp")
    out = capsys.readouterr().out
    assert code == 1
    run_dir = next(world.logs.glob("ds_run_md4_*"))
    summary = (run_dir / "summary.md").read_text()
    assert re.search(r"- embed mn_bace: exit code 3; log .*embed_mn_bace_md4_.*\.log; last line: boom mn_bace", summary)
    assert "- probe_linear mn_bace: not run: the embedding of this task failed" in summary
    assert "EKSİK: 3 iş" in summary  # embed + two probes of mn_bace
    rows = [line for line in summary.splitlines() if line.startswith("| ") and not line.startswith("| görev")]
    assert len(rows) == 8 and all(line.rstrip(" |").split("|")[-1].strip() for line in rows)  # no empty cell
    assert re.search(r"\| ec \| mlp_probe \| fmax \(test/all\) \| 0\.500 ± 0\.000 \(n_tohum=3", summary)
    assert re.search(r"\| mn_bace \| mlp_probe \| roc_auc \(test/all\) \| HATA: embed: exit code 3", summary)
    assert re.search(r"\| lba.identity60.joint \| mlp_probe.identity60.joint \| rmse \(test/all\) \| 1\.500", summary)
    probes = [e for e in _events(world) if e["kind"] == "probe"]
    assert sorted((e["task"], e["split_tag"]) for e in probes) == sorted(
        [("ec", None)] * 2 + [("lba", "identity30")] * 2 + [("lba", "identity60")] * 2)
    assert all(e["gpu"] == "" for e in probes)
    assert "durum: koşan" in out and "son yazma" in out  # status lines of the running jobs
    assert ledger.verify(world.results)["lines"]["measurements"] > 0
    # a second run skips what is complete and retries only what failed
    monkeypatch.delenv("FAKE_FAIL")
    capsys.readouterr()
    assert world.main("--model", "md4", "--tasks", "ec,mn_bace,lba", "--gpus", "5", "--heads", "linear,mlp") == 0
    out = capsys.readouterr().out
    assert "ÇALIŞACAK  embed mn_bace" in out and "ATLANACAK  embed ec: zaten tamam" in out
    assert "ATLANACAK  probe lba.identity30.joint mlp: zaten tamam" in out


def test_exit_zero_without_outputs_is_a_failure(world, monkeypatch):
    world.processed("ec")
    monkeypatch.setenv("FAKE_SILENT", "ec")
    assert world.main("--model", "md4", "--tasks", "ec", "--gpus", "1", "--heads", "mlp") == 1
    summary = next(world.logs.glob("ds_run_md4_*/summary.md")).read_text()
    assert "- embed ec: exit code 0 but embedding files missing" in summary


def test_log_names_are_unique_and_carry_the_child_pid(world):
    world.processed("ec", "mn_bace", "mn_esol", "mn_hiv", "lba.identity30.joint", "lba.identity60.joint")
    world.main("--model", "md4", "--tasks", "ec,mn_bace,mn_esol,mn_hiv,lba", "--gpus", "1", "--per-gpu", "6",
               "--probe-jobs", "4", "--heads", "linear,mlp")
    logs = sorted(p.name for p in next(world.logs.glob("ds_run_md4_*")).glob("*.log"))
    assert len(logs) == len(set(logs)) == 6 + 12 + 1 + 1  # + the final ledger report
    pids = {e["pid"] for e in _events(world)}
    for name in logs:
        match = LOG_NAME.match(name)
        assert match, name
        if match.group(1) not in ("ledger_add", "ledger_report"):
            assert int(match.group(5)) in pids
    assert not list(next(world.logs.glob("ds_run_md4_*")).glob(".starting_*"))


def test_each_gpu_runs_at_most_per_gpu_jobs(world, monkeypatch):
    keys = ("ec", "mn_bace", "mn_esol", "mn_hiv", "mn_bbbp", "mn_tox21")
    world.processed(*keys)
    monkeypatch.setenv("FAKE_SLEEP", "0.4")
    world.main("--model", "md4", "--tasks", ",".join(keys), "--gpus", "3,4", "--per-gpu", "2", "--heads", "mlp")
    embeds = [e for e in _events(world) if e["kind"] == "embed"]
    assert len(embeds) == 6 and {e["gpu"] for e in embeds} == {"3", "4"}
    for gpu in ("3", "4"):
        spans = [(e["start"], e["end"]) for e in embeds if e["gpu"] == gpu]
        peak = max(sum(1 for s, e in spans if s <= t < e) for t, _ in spans)
        assert peak <= 2
    first_probe = min(e["start"] for e in _events(world) if e["kind"] == "probe")
    assert first_probe >= max(e["end"] for e in embeds)  # probes start after every embed has ended


def test_random_init_uses_its_own_model_id(world):
    world.processed("ec")
    world.main("--model", "md4", "--random-init", "--tasks", "ec", "--gpus", "1", "--heads", "mlp")
    assert (world.outputs / "downstream" / "ec" / "md4.random_init" / "embeddings" / "test.json").is_file()
    assert (world.outputs / "downstream" / "ec" / "md4.random_init" / "mlp_probe.mean_atoms" / "seed2"
            / "metrics.json").is_file()


# -- Make --------------------------------------------------------------------------------------------------

def _make(*args, makefile=None):
    command = ["make", "--no-print-directory", "-n", *(["-f", str(makefile)] if makefile else []), *args]
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    return re.sub(r"\d{8}_\d{6}", "<stamp>", " ".join(result.stdout.replace("\\\n", " ").split())), result


def test_ds_run_make_target(tmp_path):
    text, result = _make("ds-run", "MODEL=g1", "TASKS=ec,lba", "GPU=2,3", "PER_GPU=3", "PLAN=1")
    assert result.returncode == 0, result.stderr
    assert ('python -u -m downstream.run_all --model "g1" --tasks ec,lba --gpus "2,3" --per-gpu 3 --heads linear,mlp '
            "--seeds 0 1 2 --probe-jobs 1 --threads 8 --log-root logs/") in text and text.endswith("--plan")
    assert 'MIND_RESULTS_ROOT="/home/yusuf/mind_results"' in text
    text, _ = _make("ds-run", "MODEL=/x/b.ckpt", "NAME=g1", "GPU=2", "RANDOM_INIT=1", "DETACH=1", "OUTPUTS_ROOT=/o")
    assert '--name "g1"' in text and "--random-init" in text and '--outputs-root "/o"' in text
    assert "setsid nohup" in text and "> $log 2>&1 < /dev/null &" in text
    guard = subprocess.run(["make", "--no-print-directory", "ds-run"], cwd=REPO, capture_output=True, text=True)
    assert guard.returncode != 0 and "set MODEL" in guard.stdout + guard.stderr


@pytest.mark.parametrize("args", [
    ["ds-embed", "TASK=ec", "MODEL=m", "GPU=1"], ["ds-embed", "TASK=lba", "MODEL=m", "GPU=1", "SPLIT_TAG=identity60"],
    ["ds-probe", "TASK=ec", "MODEL_ID=m", "HEAD=mlp", "FOREGROUND=1"], ["ds-finetune", "TASK=ec", "MODEL=m", "GPU=1"],
    ["ds-snapshot", "CKPT=c", "NAME=n"], ["ds-report", "TASK=ec"], ["ds-ledger-add", "OUTPUTS_ROOT=/o"],
    ["ds-compare", "A=a", "B=b"], ["ds-summary"], ["ds-missing"], ["ds-ledger-verify"],
])
def test_existing_targets_are_unchanged(tmp_path, args):
    head = tmp_path / "Makefile.HEAD"
    head.write_text(subprocess.run(["git", "show", "HEAD:Makefile"], cwd=REPO, capture_output=True, text=True,
                                   check=True).stdout)
    assert _make(*args)[0] == _make(*args, makefile=head)[0]


# -- COMPARE_TO, final ledger report, COUNTERFACTUAL ---------------------------------------------------------

SHA_WRAPPER = r'''
import json, os, runpy, sys
from pathlib import Path
fake, args = sys.argv[1], sys.argv[3:]
value = lambda flag: args[args.index(flag) + 1] if flag in args else None
model = (value("--model") or value("--model-id")).removesuffix(".random_init")
card = Path(value("--outputs-root")) / "models" / model / "card.json"
os.environ["FAKE_SHA"] = json.loads(card.read_text())["sha256"]  # each model's results carry its own checkpoint sha
with open(os.environ["MODEL_EVENTS"], "a") as handle:
    handle.write(json.dumps({"kind": sys.argv[2], "model": model, "random_init": "--random-init" in args,
                             "model_id": value("--model-id")}) + "\n")
sys.argv = [fake] + sys.argv[2:]
runpy.run_path(fake, run_name="__main__")
'''
FAKE_CF = r'''
import argparse, hashlib, json, sys
from pathlib import Path
p = argparse.ArgumentParser()
for name in ("--threads", "--ckpt", "--split", "--seed", "--output-dir", "--device", "--domains", "--counterfactual",
             "--config-yaml-path"):
    p.add_argument(name)
a = p.parse_args()
if a.counterfactual == "masked_pos_random":
    print("cf boom", flush=True)
    sys.exit(4)
out = Path(a.output_dir)
out.mkdir(parents=True)
sha = hashlib.sha256(Path(a.ckpt).read_bytes()).hexdigest()
summary = {"n_masked": 10, "top1_acc_real": 0.5, "top3_acc_real": 0.7, "ce_real": 1.0, "ece_real": 0.1,
           "top1_acc_virtual": None}
prov = {"checkpoint": {"path": a.ckpt, "sha256": sha, "epoch": 3}, "git": {"commit": "f" * 40, "dirty": False},
        "split": a.split, "seed": int(a.seed), "masking_protocol": {"mlm_mask_ratio": 0.15},
        "counterfactual": {"mode": a.counterfactual, "domains": ["unimol"]}}
(out / "metrics.json").write_text(json.dumps({"provenance": prov, "plan": {}, "losses": {}, "mask_digest": {},
                                               "domain_mlm": {"unimol": {"batches": 1, "summary": summary}},
                                               "counterfactual": {"mode": a.counterfactual}}))
'''


@pytest.fixture
def two_models(world, monkeypatch):
    """md4 (snapshot) plus a new checkpoint file; the fake jobs write each model's own checkpoint sha."""
    wrapper = world.tmp / "sha_wrapper.py"
    wrapper.write_text(SHA_WRAPPER)
    cf = world.tmp / "fake_cf.py"
    cf.write_text(FAKE_CF)
    monkeypatch.setenv("MODEL_EVENTS", str(world.tmp / "model_events.jsonl"))
    fake = world.commands["embed"][1]
    world.commands.update(embed=[sys.executable, str(wrapper), fake, "embed"],
                          probe=[sys.executable, str(wrapper), fake, "probe"], cf=[sys.executable, str(cf)])
    world.ckpt = _tiny_ckpt(world.tmp / "new.ckpt")
    world.processed("ec", "mn_bace")
    world.model_events = lambda: ([json.loads(line) for line in (world.tmp / "model_events.jsonl").read_text()
                                   .splitlines()] if (world.tmp / "model_events.jsonl").exists() else [])
    return world


def test_compare_to_completes_every_model_skips_done_and_compares(two_models, capsys):
    w = two_models
    assert w.main("--model", "md4", "--tasks", "ec,mn_bace", "--gpus", "2", "--heads", "mlp") == 0  # md4 complete
    done_before = len(w.model_events())
    capsys.readouterr()
    code = w.main("--model", str(w.ckpt), "--name", "newm", "--compare-to", "md4,md4.random_init",
                  "--tasks", "ec,mn_bace", "--gpus", "2,3", "--heads", "mlp", "--bootstrap", "20")
    out = capsys.readouterr().out
    assert code == 0, out
    events = w.model_events()[done_before:]
    assert not [e for e in events if e["model"] == "md4" and not e["random_init"] and e["model_id"] != "md4.random_init"]
    assert {(e["kind"], e["model"]) for e in events if e["kind"] == "embed"} == {("embed", "newm"), ("embed", "md4")}
    assert all(e["random_init"] for e in events if e["kind"] == "embed" and e["model"] == "md4")
    assert "Model: md4 (model_id md4)" in out and "ATLANACAK  embed ec: zaten tamam" in out
    run_dir = next(w.logs.glob("ds_run_newm_*"))
    names = sorted(p.name for p in run_dir.glob("*.log"))
    assert any(n.startswith("compare_md4__vs__newm_") for n in names), names
    assert any(n.startswith("compare_md4.random_init__vs__newm_") for n in names), names
    assert any(n.startswith("ledger_report_all_") for n in names), names
    assert any(n.startswith("embed_ec_md4.random_init_") for n in names), names  # log names carry their own model
    summary = (run_dir / "summary.md").read_text()
    for heading in ("# ds-run: newm", "# ds-run: md4\n", "# ds-run: md4.random_init"):
        assert heading in summary, heading
    assert "## EKSİK" not in summary  # no failed job (missing catalog cells are listed per model, that is fine)
    assert (w.results / "reports" / "ozet_hoca.md").is_file()
    assert (w.results / "reports" / "compare" / "md4__vs__newm.md").is_file()
    models = {row["model_key"] for row in ledger.read_records(w.results, "models")}
    assert {"newm", "md4"} <= models


def test_compare_to_plan_lists_everything_and_writes_nothing(two_models, capsys):
    w = two_models
    before = sorted(p for p in w.tmp.rglob("*"))
    code = w.main("--model", str(w.ckpt), "--name", "newm", "--compare-to", "md4,md4.random_init", "--tasks",
                  "ec,mn_bace", "--gpus", "2", "--counterfactual", "--cf-modes", "none", "masked_pos_noise", "--plan")
    out = capsys.readouterr().out
    assert code == 0, out
    assert sorted(p for p in w.tmp.rglob("*")) == before and not w.logs.exists()
    assert "Model: newm (model_id newm)" in out and "Model: md4 (model_id md4.random_init)" in out
    assert "ÇALIŞACAK  compare A=md4 B=newm" in out and "ÇALIŞACAK  compare A=md4.random_init B=newm" in out
    assert "ÇALIŞACAK  counterfactual none" in out and "ÇALIŞACAK  counterfactual masked_pos_noise" in out
    assert "ledger report" in out and "PLAN: hiçbir şey koşulmadı" in out


def test_a_failing_compare_model_job_is_listed_and_the_rest_goes_on(two_models, capsys, monkeypatch):
    w = two_models
    monkeypatch.setenv("FAKE_FAIL", "mn_bace")
    code = w.main("--model", str(w.ckpt), "--name", "newm", "--compare-to", "md4", "--tasks", "ec,mn_bace",
                  "--gpus", "2", "--heads", "mlp", "--bootstrap", "20")
    assert code == 1
    summary = (next(w.logs.glob("ds_run_newm_*")) / "summary.md").read_text()
    assert re.search(r"- embed mn_bace: exit code 3", summary)  # the main model keeps its labels
    assert re.search(r"- \[md4\] embed mn_bace: exit code 3", summary)
    assert re.search(r"\| ec \| mlp_probe \| fmax \(test/all\) \| 0\.500", summary)
    assert (w.results / "reports" / "ozet_hoca.md").is_file()


@pytest.mark.parametrize("args, message", [
    (["--model", "md4", "--compare-to", "nosuch"], "no snapshot or alias"),
    (["--model", "md4", "--compare-to", "md4"], "is the model itself"),
    (["--model", "md4", "--counterfactual"], "needs MODEL=<path/to.ckpt>"),
])
def test_bad_compare_and_counterfactual_arguments_stop_before_anything(world, capsys, args, message):
    world.processed("ec")
    assert world.main(*args, "--tasks", "ec", "--gpus", "2") == 2
    assert message in capsys.readouterr().err and not world.logs.exists()


def test_counterfactual_runs_every_mode_goes_on_after_a_failure_and_adds_to_the_ledger(two_models, capsys):
    w = two_models
    code = w.main("--model", str(w.ckpt), "--name", "newm", "--tasks", "ec", "--gpus", "2,3", "--heads", "mlp",
                  "--counterfactual", "--cf-modes", "none", "masked_pos_noise", "masked_pos_random")
    out = capsys.readouterr().out
    assert code == 1
    done = sorted(p.name for p in (w.outputs / "eval").glob("cf_newm_*/*") if (p / "metrics.json").is_file())
    assert done == ["masked_pos_noise", "none"]
    assert re.search(r"HATA counterfactual newm\.masked_pos_random .*exit code 4", out)
    rows = [r for r in ledger.read_records(w.results, "measurements") if r["task"] == "counterfactual"]
    assert {r["model_key"] for r in rows} == {"newm"} and {r["protocol_id"] for r in rows} == {"none", "masked_pos_noise"}
    summary = (next(w.logs.glob("ds_run_newm_*")) / "summary.md").read_text()
    assert "counterfactual newm.masked_pos_random" in summary


def test_ds_run_make_target_compare_and_counterfactual():
    text, result = _make("ds-run", "MODEL=/x/b.ckpt", "NAME=n", "GPU=2", "COMPARE_TO=a,a.random_init",
                         "COUNTERFACTUAL=1", "PLAN=1")
    assert result.returncode == 0, result.stderr
    assert ('--compare-to "a,a.random_init" --bootstrap 1000 --counterfactual --cf-split test --cf-seed 0 '
            "--cf-domains unimol --cf-modes none masked_pos_noise masked_pos_random") in text
    assert text.endswith("--plan")
    text, _ = _make("ds-run", "MODEL=/x/b.ckpt", "NAME=n", "GPU=2", "COUNTERFACTUAL=1", "EVAL_CONFIG=cfg")
    assert "--cf-eval-config core/cfg.yaml" in text
    head = subprocess.run(["git", "show", "HEAD:Makefile"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    assert "eval-counterfactual:" in head  # the sweep ds-run mirrors exists in the Makefile it ships with


# -- detached by default: closing the terminal does not stop ds-probe / ds-run ---------------------------------

FAKE_PYTHON = """#!/bin/bash
echo "fake python $*"
ps -o sid=,pgid=,tty= -p $$ > "$FAKE_INFO.tmp"
echo "$$" >> "$FAKE_INFO.tmp"
mv "$FAKE_INFO.tmp" "$FAKE_INFO"
sleep "${FAKE_SLEEP:-1}"
exit "${FAKE_EXIT:-0}"
"""


def _wait_for(path: Path, seconds: float = 20.0) -> None:
    import time
    deadline = time.monotonic() + seconds
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.05)


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


@pytest.mark.parametrize("target, exit_code", [
    (["ds-probe", "TASK=ec", "MODEL_ID=m", "HEAD=mlp"], 0),
    (["ds-probe", "TASK=ec", "MODEL_ID=m", "HEAD=mlp"], 3),
    (["ds-run", "MODEL=m", "GPU=2"], 0),
])
def test_ds_jobs_survive_the_terminal_hangup_and_log_their_exit_code(tmp_path, target, exit_code):
    import signal
    import time
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text(FAKE_PYTHON)
    (bin_dir / "python").chmod(0o755)
    info, logs = tmp_path / "info.txt", tmp_path / "logs"
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "FAKE_INFO": str(info),
           "FAKE_SLEEP": "2", "FAKE_EXIT": str(exit_code)}
    # the "terminal": a shell in its own session running make, then lingering like an interactive login would
    make = " ".join(["make", "--no-print-directory", *target, "CONDA_RUN=", f"LOG_MONTH_DIR={logs}"])
    terminal = subprocess.Popen(["bash", "-c", f"{make}; sleep 60"], cwd=REPO, env=env, start_new_session=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        _wait_for(info)
        sid, pgid, tty, pid = info.read_text().split()
        pid = int(pid)
        assert int(sid) != os.getsid(terminal.pid) and int(pgid) != terminal.pid  # its own session and group
        assert tty == "?"  # no controlling terminal
        os.killpg(terminal.pid, signal.SIGHUP)  # the SSH connection closes
        terminal.wait(10)
        assert terminal.returncode == -signal.SIGHUP
        assert _alive(pid), "the detached job died with the terminal"
        deadline = time.monotonic() + 20
        while _alive(pid):
            assert time.monotonic() < deadline
            time.sleep(0.05)
    finally:
        if terminal.poll() is None:
            os.killpg(terminal.pid, signal.SIGKILL)
    log, = logs.glob("ds_*.log")
    deadline = time.monotonic() + 10
    while "EXIT_CODE=" not in log.read_text():
        assert time.monotonic() < deadline
        time.sleep(0.05)
    lines = log.read_text().splitlines()
    assert lines[0].startswith("fake python -u -m downstream.") and lines[-1] == f"EXIT_CODE={exit_code}"
    status = subprocess.run(["make", "--no-print-directory", "ds-status", f"LOG_MONTH_DIR={logs}"], cwd=REPO,
                            capture_output=True, text=True)
    assert f"EXIT_CODE={exit_code}  {log}" in status.stdout


def test_ds_run_is_detached_by_default_and_plan_or_foreground_stay_in_the_terminal():
    text, _ = _make("ds-run", "MODEL=m", "GPU=2")
    assert "setsid nohup bash -c" in text and "EXIT_CODE=$code" in text and "> $log 2>&1 < /dev/null &" in text
    for extra in ("PLAN=1", "FOREGROUND=1"):
        text, _ = _make("ds-run", "MODEL=m", "GPU=2", extra)
        assert "setsid" not in text and "python -u -m downstream.run_all" in text
    text, _ = _make("ds-probe", "TASK=ec", "MODEL_ID=m", "HEAD=mlp", "FOREGROUND=1")
    assert "set -o pipefail;" in text and "| tee logs/" in text and "setsid" not in text
