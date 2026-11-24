"""``downstream.prepare_all`` (``make ds-prepare``): selection, shared steps, states, idempotence, failures, disk.

CPU only, no network and no real data. The planner is driven by ``TaskInfo`` objects built on a temporary tree
(fake archives with real checksums, fake processed files); the commands of the todo steps are replaced by a runner
that writes the outputs a prepare CLI would. Two tests read the real registry and task.yaml files (no data needed).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from downstream import prepare_all
from downstream.common.prepare_spec import GIB, Archive, DownloadSpec, verify_archive
from downstream.prepare_all import PrepareError, TaskInfo

REPO = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")


def _archive(root: Path, name: str, payload: bytes = b"archive-bytes") -> Archive:
    return Archive(root / "raw" / name, f"https://example.invalid/{name}", len(payload),
                   md5=hashlib.md5(payload).hexdigest(), sha256=hashlib.sha256(payload).hexdigest())


def _info(root: Path, name: str, family: str, *, archive: str = None, processed: str = None, unavailable=None,
          spec: bool = True, gib: float = 1.0) -> TaskInfo:
    archive_obj = _archive(root, archive or f"{family}.zip")
    done_marker = root / "raw" / f"{archive or family}_extracted"
    return TaskInfo(
        name=name, family=family, module=f"fake.tasks.{family}.prepare",
        spec=DownloadSpec((archive_obj,), (done_marker,), unavailable=unavailable, extracted_gib=gib) if spec else None,
        processed=(root / "processed" / (processed or name),), splits=SPLITS, provenance=root / f"{family}_prov.json",
        processed_gib=0.5,
        cli_args=lambda step, task, workers, threads: ["--task", task] if family == "mn" else
        (["--workers", str(workers)] if step == "process" else []))


def _tasks(root: Path, *, reaction_unavailable: bool = True) -> list:
    return [
        _info(root, "ec", "ec"),
        _info(root, "go_mf", "go", archive="go.zip", processed="go"),
        _info(root, "go_bp", "go", archive="go.zip", processed="go"),
        _info(root, "mn_a", "mn", archive="mn.tar"),
        _info(root, "mn_b", "mn", archive="mn.tar"),
        _info(root, "reaction", "reaction", unavailable="source returns HTTP 404" if reaction_unavailable else None),
    ]


def _finish_download(info: TaskInfo, payload: bytes = b"archive-bytes") -> None:
    for archive in info.spec.archives:
        archive.path.parent.mkdir(parents=True, exist_ok=True)
        archive.path.write_bytes(payload)
    for output in info.spec.outputs:
        output.mkdir(parents=True, exist_ok=True)


def _finish_process(info: TaskInfo) -> None:
    for directory in info.processed:
        directory.mkdir(parents=True, exist_ok=True)
        for split in info.splits:
            (directory / f"{split}.pt").write_bytes(b"split-" + split.encode())


def _plan(infos, explicit=(), **kwargs):
    return prepare_all.build_plan(infos, [], explicit, free_bytes=lambda path: 10 ** 15, **kwargs)


FAMILIES = {"ec": "ec", "go_mf": "go", "go_bp": "go", "mn_a": "mn", "mn_b": "mn", "reaction": "reaction"}
REGISTRY = tuple(FAMILIES)


# ---------------------------------------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------------------------------------

def test_select_all_family_names_and_exclude():
    def pick(text, exclude=""):
        return prepare_all.select(text, exclude, registry=REGISTRY, families=FAMILIES)

    assert pick("all")[0] == list(REGISTRY) and pick("all")[2] == set()  # nothing named explicitly
    assert pick("mn")[0] == ["mn_a", "mn_b"] and pick("mn")[2] == {"mn_a", "mn_b"}
    assert pick("ec,go_bp")[0] == ["ec", "go_bp"]
    assert pick("ec go")[0] == ["ec", "go_mf", "go_bp"]  # spaces work like commas (Make EXCLUDE="a b")
    selected, excluded, explicit = pick("all", "reaction mn_b")
    assert selected == ["ec", "go_mf", "go_bp", "mn_a"] and excluded == ["mn_b", "reaction"]
    assert pick("mn,ec", "mn")[0] == ["ec"]
    assert pick("all,reaction")[2] == {"reaction"}  # named on top of ``all``: explicit


@pytest.mark.parametrize("tasks, exclude, message", [
    ("ec,nope", "", r"unknown task or family name\(s\) \['nope'\]"),
    ("all", "typo", r"\['typo'\]"),
    ("", "", "--tasks is empty"),
    ("mn", "mn_a mn_b", "no task left"),
])
def test_select_refuses_bad_input(tasks, exclude, message):
    with pytest.raises(PrepareError, match=message):
        prepare_all.select(tasks, exclude, registry=REGISTRY, families=FAMILIES)


# ---------------------------------------------------------------------------------------------------------
# Plan: sharing, order, states
# ---------------------------------------------------------------------------------------------------------

def test_fresh_tree_shares_downloads_and_orders_downloads_first(tmp_path):
    plan = _plan(_tasks(tmp_path))
    assert [(s.kind, s.tasks) for s in plan.steps] == [
        ("download", ("ec",)), ("download", ("go_mf", "go_bp")), ("download", ("mn_a", "mn_b")),
        ("download", ("reaction",)),
        ("process", ("ec",)), ("process", ("go_mf", "go_bp")), ("process", ("mn_a",)), ("process", ("mn_b",)),
        ("process", ("reaction",))]
    kinds = [s.kind for s in plan.steps]
    assert kinds == sorted(kinds, key=lambda k: k != "download")
    states = {s.label: s.state for s in plan.steps}
    assert states["download ec"] == states["download mn_a,mn_b"] == "todo"
    assert states["download reaction"] == "unavailable" and states["process reaction"] == "blocked"
    assert plan.errors == []  # unavailable only because it came from "all"
    by_label = {s.label: s for s in plan.steps}
    assert by_label["process mn_b"].command[-2:] == ("--task", "mn_b")
    assert by_label["process ec"].command[-2:] == ("--workers", "16")
    assert by_label["download ec"].command[3:] == ("fake.tasks.ec.prepare", "download")
    assert by_label["download mn_a,mn_b"].command[3:6] == ("fake.tasks.mn.prepare", "download", "--task")
    # fetch: ec 13 B + go 13 B + mn 13 B are tiny; the extracted estimate dominates the need (1 GiB each x 3)
    assert plan.disk and plan.disk[0][1] >= 3 * GIB


def test_unavailable_is_an_error_only_when_named(tmp_path):
    infos = _tasks(tmp_path)
    assert _plan(infos).errors == []
    plan = _plan(infos, explicit={"reaction"})
    assert len(plan.errors) == 1 and "requested by name (reaction)" in plan.errors[0]
    assert "source returns HTTP 404" in plan.errors[0]
    # once the source is available again the same request plans normally
    assert _plan(_tasks(tmp_path, reaction_unavailable=False), explicit={"reaction"}).errors == []


def test_finished_tree_is_all_done_and_rerun_runs_nothing(tmp_path):
    infos = _tasks(tmp_path, reaction_unavailable=False)
    for info in infos:
        _finish_download(info)
        _finish_process(info)
    plan = _plan(infos)
    assert {s.state for s in plan.steps} == {"done"} and plan.errors == [] and plan.disk == []
    lines = []
    code = prepare_all.execute(plan, run=lambda command: pytest.fail(f"must not run {command}"), say=lines.append)
    assert code == 0
    text = "\n".join(lines)
    assert "verified go.zip: 13 bytes, md5+sha256 match" in text
    assert "no recorded checksums" in text  # existence only: no provenance written in this fixture


def test_done_download_with_wrong_checksum_fails_loudly(tmp_path):
    info = _info(tmp_path, "ec", "ec")
    _finish_download(info)
    _finish_process(info)
    info.spec.archives[0].path.write_bytes(b"archive-BYTES")  # same size, other content
    lines = []
    code = prepare_all.execute(_plan([info]), run=lambda c: pytest.fail("no run"), say=lines.append)
    assert code == 1 and any("FAILED verification" in line and "md5" in line for line in lines)
    assert info.spec.archives[0].path.read_bytes() == b"archive-BYTES"  # never deleted or replaced


def test_raw_data_is_not_needed_once_processed_outputs_exist(tmp_path):
    info = _info(tmp_path, "ec", "ec")
    _finish_process(info)  # raw archive removed after processing
    plan = _plan([info])
    assert [(s.kind, s.state) for s in plan.steps] == [("download", "unneeded"), ("process", "done")]
    assert plan.errors == [] and prepare_all.execute(plan, run=lambda c: pytest.fail("no run"), say=lambda t: None) == 0


def test_unavailable_download_with_finished_outputs_is_not_unavailable(tmp_path):
    info = _info(tmp_path, "reaction", "reaction", unavailable="HTTP 404")
    _finish_download(info)
    _finish_process(info)
    plan = _plan([info], explicit={"reaction"})
    assert plan.errors == [] and {s.state for s in plan.steps} == {"done"}


def test_partial_and_truncated_files_are_problems_and_never_touched(tmp_path):
    info = _info(tmp_path, "ec", "ec")
    archive = info.spec.archives[0]
    archive.path.parent.mkdir(parents=True)
    archive.partial.write_bytes(b"half")
    plan = _plan([info])
    assert plan.steps[0].state == "problem" and "interrupted download" in plan.errors[0]
    assert str(archive.partial) in plan.errors[0] and plan.steps[1].state == "problem"
    assert len(plan.errors) == 1  # the process step names its download instead of repeating the cause
    with pytest.raises(RuntimeError, match="plan has errors"):
        prepare_all.execute(plan, run=lambda c: pytest.fail("no run"))
    assert archive.partial.read_bytes() == b"half"

    archive.partial.unlink()
    archive.path.write_bytes(b"short")
    plan = _plan([info])
    assert plan.steps[0].state == "problem" and "has 5 bytes, pinned 13" in plan.errors[0]


def test_partly_built_processed_directory_is_a_problem(tmp_path):
    info = _info(tmp_path, "ec", "ec")
    _finish_download(info)
    directory = info.processed[0]
    directory.mkdir(parents=True)
    (directory / "train.pt").write_bytes(b"x")
    plan = _plan([info])
    process = plan.processes[0]
    assert process.state == "problem" and "partly built: 1 split file(s) exist, 2 are missing" in process.detail
    assert plan.errors and "never overwrites" in plan.errors[0]


def test_disk_check_refuses_when_the_need_exceeds_the_free_space(tmp_path):
    plan = prepare_all.build_plan(_tasks(tmp_path), [], [], free_bytes=lambda path: 2 * GIB)
    assert any("not enough free disk" in e for e in plan.errors)
    assert "PLAN ERRORS" in prepare_all.render(plan)


def test_task_without_download_spec_runs_its_cli_with_unknown_size(tmp_path):
    info = _info(tmp_path, "newtask", "newfam", spec=False)
    plan = _plan([info])
    download = plan.downloads[0]
    assert download.state == "todo" and download.size_unknown and "no download_spec hook" in download.detail
    assert "size unknown" in prepare_all.render(plan) and "some sizes are unknown" in prepare_all.render(plan)


# ---------------------------------------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------------------------------------

class FakeRunner:
    """Replaces the prepare CLIs: writes the outputs of a finished download/process, or fails on request."""

    def __init__(self, infos, fail=(), silent=()):
        self.by_module = {}
        for info in infos:
            self.by_module.setdefault(info.module, []).append(info)
        self.fail, self.silent, self.calls = set(fail), set(silent), []

    def __call__(self, command):
        module, step = command[3], command[4]
        self.calls.append((module, step))
        if (module, step) in self.fail:
            return 3
        if (module, step) not in self.silent:
            for info in self.by_module[module]:
                (_finish_download if step == "download" else _finish_process)(info)
        return 0


def test_run_downloads_then_processes_then_rerun_is_a_noop(tmp_path):
    infos = _tasks(tmp_path)
    runner = FakeRunner(infos)
    lines = []
    assert prepare_all.execute(_plan(infos), run=runner, say=lines.append) == 0
    downloads = [c for c in runner.calls if c[1] == "download"]
    assert runner.calls[:len(downloads)] == downloads  # every download before any process
    assert ("fake.tasks.go.prepare", "download") in runner.calls
    assert runner.calls.count(("fake.tasks.go.prepare", "download")) == 1  # shared
    assert ("fake.tasks.reaction.prepare", "download") not in runner.calls  # unavailable: not started
    assert any("skipped: source returns HTTP 404" in line for line in lines)
    runner.calls.clear()
    plan = _plan(infos)
    assert prepare_all.execute(plan, run=runner, say=lambda text: None) == 0  # reaction still unavailable, rest done
    assert runner.calls == []
    assert {s.state for s in plan.steps if "reaction" not in s.tasks} == {"done"}


def test_failed_download_skips_only_its_process_steps_and_fails_the_run(tmp_path):
    infos = _tasks(tmp_path)
    runner = FakeRunner(infos, fail={("fake.tasks.go.prepare", "download")})
    lines = []
    assert prepare_all.execute(_plan(infos), run=runner, say=lines.append) == 1
    assert ("fake.tasks.go.prepare", "process") not in runner.calls
    assert ("fake.tasks.ec.prepare", "process") in runner.calls and ("fake.tasks.mn.prepare", "process") in runner.calls
    text = "\n".join(lines)
    assert "FAILED: exit code 3" in text and "skipped: its download failed" in text
    assert "1 step(s) failed, 1 skipped because a download failed" in text


def test_exit_zero_without_outputs_is_a_failure(tmp_path):
    info = _info(tmp_path, "ec", "ec")
    runner = FakeRunner([info], silent={("fake.tasks.ec.prepare", "download")})
    lines = []
    assert prepare_all.execute(_plan([info]), run=runner, say=lines.append) == 1
    assert any("exit code 0, but the download is not complete" in line for line in lines)


def test_verify_processed_checks_recorded_checksums(tmp_path):
    directory = tmp_path / "processed" / "x"
    directory.mkdir(parents=True)
    (directory / "train.pt").write_bytes(b"abc")
    provenance = tmp_path / "provenance.json"
    record = {"kind": "process", "exit_code": 0, "processed_dir": str(directory),
              "outputs_sha256": {"train.pt": hashlib.sha256(b"abc").hexdigest()}}
    provenance.write_text(json.dumps({"runs": [record]}))
    assert "1 recorded file(s) match" in prepare_all.verify_processed([directory], provenance)[0]
    (directory / "train.pt").write_bytes(b"abd")
    with pytest.raises(RuntimeError, match="differs from the recorded"):
        prepare_all.verify_processed([directory], provenance)


def test_verify_archive_needs_a_pin(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="no pinned checksum"):
        verify_archive(Archive(path, "u", 1))
    with pytest.raises(RuntimeError, match="pinned 2"):
        verify_archive(Archive(path, "u", 2, md5="0"))


# ---------------------------------------------------------------------------------------------------------
# CLI and registry
# ---------------------------------------------------------------------------------------------------------

def test_main_picks_up_a_task_added_to_the_registry(tmp_path, capsys):
    infos = {i.name: i for i in _tasks(tmp_path)}
    infos["rsr_new"] = _info(tmp_path, "rsr_new", "rsr", archive="rsr.tar")
    code = prepare_all.main(["--tasks", "rsr", "--plan"], inspect=lambda name: infos[name], registry=tuple(infos))
    out = capsys.readouterr().out
    assert code == 0 and "rsr_new" in out and "download" in out and "process" in out
    assert not (tmp_path / "raw").exists()  # a plan writes nothing


def test_main_exit_codes(tmp_path, capsys):
    infos = {i.name: i for i in _tasks(tmp_path)}
    kwargs = {"inspect": lambda name: infos[name], "registry": tuple(infos)}
    assert prepare_all.main(["--tasks", "ec,bogus", "--plan"], **kwargs) == 2
    assert "unknown task or family name(s) ['bogus']" in capsys.readouterr().err
    assert prepare_all.main(["--tasks", "reaction", "--plan"], **kwargs) == 2  # named, unavailable
    assert "PLAN ERRORS" in capsys.readouterr().out
    assert prepare_all.main(["--tasks", "all", "--exclude", "reaction", "--plan"], **kwargs) == 0
    assert prepare_all.main(["--tasks", "ec", "--plan"], inspect=lambda n: infos[n], registry=tuple(infos),
                            run=lambda c: pytest.fail("a plan runs nothing")) == 0


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MIND_DATA_ROOT", str(tmp_path / "data"))
    return tmp_path / "data"


def test_real_registry_every_task_has_a_spec_and_shares_the_expected_archives(data_root, capsys):
    from downstream.tasks import MOLNET_TASKS, NPASS_TASKS, TASKS

    infos = [prepare_all.inspect_task(name) for name in TASKS]
    assert all(i.spec is not None for i in infos), "every registered task needs download_spec"
    plan = prepare_all.build_plan(infos, [], [], free_bytes=lambda p: 10 ** 15)
    downloads = {s.tasks: s for s in plan.downloads}
    assert ("go_mf", "go_bp", "go_cc") in downloads and tuple(MOLNET_TASKS) in downloads
    assert len(plan.downloads) == 8 and len(plan.processes) == 4 + 1 + len(MOLNET_TASKS) + len(NPASS_TASKS) + 1  # ec lba go fold reaction + mn + npass + rsr
    pinned = sum(a.bytes for s in plan.downloads for a in s.spec.archives if a.bytes)
    assert 8.5 * GIB < pinned < 9.5 * GIB
    assert downloads[("reaction",)].state == "unavailable" and plan.errors == []
    assert all(s.state in ("todo", "unavailable", "blocked") for s in plan.steps)
    # named: refused loudly, nothing written
    assert prepare_all.main(["--tasks", "reaction", "--plan"]) == 2
    assert "requested by name (reaction) but unavailable" in capsys.readouterr().out
    assert not data_root.exists()


def test_make_dry_run_commands():
    env = {k: v for k, v in os.environ.items() if k not in ("MIND_DATA_ROOT",)}

    def make(*args):
        result = subprocess.run(["make", "--no-print-directory", "-n", "ds-prepare", *args], cwd=REPO, env=env,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        return " ".join(result.stdout.replace("\\\n", " ").split())

    plan_only = make("PLAN=1", 'TASKS=molnet', 'EXCLUDE=mn_pcba')
    assert 'python -u -m downstream.prepare_all --tasks "molnet" --exclude "mn_pcba"' in plan_only
    assert plan_only.count("downstream.prepare_all") == 1 and "setsid" not in plan_only
    detached = make("WORKERS=4")
    assert detached.count("downstream.prepare_all") == 2  # the plan in the terminal, then the detached run
    assert "--plan" in detached.split("downstream.prepare_all")[1] and "--plan" not in detached.split("downstream.prepare_all")[2]
    assert "setsid nohup" in detached and 'echo "EXIT_CODE=$code"' in detached.replace("$$", "$") and "--workers 4" in detached
    assert '--tasks "all"' in detached
