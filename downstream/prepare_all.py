"""Download, extract and process the data of the selected downstream tasks in one command (``make ds-prepare``).

    python -m downstream.prepare_all [--tasks all|molnet|ec,go_mf,...] [--exclude "reaction mn_pcba"]
        [--workers N] [--threads N] [--plan]

The tasks come from the registry ``downstream.tasks.TASKS`` (``all``), from a family name (the package of the task
module: ``molnet`` = the 14 ``mn_*`` tasks, ``go`` = go_mf/go_bp/go_cc) or from task names; ``--exclude`` removes
names or families again. A task registered like the existing ones (module ``downstream.tasks.<family>.*`` with a
``downstream.tasks.<family>.prepare`` CLI taking ``download`` / ``process``) is picked up without a change here.

Plan. Every selected task is resolved to two steps that are shared where the inputs are the same:

* ``download``: tasks whose archives are the same files run one download (the 14 MoleculeNet tasks, one archive;
  go_mf/go_bp/go_cc, one archive). All downloads run before any processing.
* ``process``: tasks that write the same processed directory run one process (the three GO branches).

State of a step, from files only (no hashing, no network): ``done`` (archives of the pinned size and every output of
the download are present; every split file of the processed directory exists), ``todo``, ``unavailable`` (the source
cannot be fetched), ``blocked`` (its download is unavailable), ``unneeded`` (a download whose processed
outputs all exist: the raw data may have been removed) or ``problem``. Problems are never repaired and never
overwritten: a ``<archive>.partial`` file, an archive of the wrong size, a processed directory with only some of its
split files. The message names the path; the run refuses to start while any problem exists.
An unavailable task is listed and skipped when it comes from ``all`` or a family, and is an error when named.

Run. ``done`` steps are re-verified, not redone: archives against their pinned size and checksums, processed
split files against the sha256 recorded in ``provenance.json`` (without a recorded checksum only their existence is
checked, and the line says so). ``todo`` steps run the task's own prepare CLI (``python -m
downstream.tasks.<family>.prepare <step> ...``), each of which is idempotent on files already present. A failed step
is reported and the steps that do not depend on it go on; the exit code is 1 if any step failed. After a step its
state is re-checked: exit 0 without its outputs counts as a failure.

Disk. Before anything runs, the estimated need (archives still to fetch, extracted trees, processed directories; the
last two from ``disk_estimate_gib`` of task.yaml, measured on a finished build) is compared with the free space of
each file system involved (10 % margin).

Exit codes: 0 all steps done, 1 a step failed, 2 the plan has problems (nothing was run).
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from downstream.common.prepare_spec import GIB, Archive, DownloadSpec, disk_estimate, spec_key, verify_archive
from downstream.tasks import TASKS, load_task

DISK_MARGIN = 1.10  # required free space = estimated need x this
EXTRACT_FALLBACK = 6.0  # extracted size / archive size when a task.yaml has no disk estimate (measured 2.6 to 7)
DONE, PENDING, UNAVAILABLE, BLOCKED, PROBLEM, UNNEEDED = "done", "todo", "unavailable", "blocked", "problem", "unneeded"


class PrepareError(RuntimeError):
    """The selection is invalid (unknown or empty)."""


@dataclass(frozen=True)
class TaskInfo:
    """What the planner needs to know about one task (``inspect_task`` fills it from the registry)."""

    name: str
    family: str
    module: str  # the prepare CLI module, e.g. downstream.tasks.ec.prepare
    spec: Optional[DownloadSpec]  # None: the prepare module has no download_spec hook
    processed: Tuple[Path, ...]  # processed directories this task needs
    splits: Tuple[str, ...]  # a directory is built when every <split>.pt exists
    provenance: Path
    processed_gib: Optional[float]  # estimate per processed directory
    cli_args: Callable[..., list] = lambda step, **_: []


@dataclass
class Step:
    kind: str  # "download" | "process"
    tasks: Tuple[str, ...]
    module: str
    command: Tuple[str, ...]
    state: str
    detail: str = ""
    spec: Optional[DownloadSpec] = None
    dirs: Tuple[Path, ...] = ()
    splits: Tuple[str, ...] = ()
    provenance: Optional[Path] = None
    fetch_bytes: int = 0
    need_bytes: int = 0
    size_unknown: bool = False
    requires: Optional[Tuple[str, ...]] = None  # key of the download step a process step follows
    key: Tuple = ()
    explicit: Tuple[str, ...] = ()  # tasks of this step that were named (not only reached through ``all``)
    derived: bool = False  # the state follows from the download step; the cause is reported there
    anchor: Optional[Path] = None  # path whose file system receives the writes

    @property
    def label(self) -> str:
        return f"{self.kind} {','.join(self.tasks)}"


@dataclass
class Plan:
    selected: List[str]
    excluded: List[str]
    steps: List[Step] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    disk: List[Tuple[Path, int, int]] = field(default_factory=list)  # (anchor, need bytes, free bytes) per file system

    @property
    def downloads(self) -> List[Step]:
        return [s for s in self.steps if s.kind == "download"]

    @property
    def processes(self) -> List[Step]:
        return [s for s in self.steps if s.kind == "process"]


# ---------------------------------------------------------------------------------------------------------
# Registry access
# ---------------------------------------------------------------------------------------------------------

def inspect_task(name: str) -> TaskInfo:
    """Read one registered task: its config, processed directories and the hooks of its prepare module."""
    module = load_task(name)
    parts = module.__name__.split(".")
    if len(parts) < 4 or parts[:2] != ["downstream", "tasks"]:
        raise ValueError(f"task {name!r}: module {module.__name__!r} is not downstream.tasks.<family>.<module>")
    family, prepare_name = parts[2], ".".join(parts[:3]) + ".prepare"
    config = module.load_task_config(module.TASK_YAML)
    prepare = importlib.import_module(prepare_name)
    spec = prepare.download_spec(config) if hasattr(prepare, "download_spec") else None
    tags = config.get("split_tags")  # a task with several processed directories (LBA: one per split tag)
    directories = [module.processed_dir(config, tag) for tag in tags] if tags else [module.processed_dir(config)]
    return TaskInfo(
        name=name, family=family, module=prepare_name, spec=spec, processed=tuple(Path(d) for d in directories),
        splits=tuple(config["splits"]), provenance=Path(config["data_root"]) / "provenance.json",
        processed_gib=disk_estimate(config, "processed"), cli_args=getattr(prepare, "cli_args", lambda step, **_: []))


# ---------------------------------------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------------------------------------

def _tokens(text: str) -> List[str]:
    return [t for t in re.split(r"[,\s]+", text.strip()) if t]


def resolve_names(text: str, families: Dict[str, str], registry: Sequence[str]) -> List[str]:
    """Task names of ``text``: ``all``, family names and task names, in registry order, without duplicates."""
    chosen = set()
    unknown = []
    for token in _tokens(text):
        if token == "all":
            chosen.update(registry)
        elif token in registry:
            chosen.add(token)
        elif token in set(families.values()):
            chosen.update(name for name, family in families.items() if family == token)
        else:
            unknown.append(token)
    if unknown:
        raise PrepareError(f"unknown task or family name(s) {unknown}; choose from all, "
                           f"{', '.join(sorted(set(families.values())))}, {', '.join(registry)}")
    return [name for name in registry if name in chosen]


def select(tasks_text: str, exclude_text: str, *, registry: Sequence[str],
           families: Dict[str, str]) -> Tuple[List[str], List[str], set]:
    """(selected names, excluded names, names requested explicitly, i.e. not only through ``all``)."""
    if not _tokens(tasks_text):
        raise PrepareError("--tasks is empty; give all, a family (molnet, go, ...) or task names")
    requested = resolve_names(tasks_text, families, registry)
    named = " ".join(t for t in _tokens(tasks_text) if t != "all")
    explicit = set(resolve_names(named, families, registry)) if named else set()
    excluded = resolve_names(exclude_text, families, registry) if _tokens(exclude_text) else []
    selected = [name for name in requested if name not in excluded]
    if not selected:
        raise PrepareError(f"no task left: --tasks {tasks_text!r} minus --exclude {exclude_text!r}")
    return selected, [name for name in excluded if name in requested], explicit & set(selected)


# ---------------------------------------------------------------------------------------------------------
# State of a step (files only)
# ---------------------------------------------------------------------------------------------------------

def _gib(size: float) -> str:
    return f"{size / GIB:.2f} GiB"


def download_state(spec: DownloadSpec) -> Tuple[str, str, int]:
    """(state, detail, bytes still to fetch). ``problem`` for a partial file or an archive of the wrong size."""
    problems, missing = [], []
    for archive in spec.archives:
        if archive.partial.exists():
            problems.append(f"{archive.partial} ({archive.partial.stat().st_size} bytes) is left from an interrupted "
                            "download; inspect it, delete it and rerun (it is never overwritten or resumed)")
        elif archive.path.exists():
            if not archive.path.is_file():
                problems.append(f"{archive.path} exists and is not a file")
            elif archive.bytes is not None and archive.path.stat().st_size != archive.bytes:
                problems.append(f"{archive.path} has {archive.path.stat().st_size} bytes, pinned {archive.bytes}: "
                                "truncated or another file; move it aside and rerun")
        else:
            missing.append(archive)
    if problems:
        return PROBLEM, "; ".join(problems), 0
    absent_outputs = [o for o in spec.outputs if not o.exists()]
    if not missing and not absent_outputs:
        return DONE, "archives and outputs present", 0
    if spec.unavailable:
        return UNAVAILABLE, spec.unavailable, 0
    if missing:
        return PENDING, f"{len(missing)} archive(s) to fetch", sum(a.bytes or 0 for a in missing)
    return PENDING, f"archives present, {len(absent_outputs)} output(s) to extract or write", 0


def processed_state(dirs: Sequence[Path], splits: Sequence[str]) -> Tuple[str, str]:
    """``done`` when every split file of every directory exists, ``todo`` when none does, else ``problem``."""
    present, absent = [], []
    for directory in dirs:
        for split in splits:
            (present if (directory / f"{split}.pt").is_file() else absent).append(directory / f"{split}.pt")
    if not absent:
        return DONE, "all split files present"
    if not present:
        return PENDING, f"{len(absent)} split file(s) to build"
    return PROBLEM, (f"partly built: {len(present)} split file(s) exist, {len(absent)} are missing "
                     f"({', '.join(str(p) for p in absent[:3])}{', ...' if len(absent) > 3 else ''}); process never "
                     "overwrites an existing file: move the directory aside and rerun")


# ---------------------------------------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------------------------------------

def _command(module: str, step: str, extra: Iterable[str]) -> Tuple[str, ...]:
    return (sys.executable, "-u", "-m", module, step, *extra)


def _anchor(path: Path) -> Path:
    """The nearest existing ancestor of ``path`` (the file system a write there would use)."""
    path = Path(os.path.abspath(path))
    while not path.exists():
        path = path.parent
    return path


def build_plan(infos: Sequence[TaskInfo], excluded: Sequence[str] = (), explicit: Iterable[str] = (), *,
               workers: int = 16, threads: int = 8,
               free_bytes: Callable[[Path], int] = lambda p: shutil.disk_usage(p).free) -> Plan:
    """Steps, states, errors and the disk check for ``infos`` (registry order); touches no file."""
    explicit = set(explicit)
    plan = Plan(selected=[i.name for i in infos], excluded=list(excluded))
    downloads: Dict[Tuple, Step] = {}
    processes: Dict[Tuple, Step] = {}
    for info in infos:
        extra = lambda step: list(info.cli_args(step, task=info.name, workers=workers, threads=threads))  # noqa: E731
        # --- download
        if info.spec is not None:
            key = ("download", spec_key(info.spec))
        else:
            key = ("download", info.module, str(info.provenance.parent))
        step = downloads.get(key)
        if step is None:
            if info.spec is not None:
                state, detail, fetch = download_state(info.spec)
                extracted = info.spec.extracted_gib
                if state == PENDING:
                    fetched_size = sum(a.bytes or 0 for a in info.spec.archives if not a.path.exists())
                    unknown = any(a.bytes is None for a in info.spec.archives if not a.path.exists())
                    need = fetched_size + int((extracted if extracted is not None else
                                               EXTRACT_FALLBACK * sum(a.bytes or 0 for a in info.spec.archives) / GIB
                                               ) * GIB)
                else:
                    unknown, need = False, 0
                anchor = _anchor(info.spec.archives[0].path.parent) if info.spec.archives else _anchor(info.provenance)
            else:
                state, detail, fetch, unknown, need = PENDING, "no download_spec hook: size and state unknown", 0, True, 0
                anchor = _anchor(info.provenance.parent)
            step = Step("download", (), info.module, _command(info.module, "download", extra("download")), state,
                        detail, spec=info.spec, fetch_bytes=fetch, need_bytes=need, size_unknown=unknown, key=key,
                        anchor=anchor)
            downloads[key] = step
            plan.steps.append(step)
        step.tasks += (info.name,)
        step.explicit += (info.name,) if info.name in explicit else ()
        # --- process
        pkey = ("process", info.module, tuple(str(d) for d in info.processed))
        pstep = processes.get(pkey)
        if pstep is None:
            state, detail = processed_state(info.processed, info.splits)
            need = int((info.processed_gib or 0.0) * GIB * len(info.processed)) if state == PENDING else 0
            pstep = Step("process", (), info.module, _command(info.module, "process", extra("process")), state,
                         detail, dirs=info.processed, splits=info.splits, provenance=info.provenance, need_bytes=need,
                         size_unknown=state == PENDING and info.processed_gib is None, requires=key,
                         key=pkey, anchor=_anchor(info.processed[0].parent))
            processes[pkey] = pstep
            plan.steps.append(pstep)
        pstep.tasks += (info.name,)
        pstep.explicit += (info.name,) if info.name in explicit else ()
    plan.steps.sort(key=lambda s: s.kind != "download")  # stable: downloads first, registry order inside each kind
    _resolve_blocked(plan, downloads)
    _collect_errors(plan)
    _check_disk(plan, free_bytes)
    return plan


def _resolve_blocked(plan: Plan, downloads: Dict[Tuple, Step]) -> None:
    """A process step follows its download: it is blocked by an unavailable one; a download whose every process step
    is done is not needed (the raw data may have been removed after processing)."""
    for step in plan.processes:
        download = downloads[step.requires]
        if step.state == PENDING and download.state in (UNAVAILABLE, PROBLEM):
            step.derived = True
            step.state = BLOCKED if download.state == UNAVAILABLE else PROBLEM
            step.detail = (f"its download is unavailable: {download.detail}" if download.state == UNAVAILABLE
                           else "its download step has a problem (see above)")
    for key, download in downloads.items():
        dependents = [s for s in plan.processes if s.requires == key]
        if download.state in (PENDING, UNAVAILABLE) and dependents and all(s.state == DONE for s in dependents):
            download.state, download.detail = UNNEEDED, "processed outputs are complete; the raw data is not needed"
            download.fetch_bytes = download.need_bytes = 0
            download.size_unknown = False


def _collect_errors(plan: Plan) -> None:
    for step in plan.steps:
        if step.state == PROBLEM and not step.derived:
            plan.errors.append(f"{step.label}: {step.detail}")
        if step.state == UNAVAILABLE and step.explicit:
            plan.errors.append(f"{step.label}: requested by name ({', '.join(step.explicit)}) but unavailable: "
                               f"{step.detail}. Exclude it or make the source available")


def _check_disk(plan: Plan, free_bytes: Callable[[Path], int]) -> None:
    by_device: Dict[int, List] = {}
    for step in plan.steps:
        if step.state != PENDING or not step.need_bytes or step.anchor is None:
            continue
        entry = by_device.setdefault(os.stat(step.anchor).st_dev, [step.anchor, 0])
        entry[1] += step.need_bytes
    for anchor, need in by_device.values():
        free = free_bytes(anchor)
        plan.disk.append((anchor, need, free))
        if need * DISK_MARGIN > free:
            plan.errors.append(f"not enough free disk under {anchor}: the plan needs about {_gib(need)} (+10 % margin), "
                               f"{_gib(free)} are free")


# ---------------------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------------------

def _task_text(tasks: Sequence[str]) -> str:
    return ",".join(tasks) if len(tasks) <= 4 else f"{tasks[0]} .. {tasks[-1]} ({len(tasks)} tasks)"


def render(plan: Plan, *, tasks_text: str = "all", exclude_text: str = "") -> str:
    lines = [f"ds-prepare plan: {len(plan.selected)} task(s) selected (TASKS={tasks_text}"
             f"{', EXCLUDE=' + exclude_text if exclude_text.strip() else ''}"
             f"{'; excluded: ' + ' '.join(plan.excluded) if plan.excluded else ''})", ""]
    rows = []
    for number, step in enumerate(plan.steps, start=1):
        if step.kind == "download" and step.spec is not None:
            size = sum(a.bytes or 0 for a in step.spec.archives)
            size_text = (f"{_gib(size)} archive" + ("" if all(a.bytes is not None for a in step.spec.archives)
                                                    else " (partly unpinned)")) if size else "size unknown"
        elif step.kind == "download":
            size_text = "size unknown"
        else:
            size_text = f"{len(step.dirs)} dir(s)"
        rows.append((str(number), step.kind, step.state, _task_text(step.tasks), size_text, step.detail))
    header = ("#", "step", "state", "tasks", "size", "detail")
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(5)]
    for row in [header, *rows]:
        cells = [row[i].ljust(widths[i]) for i in range(5)]
        lines.append("  ".join(cells) + "  " + row[5])
    todo = [s for s in plan.steps if s.state == PENDING]
    fetch = sum(s.fetch_bytes for s in todo)
    need = sum(s.need_bytes for s in todo)
    lines += ["", f"Steps: {len(todo)} to run, {sum(s.state == DONE for s in plan.steps)} done (re-verified, not redone), "
              f"{sum(s.state in (UNAVAILABLE, BLOCKED) for s in plan.steps)} unavailable or blocked, "
              f"{sum(s.state == UNNEEDED for s in plan.steps)} not needed, "
              f"{sum(s.state == PROBLEM for s in plan.steps)} with a problem.",
              f"To fetch: {_gib(fetch)}. Estimated disk need: {_gib(need)}"
              + (" (at least: some sizes are unknown)" if any(s.size_unknown for s in todo) else "") + "."]
    for anchor, need_bytes, free in plan.disk:
        lines.append(f"Disk under {anchor}: need about {_gib(need_bytes)} (+10 % margin), free {_gib(free)}.")
    if plan.errors:
        lines += ["", "PLAN ERRORS (nothing is run):", *[f"  - {e}" for e in plan.errors]]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------------------------------------

def verify_processed(directories: Sequence[Path], provenance: Path) -> List[str]:
    """Re-check the recorded sha256 of processed files; returns one line per directory (raises on a difference)."""
    import json

    from downstream.common.registry import sha256_file

    runs = json.loads(provenance.read_text()).get("runs", []) if provenance.is_file() else []
    lines = []
    for directory in directories:
        records = [r for r in runs if (r.get("kind") == "process" or r.get("step") == "process")
                   and r.get("exit_code") == 0 and r.get("outputs_sha256")
                   and Path(str(r.get("processed_dir", ""))).name == directory.name]
        if not records:
            lines.append(f"{directory}: split files present; no recorded checksums in {provenance.name}, existence only")
            continue
        recorded = records[-1]["outputs_sha256"]
        for name, expected in recorded.items():
            target = directory / name
            if not target.is_file():
                raise RuntimeError(f"{target} is recorded in {provenance} but missing")
            found = sha256_file(target)
            if found != expected:
                raise RuntimeError(f"{target}: sha256 {found} differs from the recorded {expected}; the file changed "
                                   "after processing. Move the directory aside and rerun")
        lines.append(f"{directory}: {len(recorded)} recorded file(s) match their sha256")
    return lines


def run_command(command: Sequence[str]) -> int:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    return subprocess.run(list(command), env=env).returncode


def _recheck(step: Step) -> Optional[str]:
    """After a step ran: the reason it did not produce its outputs, else None."""
    if step.kind == "download":
        if step.spec is None:
            return None
        state, detail, _ = download_state(step.spec)
        return None if state == DONE else f"exit code 0, but the download is not complete ({state}: {detail})"
    state, detail = processed_state(step.dirs, step.splits)
    return None if state == DONE else f"exit code 0, but the processed files are not complete ({state}: {detail})"


def execute(plan: Plan, *, run: Callable[[Sequence[str]], int] = run_command,
            say: Callable[[str], None] = lambda text: print(text, flush=True)) -> int:
    """Run the plan; returns 0 when every step is done, else 1."""
    if plan.errors:
        raise RuntimeError("the plan has errors; execute() only runs a clean plan")
    results: Dict[Tuple, str] = {}
    outcomes: List[Tuple[Step, str]] = []
    total = len(plan.steps)
    for number, step in enumerate(plan.steps, start=1):
        say(f"=== [{number}/{total}] {step.label}: {step.state}")
        if step.state in (UNAVAILABLE, BLOCKED, UNNEEDED):
            say(f"skipped: {step.detail}")
            outcome = "skipped"
        elif step.kind == "process" and results.get(step.requires) == "failed":
            say("skipped: its download failed")
            outcome = "skipped (download failed)"
        elif step.state == DONE:
            try:
                if step.kind == "download":
                    for line in (verify_archive(a) for a in (step.spec.archives if step.spec else ())):
                        say(f"verified {line}")
                else:
                    for line in verify_processed(step.dirs, step.provenance):
                        say(f"verified {line}")
                outcome = "done (verified)"
            except RuntimeError as exc:
                say(f"FAILED verification: {exc}")
                outcome = "failed"
        else:
            say("$ " + " ".join(step.command))
            code = run(step.command)
            problem = f"exit code {code}" if code else _recheck(step)
            if problem:
                say(f"FAILED: {problem}")
                outcome = "failed"
            else:
                outcome = "done"
        results[step.key] = outcome
        outcomes.append((step, outcome))
    say("=== summary")
    for step, outcome in outcomes:
        say(f"{outcome:<24} {step.label}")
    failed = [s for s, o in outcomes if o == "failed"]
    skipped = [s for s, o in outcomes if o.startswith("skipped (download")]
    if failed or skipped:
        say(f"{len(failed)} step(s) failed, {len(skipped)} skipped because a download failed")
        return 1
    return 0


# ---------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------

def main(argv: Optional[list] = None, *, inspect: Callable[[str], TaskInfo] = inspect_task,
         registry: Sequence[str] = TASKS, run: Callable[[Sequence[str]], int] = run_command) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks", default="all", help="all, a family (molnet, go, ...) or a comma list of task names")
    parser.add_argument("--exclude", default="", help="task or family names to leave out (comma or space separated)")
    parser.add_argument("--workers", type=int, default=16, help="structure-parsing workers of the process steps")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads of the steps that bound torch threads")
    parser.add_argument("--plan", action="store_true", help="print the plan and exit (writes nothing)")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.threads < 1:
        parser.error("--workers and --threads must be at least 1")
    infos: Dict[str, TaskInfo] = {}
    problems = []
    for name in registry:
        try:
            infos[name] = inspect(name)
        except Exception as exc:  # noqa: BLE001 - reported with the task name, never hidden
            problems.append(f"task {name}: cannot be inspected: {type(exc).__name__}: {exc}")
    families = {name: info.family for name, info in infos.items()}
    try:
        selected, excluded, explicit = select(args.tasks, args.exclude, registry=list(registry), families=families)
    except PrepareError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    broken = [p for p in problems if any(p.startswith(f"task {n}:") for n in selected)]
    if broken:
        print("ERROR: a selected task cannot be inspected:\n  " + "\n  ".join(broken), file=sys.stderr)
        return 2
    plan = build_plan([infos[n] for n in selected], excluded, explicit, workers=args.workers, threads=args.threads)
    print(render(plan, tasks_text=args.tasks, exclude_text=args.exclude), end="", flush=True)
    if plan.errors:
        return 2
    if args.plan:
        return 0
    return execute(plan, run=run)


if __name__ == "__main__":
    raise SystemExit(main())
