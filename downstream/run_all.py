"""Embed, probe, fine-tune and ledger every task for one MIND model, then print its final metrics (``make ds-run``).

    python -m downstream.run_all --model <snapshot | path/to.ckpt> [--name NAME] --tasks all|ec,go_mf,...
        --gpus 2,3 --heads linear,mlp[,finetune] --per-gpu 3 [--random-init] [--compare-to a,b.random_init]
        [--counterfactual] [--plan]

``--heads`` selects the protocols: ``linear`` / ``mlp`` are frozen-encoder probes, ``finetune`` is end-to-end
fine-tuning (``downstream.finetune`` with the task.yaml ``finetune`` settings). ``--compare-to`` models (snapshot
names; ``.random_init`` = RANDOM_INIT=1) go through steps 2-5 the same way (what is complete is skipped):

1. A checkpoint path is snapshotted as NAME (``downstream.common.registry.snapshot_model``). NAME already naming
   another checkpoint (sha256 differs from the snapshot card or from the results ledger) stops everything.
2. Embeddings (``downstream.embed``), one job per task (LBA: one per required split tag), on the GPUs given,
   at most ``--per-gpu`` jobs per GPU. Splits that already have an embedding (.npz and .json) are not redone.
   Without a probe head no embedding is computed.
3. When every embedding job has ended: probes (``downstream.probe``, CPU) per task x head, seeds missing only.
   Then, with ``finetune`` in ``--heads``: one fine-tune job per task (seeds missing only, a seed counts as done when
   its ``metrics.json`` exists and its config.json records the current settings and processed directory), at most
   ``--ft-per-gpu`` jobs per GPU, the longest first (estimate: task.yaml ``finetune_seconds_per_epoch`` x epochs x
   seeds), each next job on the first free slot. After every finished probe or fine-tune job its results are added to
   the ledger (``ledger add`` is idempotent: files already ingested are skipped), so an interrupted run keeps them.
4. ``downstream.ledger add`` of the outputs root into ``$MIND_RESULTS_ROOT``.
5. One table: the catalog's primary metric for every task x head (a failed or skipped one says why), and the
   required cells still missing for this model (``ledger_report.missing_cells``); one table per model.
6. ``downstream.ledger_compare --a <each compare-to model> --b <model>``, then ``downstream.ledger_report report``
   (what ``make ds-compare`` and ``make ds-ledger-report`` run).

``--counterfactual`` (only for a checkpoint path): after the embeddings, the ``make eval-counterfactual`` sweep
(``core.evaluate_pretrain`` with the target's arguments, one mode per job, one job per GPU) on the snapshot copy,
written to ``<outputs root>/eval/cf_<name>_<stamp>/<mode>`` and added to the ledger with step 4. The Make target
itself detaches and returns at once, so ds-run cannot wait for it; it refuses to start if the target is gone.

``--model <snapshot>.random_init`` is the same as ``--model <snapshot> --random-init``.

``all`` = EC, GO-MF/BP/CC, Fold, LBA (the catalog's required split tags, joint representation), RSR and the 14
MoleculeNet tasks. Reaction has no processed data and runs only when named. A task whose domain the
model was not pretrained on (card ``domains``; protein needs ``pdb``, RSR ``rna``, MoleculeNet ``unimol``) is skipped and
listed under ``all`` and refused when named explicitly.

A failing job (non-zero exit, or exit 0 without its output files) is reported with its log and the others go on;
the run ends with the list of everything missing and exit code 1. Jobs have no timeout; a status line with each
running PID, its age and its log's size and last write is printed every ``--status-every`` seconds.

Logs: ``<log root>/ds_run_<model>_<stamp>_<pid>/<kind>_<task>_<model>_<YYYYMMDD_HHMMSS_mmm>_<child pid>.log``
(kind ``embed`` / ``probe_linear`` / ``probe_mlp`` / ``finetune`` / ``counterfactual`` / ``ledger_add`` /
``compare`` / ``ledger_report``; ``<model>`` is the job's own model), next to ``summary.md`` and ``missing.csv``; ``make ds-embed``
and ``ds-probe`` logs are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from downstream import ledger, ledger_report
from downstream.common import registry, split_replicates
from downstream.finetune_settings import resolve_settings, same_settings
from downstream.tasks import MOLNET_TASKS, NPASS_TASKS, TASKS, load_task

HEADS = ("linear", "mlp")  # downstream.common.heads.HEADS (not imported: it pulls torch into the plan step)
FINETUNE = "finetune"  # the --heads entry that selects end-to-end fine-tuning
PROTOCOL_CHOICES = HEADS + (FINETUNE,)
ALL_TASKS = ("ec", "go_mf", "go_bp", "go_cc", "fold", "lba", "rsr", "ss_q3", "ss_q8") + MOLNET_TASKS + NPASS_TASKS
# Pretraining domains (card "domains", the enabled multi_domain datasets) a task's input needs. The LBA joint graph
# is a protein graph (ligand atoms with pos_code sm).
MODALITY_DOMAINS = {"protein": ("pdb",), "complex": ("pdb",), "small_molecule": ("unimol",), "rna": ("rna",)}
HF_PREFIXES = ("esm2_", "saprot_")


class PlanError(RuntimeError):
    """The run cannot start (bad arguments, model identity conflict, explicitly named task that cannot run)."""


@dataclass(frozen=True)
class Unit:
    """One embedding set: a task, plus LBA's split tag and representation, or the split tag of a per-seed-split task.

    A task with one split per seed (NPASS) has a unit per split tag and no representation: its protocols carry no tag
    (the cells aggregate over the tags), and the unit's own seed is the one of its tag.
    """

    task: str
    split_tag: Optional[str] = None
    representation: Optional[str] = None

    @property
    def key(self) -> str:
        return ".".join(part for part in (self.task, self.split_tag, self.representation) if part)

    @property
    def prefix(self) -> str:
        return "".join(f"{part}." for part in (self.split_tag, self.representation) if part)

    def protocol(self, head: str) -> str:
        """Results protocol of a probe head or of ``finetune`` (fine-tuning of LBA is joint-only)."""
        base = FINETUNE if head == FINETUNE else f"{head}_probe"
        return base + (f".{self.split_tag}.{self.representation}" if self.representation else "")

    def cli(self) -> list:
        return (["--split-tag", self.split_tag] if self.split_tag else []) + (
            ["--representation", self.representation] if self.representation else [])


# ---------------------------------------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------------------------------------

def expand_tasks(text: str, catalog: dict, seeds: Optional[list] = None) -> tuple[list, bool]:
    """Units for ``all`` or a comma list, and whether the tasks were named explicitly.

    A task with one split per seed (catalog ``split_replicates``) gets one unit per tag of ``seeds``.
    """
    explicit = text.strip() != "all"
    names = list(ALL_TASKS) if not explicit else [t.strip() for t in text.split(",") if t.strip()]
    unknown = [t for t in names if t not in TASKS]
    if unknown or not names:
        raise PlanError(f"unknown task(s) {unknown or names}; choose 'all' or a comma list of {', '.join(TASKS)}")
    units = []
    for name in dict.fromkeys(names):
        if name == "lba":
            for protocol in catalog["tasks"]["lba"]["required_protocols"]:
                _, split_tag, representation = protocol.split(".")
                units.append(Unit("lba", split_tag, representation))
        elif catalog["tasks"].get(name, {}).get("split_replicates"):
            if not seeds:
                raise PlanError(f"{name} has one split per seed; give the seeds whose splits to run (--seeds)")
            units += [Unit(name, split_replicates.tag_for_seed(seed)) for seed in dict.fromkeys(seeds)]
        else:
            units.append(Unit(name))
    return units, explicit


def describe_task(unit: Unit) -> dict:
    """Modality, split names, probe and fine-tune pooling and processed split files of one unit (reads task.yaml)."""
    task = load_task(unit.task)
    config = task.load_task_config(task.TASK_YAML)
    replicated = split_replicates.is_replicated(config)
    if replicated and unit.split_tag not in config["split_tags"]:
        raise ValueError(f"split tag {unit.split_tag!r} has no split: {unit.task} draws splits for seeds "
                         f"{config['split_seeds']} (task.yaml split.seeds)")
    directory = task.processed_dir(config, unit.split_tag) if unit.split_tag else task.processed_dir(config)
    splits = list(config["splits"])
    cost = config["finetune_seconds_per_epoch"]
    cost = cost[unit.task] if isinstance(cost, dict) else cost
    return {"modality": config["modality"], "splits": splits, "replicated": replicated,
            "pooling": config["probe"]["pooling"],
            "finetune_pooling": config["finetune"]["pooling"], "finetune": config["finetune"],
            "finetune_seconds": float(cost) * int(config["finetune"]["epochs"]),
            "processed": {split: Path(directory) / f"{split}.pt" for split in splits}}


def _sha256_stable(path: Path) -> str:
    before = path.stat()
    sha = registry.sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise PlanError(f"{path} changed while hashing (a live run rewriting it?); try again when it is stable")
    return sha


@dataclass
class ModelPlan:
    name: str
    model_id: str
    sha256: str
    domains: list
    action: str  # "snapshot exists" | "create snapshot" | "snapshot exists (same checkpoint)"
    ckpt: Optional[Path] = None
    epoch: Optional[int] = None
    notes: list = field(default_factory=list)


def resolve_model(model: str, name: Optional[str], *, random_init: bool, outputs_root: Path,
                  results_root: Optional[Path]) -> ModelPlan:
    """Snapshot to use (or to create) for ``model``; every identity conflict is a ``PlanError``."""
    if model.startswith(HF_PREFIXES):
        raise PlanError(f"{model!r} is an HF baseline; ds-run drives MIND models only (use make ds-embed / ds-probe)")
    is_path = model.endswith(".ckpt") or os.sep in model
    if is_path:
        ckpt = Path(model).resolve()
        if not name:
            raise PlanError("a checkpoint path needs NAME (the snapshot name the results are recorded under)")
        if not ckpt.is_file():
            raise PlanError(f"checkpoint not found: {ckpt}")
        sha = _sha256_stable(ckpt)
        try:
            registry._check_name("snapshot name", name)
        except ValueError as exc:
            raise PlanError(str(exc)) from exc
        card_path = outputs_root / "models" / name / "card.json"
        if card_path.is_file():
            card = json.loads(card_path.read_text())
            if card["sha256"] != sha:
                raise PlanError(f"snapshot {name!r} ({card_path}) is checkpoint sha256 {card['sha256']}, but {ckpt} is "
                                f"{sha}; one name must name one checkpoint: choose another NAME")
            plan = ModelPlan(name, name, sha, list(card.get("domains") or []), "snapshot exists (same checkpoint)",
                             ckpt, card.get("epoch"))
        elif (outputs_root / "models" / name).exists():
            raise PlanError(f"{outputs_root / 'models' / name} exists without a card.json; inspect it by hand")
        else:
            summary = registry._checkpoint_summary(ckpt)
            plan = ModelPlan(name, name, sha, summary["domains"], "create snapshot", ckpt, summary["epoch"])
            hiding = (summary["hparams"] or {}).get("unimol_masked_geometry_hiding")
            plan.notes.append(f"unimol_masked_geometry_hiding = {hiding}")
    else:
        if name and name != model:
            raise PlanError(f"NAME={name} is only for a checkpoint path; MODEL={model} is already a snapshot name")
        try:
            resolved = registry.resolve_model(model, root=outputs_root)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            raise PlanError(f"MODEL={model}: {exc}") from exc
        card = resolved.card
        plan = ModelPlan(resolved.model_id, resolved.model_id, card["sha256"], list(card.get("domains") or []),
                         "snapshot exists", None, card.get("epoch"))
        if resolved.model_id != model:
            plan.notes.append(f"alias {model} -> {resolved.model_id}")
    if not plan.domains:
        raise PlanError(f"snapshot {plan.name}: no pretraining domains in its card; the domain check cannot run")
    if results_root is not None and ledger.ledger_dir(results_root).exists():
        for row in ledger.read_records(results_root, "models") + ledger.read_records(results_root, "measurements"):
            key = row["model_key"]
            if key in (plan.name, plan.name + registry.RANDOM_INIT_SUFFIX):
                sha = row.get("sha256", row.get("model_sha256"))
                if sha != plan.sha256:
                    raise PlanError(f"the results ledger has {key!r} with checkpoint sha256 {sha}, not {plan.sha256}; "
                                    "one name must name one checkpoint")
    if random_init:
        plan.model_id = plan.name + registry.RANDOM_INIT_SUFFIX
    return plan


def _known_timings(unit: Unit, splits: list, outputs_root: Path, model_id: str) -> list:
    """Recorded embedding seconds of other models for the same task (the ``seconds`` field of embed.py)."""
    found = []
    for directory in sorted((outputs_root / "downstream" / unit.task).glob("*/embeddings")):
        if directory.parent.name == model_id:
            continue
        total, used = 0.0, []
        for split in splits:
            path = directory / f"{unit.prefix}{split}.json"
            if path.is_file():
                record = json.loads(path.read_text())
                if isinstance(record.get("seconds"), (int, float)):
                    total += record["seconds"]
                    used.append(split)
        if used:
            found.append(f"{directory.parent.name}: {total:.1f} s over {'/'.join(used)} (embed.py 'seconds'; {directory})")
    return found[:2]


def unit_seeds(unit: Unit, info: dict, seeds: list) -> list:
    """Seeds a unit trains: all of them, or (one split per seed) the seed whose split is the unit's tag."""
    return split_replicates.seeds_of_tag(unit.split_tag, seeds) if info.get("replicated") else list(seeds)


def plan_units(units: list, explicit: bool, model: ModelPlan, *, heads: list, seeds: list, outputs_root: Path,
               describe: Callable[[Unit], dict] = None) -> dict:
    """What runs, what is skipped and what already cannot run, per unit and head (``finetune`` included)."""
    describe = describe or describe_task
    probe_heads = [head for head in heads if head != FINETUNE]
    plan = {"embed": [], "embed_done": [], "probe": [], "probe_done": [], "finetune": [], "finetune_done": [],
            "domain_skips": [], "failed": [], "notes": [], "timings": {}, "units": [], "info": {}}
    for unit in units:
        try:
            info = describe(unit)
        except Exception as exc:  # noqa: BLE001 - any config error is reported for this unit, the others go on
            plan["failed"].append((f"config {unit.key}", f"task config unreadable: {type(exc).__name__}: {exc}"))
            continue
        plan["info"][unit] = info
        needed = MODALITY_DOMAINS.get(info["modality"])
        if needed is None:
            raise PlanError(f"{unit.key}: modality {info['modality']!r} has no domain rule in MODALITY_DOMAINS")
        lacking = [d for d in needed if d not in model.domains]
        if lacking:
            reason = (f"domain mismatch: the task has modality={info['modality']} and needs {lacking}, model "
                      f"{model.name} has domains={model.domains}")
            if explicit:
                raise PlanError(f"{unit.key}: {reason}. Run it on purpose with make ds-embed / ds-probe.")
            plan["domain_skips"].append((unit, reason))
            continue
        if info["modality"] == "complex" and not {"unimol", "metabolite"} & set(model.domains):
            plan["notes"].append(f"{unit.key}: ligand atoms, but {model.name} saw no small-molecule domain")
        plan["units"].append(unit)
        seeds_here = unit_seeds(unit, info, seeds)
        if FINETUNE in heads:
            _plan_finetune(plan, unit, info, model, seeds=seeds_here, outputs_root=outputs_root)
        if not probe_heads:
            continue
        missing, broken, stale = [], [], []
        for split in info["splits"]:
            npz = registry.embeddings_path(unit.task, model.model_id, unit.prefix + split, root=outputs_root)
            if npz.with_suffix(".json").is_file() and npz.is_file():
                # The embedding path carries no data revision: one built from another processed directory is not done.
                built_from = Path(json.loads(npz.with_suffix(".json").read_text()).get("processed") or "").parent.name
                if built_from != info["processed"][split].parent.name:
                    stale.append(f"{npz.name} built from processed directory {built_from!r}")
                continue
            if npz.is_file() or npz.with_suffix(".json").is_file():
                broken.append(str(npz))
            missing.append(split)
        no_data = [str(info["processed"][s]) for s in missing if not info["processed"][s].is_file()]
        if broken:
            reason = (f"half-written embedding {broken} (.npz without .json or the reverse: an embed still running "
                      "or one that crashed); inspect it by hand")
            plan["failed"].append((f"embed {unit.key}", reason))
        elif stale:
            plan["failed"].append((f"embed {unit.key}", f"{stale}, but the task reads "
                                   f"{info['processed'][info['splits'][0]].parent.name!r}; use a new OUTPUTS_ROOT"))
        elif no_data:
            plan["failed"].append((f"embed {unit.key}", f"no processed data: {no_data}; build it with "
                                  f"`make ds-prepare TASKS={unit.task}`"))
        elif missing:
            plan["embed"].append((unit, missing))
            plan["timings"][unit] = _known_timings(unit, missing, outputs_root, model.model_id)
        else:
            plan["embed_done"].append(unit)
        for head in probe_heads:
            done = [s for s in seeds_here if (registry.results_dir(unit.task, model.model_id, unit.protocol(head), s,
                                                                   pooling=info["pooling"], root=outputs_root)
                                              / "metrics.json").is_file()]
            todo = [s for s in seeds_here if s not in done]
            if todo:
                plan["probe"].append((unit, head, todo))
            else:
                plan["probe_done"].append((unit, head))
    return plan


def _plan_finetune(plan: dict, unit: Unit, info: dict, model: ModelPlan, *, seeds: list, outputs_root: Path) -> None:
    """Seeds of one unit still to fine-tune; missing processed data, a non-joint LBA unit or a finished seed trained on
    another processed directory (results paths carry no data revision) cannot run."""
    label = f"{FINETUNE} {unit.key}"
    no_data = [str(path) for split, path in info["processed"].items() if not Path(path).is_file()]
    if unit.representation not in (None, "joint"):
        plan["failed"].append((label, "fine-tuning supports the joint LBA representation only"))
        return
    if no_data:
        plan["failed"].append((label, f"no processed data: {no_data}; build it with `make ds-prepare TASKS={unit.task}`"))
        return
    expected = sorted({Path(path).parent.name for path in info["processed"].values()})
    settings = resolve_settings(info["finetune"], random_init=model.model_id.endswith(registry.RANDOM_INIT_SUFFIX))
    todo, stale = [], []
    for seed in seeds:
        directory = registry.results_dir(unit.task, model.model_id, unit.protocol(FINETUNE), seed,
                                         pooling=info["finetune_pooling"], root=outputs_root)
        if not (directory / "metrics.json").is_file():
            todo.append(seed)
            continue
        config = json.loads((directory / "config.json").read_text()) if (directory / "config.json").is_file() else {}
        built = sorted({Path(entry["path"]).parent.name for entry in (config.get("processed") or {}).values()})
        if built != expected:
            stale.append(f"seed{seed} trained on {built or 'unrecorded processed data'}")
        elif not same_settings(config.get("finetune"), settings):
            stale.append(f"seed{seed} trained with other finetune settings than task.yaml")
    if stale:
        plan["failed"].append((label, f"{stale}, but the task reads {expected} with the current task.yaml finetune "
                                      "settings; use a new OUTPUTS_ROOT"))
    elif todo:
        plan["finetune"].append((unit, todo))
    else:
        plan["finetune_done"].append(unit)


def finetune_seconds(plan: dict, unit: Unit, seeds: list) -> float:
    """Estimated seconds of one fine-tune job (task.yaml estimate per epoch x epochs x seeds); orders the queue."""
    return plan["info"][unit]["finetune_seconds"] * len(seeds)


def plan_text(model: ModelPlan, plan: dict, *, outputs_root: Path, results_root: Optional[Path], gpus: list,
              per_gpu: int, probe_jobs: int, heads: list, seeds: list, ft_per_gpu: int = 1) -> str:
    lines = [f"Model: {model.name} (model_id {model.model_id}), sha256 {model.sha256}, epoch {model.epoch}, "
             f"domains {model.domains}; {model.action}" + (f" from {model.ckpt}" if model.ckpt else "")]
    lines += [f"  not: {note}" for note in model.notes]
    lines += [f"Outputs root: {outputs_root}", f"Results ledger: {results_root}",
              f"GPU: {','.join(gpus) or '-'} x {per_gpu} jobs per GPU; probes: {probe_jobs} at a time on CPU; "
              f"heads {','.join(heads)}; seeds {' '.join(map(str, seeds))}", "",
              f"1. Embed ({len(plan['embed'])} jobs):"]
    for unit, splits in plan["embed"]:
        lines.append(f"   WILL RUN  embed {unit.key}: splits {' '.join(splits)}")
        known = plan["timings"].get(unit)
        lines += [f"     known timing: {entry}" for entry in known] if known else [
            "     known timing: none (no recorded embed time for this task)"]
    lines += [f"   WILL SKIP  embed {unit.key}: already complete" for unit in plan["embed_done"]]
    lines.append(f"2. Probe ({len(plan['probe'])} jobs, after every embed has ended; no timing recorded):")
    lines += [f"   WILL RUN  probe {unit.key} {head}: seeds {' '.join(map(str, todo))}"
              for unit, head, todo in plan["probe"]]
    lines += [f"   WILL SKIP  probe {unit.key} {head}: already complete" for unit, head in plan["probe_done"]]
    step = 3
    if plan["finetune"] or plan["finetune_done"]:
        lines.append(f"3. Fine-tune ({len(plan['finetune'])} jobs after the probes, longest first, at most "
                     f"{ft_per_gpu} per GPU; task.yaml finetune settings; hours = task.yaml estimate for one job alone "
                     "on a GPU):")
        lines += [f"   WILL RUN  finetune {unit.key}: seeds {' '.join(map(str, todo))} (about "
                  f"{plan['info'][unit]['finetune_seconds'] * len(todo) / 3600:.1f} h)"
                  for unit, todo in sorted(plan["finetune"], key=lambda item: -finetune_seconds(plan, *item))]
        lines += [f"   WILL SKIP  finetune {unit.key}: already complete" for unit in plan["finetune_done"]]
        step = 4
    lines += [f"{step}. ledger add (outputs root -> results ledger)",
              f"{step + 1}. summary table + missing required cells"]
    if plan["domain_skips"]:
        lines += ["", "Skipped tasks (domain mismatch):"] + [f"   SKIPPED  {u.key}: {r}" for u, r in plan["domain_skips"]]
    if plan["notes"]:
        lines += ["", "Notes:"] + [f"   {note}" for note in plan["notes"]]
    if plan["failed"]:
        lines += ["", f"Cannot run ({len(plan['failed'])}):"] + [f"   ERROR  {label}: {reason}"
                                                           for label, reason in plan["failed"]]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------------------------------------

def _stamp() -> str:
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"


@dataclass
class Job:
    kind: str  # embed | probe_linear | probe_mlp
    unit: Unit
    argv: Callable[[str], list]  # slot -> argv
    env: Callable[[str], dict]  # slot -> extra environment
    check: Callable[[], Optional[str]]  # None when the expected outputs exist
    head: Optional[str] = None
    model_id: Optional[str] = None  # the model in the log name (default: the Runner's)
    seconds: float = 0.0  # estimated duration (fine-tune jobs; orders the queue)

    @property
    def label(self) -> str:
        return f"{self.kind} {self.unit.key}"


class Runner:
    """Runs jobs on slots (one slot = one concurrent job on one GPU or on the CPU), waits for all, no timeout."""

    def __init__(self, run_dir: Path, model_id: str, *, status_every: float, poll: float, out=print):
        self.run_dir, self.model_id, self.status_every, self.poll, self.out = run_dir, model_id, status_every, poll, out
        self.counter = 0

    def _launch(self, job: Job, slot: str):
        stamp = _stamp()
        self.counter += 1
        starting = self.run_dir / f".starting_{os.getpid()}_{self.counter}.log"
        handle = open(starting, "x")
        try:
            process = subprocess.Popen(job.argv(slot), stdout=handle, stderr=subprocess.STDOUT,
                                       env={**os.environ, **job.env(slot)}, cwd=registry.REPO_ROOT)
        except OSError:
            handle.close()
            starting.unlink()
            raise
        handle.close()  # the child keeps its own descriptor
        log = self.run_dir / f"{job.kind}_{job.unit.key}_{job.model_id or self.model_id}_{stamp}_{process.pid}.log"
        try:
            os.link(starting, log)  # never replaces an existing log
        except FileExistsError:
            process.kill()
            raise
        starting.unlink()
        self.out(f"[{datetime.now():%H:%M:%S}] STARTED {job.label} ({slot}) pid {process.pid} log {log}")
        return process, log, time.monotonic()

    def _status(self, running: dict, pending: int, finished: int, failed: int) -> None:
        parts = []
        now = time.time()
        for slot, (job, process, log, started) in running.values():
            stat = log.stat()
            parts.append(f"{job.label} ({slot}) pid {process.pid} {int(time.monotonic() - started)} s, log "
                         f"{stat.st_size} B, last write {int(now - stat.st_mtime)} s ago")
        self.out(f"[{datetime.now():%H:%M:%S}] status: running {len(running)}, pending {pending}, finished {finished}, "
                 f"failed {failed}" + "".join(f"\n    {p}" for p in parts))

    def run(self, jobs: List[Job], slots: List[str],
            on_done: Optional[Callable[[Job, bool], None]] = None) -> Dict[int, tuple]:
        """``{job index: (ok, reason, log)}``; every job runs even when others fail. ``on_done(job, ok)`` runs in this
        process after each job ends (the next jobs start after it returns)."""
        pending = deque(enumerate(jobs))
        free = list(range(len(slots)))
        running, results = {}, {}
        last_status = time.monotonic()
        while pending or running:
            while pending and free:
                index, job = pending.popleft()
                slot_index = free.pop(0)
                process, log, started = self._launch(job, slots[slot_index])
                running[index] = (slot_index, (job, process, log, started))
            for index in list(running):
                slot_index, (job, process, log, started) = running[index]
                code = process.poll()
                if code is None:
                    continue
                del running[index]
                free.append(slot_index)
                free.sort()
                if code != 0:
                    tail = _tail(log)
                    reason = f"exit code {code}; log {log}" + (f"; last line: {tail}" if tail else "")
                else:
                    problem = job.check()
                    reason = None if problem is None else f"exit code 0 but {problem}; log {log}"
                results[index] = (reason is None, reason, log)
                self.out(f"[{datetime.now():%H:%M:%S}] {'FINISHED' if reason is None else 'ERROR'} {job.label} pid "
                         f"{process.pid} ({int(time.monotonic() - started)} s)" + (f": {reason}" if reason else ""))
                if on_done is not None:
                    on_done(job, reason is None)
            if running and time.monotonic() - last_status >= self.status_every:
                view = {i: (slots[s], rest) for i, (s, rest) in running.items()}
                self._status(view, len(pending), sum(r[0] for r in results.values()),
                             sum(not r[0] for r in results.values()))
                last_status = time.monotonic()
            if pending or running:
                time.sleep(self.poll)
        return results


def _tail(path: Path) -> str:
    try:
        lines = [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
    except OSError as exc:
        return f"(log unreadable: {exc})"
    return lines[-1][:300] if lines else ""


def default_commands() -> dict:
    python = [sys.executable, "-u", "-m"]
    return {"embed": python + ["downstream.embed"], "probe": python + ["downstream.probe"],
            "ledger": python + ["downstream.ledger"], "report": python + ["downstream.ledger_report"],
            "compare": python + ["downstream.ledger_compare"], "cf": python + ["core.evaluate_pretrain"],
            "finetune": python + ["downstream.finetune"]}


def _thread_env(threads: int) -> dict:
    return {name: str(threads) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}


def build_jobs(plan: dict, model: ModelPlan, *, commands: dict, outputs_root: Path, threads: int,
               random_init: bool) -> tuple[list, list, list]:
    embed_jobs, probe_jobs, finetune_jobs = [], [], []
    common = ["--threads", str(threads), "--outputs-root", str(outputs_root)]
    for unit, splits in plan["embed"]:
        def argv(slot, unit=unit, splits=splits):
            return (commands["embed"] + common + ["--task", unit.task, "--model", model.name, "--device", "cuda:0"]
                    + (["--random-init"] if random_init else []) + unit.cli() + ["--splits", *splits])

        def check(unit=unit, splits=splits):
            absent = [str(p) for s in splits for p in (registry.embeddings_path(unit.task, model.model_id,
                                                                                unit.prefix + s, root=outputs_root),)
                      if not (p.is_file() and p.with_suffix(".json").is_file())]
            return f"embedding files missing: {absent}" if absent else None
        embed_jobs.append(Job("embed", unit, argv, lambda slot: {"CUDA_VISIBLE_DEVICES": slot, **_thread_env(threads)},
                              check))
    for unit, head, seeds in plan["probe"]:
        pooling = plan["info"][unit]["pooling"]

        def argv(slot, unit=unit, head=head, seeds=seeds):
            return (commands["probe"] + common + ["--task", unit.task, "--model-id", model.model_id, "--head", head,
                                                  "--seeds", *map(str, seeds)] + unit.cli())

        def check(unit=unit, head=head, seeds=seeds, pooling=pooling):
            absent = [s for s in seeds if not (registry.results_dir(unit.task, model.model_id, unit.protocol(head), s,
                                                                    pooling=pooling, root=outputs_root)
                                               / "metrics.json").is_file()]
            return f"metrics.json missing for seeds {absent}" if absent else None
        probe_jobs.append(Job(f"probe_{head}", unit, argv, lambda slot: {"CUDA_VISIBLE_DEVICES": "",
                                                                         **_thread_env(threads)}, check, head))
    for unit, seeds in plan["finetune"]:
        pooling = plan["info"][unit]["finetune_pooling"]

        def argv(slot, unit=unit, seeds=seeds):
            return (commands["finetune"] + common + ["--task", unit.task, "--model", model.name, "--device", "cuda:0",
                                                     "--seeds", *map(str, seeds)]
                    + (["--random-init"] if random_init else []) + (["--split-tag", unit.split_tag] if unit.split_tag
                                                                    else []))

        def check(unit=unit, seeds=seeds, pooling=pooling):
            absent = [s for s in seeds if not (registry.results_dir(unit.task, model.model_id, unit.protocol(FINETUNE),
                                                                    s, pooling=pooling, root=outputs_root)
                                               / "metrics.json").is_file()]
            return f"metrics.json missing for seeds {absent}" if absent else None
        finetune_jobs.append(Job(FINETUNE, unit, argv, lambda slot: {"CUDA_VISIBLE_DEVICES": slot,
                                                                     **_thread_env(threads)}, check, FINETUNE,
                                 seconds=finetune_seconds(plan, unit, seeds)))
    return embed_jobs, probe_jobs, finetune_jobs


def summary(results_root: Path, model: ModelPlan, plan: dict, heads: list, failures: list) -> tuple[str, list]:
    """Markdown table of the primary metric per unit x head, plus the missing required cells of this model."""
    state = ledger_report.load_state(results_root)
    catalog = state["catalog"]
    failed_units = {}
    for label, reason in failures:
        kind, _, key = label.partition(" ")
        failed_units.setdefault(key, []).append((kind, f"{kind}: {reason}"))
    lines = [f"# ds-run: {model.model_id}", "", f"Ledger: {results_root} (measurements sha256 "
             f"{state['header']['measurements_sha256']}); cell = mean ± std (ddof={ledger_report.STD_DDOF}) over "
             "seeds.", "", f"Primary metric only; all catalog metrics of a task are in "
             f"{results_root / 'reports' / 'tasks'}/<task>.md (written by the ledger report at the end of the run).", "",
             "| task | protocol | primary metric | value |", "|---|---|---|---|"]
    shown = {}  # (task, protocol) -> its units: the tags of a per-seed-split task share one row
    for unit in plan["units"]:
        for head in heads:
            shown.setdefault((unit.task, unit.protocol(head), head), []).append(unit)
    for (task, protocol, head), group in shown.items():
        spec = catalog["tasks"][task]
        primary = spec["primary"]
        found = ledger_report.lookup(state, task, model.model_id, protocol, primary)
        text = found["text"]
        kinds = (FINETUNE,) if head == FINETUNE else ("embed", f"probe_{head}")
        problems = [text for unit in group for kind, text in failed_units.get(unit.key, []) if kind in kinds]
        if problems and not found["filled"]:
            text = "ERROR: " + " | ".join(problems)
        label = group[0].key if len(group) == 1 else task
        lines.append(f"| {label} | {protocol} | {primary['metric']} ({primary['eval_split']}/{primary['subset']}) "
                     f"| {text} |")
    for unit, reason in plan["domain_skips"]:
        lines.append(f"| {unit.key} | - | - | SKIPPED: {reason} |")
    tasks = sorted({unit.task for unit in plan["units"]})
    missing = ledger_report.missing_cells(state, [model.model_id], tasks) if tasks else []
    counts = Counter(entry["category"] for entry in missing)
    lines += ["", f"## Missing required cells ({len(missing)})", "",
              "Categories: " + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none"), ""]
    lines += [f"- {e['task']} / {e['protocol_id']} / {e['eval_split']} / {e['subset']} / {e['metric']}: {e['category']}"
              for e in missing]
    return "\n".join(lines) + "\n", missing


CF_MODES = ("none", "masked_pos_noise", "masked_pos_random", "masked_dist_shuffle", "esa_edges_remove",
            "esa_edges_shuffle", "masked_edges_remove")  # Makefile CF_MODES (eval-counterfactual)


def resolve_compare_to(text: str, model: ModelPlan, *, outputs_root: Path, results_root: Optional[Path]) -> list:
    """``(ModelPlan, random_init)`` of every COMPARE_TO entry; ``<snapshot>.random_init`` = RANDOM_INIT=1."""
    refs = [t.strip() for t in text.split(",") if t.strip()]
    if len(set(refs)) != len(refs):
        raise PlanError(f"COMPARE_TO={text}: a model is listed twice")
    found = []
    for ref in refs:
        random_init = ref.endswith(registry.RANDOM_INIT_SUFFIX)
        base = ref[: -len(registry.RANDOM_INIT_SUFFIX)] if random_init else ref
        other = resolve_model(base, None, random_init=random_init, outputs_root=outputs_root, results_root=results_root)
        if other.model_id == model.model_id:
            raise PlanError(f"COMPARE_TO={ref} is the model itself ({model.model_id})")
        found.append((other, random_init))
    return found


def check_counterfactual(model: ModelPlan, *, random_init: bool, modes: list) -> None:
    """COUNTERFACTUAL=1 mirrors ``make eval-counterfactual`` on the checkpoint MODEL names (a pretraining ckpt)."""
    if model.ckpt is None:
        raise PlanError(f"COUNTERFACTUAL=1 needs MODEL=<path/to.ckpt> (a pretraining checkpoint); MODEL={model.name} "
                        "is a snapshot name")
    if random_init:
        raise PlanError("COUNTERFACTUAL=1 evaluates the pretrained checkpoint; it cannot go with RANDOM_INIT=1")
    makefile = registry.REPO_ROOT / "Makefile"
    if "\neval-counterfactual:" not in (makefile.read_text() if makefile.is_file() else ""):
        raise PlanError(f"COUNTERFACTUAL=1: {makefile} has no eval-counterfactual target to mirror")
    if not (registry.REPO_ROOT / "core" / "evaluate_pretrain.py").is_file():
        raise PlanError("COUNTERFACTUAL=1: core/evaluate_pretrain.py is missing in this checkout")
    unknown = [m for m in modes if m not in CF_MODES]
    if not modes or unknown:
        raise PlanError(f"CF_MODES {unknown or modes}: choose from {' '.join(CF_MODES)}")


def extra_plan_text(model: ModelPlan, others: list, args, gpus: list, outputs_root: Path) -> str:
    lines = ["", "Then (when the embeddings of all models have finished):"]
    if args.counterfactual:
        target = outputs_root / "eval" / f"cf_{model.name}_<stamp>"
        lines += [f"   WILL RUN  counterfactual {mode}: core.evaluate_pretrain --ckpt "
                  f"{outputs_root / 'models' / model.name / 'model.ckpt'} --split {args.cf_split} --seed {args.cf_seed} "
                  f"--domains {args.cf_domains} -> {target}/{mode} (same arguments as make eval-counterfactual; "
                  f"GPU {','.join(gpus)}, 1 per GPU)" for mode in args.cf_modes]
    lines.append("   probes, then one ledger add" + (" (+ counterfactual dizinleri)" if args.counterfactual else ""))
    lines += [f"   WILL RUN  compare A={other.model_id} B={model.model_id} (bootstrap {args.bootstrap})"
              for other, _ in others]
    lines.append("   WILL RUN  ledger report (same as make ds-ledger-report: tasks/, timeline/, summary.md, "
                 "missing_cells.*)")
    return "\n".join(lines) + "\n"


def counterfactual_jobs(model: ModelPlan, args, *, commands: dict, outputs_root: Path) -> tuple[list, list]:
    """One job per mode, the arguments of ``make eval-counterfactual`` (one GPU per job)."""
    ckpt = outputs_root / "models" / model.name / "model.ckpt"
    root = outputs_root / "eval" / f"cf_{model.name}_{_stamp()}"
    jobs, dirs = [], []
    for mode in args.cf_modes:
        out = root / mode

        def argv(slot, mode=mode, out=out):
            return (commands["cf"] + ["--threads", str(args.threads), "--ckpt", str(ckpt), "--split", args.cf_split,
                                      "--seed", str(args.cf_seed), "--output-dir", str(out), "--device", "cuda:0",
                                      "--domains", args.cf_domains, "--counterfactual", mode]
                    + (["--config-yaml-path", args.cf_eval_config] if args.cf_eval_config else []))

        def check(out=out):
            return None if (out / "metrics.json").is_file() else f"{out / 'metrics.json'} missing"
        jobs.append(Job("counterfactual", Unit(f"{model.name}.{mode}"), argv,
                        lambda slot: {"CUDA_VISIBLE_DEVICES": slot, **_thread_env(args.threads)}, check,
                        model_id=model.model_id))
        dirs.append(out)
    return jobs, dirs


def _cpu_job(kind: str, key: str, argv: list, model_id: str) -> Job:
    return Job(kind, Unit(key), lambda slot: argv, lambda slot: {"CUDA_VISIBLE_DEVICES": ""}, lambda: None,
               model_id=model_id)


def main(argv=None, *, commands: Optional[dict] = None, describe: Optional[Callable] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="snapshot name/alias, or a .ckpt path (then --name)")
    parser.add_argument("--name", default=None, help="snapshot name for a checkpoint path")
    parser.add_argument("--tasks", default="all", help="'all' or a comma list (lba = its required split tags)")
    parser.add_argument("--gpus", default="", help="comma list of GPU indices for the embeddings, e.g. 2,3")
    parser.add_argument("--per-gpu", type=int, default=1, help="concurrent embedding jobs per GPU")
    parser.add_argument("--ft-per-gpu", type=int, default=1, help="concurrent fine-tune jobs per GPU")
    parser.add_argument("--heads", default="linear,mlp", help=f"comma list of {', '.join(PROTOCOL_CHOICES)}")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--probe-jobs", type=int, default=1, help="concurrent probe jobs (CPU)")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads of every child process")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--compare-to", default="", help="comma list of models (snapshot[.random_init]) completed the "
                                                         "same way, then compared: ds-compare A=<each> B=<model>")
    parser.add_argument("--bootstrap", type=int, default=1000, help="paired bootstrap resamples of each compare")
    parser.add_argument("--counterfactual", action="store_true", help="also the eval-counterfactual sweep (ckpt path)")
    parser.add_argument("--cf-modes", nargs="+", default=list(CF_MODES))
    parser.add_argument("--cf-split", default="test", choices=("val", "test"))
    parser.add_argument("--cf-seed", type=int, default=0)
    parser.add_argument("--cf-domains", default="unimol")
    parser.add_argument("--cf-eval-config", default=None, help="core/<EVAL_CONFIG>.yaml (default: the ckpt's config)")
    parser.add_argument("--outputs-root", type=Path, default=None, help="default: $MIND_OUTPUTS_ROOT or <repo>/outputs")
    parser.add_argument("--results-root", type=Path, default=None, help=f"default: ${ledger.ENV_VAR}")
    parser.add_argument("--log-root", type=Path, default=Path("logs") / datetime.now().strftime("%B%Y"))
    parser.add_argument("--plan", action="store_true", help="print what would run and stop (writes nothing)")
    parser.add_argument("--status-every", type=float, default=300.0, help="seconds between status lines")
    parser.add_argument("--poll", type=float, default=5.0, help="seconds between process checks")
    args = parser.parse_args(argv)
    commands = {**default_commands(), **(commands or {})}
    try:
        heads = [h.strip() for h in args.heads.split(",") if h.strip()]
        if not heads or any(h not in PROTOCOL_CHOICES for h in heads) or len(set(heads)) != len(heads):
            raise PlanError(f"HEADS={args.heads}: choose from {PROTOCOL_CHOICES}, each at most once")
        if min(args.per_gpu, args.ft_per_gpu, args.probe_jobs, args.threads, args.bootstrap) < 1:
            raise PlanError("--per-gpu, --ft-per-gpu, --probe-jobs, --threads and --bootstrap must be at least 1")
        if args.model.endswith(registry.RANDOM_INIT_SUFFIX) and not args.model.endswith(".ckpt"):
            args.model, args.random_init = args.model[: -len(registry.RANDOM_INIT_SUFFIX)], True
        gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
        outputs_root = registry.outputs_root(args.outputs_root).resolve()
        results_root = ledger.results_root(args.results_root)
        catalog = ledger.load_catalog()
        units, explicit = expand_tasks(args.tasks, catalog, args.seeds)
        model = resolve_model(args.model, args.name, random_init=args.random_init, outputs_root=outputs_root,
                              results_root=results_root)
        others = resolve_compare_to(args.compare_to, model, outputs_root=outputs_root, results_root=results_root)
        if args.counterfactual:
            check_counterfactual(model, random_init=args.random_init, modes=args.cf_modes)
        plan = plan_units(units, explicit, model, heads=heads, seeds=args.seeds, outputs_root=outputs_root,
                          describe=describe)
        runs = [(model, plan, args.random_init)] + [
            (other, plan_units(units, explicit, other, heads=heads, seeds=args.seeds, outputs_root=outputs_root,
                               describe=describe), random_init) for other, random_init in others]
        n_embed = sum(len(p["embed"]) for _, p, _ in runs)
        n_finetune = sum(len(p["finetune"]) for _, p, _ in runs)
        if (n_embed or n_finetune or args.counterfactual) and (not gpus or not all(g.isdigit() for g in gpus)):
            raise PlanError(f"{n_embed} embedding job(s), {n_finetune} fine-tune job(s)"
                            + (" and the counterfactual sweep" if args.counterfactual else "")
                            + f" to run but GPU={args.gpus!r}: give GPU indices like GPU=2,3 (ds-run does not pick "
                            "GPUs itself)")
    except (PlanError, ledger.LedgerError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    texts = []
    for index, (m, p, _) in enumerate(runs):
        text = plan_text(m, p, outputs_root=outputs_root, results_root=results_root, gpus=gpus,
                         per_gpu=args.per_gpu, probe_jobs=args.probe_jobs, heads=heads, seeds=args.seeds,
                         ft_per_gpu=args.ft_per_gpu)
        texts.append(text if index == 0 else f"=== COMPARE_TO {m.model_id} ===\n{text}")
    if others or args.counterfactual:
        texts.append(extra_plan_text(model, others, args, gpus, outputs_root))
    text = "\n".join(texts)
    print(text, flush=True)
    if args.plan:
        print("PLAN: nothing was run, no file was written.")
        return 1 if any(p["failed"] for _, p, _ in runs) else 0

    run_dir = Path(args.log_root) / f"ds_run_{model.model_id}_{_stamp()}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "plan.txt").write_text(text)
    print(f"Run directory: {run_dir.resolve()}", flush=True)
    failures = {m.model_id: list(p["failed"]) for m, p, _ in runs}  # per model, labels "<kind> <unit key>"
    extra_failures = []  # counterfactual, ledger add, compare, report
    if model.action == "create snapshot":
        destination = registry.snapshot_model(model.ckpt, model.name, root=outputs_root)
        card = json.loads((destination / "card.json").read_text())
        if card["sha256"] != model.sha256:
            raise RuntimeError(f"snapshot {destination} has sha256 {card['sha256']}, the plan hashed {model.sha256}")
        print(f"snapshot: {destination}", flush=True)
    embed_jobs, probe_jobs, finetune_jobs = [], [], []
    for m, p, random_init in runs:
        embeds, probes, finetunes = build_jobs(p, m, commands=commands, outputs_root=outputs_root,
                                               threads=args.threads, random_init=random_init)
        for job in embeds + probes + finetunes:
            job.model_id = m.model_id
        embed_jobs += embeds
        probe_jobs += probes
        finetune_jobs += finetunes
    runner = Runner(run_dir, model.model_id, status_every=args.status_every, poll=args.poll,
                    out=lambda message: print(message, flush=True))
    embed_failed = set()
    slots = [g for _ in range(args.per_gpu) for g in gpus]  # interleaved: jobs spread over the GPUs first
    for index, (ok, reason, _) in runner.run(embed_jobs, slots).items():
        if not ok:
            job = embed_jobs[index]
            failures[job.model_id].append((job.label, reason))
            embed_failed.add((job.model_id, job.unit))
    eval_dirs = []
    if args.counterfactual:
        cf_jobs, cf_dirs = counterfactual_jobs(model, args, commands=commands, outputs_root=outputs_root)
        for index, (ok, reason, _) in runner.run(cf_jobs, gpus).items():
            if ok:
                eval_dirs.append(cf_dirs[index])
            else:
                extra_failures.append((cf_jobs[index].label, reason))
    # Probes need the embedding: only an embedding that could not run blocks them (not a fine-tune problem).
    failed_before = {m.model_id: {label.partition(" ")[2] for label, _ in p["failed"]
                                  if label.partition(" ")[0] == "embed"} for m, p, _ in runs}
    runnable = []
    for job in probe_jobs:
        if (job.model_id, job.unit) in embed_failed or job.unit.key in failed_before[job.model_id]:
            failures[job.model_id].append((f"{job.kind} {job.unit.key}", "not run: the embedding of this task failed"))
        else:
            runnable.append(job)
    def ledger_add(key: str, extra: list) -> Job:
        return _cpu_job("ledger_add", key, commands["ledger"] + ["--results-root", str(results_root), "add",
                                                                 "--outputs-root", str(outputs_root)] + extra,
                        model.model_id)

    def ingest_finished(job: Job, ok: bool) -> None:
        """Add the results of a finished job to the ledger now (sequential, in this process; re-adding is a no-op)."""
        if not ok:
            return
        (added, reason, _), = runner.run([ledger_add(f"after_{job.kind}_{job.unit.key}", [])], ["cpu0"]).values()
        if not added:
            extra_failures.append((f"ledger_add after {job.label}", reason))

    for index, (ok, reason, _) in runner.run(runnable, [f"cpu{i}" for i in range(args.probe_jobs)],
                                             on_done=ingest_finished).items():
        if not ok:
            failures[runnable[index].model_id].append((runnable[index].label, reason))
    finetune_jobs.sort(key=lambda job: -job.seconds)  # longest first; the small ones fill the gaps
    ft_slots = [g for _ in range(args.ft_per_gpu) for g in gpus]  # interleaved like the embedding slots
    for index, (ok, reason, _) in runner.run(finetune_jobs, ft_slots, on_done=ingest_finished).items():
        if not ok:
            failures[finetune_jobs[index].model_id].append((finetune_jobs[index].label, reason))
    add = ledger_add("all", [item for d in eval_dirs for item in ("--eval-dir", str(d))])
    (ok, reason, _), = runner.run([add], ["cpu0"]).values()
    if not ok:
        extra_failures.append(("ledger_add all", reason))
    tables, missing = [], []
    for m, p, _ in runs:
        try:
            table, cells = summary(results_root, m, p, heads, failures[m.model_id])
        except Exception as exc:  # noqa: BLE001 - the summary failing must not hide the job failures listed below
            table, cells = f"# ds-run: {m.model_id}\n\nSummary table could not be produced: {type(exc).__name__}: {exc}\n", []
            extra_failures.append((f"summary {m.model_id}", str(exc)))
        tables.append(table)
        missing += cells
    final = [_cpu_job("compare", f"{other.model_id}__vs__{model.model_id}",
                      commands["compare"] + ["--a", other.model_id, "--b", model.model_id, "--bootstrap",
                                             str(args.bootstrap), "--results-root", str(results_root)], model.model_id)
             for other, _, _ in runs[1:]]
    final.append(_cpu_job("ledger_report", "all", commands["report"] + ["report", "--results-root", str(results_root)],
                          model.model_id))
    for index, (ok, reason, _) in runner.run(final, ["cpu0"]).items():
        if not ok:
            extra_failures.append((final[index].label, reason))
    table = "\n".join(tables)
    listed = failures[model.model_id] + [(f"[{m.model_id}] {label}", reason) for m, _, _ in runs[1:]
                                         for label, reason in failures[m.model_id]] + extra_failures
    if listed:
        table += f"\n## MISSING: {len(listed)} job(s) failed or could not run\n\n" + "".join(
            f"- {label}: {reason}\n" for label, reason in listed)
    (run_dir / "summary.md").write_text(table)
    with open(run_dir / "missing.csv", "w", newline="") as handle:
        if missing:
            writer = csv.DictWriter(handle, fieldnames=list(missing[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(missing)
    print("\n" + table + f"\n(summary: {run_dir.resolve() / 'summary.md'}; ledger reports: {results_root / 'reports'})",
          flush=True)
    return 1 if listed else 0


if __name__ == "__main__":
    raise SystemExit(main())
