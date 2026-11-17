"""ECFP4 fingerprint baseline for the MoleculeNet tasks (CPU only).

    python -m downstream.baselines.ecfp --tasks all --heads linear,mlp --seeds 0 1 2 [--models probe|tuned|both]
        [--feature bits|counts] [--n-bits 2048] [--threads 8] [--jobs 1] [--outputs-root DIR] [--log-root DIR]
        [--no-ledger] [--plan]

The baseline is a frozen-feature probe whose feature is a Morgan fingerprint of the stored SMILES (radius 2 =
ECFP4) instead of a pooled encoder embedding. Everything after the feature is the code of the MIND probes:
``downstream.probe`` (linear or MLP head, masked BCE / masked MSE on train-standardized targets, the same epoch
selection on the validation metric, the same evaluation functions and the same ``task.yaml`` probe settings), run
once per seed, on the Uni-Mol scaffold split of the processed MoleculeNet files.

Steps per task (what already exists is skipped, so an interrupted run can be repeated):

1. Fingerprints of every split, written like an encoder embedding to
   ``<outputs root>/downstream/<task>/ecfp4/embeddings/<split>.{npz,json}`` with pooling name
   ``ecfp4_<variant><n_bits>`` (``ecfp4_bits2048``). The json record carries the RDKit version, radius, size and
   variant. A SMILES RDKit cannot parse raises (it is never replaced by an all-zero row).
2. ``--models probe`` (model ``ecfp4``): ``python -m downstream.probe`` per head with the seeds that have no
   ``metrics.json`` yet. ``--models tuned`` (model ``ecfp4_tuned``): ``downstream.baselines.ecfp_tuned``, the
   validation-tuned scikit-learn logistic / ridge regression on the same fingerprints, one seed. ``both`` (default)
   runs both; ``--jobs`` processes at a time, the largest tasks first.
3. ``downstream.ledger add`` of the outputs root and the ledger reports (``$MIND_RESULTS_ROOT``), unless
   ``--no-ledger``; the final table lists the primary metric per task and head.

The probe runs with ``--feature-scaling none``: encoder embeddings are standardized with train statistics, but for a
sparse bit vector that would turn a bit set in one train molecule into a value of about 40 (and a bit never set in
train into 1e6 when a test molecule has it); in a check on ESOL the standardized linear head was worse than the
train-mean predictor. ``downstream.probe`` and the ledger refuse the two combinations that would mix the settings (a
fingerprint standardized, an embedding unscaled).

Count fingerprints (``--feature counts``) are stored as uint16 and refused above 65535. No GPU is used: the
processes run with ``CUDA_VISIBLE_DEVICES=""``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from downstream import ledger, ledger_report
from downstream.common.registry import REPO_ROOT, embeddings_path, git_commit, outputs_root, results_dir, sha256_file
from downstream.tasks import MOLNET_TASKS, load_task

MODEL_ID = "ecfp4"
TUNED_MODEL_ID = "ecfp4_tuned"  # downstream.baselines.ecfp_tuned
TUNED_PROTOCOL = "sklearn_tuned"
TUNED_SEED = 0
MODEL_CHOICES = ("probe", "tuned", "both")
RADIUS = 2  # ECFP4: diameter 4 = radius 2
VARIANTS = ("bits", "counts")
DEFAULT_BITS = 2048
HEADS = ("linear", "mlp")  # downstream.common.heads.HEADS (not imported: that pulls torch into the plan step)
MAX_COUNT = np.iinfo(np.uint16).max
FEATURE_SCALING = "none"  # the probe's --feature-scaling for fingerprints (see the module docstring)


@dataclass(frozen=True)
class FingerprintSpec:
    """One fingerprint definition; ``pooling`` names it in the results path and in the ledger."""

    variant: str = "bits"
    n_bits: int = DEFAULT_BITS

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"fingerprint variant must be one of {VARIANTS}, got {self.variant!r}")
        if not isinstance(self.n_bits, int) or isinstance(self.n_bits, bool) or self.n_bits < 8:
            raise ValueError(f"n_bits must be an integer of at least 8, got {self.n_bits!r}")

    @property
    def pooling(self) -> str:
        return f"ecfp{2 * RADIUS}_{self.variant}{self.n_bits}"

    def describe(self) -> dict:
        """Everything that defines the feature; written to every embedding record and compared on reuse."""
        import rdkit

        return {"algorithm": "Morgan fingerprint (ECFP4)", "radius": RADIUS, "n_bits": self.n_bits,
                "variant": self.variant, "rdkit_version": rdkit.__version__, "include_chirality": False,
                "use_bond_types": True, "generator": "rdFingerprintGenerator.GetMorganGenerator",
                "folding": "hashed identifiers modulo n_bits",
                "invalid_smiles": "refused (ValueError)", "dtype": "uint8" if self.variant == "bits" else "uint16"}


def fingerprint_matrix(smiles: Sequence[str], ids: Sequence[str], spec: FingerprintSpec) -> np.ndarray:
    """``[N, n_bits]`` fingerprints (uint8 bits, or uint16 counts); an unparsable SMILES raises naming its id."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    if len(smiles) != len(ids):
        raise ValueError(f"{len(smiles)} SMILES but {len(ids)} ids")
    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=spec.n_bits)
    out = np.zeros((len(smiles), spec.n_bits), dtype=np.uint8 if spec.variant == "bits" else np.uint16)
    invalid = []
    for row, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            invalid.append(f"{ids[row]} ({text!r})")
            continue
        if spec.variant == "bits":
            out[row] = generator.GetFingerprintAsNumPy(molecule)
        else:
            counts = generator.GetCountFingerprintAsNumPy(molecule)
            if int(counts.max()) > MAX_COUNT:
                raise ValueError(f"{ids[row]}: a fingerprint count of {int(counts.max())} exceeds {MAX_COUNT}")
            out[row] = counts
    if invalid:
        raise ValueError(f"RDKit cannot parse {len(invalid)} of {len(smiles)} SMILES, first: {invalid[:5]}")
    return out


# ---------------------------------------------------------------------------------------------------------
# Embedding files
# ---------------------------------------------------------------------------------------------------------

def _config(task_name: str) -> dict:
    task = load_task(task_name)
    return task.load_task_config(task.TASK_YAML)


def _processed_paths(task_name: str, config: dict) -> dict:
    task = load_task(task_name)
    directory = Path(task.processed_dir(config))
    return {split: directory / f"{split}.pt" for split in config["splits"]}


def embedding_state(task_name: str, spec: FingerprintSpec, root: Path) -> tuple[list, list]:
    """``(splits to write, splits done)``; a half-written pair or a record of another definition raises."""
    config = _config(task_name)
    missing, done = [], []
    for split in config["splits"]:
        npz = embeddings_path(task_name, MODEL_ID, split, root=root)
        record_path = npz.with_suffix(".json")
        if npz.is_file() and record_path.is_file():
            record = json.loads(record_path.read_text())
            wanted = spec.describe()
            found = {key: (record.get("fingerprint") or {}).get(key) for key in wanted}
            if found != wanted:
                raise RuntimeError(f"{record_path} was written with fingerprint {found}, this run asks for {wanted}; "
                                   "use another outputs root")
            done.append(split)
        elif npz.is_file() or record_path.is_file():
            raise RuntimeError(f"half-written embedding {npz} (.npz without .json or the reverse); inspect it by hand")
        else:
            missing.append(split)
    return missing, done


def write_task_embeddings(task_name: str, spec: FingerprintSpec, root: Path) -> list:
    """Fingerprints of every split of ``task_name``; returns the json records written. Never overwrites."""
    task = load_task(task_name)
    config = _config(task_name)
    paths = _processed_paths(task_name, config)
    started = datetime.now(timezone.utc).isoformat()
    commit = git_commit()
    records = []
    for name, path in paths.items():
        timer = time.monotonic()
        out = embeddings_path(task_name, MODEL_ID, name, root=root)
        if out.exists() or out.with_suffix(".json").exists():
            raise FileExistsError(f"refusing to overwrite {out}")
        sha = sha256_file(path)
        split = task.ProcessedSplit(path)
        if not hasattr(split, "smiles"):
            raise ValueError(f"{task_name}: the processed split has no SMILES; a fingerprint needs them")
        matrix = fingerprint_matrix(split.smiles, split.ids, spec)
        out.parent.mkdir(parents=True, exist_ok=True)
        partial = out.with_name(out.stem + ".partial.npz")
        np.savez_compressed(partial, chains=np.array(split.chains), labels=np.asarray(split.labels),
                            **{f"emb_{spec.pooling}": matrix})
        partial.replace(out)
        record = {"task": task_name, "started_utc": started, "git": commit, "device": "cpu", "split": name,
                  "split_tag": None, "representation": None, "model_id": MODEL_ID, "fingerprint": spec.describe(),
                  "processed": str(path), "processed_sha256": sha, "chains": len(split.chains),
                  "poolings": [spec.pooling], "dim": spec.n_bits, "seconds": round(time.monotonic() - timer, 1)}
        out.with_suffix(".json").write_text(json.dumps(record, indent=1))
        records.append(record)
    return records


# ---------------------------------------------------------------------------------------------------------
# Plan and run
# ---------------------------------------------------------------------------------------------------------

def expand_tasks(text: str) -> list:
    """``all`` / ``molnet`` or a comma list of ``mn_*`` tasks; any other task is refused (it has no SMILES)."""
    names = [t.strip() for t in text.split(",") if t.strip()]
    if not names:
        raise ValueError("--tasks is empty")
    expanded = []
    for name in names:
        expanded += list(MOLNET_TASKS) if name in ("all", "molnet") else [name]
    unknown = [t for t in expanded if t not in MOLNET_TASKS]
    if unknown:
        raise ValueError(f"the fingerprint baseline covers the MoleculeNet tasks only; unknown or unsupported: "
                         f"{unknown}. Choose from {', '.join(MOLNET_TASKS)}")
    return list(dict.fromkeys(expanded))


def tuned_metrics_path(task_name: str, spec: FingerprintSpec, root: Path) -> Path:
    return results_dir(task_name, TUNED_MODEL_ID, TUNED_PROTOCOL, TUNED_SEED, pooling=spec.pooling, root=root) / "metrics.json"


def plan_task(task_name: str, spec: FingerprintSpec, heads: Sequence[str], seeds: Sequence[int], root: Path,
              models: str = "both") -> dict:
    config = _config(task_name)
    missing, _ = embedding_state(task_name, spec, root)
    probes = []
    for head in heads if models in ("probe", "both") else ():
        todo = [s for s in seeds if not (results_dir(task_name, MODEL_ID, f"{head}_probe", s, pooling=spec.pooling,
                                                     root=root) / "metrics.json").is_file()]
        if todo:
            probes.append((head, todo))
    paths = _processed_paths(task_name, config)
    absent = [str(p) for p in paths.values() if not p.is_file()]
    tuned = models in ("tuned", "both") and not tuned_metrics_path(task_name, spec, root).is_file()
    return {"task": task_name, "embed_splits": missing, "probes": probes, "tuned": tuned, "no_data": absent,
            "train_molecules": config["expected_counts"]["train"] * config["num_targets"]}


def plan_text(plans: list, spec: FingerprintSpec, root: Path, results: Optional[Path], jobs: int, threads: int) -> str:
    lines = [f"ECFP4 baseline: model id {MODEL_ID}, pooling {spec.pooling}, {spec.describe()}",
             f"outputs root {root}", f"results root {results}", f"{jobs} job(s) at a time, {threads} thread(s) (or fit workers) each", ""]
    for plan in plans:
        todo = "; ".join(f"{head}: seeds {seeds}" for head, seeds in plan["probes"]) or "all probes done"
        todo += "; tuned sklearn: seed 0" if plan["tuned"] else "; tuned sklearn done or not requested"
        embed = f"fingerprints of {plan['embed_splits']}" if plan["embed_splits"] else "fingerprints done"
        data = f"  NO PROCESSED DATA: {plan['no_data']}" if plan["no_data"] else ""
        lines.append(f"{plan['task']}: {embed}; {todo}{data}")
    return "\n".join(lines)


def _thread_env(threads: int) -> dict:
    return {name: str(threads) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}


def _probe_command(task_name: str, head: str, seeds: Sequence[int], spec: FingerprintSpec, root: Path,
                   threads: int) -> list:
    return [sys.executable, "-u", "-m", "downstream.probe", "--task", task_name, "--model-id", MODEL_ID, "--head", head,
            "--pooling", spec.pooling, "--seeds", *map(str, seeds), "--threads", str(threads), "--outputs-root", str(root),
            "--feature-scaling", FEATURE_SCALING]


def _tuned_command(task_name: str, spec: FingerprintSpec, root: Path, threads: int) -> list:
    return [sys.executable, "-u", "-m", "downstream.baselines.ecfp_tuned", "--task", task_name, "--feature", spec.variant,
            "--n-bits", str(spec.n_bits), "--threads", str(threads), "--outputs-root", str(root)]


def _run_probe(label: str, argv: list, log: Path, env_threads: int, expected: list) -> Optional[str]:
    """Run one probe or tuned-fit process (``env_threads`` BLAS/OpenMP threads); ``None`` on success, else the reason."""
    with open(log, "w") as handle:
        started = time.monotonic()
        code = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, cwd=REPO_ROOT,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": "", **_thread_env(env_threads)}).returncode
    seconds = int(time.monotonic() - started)
    missing = [str(path) for path in expected if not path.is_file()]
    if code != 0 or missing:
        tail = [line for line in log.read_text(errors="replace").splitlines() if line.strip()][-1:]
        reason = f"exit code {code}" + (f", metrics.json missing: {missing}" if missing else "") + f"; log {log}" \
                 + (f"; last line: {tail[0][:300]}" if tail else "")
        print(f"[{datetime.now():%H:%M:%S}] ERROR {label} ({seconds} s): {reason}", flush=True)
        return reason
    print(f"[{datetime.now():%H:%M:%S}] FINISHED {label} ({seconds} s)", flush=True)
    return None


def summary_table(results: Path, tasks: Sequence[str], heads: Sequence[str], models: str = "both") -> str:
    """Primary metric per task and head from the ledger (what ``make ds-summary`` shows for any model)."""
    state = ledger_report.load_state(results)
    catalog = state["catalog"]
    lines = [f"# ECFP4 baseline ({MODEL_ID})", "", f"Ledger: {results}; cell = mean ± std over seeds.", "",
             "| task | model / protocol | primary metric | value |", "|---|---|---|---|"]
    for task_name in tasks:
        primary = catalog["tasks"][task_name]["primary"]
        wanted = [(MODEL_ID, f"{head}_probe") for head in heads if models in ("probe", "both")]
        wanted += [(TUNED_MODEL_ID, TUNED_PROTOCOL)] if models in ("tuned", "both") else []
        for model, protocol in wanted:
            found = ledger_report.lookup(state, task_name, model, protocol, primary)
            lines.append(f"| {task_name} | {model} / {protocol} | {primary['metric']} "
                         f"({primary['eval_split']}/{primary['subset']}) | {found['text']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks", default="all", help="'all' / 'molnet' (the 14 MoleculeNet tasks) or a comma list")
    parser.add_argument("--heads", default="linear,mlp")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--feature", default="bits", choices=VARIANTS, help="bit vector (default) or count vector")
    parser.add_argument("--models", default="both", choices=MODEL_CHOICES,
                        help="probe: ecfp4 (MIND probe heads); tuned: ecfp4_tuned (scikit-learn, validation-tuned); both")
    parser.add_argument("--n-bits", type=int, default=DEFAULT_BITS)
    parser.add_argument("--threads", type=int, default=8,
                        help="CPU threads of every probe process; worker processes of every tuned fit")
    parser.add_argument("--jobs", type=int, default=1, help="concurrent probe / tuned-fit processes")
    parser.add_argument("--outputs-root", type=Path, default=None, help="default: $MIND_OUTPUTS_ROOT or <repo>/outputs")
    parser.add_argument("--results-root", type=Path, default=None, help=f"default: ${ledger.ENV_VAR}")
    parser.add_argument("--log-root", type=Path, default=Path("logs") / datetime.now().strftime("%B%Y"))
    parser.add_argument("--no-ledger", action="store_true", help="stop after the probes (no ledger add, no report)")
    parser.add_argument("--plan", action="store_true", help="print what would run and stop (writes nothing)")
    args = parser.parse_args(argv)
    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    if not heads or any(h not in HEADS for h in heads):
        parser.error(f"--heads: choose from {HEADS}")
    if args.threads < 1 or args.jobs < 1:
        parser.error("--threads and --jobs must be at least 1")
    try:
        spec = FingerprintSpec(args.feature, args.n_bits)
        tasks = expand_tasks(args.tasks)
        root = outputs_root(args.outputs_root).resolve()
        results = None if args.no_ledger and args.plan else ledger.results_root(args.results_root)
        plans = [plan_task(t, spec, heads, args.seeds, root, args.models) for t in tasks]
    except (ValueError, RuntimeError, ledger.LedgerError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    text = plan_text(plans, spec, root, results, args.jobs, args.threads)
    print(text, flush=True)
    if any(plan["no_data"] for plan in plans):
        print("ERROR: processed data is missing for the tasks above (make ds-process TASK=<task>)", file=sys.stderr)
        return 2
    if args.plan:
        print("PLAN: nothing was run, no file was written.")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.log_root) / f"ds_ecfp_{stamp}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "plan.txt").write_text(text)
    print(f"Run directory: {run_dir.resolve()}", flush=True)
    failures = []
    for plan in plans:
        if not plan["embed_splits"]:
            continue
        try:
            started = time.monotonic()
            write_task_embeddings(plan["task"], spec, root)
            print(f"[{datetime.now():%H:%M:%S}] FINISHED fingerprints {plan['task']} ({int(time.monotonic() - started)} s)",
                  flush=True)
            plan["embed_ok"] = True
        except Exception as exc:  # noqa: BLE001 - reported for this task, the others go on
            failures.append((f"fingerprints {plan['task']}", f"{type(exc).__name__}: {exc}"))
            plan["embed_ok"] = False
            print(f"[{datetime.now():%H:%M:%S}] ERROR fingerprints {plan['task']}: {exc}", flush=True)
    work = []
    for plan in plans:
        if plan.get("embed_ok") is False:
            failures += [(f"probe_{head} {plan['task']}", "not run: the fingerprints of this task failed")
                         for head, _ in plan["probes"]]
            failures += [(f"tuned {plan['task']}", "not run: the fingerprints of this task failed")] * plan["tuned"]
            continue
        for head, seeds in plan["probes"]:
            expected = [results_dir(plan["task"], MODEL_ID, f"{head}_probe", s, pooling=spec.pooling, root=root)
                        / "metrics.json" for s in seeds]
            label = f"probe_{head} {plan['task']}"
            work.append((plan["train_molecules"], label, _probe_command(plan["task"], head, seeds, spec, root, args.threads),
                         run_dir / f"{label.replace(' ', '_')}.log", expected, args.threads))
        if plan["tuned"]:  # liblinear / ridge are single-threaded; the fit workers are the threads
            label = f"tuned {plan['task']}"
            work.append((plan["train_molecules"], label, _tuned_command(plan["task"], spec, root, args.threads),
                         run_dir / f"{label.replace(' ', '_')}.log", [tuned_metrics_path(plan["task"], spec, root)], 1))
    work.sort(key=lambda item: -item[0])  # the largest jobs first keeps the pool busy until the end
    work = [item[1:] for item in work]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [(item[0], pool.submit(_run_probe, item[0], item[1], item[2], item[4], item[3])) for item in work]
        failures += [(label, reason) for label, future in futures if (reason := future.result()) is not None]
    if not args.no_ledger:
        try:
            record = ledger.ingest(results, outputs_root=root, command=" ".join(sys.argv))
            print(f"ledger add: {record['rows_added']} rows from {record['files_seen']} files "
                  f"({record['files_skipped_same_sha']} already known)", flush=True)
            written = ledger_report.write_reports(results)
            print(f"ledger reports: {len(written)} files under {results / 'reports'}", flush=True)
            table = summary_table(results, tasks, heads, args.models)
        except Exception as exc:  # noqa: BLE001 - the job failures below must still be listed
            failures.append(("ledger add / report", f"{type(exc).__name__}: {exc}"))
            table = f"# ECFP4 baseline\n\nLedger step failed: {exc}\n"
    else:
        table = "# ECFP4 baseline\n\nNo ledger step (--no-ledger).\n"
    if failures:
        table += f"\n## MISSING: {len(failures)} job(s) failed or could not run\n\n" + "".join(
            f"- {label}: {reason}\n" for label, reason in failures)
    (run_dir / "summary.md").write_text(table)
    print("\n" + table, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
