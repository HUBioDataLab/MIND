"""Compare two models cell by cell, with a paired bootstrap over the saved test predictions.

    python -m downstream.ledger_compare --a hyp_md4_s42_ep98 --b <g1 snapshot> [--tasks ec lba] [--bootstrap 1000]

Writes ``$MIND_RESULTS_ROOT/reports/compare/<A>__vs__<B>.md|.csv``. Only cells with the same comparable
setting (see ``downstream.ledger_report``: protocol, pooling, data revision and data fingerprint, settings
hash) are compared; every other cell is listed as "only in A" / "only in B" with the reason.

Three noise sources are reported separately:

1. probe seed: std over run seeds of each model (``ledger_report.STD_DDOF``);
2. test sample: paired bootstrap 95 % interval of the difference. Each replicate draws one set of test-sample
   indices, applies it to A and B and to every common run seed, and averages the signed difference over seeds.
   Chains are aligned by id; a different test-chain set or labels is an error, and each seed's predictions must
   reproduce the ledger value before they are used;
3. pretraining run: not measured (one pretraining seed per model); the report says so in a warning box.

Sign: ``diff_b_better`` = B - A for higher-is-better metrics and A - B for lower-is-better ones, so + always
means B is better. Decision: "B daha iyi" when the interval's lower bound is > 0 and |diff| > 2 x the larger
seed std; "B daha kötü" symmetrically; otherwise "ayırt edilemez". Cells without an interval (subsets whose
mask is not in predictions.npz, validation rows, metrics without a bootstrap implementation) get "karar yok".
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from downstream import ledger, ledger_report
from downstream.common import metrics as M

REPRODUCE_TOLERANCE = 1e-4
_SECTION_KEY = ("protocol_family", "protocol_id", "pooling", "data_revision", "split_tag", "representation",
                "protocol_settings_sha256", "data_fingerprint")


# ---------------------------------------------------------------------------------------------------------
# Resampled metric functions: prepare(labels, scores) -> f(indices) -> value (ValueError when undefined)
# ---------------------------------------------------------------------------------------------------------

def _fmax_fn(labels: np.ndarray, scores: np.ndarray) -> Callable:
    """Protein-centric Fmax as in ``metrics.protein_fmax``, from per-protein counts so resampling is cheap."""
    labels, scores = M._check(labels, scores)
    hits = np.stack([((scores >= t) & labels).sum(axis=1) for t in M.THRESHOLDS]).astype(np.float64)
    predicted = np.stack([(scores >= t).sum(axis=1) for t in M.THRESHOLDS]).astype(np.float64)
    true = labels.sum(axis=1).astype(np.float64)

    def fmax(idx):
        h, p = hits[:, idx], predicted[:, idx]
        covered = p > 0
        n_covered = covered.sum(axis=1)
        precision = np.where(n_covered > 0, (np.where(covered, h / np.maximum(p, 1), 0)).sum(axis=1)
                             / np.maximum(n_covered, 1), 0.0)
        recall = (h / true[idx]).mean(axis=1)
        valid = (n_covered > 0) & (precision + recall > 0)
        f1 = np.where(valid, 2 * precision * recall / np.where(valid, precision + recall, 1), 0.0)
        return float(f1.max())
    return fmax


def metric_fn(kind: str, metric: str, labels: np.ndarray, scores: np.ndarray) -> Optional[Callable]:
    """Resampling function for one (kind, metric), or None when no bootstrap is implemented for it."""
    if kind == "multilabel" and metric == "fmax":
        return _fmax_fn(labels, scores)
    if kind == "multilabel" and metric == "top1_accuracy":
        checked_labels, checked_scores = M._check(labels, scores)
        correct = checked_labels[np.arange(len(checked_labels)), checked_scores.argmax(axis=1)].astype(np.float64)
        return lambda idx: float(correct[idx].mean())
    if kind == "multiclass" and metric in ("accuracy", "balanced_accuracy"):
        targets, checked_scores = M._check_multiclass(labels, scores)
        correct = (checked_scores.argmax(axis=1) == targets).astype(np.float64)
        if metric == "accuracy":
            return lambda idx: float(correct[idx].mean())

        def balanced(idx):  # mean recall over the classes present in the resample, as metrics.balanced_accuracy
            counts = np.bincount(targets[idx])
            hits = np.bincount(targets[idx], weights=correct[idx], minlength=len(counts))
            present = counts > 0
            return float(np.mean(hits[present] / counts[present]))
        return balanced
    if kind == "regression" and metric in ("rmse", "mae", "pearson_r", "spearman_rho"):
        function = {"rmse": M.rmse, "mae": M.mae, "pearson_r": M.pearson_r, "spearman_rho": M.spearman_rho}[metric]
        return lambda idx: function(labels.reshape(-1)[idx], scores.reshape(-1)[idx])
    if kind == "multilabel_masked" and metric in ("roc_auc", "prc_auc"):
        function = M.multitask_roc_auc if metric == "roc_auc" else M.multitask_prc_auc
        return lambda idx: function(labels[idx], scores[idx])[metric]
    if kind == "multitarget_regression" and metric in ("rmse", "mae"):
        return lambda idx: M.multitarget_regression_errors(labels[idx], scores[idx])[metric]
    return None


def _split_arrays(path: Path, eval_split: str):
    """(chains, scores, labels) of one test split from predictions.npz (multi-class extras are prefixed)."""
    with np.load(path) as data:
        if f"{eval_split}_chains" in data:
            prefix = eval_split
        elif "test_chains" in data:
            prefix = "test"  # the first test split of the task (the integrity check below confirms the pairing)
        else:
            raise ValueError(f"{path} has no test predictions")
        return (data[f"{prefix}_chains"].astype(str), data[f"{prefix}_scores"], data[f"{prefix}_labels"])


def _aligned(path: Path, eval_split: str, order: Optional[np.ndarray]):
    chains, scores, labels = _split_arrays(path, eval_split)
    if order is None:
        return chains, scores, labels
    position = {chain: i for i, chain in enumerate(chains)}
    if len(position) != len(chains) or set(position) != set(order.tolist()):
        raise ValueError(f"{path}: test chains differ from the other model's ({len(chains)} vs {len(order)} ids, "
                         f"{len(set(position) ^ set(order.tolist()))} not shared); paired comparison is impossible")
    index = np.array([position[chain] for chain in order])
    return order, scores[index], labels[index]


def _ci_reason(row: dict, spec: dict) -> Optional[str]:
    if row["eval_split"] == "val":
        return "doğrulama etiketleri predictions.npz'de yok"
    if row["subset"] != "all":
        return "alt küme maskesi predictions.npz'de yok"
    return None


def _parts(row: dict, spec: dict) -> list:
    if row["derived_from"]:
        for derived in spec.get("derived_averages") or []:
            if derived["eval_split"] == row["eval_split"]:
                return list(derived["of"])
        raise ValueError(f"{row['source_metrics_path']}: derived row {row['eval_split']} has no catalog definition")
    return [row["eval_split"]]


def paired_bootstrap(rows_a: list, rows_b: list, spec: dict, *, sign: float, n_boot: int,
                     rng_seed: int) -> dict:
    """Interval of the signed (B better = +) difference, paired over test samples and common seeds."""
    by_seed_a = {r["run_seed"]: r for r in rows_a}
    by_seed_b = {r["run_seed"]: r for r in rows_b}
    seeds = sorted(set(by_seed_a) & set(by_seed_b))
    first = rows_a[0]
    reason = _ci_reason(first, spec)
    if reason is None and not seeds:
        reason = "ortak koşu tohumu yok"
    if reason is None and any(by_seed_a[s]["value"] is None or by_seed_b[s]["value"] is None for s in seeds):
        reason = "değer bazı tohumlarda tanımsız"
    if reason is not None:
        return {"ci_low": None, "ci_high": None, "n_boot_failed": None, "common_seeds": seeds, "ci_reason": reason}
    parts = _parts(first, spec)
    orders = {}
    functions = []  # per seed: (f_a, f_b) lists over parts
    for seed in seeds:
        per_model = []
        for row in (by_seed_a[seed], by_seed_b[seed]):
            path = Path(row["source_metrics_path"]).with_name("predictions.npz")
            if not path.is_file():
                return {"ci_low": None, "ci_high": None, "n_boot_failed": None, "common_seeds": seeds,
                        "ci_reason": f"predictions.npz yok: {path}"}
            fns = []
            for part in parts:
                chains, scores, labels = _aligned(path, part, orders.get(part))
                if part not in orders:
                    orders[part] = chains
                    orders[part, "labels"] = labels
                elif not np.array_equal(labels, orders[part, "labels"], equal_nan=True):
                    raise ValueError(f"{path}: labels of {part} differ from the other run's for the same chains")
                function = metric_fn(row["kind"], row["metric"], labels, scores)
                if function is None:
                    return {"ci_low": None, "ci_high": None, "n_boot_failed": None, "common_seeds": seeds,
                            "ci_reason": f"bootstrap {row['kind']}/{row['metric']} için tanımlı değil"}
                fns.append(function)
            full = float(np.mean([f(np.arange(len(orders[p]))) for f, p in zip(fns, parts)]))
            if abs(full - row["value"]) > REPRODUCE_TOLERANCE:
                raise ValueError(f"{path}: predictions.npz does not reproduce the ledger value of "
                                 f"{row['metric']} {row['eval_split']} (seed {seed}): {full:.6f} vs {row['value']:.6f}")
            per_model.append(fns)
        functions.append(per_model)
    rng = np.random.default_rng(rng_seed)
    sizes = [len(orders[p]) for p in parts]
    differences, failed = [], 0
    for _ in range(n_boot):
        draws = [rng.integers(0, n, n) for n in sizes]
        try:
            values = [np.mean([fb(i) for fb, i in zip(f_b, draws)]) - np.mean([fa(i) for fa, i in zip(f_a, draws)])
                      for f_a, f_b in functions]
        except ValueError:
            failed += 1
            continue
        if not np.all(np.isfinite(values)):
            failed += 1
            continue
        differences.append(sign * float(np.mean(values)))
    if not differences:
        return {"ci_low": None, "ci_high": None, "n_boot_failed": failed, "common_seeds": seeds,
                "ci_reason": "her bootstrap örneğinde metrik tanımsız"}
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"ci_low": float(low), "ci_high": float(high), "n_boot_failed": failed, "common_seeds": seeds,
            "ci_reason": None}


# ---------------------------------------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------------------------------------

def _cells(state: dict, model: str, tasks: Optional[list]) -> dict:
    cells = defaultdict(list)
    for task in tasks or sorted({r["task"] for r in state["rows"]}):
        for section in ledger_report.sections(state, task):
            for row in section["rows"]:
                if row["model_key"] == model:
                    cells[(task,) + tuple(section[f] for f in _SECTION_KEY)
                          + (row["eval_split"], row["subset"], row["metric"])].append(row)
    return cells


def _decision(diff, low, high, std_a, std_b, reason) -> str:
    if diff is None:
        return "karar yok: değer tanımsız"
    if low is None:
        return f"karar yok: {reason}"
    if std_a is None or std_b is None:
        return "karar yok: tek tohum, probe-tohumu std'si tanımsız"
    margin = 2 * max(std_a, std_b)
    if low > 0 and abs(diff) > margin:
        return "B daha iyi"
    if high < 0 and abs(diff) > margin:
        return "B daha kötü"
    return "ayırt edilemez"


def compare(state: dict, a: str, b: str, *, tasks: Optional[list] = None, n_boot: int = 1000,
            rng_seed: int = 0) -> dict:
    if a == b:
        raise ValueError("A and B must be different models")
    cells_a, cells_b = _cells(state, a, tasks), _cells(state, b, tasks)
    if not cells_a and not cells_b:
        raise ValueError(f"no valid ledger rows for {a!r} or {b!r}" + (f" in tasks {tasks}" if tasks else ""))
    pairs = []
    for key in sorted(set(cells_a) & set(cells_b), key=str):
        rows_a, rows_b = cells_a[key], cells_b[key]
        task, metric = key[0], key[-1]
        info = ledger.metric_info(state["catalog"], task, metric)
        stats_a, stats_b = ledger_report.cell_stats(rows_a), ledger_report.cell_stats(rows_b)
        if rows_a[0]["unit"] != rows_b[0]["unit"]:
            raise ValueError(f"{key}: units differ ({rows_a[0]['unit']} vs {rows_b[0]['unit']})")
        if info["direction"] is None:
            continue  # metrics outside the catalog have no direction; they stay in the task tables only
        sign = 1.0 if info["direction"] == "higher" else -1.0
        diff = (None if stats_a["mean"] is None or stats_b["mean"] is None
                else sign * (stats_b["mean"] - stats_a["mean"]))
        spec = ledger.task_spec(state["catalog"], task)
        boot = paired_bootstrap(rows_a, rows_b, spec, sign=sign, n_boot=n_boot, rng_seed=rng_seed)
        pairs.append({"task": task, **dict(zip(_SECTION_KEY, key[1:9])), "eval_split": key[9], "subset": key[10],
                      "metric": metric, "direction": info["direction"], "unit": info["unit"],
                      "is_primary": rows_a[0]["is_primary"], "mean_a": stats_a["mean"], "mean_b": stats_b["mean"],
                      "std_a": stats_a["std"], "std_b": stats_b["std"], "n_seeds_a": stats_a["n_seeds"],
                      "n_seeds_b": stats_b["n_seeds"], "diff_b_better": diff, **boot, "n_boot": n_boot,
                      "rng_seed": rng_seed,
                      "decision": _decision(diff, boot["ci_low"], boot["ci_high"], stats_a["std"], stats_b["std"],
                                            boot["ci_reason"])})

    def only(mine, other, name):
        loose = {(k[0], k[2], k[9], k[10], k[11]) for k in other}
        return [{"task": k[0], "protocol_id": k[2], "pooling": k[3], "data_revision": k[4], "eval_split": k[9],
                 "subset": k[10], "metric": k[11],
                 "reason": ("farklı ayar (pooling / veri sürümü / veri parmak izi / protokol ayarı farklı)"
                            if (k[0], k[2], k[9], k[10], k[11]) in loose else f"{name}'de yok")}
                for k in sorted(set(mine) - set(other), key=str)]
    return {"a": a, "b": b, "pairs": pairs, "only_a": only(cells_a, cells_b, "B"), "only_b": only(cells_b, cells_a, "A"),
            "n_boot": n_boot, "rng_seed": rng_seed}


def _num(value, digits=3) -> str:
    return "tanımsız" if value is None else f"{value:.{digits}f}"


def compare_markdown(state: dict, result: dict) -> str:
    a, b = result["a"], result["b"]
    lines = ledger_report.header_lines(state, f"Karşılaştırma: A = {a}, B = {b}")
    with_ci = [p for p in result["pairs"] if p["ci_low"] is not None]
    counts = defaultdict(int)
    for pair in result["pairs"]:
        counts[pair["decision"].split(":")[0]] += 1
    lines += [f"> UYARI: {ledger_report.PRETRAIN_SEED_WARNING} Bu rapor \"B şunu iyileştirdi\" diyemez; en fazla \"bu iki "
              "koşu arasında probe ve test gürültüsünün ötesinde fark var\" der.", "",
              "Fark = B − A, yöne göre işaretli: + her zaman B daha iyi demek (RMSE/MAE gibi düşük-iyi metriklerde "
              "işaret çevrildi).", "", "Üç gürültü kaynağı ayrı ayrı:", "",
              f"1. Probe tohumu (başın rastgele başlatılması): std_A ve std_B sütunları (tohumlar arası, std: ddof="
              f"{ledger_report.STD_DDOF} (örnek std); tek tohumda tanımsız).",
              f"2. Test örneklemi: eşli bootstrap %95 güven aralığı ({result['n_boot']} yeniden örnekleme, RNG tohumu "
              f"{result['rng_seed']}; aynı örnek indeksleri iki modele ve tüm ortak tohumlara uygulanır).",
              "3. Ön-eğitim koşusu (aynı ayar, başka ön-eğitim tohumu): ÖLÇÜLMEDİ (yukarıdaki uyarı).", "",
              "Karar: \"B daha iyi\" = aralığın alt sınırı > 0 ve |fark| > 2 × max(std_A, std_B); \"B daha kötü\" "
              "simetrik; diğer her durum \"ayırt edilemez\". Aralık yoksa \"karar yok\" ve nedeni.", "",
              f"Çoklu karşılaştırma: {len(with_ci)} karşılaştırma %95 düzeyinde yapıldı; gerçek fark olmasa bile "
              f"şans eseri yaklaşık {0.05 * len(with_ci):.1f} tanesinin aralığı sıfırı dışlayabilir (beklenen yanlış "
              "alarm).", "",
              "Karar sayıları: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), "",
              "| görev | protokol | pooling | veri | eval_split | subset | metric | birim | A | B | fark (+ = B iyi) | "
              "std_A | std_B | %95 GA | ortak tohum | karar |", "|" + "---|" * 16]
    for p in sorted(result["pairs"], key=lambda p: (p["task"], not p["is_primary"], p["eval_split"], p["subset"],
                                                    p["metric"])):
        interval = "-" if p["ci_low"] is None else f"[{p['ci_low']:+.3f}, {p['ci_high']:+.3f}]"
        if p["n_boot_failed"]:
            interval += f" ({p['n_boot_failed']} örnek tanımsız)"
        lines.append(f"| {p['task']} | {p['protocol_id']} | {p['pooling']} | {p['data_revision']} | {p['eval_split']} | "
                     f"{p['subset']} | {p['metric']} | {p['unit']} | {_num(p['mean_a'])} | {_num(p['mean_b'])} | "
                     f"{'tanımsız' if p['diff_b_better'] is None else format(p['diff_b_better'], '+.3f')} | "
                     f"{_num(p['std_a'])} | {_num(p['std_b'])} | {interval} | {len(p['common_seeds'])} | {p['decision']} |")
    for name, entries in (("A", result["only_a"]), ("B", result["only_b"])):
        lines += ["", f"## Yalnız {name}'de olan hücreler ({len(entries)})", ""]
        lines += [f"- {e['task']} / {e['protocol_id']} / {e['pooling']} / {e['data_revision']} / {e['eval_split']} / "
                  f"{e['subset']} / {e['metric']}: {e['reason']}" for e in entries] or ["- yok"]
    return "\n".join(lines) + "\n"


def write_compare(root: Optional[Path], a: str, b: str, *, tasks: Optional[list] = None, n_boot: int = 1000,
                  rng_seed: int = 0, state: Optional[dict] = None) -> list:
    root = ledger.results_root(root)
    state = state or ledger_report.load_state(root)
    result = compare(state, a, b, tasks=tasks, n_boot=n_boot, rng_seed=rng_seed)
    written: list = []
    directory = root / "reports" / "compare"
    ledger_report._write(directory / f"{a}__vs__{b}.md", compare_markdown(state, result), written)
    rows = [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in p.items()} for p in result["pairs"]]
    ledger_report._write(directory / f"{a}__vs__{b}.csv", ledger_report._csv_text(rows), written)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--results-root", type=Path, default=None, help=f"default: ${ledger.ENV_VAR}")
    args = parser.parse_args(argv)
    if args.bootstrap < 1:
        parser.error("--bootstrap must be at least 1")
    written = write_compare(args.results_root, args.a, args.b, tasks=args.tasks, n_boot=args.bootstrap,
                            rng_seed=args.rng_seed)
    print(json.dumps([str(p) for p in written], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
