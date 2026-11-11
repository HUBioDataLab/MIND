"""Reports generated from the results ledger (rewritten on every run; the ledger is the source).

    python -m downstream.ledger_report report  [--models A B ...]   tasks/, timeline/, ozet_hoca.md, eksik_hucreler.*
    python -m downstream.ledger_report summary [--models A B ...]   ozet_hoca.md only
    python -m downstream.ledger_report missing [--models A B ...]   eksik_hucreler.md/.csv only

Reports go to ``$MIND_RESULTS_ROOT/reports/``. A column only ever holds one comparable setting: rows are
grouped by (task, protocol family, protocol id, pooling, data revision, split tag, representation,
protocol-settings hash, per-run data fingerprint over the evaluated processed files); a different value of
any of these starts a new "Ayar" section. Superseded / invalid / smoke rows leave the task tables and stay in
the timeline. No cell is left empty: a missing cell names its category (EKSİK:..., TANIMSIZ:..., KOD-YOK,
YALNIZ-ESKİ-SÜRÜM). Literature numbers (``<root>/literature/literature.csv``, local, stage 5) are shown in a
separate column in their own unit and protocol; they are never mixed into model columns.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from downstream import ledger

STD_DDOF = 1  # sample std (n-1); downstream.report keeps ddof=0, so the same seeds can show two std values
# Professor-page / missing-list models when neither --models nor <root>/comparisons.yaml names them (decision of
# 9 Nov 2026): pre-G1 four-domain model, protein-only, the random-init control, the post-G1 model (snapshot name
# fixed when its training ends; until then it shows as EKSİK:model-henüz-yok) and the 330-epoch reference.
DEFAULT_SUMMARY_MODELS = ("hyp_md4_s42_ep98", "hyp_protein_only_s42_ep98", "hyp_md4_s42_ep98.random_init",
                          "g1h_1m_100ep_s42_ep<N>", "mind4_1m_packed_ep330_final")
LITERATURE_COLUMNS = ("task", "paper", "model", "metric", "eval_split", "subset", "value", "std", "unit", "protocol",
                      "n_seeds", "split_note", "pdf", "page", "table")
LITERATURE_REQUIRED = ("task", "paper", "model", "metric", "eval_split", "subset", "value", "unit", "protocol", "pdf",
                       "page", "table")
NON_SUMMARY_FAMILIES = ("pretraining", "diagnostic")
PRETRAIN_SEED_WARNING = ("ön-eğitim tohumu gürültüsü ölçülmedi: her model tek bir ön-eğitim koşusundan (tek tohum) "
                         "geliyor; aynı ayarla başka tohumla eğitilmiş bir model ne kadar farklı sonuç verirdi, "
                         "bilinmiyor.")
_SECTION_FIELDS = ("task", "protocol_family", "protocol_id", "pooling", "data_revision", "split_tag", "representation",
                   "protocol_settings_sha256")
_FAMILY_ORDER = {"frozen_probe": 0, "finetune": 1, "evaluation": 2}


# ---------------------------------------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------------------------------------

def load_literature(root: Path, catalog: dict) -> Optional[list]:
    """Rows of ``<root>/literature/literature.csv`` (None when the file does not exist yet)."""
    path = Path(root) / "literature" / "literature.csv"
    if not path.exists():
        return None
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != LITERATURE_COLUMNS:
            raise ValueError(f"{path}: columns must be {','.join(LITERATURE_COLUMNS)}, got {reader.fieldnames}")
        rows = list(reader)
    for number, row in enumerate(rows, start=2):
        where = f"{path} line {number}"
        for field in LITERATURE_REQUIRED:
            if not (row.get(field) or "").strip():
                raise ValueError(f"{where}: field {field!r} is empty (unit, protocol and pdf/page/table are required)")
        if row["task"] not in catalog["tasks"]:
            raise ValueError(f"{where}: task {row['task']!r} is not in the ledger catalog")
        if row["metric"] not in catalog["metrics"]:
            raise ValueError(f"{where}: metric {row['metric']!r} is not in the ledger catalog")
        try:
            row["value_float"] = float(row["value"])
            row["std_float"] = float(row["std"]) if row["std"].strip() else None
        except ValueError as exc:
            raise ValueError(f"{where}: value/std must be numbers ({exc})") from exc
    return rows


def load_state(root: Optional[Path] = None, *, catalog: Optional[dict] = None) -> dict:
    """Verified ledger with annotations applied, plus the header every report prints."""
    root = ledger.results_root(root)
    summary = ledger.verify(root)
    catalog = catalog or ledger.load_catalog()
    records = {name: ledger.read_records(root, name) for name in ledger.LEDGER_FILES}
    rows = ledger.apply_annotations(records["measurements"], records["annotations"])
    # Keyed by the source file and its content hash, not by the ingest: a row recomputed later from the same
    # file (``ledger recompute-macro-aupr``) keeps the setting of that file's other rows.
    fingerprints = defaultdict(set)
    for row in rows:
        fingerprints[row["source_metrics_path"], row["source_metrics_sha256"]].add(
            (row["eval_split"], row["eval_processed_sha256"]))
    for row in rows:
        row["data_fingerprint"] = ledger.sha256_json(
            sorted(fingerprints[row["source_metrics_path"], row["source_metrics_sha256"]], key=str))
    header = {"generated_utc": ledger.now_utc(), "n_rows": len(rows), "measurements_sha256": summary["measurements_sha256"],
              "catalog_sha256": catalog["sha256"], "code_git": ledger._code_git(),
              "n_annotations": len(records["annotations"])}
    embedding_sets = {(entry["task"], entry["model_key"]) for ingest in records["ingests"]
                      for entry in ingest.get("embedding_sets", [])}
    return {"root": root, "rows": rows, "models": records["models"], "annotations": records["annotations"],
            "catalog": catalog, "header": header, "literature": load_literature(root, catalog),
            "embedding_sets": embedding_sets}


def header_lines(state: dict, title: str) -> list:
    h = state["header"]
    return [f"# {title}", "",
            f"Üretildi: {h['generated_utc']} | defter: measurements.jsonl {h['n_rows']} satır, sha256 "
            f"{h['measurements_sha256']} | katalog sha256 {h['catalog_sha256']} | kod {h['code_git']} | "
            f"uygulanan not: {h['n_annotations']} | std: ddof={STD_DDOF} (örnek std), tohumlar arası; ds-report ddof=0 "
            "kullanır", ""]


# ---------------------------------------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------------------------------------

def _latest(rows: list) -> tuple[list, dict]:
    """Newest row per (model, seed, cell) within one setting; older values are returned for a footnote."""
    newest, older = {}, defaultdict(list)
    for row in sorted(rows, key=lambda r: r["ingested_utc"]):
        key = (row["model_key"], row["run_seed"], row["eval_split"], row["subset"], row["metric"])
        if key in newest:
            older[key].append(newest[key])
        newest[key] = row
    return list(newest.values()), older


def cell_stats(rows: list) -> dict:
    """Mean/std over seeds of one (setting, model, cell); undefined seeds are counted with their reasons."""
    values = [r["value"] for r in rows if r["value"] is not None]
    reasons = sorted({r["undefined_reason"] for r in rows if r["value"] is None})
    expected = rows[0]["expected_seeds"] or []
    samples = sorted({r["n_samples"] for r in rows if r["n_samples"] is not None})
    return {"n_seeds": len(values), "n_rows": len(rows), "expected": len(expected),
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=STD_DDOF)) if len(values) > STD_DDOF else None, "reasons": reasons,
            "n_samples": samples, "seeds": sorted(r["run_seed"] for r in rows),
            "missing_provenance": sorted({m for r in rows for m in r["missing_provenance"]}),
            "caveats": sorted({c for r in rows for c in r["caveats"]})}


def format_cell(stats: dict) -> str:
    if stats["mean"] is None:
        return "TANIMSIZ:" + "; ".join(stats["reasons"])
    samples = stats["n_samples"]
    n = "?" if not samples else str(samples[0]) if len(samples) == 1 else f"{samples[0]}-{samples[-1]}"
    spread = f"± {stats['std']:.3f}" if stats["std"] is not None else f"(std tanımsız (n={stats['n_seeds']}))"
    text = f"{stats['mean']:.3f} {spread} (n_tohum={stats['n_seeds']}, n_örnek={n})"
    if stats["expected"] and stats["n_rows"] < stats["expected"]:
        text += f" EKSİK:tohum {stats['n_rows']}/{stats['expected']}"
    if stats["reasons"]:
        text += f" TANIMSIZ {stats['n_rows'] - stats['n_seeds']} tohumda: {'; '.join(stats['reasons'])}"
    if stats["missing_provenance"]:
        text += " †"
    if stats["caveats"]:
        text += " ‡"
    return text


def sections(state: dict, task: str, *, statuses=("valid",)) -> list:
    """Comparable settings of ``task`` (see module docstring), each with its models and rows."""
    groups = defaultdict(list)
    for row in state["rows"]:
        if row["task"] == task and row["status"] in statuses:
            groups[tuple(row[f] for f in _SECTION_FIELDS) + (row["data_fingerprint"],)].append(row)
    out = []
    for key, rows in groups.items():
        latest, older = _latest(rows)
        entry = dict(zip(_SECTION_FIELDS + ("data_fingerprint",), key))
        entry.update(models=sorted({r["model_key"] for r in latest}), rows=latest, older=older)
        out.append(entry)
    return sorted(out, key=lambda s: (_FAMILY_ORDER.get(s["protocol_family"], 9), s["protocol_id"], str(s["pooling"]),
                                      str(s["data_revision"]), s["data_fingerprint"]))


def _cell_order(spec: dict, rows: list) -> list:
    required = [(c["eval_split"], c["subset"], c["metric"]) for c in spec["required"]]
    present = sorted({(r["eval_split"], r["subset"], r["metric"]) for r in rows} - set(required),
                     key=lambda c: (c[0] != "test", c[0], c[1], not any(r["is_catalog_metric"] for r in rows
                                                                         if r["metric"] == c[2]), c[2]))
    return required + present


def _literature_cell(state: dict, task: str, eval_split: str, subset: str, metric: str) -> str:
    if state["literature"] is None:
        return "literatür dosyası yok (aşama 5)"
    rows = [r for r in state["literature"] if (r["task"], r["eval_split"], r["subset"], r["metric"]) ==
            (task, eval_split, subset, metric)]
    if not rows:
        return "YOK"
    direction = state["catalog"]["metrics"][metric]["direction"]
    parts = []
    for unit in sorted({r["unit"] for r in rows}):
        candidates = [r for r in rows if r["unit"] == unit]
        best = (max if direction == "higher" else min)(candidates, key=lambda r: r["value_float"])
        seeds = f", n_tohum={best['n_seeds']}" if best["n_seeds"].strip() else ""
        parts.append(f"{best['value']} {unit} ({best['model']}, {best['paper']}, {best['protocol']}{seeds}; "
                     f"{best['pdf']} s.{best['page']} {best['table']})")
    return "; ".join(parts)


def _section_title(index: int, section: dict) -> str:
    return (f"### Ayar {index}: {section['protocol_family']} / {section['protocol_id']} / pooling={section['pooling']}"
            f" / veri={section['data_revision']} / split_tag={section['split_tag']} / "
            f"representation={section['representation']} / ayar-sha={str(section['protocol_settings_sha256'])[:12]}"
            f" / veri-parmak-izi={section['data_fingerprint'][:12]}")


def task_report(state: dict, task: str) -> tuple[str, list]:
    """Markdown and long-format rows of one task's tables (valid rows only)."""
    spec = ledger.task_spec(state["catalog"], task)
    lines = header_lines(state, f"Görev tablosu: {task}")
    lines += [f"Aile: {spec['family']} | tür: {spec['kind']} | zorunlu protokoller: {spec['required_protocols']}",
              "Hücre: ort ± std (n_tohum, n_örnek). † eksik iz bilgisi (dipnot), ‡ not/uyarı, KOD-YOK: makale istiyor "
              "ama kodumuz hesaplamıyor. Literatür sütunu ayrı protokoldür (çoğu tam ince ayar), kendi biriminde.", ""]
    out_rows = []
    task_sections = sections(state, task)
    if not task_sections:
        lines += ["Bu görev için geçerli (valid) ölçüm yok. Eksikler için eksik_hucreler.md.", ""]
    for index, section in enumerate(task_sections, start=1):
        models = section["models"]
        lines += [_section_title(index, section), "",
                  "| eval_split | subset | metric | yön | birim | " + " | ".join(models)
                  + " | literatür (en iyi; ayrı protokol) |", "|" + "---|" * (6 + len(models))]
        by_cell = defaultdict(list)
        for row in section["rows"]:
            by_cell[(row["eval_split"], row["subset"], row["metric"]), row["model_key"]].append(row)
        code_missing = {(c["eval_split"], c["subset"], c["metric"]) for c in spec["required"]
                        if c.get("code", "present") == "missing"}
        footnotes = []
        for cell in _cell_order(spec, section["rows"]):
            info = ledger.metric_info(state["catalog"], task, cell[2])
            texts = []
            for model in models:
                cell_rows = by_cell.get((cell, model))
                if cell in code_missing:
                    text, stats = "KOD-YOK", None
                elif not cell_rows:
                    text, stats = "EKSİK:metrik-dosyada-yok", None
                else:
                    stats = cell_stats(cell_rows)
                    text = format_cell(stats)
                    previous = [(old["value"], old["ingested_utc"]) for r in cell_rows
                                for old in section["older"].get((model, r["run_seed"]) + cell, [])]
                    if previous:
                        text += " *"
                        footnotes.append(f"* {model} {'/'.join(cell)}: önceki değer "
                                         + ", ".join(f"{v:.3f} ({t})" if v is not None else f"tanımsız ({t})"
                                                     for v, t in previous))
                    if stats["missing_provenance"]:
                        footnotes.append(f"† {model} {'/'.join(cell)}: eksik iz bilgisi {stats['missing_provenance']}")
                    if stats["caveats"]:
                        footnotes.append(f"‡ {model} {'/'.join(cell)}: {'; '.join(stats['caveats'])}")
                texts.append(text)
                out_rows.append({"task": task, "section": index, **{f: section[f] for f in _SECTION_FIELDS[1:]},
                                 "data_fingerprint": section["data_fingerprint"], "model_key": model,
                                 "eval_split": cell[0], "subset": cell[1], "metric": cell[2],
                                 "unit": info["unit"], "direction": info["direction"],
                                 "mean": stats["mean"] if stats else None, "std": stats["std"] if stats else None,
                                 "n_seeds": stats["n_seeds"] if stats else 0,
                                 "expected_seeds": stats["expected"] if stats else None, "cell": text})
            label = "katalog dışı" if not info["is_catalog_metric"] else info["direction"]
            lines.append(f"| {cell[0]} | {cell[1]} | {cell[2]} | {label} | {info['unit'] or '-'} | "
                         + " | ".join(texts) + f" | {_literature_cell(state, task, *cell)} |")
        lines.append("")
        lines += sorted(set(footnotes)) + ([""] if footnotes else [])
    return "\n".join(lines) + "\n", out_rows


def timeline_report(state: dict, task: str) -> tuple[str, list]:
    """Every status (superseded included), ordered by model date, then by measurement date."""
    model_dates = {m["model_key"]: m.get("created_utc") for m in state["models"]}
    groups = defaultdict(list)
    for row in state["rows"]:
        if row["task"] == task:
            groups[tuple(row[f] for f in _SECTION_FIELDS) + (row["data_fingerprint"], row["model_key"], row["status"],
                                                             row["eval_split"], row["subset"], row["metric"])].append(row)
    out_rows = []
    for key, rows in groups.items():
        latest, _ = _latest(rows)
        stats = cell_stats(latest)
        out_rows.append({"model_key": key[9], "model_created_utc": model_dates.get(key[9]),
                         "train_epoch": latest[0]["train_epoch"], "first_ingested_utc": min(r["ingested_utc"] for r in rows),
                         "status": key[10], "status_reason": latest[0]["status_reason"], "protocol_family": key[1],
                         "protocol_id": key[2], "pooling": key[3], "data_revision": key[4],
                         "eval_split": key[11], "subset": key[12], "metric": key[13], "is_primary": latest[0]["is_primary"],
                         "mean": stats["mean"], "std": stats["std"], "n_seeds": stats["n_seeds"], "cell": format_cell(stats)})
    out_rows.sort(key=lambda r: (str(r["model_created_utc"]), r["first_ingested_utc"], r["model_key"], r["protocol_id"],
                                 r["eval_split"], r["subset"], r["metric"]))
    lines = header_lines(state, f"Zaman çizgisi: {task}")
    lines += ["Tüm durumlar dahil (superseded/invalid/smoke burada görünür, görev tablosunda görünmez). Yalnız birincil "
              "metrik; tüm metrikler CSV'de.", "",
              "| model | model tarihi | ön-eğitim epoch | ilk ekleme | durum | protokol | pooling | veri | hücre |",
              "|---|---|---|---|---|---|---|---|---|"]
    for row in out_rows:
        if row["is_primary"]:
            status = row["status"] + (f" ({row['status_reason']})" if row["status_reason"] else "")
            lines.append(f"| {row['model_key']} | {row['model_created_utc'] or '-'} | {row['train_epoch'] if row['train_epoch'] is not None else '-'} | "
                         f"{row['first_ingested_utc']} | {status} | {row['protocol_id']} | {row['pooling']} | "
                         f"{row['data_revision']} | {row['cell']} |")
    return "\n".join(lines) + "\n", out_rows


# ---------------------------------------------------------------------------------------------------------
# Expected cells: professor page and missing list
# ---------------------------------------------------------------------------------------------------------

def summary_tasks(catalog: dict) -> list:
    return [task for task, spec in catalog["tasks"].items() if spec["family"] not in NON_SUMMARY_FAMILIES]


def _hint(task: str, model: str, protocol: str, category: str) -> str:
    head, *qualifiers = protocol.split(".")
    lba = f" SPLIT_TAG={qualifiers[0]} REPRESENTATION={qualifiers[1]}" if len(qualifiers) == 2 else ""
    probe = (f"make ds-probe TASK={task} MODEL_ID={model} HEAD={ledger.PROBE_HEADS.get(head, head)}{lba}"
             if head in ledger.PROBE_HEADS else f"make ds-finetune TASK={task} MODEL={model} GPU=<id>")
    if category == "KOD-YOK":
        return "metrik kodu yok: downstream/common/metrics.py'ye eklenmeli (ayrı iş)"
    if category == "EKSİK:model-henüz-yok":
        return (f"model defterde hiç yok: make ds-snapshot CKPT=<ckpt> NAME={model}, make ds-embed TASK={task} "
                f"MODEL={model} GPU=<id>, {probe}, sonra make ds-ledger-add")
    if category == "EKSİK:koşulmadı":
        return f"make ds-embed TASK={task} MODEL={model} GPU=<id>, sonra {probe}, sonra make ds-ledger-add"
    if category == "EKSİK:yalnız-embedding":
        return f"{probe}, sonra make ds-ledger-add"
    if category == "EKSİK:metrik-dosyada-yok":
        return "predictions.npz'den yeniden hesap (aşama 7) ya da probe'u yeni kodla yeniden koş"
    if category == "YALNIZ-ESKİ-SÜRÜM":
        return f"yeni veri sürümünde yeniden koş: {probe}"
    if category.startswith("EKSİK:tohum"):
        return f"eksik tohumları koş: {probe} SEEDS=\"...\""
    return "-"


def lookup(state: dict, task: str, model: str, protocol: str, cell: dict) -> dict:
    """State of one expected cell: ``category`` None when complete, else the missing category."""
    key = (cell["eval_split"], cell["subset"], cell["metric"])
    if cell.get("code", "present") == "missing":
        return {"category": "KOD-YOK", "text": "KOD-YOK", "filled": False}
    candidates = [s for s in sections(state, task) if s["protocol_id"] == protocol and model in s["models"]]
    found = []
    for section in candidates:
        rows = [r for r in section["rows"] if r["model_key"] == model and (r["eval_split"], r["subset"], r["metric"]) == key]
        if rows:
            found.append(cell_stats(rows))
    if found:
        best = max(found, key=lambda s: (s["n_seeds"], s["n_rows"]))
        category = None
        if best["mean"] is None:
            category = "TANIMSIZ:" + "; ".join(best["reasons"])
        elif best["expected"] and best["n_rows"] < best["expected"]:
            category = f"EKSİK:tohum {best['n_rows']}/{best['expected']}"
        text = format_cell(best) if len(found) == 1 else f"BİRDEN-FAZLA-AYAR({len(found)}): bkz. tasks/{task}.md"
        return {"category": category, "text": text, "filled": best["mean"] is not None}
    matching_any = [r for r in state["rows"] if r["task"] == task and r["model_key"] == model
                    and r["protocol_id"] == protocol and (r["eval_split"], r["subset"], r["metric"]) == key]
    if matching_any:
        category = "YALNIZ-ESKİ-SÜRÜM"
    elif candidates:
        category = "EKSİK:metrik-dosyada-yok"
    elif (task, model) in state["embedding_sets"]:
        category = "EKSİK:yalnız-embedding"
    elif not _known_model(state, model):
        category = "EKSİK:model-henüz-yok"
    else:
        category = "EKSİK:koşulmadı"
    return {"category": category, "text": category, "filled": False}


def _known_model(state: dict, model: str) -> bool:
    """A model the ledger has seen at all (any row, any status, a card or an embedding set)."""
    return (any(r["model_key"] == model for r in state["rows"]) or any(m["model_key"] == model for m in state["models"])
            or any(entry[1] == model for entry in state["embedding_sets"]))


def missing_cells(state: dict, models: list, tasks: Optional[list] = None) -> list:
    out = []
    for task in tasks or summary_tasks(state["catalog"]):
        spec = state["catalog"]["tasks"][task]
        for model in models:
            for protocol in spec["required_protocols"]:
                for cell in spec["required"]:
                    found = lookup(state, task, model, protocol, cell)
                    if found["category"] is not None:
                        out.append({"task": task, "model_key": model, "protocol_id": protocol,
                                    "eval_split": cell["eval_split"], "subset": cell["subset"], "metric": cell["metric"],
                                    "category": found["category"], "source": cell.get("source", ""),
                                    "hint": _hint(task, model, protocol, found["category"])})
    return out


def summary_page(state: dict, models: list) -> str:
    catalog = state["catalog"]
    tasks = summary_tasks(catalog)
    lines = header_lines(state, "Özet (hoca sayfası)")
    lines += ["Her görevin birincil metriği, donmuş kodlayıcı + küçük baş (probe) ile; hücre = tohumlar üzerinde "
              "ort ± std. Literatür sütunu farklı protokoldür (çoğu tam ince ayar) ve kendi birimindedir; doğrudan "
              "kıyas değildir. Ayrıntı: tasks/<görev>.md.", "", f"> UYARI: {PRETRAIN_SEED_WARNING}", "",
              "| görev | birincil metrik | " + " | ".join(f"`{m}`" for m in models)
              + " | literatür (en iyi; ayrı protokol) |",
              "|" + "---|" * (3 + len(models))]
    for task in tasks:
        spec = catalog["tasks"][task]
        primary = spec["primary"]
        cells = [lookup(state, task, model, spec["summary_protocol"], primary)["text"] for model in models]
        label = f"{primary['metric']} ({primary['eval_split']}/{primary['subset']}, {spec['summary_protocol']})"
        lines.append(f"| {task} | {label} | " + " | ".join(cells) + " | "
                     + _literature_cell(state, task, primary["eval_split"], primary["subset"], primary["metric"]) + " |")
    required = sum(len(catalog["tasks"][t]["required"]) for t in tasks)
    coded = sum(1 for t in tasks for c in catalog["tasks"][t]["required"] if c.get("code", "present") == "present")
    expected = filled = 0
    for task in tasks:
        spec = catalog["tasks"][task]
        for model in models:
            for protocol in spec["required_protocols"]:
                for cell in spec["required"]:
                    if cell.get("code", "present") == "present":
                        expected += 1
                        filled += lookup(state, task, model, protocol, cell)["filled"]
    lines += ["", "## Sayım", "",
              f"- Katalog: {len(tasks)} görevde {required} zorunlu metrik hücresi (görev × split × alt küme × metrik; "
              f"bir makale raporladıysa zorunlu). Kodumuz bunların {coded} tanesini hesaplıyor, {required - coded} KOD-YOK.",
              f"- Seçili {len(models)} model için (zorunlu protokoller dahil): beklenen {expected} hücre, dolu {filled}, "
              f"eksik {expected - filled}. Liste: eksik_hucreler.md.",
              f"- Defterdeki toplam ölçüm satırı (tohum × metrik × split, tüm durumlar): {state['header']['n_rows']}.", ""]
    return "\n".join(lines) + "\n"


def missing_markdown(state: dict, models: list, missing: list) -> str:
    lines = header_lines(state, "Eksik hücreler")
    counts = defaultdict(int)
    for entry in missing:
        counts[entry["category"].split(" ")[0] if entry["category"].startswith("EKSİK:tohum") else entry["category"]] += 1
    lines += [f"Modeller: {', '.join(models)}. Beklenen = katalogdaki zorunlu hücreler × zorunlu protokoller.", "",
              "Kategori sayıları: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), "",
              "| görev | model | protokol | eval_split | subset | metric | kategori | nasıl tamamlanır |", "|---|---|---|---|---|---|---|---|"]
    for entry in missing:
        lines.append(f"| {entry['task']} | {entry['model_key']} | {entry['protocol_id']} | {entry['eval_split']} | "
                     f"{entry['subset']} | {entry['metric']} | {entry['category']} | {entry['hint']} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------------------------------

def report_models(root: Path, models: Optional[list]) -> list:
    """``--models``, else ``summary_models`` in <root>/comparisons.yaml, else ``DEFAULT_SUMMARY_MODELS``."""
    if models:
        return list(models)
    path = Path(root) / "comparisons.yaml"
    if not path.exists():
        return list(DEFAULT_SUMMARY_MODELS)
    listed = (yaml.safe_load(path.read_text()) or {}).get("summary_models")
    if not listed:
        raise ValueError(f"{path} has no summary_models list")
    return list(listed)


def _csv_text(rows: list) -> str:
    buffer = io.StringIO()
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return buffer.getvalue()


def _write(path: Path, text: str, written: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text)
    partial.replace(path)
    written.append(path)


def write_reports(root: Optional[Path] = None, *, models: Optional[list] = None, which=("tasks", "timeline", "summary",
                                                                                       "missing"),
                  state: Optional[dict] = None) -> list:
    root = ledger.results_root(root)
    state = state or load_state(root)
    reports = root / "reports"
    written: list = []
    tasks = sorted({row["task"] for row in state["rows"]})
    if "tasks" in which:
        for task in tasks:
            markdown, rows = task_report(state, task)
            _write(reports / "tasks" / f"{task}.md", markdown, written)
            _write(reports / "tasks" / f"{task}.csv", _csv_text(rows), written)
    if "timeline" in which:
        for task in tasks:
            markdown, rows = timeline_report(state, task)
            _write(reports / "timeline" / f"{task}.md", markdown, written)
            _write(reports / "timeline" / f"{task}.csv", _csv_text(rows), written)
    if "summary" in which or "missing" in which:
        selected = report_models(root, models)
        if "summary" in which:
            _write(reports / "ozet_hoca.md", summary_page(state, selected), written)
        if "missing" in which:
            missing = missing_cells(state, selected)
            _write(reports / "eksik_hucreler.md", missing_markdown(state, selected, missing), written)
            _write(reports / "eksik_hucreler.csv", _csv_text(missing), written)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("report", "summary", "missing"))
    parser.add_argument("--results-root", type=Path, default=None, help=f"default: ${ledger.ENV_VAR}")
    parser.add_argument("--models", nargs="*", default=None,
                        help="default: summary_models in <root>/comparisons.yaml, else DEFAULT_SUMMARY_MODELS")
    args = parser.parse_args(argv)
    which = {"report": ("tasks", "timeline", "summary", "missing"), "summary": ("summary",),
             "missing": ("missing",)}[args.command]
    written = write_reports(args.results_root, models=args.models, which=which)
    print(json.dumps([str(p) for p in written], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
