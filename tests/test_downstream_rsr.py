"""ATOM3D RSR downstream task: ranking metrics, decoy -> RNA graph contract, processing QC, download, probe, ledger.

Everything runs on CPU with a tiny synthetic fixture (a few puzzles x a few decoys written as real LMDB files); no
network and no real data. The metric implementation is checked against scipy and a brute-force top-1 loss on random data.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import math
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from scipy import stats

from core.pretraining_config import create_pretraining_config, load_config_yaml
from core.pretraining_model import PretrainingESAModel
from data_loading.atom_position_vocab import (
    RNA_BACKBONE_ATOMS, RNA_BASE_ATOMS_BY_NUCLEOTIDE, VIRTUAL_ATOMIC_NUMBER, is_virtual_position_code,
)
from downstream import ledger, probe
from downstream.common import encoder as enc
from downstream.common import metrics as M
from downstream.common import rna_graph as rg
from downstream.tasks import TASKS, load_task
from downstream.tasks.rsr import dataset as ds
from downstream.tasks.rsr import prepare
from downstream_helpers import cif_text
from ledger_helpers import card, write_probe

REPO = Path(__file__).resolve().parent.parent
ATOM_COLUMNS = [
    "ensemble", "subunit", "structure", "model", "chain", "hetero", "insertion_code", "residue", "segid", "resname",
    "altloc", "occupancy", "bfactor", "x", "y", "z", "element", "name", "fullname", "serial_number",
]
ELEMENT = {"P": "P", "O": "O", "C": "C", "N": "N"}


# ---------------------------------------------------------------------------
# Synthetic decoys and LMDBs
# ---------------------------------------------------------------------------

def nucleotide_atoms(resname, index, drop=()):
    """Heavy atoms of one nucleotide with distinct, 3-decimal coordinates (rounded as an mmCIF would hold them)."""
    x = 6.0 * index
    atoms, k = [], 0
    for name in RNA_BACKBONE_ATOMS:
        atoms.append((name, ELEMENT[name[0]], (x + 0.5 * k, round(0.3 * (k % 3), 3), round(0.2 * (k % 2), 3))))
        k += 1
    for j, name in enumerate(RNA_BASE_ATOMS_BY_NUCLEOTIDE[resname]):
        atoms.append((name, ELEMENT[name[0]], (x + 0.4 * j, 2.0 + 0.5 * j, round(0.3 * (j % 3), 3))))
    return [atom for atom in atoms if atom[0] not in drop]


def decoy_rows(nucleotides, ensemble, *, hydrogens=True, chain="A", placeholders=()):
    """``nucleotides``: [(resname, atoms)]; hydrogens are added after the heavy atoms of each nucleotide.

    ``placeholders`` = [(residue number, atom name)]: Rosetta placeholder atoms with an empty element, as in the real
    data (RPA1 / RPA2 / RPB1 / RPB2 after the atoms of a chain-end nucleotide)."""
    rows, serial = [], 0
    for residue, (resname, atoms) in enumerate(nucleotides, start=1):
        extra = [("H5'", "H", (atoms[0][2][0] + 0.9, 0.1, 0.1)), ("H2", "H", (atoms[-1][2][0] + 0.9, 0.2, 0.3))]
        extra += [(name, "", (atoms[0][2][0] - 1.0, 5.0 + k, 0.0))
                  for k, (number, name) in enumerate(placeholders) if number == residue]
        for name, element, (x, y, z) in atoms + (extra if hydrogens else [e for e in extra if e[1] == ""]):
            rows.append([ensemble, 0, ensemble, 0, chain, " ", " ", residue, "    ", f"  {resname}", " ", 1.0, 0.0,
                         float(x), float(y), float(z), element, name, name, serial])
            serial += 1
    return rows


def frame(rows):
    return {"columns": list(ATOM_COLUMNS), "index": list(range(len(rows))), "data": rows}


def rsr_item(puzzle, index, rmsd, *, sequence="GAUC", drops=None, hydrogens=True, placeholders=(), **scores):
    """A real-looking item: ``id`` is the text of ``(puzzle, model)``, the atoms' ``ensemble`` is the model's file name
    (unique per model, NOT the puzzle), residue names carry Rosetta's leading blanks."""
    drops = drops or {}
    nucleotides = [(resname, nucleotide_atoms(resname, i, drops.get(i, ()))) for i, resname in enumerate(sequence)]
    model = f"S_{index:06d}_000"
    return {"id": str((puzzle, model)),
            "atoms": frame(decoy_rows(nucleotides, f"{model}.pdb", hydrogens=hydrogens, placeholders=placeholders)),
            "scores": {"rms": float(rmsd), "score": -10.0, **scores}, "types": {}, "file_path": "x"}


def write_lmdb(path, items, fmt="json"):
    import lmdb

    def dump(value):
        if fmt == "msgpack":
            import msgpack
            return gzip.compress(msgpack.packb(value))
        return gzip.compress(json.dumps(value).encode())

    path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), map_size=1 << 28)
    with env.begin(write=True) as txn:
        for index, item in enumerate(items):
            txn.put(str(index).encode(), dump(item))
        txn.put(b"num_examples", str(len(items)).encode())
        txn.put(b"serialization_format", fmt.encode())
        txn.put(b"id_to_idx", dump({str(i): i for i in range(len(items))}))
    env.close()


def split_items(puzzle, count, *, offset=0):
    """``count`` decoys of one puzzle, RMSDs 3, 6, 9, ... (strictly increasing, so a ranking has a clear truth)."""
    return [rsr_item(puzzle, offset + i, 3.0 * (i + 1), sequence="GAUCGA"[: 4 + (i % 3)]) for i in range(count)]


def tiny_config(tmp_path, *, counts=None, puzzles=None, qc=None):
    config = ds.load_task_config(data_root=tmp_path / "data")
    config["expected_counts"] = counts or {"train": 4, "val": 3, "test": 3}
    config["expected_puzzles"] = puzzles or {"train": 1, "val": 1, "test": 1}
    config["qc"] = {**config["qc"], **(qc or {})}
    return config


def make_data(tmp_path, *, fmt="json", train=None, val=None, test=None):
    config = tiny_config(tmp_path)
    root = ds.task_paths(config)["extracted"] / "split" / "RSR-candidates-split-by-time" / "data"
    write_lmdb(root / "train", train or split_items("rp01", 4), fmt)
    write_lmdb(root / "val", val or split_items("rp02", 3), fmt)
    write_lmdb(root / "test", test or split_items("rp03", 3), fmt)
    return config


def _tiny_model_config():
    values = load_config_yaml(str(REPO / "core" / "pretraining_config_multidomain_1m.yaml"))
    layers = len(values["layer_types"])
    values.update(hidden_dims=[32] * layers, num_heads=[4] * layers, graph_dim=32, mlp_hidden_size=64,
                  gaussian_kernels=16)
    return create_pretraining_config(**values)


# ---------------------------------------------------------------------------
# Ranking metrics against scipy (independent of the implementation under test)
# ---------------------------------------------------------------------------

def _random_puzzles(seed, sizes=(40, 25, 60)):
    rng = np.random.default_rng(seed)
    targets, predictions, groups = [], [], []
    for index, size in enumerate(sizes):
        truth = rng.gamma(4.0, 3.0, size) + 4.0 * index
        targets.append(truth)
        predictions.append(truth * rng.uniform(0.5, 1.5) + rng.normal(0, 4.0, size) + 2.0 * index)
        groups += [f"rp{index + 1:02d}"] * size
    return np.concatenate(targets), np.concatenate(predictions), np.array(groups)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ranking_metrics_match_scipy_and_brute_force(seed):
    targets, predictions, groups = _random_puzzles(seed)
    result = M.evaluate_ranking_regression(targets, predictions, groups)
    spearman, pearson, loss = [], [], []
    for label in dict.fromkeys(groups.tolist()):
        t, p = targets[groups == label], predictions[groups == label]
        spearman.append(stats.spearmanr(t, p)[0])
        pearson.append(stats.pearsonr(t, p)[0])
        best = min(range(len(p)), key=lambda i: (p[i], i))
        loss.append(t[best] - t.min())
    mean = result["mean_per_puzzle"]
    assert mean["spearman_rho"] == pytest.approx(np.mean(spearman), abs=1e-12)
    assert mean["pearson_r"] == pytest.approx(np.mean(pearson), abs=1e-12)
    assert mean["top1_loss"] == pytest.approx(np.mean(loss), abs=1e-12)
    assert mean["per_puzzle_spearman_rho"] == pytest.approx(spearman) and mean["puzzles"] == ["rp01", "rp02", "rp03"]
    assert mean["num_puzzles"] == 3 and mean["num_samples"] == targets.size
    pooled = result["global"]
    assert pooled["spearman_rho"] == pytest.approx(stats.spearmanr(targets, predictions)[0], abs=1e-12)
    assert pooled["pearson_r"] == pytest.approx(stats.pearsonr(targets, predictions)[0], abs=1e-12)
    assert pooled["rmse"] == pytest.approx(math.sqrt(np.mean((targets - predictions) ** 2)))
    assert pooled["mae"] == pytest.approx(np.mean(np.abs(targets - predictions)))
    assert mean["correlation_undefined_reason"] is None and pooled["correlation_undefined_reason"] is None
    assert mean["spearman_rho"] != pytest.approx(pooled["spearman_rho"], abs=1e-3), "mean and global must differ here"


def test_ranking_metrics_handle_ties_perfect_and_reversed_rankings():
    truth = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0])
    groups = np.array(["a"] * 4 + ["b"] * 3)
    perfect = M.evaluate_ranking_regression(truth, truth * 2 + 1, groups)
    assert perfect["mean_per_puzzle"]["spearman_rho"] == pytest.approx(1.0)
    assert perfect["mean_per_puzzle"]["top1_loss"] == 0.0
    reversed_ = M.evaluate_ranking_regression(truth, -truth, groups)
    assert reversed_["mean_per_puzzle"]["spearman_rho"] == pytest.approx(-1.0)
    assert reversed_["mean_per_puzzle"]["top1_loss"] == pytest.approx(((4.0 - 1.0) + (30.0 - 10.0)) / 2)
    # ties: scipy's average ranks, and a tie at the lowest prediction counts with the mean of the tied true values
    tied = np.array([1.0, 1.0, 2.0, 3.0, 5.0, 5.0, 7.0])
    result = M.evaluate_ranking_regression(truth, tied, groups)
    expected = np.mean([stats.spearmanr(truth[:4], tied[:4])[0], stats.spearmanr(truth[4:], tied[4:])[0]])
    assert result["mean_per_puzzle"]["spearman_rho"] == pytest.approx(expected, abs=1e-12)
    assert result["mean_per_puzzle"]["per_puzzle_top1_loss"] == pytest.approx([(1.0 + 2.0) / 2 - 1.0, (10.0 + 20.0) / 2 - 10.0])
    assert M.top1_loss(truth[:4], tied[:4]) == M.top1_loss(truth[:4][::-1], tied[:4][::-1])


def test_a_puzzle_without_a_defined_correlation_makes_the_mean_undefined_never_a_partial_mean():
    truth = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0])
    groups = np.array(["a"] * 4 + ["b"] * 3)
    constant_b = np.array([4.0, 3.0, 2.0, 1.0, 5.0, 5.0, 5.0])
    result = M.evaluate_ranking_regression(truth, constant_b, groups)
    mean = result["mean_per_puzzle"]
    assert mean["spearman_rho"] is None and mean["pearson_r"] is None
    assert "constant_prediction in 1 group(s) (b)" in mean["correlation_undefined_reason"]
    assert mean["per_puzzle_spearman_rho"][0] == pytest.approx(-1.0) and mean["per_puzzle_spearman_rho"][1] is None
    assert result["global"]["spearman_rho"] is not None
    single = M.evaluate_ranking_regression(truth[:5], truth[:5], np.array(["a"] * 4 + ["solo"]))
    assert "fewer_than_two_samples" in single["mean_per_puzzle"]["correlation_undefined_reason"]


def test_ranking_metrics_fail_loud_on_bad_inputs():
    truth = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="differ"):
        M.evaluate_ranking_regression(truth, truth[:2], np.array(["a", "a", "a"]))
    with pytest.raises(ValueError, match="NaN"):
        M.evaluate_ranking_regression(truth, np.array([1.0, np.nan, 2.0]), np.array(["a"] * 3))
    with pytest.raises(ValueError, match="non-empty string"):
        M.evaluate_ranking_regression(truth, truth, np.array(["a", "", "a"]))
    with pytest.raises(ValueError, match="non-empty string"):
        M.evaluate_ranking_regression(truth, truth, np.array([1, 1, 2]))
    with pytest.raises(ValueError, match="group labels"):
        M.evaluate_ranking_regression(truth, truth, np.array(["a", "a"]))


def test_validation_score_is_the_mean_spearman_and_needs_groups():
    targets, predictions, groups = _random_puzzles(0)
    score = probe.validation_score("ranking_regression", targets.reshape(-1, 1), predictions, groups=groups)
    assert score == pytest.approx(M.evaluate_ranking_regression(targets, predictions, groups)["mean_per_puzzle"]["spearman_rho"])
    with pytest.raises(ValueError, match="group"):
        probe.validation_score("ranking_regression", targets, predictions)
    constant = probe.validation_score("ranking_regression", targets, np.zeros_like(targets), groups=groups)
    assert constant == -float("inf"), "an undefined epoch must never be selected"
    assert probe.validation_label("ranking_regression", 0.25) == ("val_mean_spearman", 0.25)


# ---------------------------------------------------------------------------
# Registry, config, catalog
# ---------------------------------------------------------------------------

def test_task_is_registered_with_a_consistent_config():
    assert "rsr" in TASKS
    task = load_task("rsr")
    assert task.KIND == "ranking_regression" and callable(task.group_labels)
    config = task.load_task_config()
    assert config["modality"] == "rna" and config["expected_counts"] == {"train": 12479, "val": 4000, "test": 4000}
    assert config["expected_puzzles"] == {"train": 13, "val": 4, "test": 4}
    files = config["sources"][config["source"]]["files"]
    assert {entry["md5"].__len__() for entry in files.values()} == {32}
    assert files["split"]["bytes"] == 1269970448 and files["split"]["md5"] == "ba48438ec1df66c583a76cc42e5fddb5"
    catalog = ledger.load_catalog()
    spec = catalog["tasks"]["rsr"]
    assert spec["kind"] == task.KIND and spec["primary"]["metric"] == "spearman_rho"
    assert {(c["subset"], c["metric"]) for c in spec["required"]} == {
        ("mean_per_puzzle", "spearman_rho"), ("global", "spearman_rho"),
        ("mean_per_puzzle", "pearson_r"), ("global", "pearson_r")}
    assert ledger.metric_info(catalog, "rsr", "top1_loss") == {"direction": "lower", "unit": "Angstrom",
                                                              "is_catalog_metric": True}


def test_config_refuses_unimplemented_choices(tmp_path):
    import yaml

    def broken(**changes):
        values = yaml.safe_load(ds.TASK_YAML.read_text())
        for key, value in changes.items():
            target = values
            *parents, last = key.split("__")
            for parent in parents:
                target = target[parent]
            target[last] = value
        path = tmp_path / "task.yaml"
        path.write_text(yaml.safe_dump(values))
        return path

    with pytest.raises(ValueError, match="missing_atoms"):
        ds.load_task_config(broken(missing_atoms__incomplete_base="drop_structure"))
    with pytest.raises(ValueError, match="hydrogens"):
        ds.load_task_config(broken(structure__hydrogens="keep"))
    with pytest.raises(ValueError, match="expected_counts"):
        ds.load_task_config(broken(expected_counts={"train": 1, "val": 1}))
    with pytest.raises(ValueError, match="resname_aliases"):
        ds.load_task_config(broken(structure__resname_aliases={"X": "ALA"}))
    with pytest.raises(ValueError, match="puzzle_from"):
        ds.load_task_config(broken(puzzle_from="id"))


# ---------------------------------------------------------------------------
# Decoy -> RNA graph
# ---------------------------------------------------------------------------

def _settings(tmp_path=None):
    return ds.processing_settings(ds.load_task_config())


def test_read_decoy_drops_hydrogens_and_altlocs_and_reads_the_puzzle():
    item = rsr_item("rp07", 3, 12.5, sequence="GAU")
    columns = {name: i for i, name in enumerate(ATOM_COLUMNS)}
    item["atoms"]["data"][4][columns["altloc"]] = "B"        # an alternate location other than blank / A is skipped
    item["atoms"]["data"][1][columns["name"]] = "O1P"        # a legacy name is read through the adapter's aliases
    decoy = ds.read_decoy(item, _settings(), label="t")
    assert decoy.puzzle == "rp07" and decoy.target == 12.5 and decoy.source_id == item["id"]
    assert [n.resname for n in decoy.nucleotides] == ["G", "A", "U"]
    assert decoy.counts["hydrogens_skipped"] == 6 and decoy.counts["altloc_atoms_skipped"] == 1
    assert all(element != "H" for n in decoy.nucleotides for _, element, _ in n.atoms)
    assert "OP1" in {name for name, _, _ in decoy.nucleotides[0].atoms}


@pytest.mark.parametrize("mutate,reason", [
    (lambda item: item["atoms"]["data"][0].__setitem__(9, "ALA"), "unsupported_residue"),
    (lambda item: item["atoms"]["data"].append(list(item["atoms"]["data"][1])), "duplicate_atoms"),
    (lambda item: item["scores"].pop("rms"), "bad_target"),
    (lambda item: item["scores"].__setitem__("rms", float("nan")), "bad_target"),
    (lambda item: item["scores"].__setitem__("rms", -1.0), "bad_target"),
    (lambda item: item.__setitem__("id", "not a tuple"), "bad_puzzle"),
    (lambda item: item.__setitem__("id", "('rp/01', 'S_1')"), "bad_puzzle"),
    (lambda item: item.__setitem__("id", "('only one',)"), "bad_puzzle"),
    (lambda item: item.__setitem__("id", "__import__('os').getcwd()"), "bad_puzzle"),
    (lambda item: item["atoms"]["data"][3].__setitem__(16, ""), "bad_frame"),
    (lambda item: item.__setitem__("atoms", {"columns": ["x"], "index": [0], "data": [[1.0]]}), "bad_frame"),
])
def test_unconvertible_decoys_are_counted_with_their_reason(mutate, reason):
    item = rsr_item("rp01", 0, 5.0)
    mutate(item)
    result = ds.process_item(item, 0, _settings())
    assert result.status == "failed" and result.sample is None
    assert result.reason.startswith(reason), result.reason
    if reason != "bad_puzzle":   # a failure after the id was read still names its puzzle (never an empty puzzle)
        assert result.puzzle == "rp01" and result.decoy_id == "rp01/000000" and result.source_id == item["id"]


def test_rosetta_placeholder_atoms_are_removed_and_counted_and_leave_the_graph_unchanged():
    """The real pattern: RPA1/RPA2/RPB1/RPB2 atoms with an empty element, and an ensemble that is a file name."""
    plain = ds.process_item(rsr_item("rna_puzzle_14_bound", 5, 7.0, sequence="GAUC"), 5, _settings())
    marked = ds.process_item(rsr_item("rna_puzzle_14_bound", 5, 7.0, sequence="GAUC",
                                      placeholders=[(1, "RPB1"), (4, "RPA1"), (4, "RPA2")]), 5, _settings())
    assert marked.status == "ok", marked.reason
    assert marked.puzzle == "rna_puzzle_14_bound" and marked.decoy_id == "rna_puzzle_14_bound/000005"
    assert marked.report["placeholder_atoms_removed"] == 3 and plain.report["placeholder_atoms_removed"] == 0
    assert marked.report["atoms_read"] == plain.report["atoms_read"] + 3
    assert marked.report["unexpected_atoms_removed"] == 0 and marked.report["stored_nodes"] == plain.report["stored_nodes"]
    assert torch.equal(marked.sample.pos, plain.sample.pos) and torch.equal(marked.sample.z, plain.sample.z)
    # an atom without an element that is not a listed placeholder still fails the model, with its name
    item = rsr_item("rp01", 0, 5.0, placeholders=[(1, "XYZ9")])
    failed = ds.process_item(item, 0, _settings())
    assert failed.status == "failed" and failed.reason.startswith("bad_frame") and "XYZ9" in failed.reason
    # a listed name WITH an element is a real (unknown) atom: removed through the unexpected-atom rule, counted there
    item = rsr_item("rp01", 0, 5.0, sequence="GAUC")
    item["atoms"]["data"].append(["m.pdb", 0, "m.pdb", 0, "A", " ", " ", 1, "    ", "  G", " ", 1.0, 0.0, 9.0, 9.0, 9.0, "C", "RPA1", "RPA1", 0])
    counted = ds.process_item(item, 0, _settings())
    assert counted.status == "ok" and counted.report["unexpected_atoms_removed"] == 1
    assert counted.report["placeholder_atoms_removed"] == 0


def test_the_puzzle_is_read_from_the_id_not_from_the_ensemble_column():
    item = rsr_item("rna_puzzle_2_hack", 3, 4.0, sequence="GAUC")
    ensembles = {row[0] for row in item["atoms"]["data"]}
    assert ensembles == {"S_000003_000.pdb"}
    decoy = ds.read_decoy(item, _settings(), label="t")
    assert decoy.puzzle == "rna_puzzle_2_hack" and decoy.source_id == "('rna_puzzle_2_hack', 'S_000003_000')"
    assert ds.parse_puzzle({"id": ["rna_puzzle_9", "S_1"]}) == ("rna_puzzle_9", "['rna_puzzle_9', 'S_1']")


def test_complete_decoy_is_23_slots_per_nucleotide_with_v_exactly_on_padding():
    result = ds.process_item(rsr_item("rp01", 0, 5.0, sequence="GAUC"), 0, _settings())
    assert result.status == "ok", result.reason
    compact = result.sample
    assert int(compact.z.numel()) == 4 * rg.SLOTS_PER_NUCLEOTIDE and result.report["stored_nodes"] == 92
    assert result.report["nucleotides_missing_backbone"] == 0 and result.report["nucleotides_missing_base"] == 0
    sample = rg.runtime_sample(compact, rg.edge_free_schema({}), _tiny_model_config())
    codes = list(sample.pos_code)
    assert [bool(v) for v in (sample.z == VIRTUAL_ATOMIC_NUMBER).tolist()] == [is_virtual_position_code(c) for c in codes]
    assert int((sample.z == VIRTUAL_ATOMIC_NUMBER).sum()) == sum(11 - len(RNA_BASE_ATOMS_BY_NUCLEOTIDE[r]) for r in "GAUC")
    assert sample.dataset_type == "rna" and int(sample.template_reason_bits.abs().sum()) == 0
    assert not bool(sample.downstream_masked_block.any()) and sample.edge_index.shape == (2, 0)
    assert sample.block_idx.tolist() == sorted(sample.block_idx.tolist())


def test_decoy_graph_equals_the_pretraining_adapter_graph_for_the_same_atoms(tmp_path):
    from data_loading.adapters.rna_adapter import RNAAdapter
    from data_loading.cache_to_pyg import OptimizedUniversalDataset

    sequence = "GAUC"
    nucleotides = [(r, nucleotide_atoms(r, i, drop=("O4",) if i == 2 else ())) for i, r in enumerate(sequence)]
    path = tmp_path / "decoy.cif"
    path.write_text(cif_text("decoy", {"A": [(r, atoms, "ATOM") for r, atoms in nucleotides]}))
    with contextlib.redirect_stdout(io.StringIO()):
        molecule = RNAAdapter().convert_to_universal(path)
    converter = object.__new__(OptimizedUniversalDataset)
    converter.molecule_max_atoms, converter.cutoff_distance, converter.max_neighbors = None, 5.0, 64
    converter.pre_filter = converter.pre_transform = None
    reference = converter._universal_to_pyg(molecule, raise_on_error=True)

    item = {"id": str(("rp01", "x")), "atoms": frame(decoy_rows(nucleotides, "x.pdb")), "scores": {"rms": 1.0}}
    mine = ds.process_item(item, 0, _settings())
    assert mine.status == "ok" and mine.report["base_atoms_missing"] == 1
    node = rg.build_node_sample(ds.read_decoy(item, _settings(), label="t").nucleotides, "x")
    assert node.z.tolist() == reference.z.tolist() and list(node.pos_code) == list(reference.pos_code)
    real = torch.tensor([not is_virtual_position_code(c) for c in node.pos_code])
    assert torch.allclose(node.pos[real], reference.pos[real], atol=1e-4)
    assert node.block_idx.tolist() == reference.block_idx.tolist()


def test_missing_atoms_are_counted_never_repaired_or_hidden():
    item = rsr_item("rp01", 0, 5.0, sequence="GAUC", drops={0: ("P", "OP1", "OP2"), 2: ("O4",)})
    item["atoms"]["data"] += [["rp01", 0, "s", 0, "A", " ", " ", 2, "", "A", " ", 1.0, 0.0, 99.0, 0.0, 0.0, "C", "XX1", "XX1", 0],
                              ["rp01", 0, "s", 0, "A", " ", " ", 2, "", "A", " ", 1.0, 0.0, 98.0, 0.0, 0.0, "O", "OP3", "OP3", 0]]
    result = ds.process_item(item, 0, _settings())
    assert result.status == "ok", result.reason
    report = result.report
    assert report["nucleotides_missing_backbone"] == 1 and report["nucleotides_missing_base"] == 1
    assert report["base_atoms_missing"] == 1 and report["unexpected_atoms_removed"] == 1
    assert report["terminal_atoms_dropped"] == 1
    assert result.sample.z.numel() == 4 * 23 - 3   # the 5' nucleotide keeps the 9 backbone atoms it has
    sample = rg.runtime_sample(result.sample, rg.edge_free_schema({}), _tiny_model_config())
    assert rg.quarantined_nucleotides(sample) == 1, "the incomplete-backbone block is quarantined, as in pretraining"


def test_nucleotide_records_and_graph_builder_refuse_bad_input():
    with pytest.raises(ValueError, match="resname"):
        rg.NucleotideRecord("ALA", [("P", "P", (0, 0, 0))])
    with pytest.raises(ValueError, match="duplicated"):
        rg.NucleotideRecord("A", [("P", "P", (0, 0, 0)), ("P", "P", (1, 0, 0))])
    with pytest.raises(ValueError, match="unexpected atoms"):
        rg.build_node_sample([rg.NucleotideRecord("A", [("P", "P", (0, 0, 0)), ("ZZ", "C", (1, 0, 0))])], "x")
    with pytest.raises(ValueError, match="no nucleotides"):
        rg.build_node_sample([], "x")
    compact = ds.process_item(rsr_item("rp01", 0, 5.0), 0, _settings()).sample
    with pytest.raises(ValueError, match="domain"):
        rg.runtime_sample(compact, {**rg.edge_free_schema({}), "domain": "protein"}, _tiny_model_config())
    config = _tiny_model_config()
    config.geometry_neutral_virtual_slots = {"enabled": True, "protein": True, "rna": False}
    with pytest.raises(ValueError, match="geometry_neutral_virtual_slots"):
        rg.runtime_sample(compact, rg.edge_free_schema({}), config)


def test_short_models_are_dropped_and_counted():
    result = ds.process_item(rsr_item("rp01", 0, 5.0, sequence="GA"), 0, _settings())
    assert result.status == "dropped" and result.reason.startswith("too_few_nucleotides")


# ---------------------------------------------------------------------------
# LMDB discovery and processing
# ---------------------------------------------------------------------------

def test_lmdb_layout_is_discovered_and_ambiguity_is_refused(tmp_path):
    config = make_data(tmp_path)
    found = ds.lmdb_dirs(config)
    assert {name: path.name for name, path in found.items()} == {"train": "train", "val": "val", "test": "test"}
    assert [ds.lmdb_length(found[s]) for s in ("train", "val", "test")] == [4, 3, 3]
    write_lmdb(ds.task_paths(config)["extracted"] / "split" / "other" / "train", split_items("rp01", 1))
    with pytest.raises(ValueError, match="exactly one LMDB directory"):
        ds.lmdb_dirs(config)
    with pytest.raises(ValueError, match="found"):
        ds.find_lmdb_splits(tmp_path / "empty", ["train", "val", "test"])


def test_pickle_serialization_is_refused(tmp_path):
    import lmdb

    path = tmp_path / "lmdb"
    write_lmdb(path, split_items("rp01", 2))
    env = lmdb.open(str(path), map_size=1 << 26)
    with env.begin(write=True) as txn:
        txn.put(b"serialization_format", b"pkl")
    env.close()
    with pytest.raises(ValueError, match="pickle is refused"):
        ds.lmdb_length(path)


@pytest.mark.parametrize("fmt", ["json", "msgpack"])
def test_build_processed_round_trip_and_qc_files(tmp_path, fmt):
    config = make_data(tmp_path, fmt=fmt)
    wrong = tiny_config(tmp_path, counts={"train": 5, "val": 3, "test": 3})
    with pytest.raises(ValueError, match="task.yaml expects 5"):
        ds.build_processed(wrong, splits=["train"], workers=1)
    assert ds.build_processed(config, splits=["train", "val", "test"], workers=1, chunk=2) == 0
    out = ds.processed_dir(config)
    assert out.name == "v1"
    split = ds.ProcessedSplit(out / "train.pt")
    assert len(split) == 4 and split.chains == [f"rp01/{i:06d}" for i in range(4)]
    assert split.labels.shape == (4, 1) and split.labels[:, 0].tolist() == [3.0, 6.0, 9.0, 12.0]
    assert set(split.groups) == {"rp01"} and split.report["status"] == {"ok": 4}
    assert [n // 23 for n in (int(s.z.numel()) for s in split.payload["samples"])] == [4, 5, 6, 4]
    qc = json.loads((out / "qc_train.json").read_text())
    assert qc["puzzles"] == 1 and qc["models_per_puzzle_kept"] == {"rp01": 4} and qc["totals"]["hydrogens_skipped"] > 0
    rows = (out / "qc_train.tsv").read_text().splitlines()
    assert rows[0].split("\t")[:3] == ["decoy", "split", "status"] and len(rows) == 5
    assert "S_000000_000" in rows[1]    # the raw LMDB id is kept next to the model id
    # the runtime graph is deterministic (no random edges for RNA) whatever the seed
    first, second = split.runtime(1, _tiny_model_config(), seed=0), split.runtime(1, _tiny_model_config(), seed=7)
    assert torch.equal(first.pos, second.pos) and torch.equal(first.z, second.z)
    with pytest.raises(FileExistsError):   # a processed split is never overwritten
        ds.build_processed(config, splits=["train"], workers=1)


def test_qc_failure_writes_no_file_and_exits_one(tmp_path, capsys):
    train = split_items("rp01", 3) + [rsr_item("rp01", 9, 5.0, sequence="GAUC")]
    train[1]["atoms"]["data"][0][9] = "ALA"
    config = make_data(tmp_path, train=train)
    code = ds.build_processed(config, splits=["train", "val", "test"], workers=1)
    out = ds.processed_dir(config)
    assert code == 1 and not (out / "train.pt").exists() and (out / "val.pt").exists() and (out / "test.pt").exists()
    qc = json.loads((out / "qc_train.json").read_text())
    assert qc["status"] == {"ok": 3, "failed": 1} and qc["reasons"] == {"unsupported_residue": 1}
    assert "QC FAILED" in capsys.readouterr().out
    failed = [r for r in (out / "qc_train.tsv").read_text().splitlines() if "\tfailed\t" in r]
    assert len(failed) == 1 and "unsupported_residue" in failed[0]
    # the same failure is tolerated when the configured fraction allows it, and still counted
    loose = tiny_config(tmp_path, qc={"max_failed_structure_fraction": 0.3})
    loose["data_root"] = str(tmp_path / "data2")
    shutil.copytree(tmp_path / "data" / "raw", tmp_path / "data2" / "raw")
    assert ds.build_processed(loose, splits=["train"], workers=1) == 0
    assert json.loads((ds.processed_dir(loose) / "qc_train.json").read_text())["status"] == {"ok": 3, "failed": 1}


def test_wrong_puzzle_count_lost_puzzles_and_split_leaks_stop_processing(tmp_path, capsys):
    mixed = split_items("rp01", 2) + split_items("rp02", 2, offset=10)
    config = make_data(tmp_path, train=mixed)
    assert ds.build_processed(config, splits=["train"], workers=1) == 1   # 2 puzzles, 1 expected
    assert "2 puzzles, expected 1" in capsys.readouterr().out
    leaky = make_data(tmp_path / "leak", val=split_items("rp01", 3, offset=20))
    code = ds.build_processed(leaky, splits=["train", "val", "test"], workers=1)
    assert code == 1 and "SPLIT LEAK: puzzles ['rp01'] occur in both train and val" in capsys.readouterr().out


def test_incomplete_base_fraction_is_a_qc_limit(tmp_path):
    train = [rsr_item("rp01", i, 3.0 + i, drops={1: ("N6",)}) for i in range(4)]
    config = make_data(tmp_path, train=train)
    assert ds.build_processed(config, splits=["train"], workers=1) == 1
    qc = json.loads((ds.processed_dir(config) / "qc_train.json").read_text())
    assert qc["incomplete_base_fraction"] == 1.0 and qc["totals"]["nucleotides_missing_base"] == 4


def test_parallel_processing_gives_the_same_samples_as_serial(tmp_path):
    serial = make_data(tmp_path / "a")
    parallel = make_data(tmp_path / "b")
    assert ds.build_processed(serial, splits=["train"], workers=1, chunk=1) == 0
    assert ds.build_processed(parallel, splits=["train"], workers=2, chunk=1) == 0
    a, b = (ds.ProcessedSplit(ds.processed_dir(c) / "train.pt") for c in (serial, parallel))
    assert a.ids == b.ids and all(torch.equal(x.pos, y.pos) and torch.equal(x.z, y.z)
                                  for x, y in zip(a.payload["samples"], b.payload["samples"]))


def test_group_labels_refuse_ids_without_a_puzzle():
    assert ds.group_labels(["rp01/000001", "rp14/000002"]).tolist() == ["rp01", "rp14"]
    with pytest.raises(ValueError, match="not '<puzzle>"):
        ds.group_labels(["rp01"])
    with pytest.raises(ValueError, match="not '<puzzle>"):
        ds.group_labels(["/000001"])


# ---------------------------------------------------------------------------
# Download / validate
# ---------------------------------------------------------------------------

def _tar(path, members):
    with tarfile.open(path, "w:gz") as handle:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            handle.addfile(info, io.BytesIO(data))


def _fixture_source(tmp_path, config):
    """Tiny archives shaped like the record's files; returns ``{name: path}`` with pins patched into ``config``."""
    src = tmp_path / "source"
    lmdb_root = tmp_path / "lmdb_build"
    for split, items in (("train", split_items("rp01", 4)), ("val", split_items("rp02", 3)), ("test", split_items("rp03", 3))):
        write_lmdb(lmdb_root / "RSR-split" / "data" / split, items)
    src.mkdir()
    members = {str(p.relative_to(lmdb_root)): p.read_bytes() for p in sorted(lmdb_root.rglob("*")) if p.is_file()}
    _tar(src / "split.tar.gz", members)
    _tar(src / "indices.tar.gz", {f"indices/{s}_indices.txt": ("\n".join(map(str, range(n))) + "\n").encode()
                                  for s, n in (("train", 4), ("val", 3), ("test", 3))})
    (src / "README.md").write_text("readme")
    (src / "LICENSE_SA").write_text("license")
    files = config["sources"][config["source"]]["files"]
    paths = {}
    for key, name in (("split", "split.tar.gz"), ("indices", "indices.tar.gz"), ("readme", "README.md"),
                      ("license", "LICENSE_SA")):
        data = (src / name).read_bytes()
        files[key].update(md5=hashlib.md5(data).hexdigest(), bytes=len(data))
        paths[files[key]["name"]] = src / name
    return paths


def test_download_verifies_pins_records_sha256_and_checks_counts(tmp_path, monkeypatch):
    config = tiny_config(tmp_path)
    paths = _fixture_source(tmp_path, config)
    fetched = []

    def fake_fetch(url, dest):
        fetched.append(Path(url).name)
        dest.write_bytes(paths[url.split("/files/")[1].split("/content")[0]].read_bytes())
        return True

    real_fetch = prepare.fetch
    monkeypatch.setattr(prepare, "fetch", fake_fetch)
    result = prepare.download(config)
    assert result["lmdb_counts"] == {"train": 4, "val": 3, "test": 3} == result["index_counts"]
    assert len(fetched) == 4 and all(r["sha256_first_recorded"] for r in result["files"].values())
    assert set(result["lmdb_dirs"]) == {"train", "val", "test"}
    provenance = ds.task_paths(config)["provenance"]
    prepare._append_provenance(provenance, {"step": "download", **{k: v for k, v in result.items()}})
    # a second run keeps the files, re-hashes them and sees the recorded sha256
    monkeypatch.setattr(prepare, "fetch", real_fetch)   # a present file is kept: the network is never touched
    monkeypatch.setattr(prepare.requests, "get", lambda *a, **k: pytest.fail("a present file must not be fetched again"))
    again = prepare.download(config)
    assert not any(r["sha256_first_recorded"] for r in again["files"].values())
    # the recorded sha256 of a file refuses a different one even if size and md5 are re-pinned to it
    runs = json.loads(provenance.read_text())
    runs["runs"][0]["files"]["readme"]["sha256"] = "0" * 64
    provenance.write_text(json.dumps(runs))
    with pytest.raises(RuntimeError, match="recorded by an earlier download"):
        prepare.download(config)


def test_download_refuses_a_wrong_pin_wrong_counts_and_a_mismatching_index_file(tmp_path, monkeypatch):
    config = tiny_config(tmp_path)
    paths = _fixture_source(tmp_path, config)
    monkeypatch.setattr(prepare, "fetch", lambda url, dest: dest.write_bytes(
        paths[url.split("/files/")[1].split("/content")[0]].read_bytes()) > 0)
    bad = json.loads(json.dumps(config))
    bad["sources"][bad["source"]]["files"]["readme"]["md5"] = "0" * 32
    with pytest.raises(RuntimeError, match="md5"):
        prepare.download(bad)
    wrong = tiny_config(tmp_path / "w", counts={"train": 5, "val": 3, "test": 3})
    paths_w = _fixture_source(tmp_path / "w2", wrong)
    monkeypatch.setattr(prepare, "fetch", lambda url, dest: dest.write_bytes(
        paths_w[url.split("/files/")[1].split("/content")[0]].read_bytes()) > 0)
    with pytest.raises(RuntimeError, match="expected_counts"):
        prepare.download(wrong)
    with pytest.raises(RuntimeError, match="index-file counts"):
        prepare.check_counts(config, lmdb_counts={"train": 4, "val": 3, "test": 3}, indices={"train": 9, "val": 3, "test": 3})


def test_validate_converts_a_sample_read_only_and_reports_failures(tmp_path, capsys):
    config = make_data(tmp_path)
    assert prepare.validate(config, sample=2) == 0
    out = capsys.readouterr().out
    assert out.count(": ok ") == 6 and "puzzle='rp01'" in out
    assert not ds.processed_dir(config).exists()
    train = split_items("rp01", 4)
    train[0]["scores"].pop("rms")
    broken = make_data(tmp_path / "b", train=train)
    assert prepare.validate(broken, sample=2) == 1


# ---------------------------------------------------------------------------
# Encoder, probe, ledger
# ---------------------------------------------------------------------------

def test_rna_graphs_survive_the_encoder_and_embed_deterministically(cpu_flash, tmp_path):
    from downstream.embed import embed_split_mind

    config = make_data(tmp_path)
    ds.build_processed(config, splits=["train"], workers=1)
    split = ds.ProcessedSplit(ds.processed_dir(config) / "train.pt")
    torch.manual_seed(0)
    model = PretrainingESAModel(_tiny_model_config())
    path = tmp_path / "tiny.ckpt"
    torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": pl.__version__, "epoch": 0, "global_step": 0}, path)
    loaded = enc.load_mind_encoder(str(path), "cpu", domains=enc.geometry_domains("rna"))
    first = embed_split_mind(loaded, split, "cpu", seed=0, poolings=["mean_atoms", "mean_residues"], max_atoms=250)
    second = embed_split_mind(loaded, split, "cpu", seed=0, poolings=["mean_atoms", "mean_residues"], max_atoms=100)
    for mode in ("mean_atoms", "mean_residues"):
        assert first[mode].shape == (4, 32) and np.isfinite(first[mode]).all()
        np.testing.assert_allclose(first[mode], second[mode], atol=1e-5)   # batch composition does not matter
    assert not np.allclose(first["mean_atoms"][0], first["mean_atoms"][1])
    protein_only = _tiny_model_config()
    protein_only.geometry_neutral_virtual_slots = {"enabled": True, "protein": True, "rna": False}
    with pytest.raises(ValueError, match="for rna"):
        enc._check_config(protein_only, enc.geometry_domains("rna"))
    enc._check_config(protein_only)   # the protein default is unchanged
    assert enc.geometry_domains("protein") == enc.geometry_domains("small_molecule") == ("protein",)


def _write_embeddings(outputs, rng, *, puzzles):
    """Toy embeddings whose first feature carries the RMSD, so a probe can learn the ranking."""
    for split, count in {"train": 60, "val": 24, "test": 24}.items():
        names = [f"{puzzles[split][i % len(puzzles[split])]}/{i:06d}" for i in range(count)]
        labels = rng.gamma(4.0, 3.0, size=(count, 1)).astype(np.float32)
        features = np.concatenate([labels * 0.5 + rng.normal(0, 0.5, (count, 1)), rng.normal(size=(count, 5))], 1)
        path = outputs / "downstream" / "rsr" / "toy" / "embeddings" / f"{split}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, chains=np.array(names), labels=labels, emb_mean_atoms=features.astype(np.float32))
        path.with_suffix(".json").write_text(json.dumps({"split": split}))


def test_probe_trains_on_rsr_embeddings_and_writes_ranking_metrics(tmp_path):
    outputs = tmp_path / "out"
    _write_embeddings(outputs, np.random.default_rng(0),
                      puzzles={"train": ["rp01", "rp02", "rp03"], "val": ["rp14", "rp15"], "test": ["rp18", "rp19"]})
    args = ["--task", "rsr", "--model-id", "toy", "--head", "linear", "--seeds", "0", "--pooling", "mean_atoms",
            "--data-root", str(tmp_path / "data"), "--outputs-root", str(outputs)]
    assert probe.main(args) == 0
    directory = outputs / "downstream" / "rsr" / "toy" / "linear_probe.mean_atoms" / "seed0"
    metrics = json.loads((directory / "metrics.json").read_text())
    assert metrics["kind"] == "ranking_regression" and set(metrics["test"]) == {"mean_per_puzzle", "global"}
    assert set(metrics["val"]) == {"mean_per_puzzle", "global"}
    assert metrics["test"]["mean_per_puzzle"]["puzzles"] == ["rp18", "rp19"]
    assert metrics["test"]["mean_per_puzzle"]["spearman_rho"] > 0.5, "the toy signal must be learned"
    assert metrics["test"]["global"]["rmse"] > 0 and metrics["target_standardization"]["primary_metric"] is None
    with np.load(directory / "predictions.npz") as handle:
        saved = {key: handle[key] for key in handle.files}
    assert saved["test_chains"][0].startswith("rp18/") and saved["test_scores"].shape == (24,)
    # the saved test scores reproduce the metric file (original RMSD units, not standardized ones)
    recomputed = M.evaluate_ranking_regression(saved["test_labels"].reshape(-1), saved["test_scores"],
                                               ds.group_labels(saved["test_chains"]))
    assert recomputed["global"]["spearman_rho"] == pytest.approx(metrics["test"]["global"]["spearman_rho"], abs=1e-9)
    assert recomputed["mean_per_puzzle"]["top1_loss"] == pytest.approx(metrics["test"]["mean_per_puzzle"]["top1_loss"])


def test_probe_stops_when_an_id_carries_no_puzzle(tmp_path):
    outputs = tmp_path / "out"
    _write_embeddings(outputs, np.random.default_rng(0), puzzles={"train": ["rp01"], "val": ["rp14"], "test": ["rp18"]})
    path = outputs / "downstream" / "rsr" / "toy" / "embeddings" / "test.npz"
    with np.load(path) as saved:
        arrays = dict(saved)
    arrays["chains"] = np.array([name.replace("rp18/", "rp18_") for name in arrays["chains"]])
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="not '<puzzle>"):
        probe.main(["--task", "rsr", "--model-id", "toy", "--head", "linear", "--seeds", "0", "--pooling", "mean_atoms",
                    "--data-root", str(tmp_path / "data"), "--outputs-root", str(outputs)])


def test_evaluate_task_refuses_groups_for_other_kinds_and_missing_groups():
    labels = np.arange(4, dtype=np.float32).reshape(-1, 1)
    with pytest.raises(ValueError, match="ranking_regression tasks only"):
        probe.evaluate_task("regression", {}, train_labels=labels, val_labels=labels, val_scores=labels[:, 0],
                            test_labels={"test": labels}, test_scores={"test": labels[:, 0]}, subsets={},
                            val_groups=np.array(["a"] * 4))
    with pytest.raises(ValueError, match="needs val_groups"):
        probe.evaluate_task("ranking_regression", {}, train_labels=labels, val_labels=labels, val_scores=labels[:, 0],
                            test_labels={"test": labels}, test_scores={"test": labels[:, 0]}, subsets={})


def test_ledger_ingests_ranking_metrics_with_literature_aligned_cells(tmp_path):
    targets, predictions, groups = _random_puzzles(3)
    block = M.evaluate_ranking_regression(targets, predictions, groups)
    outputs = tmp_path / "outputs"
    card(outputs, "model_a", "a" * 64)
    for seed in (0, 1):
        metrics = {"task": "rsr", "kind": "ranking_regression", "model_id": "model_a", "protocol": "mlp_probe",
                   "pooling": "mean_atoms", "seed": seed, "best_epoch": 3, "split_tag": None, "representation": None,
                   "val": block, "test": block,
                   "target_standardization": {"mean": [10.0], "std": [5.0], "primary_metric": None}}
        write_probe(outputs, metrics, splits=("train", "val", "test"))
    root = tmp_path / "results"
    ledger.ingest(root, outputs_root=outputs, command="test")
    rows = [r for r in ledger.read_records(root, "measurements") if r["task"] == "rsr" and r["run_seed"] == 0]
    cells = {(r["eval_split"], r["subset"], r["metric"]): r for r in rows}
    primary = cells[("test", "mean_per_puzzle", "spearman_rho")]
    assert primary["is_primary"] and primary["is_catalog_metric"] and primary["direction"] == "higher"
    assert primary["value"] == pytest.approx(block["mean_per_puzzle"]["spearman_rho"])
    assert primary["n_samples"] == targets.size and primary["context"]["num_puzzles"] == 3
    assert cells[("test", "global", "pearson_r")]["value"] == pytest.approx(block["global"]["pearson_r"])
    assert cells[("val", "global", "spearman_rho")]["value"] == pytest.approx(block["global"]["spearman_rho"])
    assert cells[("test", "mean_per_puzzle", "top1_loss")]["unit"] == "Angstrom"
    assert {metric for (split, subset, metric) in cells if split == "test"} >= {"spearman_rho", "pearson_r", "top1_loss", "rmse"}
    assert not [r for r in rows if r["metric"].startswith("per_puzzle") or r["metric"] == "puzzles"]
    # an undefined correlation keeps its reason
    undefined = M.evaluate_ranking_regression(targets, np.zeros_like(targets), groups)
    write_probe(outputs, {**metrics, "seed": 2, "val": undefined, "test": undefined}, splits=("train", "val", "test"))
    ledger.ingest(root, outputs_root=outputs, command="test")
    row, = [r for r in ledger.read_records(root, "measurements")
            if r["task"] == "rsr" and r["run_seed"] == 2 and r["eval_split"] == "test"
            and (r["subset"], r["metric"]) == ("mean_per_puzzle", "spearman_rho")]
    assert row["value"] is None and "constant_prediction" in row["undefined_reason"]


def test_report_renders_the_ranking_table_with_literature(tmp_path):
    from downstream import report

    targets, predictions, groups = _random_puzzles(4)
    block = M.evaluate_ranking_regression(targets, predictions, groups)
    base = tmp_path / "outputs" / "downstream" / "rsr" / "toy" / "linear_probe.mean_atoms"
    for seed in (0, 1):
        (base / f"seed{seed}").mkdir(parents=True)
        (base / f"seed{seed}" / "metrics.json").write_text(json.dumps({
            "task": "rsr", "kind": "ranking_regression", "model_id": "toy", "protocol": "linear_probe", "seed": seed,
            "val": block, "test": block}))
    rows = report.collect("rsr", tmp_path / "outputs")
    assert rows[0]["kind"] == "ranking_regression" and rows[0]["test_mean_per_puzzle_spearman_rho"]["n"] == 2
    text = report.to_markdown("rsr", rows)
    assert "ATOM3D 3DCNN" in text and "0.264" in text and "0.372" in text and "Rosetta scoring function" in text
    assert "not directly comparable" in text and "top-1 loss" in text


# ---------------------------------------------------------------------------
# Fine-tuning (stand-in encoder of test_downstream_finetune; everything else is production code)
# ---------------------------------------------------------------------------

from test_downstream_finetune import encoder_calls, finetune_args, outputs  # noqa: E402,F401  (pytest fixtures)


def test_finetune_selects_epochs_by_mean_puzzle_spearman_and_writes_ranking_metrics(tmp_path, outputs, encoder_calls):
    from downstream import finetune

    config = make_data(tmp_path, train=split_items("rp01", 6), val=split_items("rp02", 4), test=split_items("rp03", 4))
    config["expected_counts"] = {"train": 6, "val": 4, "test": 4}
    assert ds.build_processed(config, splits=["train", "val", "test"], workers=1) == 0
    out, = finetune.run(finetune_args("rsr", outputs, data_root=tmp_path / "data", epochs=3))
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["kind"] == "ranking_regression" and set(metrics["test"]) == {"mean_per_puzzle", "global"}
    assert metrics["test"]["mean_per_puzzle"]["puzzles"] == ["rp03"] and metrics["test"]["global"]["num_samples"] == 4
    assert metrics["target_standardization"]["std"][0] == pytest.approx(
        np.std([3.0 * (i + 1) for i in range(6)]), rel=1e-5)
    log = (out / "run.log").read_text()
    assert "val_mean_spearman" in log and "lr_encoder 0.0001" in log
    with np.load(out / "predictions.npz") as handle:
        scores, labels = handle["test_scores"], handle["test_labels"].reshape(-1)
    assert scores.shape == (4,) and labels.tolist() == [3.0, 6.0, 9.0, 12.0]
    recomputed = M.evaluate_ranking_regression(labels, scores, np.array(["rp03"] * 4))
    assert recomputed["global"]["rmse"] == pytest.approx(metrics["test"]["global"]["rmse"], rel=1e-6)
    random_out, = finetune.run(finetune_args("rsr", outputs, data_root=tmp_path / "data", epochs=1, random_init=True))
    assert "lr_encoder 0.0003" in (random_out / "run.log").read_text()


def test_ds_run_knows_the_rna_modality_and_the_rsr_unit():
    from downstream import run_all

    units, explicit = run_all.expand_tasks("rsr", ledger.load_catalog())
    assert explicit and [u.key for u in units] == ["rsr"] and "rsr" in run_all.ALL_TASKS
    assert run_all.MODALITY_DOMAINS["rna"] == ("rna",)
    info = run_all.describe_task(units[0])
    assert info["modality"] == "rna" and info["splits"] == ["train", "val", "test"]
    assert info["pooling"] == info["finetune_pooling"] == "mean_atoms" and info["finetune_seconds"] == 90 * 50
    assert {path.parent.name for path in info["processed"].values()} == {"v1"}
    assert info["finetune"]["lr_encoder"] == 1e-4
