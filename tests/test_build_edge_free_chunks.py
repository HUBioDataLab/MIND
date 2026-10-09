"""Tests for the raw-structure -> edge-free chunk builder (split plan and worker)."""

from collections import Counter

import pytest
import torch
from Bio.PDB import MMCIFIO, PDBParser

from data_loading.build_edge_free_chunks import (
    _init_worker,
    assign_split,
    build_node_sample,
    build_plan,
    chunk_directory,
    load_or_write_plan,
    split_group,
)
from data_loading.atom_position_vocab import VIRTUAL_ATOMIC_NUMBER

IDS = [f"A0A{index:06d}" for index in range(3000)]


def _plan(ids=IDS, **overrides):
    values = dict(chunk_size=100, split_salt="salt", fractions=(0.8, 0.1, 0.1), shuffle_seed=7)
    values.update(overrides)
    return build_plan(ids, **values)


def test_split_is_deterministic_and_order_independent():
    first = {row["id"]: row["split"] for row in _plan()}
    second = {row["id"]: row["split"] for row in _plan(ids=list(reversed(IDS)))}
    assert first == second
    assert all(first[i] == assign_split(i, "salt", (0.8, 0.1, 0.1)) for i in IDS)
    counts = Counter(first.values())
    assert 0.75 < counts["train"] / len(IDS) < 0.85
    assert counts["val"] > 0 and counts["test"] > 0


def test_every_id_is_planned_once_and_chunks_are_full():
    rows = _plan()
    assert sorted(row["id"] for row in rows) == sorted(IDS)
    per_chunk = Counter((row["split"], row["chunk"]) for row in rows)
    for split in ("train", "val", "test"):
        sizes = [per_chunk[key] for key in sorted(per_chunk) if key[0] == split]
        assert all(size == 100 for size in sizes[:-1]) and 0 < sizes[-1] <= 100


def test_chunks_are_shuffled_within_split_not_manifest_order():
    rows = [row for row in _plan() if row["split"] == "train" and row["chunk"] == 0]
    train_ids = sorted(row["id"] for row in _plan() if row["split"] == "train")
    assert [row["id"] for row in rows] != train_ids[:100]
    assert _plan(shuffle_seed=8) != _plan()


def test_pdb_entry_grouping_keeps_chains_together():
    ids = [f"{entry}_{chain}" for entry in ("1abc", "2xyz", "3qqq", "4rrr") for chain in "ABCD"]
    rows = build_plan(ids, chunk_size=5, split_salt="s", fractions=(0.5, 0.25, 0.25),
                      shuffle_seed=1, group_by="pdb-entry")
    by_entry = {}
    for row in rows:
        by_entry.setdefault(split_group(row["id"], "pdb-entry"), set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_entry.values())


def test_no_split_keeps_manifest_order():
    rows = _plan(split_salt=None)
    assert [row["id"] for row in rows] == IDS
    assert {row["split"] for row in rows} == {""}
    assert [row["chunk"] for row in rows[:101:100]] == [0, 1]


def test_plan_file_must_match_on_restart(tmp_path):
    path = tmp_path / "plan.csv"
    load_or_write_plan(path, _plan())
    load_or_write_plan(path, _plan())
    with pytest.raises(ValueError, match="differs"):
        load_or_write_plan(path, _plan(shuffle_seed=99))


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        build_plan(["A", "A"], chunk_size=2, split_salt=None, fractions=(1, 0, 0), shuffle_seed=0)


def test_chunk_directory_names_are_zero_padded(tmp_path):
    assert chunk_directory(tmp_path / "p", "val", 7).name == "p_val_chunk_007"
    assert chunk_directory(tmp_path / "p", "", 12).name == "p_chunk_012"


# Three residues (ALA, SER, GLY) with ideal-ish backbone geometry.
_ATOMS = [
    ("ALA", 1, "N", "N", (0.000, 1.430, 0.000)), ("ALA", 1, "CA", "C", (0.000, 0.000, 0.000)),
    ("ALA", 1, "C", "C", (1.520, 0.000, 0.000)), ("ALA", 1, "O", "O", (2.150, 1.050, 0.000)),
    ("ALA", 1, "CB", "C", (-0.540, -0.770, 1.200)),
    ("SER", 2, "N", "N", (2.150, -1.180, 0.000)), ("SER", 2, "CA", "C", (3.600, -1.300, 0.000)),
    ("SER", 2, "C", "C", (4.100, -2.740, 0.000)), ("SER", 2, "O", "O", (3.300, -3.680, 0.000)),
    ("SER", 2, "CB", "C", (4.100, -0.540, 1.220)), ("SER", 2, "OG", "O", (5.500, -0.500, 1.300)),
    ("GLY", 3, "N", "N", (5.420, -2.900, 0.000)), ("GLY", 3, "CA", "C", (6.000, -4.240, 0.000)),
    ("GLY", 3, "C", "C", (7.520, -4.200, 0.000)), ("GLY", 3, "O", "O", (8.150, -3.150, 0.000)),
]


def _write_pdb(path):
    lines = []
    for serial, (res, seq, name, element, (x, y, z)) in enumerate(_ATOMS, start=1):
        lines.append(
            f"ATOM  {serial:5d} {name:<4s} {res:3s} A{seq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00          {element:>2s}"
        )
    path.write_text("\n".join(lines + ["END"]) + "\n")


@pytest.fixture(scope="module")
def raw_structures(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    pdb = root / "TESTPROT.pdb"
    _write_pdb(pdb)
    cif = root / "TESTCIF.cif"
    io = MMCIFIO()
    io.set_structure(PDBParser(QUIET=True).get_structure("TESTCIF", str(pdb)))
    io.save(str(cif))
    _init_worker("protein")
    return pdb, cif


def test_worker_builds_fixed_slot_protein_from_pdb(raw_structures):
    pdb, _ = raw_structures
    structure_id, sample, reason = build_node_sample(str(pdb))
    assert (structure_id, reason) == ("TESTPROT", None)
    assert sample.num_nodes == 3 * 14
    assert sample.block_symbols == ["ALA", "SER", "GLY"]
    assert sample.block_idx.tolist() == [i // 14 for i in range(42)]
    assert sample.pos_code[:5] == ["N", "CA", "C", "O", "CB"]
    real = sample.z != VIRTUAL_ATOMIC_NUMBER
    assert int(real.sum()) == len(_ATOMS)
    assert not hasattr(sample, "edge_index") or sample.edge_index is None
    assert torch.isfinite(sample.pos).all()


def test_worker_reads_cif_like_pdb(raw_structures):
    pdb, cif = raw_structures
    _, from_pdb, _ = build_node_sample(str(pdb))
    structure_id, from_cif, reason = build_node_sample(str(cif))
    assert (structure_id, reason) == ("TESTCIF", None)
    for name in ("z", "pos", "block_idx", "entity_idx"):
        assert torch.equal(getattr(from_pdb, name), getattr(from_cif, name)), name
    assert from_pdb.pos_code == from_cif.pos_code


def test_worker_reports_unreadable_file(tmp_path, raw_structures):
    broken = tmp_path / "BROKEN.pdb"
    broken.write_text("not a structure\n")
    structure_id, sample, reason = build_node_sample(str(broken))
    assert structure_id == "BROKEN" and sample is None and reason
