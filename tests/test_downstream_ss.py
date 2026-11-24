"""Secondary-structure tasks (ss_q3, ss_q8): label definitions, structure mapping with counted losses, the TAPE LMDB reader,
the resumable structure fetch, processing on a synthetic fixture, per-residue probe / fine-tune on CPU, metrics, ledger.

Synthetic inputs only: three chains (a mismatch, a residue missing from the structure, unresolved termini, a residue missing
a backbone atom, an incomplete sidechain) plus a chain whose structure cannot be found; no network, no real data.
"""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, f1_score

from downstream import finetune, ledger, probe
from downstream.common import encoder as mind_encoder
from downstream.common.metrics import evaluate_multiclass, macro_f1
from downstream.common.prepare_spec import DownloadSpec
from downstream.common.registry import embeddings_path
from downstream.tasks import TASKS, load_task
from downstream.tasks.ss import dataset, mapping as M, prepare, structures, tape_lmdb
from downstream_helpers import chain_of, cif_text
from test_downstream_finetune import encoder_calls, finetune_args, outputs, stand_in_encode  # noqa: F401

SPLITS = ("train", "valid", "cb513", "ts115", "casp12")
THREE = {"A": "ALA", "S": "SER", "G": "GLY", "L": "LEU", "K": "LYS", "V": "VAL", "E": "GLU", "F": "PHE", "T": "THR"}


def record(chain_id, sequence, ss8, observed=None):
    ss8 = np.asarray(ss8, dtype=np.int64)
    return {"id": chain_id, "primary": sequence, "ss8": ss8, "ss3": M.SS8_INDEX_TO_SS3_INDEX[ss8],
            "observed": np.ones(len(sequence), dtype=np.int64) if observed is None else np.asarray(observed, dtype=np.int64)}


def fixture_records():
    """1ABC-A: 2 unresolved N-terminal residues, a missing middle residue, one mismatch. 2DEF-A: complete structure with a
    backbone-less residue and an incomplete sidechain. 3XYZ-A: no structure file."""
    rng = np.random.default_rng(0)
    seq_a = "GSAKLVESTAKLFGVTEALK"                 # dataset residues; structure lacks the first 2 and the 7th (E)
    obs_a = [0, 0] + [1] * 18
    seq_b = "ALKVEFSTGA"
    seq_c = "AKLVE"
    return [record("1ABC-A", seq_a, rng.integers(0, 8, len(seq_a)), obs_a),
            record("2DEF-A", seq_b, rng.integers(0, 8, len(seq_b))),
            record("3XYZ-A", seq_c, rng.integers(0, 8, len(seq_c)))]


def structure_files(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    # 1ABC: structure residues = dataset[2:6] + dataset[7:] (17 residues), with the residue at dataset 3 changed (mismatch: K -> T)
    seq_a = "GSAKLVESTAKLFGVTEALK"
    kept = [seq_a[i] for i in range(2, 20) if i != 6]
    kept[1] = "T"                                   # dataset has K
    drops = {}
    res_b = list("ALKVEFSTGA")
    drops_b = {2: ("N",), 4: ("CD",)}               # residue 2 lacks N (dropped), residue 4 (GLU) lacks CD (masked)
    for pdb, chains in (("1ABC", {"A": chain_of([THREE[c] for c in kept], drops=drops)}),
                        ("2DEF", {"A": chain_of([THREE[c] for c in res_b], drops=drops_b)})):
        with gzip.open(directory / f"{pdb}.cif.gz", "wt") as handle:
            handle.write(cif_text(pdb, chains))


def write_task_data(root: Path) -> dict:
    config = dataset.load_task_config(dataset.TASK_YAML, data_root=root)
    paths = dataset.task_paths(config)
    paths["records"].mkdir(parents=True)
    paths["splits"].mkdir(parents=True)
    for split in SPLITS:
        records = fixture_records()
        dataset.write_records(paths["records"] / f"{split}.jsonl.gz", records)
        (paths["splits"] / f"{split}.txt").write_text("\n".join(r["id"] for r in records) + "\n")
    structure_files(paths["structures"])
    return config


@pytest.fixture(scope="module")
def processed_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("ss_data")
    config = write_task_data(root)
    assert dataset.build_processed(config, splits=SPLITS, workers=1, max_failed_fraction=0.5) == 0
    return root


# ---------------------------------------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------------------------------------

def test_label_definitions_follow_the_verified_tape_layout():
    assert M.SS8_CLASSES == "GHIBESTL" and M.SS3_CLASSES == "HEC"
    assert M.SS8_INDEX_TO_SS3_INDEX.tolist() == [0, 0, 0, 1, 1, 2, 2, 2]   # G,H,I -> H; B,E -> E; S,T,L -> C
    assert M.COIL_INDEX == {"ss3": 2, "ss8": 7}


def test_inconsistent_ss3_and_ss8_raise():
    good = record("1ABC-A", "AAA", [0, 4, 7])
    M.check_label_arrays("1ABC-A", "AAA", good["ss3"], good["ss8"], good["observed"])
    with pytest.raises(ValueError, match="standard collapse"):
        M.check_label_arrays("1ABC-A", "AAA", np.array([0, 0, 2]), good["ss8"], good["observed"])
    with pytest.raises(ValueError, match="lengths"):
        M.check_label_arrays("1ABC-A", "AA", good["ss3"], good["ss8"], good["observed"])


# ---------------------------------------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------------------------------------

def test_alignment_labels_residues_and_counts_every_reason():
    data = "GSAKLVESTAKL"
    structure = "AKLVSTAKL"           # lacks GS (termini) and E
    ss = {"ss8": np.arange(12) % 8, "ss3": M.SS8_INDEX_TO_SS3_INDEX[np.arange(12) % 8]}
    result = M.map_labels(structure, data, ss)
    assert result.pair.tolist() == [2, 3, 4, 5, 7, 8, 9, 10, 11]
    assert result.identical == 9 and result.mismatched == 0 and result.identity == 1.0
    assert result.labels["ss8"].tolist() == [2, 3, 4, 5, 7, 0, 1, 2, 3]
    changed = M.map_labels("AKLXSTAKL", data, ss)
    assert changed.structure_reasons[3] == "mismatch_other" and changed.labels["ss8"][3] == -1
    xdata = M.map_labels("AKLVSTAKL", "GSAKLXESTAKL", ss)
    assert xdata.structure_reasons[3] == "mismatch_dataset_x" and xdata.labels["ss8"][3] == -1
    semet = M.map_labels("AKLVSTAKL", "GSAKLXESTAKL", ss, [False, False, False, True] + [False] * 5)
    assert semet.structure_reasons[3] == "ok_selenomethionine" and semet.labels["ss8"][3] == 5 and semet.mismatched == 0


def test_a_region_of_a_longer_chain_aligns_completely_and_a_random_pair_does_not():
    chain = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    region = chain[10:24]
    ss = {"ss8": np.arange(len(region)) % 8}
    result = M.map_labels(chain, region, ss)
    assert result.identity == 1.0 and result.aligned_fraction(len(region)) == 1.0
    assert (result.labels["ss8"] >= 0).sum() == len(region) and result.structure_reasons.count("no_dataset_counterpart") == len(chain) - len(region)
    unrelated = M.map_labels(chain, "WWWPPPCCCHHHNN", ss)
    assert unrelated.identity < 0.5


def test_residue_buckets_add_up_or_raise():
    data, structure = "GSAKLVESTAKL", "AKLVSTAKL"
    ss = {"ss8": np.arange(12) % 8}
    mapping = M.map_labels(structure, data, ss)
    observed = np.array([0, 0] + [1] * 10)
    kept = [True] * 8 + [False]
    buckets = M.dataset_residue_buckets(12, observed, mapping, kept, structure, data)
    assert dict(buckets) == {M.EVALUATED: 8, "dropped_missing_backbone": 1, "not_in_structure_flagged_unresolved": 2,
                             "not_in_structure_unflagged": 1}
    assert sum(M.account_residues("c", 12, buckets).values()) == 12
    with pytest.raises(ValueError, match="sum to"):
        M.account_residues("c", 13, buckets)
    with pytest.raises(ValueError, match="unknown residue reasons"):
        M.account_residues("c", 1, M.Counter({"vanished": 1}))


# ---------------------------------------------------------------------------------------------------------
# TAPE LMDB reader (no pickled code is executed)
# ---------------------------------------------------------------------------------------------------------

def tape_record(chain_id, sequence):
    n = len(sequence)
    ss8 = (np.arange(n) % 8).astype("<i4")
    return {"id": chain_id.encode(), "primary": sequence, "ss3": M.SS8_INDEX_TO_SS3_INDEX[ss8].astype("<i4"), "ss8": ss8,
            "valid_mask": np.ones(n, "<f4"), "disorder": np.ones(n, "<i4"), "protein_length": n,
            "rsa": np.zeros(n, "<f4")}


def write_lmdb(path: Path, records, *, raw=None):
    lmdb = pytest.importorskip("lmdb")
    env = lmdb.open(str(path), map_size=1 << 24)
    with env.begin(write=True) as txn:
        txn.put(b"num_examples", pickle.dumps(len(records)))
        for index, value in enumerate(raw if raw is not None else [pickle.dumps(r) for r in records]):
            txn.put(str(index).encode(), value)
    env.close()


def test_tape_lmdb_reader_normalizes_records(tmp_path):
    write_lmdb(tmp_path / "ss.lmdb", [tape_record("1g3p-A", "ACDEFGHIK"), tape_record("3oe3-A", "KLMNP")])
    rows = list(tape_lmdb.read_lmdb(tmp_path / "ss.lmdb"))
    assert [r["id"] for r in rows] == ["1g3p-A", "3oe3-A"] and rows[0]["primary"] == "ACDEFGHIK"
    assert rows[0]["ss8"].dtype == np.int64 and rows[0]["observed"].tolist() == [1] * 9


def test_tape_lmdb_reader_refuses_code_in_a_record(tmp_path):
    class Evil:
        def __reduce__(self):
            return (eval, ("1+1",))

    write_lmdb(tmp_path / "ss.lmdb", [None], raw=[pickle.dumps({"id": b"x", "payload": Evil()})])
    with pytest.raises(pickle.UnpicklingError, match="not allowed"):
        list(tape_lmdb.read_lmdb(tmp_path / "ss.lmdb"))


def test_tape_lmdb_reader_rejects_missing_fields_and_records(tmp_path):
    broken = tape_record("1g3p-A", "ACD")
    del broken["ss8"]
    write_lmdb(tmp_path / "a.lmdb", [broken])
    with pytest.raises(ValueError, match="lacks"):
        list(tape_lmdb.read_lmdb(tmp_path / "a.lmdb"))
    write_lmdb(tmp_path / "b.lmdb", [tape_record("1g3p-A", "ACD")], raw=[])
    with pytest.raises(ValueError, match="missing"):
        list(tape_lmdb.read_lmdb(tmp_path / "b.lmdb"))


# ---------------------------------------------------------------------------------------------------------
# Structure fetch
# ---------------------------------------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status, body=b""):
        self.status_code, self._body = status, body
        self.headers = {"Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT", "ETag": '"abc"'}

    def iter_content(self, size):
        yield self._body


class FakeSession:
    def __init__(self, behaviour):
        self.behaviour, self.calls = behaviour, []

    def get(self, url, **kwargs):
        self.calls.append(url)
        return self.behaviour(url, len(self.calls))


def good_body():
    return gzip.compress(b"data_1ABC\nloop_\n")


def test_fetch_is_resumable_records_provenance_and_counts_not_found(tmp_path):
    def behaviour(url, call):
        return FakeResponse(404) if "9ZZZ" in url else FakeResponse(200, good_body())

    kwargs = dict(url_template="https://x/{pdb_id}.cif.gz", rate=1000, workers=1, sleep=lambda s: None)
    session = FakeSession(behaviour)
    counts = structures.fetch_structures(["1abc", "2def", "9zzz", "1ABC"], tmp_path, session=session, **kwargs)
    assert counts == {"ok": 2, "not_found": 1, "error": 0, "already_present": 0} and len(session.calls) == 3
    manifest = structures.read_manifest(tmp_path)
    assert manifest["1ABC"]["sha256"] == __import__("hashlib").sha256(good_body()).hexdigest()
    assert manifest["1ABC"]["etag"] == '"abc"' and manifest["9ZZZ"]["status"] == "not_found"
    again = FakeSession(behaviour)
    counts = structures.fetch_structures(["1ABC", "2DEF", "9ZZZ", "3GHI"], tmp_path, session=again, **kwargs)
    assert counts == {"ok": 1, "not_found": 1, "error": 0, "already_present": 2} and len(again.calls) == 1


def test_fetch_retries_then_records_an_error_and_refuses_untracked_files(tmp_path):
    session = FakeSession(lambda url, call: FakeResponse(503))
    counts = structures.fetch_structures(["1ABC"], tmp_path, url_template="https://x/{pdb_id}.cif.gz", rate=1000,
                                         workers=1, retries=2, session=session, sleep=lambda s: None)
    assert counts["error"] == 1 and len(session.calls) == 3
    assert "HTTP 503" in structures.read_manifest(tmp_path)["1ABC"]["error"]
    (tmp_path / "2DEF.cif.gz").write_bytes(good_body())
    with pytest.raises(RuntimeError, match="no matching ok line"):
        structures.fetch_structures(["2DEF"], tmp_path, url_template="https://x/{pdb_id}.cif.gz", session=session)


def test_fetch_rejects_a_body_that_is_not_mmcif(tmp_path):
    session = FakeSession(lambda url, call: FakeResponse(200, gzip.compress(b"<html>rate limited</html>")))
    counts = structures.fetch_structures(["1ABC"], tmp_path, url_template="https://x/{pdb_id}.cif.gz", rate=1000,
                                         workers=1, retries=0, session=session, sleep=lambda s: None)
    assert counts["error"] == 1 and not (tmp_path / "1ABC.cif.gz").exists()


def test_rate_limiter_spaces_requests():
    now, waits = [0.0], []
    limiter = structures.RateLimiter(2.0, clock=lambda: now[0], sleep=waits.append)
    for _ in range(3):
        limiter.acquire()
    assert waits == pytest.approx([0.5, 1.0])


# ---------------------------------------------------------------------------------------------------------
# Processing on the synthetic fixture
# ---------------------------------------------------------------------------------------------------------

def test_processing_counts_every_loss_and_labels_the_structure_residues(processed_root):
    config = dataset.load_task_config(dataset.TASK_YAML, data_root=processed_root)
    qc = json.loads((dataset.processed_dir(config) / "qc_cb513.json").read_text())
    assert qc["status"] == {"ok": 2, "failed": 1} and qc["reasons"] == {"structure_missing": 1}
    coverage = qc["coverage"]
    # 1ABC-A: 20 dataset residues = 16 evaluated (17 structure residues, 1 mismatch), 2 flagged unresolved, 1 unflagged
    # missing (E), 1 mismatch.  2DEF-A: 10 residues, residue 2 lacks N -> 9 evaluated.  3XYZ-A: 5 residues, not processed.
    assert coverage["residue_buckets"] == {
        "evaluated": 16 + 9, "not_in_structure_flagged_unresolved": 2, "not_in_structure_unflagged": 1,
        "sequence_mismatch": 1, "dropped_missing_backbone": 1, "chain_not_processed": 5}
    assert coverage["literature_residues"] == 20 + 10 + 5 == sum(coverage["residue_buckets"].values())
    assert coverage["evaluated_residues"] == 25 and coverage["evaluated_chains"] == 2 and coverage["literature_chains"] == 3
    assert coverage["residue_coverage"] == pytest.approx(25 / 35) and coverage["masked_residues"] == 1
    split = load_task("ss_q3").ProcessedSplit(dataset.processed_dir(config) / "cb513.pt")
    assert split.chains == ["1ABC-A", "2DEF-A"] and len(split.labels) == 25 == len(split.row_chains)
    assert [int((labels >= 0).sum()) for labels in split.sample_labels] == [16, 9]
    assert split.sample_labels[0].tolist()[1] == -1          # the mismatching residue is kept as context, unlabelled
    assert len(split.sample_labels[1]) == 9 and bool((split.sample_labels[1] >= 0).all())   # the backbone-less residue is gone
    ss8 = load_task("ss_q8").ProcessedSplit(dataset.processed_dir(config) / "cb513.pt")
    assert ss8.classes == list("GHIBESTL") and split.classes == list("HEC")
    assert np.array_equal(M.SS8_INDEX_TO_SS3_INDEX[ss8.labels], split.labels)
    tsv = (dataset.processed_dir(config) / "qc_cb513.tsv").read_text().splitlines()
    assert any(line.startswith("3XYZ-A\tcb513\tfailed\tstructure_missing") for line in tsv)


def test_a_split_above_the_failure_limit_is_not_written(tmp_path):
    config = write_task_data(tmp_path)
    assert dataset.build_processed(config, splits=["cb513"], workers=1) == 1   # 1 of 3 chains failed > 0.25
    assert not (dataset.processed_dir(config) / "cb513.pt").exists()
    assert (dataset.processed_dir(config) / "qc_cb513.json").exists()


def test_unparsable_ids_are_failed_chains_not_dropped(tmp_path):
    config = write_task_data(tmp_path)
    paths = dataset.task_paths(config)
    records = fixture_records()
    records[2]["id"] = "CB513_0007"
    path = paths["records"] / "casp12.jsonl.gz"
    path.unlink()
    dataset.write_records(path, records)
    (paths["splits"] / "casp12.txt").write_text("\n".join(r["id"] for r in records) + "\n")
    assert dataset.build_processed(config, splits=["casp12"], workers=1, max_failed_fraction=0.5) == 0
    qc = json.loads((dataset.processed_dir(config) / "qc_casp12.json").read_text())
    assert qc["reasons"] == {"id_unparsable": 1} and qc["coverage"]["residue_buckets"]["chain_not_processed"] == 5


def test_existing_processed_file_is_never_overwritten(processed_root):
    config = dataset.load_task_config(dataset.TASK_YAML, data_root=processed_root)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        dataset.build_processed(config, splits=["cb513"], workers=1, max_failed_fraction=0.5)


def test_config_refuses_a_skip_length_policy(tmp_path):
    text = dataset.TASK_YAML.read_text().replace("length_policy: crop_windows", "length_policy: skip")
    path = tmp_path / "task.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match="crop_windows"):
        dataset.load_task_config(path)


def test_test_block_carries_its_coverage_and_the_coil_fallback(processed_root):
    config = dataset.load_task_config(dataset.TASK_YAML, data_root=processed_root)
    split = load_task("ss_q3").ProcessedSplit(dataset.processed_dir(config) / "cb513.pt")
    scores = np.eye(3)[split.labels]
    block = evaluate_multiclass(split.labels, scores)
    assert block["accuracy"] == 1.0
    load_task("ss_q3").augment_test_metrics({"cb513": block}, {"cb513": split.coverage})
    assert block["num_residues"] == 25 and "num_proteins" not in block
    assert block["residue_coverage"] == pytest.approx(25 / 35) and block["literature_residues"] == 35
    lost_coil = split.coverage["lost_label_counts"]["ss3"][2]
    assert block["accuracy_all_residues_coil_fallback"] == pytest.approx((25 + lost_coil) / 35)
    text = dataset.coverage_text({"cb513": split.coverage})
    assert "SUBSET" in text and "25/35" in text
    with pytest.raises(ValueError, match="scored residues"):
        load_task("ss_q3").augment_test_metrics({"cb513": {"num_proteins": 5, "accuracy": 1.0}}, {"cb513": split.coverage})


# ---------------------------------------------------------------------------------------------------------
# Residue states, metrics
# ---------------------------------------------------------------------------------------------------------

def test_residue_states_pick_the_ca_node_of_every_residue_and_flag_masked_blocks(processed_root, monkeypatch):
    from test_downstream_molnet import tiny_model_config
    config = dataset.load_task_config(dataset.TASK_YAML, data_root=processed_root)
    split = load_task("ss_q3").ProcessedSplit(dataset.processed_dir(config) / "cb513.pt")
    samples = [split.runtime(i, tiny_model_config(), seed=0) for i in range(len(split))]
    batch = mind_encoder.collate(samples)
    batch.active_node_mask = torch.ones(batch.num_nodes, dtype=torch.bool)
    nodes = torch.arange(batch.num_nodes, dtype=torch.float32).unsqueeze(-1).repeat(1, 4)
    states, graph, masked = mind_encoder.residue_states(nodes, batch, "ca")
    ca = torch.where(batch.template_anchor_order.long() == 1)[0]
    assert states.shape == (len(ca), 4) and torch.equal(states[:, 0], ca.float())
    assert graph.tolist() == [0] * 17 + [1] * 9 and int(masked.sum()) == 1   # one incomplete sidechain (GLU of 2DEF-A)
    mean, _, _ = mind_encoder.residue_states(nodes, batch, "mean_backbone")
    anchors = torch.where(batch.template_anchor_order.long() >= 0)[0].reshape(-1, 3)
    assert torch.allclose(mean[:, 0], anchors.float().mean(1))
    with pytest.raises(ValueError, match="unknown residue mode"):
        mind_encoder.residue_states(nodes, batch, "cb")
    batch.active_node_mask[ca[0]] = False
    with pytest.raises(ValueError, match="expected 1 active anchor"):
        mind_encoder.residue_states(nodes, batch, "ca")


def test_accuracy_and_macro_f1_match_scikit_learn():
    rng = np.random.default_rng(3)
    for classes in (3, 8):
        targets = rng.integers(0, classes, 500)
        scores = rng.random((500, classes))
        scores[::7, 0] += 2.0
        predicted = scores.argmax(1)
        block = evaluate_multiclass(targets, scores)
        assert block["accuracy"] == pytest.approx(accuracy_score(targets, predicted))
        assert block["macro_f1"] == pytest.approx(f1_score(targets, predicted, average="macro"))
        assert macro_f1(targets, scores) == block["macro_f1"]
    absent = np.array([0, 0, 1, 1])
    only_two = np.eye(8)[[0, 1, 1, 1]]          # classes 2..7 occur nowhere: they must not enter the mean
    assert macro_f1(absent, only_two) == pytest.approx(f1_score(absent, only_two.argmax(1), average="macro"))


# ---------------------------------------------------------------------------------------------------------
# Probe and fine-tune on CPU
# ---------------------------------------------------------------------------------------------------------

def synthetic_embeddings(root: Path, task: str, model_id: str, label_set: str, *, rows=60, dim=8):
    rng = np.random.default_rng(7)
    classes = len(M.LABEL_SETS[label_set])
    for split in SPLITS:
        labels = np.arange(rows) % classes
        features = rng.normal(0, 0.3, (rows, dim)).astype(np.float32)
        features[np.arange(rows), labels % dim] += 3.0           # separable
        coverage = {"literature_chains": 4, "evaluated_chains": 3, "chain_coverage": 0.75, "literature_residues": rows + 10,
                    "evaluated_residues": rows, "residue_coverage": rows / (rows + 10),
                    "residue_buckets": {}, "lost_label_counts": {"ss3": [0, 0, 10], "ss8": [0] * 7 + [10]},
                    "masked_residues": 0}
        path = embeddings_path(task, model_id, split, root=root)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, chains=np.array([f"c{i // 10}" for i in range(rows)]), labels=labels, emb_ca=features)
        path.with_suffix(".json").write_text(json.dumps({"coverage": coverage, "split": split}))


@pytest.mark.parametrize("task,label_set", [("ss_q3", "ss3"), ("ss_q8", "ss8")])
def test_probe_trains_per_residue_and_marks_the_subset(tmp_path, task, label_set):
    synthetic_embeddings(tmp_path, task, "m", label_set)
    assert probe.main(["--task", task, "--model-id", "m", "--head", "linear", "--seeds", "0", "--outputs-root", str(tmp_path)]) == 0
    seed_dir = next((tmp_path / "downstream" / task / "m").glob("linear_probe.ca/seed0"))
    metrics = json.loads((seed_dir / "metrics.json").read_text())
    assert list(metrics["test"]) == ["cb513", "ts115", "casp12"]
    block = metrics["test"]["cb513"]
    assert 0.0 <= block["accuracy"] <= 1.0 and block["num_residues"] == 60 and block["literature_residues"] == 70
    assert block["residue_coverage"] == pytest.approx(60 / 70) and "macro_f1" in block
    coil = 10   # synthetic coverage: every one of the 10 lost residues is coil
    assert block["accuracy_all_residues_coil_fallback"] == pytest.approx((block["accuracy"] * 60 + coil) / 70)
    assert json.loads((seed_dir / "coverage.json").read_text())["cb513"]["evaluated_residues"] == 60


def test_ledger_ingests_the_per_residue_blocks(tmp_path):
    synthetic_embeddings(tmp_path, "ss_q3", "m", "ss3")
    probe.main(["--task", "ss_q3", "--model-id", "m", "--head", "linear", "--seeds", "0", "--outputs-root", str(tmp_path)])
    metrics = json.loads(next((tmp_path / "downstream" / "ss_q3").rglob("metrics.json")).read_text())
    blocks = dict((name, block) for name, _, block in ledger._eval_blocks(metrics, "multiclass", "t"))
    rows, samples, context = ledger._block_rows(blocks["cb513"], "multiclass", "cb513")
    names = {metric for metric, value, reason in rows}
    assert {"accuracy", "macro_f1", "residue_coverage", "chain_coverage", "accuracy_all_residues_coil_fallback"} <= names
    assert samples == 60 and context["literature_residues"] == 70
    catalog = ledger.load_catalog()
    for task in ("ss_q3", "ss_q8"):
        entry = catalog["tasks"][task]
        assert entry["kind"] == "multiclass" and entry["primary"]["eval_split"] == "cb513"
        assert {c["eval_split"] for c in entry["required"]} == {"cb513", "ts115", "casp12"}


@pytest.mark.parametrize("task", ["ss_q3", "ss_q8"])
def test_finetune_runs_per_residue_on_the_synthetic_splits(monkeypatch, tmp_path, outputs, encoder_calls, processed_root, task):
    """The stand-in encoder replaces only the ESA forward; graphs, residue states, loss, selection and metrics are production code."""
    args = finetune_args(task, outputs, data_root=processed_root, epochs=2)
    args.device = "cpu"
    directories = finetune.run(args)
    metrics = json.loads((directories[0] / "metrics.json").read_text())
    assert metrics["kind"] == "multiclass" and metrics["protocol"] == "finetune"
    block = metrics["test"]["cb513"]
    assert block["num_residues"] == 25 and block["residue_coverage"] == pytest.approx(25 / 35)
    assert 0.0 <= block["accuracy"] <= 1.0 and "macro_f1" in block
    predictions = np.load(directories[0] / "predictions.npz")
    assert predictions["test_scores"].shape == (25, 3 if task == "ss_q3" else 8)
    assert predictions["test_chains"].tolist().count("1ABC-A") == 16
    assert (directories[0] / "coverage.json").exists()


# ---------------------------------------------------------------------------------------------------------
# Registry, prepare hooks
# ---------------------------------------------------------------------------------------------------------

def test_registry_and_hooks():
    assert "ss_q3" in TASKS and "ss_q8" in TASKS
    for name in ("ss_q3", "ss_q8"):
        module = load_task(name)
        assert module.KIND == "multiclass" and module.PER_RESIDUE
    q3, q8 = (load_task(n).load_task_config(load_task(n).TASK_YAML) for n in ("ss_q3", "ss_q8"))
    assert (q3["label_count"], q8["label_count"]) == (3, 8) and load_task("ss_q3").processed_dir(q3) == load_task("ss_q8").processed_dir(q8)
    spec = prepare.download_spec(q3)
    assert isinstance(spec, DownloadSpec) and spec.archives[0].bytes == 251794897 and spec.archives[0].md5
    assert spec.archives[0].url.endswith("secondary_structure.tar.gz")
    assert any(o.name == ".complete" for o in spec.outputs) and spec.extracted_gib == 2.0
    assert prepare.cli_args("process", task="ss_q3", workers=4, threads=2) == ["--workers", "4"]
    assert prepare.cli_args("download", task="ss_q3", workers=4, threads=2) == []


def test_repeated_chain_ids_of_a_split_are_renamed_and_counted(tmp_path):
    pytest.importorskip("lmdb")
    config = dataset.load_task_config(dataset.TASK_YAML, data_root=tmp_path)
    directory = dataset.task_paths(config)["raw"] / "secondary_structure"
    directory.mkdir(parents=True)
    config["expected_chain_counts"] = {split: 2 for split in SPLITS}
    for split in SPLITS:   # distinct ids per split, except that the first chain of cb513 occurs twice (two regions)
        ids = ["1abc-A", "1abc-A"] if split == "cb513" else [f"{SPLITS.index(split)}ab{i}-A" for i in range(2)]
        write_lmdb(directory / f"secondary_structure_{split}.lmdb", [tape_record(i, "ACDEFG") for i in ids])
    result = prepare.convert_archive(config)
    assert result["duplicate_chain_regions_renamed"]["cb513"] == 1
    assert dataset.read_split_ids(config, "cb513") == ["1abc-A", "1abc-A#2"]
    assert dataset.parse_chain_id("1abc-A#2") == ("1ABC", "A") and dataset.parse_chain_id("CB513_7") is None


def test_convert_archive_checks_counts_and_split_disjointness(tmp_path):
    pytest.importorskip("lmdb")
    config = dataset.load_task_config(dataset.TASK_YAML, data_root=tmp_path)
    directory = dataset.task_paths(config)["raw"] / "secondary_structure"
    directory.mkdir(parents=True)
    for split in SPLITS:
        write_lmdb(directory / f"secondary_structure_{split}.lmdb", [tape_record(f"1ab{i}-A", "ACDEFG") for i in range(3)])
    with pytest.raises(ValueError, match="expected 8678"):
        prepare.convert_archive(config)
    config["expected_chain_counts"] = {split: 3 for split in SPLITS}
    with pytest.raises(ValueError, match="in both"):
        prepare.convert_archive(config)   # the same three ids in every split: leakage between splits is refused
