"""End-to-end CPU tests of core/evaluate_pretrain.py on a tiny synthetic four-domain split.

The production flash_varlen layers run through the ``cpu_flash`` SDPA stand-in.
"""

import hashlib
import json
import os

import pytest
import pytorch_lightning as pl
import torch
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from torch_geometric.loader import DataLoader as GeometricDataLoader

from core import evaluate_pretrain as ev
from core.pretraining_config import create_pretraining_config, load_config_yaml
from core.metrics.virtual_slot_diagnostics_callback import VirtualSlotDiagnosticsCallback
from core.pretraining_model import PretrainingESAModel
from data_loading.atom_position_vocab import (
    PROTEIN_BACKBONE_ATOMS,
    PROTEIN_SIDECHAIN_ATOM_PRIORITY,
    PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE,
    RNA_BACKBONE_ATOMS,
    RNA_BASE_ATOM_PRIORITY,
    RNA_BASE_ATOMS_BY_NUCLEOTIDE,
)
from data_loading.build_edge_free_chunks import write_edge_free_chunk

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ELEMENT = {"C": 6, "N": 7, "O": 8, "S": 16, "P": 15}
RESIDUES = ["ALA", "LEU", "SER", "TRP", "GLY", "CYS", "LYS", "ASP", "ILE", "VAL", "PRO", "HIS"]


def _graph(codes, z, pos, blocks, symbols, domain, mol_id, bonds=None):
    n = len(codes)
    src, dst = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    keep = (src != dst) & ((pos[src] - pos[dst]).norm(dim=-1) < 5.0)
    edges = torch.stack((src[keep], dst[keep]))
    data = Data(
        z=torch.tensor(z), pos=pos, edge_index=edges,
        edge_attr=(pos[edges[0]] - pos[edges[1]]).norm(dim=1, keepdim=True),
        block_idx=torch.tensor(blocks), entity_idx=torch.zeros(n, dtype=torch.long),
        pos_code=list(codes), block_symbols=list(symbols), mol_id=mol_id,
        dataset_type=domain, num_nodes=n, num_edges=torch.tensor([edges.size(1)]),
    )
    if bonds is not None:
        data.bond_edge_index = bonds
    return data


def _protein(residues, mol_id, seed):
    gen = torch.Generator().manual_seed(seed)
    codes, z, pos, blocks = [], [], [], []
    for b, residue in enumerate(residues):
        side = sorted(PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE[residue], key=PROTEIN_SIDECHAIN_ATOM_PRIORITY.index)
        names = list(PROTEIN_BACKBONE_ATOMS) + side
        origin = torch.tensor([3.8 * b, 0.0, 0.0])
        real = [origin + torch.randn(3, generator=gen) * 1.2 for _ in names]
        anchor = real[4] if len(names) > 4 else real[1]
        codes += names + [f"V{i}" for i in range(len(side) + 1, 11)]
        z += [ELEMENT[name[0]] for name in names] + [120] * (10 - len(side))
        pos += real + [anchor.clone() for _ in range(10 - len(side))]
        blocks += [b] * 14
    return _graph(codes, z, torch.stack(pos), blocks, residues, "protein", mol_id)


def _rna(nucleotides, mol_id, seed):
    gen = torch.Generator().manual_seed(seed)
    codes, z, pos, blocks = [], [], [], []
    for b, nucleotide in enumerate(nucleotides):
        base = sorted(RNA_BASE_ATOMS_BY_NUCLEOTIDE[nucleotide], key=RNA_BASE_ATOM_PRIORITY.index)
        names = list(RNA_BACKBONE_ATOMS) + base
        origin = torch.tensor([6.0 * b, 0.0, 0.0])
        real = [origin + torch.randn(3, generator=gen) * 1.5 for _ in names]
        codes += names + [f"V{i}" for i in range(len(base) + 1, 12)]
        z += [ELEMENT[name[0]] for name in names] + [120] * (11 - len(base))
        pos += real + [real[-1].clone() for _ in range(11 - len(base))]
        blocks += [b] * 23
    return _graph(codes, z, torch.stack(pos), blocks, nucleotides, "rna", mol_id)


def _small(domain, size, mol_id, seed):
    gen = torch.Generator().manual_seed(seed)
    z = [6, 8, 7, 6] + [1] * (size - 4)
    pos = torch.randn(size, 3, generator=gen) * 1.3
    bonds = None
    if domain == "coconut":
        chain = torch.arange(size - 1)
        bonds = torch.cat([torch.stack((chain, chain + 1)), torch.stack((chain + 1, chain))], dim=1)
    return _graph(["sm"] * size, z, pos, [0] * size, ["mol"], domain, mol_id, bonds=bonds)


def _write_legacy_chunk(chunk_dir, samples):
    processed = chunk_dir / "processed"
    processed.mkdir(parents=True)
    data, slices, _ = collate(Data, samples, increment=False, add_batch=False)
    torch.save((data, slices), processed / f"{chunk_dir.name}.pt")
    (processed / f"{chunk_dir.name}_metadata.json").write_text(json.dumps({
        "chunk_file": f"{chunk_dir.name}.pt", "dataset_type": samples[0].dataset_type,
        "num_samples": len(samples), "total_atoms": sum(s.num_nodes for s in samples),
        "samples": [{"idx": i, "num_atoms": s.num_nodes, "num_edges": s.edge_index.size(1), "bucket": 0}
                    for i, s in enumerate(samples)],
    }))


def _write_split(root, split, offset):
    proteins = [_protein(RESIDUES[(i * 5) % 12:] + RESIDUES[:(i * 5) % 12][: 3 + i], f"p{split}{i}", offset + i)
                for i in range(4)]
    rnas = [_rna(["A", "G", "C", "U", "G", "A"][: 3 + i], f"r{split}{i}", offset + 10 + i) for i in range(3)]
    write_edge_free_chunk(proteins, root / f"protein_{split}_chunk_0", domain="protein",
                          sequence_neighbors_k=3, source={"test": "evaluate_pretrain"})
    write_edge_free_chunk(rnas, root / f"rna_{split}_chunk_0", domain="rna",
                          sequence_neighbors_k=None, source={"test": "evaluate_pretrain"})
    _write_legacy_chunk(root / f"unimol_{split}_chunk_0",
                        [_small("unimol", 6 + i, f"u{split}{i}", offset + 20 + i) for i in range(6)])
    _write_legacy_chunk(root / f"coconut_{split}_chunk_0",
                        [_small("coconut", 7 + i, f"c{split}{i}", offset + 30 + i) for i in range(5)])


def _config(root):
    values = load_config_yaml(os.path.join(REPO, "core", "pretraining_config_multidomain_1m_coconut.yaml"))
    layers = len(values["layer_types"])
    values.update(hidden_dims=[32] * layers, num_heads=[4] * layers, graph_dim=32, mlp_hidden_size=64,
                  gaussian_kernels=16, num_workers=0, max_atoms_per_batch=400, use_wandb=False)
    datasets = values["multi_domain"]["datasets"]
    for key, name in (("pdb", "protein"), ("rna", "rna"), ("unimol", "unimol"), ("metabolite", "coconut")):
        datasets[key]["splits"] = {
            split: {"chunk_dir": str(root / f"{name}_{'val' if split == 'train' else split}"), "chunk_range": None}
            for split in ("train", "val", "test")
        }
    return create_pretraining_config(**values)


def _checkpoint(path, config, seed):
    torch.manual_seed(seed)
    model = PretrainingESAModel(config)
    torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": pl.__version__, "epoch": 3, "global_step": 77}, path)
    return path


@pytest.fixture
def setup(tmp_path, monkeypatch, cpu_flash):
    return _make_setup(tmp_path, monkeypatch)


def _make_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("MIND_INDEX_MAP_CACHE_DIR", str(tmp_path / "index_maps"))
    data = tmp_path / "data"
    data.mkdir()
    _write_split(data, "val", 0)
    _write_split(data, "test", 100)
    config = _config(data)
    ckpts = tmp_path / "ckpts"
    ckpts.mkdir()
    return {
        "tmp": tmp_path,
        "a": _checkpoint(ckpts / "a.ckpt", config, 0),
        "b": _checkpoint(ckpts / "b.ckpt", config, 1),
    }


def _run(ckpt, out, *extra, device="cpu"):
    args = ev.parse_args(["--ckpt", str(ckpt), "--split", "val", "--seed", "5", "--device", device,
                          "--output-dir", str(out), *extra])
    return ev.evaluate(args)


def _manifest(out):
    return (out / "mask_manifest.csv").read_text()


def test_end_to_end_outputs_determinism_and_untouched_checkpoint(setup):
    ckpt = setup["a"]
    sha_before = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    mtime_before = ckpt.stat().st_mtime_ns
    first = _run(ckpt, setup["tmp"] / "eval1")
    second = _run(ckpt, setup["tmp"] / "eval2")

    assert hashlib.sha256(ckpt.read_bytes()).hexdigest() == sha_before
    assert ckpt.stat().st_mtime_ns == mtime_before
    assert sorted(os.listdir(ckpt.parent)) == ["a.ckpt", "b.ckpt"]

    out = setup["tmp"] / "eval1"
    for name in ("metrics.json", "provenance.json", "tables.txt", "eval_log.txt", "mask_manifest.csv"):
        assert (out / name).is_file(), name
    for name in ("protein_elements.csv", "protein_identity_per_name.csv", "rna_identity_per_name.csv",
                 "unimol_confusion.csv", "coconut_calibration.csv", "losses_per_domain.csv",
                 "protein_per_atom.csv", "rna_per_atom.csv", "protein_distance_visible_pairs_diagnostic.csv"):
        assert (out / "tables" / name).is_file(), name

    metrics = json.loads((out / "metrics.json").read_text())
    provenance = metrics["provenance"]
    assert provenance["checkpoint"]["sha256"] == sha_before
    assert provenance["checkpoint"]["unchanged_after_run"] is True
    assert (provenance["checkpoint"]["epoch"], provenance["checkpoint"]["global_step"]) == (3, 77)
    assert provenance["git"]["commit"] and provenance["split"] == "val" and provenance["seed"] == 5
    assert set(metrics["domain_mlm"]) == {"protein", "rna", "unimol", "coconut"}
    assert set(metrics["losses"]["per_domain_batch_means"]) == {"protein", "rna", "unimol", "coconut"}
    protein = metrics["domain_mlm"]["protein"]
    assert protein["block_identity"]["n_blocks"] > 0
    assert protein["block_identity"]["n_classes"] == 18
    assert metrics["domain_mlm"]["rna"]["block_identity"]["n_blocks"] > 0
    assert protein["summary"]["n_masked_virtual"] > 0
    assert set(protein["distance_visible_pairs_diagnostic"]) == {"long_range", "short_range"}
    assert protein["embedding"]["n_nodes"] > 0
    assert metrics["domain_mlm"]["unimol"]["summary"]["n_masked_hydrogen"] >= 0
    assert metrics["protein_per_atom"] is not None and metrics["rna_per_atom"] is not None
    assert metrics["virtual_slot"]["enabled"] is True

    # Same checkpoint and settings: identical numbers and masks.
    assert first["domain_mlm"] == second["domain_mlm"]
    assert first["losses"] == second["losses"]
    assert first["mask_digest"] == second["mask_digest"]
    assert _manifest(out) == _manifest(setup["tmp"] / "eval2")


def test_masks_do_not_depend_on_checkpoint_or_batch_composition(setup):
    tmp = setup["tmp"]
    a = _run(setup["a"], tmp / "a")
    b = _run(setup["b"], tmp / "b")
    assert a["mask_digest"] == b["mask_digest"]
    assert _manifest(tmp / "a") == _manifest(tmp / "b")
    assert a["domain_mlm"]["protein"]["summary"] != b["domain_mlm"]["protein"]["summary"]

    # Different batch packing: each graph keeps its mask.
    repacked = _run(setup["a"], tmp / "repacked", "--max-atoms-per-batch", "150")
    assert repacked["plan"] != a["plan"]
    assert repacked["mask_digest"] == a["mask_digest"]

    # A domain filter keeps the remaining domains' masks; another seed changes them.
    protein_only = _run(setup["a"], tmp / "protein_only", "--domains", "protein")
    assert set(protein_only["domain_mlm"]) == {"protein"}
    assert protein_only["mask_digest"] == {"protein": a["mask_digest"]["protein"]}
    reseeded = ev.evaluate(ev.parse_args(["--ckpt", str(setup["a"]), "--split", "val", "--seed", "6",
                                          "--device", "cpu", "--output-dir", str(tmp / "seed6")]))
    assert reseeded["mask_digest"]["protein"] != a["mask_digest"]["protein"]


def test_diagnostics_do_not_change_logged_losses(setup):
    metrics = _run(setup["a"], setup["tmp"] / "diag")
    checkpoint = torch.load(setup["a"], map_location="cpu", weights_only=False)
    model = PretrainingESAModel(config=dict(checkpoint["hyper_parameters"]["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.config.seed = 5
    domains = ["protein", "rna", "unimol", "coconut"]
    dataset, indices, _ = ev.load_split_dataset(model.config, "val", domains)
    plan, _, positions = ev.plan_batches(ev.sample_entries(dataset, indices, domains), 400, None)
    loader = GeometricDataLoader(
        ev.EvalSampleDataset(dataset, positions, 5, {p: e["domain"] for e in plan for p in e["positions"]}),
        batch_sampler=[e["positions"] for e in plan])
    # Only the per-batch RNG seeding of the evaluator (long-range pair sampling) is kept.
    seeding = ev.EvalRunCallback(plan, 5, VirtualSlotDiagnosticsCallback(enabled=False))
    trainer = pl.Trainer(accelerator="cpu", devices=1, precision="bf16-mixed", logger=False,
                         enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                         callbacks=[seeding], default_root_dir=str(setup["tmp"] / "plain"))
    plain = trainer.validate(model, dataloaders=loader, verbose=False)[0]
    logged = metrics["losses"]["lightning_logged"]
    assert set(plain) == set(logged)
    for key, value in plain.items():
        assert float(value) == pytest.approx(logged[key], abs=1e-6), key


def test_refuses_existing_or_checkpoint_adjacent_output(setup):
    existing = setup["tmp"] / "exists"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="existing output directory"):
        _run(setup["a"], existing)
    with pytest.raises(ValueError, match="next to the checkpoint"):
        _run(setup["a"], setup["a"].parent / "eval")
    assert sorted(os.listdir(setup["a"].parent)) == ["a.ckpt", "b.ckpt"]
    with pytest.raises(ValueError, match="not enabled"):
        _run(setup["a"], setup["tmp"] / "bad_domain", "--domains", "qm9")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real flash_varlen kernels need CUDA")
def test_gpu_eval_is_bit_reproducible(tmp_path, monkeypatch):
    setup = _make_setup(tmp_path, monkeypatch)
    try:
        first = _run(setup["a"], tmp_path / "gpu1", device="cuda")
        second = _run(setup["a"], tmp_path / "gpu2", device="cuda")
    finally:
        torch.use_deterministic_algorithms(False)
    provenance = json.loads((tmp_path / "gpu1" / "metrics.json").read_text())["provenance"]
    assert provenance["deterministic_algorithms"] is True
    assert first["domain_mlm"] == second["domain_mlm"]
    assert first["losses"] == second["losses"]
    assert first["mask_digest"] == second["mask_digest"]
