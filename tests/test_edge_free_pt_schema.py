"""Production-contract tests for the versioned edge-free protein/RNA PT layout."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch_cluster import radius_graph
from torch_geometric.data import Batch, Data
from torch_geometric.data.collate import collate

from core.virtual_slot_graph import build_geometry_neutral_graph_view
from data_loading.atom_position_vocab import (
    PROTEIN_BACKBONE_ATOMS,
    RNA_BACKBONE_ATOMS,
    RNA_BASE_ATOMS_BY_NUCLEOTIDE,
    VIRTUAL_ATOMIC_NUMBER,
)
from data_loading.build_edge_free_chunks import (
    _init_worker,
    build_node_sample,
    build_plan,
    write_edge_free_chunk,
)
from data_loading.lazy_universal_dataset import LazyUniversalDataset
from data_loading.protein_dynamic_random_edges import OptimizedProteinDynamicRandomEdges
from data_loading.pt_schema import (
    PT_SCHEMA_NAME,
    EdgeFreeRuntimeContract,
    edge_free_runtime_contract_from_config,
    encode_edge_free_sample,
    regenerate_protein_sequence_edges,
    split_chunk_payload,
    validate_edge_free_schema,
)

K = 3


def _production_config(**overrides):
    values = dict(
        geometry_neutral_virtual_slots={"enabled": True, "protein": True, "rna": True},
        masked_slot_template={"enabled": False, "protein": False, "rna": False},
        apply_attention_on="node",
        use_dynamic_random_edges=True,
        replace_existing_random_edges=True,
        sequence_neighbors_k=K,
        random_edge_min_distance=10.0,
        cutoff_distance=5.0,
        max_spatial_neighbors=48,
        max_neighbors=64,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _stored_edges(pos, extra=None):
    """Legacy-style cache edges: radius graph over all slots (V included) plus extras."""
    edges = radius_graph(pos, r=5.0, loop=False, max_num_neighbors=64)
    if extra is not None and extra.numel():
        edges = torch.cat((edges, extra), dim=1)
    return edges


def _protein_sample(num_residues, mol_id, entity_split=None, seed=0):
    generator = torch.Generator().manual_seed(seed)
    block_codes = list(PROTEIN_BACKBONE_ATOMS) + ["CB"] + [f"V{i}" for i in range(2, 11)]
    block_types = [7, 6, 6, 8, 6] + [VIRTUAL_ATOMIC_NUMBER] * 9
    pos, z, block_idx, entity_idx, pos_code = [], [], [], [], []
    for residue in range(num_residues):
        angle = 0.6 * residue
        ca = torch.tensor([3.8 * residue * 0.55, 6.0 * torch.cos(torch.tensor(angle)), 6.0 * torch.sin(torch.tensor(angle))])
        offsets = torch.randn((14, 3), generator=generator) * 1.2
        offsets[1] = 0.0
        pos.append(ca + offsets)
        z += block_types
        block_idx += [residue] * 14
        entity = 1 if entity_split is not None and residue >= entity_split else 0
        entity_idx += [entity] * 14
        pos_code += block_codes
    pos = torch.cat(pos).to(torch.float32)
    num_nodes = pos.size(0)
    ca_nodes = [i * 14 + 1 for i in range(num_residues)]
    tier1, tier3 = [], []
    for i, ca_i in enumerate(ca_nodes):
        for j, ca_j in enumerate(ca_nodes):
            same_entity = entity_idx[ca_i] == entity_idx[ca_j]
            if same_entity and 0 < abs(i - j) <= K:
                tier1.append((ca_i, ca_j))
            elif abs(i - j) > K and float((pos[ca_i] - pos[ca_j]).norm()) >= 10.0 and (i + j) % 5 == 0:
                tier3.append((ca_i, ca_j))
    extra = torch.tensor(tier1 + tier3, dtype=torch.long).t()
    edge_index = _stored_edges(pos, extra)
    return Data(
        pos=pos,
        z=torch.tensor(z, dtype=torch.long),
        block_idx=torch.tensor(block_idx, dtype=torch.long),
        entity_idx=torch.tensor(entity_idx, dtype=torch.long),
        pos_code=pos_code,
        block_symbols=["ALA"] * num_residues,
        mol_id=mol_id,
        dataset_type="protein",
        edge_index=edge_index,
        edge_attr=(pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=1, keepdim=True),
        num_edges=torch.tensor([edge_index.size(1)]),
        num_nodes=num_nodes,
    )


def _rna_sample(num_nucleotides, mol_id, seed=0):
    generator = torch.Generator().manual_seed(seed)
    base_codes = list(RNA_BASE_ATOMS_BY_NUCLEOTIDE["C"])
    block_codes = list(RNA_BACKBONE_ATOMS) + base_codes + [
        f"V{i}" for i in range(len(base_codes) + 1, 12)
    ]
    block_types = [
        15 if code == "P" else 8 if code.startswith("O") else 7 if code.startswith("N") else 6
        for code in block_codes
    ]
    block_types[-3:] = [VIRTUAL_ATOMIC_NUMBER] * 3
    pos, z, block_idx, pos_code = [], [], [], []
    for block in range(num_nucleotides):
        center = torch.tensor([6.0 * block, 0.0, 0.0])
        pos.append(center + torch.randn((23, 3), generator=generator) * 1.5)
        z += block_types
        block_idx += [block] * 23
        pos_code += block_codes
    pos = torch.cat(pos).to(torch.float32)
    edge_index = _stored_edges(pos)
    return Data(
        pos=pos,
        z=torch.tensor(z, dtype=torch.long),
        block_idx=torch.tensor(block_idx, dtype=torch.long),
        entity_idx=torch.zeros(pos.size(0), dtype=torch.long),
        pos_code=pos_code,
        block_symbols=["C"] * num_nucleotides,
        mol_id=mol_id,
        dataset_type="rna",
        edge_index=edge_index,
        edge_attr=(pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=1, keepdim=True),
        num_edges=torch.tensor([edge_index.size(1)]),
        num_nodes=pos.size(0),
    )


def _unimol_sample(mol_id, seed=0):
    generator = torch.Generator().manual_seed(seed)
    pos = (torch.randn((9, 3), generator=generator) * 1.4).to(torch.float32)
    edge_index = _stored_edges(pos)
    return Data(
        pos=pos,
        z=torch.tensor([6, 6, 8, 7, 6, 6, 8, 6, 16], dtype=torch.long),
        block_idx=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long),
        entity_idx=torch.zeros(9, dtype=torch.long),
        pos_code=["sm"] * 9,
        block_symbols=["CCO", "CN", "CS"],
        mol_id=mol_id,
        dataset_type="unimol",
        edge_index=edge_index,
        edge_attr=(pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=1, keepdim=True),
        num_edges=torch.tensor([edge_index.size(1)]),
        num_nodes=9,
    )


def _write_legacy_chunk(root: Path, prefix: str, samples, dataset_type: str):
    processed = root / f"{prefix}_chunk_0" / "processed"
    processed.mkdir(parents=True)
    data, slices, _ = collate(samples[0].__class__, data_list=samples, increment=False, add_batch=False)
    pt_file = processed / f"{prefix}.pt"
    torch.save((data, slices), pt_file)
    metadata = {
        "chunk_file": pt_file.name,
        "num_samples": len(samples),
        "dataset_type": dataset_type,
        "total_atoms": sum(int(s.z.numel()) for s in samples),
        "total_edges": sum(int(s.edge_index.size(1)) for s in samples),
        "samples": [
            {"idx": i, "num_atoms": int(s.z.numel()), "num_edges": int(s.edge_index.size(1)), "bucket": 0}
            for i, s in enumerate(samples)
        ],
    }
    metadata_file = processed / f"{prefix}_metadata.json"
    metadata_file.write_text(json.dumps(metadata))
    return pt_file, metadata_file


def _write_edge_free(root: Path, prefix: str, samples, domain):
    chunk_dir = root / f"{prefix}_edgefree_chunk_000"
    write_edge_free_chunk(
        samples, chunk_dir, domain=domain,
        sequence_neighbors_k=K if domain == "protein" else None,
        source={"test": prefix},
    )
    return next((chunk_dir / "processed").glob("*.pt"))


@pytest.fixture
def chunks(tmp_path, monkeypatch):
    monkeypatch.setenv("MIND_INDEX_MAP_CACHE_DIR", str(tmp_path / "index_maps"))
    proteins = [
        _protein_sample(9, "P1", seed=1),
        _protein_sample(12, "P2", entity_split=6, seed=2),
        _protein_sample(5, "P3", seed=3),
    ]
    rnas = [_rna_sample(4, "R1", seed=4), _rna_sample(3, "R2", seed=5)]
    unimols = [_unimol_sample("U1", seed=6), _unimol_sample("U2", seed=7)]
    protein_pt, protein_meta = _write_legacy_chunk(tmp_path, "prot", proteins, "protein")
    rna_pt, rna_meta = _write_legacy_chunk(tmp_path, "rna", rnas, "rna")
    unimol_pt, _ = _write_legacy_chunk(tmp_path, "unimol", unimols, "unimol")
    return SimpleNamespace(
        root=tmp_path,
        protein_legacy=protein_pt,
        protein_metadata=protein_meta,
        protein_edge_free=_write_edge_free(tmp_path, "prot", proteins, "protein"),
        rna_legacy=rna_pt,
        rna_metadata=rna_meta,
        rna_edge_free=_write_edge_free(tmp_path, "rna", rnas, "rna"),
        unimol_legacy=unimol_pt,
    )


def _dataset(files, contract=None, transform=None, load_metadata=True):
    return LazyUniversalDataset(
        chunk_pt_files=[str(path) for path in files],
        transform=transform,
        max_cache_chunks=8,
        verbose=False,
        load_metadata=load_metadata,
        edge_free_runtime=contract,
    )


def _contract():
    return edge_free_runtime_contract_from_config(_production_config())


def _assert_same_field(name, legacy_value, decoded_value):
    assert type(legacy_value) is type(decoded_value), name
    if torch.is_tensor(legacy_value):
        assert legacy_value.dtype == decoded_value.dtype, name
        # Edge counts differ by design; every other dimension must match.
        if name == "edge_index":
            assert legacy_value.shape[0] == decoded_value.shape[0], name
        else:
            assert legacy_value.shape[1:] == decoded_value.shape[1:], name
    else:
        assert legacy_value == decoded_value, name


@pytest.mark.parametrize("domain", ["protein", "rna"])
def test_decoded_sample_matches_legacy_runtime_contract(chunks, domain):
    legacy = _dataset([getattr(chunks, f"{domain}_legacy")])
    edge_free = _dataset([getattr(chunks, f"{domain}_edge_free")], contract=_contract())
    assert len(legacy) == len(edge_free)
    for index in range(len(legacy)):
        old, new = legacy[index], edge_free[index]
        assert set(old.keys()) == set(new.keys())
        for name in old.keys():
            _assert_same_field(name, old[name], new[name])
        for name in ("z", "pos", "block_idx", "entity_idx"):
            assert torch.equal(old[name], new[name]), name
        for name in ("template_domain_id", "template_slot_order", "template_anchor_order",
                     "template_virtual_slot", "template_reason_bits"):
            assert torch.equal(old[name], new[name]), name
        assert old.pos_code == new.pos_code
        assert torch.allclose(
            new.edge_attr[:, 0],
            (new.pos[new.edge_index[0]] - new.pos[new.edge_index[1]]).norm(dim=1),
        )
        if domain == "rna":
            assert new.edge_index.shape == (2, 0)


def test_protein_tier1_regeneration_respects_entities_and_gap():
    sample = _protein_sample(12, "P", entity_split=6)
    edges = regenerate_protein_sequence_edges(
        sample.pos_code, sample.block_idx, sample.entity_idx, K
    )
    source, target = edges
    ca = torch.tensor([code == "CA" for code in sample.pos_code])
    assert bool(ca[source].all() and ca[target].all())
    gap = (sample.block_idx[source] - sample.block_idx[target]).abs()
    assert bool(((gap > 0) & (gap <= K)).all())
    assert bool((sample.entity_idx[source] == sample.entity_idx[target]).all())
    # 6 + 6 residues in two entities: each entity has 2*(5+4+3) directed edges.
    assert edges.size(1) == 2 * 2 * (5 + 4 + 3)


def _masked_batch(samples, masked_blocks):
    batch = Batch.from_data_list(samples)
    batch.original_types = batch.z.clone()
    mlm_mask = torch.zeros(batch.z.numel(), dtype=torch.bool)
    for graph_index, block_id in masked_blocks:
        mlm_mask |= (batch.batch == graph_index) & (batch.block_idx == block_id)
    batch.mlm_mask = mlm_mask
    masked = batch.z.clone()
    masked[mlm_mask] = 0
    batch.masked_types = masked
    batch.z = masked
    batch.x = masked
    return batch


def test_geometry_neutral_physical_graph_is_identical(chunks):
    config = _production_config()
    transform = OptimizedProteinDynamicRandomEdges(
        num_random_edges=4, min_distance=10.0, sequence_neighbors_k=K,
        replace_existing_random_edges=True,
    )
    legacy = _dataset(
        [chunks.protein_legacy, chunks.rna_legacy, chunks.unimol_legacy], transform=transform
    )
    edge_free = _dataset(
        [chunks.protein_edge_free, chunks.rna_edge_free, chunks.unimol_legacy],
        contract=_contract(), transform=transform,
    )
    order = [0, 5, 1, 3, 2, 6, 4]

    def load(dataset):
        samples = []
        for position, index in enumerate(order):
            torch.manual_seed(1000 + position)
            samples.append(dataset[index])
        return samples

    masked_blocks = [(0, 2), (0, 5), (1, 1), (2, 3), (2, 8), (3, 0), (4, 2), (6, 1)]
    old_batch = _masked_batch(load(legacy), masked_blocks)
    new_batch = _masked_batch(load(edge_free), masked_blocks)
    old_view = build_geometry_neutral_graph_view(old_batch, config)
    new_view = build_geometry_neutral_graph_view(new_batch, config)

    assert old_view.physical_edge_index.size(1) > 0
    for name in ("physical_edge_index", "slot_readout_edge_index", "compact_edge_index",
                 "compact_pos", "active_node_mask", "target_slot_mask",
                 "centered_full_pos_for_losses"):
        assert torch.equal(getattr(old_view, name), getattr(new_view, name)), name
    # The stored edge path really did carry many more cached edges.
    assert old_batch.edge_index.size(1) > 3 * new_batch.edge_index.size(1)


def test_mixed_legacy_and_edge_free_collation(chunks):
    legacy = _dataset([chunks.protein_legacy, chunks.unimol_legacy])
    edge_free = _dataset([chunks.protein_edge_free, chunks.rna_edge_free], contract=_contract())
    batch = Batch.from_data_list([legacy[3], edge_free[0], edge_free[4], legacy[1]])
    assert batch.edge_attr.size(0) == batch.edge_index.size(1)
    assert batch.num_graphs == 4
    assert batch.z.dtype == torch.long and batch.block_idx.dtype == torch.long


def test_edge_free_chunk_is_much_smaller(chunks):
    assert chunks.protein_edge_free.stat().st_size * 3 < chunks.protein_legacy.stat().st_size


def test_written_schema_and_metadata(chunks):
    payload = torch.load(chunks.protein_edge_free, weights_only=False)
    data, slices, schema = split_chunk_payload(payload, origin="test")
    assert set(data.keys()) == {
        "z", "pos", "block_idx", "entity_idx", "pos_code_ids",
        "block_symbols", "mol_id", "dataset_type",
    }
    assert data.z.dtype == torch.uint8 and data.block_idx.dtype == torch.int32
    assert data.entity_idx.dtype == torch.int16 and data.pos_code_ids.dtype == torch.uint8
    assert schema["sequence_neighbors_k"] == K
    assert schema["source"] == {"test": "prot"}
    metadata = json.loads(
        (chunks.protein_edge_free.parent / f"{chunks.protein_edge_free.stem}_metadata.json").read_text()
    )
    assert metadata[PT_SCHEMA_NAME] == schema
    assert [s["mol_id"] for s in metadata["samples"]] == ["P1", "P2", "P3"]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"geometry_neutral_virtual_slots": {"enabled": False, "protein": False, "rna": False}},
         "geometry_neutral_virtual_slots"),
        ({"masked_slot_template": {"enabled": True, "protein": True, "rna": True}},
         "masked_slot_template"),
        ({"apply_attention_on": "edge"}, "apply_attention_on"),
        ({"use_dynamic_random_edges": False}, "use_dynamic_random_edges"),
        ({"replace_existing_random_edges": False}, "replace_existing_random_edges"),
    ],
)
def test_incompatible_config_is_refused_at_dataset_init(chunks, overrides, message):
    contract = edge_free_runtime_contract_from_config(_production_config(**overrides))
    with pytest.raises(ValueError, match=message):
        _dataset([chunks.protein_edge_free], contract=contract)


def test_rna_edge_free_does_not_depend_on_protein_only_settings(chunks):
    contract = edge_free_runtime_contract_from_config(
        _production_config(use_dynamic_random_edges=False)
    )
    dataset = _dataset([chunks.rna_edge_free], contract=contract)
    assert dataset[0].edge_index.shape == (2, 0)


def test_missing_contract_is_refused(chunks):
    with pytest.raises(ValueError, match="edge_free_runtime contract"):
        _dataset([chunks.protein_edge_free], contract=None)
    lazy = _dataset([chunks.protein_edge_free], contract=None, load_metadata=False)
    with pytest.raises(ValueError, match="edge_free_runtime contract"):
        lazy[0]


def test_sequence_neighbor_mismatch_is_refused(chunks):
    contract = edge_free_runtime_contract_from_config(_production_config(sequence_neighbors_k=4))
    dataset = _dataset([chunks.protein_edge_free], contract=contract)
    with pytest.raises(ValueError, match="sequence_neighbors_k"):
        dataset[0]


def test_metadata_without_schema_is_refused(chunks):
    metadata_file = chunks.protein_edge_free.parent / f"{chunks.protein_edge_free.stem}_metadata.json"
    metadata = json.loads(metadata_file.read_text())
    del metadata[PT_SCHEMA_NAME]
    metadata_file.write_text(json.dumps(metadata))
    dataset = _dataset([chunks.protein_edge_free], contract=_contract())
    with pytest.raises(ValueError, match="does not match its metadata"):
        dataset[0]


def test_legacy_chunks_ignore_edge_free_contract(chunks):
    contract = edge_free_runtime_contract_from_config(
        _production_config(apply_attention_on="edge")
    )
    dataset = _dataset([chunks.protein_legacy], contract=contract)
    assert dataset[0].edge_index.size(1) > 0


def test_unknown_schema_version_is_refused(chunks):
    _, _, schema = split_chunk_payload(
        torch.load(chunks.protein_edge_free, weights_only=False), origin="test"
    )
    schema["version"] = 99
    with pytest.raises(ValueError, match="unsupported PT schema"):
        validate_edge_free_schema(schema, origin="test")
    schema["version"] = 1
    schema["pos_code_vocabulary"] = list(reversed(schema["pos_code_vocabulary"]))
    with pytest.raises(ValueError, match="vocabulary hash"):
        validate_edge_free_schema(schema, origin="test")


def test_writer_refuses_overwrite_and_leaves_no_partial(chunks):
    with pytest.raises(FileExistsError):
        _write_edge_free(chunks.root, "prot", [_protein_sample(3, "X")], "protein")
    assert not list(chunks.root.glob("*.partial"))


def test_encoding_fails_loud_on_out_of_range_values():
    sample = _protein_sample(3, "RANGE")
    index = {code: i for i, code in enumerate(sorted(set(sample.pos_code)))}
    sample.entity_idx = sample.entity_idx.clone()
    sample.entity_idx[0] = 40000
    with pytest.raises(ValueError, match="entity_idx range"):
        encode_edge_free_sample(sample, "protein", index)
    sample = _protein_sample(3, "CODE")
    with pytest.raises(ValueError, match="vocabulary: mol_id=CODE"):
        encode_edge_free_sample(sample, "protein", {"N": 0})
    with pytest.raises(ValueError, match="sample domain"):
        encode_edge_free_sample(_rna_sample(2, "R"), "protein", index)
