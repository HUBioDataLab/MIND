"""Bounded loader -> collate -> compact encoder and COCONUT RNG regressions."""

import json

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.data.collate import collate
from torch_geometric.utils import to_dense_batch

from core.molecular_encoder import UniversalMolecularEncoder
from core.pretraining_config import create_pretraining_config
from core.pretraining_model import PretrainingESAModel
from core.virtual_slot_graph import build_geometry_neutral_graph_view
from data_loading.atom_position_vocab import (
    PROTEIN_BACKBONE_ATOMS, RNA_BACKBONE_ATOMS,
    RNA_BASE_ATOMS_BY_NUCLEOTIDE,
)
from data_loading.build_edge_free_chunks import write_edge_free_chunk
from data_loading.lazy_universal_dataset import LazyUniversalDataset
from data_loading.masking_utils import select_mask_indices
from data_loading.pt_schema import edge_free_runtime_contract_from_config


def _config(**overrides):
    values = dict(
        pretraining_tasks=["mlm"], seed=42, hidden_dims=[16], graph_dim=16,
        num_heads=[2], layer_types=["S"], gaussian_kernels=4,
        use_posenc=False, use_bfloat16=False, use_equivariant_features=False,
        use_domain_local_slot_embedding=True,
        use_dynamic_random_edges=True, replace_existing_random_edges=True,
        xformers_or_torch_attn="torch", protein_block_mlm=True, rna_block_mlm=True,
        geometry_neutral_virtual_slots={"enabled": True, "protein": True, "rna": True},
        coconut_mask_ratio=0.5,
    )
    values.update(overrides)
    return create_pretraining_config(**values)


def _sample(domain, *, bonds=None, size=None, blocks=1):
    if domain == "protein":
        codes = list(PROTEIN_BACKBONE_ATOMS) + ["CB"] + [f"V{i}" for i in range(2, 11)]
        real_count = 5
    elif domain == "rna":
        base = list(RNA_BASE_ATOMS_BY_NUCLEOTIDE["C"])
        codes = list(RNA_BACKBONE_ATOMS) + base + [f"V{i}" for i in range(len(base) + 1, 12)]
        real_count = 12 + len(base)
    else:
        codes = ["sm"] * (size or 4)
        real_count = len(codes)
    block_size = len(codes)
    z = torch.tensor([
        120 if i >= real_count else 15 if code == "P" else
        7 if code.startswith("N") else 8 if code.startswith("O") else 6
        for i, code in enumerate(codes)
    ])
    pos = torch.zeros(len(codes), 3)
    pos[:real_count, 0] = torch.arange(real_count) * 1.5
    # Padding overlaps a real atom; compact removal must make it irrelevant.
    pos[real_count:] = pos[0]
    z = z.repeat(blocks)
    pos = torch.cat([pos + torch.tensor([50.0 * block, 0, 0]) for block in range(blocks)])
    codes = codes * blocks
    nodes = torch.arange(len(codes))
    src = nodes.repeat_interleave(len(codes))
    dst = nodes.repeat(len(codes))
    edges = torch.stack((src[src != dst], dst[src != dst]))
    data = Data(
        z=z, pos=pos, edge_index=edges,
        edge_attr=(pos[edges[0]] - pos[edges[1]]).norm(dim=1, keepdim=True),
        block_idx=torch.arange(blocks).repeat_interleave(block_size),
        entity_idx=torch.zeros(len(codes), dtype=torch.long),
        pos_code=codes, block_symbols=["ALA" if domain == "protein" else "C"] * blocks,
        mol_id=f"fixture_{domain}", dataset_type=domain, num_nodes=len(codes),
        num_edges=torch.tensor([edges.size(1)]),
    )
    if bonds is not None:
        data.bond_edge_index = bonds
    return data


def _write_legacy(tmp_path, sample):
    path = tmp_path / f"{sample.dataset_type}.pt"
    data, slices, _ = collate(Data, [sample], increment=False, add_batch=False)
    torch.save((data, slices), path)
    path.with_name(f"{path.stem}_metadata.json").write_text(json.dumps({
        "chunk_file": path.name, "dataset_type": sample.dataset_type,
        "num_samples": 1, "total_atoms": sample.num_nodes,
        "samples": [{"idx": 0, "num_atoms": sample.num_nodes,
                     "num_edges": sample.edge_index.size(1), "bucket": 0}],
    }))
    return path


@pytest.fixture
def samples(tmp_path, monkeypatch):
    monkeypatch.setenv("MIND_INDEX_MAP_CACHE_DIR", str(tmp_path / "index_maps"))
    paths = []
    for domain in ("protein", "rna", "unimol", "coconut"):
        bonds = torch.tensor([[0, 1], [1, 0]]) if domain == "coconut" else None
        sample = _sample(domain, bonds=bonds)
        if domain in {"protein", "rna"}:
            chunk_dir = tmp_path / f"{domain}_edgefree_chunk_0"
            write_edge_free_chunk(
                [sample], chunk_dir, domain=domain, source={"test": "coconut_contract"},
                sequence_neighbors_k=3 if domain == "protein" else None,
            )
            paths.append(next((chunk_dir / "processed").glob("*.pt")))
        else:
            paths.append(_write_legacy(tmp_path, sample))
    dataset = LazyUniversalDataset(
        [str(path) for path in paths], verbose=False, load_metadata=True,
        edge_free_runtime=edge_free_runtime_contract_from_config(_config()),
    )
    return {domain: dataset[i] for i, domain in enumerate(("protein", "rna", "unimol", "coconut"))}


def _batch(samples):
    batch = Batch.from_data_list([sample.clone() for sample in samples])
    batch.original_types = batch.z.clone()
    batch.mlm_mask = torch.zeros_like(batch.z, dtype=torch.bool)
    batch.masked_types = batch.z.clone()
    batch.x = batch.z.clone()
    return batch


class _EncoderCaptured(Exception):
    pass


def _compact_encode(model, batch):
    """Execute the production forward through the real encoder, stop before ESA."""
    captured = {}

    def capture(encoder, args, output):
        x, pos, compact = args
        dense, _ = to_dense_batch(x, compact.batch)
        distances = (pos[compact.edge_index[0]] - pos[compact.edge_index[1]]).norm(dim=1)
        captured.update(
            batch=compact, embeddings=output.detach(),
            coordination=encoder._compute_chemical_coordination(
                dense, compact, edge_distances=distances,
            ),
        )
        raise _EncoderCaptured

    hook = model.encoder.register_forward_hook(capture)
    try:
        with torch.no_grad(), pytest.raises(_EncoderCaptured):
            model(batch)
    finally:
        hook.remove()
    return captured


@pytest.mark.parametrize("domain", ["protein", "rna", "unimol"])
def test_other_domain_features_and_embeddings_independent_of_coconut(samples, domain):
    model = PretrainingESAModel(_config()).eval()
    alone = _compact_encode(model, _batch([samples[domain]]))
    for order in ([domain, "coconut"], ["coconut", domain]):
        mixed = _compact_encode(model, _batch([samples[key] for key in order]))
        graph = order.index(domain)
        n = alone["embeddings"].size(0)
        assert torch.equal(alone["coordination"][0, :n], mixed["coordination"][graph, :n])
        torch.testing.assert_close(
            alone["embeddings"], mixed["embeddings"][mixed["batch"].batch == graph],
            rtol=1e-5, atol=1e-6,
        )
        coconut = order.index("coconut")
        assert mixed["coordination"][coconut, :4].tolist() == [1, 1, 0, 0]
    if domain == "protein":
        assert alone["coordination"][0, :4].tolist() == [1, 2, 2, 2]


def test_three_domain_proxy_degrees_and_embeddings_preserved(samples):
    model = PretrainingESAModel(_config()).eval()
    order = ["protein", "rna", "unimol"]
    mixed = _compact_encode(model, _batch([samples[key] for key in order]))
    for graph, key in enumerate(order):
        alone = _compact_encode(model, _batch([samples[key]]))
        n = alone["embeddings"].size(0)
        expected = [1] + [2] * (n - 2) + [1]
        assert mixed["coordination"][graph, :n].tolist() == expected
        assert not mixed["coordination"][graph, n:].any()  # Dense padding.
        torch.testing.assert_close(alone["embeddings"], mixed["embeddings"][mixed["batch"].batch == graph])


def test_masked_protein_backbone_keeps_1221_next_to_coconut(samples):
    model = PretrainingESAModel(_config()).eval()
    results = []
    for order in (["protein"], ["protein", "coconut"], ["coconut", "protein"]):
        batch = _batch([samples[key] for key in order])
        protein = batch.batch == order.index("protein")
        batch.mlm_mask[protein] = True
        batch.z[protein] = 0
        batch.x = batch.z.clone()
        result = _compact_encode(model, batch)
        graph = order.index("protein")
        assert result["coordination"][graph, :4].tolist() == [1, 2, 2, 1]
        results.append(result["embeddings"][result["batch"].batch == graph])
    torch.testing.assert_close(results[0], results[1])
    torch.testing.assert_close(results[0], results[2])


def test_loader_distinguishes_legacy_empty_and_missing_bonds(samples):
    missing = LazyUniversalDataset._normalize_sample_for_collate(_sample("unimol"))
    empty = LazyUniversalDataset._normalize_sample_for_collate(
        _sample("unimol", bonds=torch.empty((2, 0), dtype=torch.long))
    )
    assert missing.has_explicit_bonds.tolist() == empty.has_explicit_bonds.tolist() == [False]
    assert samples["coconut"].has_explicit_bonds.tolist() == [True]


def test_compact_reindex_keeps_source_even_when_all_bonds_removed(samples):
    # An explicit protein list whose sole bond touches a hidden sidechain slot.
    protein = samples["protein"].clone()
    protein.bond_edge_index = torch.tensor([[0, 4], [4, 0]])
    protein = LazyUniversalDataset._normalize_sample_for_collate(protein)
    batch = _batch([protein, samples["coconut"]])
    batch.mlm_mask[:14] = True
    batch.z[:14] = 0
    batch.x = batch.z.clone()
    view = build_geometry_neutral_graph_view(batch, _config())
    assert view.compact_to_full.tolist() == [0, 1, 2, 3, 14, 15, 16, 17]
    result = _compact_encode(PretrainingESAModel(_config()).eval(), batch)
    compact = result["batch"]
    assert compact.bond_edge_index.tolist() == [[4, 5], [5, 4]]
    assert compact.has_explicit_bonds.tolist() == [True, True]
    assert result["coordination"][0, :4].tolist() == [0, 0, 0, 0]
    assert result["coordination"][1, :4].tolist() == [1, 1, 0, 0]


@pytest.mark.parametrize("bad_bond,match", [
    (torch.tensor([[-1], [0]]), "bond_edge_index.*out of range"),
    (torch.tensor([[0], [18]]), "bond_edge_index.*out of range"),
    (torch.tensor([[0], [14]]), "bond_edge_index.*cross-graph"),
    (torch.tensor([[0.0], [1.0]]), "bond_edge_index.*torch.long"),
])
def test_bad_bonds_fail_before_compact_indexing(samples, bad_bond, match):
    batch = _batch([samples["protein"], samples["coconut"]])
    batch.bond_edge_index = bad_bond
    with pytest.raises((ValueError, TypeError), match=match):
        _compact_encode(PretrainingESAModel(_config()).eval(), batch)


def test_bad_bond_source_metadata_fails_loud(samples):
    encoder = UniversalMolecularEncoder(_config())
    batch = _batch([samples["unimol"], samples["coconut"]])
    dense, _ = to_dense_batch(batch.z, batch.batch)
    distances = (batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]).norm(dim=1)
    for availability in (torch.tensor([True]), torch.tensor([0, 1]),
                         torch.tensor([False, False]), [False, True]):
        batch.has_explicit_bonds = availability
        with pytest.raises((ValueError, TypeError), match="has_explicit_bonds"):
            encoder._compute_chemical_coordination(dense, batch, edge_distances=distances)


@pytest.mark.parametrize("method", ["random_atom", "random_walk"])
def test_coconut_validation_preserves_other_domains_mask_rng(method):
    raw = [_sample("coconut", bonds=torch.tensor([[0, 1], [1, 0]]), size=32),
           _sample("protein", blocks=6), _sample("rna", blocks=6),
           _sample("unimol", size=32)]
    normalized = [LazyUniversalDataset._normalize_sample_for_collate(sample) for sample in raw]
    model = PretrainingESAModel(_config(
        coconut_masking_method=method,
        geometry_neutral_virtual_slots={"enabled": False},
        protein_block_mlm_ratio=0.5, rna_block_mlm_ratio=0.5,
    ))
    reference = Batch.from_data_list([sample.clone() for sample in normalized[1:]])
    mixed = Batch.from_data_list([sample.clone() for sample in normalized])
    model._ensure_mlm_attributes(reference, batch_idx=9, is_validation=True)
    model._ensure_mlm_attributes(mixed, batch_idx=9, is_validation=True)
    assert torch.equal(reference.mlm_mask, mixed.mlm_mask[raw[0].num_nodes:])
    for graph in (0, 1):
        for block in range(6):
            mask = reference.mlm_mask[(reference.batch == graph) & (reference.block_idx == block)]
            assert bool(mask.all()) or not bool(mask.any())


def test_proxy_threshold_outgoing_count_and_unbatched_output_preserved():
    sample = _sample("unimol")
    # Deliberately asymmetric edges: count sources only, including duplicates.
    sample.edge_index = torch.tensor([[0, 1, 1, 2], [1, 2, 3, 3]])
    distances = torch.tensor([0.5, 1.85, 1.85, 1.851])
    encoder = UniversalMolecularEncoder(_config())
    dense = sample.z.unsqueeze(0)
    result = encoder._compute_chemical_coordination(dense, sample, edge_distances=distances)
    assert result.tolist() == [[0, 2, 0, 0]]
    assert result.dtype == torch.float32 and result.device == dense.device


def test_unbatched_encoder_matches_single_graph_batch():
    sample = LazyUniversalDataset._normalize_sample_for_collate(_sample("unimol"))
    encoder = UniversalMolecularEncoder(_config()).eval()
    batched = Batch.from_data_list([sample.clone()])
    torch.testing.assert_close(
        encoder(sample.z, sample.pos, sample),
        encoder(batched.z, batched.pos, batched),
    )


@pytest.mark.parametrize("edges", [torch.tensor([[-1], [0]]), torch.tensor([[0], [4]]),
                                   torch.tensor([0, 1])])
def test_loader_rejects_bad_bond_payload_with_sample_identity(edges):
    sample = _sample("coconut", bonds=edges)
    with pytest.raises(ValueError, match="bond_edge_index.*mol_id=fixture_coconut"):
        LazyUniversalDataset._normalize_sample_for_collate(sample)


@pytest.mark.parametrize("method", ["random_atom", "random_walk"])
@pytest.mark.parametrize("batched", [True, False])
def test_coconut_validation_masks_repeat_and_training_uses_global_rng(method, batched):
    sample = _sample("coconut", size=32)
    nodes = torch.arange(31)
    sample.bond_edge_index = torch.stack((torch.cat((nodes, nodes + 1)), torch.cat((nodes + 1, nodes))))
    sample = LazyUniversalDataset._normalize_sample_for_collate(sample)
    model = PretrainingESAModel(_config(coconut_masking_method=method))

    def masked(validation, batch_idx=7):
        batch = Batch.from_data_list([sample.clone()]) if batched else sample.clone()
        model._ensure_mlm_attributes(batch, batch_idx=batch_idx, is_validation=validation)
        return batch.mlm_mask

    validation_rng = torch.get_rng_state()
    first = masked(True)
    assert torch.equal(validation_rng, torch.get_rng_state())
    torch.rand(100)  # Unrelated global RNG activity must not change validation.
    assert torch.equal(first, masked(True))
    assert not torch.equal(first, masked(True, batch_idx=8))
    torch.manual_seed(531)
    before = torch.get_rng_state()
    training = masked(False)
    assert not torch.equal(before, torch.get_rng_state())
    torch.manual_seed(531)
    assert torch.equal(training, masked(False))


@pytest.mark.parametrize("method", ["random_atom", "random_walk"])
@pytest.mark.parametrize("output_device,generator_device", [("cpu", "cpu"), ("cuda", "cpu"), ("cpu", "cuda"), ("cuda", "cuda")])
@pytest.mark.parametrize("has_bonds", [True, False])
def test_mask_generator_device_is_independent_of_index_device(method, output_device, generator_device, has_bonds):
    if "cuda" in (output_device, generator_device) and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; generator/index device check not executed")
    nodes = torch.arange(19, device=output_device)
    bonds = torch.stack((torch.cat((nodes, nodes + 1)), torch.cat((nodes + 1, nodes)))) if has_bonds else None

    def select():
        generator = torch.Generator(device=generator_device).manual_seed(71)
        return select_mask_indices(
            20, 0.5, "random_atom", bond_edge_index=bonds, dataset_type="coconut",
            coconut_method=method, device=torch.device(output_device), generator=generator,
        )

    a, b = select(), select()
    assert a.device.type == output_device and a.dtype == torch.long
    assert torch.equal(a, b)
