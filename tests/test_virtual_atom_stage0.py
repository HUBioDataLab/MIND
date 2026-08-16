import json
from types import SimpleNamespace

import torch
from torch_geometric.data import Batch, Data, InMemoryDataset

from core.pretraining_model import PretrainingESAModel
from core.pretraining_config import PretrainingConfig
from core.pretraining_tasks import PretrainingTasks
from core.metrics.rna_atom_types import pos_code_to_element
from data_loading.adapters.rna_adapter import RNAAdapter, normalize_rna_element
from data_loading.atom_position_vocab import (
    RARE_ATOMIC_NUMBER,
    RNA_BACKBONE_ATOMS,
    VIRTUAL_ATOMIC_NUMBER,
    has_exact_rna_backbone,
    rna_backbone_contract_violations,
)
from data_loading.audit_virtual_atom_baseline import audit_cache
from data_loading.cache_to_pyg import atom_to_model_id, element_to_atomic_number
from data_loading.data_types import UniversalAtom
from data_loading.lazy_universal_dataset import normalize_legacy_atom_ids


class _MaskHarness:
    _ensure_mlm_attributes = PretrainingESAModel._ensure_mlm_attributes
    _graph_dtype_list = staticmethod(PretrainingESAModel._graph_dtype_list)

    def __init__(self):
        self.config = SimpleNamespace(
            pretraining_tasks=["mlm"],
            protein_block_mlm=True,
            protein_block_mlm_ratio=0.5,
            mlm_mask_ratio=0.15,
        )
        self.current_epoch = 2


def _mixed_batch() -> Batch:
    protein = Data(
        pos=torch.zeros((28, 3)),
        z=torch.full((28,), 6, dtype=torch.long),
        block_idx=torch.tensor([0] * 14 + [1] * 14),
        dataset_type="pdb",
    )
    rna = Data(
        pos=torch.zeros((20, 3)),
        z=torch.full((20,), 7, dtype=torch.long),
        block_idx=torch.zeros(20, dtype=torch.long),
        dataset_type="rna",
    )
    return Batch.from_data_list([protein, rna])


def test_validation_and_seeded_training_masks_are_reproducible():
    harness = _MaskHarness()

    validation_a = _mixed_batch()
    validation_b = _mixed_batch()
    harness._ensure_mlm_attributes(validation_a, batch_idx=7, is_validation=True)
    harness._ensure_mlm_attributes(validation_b, batch_idx=7, is_validation=True)
    assert torch.equal(validation_a.mlm_mask, validation_b.mlm_mask)

    # Protein masking is residue-wise: one of the two 14-atom blocks is whole.
    protein_mask = validation_a.mlm_mask[:28].reshape(2, 14)
    assert protein_mask.all(dim=1).sum().item() == 1
    assert (~protein_mask).all(dim=1).sum().item() == 1
    # RNA remains atom-level: int(20 * 0.15) == 3 atoms.
    assert validation_a.mlm_mask[28:].sum().item() == 3

    torch.manual_seed(42)
    training_a = _mixed_batch()
    harness._ensure_mlm_attributes(training_a, batch_idx=0, is_validation=False)
    torch.manual_seed(42)
    training_b = _mixed_batch()
    harness._ensure_mlm_attributes(training_b, batch_idx=0, is_validation=False)
    assert torch.equal(training_a.mlm_mask, training_b.mlm_mask)


def test_cache_audit_respects_per_graph_local_edge_indices(tmp_path):
    graph_a = Data(
        pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        z=torch.tensor([6, 119]),
        block_idx=torch.tensor([0, 0]),
        entity_idx=torch.tensor([0, 0]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.tensor([[1.0], [1.0]]),
        pos_code=["C1'", "V1"],
        block_symbols=["A"],
    )
    graph_b = Data(
        pos=torch.tensor([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        z=torch.tensor([6, 119]),
        block_idx=torch.tensor([0, 0]),
        entity_idx=torch.tensor([0, 0]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.tensor([[0.0], [0.0]]),
        pos_code=["C1'", "V1"],
        block_symbols=["A"],
    )
    data, slices = InMemoryDataset.collate([graph_a, graph_b])
    cache_path = tmp_path / "tiny.pt"
    torch.save((data, slices), cache_path)
    metadata_path = tmp_path / "tiny_metadata.json"
    metadata_path.write_text(json.dumps({
        "num_samples": 2,
        "total_atoms": 4,
        "total_edges": 4,
    }))

    report = audit_cache(
        name="tiny",
        cache_path=cache_path,
        metadata_path=metadata_path,
        expected_slots=2,
        edge_chunk_size=1,
    )
    assert report["geometry"]["zero_distance_directed_edges"] == 2
    assert report["geometry"]["zero_distance_directed_edges_touching_virtual"] == 2
    assert report["degree"]["zero_source_degree_atoms"] == 0
    assert report["integrity"]["edge_attr_distance_mismatch_over_1e-5"] == 0
    assert report["slots"]["non_exact_expected_blocks"] == 0


def test_rna_bromine_normalization_does_not_change_real_boron():
    assert normalize_rna_element("B", "BR") == "BR"
    assert normalize_rna_element("Br", "BR") == "BR"
    assert normalize_rna_element("B", "B") == "B"
    assert element_to_atomic_number("BR") == 35
    assert element_to_atomic_number("B") == RARE_ATOMIC_NUMBER
    assert element_to_atomic_number("V") == RARE_ATOMIC_NUMBER
    assert atom_to_model_id(SimpleNamespace(
        element="V", pos_code="V1", is_virtual=True
    )) == VIRTUAL_ATOMIC_NUMBER
    assert atom_to_model_id(SimpleNamespace(
        element="V", pos_code="sm", is_virtual=False
    )) == RARE_ATOMIC_NUMBER
    assert pos_code_to_element("BR") == "Br"
    assert pos_code_to_element("B") == "UNK"


def test_legacy_virtual_id_migration_preserves_real_rare_atoms():
    sample = Data(
        z=torch.tensor([119, 119, 5, 119]),
        pos_code=["V1", "B", "BR", "BR"],
    )
    migrated = normalize_legacy_atom_ids(sample)
    assert migrated.z.tolist() == [120, 119, 35, 35]


def test_mlm_head_accepts_distinct_rare_and_virtual_targets():
    config = PretrainingConfig(graph_dim=16, atom_types=121)
    tasks = PretrainingTasks(config)
    data = Data(
        mlm_mask=torch.tensor([True, True]),
        original_types=torch.tensor([RARE_ATOMIC_NUMBER, VIRTUAL_ATOMIC_NUMBER]),
        masked_types=torch.tensor([0, 0]),
    )
    loss = tasks.mlm_loss(torch.randn(2, 16), data)
    assert torch.isfinite(loss)


def test_23_slot_standardization_repairs_legacy_bromine_atom():
    legacy_bromine = UniversalAtom(
        element="B",
        position=(0.0, 0.0, 0.0),
        pos_code="BR",
        block_idx=0,
        atom_idx_in_block=0,
        entity_idx=0,
    )
    standardized = RNAAdapter(use_23atom_uniform=True)._standardize_residue_to_23_atoms(
        [legacy_bromine], "C", 0
    )
    repaired = next(atom for atom in standardized if atom.pos_code == "BR")
    assert repaired.element == "BR"
    assert element_to_atomic_number(repaired.element) == 35


def test_rna_template_eligibility_requires_each_backbone_position_once():
    assert has_exact_rna_backbone(RNA_BACKBONE_ATOMS)

    missing_codes = RNA_BACKBONE_ATOMS[:-1]
    missing, duplicates = rna_backbone_contract_violations(missing_codes)
    assert missing == ("C1'",)
    assert duplicates == {}
    assert not has_exact_rna_backbone(missing_codes)

    duplicated_codes = [*RNA_BACKBONE_ATOMS, "OP1"]
    missing, duplicates = rna_backbone_contract_violations(duplicated_codes)
    assert missing == ()
    assert duplicates == {"OP1": 2}
    assert not has_exact_rna_backbone(duplicated_codes)
