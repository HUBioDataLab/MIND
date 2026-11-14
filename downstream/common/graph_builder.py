"""Policy-checked residue records -> the exact fixed-slot protein graph used in training.

Build time mirrors ``data_loading.build_edge_free_chunks.build_node_sample``
(adapter 14-slot standardization -> ``_universal_to_pyg`` node fields ->
``preposition_visible_virtuals``) and stores the compact edge-free encoding
(``data_loading.pt_schema``). Load time mirrors ``LazyUniversalDataset.__getitem__``
(``decode_edge_free_sample`` -> ``normalize_legacy_atom_ids`` -> collate
normalization -> ``attach_runtime_template_metadata``) followed by the protein
random-edge transform of ``core.train_pretrain``, run under a per-chain fixed seed.

``build_edge_free_chunks`` is deliberately not imported: importing it sets
``CUDA_VISIBLE_DEVICES=""`` for the whole process.

Residues must already have passed ``missing_atoms.apply_policy``; this module
refuses incomplete residues that are not flagged as masked.

``block_idx`` is the residue's position in the parsed chain, relative to the first
residue of the sample, so residues dropped by the policy leave a gap and Tier-1
sequence edges never join residues that are not sequence neighbours. Processed
data records this as ``schema["source"]["block_index"]``; data without it is refused.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
from typing import Optional, Sequence

import torch
from torch_geometric.data import Data

from core.masked_slot_mapping import attach_runtime_template_metadata
from data_loading.atom_position_vocab import (
    PROTEIN_BACKBONE_ATOMS,
    PROTEIN_SIDECHAIN_ATOM_PRIORITY,
    VIRTUAL_ATOMIC_NUMBER,
    is_virtual_position_code,
)
from data_loading.data_types import UniversalAtom, UniversalBlock, UniversalMolecule
from data_loading.lazy_universal_dataset import LazyUniversalDataset, normalize_legacy_atom_ids
from data_loading.precompute_visible_virtual_templates import preposition_visible_virtuals
from data_loading.protein_dynamic_random_edges import OptimizedProteinDynamicRandomEdges
from data_loading.pt_schema import (
    build_edge_free_schema,
    decode_edge_free_sample,
    edge_free_runtime_contract_from_config,
    encode_edge_free_sample,
)
from downstream.common.missing_atoms import residue_completeness
from downstream.common.structure_io import ResidueRecord

SLOTS_PER_RESIDUE = 14
MASK_TOKEN = 0  # the pretraining MLM input token (PretrainingESAModel._ensure_mlm_attributes)
POS_CODE_VOCABULARY = sorted(
    set(PROTEIN_BACKBONE_ATOMS) | set(PROTEIN_SIDECHAIN_ATOM_PRIORITY)
    | {f"V{index}" for index in range(1, 11)}
)
POS_CODE_INDEX = {code: index for index, code in enumerate(POS_CODE_VOCABULARY)}
BLOCK_INDEX = "chain_position"  # schema["source"]["block_index"] of data built by this module

_ADAPTER = None
_CONVERTER = None


def _adapter():
    global _ADAPTER
    if _ADAPTER is None:
        from data_loading.adapters.protein_adapter import ProteinAdapter

        with contextlib.redirect_stdout(io.StringIO()):  # adapter prints a banner
            _ADAPTER = ProteinAdapter()
    return _ADAPTER


def _converter():
    """Config-only converter, as in ``build_edge_free_chunks._init_worker``."""
    global _CONVERTER
    if _CONVERTER is None:
        from data_loading.cache_to_pyg import OptimizedUniversalDataset

        converter = object.__new__(OptimizedUniversalDataset)
        converter.molecule_max_atoms = None
        converter.cutoff_distance = 5.0
        converter.max_neighbors = 64
        converter.pre_filter = None
        converter.pre_transform = None
        _CONVERTER = converter
    return _CONVERTER


def residues_to_molecule(residues: Sequence[ResidueRecord], masked: Sequence[bool], mol_id: str) -> UniversalMolecule:
    """Standardize each residue with the training adapter's own 14-slot routine."""
    if len(residues) != len(masked):
        raise ValueError(f"{mol_id}: residues and mask flags differ in length")
    blocks = []
    for block_idx, (residue, is_masked) in enumerate(zip(residues, masked)):
        status = residue_completeness(residue)
        if status.missing_backbone or status.unexpected:
            raise ValueError(
                f"{mol_id} residue {residue.resname}{residue.resseq}: missing backbone "
                f"{status.missing_backbone} / unexpected atoms {status.unexpected} reached the graph builder"
            )
        if status.missing_sidechain and not is_masked:
            raise ValueError(
                f"{mol_id} residue {residue.resname}{residue.resseq}: incomplete sidechain "
                f"{status.missing_sidechain} is not masked; the adapter would shift atoms into wrong slots"
            )
        names = list(PROTEIN_BACKBONE_ATOMS) + [n for n in residue.atoms if n not in PROTEIN_BACKBONE_ATOMS]
        atoms = [
            UniversalAtom(
                element=residue.atoms[name][0],
                position=tuple(residue.atoms[name][1].tolist()),
                pos_code=name,
                block_idx=block_idx,
                atom_idx_in_block=index,
                entity_idx=0,
            )
            for index, name in enumerate(names)
        ]
        standardized = _adapter()._standardize_residue_to_14_atoms(atoms, residue.resname, block_idx, entity_idx=0)
        blocks.append(UniversalBlock(symbol=residue.resname, atoms=standardized))
    return UniversalMolecule(id=mol_id, dataset_type="protein", blocks=blocks, properties={})


def _block_ids(sequence_index: Optional[Sequence[int]], num_residues: int, mol_id: str) -> torch.Tensor:
    if sequence_index is None:
        return torch.arange(num_residues)
    ids = torch.as_tensor(list(sequence_index), dtype=torch.long)
    if ids.numel() != num_residues:
        raise ValueError(f"{mol_id}: sequence_index has {ids.numel()} entries for {num_residues} residues")
    if int(ids.min()) < 0 or bool((ids[1:] <= ids[:-1]).any()):
        raise ValueError(f"{mol_id}: sequence_index must be non-negative and strictly increasing")
    return ids - ids[0]


def build_node_sample(
    residues: Sequence[ResidueRecord], masked: Sequence[bool], mol_id: str, *,
    sequence_index: Optional[Sequence[int]],
) -> Data:
    """Node-only sample identical to an edge-free chunk entry, plus ``downstream_masked_block``.

    ``sequence_index`` gives each residue's position in its parsed chain (``PolicyResult.sequence_index``);
    ``None`` states that the residues are consecutive in the chain.
    """
    if not residues:
        raise ValueError(f"{mol_id}: no residues to convert")
    block_ids = _block_ids(sequence_index, len(residues), mol_id)
    molecule = residues_to_molecule(residues, masked, mol_id)
    data = _converter()._universal_to_pyg(molecule, raise_on_error=True)
    if data is None:
        raise ValueError(f"{mol_id}: converter returned no atoms")
    sample = Data(
        z=data.z, pos=data.pos, block_idx=data.block_idx, entity_idx=data.entity_idx,
        pos_code=list(data.pos_code), block_symbols=list(data.block_symbols),
        mol_id=data.mol_id, dataset_type=data.dataset_type, num_nodes=int(data.z.numel()),
    )
    sample, result, original_dataset_type = preposition_visible_virtuals(sample, "protein")
    node = Data(
        z=sample.z, pos=result.input_pos, block_idx=sample.block_idx,
        entity_idx=sample.entity_idx, pos_code=list(sample.pos_code),
        block_symbols=list(sample.block_symbols), mol_id=sample.mol_id,
        dataset_type=original_dataset_type, num_nodes=int(sample.z.numel()),
    )
    validate_fixed_slot_layout(node, len(residues))
    node.block_idx = block_ids.to(node.block_idx.dtype).repeat_interleave(SLOTS_PER_RESIDUE)
    node.downstream_masked_block = torch.tensor(list(masked), dtype=torch.bool).repeat_interleave(SLOTS_PER_RESIDUE)
    return node


def validate_fixed_slot_layout(sample: Data, num_residues: int) -> None:
    """14 nodes per residue, contiguous blocks, V id 120 exactly on V position codes."""
    mol_id = getattr(sample, "mol_id", "?")
    if int(sample.z.numel()) != SLOTS_PER_RESIDUE * num_residues:
        raise ValueError(f"{mol_id}: {sample.z.numel()} nodes for {num_residues} residues (expected 14 each)")
    expected_blocks = torch.arange(num_residues).repeat_interleave(SLOTS_PER_RESIDUE)
    if not torch.equal(sample.block_idx.long(), expected_blocks):
        raise ValueError(f"{mol_id}: block_idx is not contiguous 14-node blocks")
    virtual_codes = torch.tensor([is_virtual_position_code(c) for c in sample.pos_code])
    if not torch.equal(virtual_codes, sample.z == VIRTUAL_ATOMIC_NUMBER):
        raise ValueError(f"{mol_id}: V id 120 does not coincide with V position codes")
    backbone = [list(sample.pos_code[i * 14:i * 14 + 4]) for i in range(num_residues)]
    if any(codes != list(PROTEIN_BACKBONE_ATOMS) for codes in backbone):
        raise ValueError(f"{mol_id}: a block does not start with N, CA, C, O")


def edge_free_schema(sequence_neighbors_k: int, source: dict) -> dict:
    return build_edge_free_schema("protein", POS_CODE_VOCABULARY, sequence_neighbors_k,
                                  {**source, "block_index": BLOCK_INDEX})


def compact_sample(node: Data) -> Data:
    """Compact uint8/int16 storage form (``encode_edge_free_sample``) plus the mask flags."""
    compact = encode_edge_free_sample(node, "protein", POS_CODE_INDEX)
    compact.downstream_masked_block = node.downstream_masked_block.clone()
    return compact


def chain_seed(seed: int, mol_id: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{mol_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def random_edge_transform(model_config) -> OptimizedProteinDynamicRandomEdges:
    """Same arguments as ``create_pretraining_data_transforms``, without the global seed side effect."""
    if not bool(getattr(model_config, "use_dynamic_random_edges", False)):
        raise ValueError("model config has use_dynamic_random_edges=false; protein hybrid edges need it")
    return OptimizedProteinDynamicRandomEdges(
        num_random_edges=getattr(model_config, "num_random_edges", 8),
        min_distance=getattr(model_config, "random_edge_min_distance", 10.0),
        sequence_neighbors_k=getattr(model_config, "sequence_neighbors_k", 3),
        replace_existing_random_edges=getattr(model_config, "replace_existing_random_edges", False),
        seed=None,
    )


def runtime_sample(compact: Data, schema: dict, model_config, *, seed: int) -> Data:
    """Decode a stored chain exactly like the training loader and add seeded random edges."""
    block_index = (schema.get("source") or {}).get("block_index")
    if block_index != BLOCK_INDEX:
        raise ValueError(
            f"{getattr(compact, 'mol_id', '?')}: processed data has block_index={block_index!r}, expected "
            f"{BLOCK_INDEX!r}; it was built before dropped residues kept their sequence gap. "
            "Reprocess it under a new processed revision"
        )
    contract = edge_free_runtime_contract_from_config(model_config)
    sample = decode_edge_free_sample(compact, schema, contract)
    sample = normalize_legacy_atom_ids(sample)
    sample = LazyUniversalDataset._normalize_sample_for_collate(sample)
    sample.dataset_type = "protein"
    sample = attach_runtime_template_metadata(sample)
    if bool((sample.template_reason_bits != 0).any()):
        raise ValueError(f"{sample.mol_id}: block layout failed the template slot contract")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(chain_seed(seed, str(sample.mol_id)))
        sample = random_edge_transform(model_config)(sample)
    masked = getattr(compact, "downstream_masked_block", None)
    if masked is None or masked.numel() != sample.num_nodes:
        raise ValueError(f"{sample.mol_id}: compact sample lacks downstream_masked_block")
    sample.downstream_masked_block = masked.bool().clone()
    return sample
