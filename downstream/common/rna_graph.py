"""Nucleotide records -> the fixed-slot RNA graph of pretraining (12 backbone + 11 base slots per nucleotide).

Build time mirrors the RNA branch of ``data_loading.build_edge_free_chunks.build_node_sample``: the training
adapter's own ``_standardize_residue_to_23_atoms`` (base atoms in priority order, V padding at the base centre),
``_universal_to_pyg`` node fields, ``preposition_visible_virtuals``, and the compact edge-free encoding of
``data_loading.pt_schema``. Load time mirrors ``LazyUniversalDataset.__getitem__`` for an RNA chunk
(``decode_edge_free_sample`` -> ``normalize_legacy_atom_ids`` -> collate normalization ->
``attach_runtime_template_metadata``). RNA has no stored, regenerated or random edges: every edge is built inside the
model from the geometry-neutral view, so a runtime sample is deterministic.

``build_edge_free_chunks`` is deliberately not imported: importing it sets ``CUDA_VISIBLE_DEVICES=""``.

What the adapter rules do to real structures (all counted by ``nucleotide_report``, none silent):

* a nucleotide whose backbone lacks canonical positions (for example the 5' end, which has no phosphate) keeps the
  atoms it has; the template mapping quarantines such a block (never a masking target), exactly as in pretraining;
* a base atom that is missing is not invented: the standardizer fills the 11 base slots with the real atoms in
  priority order and pads the rest with V at the base centre, so atoms after a gap move to an earlier slot;
* atoms whose name is not a canonical backbone, base or terminal position are removed here and counted.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch_geometric.data import Data

from core.masked_slot_mapping import attach_runtime_template_metadata
from data_loading.atom_position_vocab import (
    RNA_BACKBONE_ATOMS,
    RNA_BASE_ATOM_PRIORITY,
    RNA_BASE_ATOMS_BY_NUCLEOTIDE,
    RNA_TERMINAL_ATOMS,
    VIRTUAL_ATOMIC_NUMBER,
    count_p_only_rna_blocks,
    is_virtual_position_code,
    normalize_rna_position_code,
)
from data_loading.data_types import UniversalAtom, UniversalBlock, UniversalMolecule
from data_loading.lazy_universal_dataset import LazyUniversalDataset, normalize_legacy_atom_ids
from data_loading.precompute_visible_virtual_templates import preposition_visible_virtuals
from data_loading.pt_schema import (
    build_edge_free_schema,
    decode_edge_free_sample,
    edge_free_runtime_contract_from_config,
    encode_edge_free_sample,
)

SLOTS_PER_NUCLEOTIDE = 23
BASE_SLOTS = 11
MASK_TOKEN = 0
KNOWN_NUCLEOTIDES = tuple(sorted(RNA_BASE_ATOMS_BY_NUCLEOTIDE))  # A, C, G, U
POS_CODE_VOCABULARY = sorted(
    set(RNA_BACKBONE_ATOMS) | set(RNA_BASE_ATOM_PRIORITY) | {f"V{index}" for index in range(1, BASE_SLOTS + 1)}
)
POS_CODE_INDEX = {code: index for index, code in enumerate(POS_CODE_VOCABULARY)}
KNOWN_POSITIONS = set(RNA_BACKBONE_ATOMS) | set(RNA_BASE_ATOM_PRIORITY) | set(RNA_TERMINAL_ATOMS)

_ADAPTER = None
_CONVERTER = None


@dataclass
class NucleotideRecord:
    """One nucleotide: ``resname`` in A/C/G/U and its heavy atoms as ``(name, element, (x, y, z))``.

    Names are canonical (legacy aliases already mapped), unique within the nucleotide and in the order the
    structure lists them; hydrogens and alternate locations are removed by the reader.
    """

    resname: str
    atoms: List[Tuple[str, str, Tuple[float, float, float]]]
    label: str = ""

    def __post_init__(self) -> None:
        if self.resname not in KNOWN_NUCLEOTIDES:
            raise ValueError(f"nucleotide {self.label or '?'}: resname {self.resname!r} is not one of {KNOWN_NUCLEOTIDES}")
        names = [name for name, _, _ in self.atoms]
        if len(set(names)) != len(names):
            duplicated = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(f"nucleotide {self.label or '?'}: duplicated atom names {duplicated}")
        if not self.atoms:
            raise ValueError(f"nucleotide {self.label or '?'}: no atoms")


@dataclass(frozen=True)
class NucleotideReport:
    """Completeness of one nucleotide against the canonical atom lists (counts only; nothing is repaired)."""

    missing_backbone: Tuple[str, ...]
    missing_base: Tuple[str, ...]
    unexpected: Tuple[str, ...]
    terminal: Tuple[str, ...]


def nucleotide_report(record: NucleotideRecord) -> NucleotideReport:
    names = {name for name, _, _ in record.atoms}
    return NucleotideReport(
        missing_backbone=tuple(name for name in RNA_BACKBONE_ATOMS if name not in names),
        missing_base=tuple(name for name in RNA_BASE_ATOMS_BY_NUCLEOTIDE[record.resname] if name not in names),
        unexpected=tuple(sorted(names - KNOWN_POSITIONS)),
        terminal=tuple(sorted(names & set(RNA_TERMINAL_ATOMS))),
    )


def canonical_atom_name(name: str) -> str:
    return normalize_rna_position_code(name)


def canonical_element(element: str, name: str) -> str:
    """The adapter's element normalization (a legacy bromine record labelled B is bromine)."""
    from data_loading.adapters.rna_adapter import normalize_rna_element

    return normalize_rna_element(element, name)


def _adapter():
    global _ADAPTER
    if _ADAPTER is None:
        from data_loading.adapters.rna_adapter import RNAAdapter

        with contextlib.redirect_stdout(io.StringIO()):  # the adapter prints a banner
            _ADAPTER = RNAAdapter()
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


def nucleotides_to_molecule(nucleotides: Sequence[NucleotideRecord], mol_id: str) -> UniversalMolecule:
    """Standardize each nucleotide with the training adapter's own 23-slot routine."""
    blocks = []
    for block_idx, record in enumerate(nucleotides):
        unexpected = nucleotide_report(record).unexpected
        if unexpected:
            raise ValueError(f"{mol_id} nucleotide {record.label}: unexpected atoms {unexpected} reached the graph builder")
        backbone = [item for name in RNA_BACKBONE_ATOMS for item in record.atoms if item[0] == name]
        rest = [item for item in record.atoms if item[0] not in RNA_BACKBONE_ATOMS]
        atoms = [
            UniversalAtom(
                element=element, position=tuple(float(value) for value in position), pos_code=name,
                block_idx=block_idx, atom_idx_in_block=index, entity_idx=0,
            )
            for index, (name, element, position) in enumerate(backbone + rest)
        ]
        blocks.append(UniversalBlock(symbol=record.resname,
                                     atoms=_adapter()._standardize_residue_to_23_atoms(atoms, record.resname, block_idx)))
    return UniversalMolecule(id=mol_id, dataset_type="rna", blocks=blocks, properties={})


def validate_fixed_slot_layout(sample: Data, num_nucleotides: int) -> None:
    """23 nodes per nucleotide for a complete backbone, contiguous blocks, V id 120 exactly on V position codes."""
    mol_id = getattr(sample, "mol_id", "?")
    block_idx = sample.block_idx.long()
    if int(block_idx.numel()) != int(sample.z.numel()) or len(sample.pos_code) != int(sample.z.numel()):
        raise ValueError(f"{mol_id}: inconsistent node fields")
    if int(block_idx.max()) + 1 != num_nucleotides or bool((block_idx[1:] < block_idx[:-1]).any()):
        raise ValueError(f"{mol_id}: block_idx is not {num_nucleotides} contiguous, ordered blocks")
    virtual_codes = torch.tensor([is_virtual_position_code(code) for code in sample.pos_code])
    if not torch.equal(virtual_codes, sample.z == VIRTUAL_ATOMIC_NUMBER):
        raise ValueError(f"{mol_id}: V id 120 does not coincide with V position codes")
    for block in range(num_nucleotides):
        nodes = (block_idx == block).nonzero().flatten().tolist()
        codes = [sample.pos_code[node] for node in nodes]
        if sum(not is_virtual_position_code(code) for code in codes) == 0:
            raise ValueError(f"{mol_id}: block {block} holds no real atom")
        if len(codes) < BASE_SLOTS or not all(is_virtual_position_code(c) or c in POS_CODE_INDEX for c in codes):
            raise ValueError(f"{mol_id}: block {block} breaks the fixed-slot layout ({codes})")
        if len(codes) - BASE_SLOTS != sum(code in RNA_BACKBONE_ATOMS for code in codes):
            raise ValueError(f"{mol_id}: block {block} does not end in exactly {BASE_SLOTS} base slots ({codes})")


def build_node_sample(nucleotides: Sequence[NucleotideRecord], mol_id: str) -> Data:
    """Node-only RNA sample identical to an edge-free chunk entry, plus ``downstream_masked_block`` (all False)."""
    if not nucleotides:
        raise ValueError(f"{mol_id}: no nucleotides to convert")
    molecule = nucleotides_to_molecule(nucleotides, mol_id)
    if count_p_only_rna_blocks(molecule):
        raise ValueError(f"{mol_id}: contains a P-only nucleotide block (the pretraining builder refuses it)")
    data = _converter()._universal_to_pyg(molecule, raise_on_error=True)
    if data is None:
        raise ValueError(f"{mol_id}: converter returned no atoms")
    sample = Data(
        z=data.z, pos=data.pos, block_idx=data.block_idx, entity_idx=data.entity_idx,
        pos_code=list(data.pos_code), block_symbols=list(data.block_symbols),
        mol_id=data.mol_id, dataset_type=data.dataset_type, num_nodes=int(data.z.numel()),
    )
    sample, result, original_dataset_type = preposition_visible_virtuals(sample, "rna")
    node = Data(
        z=sample.z, pos=result.input_pos, block_idx=sample.block_idx,
        entity_idx=sample.entity_idx, pos_code=list(sample.pos_code),
        block_symbols=list(sample.block_symbols), mol_id=sample.mol_id,
        dataset_type=original_dataset_type, num_nodes=int(sample.z.numel()),
    )
    validate_fixed_slot_layout(node, len(nucleotides))
    node.downstream_masked_block = torch.zeros(int(node.z.numel()), dtype=torch.bool)
    return node


def edge_free_schema(source: dict) -> dict:
    return build_edge_free_schema("rna", POS_CODE_VOCABULARY, None, dict(source))


def compact_sample(node: Data) -> Data:
    """Compact uint8/int16 storage form (``encode_edge_free_sample``) plus the mask flags."""
    compact = encode_edge_free_sample(node, "rna", POS_CODE_INDEX)
    compact.downstream_masked_block = node.downstream_masked_block.clone()
    return compact


def runtime_sample(compact: Data, schema: dict, model_config) -> Data:
    """Decode a stored RNA graph like the training loader; no random edges exist for RNA."""
    if schema.get("domain") != "rna":
        raise ValueError(f"{getattr(compact, 'mol_id', '?')}: schema domain {schema.get('domain')!r}, expected 'rna'")
    contract = edge_free_runtime_contract_from_config(model_config)
    sample = decode_edge_free_sample(compact, schema, contract)
    sample = normalize_legacy_atom_ids(sample)
    sample = LazyUniversalDataset._normalize_sample_for_collate(sample)
    sample.dataset_type = "rna"
    sample = attach_runtime_template_metadata(sample)
    masked = getattr(compact, "downstream_masked_block", None)
    if masked is None or masked.numel() != sample.num_nodes:
        raise ValueError(f"{sample.mol_id}: compact sample lacks downstream_masked_block")
    sample.downstream_masked_block = masked.bool().clone()
    return sample


def quarantined_nucleotides(sample: Data) -> int:
    """Nucleotides whose block the template mapping quarantines (incomplete backbone; never a masking target)."""
    bits = sample.template_reason_bits
    blocks = sample.block_idx.long()
    return int(torch.unique(blocks[bits != 0]).numel())
