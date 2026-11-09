"""Protein-ligand complex graphs built from the same training functions as EC.

Two representations share one stored record:

``joint``
    One graph holding the pocket (14-slot protein blocks, built by
    ``downstream.common.graph_builder``) and the ligand (one atom per block,
    ``pos_code`` ``sm`` exactly as in the UniMol chunks). The graph is labelled
    ``dataset_type="protein"`` so that
    ``core.virtual_slot_graph.build_geometry_neutral_graph_view`` removes the
    visible V padding and rebuilds the radius graph over *every* node of the
    graph, which is what produces protein-ligand interface edges.

``separate``
    The pocket alone (identical to the EC protein path) and the ligand alone as
    a ``dataset_type="unimol"`` graph with ``_universal_to_pyg`` radius edges.

Ligand bonds are deliberately **not** attached. ``resolve_bond_sources`` reports
explicit bonds per *graph*, and ``UniversalMolecularEncoder._compute_chemical_coordination``
then suppresses the <=1.85 A distance proxy for the whole graph. Attaching ligand
bonds to a joint graph would therefore zero the chemical-coordination feature of
every pocket atom. Pretraining has no explicit bonds for proteins or for UniMol,
so the distance proxy is the in-distribution choice for both parts.

Block ids are sequence-faithful, not pocket-local: a pocket residue keeps the
index it has inside its own chain, offset per chain, so Tier-1 CA sequence edges
only connect residues that really are sequence neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import torch
from torch_cluster import radius_graph
from torch_geometric.data import Data

from core.masked_slot_mapping import attach_runtime_template_metadata
from data_loading.cache_to_pyg import element_to_atomic_number
from data_loading.lazy_universal_dataset import LazyUniversalDataset, normalize_legacy_atom_ids
from data_loading.pt_schema import (
    build_edge_free_schema,
    decode_edge_free_sample,
    edge_free_runtime_contract_from_config,
)
from downstream.common.graph_builder import (
    POS_CODE_VOCABULARY,
    SLOTS_PER_RESIDUE,
    chain_seed,
    random_edge_transform,
)

LIGAND_POS_CODE = "sm"              # the pos_code every UniMol pretraining atom carries
LIGAND_DATASET_TYPE = "unimol"
PROTEIN_ENTITY = 0
LIGAND_ENTITY = 1
PROTEIN_CHAIN_STRIDE = 1_000_000    # block ids: chain_ordinal * stride + residue index in chain
LIGAND_BLOCK_OFFSET = 1_500_000_000  # one block per ligand atom, far above every protein block id
MAX_PROTEIN_CHAINS = PROTEIN_CHAIN_STRIDE  # guard below; the real limit is int32 on block_idx
COMPLEX_POS_CODE_VOCABULARY = sorted(set(POS_CODE_VOCABULARY) | {LIGAND_POS_CODE})


def complex_edge_free_schema(sequence_neighbors_k: int, source: dict) -> dict:
    """Protein edge-free schema whose pos_code vocabulary also holds the ligand code."""
    return build_edge_free_schema(
        "protein", COMPLEX_POS_CODE_VOCABULARY, sequence_neighbors_k, source
    )


@dataclass(frozen=True)
class LigandAtoms:
    """Heavy ligand atoms in input order."""

    elements: Tuple[str, ...]
    positions: np.ndarray  # [L, 3] float32

    def __post_init__(self) -> None:
        if self.positions.ndim != 2 or self.positions.shape != (len(self.elements), 3):
            raise ValueError(
                f"ligand positions {self.positions.shape} do not match {len(self.elements)} elements"
            )
        if not np.isfinite(self.positions).all():
            raise ValueError("ligand positions contain NaN or inf")

    def model_ids(self) -> torch.Tensor:
        return torch.tensor(
            [element_to_atomic_number(element) for element in self.elements], dtype=torch.long
        )


def protein_block_ids(chain_ordinals: Sequence[int], chain_residue_indices: Sequence[int]) -> torch.Tensor:
    """Sequence-faithful block ids: ``chain_ordinal * stride + residue index inside the chain``."""
    if len(chain_ordinals) != len(chain_residue_indices):
        raise ValueError("chain ordinals and residue indices differ in length")
    ids = []
    for ordinal, index in zip(chain_ordinals, chain_residue_indices):
        if not 0 <= int(ordinal) < MAX_PROTEIN_CHAINS:
            raise ValueError(f"chain ordinal {ordinal} outside [0, {MAX_PROTEIN_CHAINS})")
        if not 0 <= int(index) < PROTEIN_CHAIN_STRIDE:
            raise ValueError(f"residue index {index} outside [0, {PROTEIN_CHAIN_STRIDE})")
        value = int(ordinal) * PROTEIN_CHAIN_STRIDE + int(index)
        if value >= LIGAND_BLOCK_OFFSET:
            raise ValueError(f"protein block id {value} collides with the ligand block range")
        ids.append(value)
    if len(set(ids)) != len(ids):
        raise ValueError("protein block ids are not unique")
    return torch.tensor(ids, dtype=torch.long)


def remap_protein_blocks(node: Data, block_ids: torch.Tensor) -> Data:
    """Replace the contiguous 0..R-1 block ids of a built protein sample.

    ``graph_builder.build_node_sample`` validates the fixed-slot layout with
    contiguous ids. Pocket residues are not contiguous in their chain, so the
    real sequence indices are written back afterwards.
    """
    num_residues = int(block_ids.numel())
    if int(node.z.numel()) != num_residues * SLOTS_PER_RESIDUE:
        raise ValueError(
            f"{getattr(node, 'mol_id', '?')}: {node.z.numel()} nodes for {num_residues} residues"
        )
    if not torch.equal(
        node.block_idx.long(), torch.arange(num_residues).repeat_interleave(SLOTS_PER_RESIDUE)
    ):
        raise ValueError(f"{getattr(node, 'mol_id', '?')}: block ids are not contiguous before remapping")
    if bool((block_ids[1:] <= block_ids[:-1]).any()):
        raise ValueError(f"{getattr(node, 'mol_id', '?')}: block ids must be strictly increasing")
    node.block_idx = block_ids.repeat_interleave(SLOTS_PER_RESIDUE)
    return node


def _ligand_nodes(ligand: LigandAtoms, *, entity: int, block_offset: int) -> dict:
    count = len(ligand.elements)
    return {
        "z": ligand.model_ids(),
        "pos": torch.from_numpy(np.asarray(ligand.positions, dtype=np.float32)),
        "block_idx": torch.arange(count, dtype=torch.long) + int(block_offset),
        "entity_idx": torch.full((count,), int(entity), dtype=torch.long),
        "pos_code": [LIGAND_POS_CODE] * count,
        "block_symbols": [str(element) for element in ligand.elements],
    }


def _finalize(sample: Data, dataset_type: str, masked: torch.Tensor, model_config, *, seed: int) -> Data:
    """Loader-equivalent normalization, template metadata and seeded random edges."""
    sample = normalize_legacy_atom_ids(sample)
    sample = LazyUniversalDataset._normalize_sample_for_collate(sample)
    sample.dataset_type = dataset_type
    sample = attach_runtime_template_metadata(sample)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(chain_seed(seed, str(sample.mol_id)))
        sample = random_edge_transform(model_config)(sample)
    if masked.numel() != sample.num_nodes:
        raise ValueError(f"{sample.mol_id}: mask flags do not cover every node")
    sample.downstream_masked_block = masked.bool().clone()
    # Every LBA sample carries the same field set, so a pocket graph and a
    # ligand graph can share one collated batch in the separate representation.
    sample.num_edges = torch.tensor([int(sample.edge_index.size(1))], dtype=torch.long)
    return sample


def check_protein_template_contract(sample: Data) -> None:
    """Template reason bits must be clean on the protein part.

    ``attach_runtime_template_metadata`` validates every block against the
    14-slot protein template because the graph is labelled ``protein``. Ligand
    blocks fail that template by construction; they are never masked, so
    ``map_masked_template_slots`` ignores their bits. Only the protein blocks
    have to be clean.
    """
    protein = sample.entity_idx.long() == PROTEIN_ENTITY
    bad = sample.template_reason_bits[protein] != 0
    if bool(bad.any()):
        blocks = sorted({int(b) for b in sample.block_idx[protein][bad].tolist()})
        raise ValueError(
            f"{sample.mol_id}: protein blocks {blocks[:10]} failed the template slot contract"
        )


def joint_runtime_sample(
    compact_protein: Data,
    ligand: LigandAtoms,
    schema: dict,
    model_config,
    *,
    seed: int,
) -> Data:
    """Pocket + ligand as one ``dataset_type="protein"`` graph."""
    contract = edge_free_runtime_contract_from_config(model_config)
    protein = decode_edge_free_sample(compact_protein, schema, contract)
    masked_protein = getattr(compact_protein, "downstream_masked_block", None)
    if masked_protein is None or masked_protein.numel() != protein.num_nodes:
        raise ValueError(f"{protein.mol_id}: stored pocket lacks downstream_masked_block")

    nodes = _ligand_nodes(ligand, entity=LIGAND_ENTITY, block_offset=LIGAND_BLOCK_OFFSET)
    # Ligand nodes are appended, so every protein node index (and the Tier-1 CA
    # edges decoded above) stays valid.
    sample = Data(
        z=torch.cat((protein.z, nodes["z"])),
        pos=torch.cat((protein.pos, nodes["pos"])),
        block_idx=torch.cat((protein.block_idx, nodes["block_idx"])),
        entity_idx=torch.cat((protein.entity_idx, nodes["entity_idx"])),
        pos_code=list(protein.pos_code) + nodes["pos_code"],
        block_symbols=list(protein.block_symbols) + nodes["block_symbols"],
        mol_id=protein.mol_id,
        edge_index=protein.edge_index,
        edge_attr=protein.edge_attr,
        num_nodes=int(protein.num_nodes) + len(ligand.elements),
    )
    masked = torch.cat(
        (masked_protein.bool(), torch.zeros(len(ligand.elements), dtype=torch.bool))
    )
    sample = _finalize(sample, "protein", masked, model_config, seed=seed)
    check_protein_template_contract(sample)
    return sample


def pocket_runtime_sample(compact_protein: Data, schema: dict, model_config, *, seed: int) -> Data:
    """Pocket alone; identical to the EC protein runtime path."""
    contract = edge_free_runtime_contract_from_config(model_config)
    protein = decode_edge_free_sample(compact_protein, schema, contract)
    masked = getattr(compact_protein, "downstream_masked_block", None)
    if masked is None or masked.numel() != protein.num_nodes:
        raise ValueError(f"{protein.mol_id}: stored pocket lacks downstream_masked_block")
    sample = _finalize(protein, "protein", masked.bool(), model_config, seed=seed)
    check_protein_template_contract(sample)
    return sample


def ligand_runtime_sample(ligand: LigandAtoms, mol_id: str, model_config, *, seed: int) -> Data:
    """Ligand alone as a UniMol-style small-molecule graph with radius edges."""
    nodes = _ligand_nodes(ligand, entity=PROTEIN_ENTITY, block_offset=0)
    count = len(ligand.elements)
    if count < 1:
        raise ValueError(f"{mol_id}: ligand has no heavy atom")
    cutoff = float(getattr(model_config, "cutoff_distance", 5.0))
    max_neighbors = int(getattr(model_config, "max_neighbors", 64))
    if count > 1:
        edge_index = radius_graph(
            nodes["pos"], r=cutoff, batch=None, loop=False, max_num_neighbors=max_neighbors
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.linalg.vector_norm(
        nodes["pos"][edge_index[0]] - nodes["pos"][edge_index[1]], dim=1, keepdim=True
    ).to(torch.float32)
    sample = Data(
        z=nodes["z"], pos=nodes["pos"], block_idx=nodes["block_idx"], entity_idx=nodes["entity_idx"],
        pos_code=nodes["pos_code"], block_symbols=nodes["block_symbols"], mol_id=str(mol_id),
        edge_index=edge_index, edge_attr=edge_attr, num_nodes=count,
    )
    return _finalize(
        sample, LIGAND_DATASET_TYPE, torch.zeros(count, dtype=torch.bool), model_config, seed=seed
    )


def cross_entity_edge_count(sample: Data, edge_index: torch.Tensor) -> int:
    """Edges whose endpoints sit in different entities (protein <-> ligand)."""
    if edge_index.numel() == 0:
        return 0
    entity = sample.entity_idx.long()
    return int((entity[edge_index[0]] != entity[edge_index[1]]).sum())


def measured_physical_edges(sample: Data, model_config) -> Tuple[int, int]:
    """``(physical edges, cross-entity edges)`` of the encoder graph, without running the model.

    Mirrors ``PretrainingESAModel.forward``: no masking, ``original_types = z``.
    """
    from torch_geometric.data import Batch

    from core.virtual_slot_graph import build_geometry_neutral_graph_view

    batch = Batch.from_data_list([sample])
    batch.original_types = batch.z.clone()
    batch.mlm_mask = batch.downstream_masked_block.bool().clone()
    view = build_geometry_neutral_graph_view(batch, model_config)
    if view is None:
        raise ValueError("geometry_neutral_virtual_slots is disabled in this config")
    edges = view.physical_edge_index
    return int(edges.size(1)), cross_entity_edge_count(batch, edges)


def pocket_sequence(resnames: Sequence[str]) -> str:
    from downstream.common.structure_io import THREE_TO_ONE

    return "".join(THREE_TO_ONE[name] for name in resnames)


def ligand_from_rows(elements: Sequence[str], positions: Sequence[Sequence[float]]) -> LigandAtoms:
    return LigandAtoms(
        elements=tuple(str(element) for element in elements),
        positions=np.asarray(positions, dtype=np.float32).reshape(len(elements), 3),
    )


def stored_ligand(ligand: LigandAtoms) -> dict:
    return {"elements": list(ligand.elements), "positions": torch.from_numpy(
        np.asarray(ligand.positions, dtype=np.float32))}


def ligand_from_stored(record: dict) -> LigandAtoms:
    return LigandAtoms(
        elements=tuple(record["elements"]),
        positions=record["positions"].numpy().astype(np.float32),
    )


def complex_entity_mask(batch, entity: int) -> torch.Tensor:
    entity_idx = getattr(batch, "entity_idx", None)
    if entity_idx is None or entity_idx.numel() != batch.z.numel():
        raise ValueError("complex pooling requires entity_idx with one value per node")
    return entity_idx.long() == int(entity)


__all__ = [
    "LIGAND_POS_CODE", "LIGAND_DATASET_TYPE", "PROTEIN_ENTITY", "LIGAND_ENTITY",
    "COMPLEX_POS_CODE_VOCABULARY", "LigandAtoms", "complex_edge_free_schema",
    "protein_block_ids", "remap_protein_blocks", "joint_runtime_sample",
    "pocket_runtime_sample", "ligand_runtime_sample", "cross_entity_edge_count",
    "measured_physical_edges", "ligand_from_rows", "stored_ligand", "ligand_from_stored",
    "complex_entity_mask", "check_protein_template_contract", "pocket_sequence",
]
