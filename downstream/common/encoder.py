"""Load a MIND checkpoint as a frozen (or random-init) encoder and pool node states.

Only the pre-PMA node states of ``PretrainingESAModel.forward`` are used; the
graph embedding comes from the PMA decoder, which no pretraining loss trains.

Encoder input contract (``core.virtual_slot_graph.build_geometry_neutral_graph_view``):
``batch.original_types`` holds the pre-mask ids and ``batch.mlm_mask`` the
selected blocks. Downstream chains select no block, except residues flagged
``downstream_masked_block`` by the ``mask`` missing-atom policy; those get the
pretraining treatment (all 14 slots selected, ids set to the mask token 0).

Pooling uses ``active_node_mask & ~mlm_mask``: real atoms that ESA encoded
with their true atom type. Visible V padding, latent sidechain slots and the
mask-token backbone of masked residues are excluded.
"""

from __future__ import annotations

import contextlib
from typing import Dict, Iterable, List

import torch
from torch_geometric.data import Batch, Data

from core.pretraining_model import PretrainingESAModel
from downstream.common.graph_builder import MASK_TOKEN

POOLINGS = ("mean_atoms", "ca", "mean_residues")
# Complex (protein + ligand) poolings. They need ``batch.entity_idx`` and are
# never requested by the single-entity protein tasks, so EC is unaffected.
COMPLEX_POOLINGS = ("mean_pocket", "mean_ligand", "concat_pocket_ligand")
ALL_POOLINGS = POOLINGS + COMPLEX_POOLINGS
CA_ANCHOR_ORDER = 1  # attach_runtime_template_metadata: N=0, CA=1, C=2
PROTEIN_ENTITY = 0
LIGAND_ENTITY = 1


def geometry_domains(modality: str) -> tuple:
    """Geometry-neutral domains a task's graphs need: RNA graphs ``rna``, everything else ``protein`` (as before)."""
    return ("rna",) if modality == "rna" else ("protein",)


def _check_config(config, domains=("protein",)) -> None:
    settings = getattr(config, "geometry_neutral_virtual_slots", None) or {}
    missing = [domain for domain in domains if not (settings.get("enabled", False) and settings.get(domain, False))]
    if missing:
        raise ValueError(f"downstream embedding requires geometry_neutral_virtual_slots for {' and '.join(missing)}")
    if getattr(config, "apply_attention_on", "node") != "node":
        raise ValueError("downstream embedding supports node attention only")


def load_mind_encoder(ckpt_path: str, device, *, random_init: bool = False, init_seed: int = 0,
                      domains=("protein",)) -> PretrainingESAModel:
    """Strictly load a checkpoint, or build the same architecture with untrained weights.

    ``domains`` are the geometry-neutral domains the task's graphs need (``geometry_domains``)."""
    device = torch.device(device)
    if random_init:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = checkpoint["hyper_parameters"]["config"]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(init_seed))
            model = PretrainingESAModel(config=dict(config))
    else:
        model = PretrainingESAModel.load_from_checkpoint(ckpt_path, map_location=device, strict=True)
    _check_config(model.config, domains)
    return model.to(device).eval()


def autocast_for(model: PretrainingESAModel, device: torch.device):
    """Match the training precision (Lightning ``bf16-mixed`` when ``use_bfloat16``)."""
    if bool(getattr(model.config, "use_bfloat16", False)):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def collate(samples: List[Data]) -> Batch:
    return Batch.from_data_list(samples)


def prepare_encoder_batch(batch: Batch) -> Batch:
    """Set the MLM fields the geometry-neutral view requires (no block selected unless flagged)."""
    masked = getattr(batch, "downstream_masked_block", None)
    if masked is None or masked.numel() != batch.z.numel():
        raise ValueError("batch lacks downstream_masked_block (one flag per node)")
    masked = masked.bool()
    batch.original_types = batch.z.clone()
    batch.mlm_mask = masked.clone()
    z = batch.z.clone()
    z[masked] = MASK_TOKEN
    batch.z = z
    batch.x = z
    return batch


def encode(model: PretrainingESAModel, batch: Batch):
    """Return full-index node states and the pooling mask (gradients allowed)."""
    batch = prepare_encoder_batch(batch)
    device = batch.z.device
    with autocast_for(model, device):
        _, node_embeddings = model(batch)
    pool_mask = batch.active_node_mask & ~batch.mlm_mask
    return node_embeddings.float(), pool_mask


def _residue_ids(batch: Batch) -> torch.Tensor:
    graph, block = batch.batch.long(), batch.block_idx.long()
    start = torch.ones_like(graph, dtype=torch.bool)
    start[1:] = (graph[1:] != graph[:-1]) | (block[1:] != block[:-1])
    return torch.cumsum(start.long(), dim=0) - 1


def _masked_mean(values: torch.Tensor, groups: torch.Tensor, mask: torch.Tensor, num_groups: int):
    sums = values.new_zeros((num_groups, values.size(-1))).index_add_(0, groups[mask], values[mask])
    counts = values.new_zeros(num_groups).index_add_(0, groups[mask], values.new_ones(int(mask.sum())))
    return sums / counts.clamp_min(1).unsqueeze(-1), counts


def _entity_mask(batch: Batch, entity: int) -> torch.Tensor:
    entity_idx = getattr(batch, "entity_idx", None)
    if entity_idx is None or entity_idx.numel() != batch.z.numel():
        raise ValueError("complex pooling requires batch.entity_idx with one value per node")
    return entity_idx.long() == int(entity)


def pool(node_embeddings: torch.Tensor, batch: Batch, pool_mask: torch.Tensor, modes: Iterable[str]) -> Dict[str, torch.Tensor]:
    """Chain embeddings ``{mode: [num_graphs, D]}``; fails if a chain has nothing to pool."""
    num_graphs = int(batch.num_graphs)
    graph = batch.batch.long()
    out = {}
    for mode in modes:
        if mode == "mean_pocket":
            pooled, counts = _masked_mean(
                node_embeddings, graph, pool_mask & _entity_mask(batch, PROTEIN_ENTITY), num_graphs)
        elif mode == "mean_ligand":
            pooled, counts = _masked_mean(
                node_embeddings, graph, pool_mask & _entity_mask(batch, LIGAND_ENTITY), num_graphs)
        elif mode == "concat_pocket_ligand":
            # The protein part outnumbers the ligand roughly 9:1, so a single
            # mean over the complex would drown the ligand. Both parts are
            # already contextualized by the joint ESA attention.
            protein, protein_counts = _masked_mean(
                node_embeddings, graph, pool_mask & _entity_mask(batch, PROTEIN_ENTITY), num_graphs)
            ligand, ligand_counts = _masked_mean(
                node_embeddings, graph, pool_mask & _entity_mask(batch, LIGAND_ENTITY), num_graphs)
            pooled = torch.cat((protein, ligand), dim=-1)
            counts = torch.minimum(protein_counts, ligand_counts)
        elif mode == "mean_atoms":
            pooled, counts = _masked_mean(node_embeddings, graph, pool_mask, num_graphs)
        elif mode == "ca":
            ca = pool_mask & (batch.template_anchor_order.long() == CA_ANCHOR_ORDER)
            pooled, counts = _masked_mean(node_embeddings, graph, ca, num_graphs)
        elif mode == "mean_residues":
            residue = _residue_ids(batch)
            num_residues = int(residue.max()) + 1
            residue_mean, residue_counts = _masked_mean(node_embeddings, residue, pool_mask, num_residues)
            residue_graph = torch.zeros(num_residues, dtype=torch.long, device=graph.device)
            residue_graph[residue] = graph
            pooled, counts = _masked_mean(residue_mean, residue_graph, residue_counts > 0, num_graphs)
        else:
            raise ValueError(f"unknown pooling {mode!r}; choose from {ALL_POOLINGS}")
        if bool((counts == 0).any()):
            empty = torch.where(counts == 0)[0].tolist()
            raise ValueError(f"pooling {mode!r}: graphs {empty} have no unmasked real atom")
        out[mode] = pooled
    return out


@torch.no_grad()
def embed_batch(model: PretrainingESAModel, samples: List[Data], device, modes=POOLINGS) -> Dict[str, torch.Tensor]:
    """Frozen-encoder chain embeddings for a list of runtime samples (float32, CPU)."""
    batch = collate(samples).to(device)
    node_embeddings, pool_mask = encode(model, batch)
    return {mode: value.cpu() for mode, value in pool(node_embeddings, batch, pool_mask, modes).items()}
