"""Pure coordinate corruption for protein/RNA template slots.

The function in this module never mutates its input batch. It preserves a
bitwise clone of the loaded coordinates as ``clean_pos`` and returns a second
``input_pos`` tensor. Fully masked side/base slots receive the complete fixed
local template; visible residues keep every real atom in place while only
their virtual padding moves to its corresponding canonical slot.
"""

from dataclasses import dataclass
import time
from typing import Iterable, Tuple

import torch
from torch_cluster import knn, radius_graph

from core.masked_slot_mapping import (
    MaskedSlotMapping,
    QuarantinedSlotBlock,
    map_all_template_slots,
    map_masked_template_slots,
)
from core.virtual_atom_templates import (
    protein_sidechain_template,
    rna_base_template,
)
from data_loading.atom_position_vocab import (
    VIRTUAL_ATOMIC_NUMBER,
    is_virtual_position_code,
)


RNA_CHIRALITY_EPSILON_ANGSTROM = 1.0e-6
FRAME_EPSILON_ANGSTROM = 1.0e-6


@dataclass(frozen=True)
class AppliedTemplateBlock:
    graph_index: int
    block_id: int
    domain: str
    node_indices: torch.Tensor


@dataclass(frozen=True)
class RelocatedContextVirtualBlock:
    graph_index: int
    block_id: int
    domain: str
    node_indices: torch.Tensor


@dataclass(frozen=True)
class CoordinateQuarantinedBlock:
    graph_index: int
    block_id: int
    domain: str
    reason: str


@dataclass(frozen=True)
class MaskedCoordinateCorruption:
    clean_pos: torch.Tensor
    input_pos: torch.Tensor
    template_node_mask: torch.Tensor
    edge_rebuild_node_mask: torch.Tensor
    target_template_node_mask: torch.Tensor
    context_virtual_node_mask: torch.Tensor
    applied_blocks: Tuple[AppliedTemplateBlock, ...]
    context_virtual_blocks: Tuple[RelocatedContextVirtualBlock, ...]
    mapping_quarantines: Tuple[QuarantinedSlotBlock, ...]
    coordinate_quarantines: Tuple[CoordinateQuarantinedBlock, ...]
    context_mapping_quarantines: Tuple[QuarantinedSlotBlock, ...]
    context_coordinate_quarantines: Tuple[CoordinateQuarantinedBlock, ...]


@dataclass(frozen=True)
class RebuiltTemplateEdges:
    edge_index: torch.Tensor
    edge_attr: torch.Tensor | None
    rebuilt_edge_mask: torch.Tensor
    removed_edges: int
    added_edges: int
    rebuild_ms: float


def _batched_global_templates(
    *,
    clean_pos: torch.Tensor,
    mapped_blocks: Tuple,
) -> tuple[dict, dict]:
    """Build local-frame templates on the batch device without full-node CPU copies."""
    templates = {}
    reasons = {}

    for domain in ("protein", "rna"):
        domain_blocks = [block for block in mapped_blocks if block.domain == domain]
        if not domain_blocks:
            continue

        anchor_rows = []
        keys = []
        for block in domain_blocks:
            key = (block.graph_index, block.block_id, block.domain)
            if block.anchor_node_indices is None:
                raise RuntimeError("template mapping is missing numeric anchor nodes")
            anchor_rows.append(
                clean_pos[block.anchor_node_indices.to(clean_pos.device)]
            )
            keys.append(key)

        anchors = torch.stack(anchor_rows)
        finite = torch.isfinite(anchors).all(dim=(1, 2))
        if domain == "protein":
            n, origin, c = anchors.unbind(dim=1)
            x_vector = c - origin
            y_source = n - origin
        else:
            c2, origin, c4, o2 = anchors.unbind(dim=1)
            x_vector = c4 - origin
            y_source = c2 - origin

        x_norm = torch.linalg.vector_norm(x_vector, dim=1)
        y_source_norm = torch.linalg.vector_norm(y_source, dim=1)
        safe_x_norm = x_norm.clamp_min(FRAME_EPSILON_ANGSTROM)
        x_axis = x_vector / safe_x_norm[:, None]
        y_perpendicular = y_source - (
            (y_source * x_axis).sum(dim=1, keepdim=True) * x_axis
        )
        perpendicular_norm = torch.linalg.vector_norm(y_perpendicular, dim=1)
        safe_perpendicular_norm = perpendicular_norm.clamp_min(
            FRAME_EPSILON_ANGSTROM
        )
        y_axis = y_perpendicular / safe_perpendicular_norm[:, None]
        z_axis = torch.linalg.cross(x_axis, y_axis, dim=1)
        z_axis = z_axis / torch.linalg.vector_norm(
            z_axis, dim=1
        ).clamp_min(FRAME_EPSILON_ANGSTROM)[:, None]
        y_axis = torch.linalg.cross(z_axis, x_axis, dim=1)
        basis = torch.stack((x_axis, y_axis, z_axis), dim=2)

        if domain == "protein":
            local_template = protein_sidechain_template(
                dtype=anchors.dtype, device=anchors.device
            )
        else:
            local_template = rna_base_template(
                dtype=anchors.dtype, device=anchors.device
            )
        global_templates = (
            local_template.unsqueeze(0) @ basis.transpose(1, 2)
            + origin[:, None, :]
        )
        output_finite = torch.isfinite(global_templates).all(dim=(1, 2))

        for index, key in enumerate(keys):
            reason = None
            if not bool(finite[index]):
                reason = "frame:nonfinite_anchor"
            elif x_norm[index] <= FRAME_EPSILON_ANGSTROM:
                reason = "frame:coincident_anchor"
            elif y_source_norm[index] <= FRAME_EPSILON_ANGSTROM:
                reason = "frame:coincident_anchor"
            elif perpendicular_norm[index] <= FRAME_EPSILON_ANGSTROM:
                reason = "frame:collinear_anchors"
            elif domain == "rna":
                signed_o2_z = torch.dot(o2[index] - origin[index], z_axis[index])
                if signed_o2_z <= RNA_CHIRALITY_EPSILON_ANGSTROM:
                    reason = "noncanonical_rna_orientation"
            if reason is None and not bool(output_finite[index]):
                reason = "nonfinite_template_coordinate"

            if reason is None:
                templates[key] = global_templates[index]
            else:
                reasons[key] = reason

    return templates, reasons


def build_masked_coordinate_corruption(
    batch: object,
    *,
    mapping: MaskedSlotMapping | None = None,
    domains: Iterable[str] = ("protein", "rna"),
    relocate_context_virtuals: bool = True,
) -> MaskedCoordinateCorruption:
    """Return clean/input coordinate views without modifying ``batch``.

    Selected-target failures are carried through from Stage 3.3 and remain
    fatal at integration time. Invalid visible legacy blocks are left
    unchanged and reported separately as context quarantines.
    """
    pos = getattr(batch, "pos", None)
    if pos is None or not isinstance(pos, torch.Tensor):
        raise ValueError("batch.pos tensor is required for coordinate corruption")
    if pos.ndim != 2 or tuple(pos.shape[1:]) != (3,):
        raise ValueError(f"batch.pos must have shape [N, 3], got {tuple(pos.shape)}")
    if not pos.is_floating_point():
        raise TypeError("batch.pos must use a floating-point dtype")

    num_nodes = pos.shape[0]
    if getattr(batch, "z", None) is None or batch.z.numel() != num_nodes:
        raise ValueError("batch.z must have one value per coordinate")
    if mapping is None:
        mapping = map_masked_template_slots(batch, domains=domains)

    clean_pos = pos.clone()
    input_pos = pos.clone()
    target_template_node_mask = torch.zeros(
        num_nodes, dtype=torch.bool, device=pos.device
    )
    context_virtual_node_mask = torch.zeros_like(target_template_node_mask)
    enabled_domains = {str(domain).lower() for domain in domains}
    if not relocate_context_virtuals:
        original_types = getattr(batch, "original_types", None)
        if original_types is None or original_types.numel() != num_nodes:
            raise ValueError(
                "prepositioned visible virtuals require batch.original_types"
            )
        graph_types = getattr(batch, "dataset_type", None)
        if isinstance(graph_types, (list, tuple)):
            graph_types = [str(value).lower() for value in graph_types]
        else:
            graph_count = int(getattr(batch, "num_graphs", 1) or 1)
            graph_types = [str(graph_types).lower()] * graph_count
        enabled_graph = torch.tensor(
            [
                (graph_type in ("pdb", "protein") and "protein" in enabled_domains)
                or (graph_type == "rna" and "rna" in enabled_domains)
                for graph_type in graph_types
            ],
            dtype=torch.bool,
            device=pos.device,
        )
        context_virtual_node_mask = (
            (original_types.to(pos.device) == VIRTUAL_ATOMIC_NUMBER)
            & ~batch.mlm_mask.bool()
            & enabled_graph[batch.batch]
        )
    applied_blocks = []
    context_virtual_blocks = []
    coordinate_quarantines = []
    context_coordinate_quarantines = []
    target_node_parts = []
    context_node_parts = []
    destination_node_parts = []
    destination_coordinate_parts = []

    if relocate_context_virtuals:
        all_mapping = map_all_template_slots(batch, domains=domains)
        frame_blocks = all_mapping.mapped_blocks
    else:
        all_mapping = MaskedSlotMapping(
            mapped_blocks=(),
            quarantined_blocks=(),
            node_mask=torch.zeros_like(target_template_node_mask),
        )
        frame_blocks = mapping.mapped_blocks
    global_templates, global_template_reasons = _batched_global_templates(
        clean_pos=clean_pos,
        mapped_blocks=frame_blocks,
    )

    selected_block_keys = {
        (block.graph_index, block.block_id, block.domain)
        for block in mapping.mapped_blocks
    }
    for mapped_block in mapping.mapped_blocks:
        block_key = (
            mapped_block.graph_index,
            mapped_block.block_id,
            mapped_block.domain,
        )
        global_template = global_templates.get(block_key)
        reason = global_template_reasons.get(block_key)
        if reason is not None:
            coordinate_quarantines.append(
                CoordinateQuarantinedBlock(
                    graph_index=mapped_block.graph_index,
                    block_id=mapped_block.block_id,
                    domain=mapped_block.domain,
                    reason=reason,
                )
            )
            continue

        slot_nodes = mapped_block.node_indices
        if slot_nodes.numel() != global_template.shape[0]:
            raise ValueError(
                f"{mapped_block.domain} mapped slot count changed after validation: "
                f"{slot_nodes.numel()} != {global_template.shape[0]}"
            )
        target_node_parts.append(slot_nodes)
        destination_node_parts.append(slot_nodes)
        destination_coordinate_parts.append(global_template)
        applied_blocks.append(
            AppliedTemplateBlock(
                graph_index=mapped_block.graph_index,
                block_id=mapped_block.block_id,
                domain=mapped_block.domain,
                node_indices=slot_nodes,
            )
        )

    for mapped_block in all_mapping.mapped_blocks:
        block_key = (
            mapped_block.graph_index,
            mapped_block.block_id,
            mapped_block.domain,
        )
        if block_key in selected_block_keys:
            continue

        if mapped_block.virtual_slot_mask is None:
            virtual_local_indices = torch.tensor(
                [
                    index
                    for index, code in enumerate(mapped_block.pos_codes)
                    if is_virtual_position_code(code)
                ],
                dtype=torch.long,
            )
        else:
            virtual_local_indices = torch.where(mapped_block.virtual_slot_mask)[0]
        if virtual_local_indices.numel() == 0:
            continue

        slot_nodes = mapped_block.node_indices
        virtual_nodes = slot_nodes[virtual_local_indices]
        context_node_parts.append(virtual_nodes)
        if relocate_context_virtuals:
            global_template = global_templates.get(block_key)
            reason = global_template_reasons.get(block_key)
            if reason is not None:
                context_node_parts.pop()
                context_coordinate_quarantines.append(
                    CoordinateQuarantinedBlock(
                        graph_index=mapped_block.graph_index,
                        block_id=mapped_block.block_id,
                        domain=mapped_block.domain,
                        reason=reason,
                    )
                )
                continue
            destination_node_parts.append(virtual_nodes)
            destination_coordinate_parts.append(global_template[virtual_local_indices])
        context_virtual_blocks.append(
            RelocatedContextVirtualBlock(
                graph_index=mapped_block.graph_index,
                block_id=mapped_block.block_id,
                domain=mapped_block.domain,
                node_indices=virtual_nodes,
            )
        )

    if not relocate_context_virtuals and bool(context_virtual_node_mask.any()):
        context_indices = torch.where(context_virtual_node_mask)[0]
        context_pairs = torch.stack(
            (batch.batch[context_indices], batch.block_idx[context_indices]), dim=1
        )
        for graph_index, block_id in torch.unique(context_pairs, dim=0).cpu().tolist():
            pair_mask = (
                (batch.batch[context_indices] == graph_index)
                & (batch.block_idx[context_indices] == block_id)
            )
            node_indices = context_indices[pair_mask]
            graph_type = graph_types[graph_index]
            domain = "rna" if graph_type == "rna" else "protein"
            context_virtual_blocks.append(
                RelocatedContextVirtualBlock(
                    graph_index=graph_index,
                    block_id=block_id,
                    domain=domain,
                    node_indices=node_indices.detach().cpu(),
                )
            )

    if target_node_parts:
        target_nodes = torch.cat(target_node_parts).to(pos.device)
        target_template_node_mask[target_nodes] = True
    if context_node_parts:
        context_nodes = torch.cat(context_node_parts).to(pos.device)
        context_virtual_node_mask[context_nodes] = True
    if destination_node_parts:
        destination_nodes = torch.cat(destination_node_parts).to(pos.device)
        destination_coordinates = torch.cat(destination_coordinate_parts).to(
            device=pos.device, dtype=pos.dtype
        )
        input_pos[destination_nodes] = destination_coordinates

    template_node_mask = target_template_node_mask | context_virtual_node_mask
    edge_rebuild_node_mask = target_template_node_mask.clone()
    if relocate_context_virtuals:
        edge_rebuild_node_mask |= context_virtual_node_mask

    return MaskedCoordinateCorruption(
        clean_pos=clean_pos,
        input_pos=input_pos,
        template_node_mask=template_node_mask,
        edge_rebuild_node_mask=edge_rebuild_node_mask,
        target_template_node_mask=target_template_node_mask,
        context_virtual_node_mask=context_virtual_node_mask,
        applied_blocks=tuple(applied_blocks),
        context_virtual_blocks=tuple(context_virtual_blocks),
        mapping_quarantines=mapping.quarantined_blocks,
        coordinate_quarantines=tuple(coordinate_quarantines),
        context_mapping_quarantines=all_mapping.quarantined_blocks,
        context_coordinate_quarantines=tuple(context_coordinate_quarantines),
    )


def rebuild_template_incident_edges(
    batch: object,
    input_pos: torch.Tensor,
    template_node_mask: torch.Tensor,
    *,
    cutoff: float = 5.0,
    max_neighbors: int = 48,
) -> RebuiltTemplateEdges:
    """Replace only edges touching template nodes with spatial input edges."""
    started = time.perf_counter()
    edge_index = batch.edge_index
    edge_attr = getattr(batch, "edge_attr", None)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("batch.edge_index must have shape [2, E]")
    if cutoff <= 0 or max_neighbors <= 0:
        raise ValueError("cutoff and max_neighbors must be positive")
    if edge_attr is not None and edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError("batch.edge_attr must align with batch.edge_index")

    touches_template = (
        template_node_mask[edge_index[0]] | template_node_mask[edge_index[1]]
    )
    keep = ~touches_template
    kept_edges = edge_index[:, keep]
    kept_attr = edge_attr[keep] if edge_attr is not None else None
    num_nodes = input_pos.shape[0]
    moved_nodes = int(template_node_mask.sum().item())
    use_targeted_rebuild = moved_nodes * 4 <= num_nodes
    if moved_nodes and num_nodes > 1 and use_targeted_rebuild:
        template_nodes = torch.where(template_node_mask)[0]
        assignments = knn(
            input_pos,
            input_pos[template_nodes],
            k=int(max_neighbors) + 1,
            batch_x=batch.batch,
            batch_y=batch.batch[template_nodes],
        )
        sources = template_nodes[assignments[0]]
        targets = assignments[1]
        distances = torch.linalg.vector_norm(
            input_pos[sources] - input_pos[targets], dim=1
        )
        valid = (sources != targets) & (distances <= float(cutoff))
        sources = sources[valid]
        targets = targets[valid]
        candidate_edges = torch.stack(
            (torch.cat((sources, targets)), torch.cat((targets, sources))), dim=0
        )
    elif moved_nodes and num_nodes > 1:
        candidate_edges = radius_graph(
            input_pos,
            r=float(cutoff),
            batch=batch.batch,
            loop=False,
            max_num_neighbors=int(max_neighbors),
        )
        candidate_touches_template = (
            template_node_mask[candidate_edges[0]]
            | template_node_mask[candidate_edges[1]]
        )
        candidate_edges = candidate_edges[:, candidate_touches_template]

    if moved_nodes and num_nodes > 1:
        keys = candidate_edges[0] * num_nodes + candidate_edges[1]
        unique_keys = torch.unique(keys, sorted=True)
        new_edges = torch.stack(
            (unique_keys // num_nodes, unique_keys % num_nodes), dim=0
        )
    else:
        new_edges = torch.empty(
            (2, 0), dtype=torch.long, device=edge_index.device
        )

    rebuilt_edge_index = torch.cat((kept_edges, new_edges), dim=1)
    if edge_attr is None:
        rebuilt_edge_attr = None
    else:
        new_distances = torch.linalg.vector_norm(
            input_pos[new_edges[0]] - input_pos[new_edges[1]], dim=1
        )
        if edge_attr.ndim == 1:
            new_attr = new_distances.to(edge_attr.dtype)
        elif edge_attr.ndim == 2 and edge_attr.shape[1] == 1:
            new_attr = new_distances.to(edge_attr.dtype).unsqueeze(1)
        else:
            raise ValueError(
                "template edge rebuild supports scalar distance edge_attr only"
            )
        rebuilt_edge_attr = torch.cat((kept_attr, new_attr), dim=0)

    rebuilt_edge_mask = torch.zeros(
        rebuilt_edge_index.shape[1], dtype=torch.bool, device=edge_index.device
    )
    rebuilt_edge_mask[kept_edges.shape[1]:] = True
    return RebuiltTemplateEdges(
        edge_index=rebuilt_edge_index,
        edge_attr=rebuilt_edge_attr,
        rebuilt_edge_mask=rebuilt_edge_mask,
        removed_edges=int(touches_template.sum().item()),
        added_edges=int(new_edges.shape[1]),
        rebuild_ms=(time.perf_counter() - started) * 1000.0,
    )


def apply_masked_slot_template_(batch: object, config: object) -> dict:
    """Apply the enabled template feature to one batch in place."""
    template_started_at = time.perf_counter()
    settings = getattr(config, "masked_slot_template", None) or {}
    if not settings.get("enabled", False):
        return {}
    if hasattr(batch, "template_node_mask"):
        return getattr(batch, "virtual_template_stats", {})
    if "coordinate_denoising" in getattr(config, "pretraining_tasks", ()):
        raise ValueError(
            "masked slot templates and coordinate_denoising cannot share a run"
        )

    domains = tuple(
        domain for domain in ("protein", "rna") if settings.get(domain, False)
    )
    coordinate_result = build_masked_coordinate_corruption(
        batch,
        domains=domains,
        relocate_context_virtuals=not settings.get(
            "visible_virtuals_prepositioned", False
        ),
    )

    # RNA source audits deliberately retain a small number of malformed or
    # opposite-handed nucleotide blocks as visible molecular context.  They
    # cannot define the canonical template frame and therefore must never be
    # MLM targets.  Veto only the audited RNA failure modes here; unexpected
    # RNA failures and every protein failure remain fatal below.
    allowed_mapping_reasons = frozenset(
        ("missing_backbone_position", "duplicated_backbone_position")
    )
    allowed_coordinate_reasons = frozenset(("noncanonical_rna_orientation",))
    excluded_target_keys = set()
    unexpected_mapping = []
    unexpected_coordinate = []
    for item in coordinate_result.mapping_quarantines:
        if (
            item.domain == "rna"
            and set(item.reasons)
            and set(item.reasons) <= allowed_mapping_reasons
        ):
            excluded_target_keys.add((item.graph_index, item.block_id))
        else:
            unexpected_mapping.append(item)
    for item in coordinate_result.coordinate_quarantines:
        if item.domain == "rna" and item.reason in allowed_coordinate_reasons:
            excluded_target_keys.add((item.graph_index, item.block_id))
        else:
            unexpected_coordinate.append(item)

    excluded_target_nodes = 0
    if excluded_target_keys and not unexpected_mapping and not unexpected_coordinate:
        quarantine_node_mask = torch.zeros_like(batch.mlm_mask, dtype=torch.bool)
        for graph_index, block_id in excluded_target_keys:
            quarantine_node_mask |= (
                (batch.batch == graph_index) & (batch.block_idx == block_id)
            )
        excluded_target_nodes = int(quarantine_node_mask.sum().item())
        batch.mlm_mask[quarantine_node_mask] = False
        batch.z[quarantine_node_mask] = batch.original_types[quarantine_node_mask]
        batch.masked_types[quarantine_node_mask] = batch.original_types[
            quarantine_node_mask
        ]
        coordinate_result = build_masked_coordinate_corruption(
            batch,
            domains=domains,
            relocate_context_virtuals=not settings.get(
                "visible_virtuals_prepositioned", False
            ),
        )

    if coordinate_result.mapping_quarantines or coordinate_result.coordinate_quarantines:
        mapping_reasons = [
            reason
            for item in coordinate_result.mapping_quarantines
            for reason in item.reasons
        ]
        coordinate_reasons = [
            item.reason for item in coordinate_result.coordinate_quarantines
        ]
        raise RuntimeError(
            "masked slot template rejected selected blocks: "
            + ", ".join(mapping_reasons + coordinate_reasons)
        )
    if not bool(coordinate_result.template_node_mask.any()):
        # A cross-modal batch may legitimately contain no graph from either
        # enabled template domain (for example an UniMol-only late batch).
        # Malformed selected protein/RNA blocks are handled above as explicit
        # quarantines; an empty, quarantine-free mapping is a true no-op.
        return {
            "templated_blocks": 0,
            "templated_nodes": 0,
            "target_template_nodes": 0,
            "excluded_quarantined_target_blocks": len(excluded_target_keys),
            "excluded_quarantined_target_nodes": excluded_target_nodes,
            "context_virtual_blocks": 0,
            "context_virtual_nodes": 0,
            "context_mapping_quarantines": 0,
            "context_coordinate_quarantines": 0,
            "removed_edges": 0,
            "added_edges": 0,
            "excluded_short_edges": 0,
            "excluded_long_pairs": 0,
            "edge_rebuild_ms": 0.0,
            "template_total_ms": (time.perf_counter() - template_started_at) * 1000.0,
        }

    rebuilt = rebuild_template_incident_edges(
        batch,
        coordinate_result.input_pos,
        coordinate_result.edge_rebuild_node_mask,
        cutoff=float(getattr(config, "cutoff_distance", 5.0)),
        max_neighbors=int(getattr(config, "max_spatial_neighbors", 48)),
    )
    batch.clean_pos = coordinate_result.clean_pos
    batch.input_pos = coordinate_result.input_pos
    batch.pos = coordinate_result.input_pos
    batch.template_node_mask = coordinate_result.template_node_mask
    batch.template_edge_rebuild_node_mask = coordinate_result.edge_rebuild_node_mask
    batch.target_template_node_mask = coordinate_result.target_template_node_mask
    batch.context_virtual_node_mask = coordinate_result.context_virtual_node_mask
    batch.edge_index = rebuilt.edge_index
    if rebuilt.edge_attr is not None:
        batch.edge_attr = rebuilt.edge_attr
    batch.template_rebuilt_edge_mask = rebuilt.rebuilt_edge_mask
    templated_nodes = int(coordinate_result.template_node_mask.sum().item())
    total_nodes = int(coordinate_result.template_node_mask.numel())
    eligible_nodes = total_nodes - templated_nodes
    batch.virtual_template_stats = {
        "templated_blocks": len(coordinate_result.applied_blocks),
        "templated_nodes": templated_nodes,
        "target_template_nodes": int(
            coordinate_result.target_template_node_mask.sum().item()
        ),
        "excluded_quarantined_target_blocks": len(excluded_target_keys),
        "excluded_quarantined_target_nodes": excluded_target_nodes,
        "context_virtual_blocks": len(
            coordinate_result.context_virtual_blocks
        ),
        "context_virtual_nodes": int(
            coordinate_result.context_virtual_node_mask.sum().item()
        ),
        "context_mapping_quarantines": len(
            coordinate_result.context_mapping_quarantines
        ),
        "context_coordinate_quarantines": len(
            coordinate_result.context_coordinate_quarantines
        ),
        "removed_edges": rebuilt.removed_edges,
        "added_edges": rebuilt.added_edges,
        "excluded_short_edges": int((
            coordinate_result.template_node_mask[rebuilt.edge_index[0]]
            | coordinate_result.template_node_mask[rebuilt.edge_index[1]]
        ).sum().item()),
        "excluded_long_pairs": (
            total_nodes * (total_nodes - 1) // 2
            - eligible_nodes * (eligible_nodes - 1) // 2
        ),
        "edge_rebuild_ms": rebuilt.rebuild_ms,
        "template_total_ms": (time.perf_counter() - template_started_at) * 1000.0,
    }
    return batch.virtual_template_stats
