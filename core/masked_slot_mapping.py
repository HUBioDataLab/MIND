"""Read-only runtime mapping from masked blocks to template slot nodes.

This module does not alter a batch.  It derives protein sidechain and RNA base
slot indices from existing ``batch``, ``block_idx``, ``mlm_mask``, and
``pos_code`` fields, and quarantines malformed selected blocks with explicit
reason codes.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch

from data_loading.atom_position_vocab import (
    PROTEIN_BACKBONE_ATOMS,
    PROTEIN_SIDECHAIN_ATOM_PRIORITY,
    PROTEIN_TERMINAL_ATOMS,
    RNA_BACKBONE_ATOMS,
    RNA_BASE_ATOM_PRIORITY,
    RNA_TERMINAL_ATOMS,
    is_virtual_position_code,
    normalize_rna_position_code,
)


PROTEIN_GRAPH_TYPES = frozenset(("pdb", "protein"))
RNA_GRAPH_TYPES = frozenset(("rna",))


@dataclass(frozen=True)
class MappedSlotBlock:
    graph_index: int
    block_id: int
    domain: str
    node_indices: torch.Tensor
    pos_codes: Tuple[str, ...]
    anchor_node_indices: torch.Tensor | None = None
    virtual_slot_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class QuarantinedSlotBlock:
    graph_index: int
    block_id: int
    domain: str
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class MaskedSlotMapping:
    mapped_blocks: Tuple[MappedSlotBlock, ...]
    quarantined_blocks: Tuple[QuarantinedSlotBlock, ...]
    node_mask: torch.Tensor


_TEMPLATE_REASON_BITS = {
    "missing_backbone_position": 1 << 0,
    "duplicated_backbone_position": 1 << 1,
    "terminal_atom_in_fixed_block": 1 << 2,
    "wrong_slot_count": 1 << 3,
    "noncanonical_backbone_slot_partition": 1 << 4,
    "duplicated_slot_position": 1 << 5,
    "unsupported_real_slot_position": 1 << 6,
    "real_slot_after_virtual_padding": 1 << 7,
    "virtual_slot_index_mismatch": 1 << 8,
}


def attach_runtime_template_metadata(data: object) -> object:
    """Attach numeric template roles without changing the persisted ``.pt`` cache."""
    num_nodes = int(data.z.numel())
    domain_value = getattr(data, "dataset_type", "")
    if isinstance(domain_value, (list, tuple)):
        domain_value = domain_value[0] if domain_value else ""
    domain = _domain_for_graph_type(str(domain_value).lower())

    domain_ids = torch.zeros(num_nodes, dtype=torch.int8)
    slot_orders = torch.full((num_nodes,), -1, dtype=torch.int8)
    anchor_orders = torch.full((num_nodes,), -1, dtype=torch.int8)
    virtual_slots = torch.zeros(num_nodes, dtype=torch.bool)
    reason_bits = torch.zeros(num_nodes, dtype=torch.int16)
    if domain is None or num_nodes == 0:
        data.template_domain_id = domain_ids
        data.template_slot_order = slot_orders
        data.template_anchor_order = anchor_orders
        data.template_virtual_slot = virtual_slots
        data.template_reason_bits = reason_bits
        return data

    pos_codes = flatten_pos_codes(data.pos_code, num_nodes)
    block_indices = data.block_idx.detach().cpu().tolist()
    grouped_nodes = {}
    for node_index, block_id in enumerate(block_indices):
        grouped_nodes.setdefault(int(block_id), []).append(node_index)

    domain_id = 1 if domain == "protein" else 2
    anchor_codes = (
        ("N", "CA", "C")
        if domain == "protein"
        else ("C2'", "C3'", "C4'", "O2'")
    )
    for block_nodes in grouped_nodes.values():
        domain_ids[block_nodes] = domain_id
        block_codes = tuple(pos_codes[index] for index in block_nodes)
        slot_codes, slot_local_indices, reasons = _validate_block_slots(
            block_codes, domain=domain
        )
        bitmask = 0
        for reason in reasons:
            bitmask |= _TEMPLATE_REASON_BITS[reason]
        if bitmask:
            reason_bits[block_nodes] = bitmask

        normalized_codes = (
            tuple(code.strip() for code in block_codes)
            if domain == "protein"
            else tuple(normalize_rna_position_code(code) for code in block_codes)
        )
        for anchor_order, anchor_code in enumerate(anchor_codes):
            matches = [
                local_index
                for local_index, code in enumerate(normalized_codes)
                if code == anchor_code
            ]
            if len(matches) == 1:
                anchor_orders[block_nodes[matches[0]]] = anchor_order
        for slot_order, (local_index, code) in enumerate(
            zip(slot_local_indices, slot_codes)
        ):
            node_index = block_nodes[local_index]
            slot_orders[node_index] = slot_order
            virtual_slots[node_index] = is_virtual_position_code(code)

    data.template_domain_id = domain_ids
    data.template_slot_order = slot_orders
    data.template_anchor_order = anchor_orders
    data.template_virtual_slot = virtual_slots
    data.template_reason_bits = reason_bits
    return data


def _decode_reason_bits(value: int) -> Tuple[str, ...]:
    return tuple(
        reason for reason, bit in _TEMPLATE_REASON_BITS.items() if value & bit
    )


def flatten_pos_codes(raw_pos_codes: object, expected_nodes: int) -> Tuple[str, ...]:
    """Flatten PyG's list-of-lists representation without changing order."""
    if isinstance(raw_pos_codes, str) or raw_pos_codes is None:
        raise ValueError("batch.pos_code must be a flat or graph-nested sequence")
    if not isinstance(raw_pos_codes, (list, tuple)):
        raw_pos_codes = list(raw_pos_codes)

    if raw_pos_codes and isinstance(raw_pos_codes[0], (list, tuple)):
        flattened = tuple(
            str(code) for graph_codes in raw_pos_codes for code in graph_codes
        )
    else:
        flattened = tuple(str(code) for code in raw_pos_codes)

    if len(flattened) != expected_nodes:
        raise ValueError(
            "batch.pos_code length does not match nodes: "
            f"{len(flattened)} != {expected_nodes}"
        )
    return flattened


def _graph_types(batch: object) -> list[str]:
    value = getattr(batch, "dataset_type", None)
    if value is None:
        raise ValueError("batch.dataset_type is required for masked slot mapping")
    if isinstance(value, (list, tuple)):
        return [str(item).lower() for item in value]
    num_graphs = int(getattr(batch, "num_graphs", 1) or 1)
    return [str(value).lower()] * num_graphs


def _domain_for_graph_type(graph_type: str) -> str | None:
    if graph_type in PROTEIN_GRAPH_TYPES:
        return "protein"
    if graph_type in RNA_GRAPH_TYPES:
        return "rna"
    return None


def _validate_block_slots(
    codes: Sequence[str],
    *,
    domain: str,
) -> tuple[Tuple[str, ...], Tuple[int, ...], Tuple[str, ...]]:
    if domain == "protein":
        normalized = tuple(str(code).strip() for code in codes)
        backbone = PROTEIN_BACKBONE_ATOMS
        terminal = frozenset(PROTEIN_TERMINAL_ATOMS)
        priority = PROTEIN_SIDECHAIN_ATOM_PRIORITY
        expected_slot_count = 10
    elif domain == "rna":
        normalized = tuple(normalize_rna_position_code(code) for code in codes)
        backbone = RNA_BACKBONE_ATOMS
        terminal = frozenset(RNA_TERMINAL_ATOMS)
        priority = RNA_BASE_ATOM_PRIORITY
        expected_slot_count = 11
    else:
        raise ValueError(f"unsupported slot-mapping domain: {domain}")

    reasons = []
    counts = Counter(normalized)
    if any(counts[code] == 0 for code in backbone):
        reasons.append("missing_backbone_position")
    if any(counts[code] > 1 for code in backbone):
        reasons.append("duplicated_backbone_position")
    if any(code in terminal for code in normalized):
        reasons.append("terminal_atom_in_fixed_block")

    slot_local_indices = tuple(
        index
        for index, code in enumerate(normalized)
        if code not in backbone and code not in terminal
    )
    stored_slot_codes = tuple(normalized[index] for index in slot_local_indices)
    if len(stored_slot_codes) != expected_slot_count:
        reasons.append("wrong_slot_count")

    expected_slot_start = len(normalized) - expected_slot_count
    if slot_local_indices != tuple(range(max(0, expected_slot_start), len(normalized))):
        reasons.append("noncanonical_backbone_slot_partition")
    if len(stored_slot_codes) != len(set(stored_slot_codes)):
        reasons.append("duplicated_slot_position")

    priority_index = {code: index for index, code in enumerate(priority)}
    real_pairs = [
        (local_index, code)
        for local_index, code in zip(slot_local_indices, stored_slot_codes)
        if not is_virtual_position_code(code)
    ]
    virtual_pairs = [
        (local_index, code)
        for local_index, code in zip(slot_local_indices, stored_slot_codes)
        if is_virtual_position_code(code)
    ]
    real_codes = [code for _, code in real_pairs]
    # Protein slot names are fixed by the 20 canonical residues. RNA also
    # admits supported modified bases whose real atom labels (for example
    # CM5, C5M, or S4) are not exhaustively enumerable here. The RNA adapter
    # has already constrained those blocks to 12 backbone + 11 ordered
    # base/V slots, so retain their adapter order instead of quarantining the
    # whole block solely because a modified atom name is absent from this
    # priority table.
    if domain == "protein" and any(
        code not in priority_index for code in real_codes
    ):
        reasons.append("unsupported_real_slot_position")

    virtual_seen = False
    for code in stored_slot_codes:
        if is_virtual_position_code(code):
            virtual_seen = True
        elif virtual_seen:
            reasons.append("real_slot_after_virtual_padding")
            break

    expected_virtual_codes = tuple(
        f"V{slot_number}"
        for slot_number in range(len(real_pairs) + 1, expected_slot_count + 1)
    )
    if tuple(code for _, code in virtual_pairs) != expected_virtual_codes:
        reasons.append("virtual_slot_index_mismatch")

    # Old caches and the current adapter can store recognized real atoms in
    # different orders. Canonicalize the returned *view* from pos_code while
    # leaving batch node order untouched. Virtual suffixes already encode
    # their absolute slot numbers and remain after all real slots.
    if any(code not in priority_index for code in real_codes):
        canonical_real_pairs = real_pairs
    else:
        canonical_real_pairs = sorted(
            real_pairs, key=lambda pair: priority_index[pair[1]]
        )
    canonical_pairs = canonical_real_pairs + virtual_pairs
    canonical_slot_local_indices = tuple(index for index, _ in canonical_pairs)
    canonical_slot_codes = tuple(code for _, code in canonical_pairs)

    return (
        canonical_slot_codes,
        canonical_slot_local_indices,
        tuple(dict.fromkeys(reasons)),
    )


def _map_template_slots(
    batch: object,
    *,
    domains: Iterable[str],
    masked_only: bool,
) -> MaskedSlotMapping:
    numeric_fields = (
        "template_domain_id",
        "template_slot_order",
        "template_anchor_order",
        "template_virtual_slot",
        "template_reason_bits",
    )
    if all(getattr(batch, name, None) is not None for name in numeric_fields):
        return _map_template_slots_numeric(
            batch, domains=domains, masked_only=masked_only
        )

    required = ["z", "batch", "block_idx", "pos_code"]
    if masked_only:
        required.append("mlm_mask")
    missing = [name for name in required if getattr(batch, name, None) is None]
    if missing:
        prefix = "masked" if masked_only else "all-block"
        raise ValueError(f"{prefix} slot mapping requires: " + ", ".join(missing))

    num_nodes = int(batch.z.numel())
    tensor_names = ["batch", "block_idx"] + (["mlm_mask"] if masked_only else [])
    for name in tensor_names:
        tensor = getattr(batch, name)
        if tensor.ndim != 1 or tensor.numel() != num_nodes:
            raise ValueError(f"batch.{name} must have one value per node")

    pos_codes = flatten_pos_codes(batch.pos_code, num_nodes)
    graph_types = _graph_types(batch)
    enabled_domains = {str(domain).lower() for domain in domains}
    unknown_domains = enabled_domains - {"protein", "rna"}
    if unknown_domains:
        raise ValueError(
            "unsupported template slot domains: "
            + ", ".join(sorted(unknown_domains))
        )

    graph_indices = batch.batch.detach().cpu().tolist()
    block_indices = batch.block_idx.detach().cpu().tolist()
    mlm_mask = (
        batch.mlm_mask.detach().cpu().bool().tolist() if masked_only else None
    )
    grouped_nodes = {}
    for node_index, (graph_index, block_id) in enumerate(
        zip(graph_indices, block_indices)
    ):
        graph_index = int(graph_index)
        graph_type = graph_types[graph_index] if graph_index < len(graph_types) else ""
        domain = _domain_for_graph_type(graph_type)
        if domain is None or domain not in enabled_domains:
            continue
        key = (graph_index, int(block_id), domain)
        grouped_nodes.setdefault(key, []).append(node_index)

    mapped_blocks = []
    quarantined_blocks = []
    template_node_mask = torch.zeros(
        num_nodes, dtype=torch.bool, device=batch.z.device
    )
    for (graph_index, block_id, domain), block_nodes in grouped_nodes.items():
        block_mask = [mlm_mask[index] for index in block_nodes] if masked_only else None
        if masked_only and not any(block_mask):
            continue

        reasons = []
        if masked_only and not all(block_mask):
            reasons.append("partially_masked_block")
        block_codes = tuple(pos_codes[index] for index in block_nodes)
        slot_codes, slot_local_indices, slot_reasons = _validate_block_slots(
            block_codes,
            domain=domain,
        )
        reasons.extend(slot_reasons)
        if reasons:
            quarantined_blocks.append(
                QuarantinedSlotBlock(
                    graph_index=graph_index,
                    block_id=block_id,
                    domain=domain,
                    reasons=tuple(dict.fromkeys(reasons)),
                )
            )
            continue

        slot_nodes = torch.tensor(
            [block_nodes[index] for index in slot_local_indices],
            dtype=torch.long,
        )
        normalized_block_codes = (
            tuple(code.strip() for code in block_codes)
            if domain == "protein"
            else tuple(normalize_rna_position_code(code) for code in block_codes)
        )
        anchor_codes = (
            ("N", "CA", "C")
            if domain == "protein"
            else ("C2'", "C3'", "C4'", "O2'")
        )
        anchor_nodes = torch.tensor(
            [
                block_nodes[normalized_block_codes.index(code)]
                for code in anchor_codes
            ],
            dtype=torch.long,
        )
        template_node_mask[slot_nodes.to(batch.z.device)] = True
        mapped_blocks.append(
            MappedSlotBlock(
                graph_index=graph_index,
                block_id=block_id,
                domain=domain,
                node_indices=slot_nodes,
                pos_codes=slot_codes,
                anchor_node_indices=anchor_nodes,
                virtual_slot_mask=torch.tensor(
                    [is_virtual_position_code(code) for code in slot_codes],
                    dtype=torch.bool,
                ),
            )
        )

    return MaskedSlotMapping(
        mapped_blocks=tuple(mapped_blocks),
        quarantined_blocks=tuple(quarantined_blocks),
        node_mask=template_node_mask,
    )


def _map_template_slots_numeric(
    batch: object,
    *,
    domains: Iterable[str],
    masked_only: bool,
) -> MaskedSlotMapping:
    """Map collated numeric roles with block loops but no node-level CPU copies."""
    num_nodes = int(batch.z.numel())
    enabled_domains = {str(domain).lower() for domain in domains}
    unknown_domains = enabled_domains - {"protein", "rna"}
    if unknown_domains:
        raise ValueError(
            "unsupported template slot domains: " + ", ".join(sorted(unknown_domains))
        )
    domain_ids = batch.template_domain_id.long()
    relevant = torch.zeros(num_nodes, dtype=torch.bool, device=batch.z.device)
    if "protein" in enabled_domains:
        relevant |= domain_ids == 1
    if "rna" in enabled_domains:
        relevant |= domain_ids == 2
    relevant_nodes = torch.where(relevant)[0]
    template_node_mask = torch.zeros(
        num_nodes, dtype=torch.bool, device=batch.z.device
    )
    if relevant_nodes.numel() == 0:
        return MaskedSlotMapping((), (), template_node_mask)

    ownership = torch.stack(
        (batch.batch[relevant_nodes], batch.block_idx[relevant_nodes]), dim=1
    )
    groups, inverse = torch.unique(
        ownership, dim=0, sorted=True, return_inverse=True
    )
    sort_order = torch.argsort(inverse, stable=True)
    sorted_nodes = relevant_nodes[sort_order]
    counts = torch.bincount(inverse)
    masked_counts = torch.zeros_like(counts)
    if masked_only:
        masked_counts.scatter_add_(
            0, inverse, batch.mlm_mask[relevant_nodes].long()
        )
    reason_by_group = torch.zeros_like(counts)
    reason_by_group.scatter_reduce_(
        0,
        inverse,
        batch.template_reason_bits[relevant_nodes].long(),
        reduce="amax",
        include_self=True,
    )
    first_positions = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=counts.device),
            torch.cumsum(counts, dim=0)[:-1],
        )
    )
    group_domains = domain_ids[sorted_nodes[first_positions]]
    group_metadata = torch.cat(
        (
            groups,
            counts[:, None],
            masked_counts[:, None],
            reason_by_group[:, None],
            group_domains[:, None],
        ),
        dim=1,
    ).detach().cpu().tolist()
    count_values = counts.detach().cpu().tolist()
    offsets = [0]
    for count in count_values:
        offsets.append(offsets[-1] + int(count))

    mapped_blocks = []
    quarantined_blocks = []
    for group_index, metadata in enumerate(group_metadata):
        block_nodes = sorted_nodes[offsets[group_index] : offsets[group_index + 1]]
        graph_index, block_id, block_count, masked_count, static_bits, domain_id = (
            int(value) for value in metadata
        )
        domain = "protein" if domain_id == 1 else "rna"
        if masked_only:
            if masked_count == 0:
                continue

        reasons = []
        if masked_only and masked_count != block_count:
            reasons.append("partially_masked_block")
        reasons.extend(_decode_reason_bits(static_bits))
        if reasons:
            quarantined_blocks.append(
                QuarantinedSlotBlock(
                    graph_index=graph_index,
                    block_id=block_id,
                    domain=domain,
                    reasons=tuple(dict.fromkeys(reasons)),
                )
            )
            continue

        slot_nodes = block_nodes[batch.template_slot_order[block_nodes] >= 0]
        slot_nodes = slot_nodes[
            torch.argsort(batch.template_slot_order[slot_nodes], stable=True)
        ]
        anchor_nodes = block_nodes[batch.template_anchor_order[block_nodes] >= 0]
        anchor_nodes = anchor_nodes[
            torch.argsort(batch.template_anchor_order[anchor_nodes], stable=True)
        ]
        expected_slots = 10 if domain == "protein" else 11
        expected_anchors = 3 if domain == "protein" else 4
        if slot_nodes.numel() != expected_slots or anchor_nodes.numel() != expected_anchors:
            raise RuntimeError(
                "numeric template metadata is inconsistent for "
                f"graph={graph_index}, block={block_id}, domain={domain}"
            )
        template_node_mask[slot_nodes] = True
        mapped_blocks.append(
            MappedSlotBlock(
                graph_index=graph_index,
                block_id=block_id,
                domain=domain,
                node_indices=slot_nodes,
                pos_codes=(),
                anchor_node_indices=anchor_nodes,
                virtual_slot_mask=batch.template_virtual_slot[slot_nodes].bool(),
            )
        )

    return MaskedSlotMapping(
        mapped_blocks=tuple(mapped_blocks),
        quarantined_blocks=tuple(quarantined_blocks),
        node_mask=template_node_mask,
    )


def map_masked_template_slots(
    batch: object,
    *,
    domains: Iterable[str] = ("protein", "rna"),
) -> MaskedSlotMapping:
    """Map fully MLM-masked protein/RNA blocks to exact template slot nodes."""
    return _map_template_slots(batch, domains=domains, masked_only=True)


def map_all_template_slots(
    batch: object,
    *,
    domains: Iterable[str] = ("protein", "rna"),
) -> MaskedSlotMapping:
    """Map every exact fixed-slot block, independent of MLM selection."""
    return _map_template_slots(batch, domains=domains, masked_only=False)
