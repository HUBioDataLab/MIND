"""Versioned compact PT schema for fixed-slot protein/RNA chunks without stored edges.

In the geometry-neutral production path, stored protein and RNA edges are
discarded and rebuilt at runtime, except protein Tier-1 CA-CA sequence edges
(``core.virtual_slot_graph._cached_structural_edge_mask``). The edge-free
layout therefore stores only node fields in compact dtypes and regenerates
Tier 1 from ``pos_code``/``block_idx``/``entity_idx`` at load time.

A chunk in this layout is saved as ``(data, slices, schema)``. Legacy loaders
that unpack ``data, slices`` fail loudly on it instead of reading a graph
without its edges. ``decode_edge_free_sample`` restores the exact in-memory
field set, dtypes and shapes of a legacy sample, so the rest of the pipeline
and mixed old/new collation are unchanged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Mapping, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce


PT_SCHEMA_NAME = "mind_pt_schema"
EDGE_FREE_LAYOUT = "edge_free_fixed_slot"
EDGE_FREE_SCHEMA_VERSION = 1
EDGE_FREE_DOMAINS = ("protein", "rna")
PROTEIN_SEQUENCE_TIER = "protein_ca_sequence"

COMPACT_NODE_DTYPES = {
    "z": torch.uint8,
    "block_idx": torch.int32,
    "entity_idx": torch.int16,
    "pos": torch.float32,
    "pos_code_ids": torch.uint8,
}
RUNTIME_INDEX_FIELDS = ("z", "block_idx", "entity_idx")
_MAX_POS_CODE_VOCABULARY = 256


def normalize_schema_domain(dataset_type: object) -> str:
    """Map persisted dataset types onto the edge-free domain names."""
    if isinstance(dataset_type, (list, tuple)):
        if len(dataset_type) != 1:
            raise ValueError(f"expected one dataset type, got {dataset_type!r}")
        dataset_type = dataset_type[0]
    value = str(dataset_type).lower()
    if value in ("pdb", "protein"):
        return "protein"
    if value == "rna":
        return "rna"
    raise ValueError(
        f"edge-free PT layout supports protein and RNA only, got {dataset_type!r}"
    )


def pos_code_vocabulary_sha256(vocabulary: list[str]) -> str:
    payload = json.dumps(list(vocabulary), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_edge_free_schema(
    domain: str,
    pos_code_vocabulary: list[str],
    sequence_neighbors_k: Optional[int],
    source: Mapping[str, object],
) -> dict:
    """Return the schema record stored in the PT and its metadata JSON."""
    domain = normalize_schema_domain(domain)
    vocabulary = [str(code) for code in pos_code_vocabulary]
    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("pos_code vocabulary contains duplicates")
    if not vocabulary or len(vocabulary) > _MAX_POS_CODE_VOCABULARY:
        raise ValueError(
            f"pos_code vocabulary size {len(vocabulary)} does not fit uint8 codes"
        )
    if domain == "protein":
        if sequence_neighbors_k is None or int(sequence_neighbors_k) <= 0:
            raise ValueError("protein edge-free schema requires sequence_neighbors_k > 0")
        sequence_neighbors_k = int(sequence_neighbors_k)
        regenerated = [PROTEIN_SEQUENCE_TIER]
    else:
        if sequence_neighbors_k is not None:
            raise ValueError("RNA edge-free schema has no sequence_neighbors_k")
        regenerated = []
    return {
        "name": PT_SCHEMA_NAME,
        "version": EDGE_FREE_SCHEMA_VERSION,
        "layout": EDGE_FREE_LAYOUT,
        "domain": domain,
        "stored_edge_tiers": [],
        "regenerated_edge_tiers": regenerated,
        "sequence_neighbors_k": sequence_neighbors_k,
        "node_dtypes": {
            name: str(dtype).replace("torch.", "")
            for name, dtype in COMPACT_NODE_DTYPES.items()
        },
        "pos_code_vocabulary": vocabulary,
        "pos_code_vocabulary_sha256": pos_code_vocabulary_sha256(vocabulary),
        "source": dict(source),
    }


def validate_edge_free_schema(schema: object, *, origin: str) -> dict:
    """Fail loudly on unknown, incomplete or internally inconsistent schemas."""
    if not isinstance(schema, Mapping):
        raise ValueError(f"{origin}: PT schema must be a mapping, got {type(schema)}")
    required = (
        "name", "version", "layout", "domain", "stored_edge_tiers",
        "regenerated_edge_tiers", "sequence_neighbors_k", "node_dtypes",
        "pos_code_vocabulary", "pos_code_vocabulary_sha256",
    )
    missing = [key for key in required if key not in schema]
    if missing:
        raise ValueError(f"{origin}: PT schema is missing {missing}")
    if schema["name"] != PT_SCHEMA_NAME:
        raise ValueError(f"{origin}: unknown PT schema name {schema['name']!r}")
    if schema["version"] != EDGE_FREE_SCHEMA_VERSION or schema["layout"] != EDGE_FREE_LAYOUT:
        raise ValueError(
            f"{origin}: unsupported PT schema version/layout "
            f"{schema['version']!r}/{schema['layout']!r}"
        )
    domain = normalize_schema_domain(schema["domain"])
    if list(schema["stored_edge_tiers"]) != []:
        raise ValueError(f"{origin}: edge-free schema declares stored edge tiers")
    expected_tiers = [PROTEIN_SEQUENCE_TIER] if domain == "protein" else []
    if list(schema["regenerated_edge_tiers"]) != expected_tiers:
        raise ValueError(
            f"{origin}: regenerated tiers {schema['regenerated_edge_tiers']!r} "
            f"do not match domain {domain}"
        )
    k = schema["sequence_neighbors_k"]
    if domain == "protein" and (not isinstance(k, int) or k <= 0):
        raise ValueError(f"{origin}: protein schema needs a positive sequence_neighbors_k")
    if domain == "rna" and k is not None:
        raise ValueError(f"{origin}: RNA schema must not set sequence_neighbors_k")
    expected_dtypes = {
        name: str(dtype).replace("torch.", "")
        for name, dtype in COMPACT_NODE_DTYPES.items()
    }
    if dict(schema["node_dtypes"]) != expected_dtypes:
        raise ValueError(
            f"{origin}: node dtypes {schema['node_dtypes']!r} != {expected_dtypes!r}"
        )
    vocabulary = list(schema["pos_code_vocabulary"])
    if pos_code_vocabulary_sha256(vocabulary) != schema["pos_code_vocabulary_sha256"]:
        raise ValueError(f"{origin}: pos_code vocabulary hash mismatch")
    if not vocabulary or len(vocabulary) > _MAX_POS_CODE_VOCABULARY:
        raise ValueError(f"{origin}: pos_code vocabulary size {len(vocabulary)} is invalid")
    return dict(schema)


def _ca_mask(pos_code: list) -> torch.Tensor:
    return torch.tensor([str(code).strip() == "CA" for code in pos_code], dtype=torch.bool)


def _coalesced(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return coalesce(edge_index.long(), num_nodes=num_nodes)


def regenerate_protein_sequence_edges(
    pos_code: list,
    block_idx: torch.Tensor,
    entity_idx: torch.Tensor,
    sequence_neighbors_k: int,
) -> torch.Tensor:
    """Tier-1 CA-CA edges: same entity, 0 < |block gap| <= k, both directions."""
    num_nodes = int(block_idx.numel())
    if len(pos_code) != num_nodes or int(entity_idx.numel()) != num_nodes:
        raise ValueError(
            "Tier-1 regeneration requires pos_code/block_idx/entity_idx of equal length: "
            f"{len(pos_code)}/{num_nodes}/{entity_idx.numel()}"
        )
    ca_nodes = torch.where(_ca_mask(pos_code))[0]
    if ca_nodes.numel() < 2:
        return torch.empty((2, 0), dtype=torch.long)
    blocks = block_idx[ca_nodes].long()
    entities = entity_idx[ca_nodes].long()
    gap = (blocks.unsqueeze(1) - blocks.unsqueeze(0)).abs()
    adjacency = (
        (gap > 0)
        & (gap <= int(sequence_neighbors_k))
        & (entities.unsqueeze(1) == entities.unsqueeze(0))
    )
    source, target = torch.where(adjacency)
    edge_index = torch.stack((ca_nodes[source], ca_nodes[target]), dim=0)
    return _coalesced(edge_index, num_nodes)


def _check_range(name: str, values: torch.Tensor, dtype: torch.dtype, mol_id: object) -> None:
    if values.numel() == 0:
        return
    info = torch.iinfo(dtype)
    low, high = int(values.min()), int(values.max())
    if low < info.min or high > info.max:
        raise ValueError(
            f"{name} range [{low}, {high}] does not fit {dtype}: mol_id={mol_id}"
        )


def encode_edge_free_sample(
    sample: Data,
    domain: str,
    pos_code_index: Mapping[str, int],
) -> Data:
    """Return the compact node-only representation of one legacy sample."""
    domain = normalize_schema_domain(domain)
    mol_id = getattr(sample, "mol_id", "?")
    sample_domain = normalize_schema_domain(getattr(sample, "dataset_type", None))
    if sample_domain != domain:
        raise ValueError(f"sample domain {sample_domain} != chunk domain {domain}: mol_id={mol_id}")
    for name in ("z", "pos", "block_idx", "entity_idx", "pos_code", "block_symbols"):
        if getattr(sample, name, None) is None:
            raise ValueError(f"{domain} sample missing '{name}' contract: mol_id={mol_id}")
    num_nodes = int(sample.z.numel())
    pos_code = list(sample.pos_code)
    if (
        tuple(sample.pos.shape) != (num_nodes, 3)
        or int(sample.block_idx.numel()) != num_nodes
        or int(sample.entity_idx.numel()) != num_nodes
        or len(pos_code) != num_nodes
    ):
        raise ValueError(f"{domain} sample node fields have inconsistent lengths: mol_id={mol_id}")
    if sample.pos.dtype != torch.float32:
        raise ValueError(f"{domain} sample pos must be float32, got {sample.pos.dtype}: mol_id={mol_id}")
    for name in RUNTIME_INDEX_FIELDS:
        _check_range(name, getattr(sample, name), COMPACT_NODE_DTYPES[name], mol_id)
    try:
        code_ids = [pos_code_index[str(code)] for code in pos_code]
    except KeyError as exc:
        raise ValueError(f"pos_code {exc} missing from chunk vocabulary: mol_id={mol_id}") from exc
    return Data(
        z=sample.z.to(COMPACT_NODE_DTYPES["z"]),
        pos=sample.pos.clone(),
        block_idx=sample.block_idx.to(COMPACT_NODE_DTYPES["block_idx"]),
        entity_idx=sample.entity_idx.to(COMPACT_NODE_DTYPES["entity_idx"]),
        pos_code_ids=torch.tensor(code_ids, dtype=COMPACT_NODE_DTYPES["pos_code_ids"]),
        block_symbols=list(sample.block_symbols),
        mol_id=str(mol_id),
        dataset_type=str(getattr(sample, "dataset_type")),
    )


def decode_edge_free_sample(
    sample: Data,
    schema: Mapping[str, object],
    contract: "EdgeFreeRuntimeContract",
) -> Data:
    """Restore the legacy in-memory sample contract from a compact sample."""
    domain = normalize_schema_domain(schema["domain"])
    mol_id = getattr(sample, "mol_id", "?")
    contract.require_domain(domain, origin=f"mol_id={mol_id}")
    for name in ("z", "pos", "block_idx", "entity_idx", "pos_code_ids"):
        if getattr(sample, name, None) is None:
            raise ValueError(f"edge-free {domain} sample missing '{name}': mol_id={mol_id}")
    vocabulary = schema["pos_code_vocabulary"]
    code_ids = sample.pos_code_ids.long()
    if code_ids.numel() and int(code_ids.max()) >= len(vocabulary):
        raise ValueError(f"pos_code id outside vocabulary: mol_id={mol_id}")

    num_nodes = int(sample.z.numel())
    pos_code = [vocabulary[index] for index in code_ids.tolist()]
    block_idx = sample.block_idx.long()
    entity_idx = sample.entity_idx.long()
    if domain == "protein":
        if int(schema["sequence_neighbors_k"]) != contract.sequence_neighbors_k:
            raise ValueError(
                "edge-free protein chunk was built for sequence_neighbors_k="
                f"{schema['sequence_neighbors_k']}, config uses {contract.sequence_neighbors_k}"
            )
        edge_index = regenerate_protein_sequence_edges(
            pos_code, block_idx, entity_idx, contract.sequence_neighbors_k
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    pos = sample.pos
    edge_attr = torch.linalg.vector_norm(
        pos[edge_index[0]] - pos[edge_index[1]], dim=1, keepdim=True
    ).to(torch.float32)

    return Data(
        num_edges=torch.tensor([edge_index.size(1)], dtype=torch.long),
        mol_id=mol_id,
        block_idx=block_idx,
        pos=pos,
        pos_code=pos_code,
        num_nodes=num_nodes,
        edge_attr=edge_attr,
        edge_index=edge_index,
        z=sample.z.long(),
        entity_idx=entity_idx,
        dataset_type=getattr(sample, "dataset_type"),
        block_symbols=list(sample.block_symbols),
    )


@dataclass(frozen=True)
class EdgeFreeRuntimeContract:
    """Runtime settings an edge-free chunk depends on, derived from the config."""

    sequence_neighbors_k: int
    incompatibilities: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def require_domain(self, domain: str, *, origin: str) -> None:
        domain = normalize_schema_domain(domain)
        reasons = tuple(self.incompatibilities.get("all", ())) + tuple(
            self.incompatibilities.get(domain, ())
        )
        if reasons:
            raise ValueError(
                f"edge-free {domain} chunk {origin} cannot be used with this config: "
                + "; ".join(reasons)
            )


def _setting(config: object, name: str, default: object) -> object:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def edge_free_runtime_contract_from_config(config: object) -> EdgeFreeRuntimeContract:
    """Collect every config choice that would make stored edges observable."""
    reasons: dict[str, list[str]] = {"all": [], "protein": [], "rna": []}
    if str(_setting(config, "apply_attention_on", "node")) == "edge":
        reasons["all"].append("apply_attention_on='edge' reads stored edges")
    template = _setting(config, "masked_slot_template", None) or {}
    if bool(_setting(template, "enabled", False)):
        reasons["all"].append("masked_slot_template rebuilds stored template edges")

    geometry = _setting(config, "geometry_neutral_virtual_slots", None) or {}
    geometry_enabled = bool(_setting(geometry, "enabled", False))
    for domain in EDGE_FREE_DOMAINS:
        if not (geometry_enabled and bool(_setting(geometry, domain, False))):
            reasons[domain].append(
                f"geometry_neutral_virtual_slots must be enabled for {domain}"
            )
    if not bool(_setting(config, "use_dynamic_random_edges", False)):
        reasons["protein"].append("use_dynamic_random_edges must be true")
    if not bool(_setting(config, "replace_existing_random_edges", False)):
        reasons["protein"].append("replace_existing_random_edges must be true")

    return EdgeFreeRuntimeContract(
        sequence_neighbors_k=int(_setting(config, "sequence_neighbors_k", 3)),
        incompatibilities={key: tuple(value) for key, value in reasons.items() if value},
    )


def split_chunk_payload(payload: object, *, origin: str) -> tuple[Data, dict, Optional[dict]]:
    """Return ``(data, slices, schema)`` for legacy 2-tuples and versioned 3-tuples."""
    if not isinstance(payload, (tuple, list)):
        raise ValueError(f"{origin}: expected a (data, slices[, schema]) tuple, got {type(payload)}")
    if len(payload) == 2:
        data, slices = payload
        return data, slices, None
    if len(payload) == 3:
        data, slices, schema = payload
        return data, slices, validate_edge_free_schema(schema, origin=origin)
    raise ValueError(f"{origin}: expected 2 or 3 chunk entries, got {len(payload)}")
