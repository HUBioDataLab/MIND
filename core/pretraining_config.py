from dataclasses import dataclass
from typing import Optional, List, Dict


MASKED_SLOT_TEMPLATE_DEFAULTS = {
    "enabled": False,
    "protein": False,
    "rna": False,
    "visible_virtuals_prepositioned": False,
}


def normalize_masked_slot_template_config(
    value: Optional[Dict[str, bool]],
    *,
    allow_enabled: bool = False,
) -> Dict[str, bool]:
    """Validate and normalize the staged masked-slot-template feature flag.

    The staged runtime contract requires an explicit global flag plus at least
    one enabled domain.
    """
    if value is None:
        return dict(MASKED_SLOT_TEMPLATE_DEFAULTS)
    if not isinstance(value, dict):
        raise TypeError(
            "masked_slot_template must be a mapping with enabled/protein/rna/"
            "visible_virtuals_prepositioned"
        )
    unknown = set(value) - set(MASKED_SLOT_TEMPLATE_DEFAULTS)
    if unknown:
        raise ValueError(
            "unknown masked_slot_template keys: " + ", ".join(sorted(unknown))
        )
    normalized = dict(MASKED_SLOT_TEMPLATE_DEFAULTS)
    for key, setting in value.items():
        if type(setting) is not bool:
            raise TypeError(f"masked_slot_template.{key} must be boolean")
        normalized[key] = setting
    if not normalized["enabled"] and (
        normalized["protein"]
        or normalized["rna"]
        or normalized["visible_virtuals_prepositioned"]
    ):
        raise ValueError(
            "masked_slot_template domain flags require enabled=true"
        )
    if normalized["enabled"] and not (
        normalized["protein"] or normalized["rna"]
    ):
        raise ValueError(
            "masked_slot_template.enabled=true requires protein and/or rna"
        )
    if normalized["enabled"] and not allow_enabled:
        raise NotImplementedError(
            "masked_slot_template flag-on behavior is disabled by this caller"
        )
    return normalized


@dataclass
class PretrainingConfig:
    """Configuration class for pretraining tasks"""
    # General model config
    num_features: int = 128
    graph_dim: int = 512
    edge_dim: int = 64
    batch_size: int = 256
    lr: float = 0.0005
    monitor_loss_name: str = "val_total_loss"
    xformers_or_torch_attn: str = "xformers"
    hidden_dims: List[int] = None
    num_heads: List[int] = None
    num_sabs: int = 4
    sab_dropout: float = 0.0
    mab_dropout: float = 0.0
    pma_dropout: float = 0.0
    apply_attention_on: str = "node"
    layer_types: List[str] = None
    use_mlps: bool = True
    set_max_items: int = 0
    early_stopping_patience: int = 30
    optimiser_weight_decay: float = 1e-10
    num_workers: int = 4
    mlp_hidden_size: int = 512
    mlp_type: str = "standard"
    attn_residual_dropout: float = 0.0
    norm_type: str = "LN"
    triu_attn_mask: bool = False
    use_bfloat16: bool = False
    is_node_task: bool = False
    use_posenc: bool = False
    num_mlp_layers: int = 3
    pre_or_post: str = "pre"
    pma_residual_dropout: float = 0
    use_mlp_ln: bool = False
    mlp_dropout: float = 0

    # Development controls (optional)
    debug_subset_n: Optional[int] = None
    debug_verbose: bool = False

    # Universal molecular configs
    # Sparse ids 0..120: mask=0, physical/common atomic numbers, rare-real=119,
    # and virtual padding=120.
    atom_types: int = 121
    bond_types: int = 4    # Single, double, triple, aromatic
    molecule_max_atoms: int = 500  # Maximum atoms per molecular system

    # 3D geometric configs
    use_3d_coordinates: bool = True
    coordinate_dim: int = 3
    gaussian_kernels: int = 128
    cutoff_distance: float = 5.0
    max_neighbors: int = 16

    # Pretraining task configs
    pretraining_tasks: List[str] = None  # ["long_range_distance", "short_range_distance", "mlm", ...]
    task_weights: Dict[str, float] = None

    # Distance prediction — long-range loss (random atom pairs, 0-20Å)
    distance_prediction_weight: float = 1.0
    distance_bins: int = 16
    max_distance: float = 20.0
    # Non-uniform boundaries: len must equal distance_bins - 1. None → uniform linear.
    distance_bin_boundaries: Optional[List[float]] = None

    # Distance prediction — short-range loss (graph edges, bounded by cutoff_distance ≤5Å)
    # Separate head/bins so we don't waste capacity on distances that never appear in edges.
    short_range_distance_bins: int = 10
    short_range_max_distance: float = 5.0
    # None → uniform linear over [0, short_range_max_distance]
    short_range_bin_boundaries: Optional[List[float]] = None

    # Long-range distance loss configuration
    long_range_ranking_margin: float = 1.0

    # Masked language modeling
    mlm_weight: float = 1.0
    mlm_mask_ratio: float = 0.15

    # Domain-specific block MLM. Protein masks complete residues; RNA masks
    # complete nucleotides. Each option is independently default-false, and
    # validation/test use a deterministic seed so masks stay stable.
    protein_block_mlm: bool = False
    protein_block_mlm_ratio: float = 0.15
    rna_block_mlm: bool = False
    rna_block_mlm_ratio: float = 0.15

    # Optional residue/nucleotide-local atom ordinal embedding. This identifies
    # a slot without exposing whether its masked target is real or virtual.
    use_domain_local_slot_embedding: bool = False

    # Keep legacy MLM exactly unchanged at 1.0. Controlled virtual-slot runs
    # can reduce the contribution of the much more frequent V targets.
    virtual_atom_loss_weight: float = 1.0

    # Legacy 14-node terminal residues can contain OXT in place of one V slot.
    # Exclude those complete blocks from protein block-MLM without rewriting
    # cached graphs. This can be enabled independently in paired controls.
    protein_block_mlm_exclude_legacy_oxt: bool = False

    # Runtime virtual-slot template. Disabled by default.
    masked_slot_template: Optional[Dict[str, bool]] = None

    # Coordinate denoising (predict added noise from node embeddings; SE(3)-invariant loss)
    coordinate_denoising_noise_std: float = 0.1
    coordinate_denoising_mask_ratio: float = 0.15  # fraction of nodes to compute loss on

    # Temperature for softmax (used in distance prediction)
    temperature: float = 0.1

    # Per-type loss tracking (for multi-domain analysis)
    compute_per_type_losses: bool = False
    log_per_type_frequency: int = 10

    # Verbose batch logging (BATCH N, dataset mix, step loss prints); set true to enable
    log_batch_stats: bool = False

    # Equivariant features (experimental)
    use_equivariant_features: bool = False  # Set to true to enable equivariant branch
    equivariant_lmax: int = 2  # Maximum angular momentum (l=0,1,2)
    equivariant_num_features: int = 16  # Number of tensor feature channels
    equivariant_fusion_method: str = "add"  # add, concat, bilinear
    equivariant_cross_connection: bool = True  # Enable cross-connection between invariant and equivariant branches


def create_pretraining_config(**kwargs) -> PretrainingConfig:
    """Create a PretrainingConfig with optional keyword overrides."""
    config = PretrainingConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)
    config.masked_slot_template = normalize_masked_slot_template_config(
        getattr(config, "masked_slot_template", None), allow_enabled=True
    )
    virtual_atom_loss_weight = float(
        getattr(config, "virtual_atom_loss_weight", 1.0)
    )
    if not 0.0 <= virtual_atom_loss_weight <= 1.0:
        raise ValueError("virtual_atom_loss_weight must be between 0.0 and 1.0")
    config.virtual_atom_loss_weight = virtual_atom_loss_weight
    template = config.masked_slot_template
    if template["enabled"]:
        if template["protein"] and not getattr(config, "protein_block_mlm", False):
            raise ValueError("protein templates require protein_block_mlm=true")
        if template["rna"] and not getattr(config, "rna_block_mlm", False):
            raise ValueError("RNA templates require rna_block_mlm=true")
        if "coordinate_denoising" in (getattr(config, "pretraining_tasks", None) or ()):
            raise ValueError(
                "masked slot templates are incompatible with coordinate_denoising"
            )
    return config
