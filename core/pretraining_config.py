from dataclasses import dataclass
from typing import Optional, List, Dict


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
    atom_types: int = 121  # Maximum atom types (119 real elements + 1 virtual atom + 1 mask token)
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

    # COCONUT / metabolite masking (random_atom vs random_walk on bond_edge_index)
    coconut_masking_method: str = "random_atom"  # random_atom | random_walk
    coconut_mask_ratio: float = 0.15
    coconut_num_random_walks: int = 1
    log_coconut_masking_stats: bool = True  # wandb masking/* metrics for coconut/metabolite
    log_coconut_masking_frequency: int = 50  # log every N training batches (0 = every step)
    # Protein-only block MLM: when enabled, MLM masking for protein/pdb graphs
    # is residue-level (all 14 atoms of selected residues masked) instead of
    # atom-level random. Other molecule types (metabolite, QM9, RNA, ...) keep
    # the atom-level behavior. Validation/test use a deterministic seed so the
    # block mask is stable across val cycles.
    protein_block_mlm: bool = False
    protein_block_mlm_ratio: float = 0.15

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
    return config
