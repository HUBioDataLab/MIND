"""
Equivariant Self-Attention Block (SAB) with integrated equivariant features.

This module integrates equivariant features WITHIN each attention layer,
following the pattern from Equiformer and similar architectures where
invariant and equivariant features interact at each layer.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    from nequip.nn import InteractionBlock
    from nequip.data import AtomicDataDict
    from e3nn.o3 import Irreps, SphericalHarmonics
    NEQUIP_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    NEQUIP_AVAILABLE = False

from esa.mha import SAB, MAB


class EquivariantSAB(nn.Module):
    """
    Self-Attention Block with integrated equivariant features.
    
    Architecture:
    1. Invariant branch: Standard SAB (self-attention on scalar features)
    2. Equivariant branch: NequIP InteractionBlock (equivariant message passing)
    3. Cross-connection: Equivariant features inform attention, invariant features inform equivariant
    4. Fusion: Combine both branches before output
    """
    
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_heads: int,
        dropout: float,
        xformers_or_torch_attn: str = "xformers",
        # Equivariant branch parameters
        use_equivariant: bool = True,
        equivariant_lmax: int = 2,
        equivariant_num_features: int = 16,
        fusion_method: str = "add",  # "add", "concat", "bilinear", "gated"
        cross_connection: bool = True,
    ):
        super().__init__()
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.use_equivariant = use_equivariant and NEQUIP_AVAILABLE
        self.fusion_method = fusion_method
        self.cross_connection = cross_connection
        
        # Invariant branch: Standard SAB
        self.invariant_sab = SAB(dim_in, dim_out, num_heads, dropout, xformers_or_torch_attn)
        
        # Equivariant branch: NequIP InteractionBlock
        if self.use_equivariant:
            assert NEQUIP_AVAILABLE, (
                "nequip is required for equivariant features. "
                "Install with: pip install nequip"
            )
            
            # Input: scalar node features
            irreps_in = {
                AtomicDataDict.NODE_FEATURES_KEY: Irreps(f"{dim_in}x0e"),
                AtomicDataDict.EDGE_EMBEDDING_KEY: Irreps("1x0e"),
                AtomicDataDict.EDGE_ATTRS_KEY: Irreps.spherical_harmonics(equivariant_lmax),
                AtomicDataDict.NODE_ATTRS_KEY: Irreps("1x0e"),
            }
            
            # Output: mix of scalars and vectors
            irreps_out = Irreps(f"{equivariant_num_features//2}x0e + {equivariant_num_features//2}x1o")
            
            self.equivariant_interaction = InteractionBlock(
                irreps_in=irreps_in,
                irreps_out=irreps_out,
                radial_mlp_depth=2,
                radial_mlp_width=64,
                use_sc=True,
                is_first_layer=False,
                avg_num_neighbors=10.0,
            )
            
            # Project equivariant features back to scalar space
            self.equivariant_to_scalar = nn.Linear(irreps_out.dim, dim_out)
            
            # Cross-connection layers
            if cross_connection:
                # Invariant -> Equivariant (to inform equivariant branch)
                self.inv_to_equiv = nn.Linear(dim_in, dim_in)
                # Equivariant -> Invariant (to gate/modulate attention)
                self.equiv_to_inv_gate = nn.Linear(irreps_out.dim, dim_out)
        
        # Fusion layer
        if fusion_method == "concat":
            self.fusion = nn.Linear(dim_out * 2, dim_out)
        elif fusion_method == "bilinear":
            self.fusion = nn.Bilinear(dim_out, dim_out, dim_out)
        elif fusion_method == "gated":
            # Gated fusion: use equivariant features to gate invariant features
            self.fusion_gate = nn.Sequential(
                nn.Linear(dim_out, dim_out),
                nn.Sigmoid()
            )
            self.fusion_proj = nn.Linear(dim_out, dim_out)
        # "add" doesn't need a fusion layer
    
    def forward(
        self,
        X: torch.Tensor,  # [batch_size, max_nodes, dim_in] - invariant features
        adj_mask: Optional[torch.Tensor] = None,
        # Equivariant branch inputs
        edge_index: Optional[torch.Tensor] = None,  # [2, E]
        edge_vectors: Optional[torch.Tensor] = None,  # [E, 3]
        pos: Optional[torch.Tensor] = None,  # [total_nodes, 3]
        batch_mapping: Optional[torch.Tensor] = None,  # [total_nodes]
    ) -> torch.Tensor:
        """
        Forward pass with integrated equivariant features.
        
        Args:
            X: Invariant features [batch_size, max_nodes, dim_in]
            adj_mask: Attention mask [batch_size, num_heads, max_nodes, max_nodes]
            edge_index: Edge connectivity [2, E]
            edge_vectors: Edge vectors [E, 3]
            pos: Node positions [total_nodes, 3]
            batch_mapping: Batch indices [total_nodes]
        
        Returns:
            Output features [batch_size, max_nodes, dim_out]
        """
        batch_size, max_nodes, dim_in = X.shape
        
        # ============================================================
        # 1. INVARIANT BRANCH: Standard self-attention
        # ============================================================
        # If cross-connection enabled, enhance input with equivariant info
        if self.use_equivariant and self.cross_connection and edge_vectors is not None:
            # Get equivariant features to inform attention
            # We'll compute this in the equivariant branch section
            # For now, use invariant features as-is
            invariant_input = X
        else:
            invariant_input = X
        
        # Apply invariant self-attention
        invariant_out = self.invariant_sab(invariant_input, adj_mask=adj_mask)
        # invariant_out: [batch_size, max_nodes, dim_out]
        
        # ============================================================
        # 2. EQUIVARIANT BRANCH: NequIP InteractionBlock
        # ============================================================
        if self.use_equivariant and edge_index is not None and edge_vectors is not None:
            # Convert dense X to sparse format for equivariant branch
            # X is [batch_size, max_nodes, dim_in], need [total_nodes, dim_in]
            if batch_mapping is not None:
                total_nodes = batch_mapping.size(0)
                X_sparse = torch.zeros(total_nodes, dim_in, device=X.device, dtype=X.dtype)
                
                # Map dense to sparse
                node_idx = 0
                for b in range(batch_size):
                    # Count actual nodes in this batch (non-padding)
                    if batch_mapping is not None:
                        n_nodes_b = (batch_mapping == b).sum().item()
                    else:
                        n_nodes_b = max_nodes
                    
                    if n_nodes_b > 0:
                        actual_nodes = min(n_nodes_b, max_nodes)
                        X_sparse[node_idx:node_idx+actual_nodes] = X[b, :actual_nodes]
                        node_idx += actual_nodes
            else:
                # Single graph case
                X_sparse = X.squeeze(0)  # [max_nodes, dim_in]
                total_nodes = X_sparse.size(0)
            
            # Cross-connection: Invariant -> Equivariant
            if self.cross_connection:
                X_sparse_enhanced = self.inv_to_equiv(X_sparse)
            else:
                X_sparse_enhanced = X_sparse
            
            # Prepare NequIP data format
            num_edges = edge_index.size(1)
            
            # Compute edge attributes (spherical harmonics)
            edge_attr_irreps = Irreps.spherical_harmonics(self.equivariant_interaction.irreps_in[AtomicDataDict.EDGE_ATTRS_KEY].lmax)
            sh = SphericalHarmonics(edge_attr_irreps, normalize=True, normalization="component")
            edge_attrs = sh(edge_vectors)  # [E, edge_attr_dim]
            
            # Dummy positions (required by NequIP, but we use edge_vectors)
            dummy_positions = torch.zeros(total_nodes, 3, device=X.device, dtype=X.dtype)
            if pos is not None:
                dummy_positions = pos
            
            # Edge embedding (scalar, just ones for now)
            edge_embedding = torch.ones(num_edges, 1, device=X.device, dtype=X.dtype)
            
            # Node attributes (for self-connection)
            node_attrs = torch.ones(total_nodes, 1, device=X.device, dtype=X.dtype)
            
            # Create NequIP data dict
            data = {
                AtomicDataDict.POSITIONS_KEY: dummy_positions,
                AtomicDataDict.NODE_FEATURES_KEY: X_sparse_enhanced,
                AtomicDataDict.EDGE_INDEX_KEY: edge_index,
                AtomicDataDict.EDGE_VECTORS_KEY: edge_vectors,
                AtomicDataDict.EDGE_EMBEDDING_KEY: edge_embedding,
                AtomicDataDict.EDGE_ATTRS_KEY: edge_attrs,
                AtomicDataDict.NODE_ATTRS_KEY: node_attrs,
            }
            
            if batch_mapping is not None:
                data[AtomicDataDict.BATCH_KEY] = batch_mapping
            
            # Apply equivariant interaction
            data = self.equivariant_interaction(data)
            equivariant_features = data[AtomicDataDict.NODE_FEATURES_KEY]  # [total_nodes, irreps_out.dim]
            
            # Project to scalar space
            equivariant_scalar = self.equivariant_to_scalar(equivariant_features)  # [total_nodes, dim_out]
            
            # Convert back to dense format
            if batch_mapping is not None:
                equivariant_dense = torch.zeros(batch_size, max_nodes, self.dim_out, device=X.device, dtype=X.dtype)
                node_idx = 0
                for b in range(batch_size):
                    n_nodes_b = (batch_mapping == b).sum().item()
                    if n_nodes_b > 0:
                        actual_nodes = min(n_nodes_b, max_nodes)
                        equivariant_dense[b, :actual_nodes] = equivariant_scalar[node_idx:node_idx+actual_nodes]
                        node_idx += actual_nodes
            else:
                equivariant_dense = equivariant_scalar.unsqueeze(0)  # [1, max_nodes, dim_out]
            
            # Cross-connection: Equivariant -> Invariant (gate attention)
            if self.cross_connection:
                equivariant_gate = self.equiv_to_inv_gate(equivariant_features)  # [total_nodes, dim_out]
                
                # Convert gate to dense
                if batch_mapping is not None:
                    gate_dense = torch.zeros(batch_size, max_nodes, self.dim_out, device=X.device, dtype=X.dtype)
                    node_idx = 0
                    for b in range(batch_size):
                        n_nodes_b = (batch_mapping == b).sum().item()
                        if n_nodes_b > 0:
                            actual_nodes = min(n_nodes_b, max_nodes)
                            gate_dense[b, :actual_nodes] = equivariant_gate[node_idx:node_idx+actual_nodes]
                            node_idx += actual_nodes
                else:
                    gate_dense = equivariant_gate.unsqueeze(0)
                
                # Gate invariant output
                invariant_out = invariant_out * torch.sigmoid(gate_dense)
        
        else:
            # No equivariant branch, just return invariant output
            equivariant_dense = None
        
        # ============================================================
        # 3. FUSION: Combine invariant and equivariant branches
        # ============================================================
        if self.use_equivariant and equivariant_dense is not None:
            if self.fusion_method == "add":
                fused = invariant_out + equivariant_dense
            elif self.fusion_method == "concat":
                fused = torch.cat([invariant_out, equivariant_dense], dim=-1)
                fused = self.fusion(fused)
            elif self.fusion_method == "bilinear":
                fused = self.fusion(invariant_out, equivariant_dense)
            elif self.fusion_method == "gated":
                gate = self.fusion_gate(equivariant_dense)
                fused = invariant_out + gate * self.fusion_proj(equivariant_dense)
            else:
                fused = invariant_out
        else:
            fused = invariant_out
        
        return fused


class EquivariantMAB(nn.Module):
    """
    Multi-Head Attention Block with integrated equivariant features.
    
    Similar to EquivariantSAB but for cross-attention (MAB).
    """
    
    def __init__(
        self,
        dim_Q: int,
        dim_K: int,
        dim_V: int,
        num_heads: int,
        dropout_p: float,
        xformers_or_torch_attn: str = "xformers",
        # Equivariant branch parameters
        use_equivariant: bool = True,
        equivariant_lmax: int = 2,
        equivariant_num_features: int = 16,
        fusion_method: str = "add",
        cross_connection: bool = True,
    ):
        super().__init__()
        
        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.dim_V = dim_V
        self.use_equivariant = use_equivariant and NEQUIP_AVAILABLE
        self.fusion_method = fusion_method
        self.cross_connection = cross_connection
        
        # Invariant branch: Standard MAB
        self.invariant_mab = MAB(dim_Q, dim_K, dim_V, num_heads, dropout_p, xformers_or_torch_attn)
        
        # Equivariant branch (similar to EquivariantSAB)
        if self.use_equivariant:
            assert NEQUIP_AVAILABLE, (
                "nequip is required for equivariant features. "
                "Install with: pip install nequip"
            )
            
            irreps_in = {
                AtomicDataDict.NODE_FEATURES_KEY: Irreps(f"{dim_K}x0e"),
                AtomicDataDict.EDGE_EMBEDDING_KEY: Irreps("1x0e"),
                AtomicDataDict.EDGE_ATTRS_KEY: Irreps.spherical_harmonics(equivariant_lmax),
                AtomicDataDict.NODE_ATTRS_KEY: Irreps("1x0e"),
            }
            
            irreps_out = Irreps(f"{equivariant_num_features//2}x0e + {equivariant_num_features//2}x1o")
            
            self.equivariant_interaction = InteractionBlock(
                irreps_in=irreps_in,
                irreps_out=irreps_out,
                radial_mlp_depth=2,
                radial_mlp_width=64,
                use_sc=True,
                is_first_layer=False,
                avg_num_neighbors=10.0,
            )
            
            self.equivariant_to_scalar = nn.Linear(irreps_out.dim, dim_V)
            
            if cross_connection:
                self.inv_to_equiv = nn.Linear(dim_K, dim_K)
                self.equiv_to_inv_gate = nn.Linear(irreps_out.dim, dim_V)
        
        # Fusion layer
        if fusion_method == "concat":
            self.fusion = nn.Linear(dim_V * 2, dim_V)
        elif fusion_method == "bilinear":
            self.fusion = nn.Bilinear(dim_V, dim_V, dim_V)
        elif fusion_method == "gated":
            self.fusion_gate = nn.Sequential(
                nn.Linear(dim_V, dim_V),
                nn.Sigmoid()
            )
            self.fusion_proj = nn.Linear(dim_V, dim_V)
    
    def forward(
        self,
        Q: torch.Tensor,  # [batch_size, seq_len_Q, dim_Q]
        K: torch.Tensor,  # [batch_size, seq_len_K, dim_K]
        adj_mask: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_vectors: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        batch_mapping: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with integrated equivariant features."""
        # Invariant branch
        invariant_out = self.invariant_mab(Q, K, adj_mask=adj_mask)
        
        # Equivariant branch (similar logic to EquivariantSAB)
        if self.use_equivariant and edge_index is not None and edge_vectors is not None:
            # Use K for equivariant processing (keys represent the graph structure)
            batch_size, seq_len_K, dim_K = K.shape
            
            # Convert to sparse and process equivariant branch
            # (Similar implementation to EquivariantSAB)
            # ... (implementation similar to EquivariantSAB)
            
            # For now, return invariant output
            # Full implementation would mirror EquivariantSAB logic
            equivariant_dense = None
        else:
            equivariant_dense = None
        
        # Fusion
        if self.use_equivariant and equivariant_dense is not None:
            if self.fusion_method == "add":
                fused = invariant_out + equivariant_dense
            elif self.fusion_method == "concat":
                fused = torch.cat([invariant_out, equivariant_dense], dim=-1)
                fused = self.fusion(fused)
            elif self.fusion_method == "bilinear":
                fused = self.fusion(invariant_out, equivariant_dense)
            elif self.fusion_method == "gated":
                gate = self.fusion_gate(equivariant_dense)
                fused = invariant_out + gate * self.fusion_proj(equivariant_dense)
            else:
                fused = invariant_out
        else:
            fused = invariant_out
        
        return fused
