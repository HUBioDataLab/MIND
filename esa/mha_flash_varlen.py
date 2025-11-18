"""
Padding-Free Multi-Head Attention using Flash Attention Variable-Length Functions.

This module implements MAB, SAB, and PMA layers without requiring padding,
using flash_attn_varlen_func for improved memory efficiency and speed on
variable-length sequences (e.g., graphs with varying number of nodes/edges).

Advantages over padded version:
- No memory wasted on padding tokens
- Faster computation (only processes real tokens)
- Up to 2x speedup on datasets with high length variance
- Lower peak memory usage

Based on the approach described in:
https://huggingface.co/blog/packing-with-FA2
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from flash_attn.bert_padding import unpad_input, pad_input


class MABFlashVarlen(nn.Module):
    """
    Multihead Attention Block (MAB) using Flash Attention variable-length functions.
    
    This version eliminates padding by using cu_seqlens (cumulative sequence lengths)
    to track boundaries between samples in a batch.
    """
    
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, dropout_p=0.0):
        super(MABFlashVarlen, self).__init__()
        
        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        
        self.fc_q = nn.Linear(dim_Q, dim_V, bias=True)
        self.fc_k = nn.Linear(dim_K, dim_V, bias=True)
        self.fc_v = nn.Linear(dim_K, dim_V, bias=True)
        self.fc_o = nn.Linear(dim_V, dim_V, bias=True)
        
        # Initialize weights
        xavier_normal_(self.fc_q.weight)
        xavier_normal_(self.fc_k.weight)
        xavier_normal_(self.fc_v.weight)
        xavier_normal_(self.fc_o.weight)
    
    
    def forward(self, Q, K, seq_lens=None, adj_mask=None):
        """
        Forward pass with optional padding-free mode.
        
        Args:
            Q: Query tensor [batch_size, seq_len_q, dim_Q] or [total_tokens, dim_Q] (unpacked)
            K: Key/Value tensor [batch_size, seq_len_k, dim_K] or [total_tokens, dim_K] (unpacked)
            seq_lens: List or tensor of sequence lengths for each sample in batch
                     If None, assumes Q and K are already padded (falls back to standard attention)
            adj_mask: Optional adjacency mask (currently not fully supported in varlen mode)
        
        Returns:
            out: Output tensor with same shape as Q
        """
        
        # Check if we're in padding-free mode
        padding_free = (seq_lens is not None)
        
        if not padding_free:
            # Fall back to standard padded attention (for compatibility)
            return self._forward_padded(Q, K, adj_mask)
        
        # Padding-free mode using Flash Attention varlen
        return self._forward_varlen(Q, K, seq_lens, adj_mask)
    
    
    def _forward_padded(self, Q, K, adj_mask=None):
        """
        Standard padded attention (original implementation for backward compatibility).
        """
        batch_size = Q.size(0)
        E_total = self.dim_V
        assert E_total % self.num_heads == 0, "Embedding dim is not divisible by nheads"
        head_dim = E_total // self.num_heads
        
        # Store original dtype for output conversion
        input_dtype = Q.dtype
        
        # Ensure input matches model dtype
        model_dtype = self.fc_q.weight.dtype
        if Q.dtype != model_dtype:
            Q = Q.to(model_dtype)
            K = K.to(model_dtype)
        
        Q = self.fc_q(Q)
        V = self.fc_v(K)
        K = self.fc_k(K)
        
        Q = Q.view(batch_size, -1, self.num_heads, head_dim)
        K = K.view(batch_size, -1, self.num_heads, head_dim)
        V = V.view(batch_size, -1, self.num_heads, head_dim)
        
        # Use PyTorch's SDPA (scaled dot-product attention)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        out = F.scaled_dot_product_attention(
            Q, K, V, 
            attn_mask=adj_mask, 
            dropout_p=self.dropout_p if self.training else 0, 
            is_causal=False
        )
        out = out.transpose(1, 2).reshape(batch_size, -1, self.num_heads * head_dim)
        
        out = out + F.mish(self.fc_o(out))
        
        # Convert back to original dtype if needed
        if input_dtype != out.dtype:
            out = out.to(input_dtype)
        
        return out
    
    
    def _forward_varlen(self, Q, K, seq_lens, adj_mask=None):
        """
        Padding-free attention using Flash Attention varlen functions.
        
        Args:
            Q: [batch_size, max_seq_len, dim_Q] - padded input
            K: [batch_size, max_seq_len, dim_K] - padded input
            seq_lens: tensor of shape [batch_size] containing actual lengths
            adj_mask: Not fully supported yet in varlen mode (TODO)
        """
        E_total = self.dim_V
        assert E_total % self.num_heads == 0, "Embedding dim is not divisible by nheads"
        head_dim = E_total // self.num_heads
        
        batch_size = Q.size(0)
        max_seq_len = Q.size(1)
        
        # Flash Attention requires fp16 or bf16
        input_dtype = Q.dtype
        use_bf16 = input_dtype != torch.float16
        if use_bf16:
            Q = Q.to(torch.bfloat16)
            K = K.to(torch.bfloat16)
        
        # Project to Q, K, V
        Q = self.fc_q(Q)  # [batch_size, max_seq_len, dim_V]
        K = self.fc_k(K)  # [batch_size, max_seq_len, dim_V]
        V = self.fc_v(K)  # Note: V comes from K in your original implementation
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, max_seq_len, self.num_heads, head_dim)
        K = K.view(batch_size, max_seq_len, self.num_heads, head_dim)
        V = V.view(batch_size, max_seq_len, self.num_heads, head_dim)
        
        # Create padding mask from sequence lengths
        # Shape: [batch_size, max_seq_len]
        padding_mask = torch.arange(max_seq_len, device=Q.device).unsqueeze(0) < seq_lens.unsqueeze(1)
        
        # Unpad Q, K, V - remove all padding tokens
        Q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(Q, padding_mask)
        K_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(K, padding_mask)
        V_unpad, _, _, _ = unpad_input(V, padding_mask)
        
        # Q_unpad, K_unpad, V_unpad are now [total_tokens, num_heads, head_dim]
        # where total_tokens = sum(seq_lens)
        
        # Apply Flash Attention on variable-length sequences
        # This is the magic - no computation on padding!
        out_unpad = flash_attn_varlen_func(
            Q_unpad, K_unpad, V_unpad,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=None,  # Uses 1/sqrt(d) by default
            causal=False
        )
        # out_unpad: [total_tokens, num_heads, head_dim]
        
        # Repad the output to restore original batch structure
        out = pad_input(out_unpad, indices_q, batch_size, max_seq_len)
        # out: [batch_size, max_seq_len, num_heads, head_dim]
        
        # Reshape back
        out = out.reshape(batch_size, max_seq_len, self.num_heads * head_dim)
        
        # Apply output projection with Mish activation
        out = out + F.mish(self.fc_o(out))
        
        # Convert back to original dtype if needed
        if input_dtype != out.dtype:
            out = out.to(input_dtype)
        
        return out


class SABFlashVarlen(nn.Module):
    """
    Self-Attention Block using padding-free Flash Attention.
    """
    
    def __init__(self, dim_in, dim_out, num_heads, dropout):
        super(SABFlashVarlen, self).__init__()
        self.mab = MABFlashVarlen(dim_in, dim_in, dim_out, num_heads, dropout)
    
    def forward(self, X, seq_lens=None, adj_mask=None):
        return self.mab(X, X, seq_lens=seq_lens, adj_mask=adj_mask)


class PMAFlashVarlen(nn.Module):
    """
    Pooling by Multihead Attention using padding-free Flash Attention.
    
    Uses learnable seed vectors to pool variable-length sequences.
    Note: For PMA, Q (seeds) and K (input) have different lengths.
    Flash Attention varlen handles this, but we need to pass separate seq_lens for Q and K.
    """
    
    def __init__(self, dim, num_heads, num_seeds, dropout):
        super(PMAFlashVarlen, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        self.num_seeds = num_seeds
        nn.init.xavier_normal_(self.S)
        self.mab = MABFlashVarlen(dim, dim, dim, num_heads, dropout_p=dropout)
    
    def forward(self, X, seq_lens=None, adj_mask=None):
        """
        Args:
            X: [batch_size, max_seq_len, dim]
            seq_lens: [batch_size] tensor of actual sequence lengths for K/V
        """
        batch_size = X.size(0)
        S = self.S.repeat(batch_size, 1, 1)  # [batch_size, num_seeds, dim]
        
        # For PMA cross-attention:
        # - Q (seeds) has fixed length = num_seeds for all samples
        # - K/V (input X) has variable length = seq_lens
        # Currently flash_varlen expects Q and K to have same seq_lens
        # So we fall back to padded mode for PMA to avoid complexity
        
        return self.mab(S, X, seq_lens=None, adj_mask=adj_mask)  # Use padded fallback


# Convenience function to convert padded batch to sequence lengths
def get_seq_lens_from_mask(mask):
    """
    Extract sequence lengths from a padding mask.
    
    Args:
        mask: Boolean tensor [batch_size, max_seq_len] where True = valid token
    
    Returns:
        seq_lens: Tensor [batch_size] with actual sequence lengths
    """
    return mask.sum(dim=1).to(torch.int32)


# Convenience function to create padding mask from sequence lengths
def create_padding_mask(seq_lens, max_len=None, device=None):
    """
    Create a padding mask from sequence lengths.
    
    Args:
        seq_lens: Tensor [batch_size] with sequence lengths
        max_len: Maximum sequence length (if None, uses max(seq_lens))
        device: Device to create mask on
    
    Returns:
        mask: Boolean tensor [batch_size, max_seq_len] where True = valid token
    """
    if device is None:
        device = seq_lens.device
    if max_len is None:
        max_len = seq_lens.max().item()
    
    batch_size = seq_lens.size(0)
    mask = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
    mask = mask < seq_lens.unsqueeze(1)
    return mask

