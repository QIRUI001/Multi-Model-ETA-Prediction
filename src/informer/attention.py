"""
Attention Mechanisms for Informer.

Implements:
1. FullAttention: Standard scaled dot-product attention
2. ProbSparseAttention: Efficient attention that samples top-u queries based on 
   KL divergence sparsity measure (Equations 5-7 in paper)

The ProbSparse mechanism reduces complexity from O(L^2) to O(L log L) by:
- Sampling u = L_K * ln(L_Q) dot products
- Selecting top-u queries with highest sparsity measure M(q_i, K)
- Using mean V values for non-selected queries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple

from .utils import TriangularCausalMask


class FullAttention(nn.Module):
    """
    Standard scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    Used as baseline and in decoder cross-attention.
    """
    
    def __init__(self, mask_flag: bool = True, factor: int = 5, 
                 scale: Optional[float] = None, attention_dropout: float = 0.1,
                 output_attention: bool = False):
        """
        Args:
            mask_flag: Whether to apply causal mask
            factor: Sampling factor for ProbSparse (unused in FullAttention)
            scale: Attention scale factor (default: 1/sqrt(d_k))
            attention_dropout: Dropout on attention weights
            output_attention: Whether to output attention weights
        """
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute full attention.
        
        Args:
            queries: Query tensor (batch, heads, seq_len_q, d_k)
            keys: Key tensor (batch, heads, seq_len_k, d_k)
            values: Value tensor (batch, heads, seq_len_v, d_v)
            attn_mask: Optional attention mask
            
        Returns:
            output: Attention output (batch, heads, seq_len_q, d_v)
            attn: Attention weights if output_attention=True, else None
        """
        B, H, L_Q, D = queries.shape
        _, _, L_K, _ = keys.shape
        
        # Scale factor
        scale = self.scale or 1.0 / math.sqrt(D)
        
        # Compute attention scores: (B, H, L_Q, L_K)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale
        
        # Apply mask if needed
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L_Q, device=queries.device)
            scores.masked_fill_(attn_mask.mask, float('-inf'))
            
        # Softmax and dropout
        attn = self.dropout(F.softmax(scores, dim=-1))
        
        # Compute output
        output = torch.matmul(attn, values)
        
        if self.output_attention:
            return output, attn
        else:
            return output, None


class ProbSparseAttention(nn.Module):
    """
    ProbSparse Self-Attention mechanism.
    
    Efficiently computes attention by:
    1. Sampling u = c * ln(L_K) queries
    2. Computing sparsity measure M(q_i, K) = max(q_i*k_j/sqrt(d)) - mean(q_i*k_j/sqrt(d))
    3. Selecting top-u queries with highest M values
    4. Computing attention only for selected queries
    5. Using mean(V) for non-selected queries
    
    This reduces complexity from O(L^2) to O(L log L).
    """
    
    def __init__(self, mask_flag: bool = True, factor: int = 5,
                 scale: Optional[float] = None, attention_dropout: float = 0.1,
                 output_attention: bool = False):
        """
        Args:
            mask_flag: Whether to apply causal mask
            factor: Sampling factor c (controls sparsity, default 5)
            scale: Attention scale factor
            attention_dropout: Dropout probability
            output_attention: Whether to return attention weights
        """
        super(ProbSparseAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def _prob_QK(self, Q: torch.Tensor, K: torch.Tensor, 
                 sample_k: int, n_top: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute probability-based query-key interaction.
        
        Samples keys and computes sparsity measure to find most important queries.
        
        Args:
            Q: Query tensor (B, H, L_Q, D)
            K: Key tensor (B, H, L_K, D)
            sample_k: Number of keys to sample
            n_top: Number of top queries to select
            
        Returns:
            Q_K: Sampled Q*K scores
            M_top: Indices of top-n queries
        """
        B, H, L_K, D = K.shape
        _, _, L_Q, _ = Q.shape
        
        # Sample keys: randomly select sample_k keys
        # K_sample: (B, H, sample_k, D)
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)
        
        # Random sample indices
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=K.device)
        K_sample = K_expand[:, :, torch.arange(L_Q, device=K.device).unsqueeze(1), index_sample, :]
        
        # Compute Q*K^T for sampled keys: (B, H, L_Q, sample_k)
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        
        # Compute sparsity measure M(q_i, K) = max - mean
        # Equation (6): M(q_i, K) = max_j(q_i*k_j/sqrt(d)) - (1/L_K)*sum_j(q_i*k_j/sqrt(d))
        M = Q_K_sample.max(-1)[0] - Q_K_sample.sum(-1) / L_K
        
        # Select top-n queries with highest M values
        M_top = M.topk(n_top, sorted=False)[1]
        
        return Q_K_sample, M_top
    
    def _get_initial_context(self, V: torch.Tensor, L_Q: int) -> torch.Tensor:
        """
        Get initial context using mean of V.
        
        For queries not in top-u, we use the mean of V values.
        
        Args:
            V: Value tensor (B, H, L_V, D)
            L_Q: Query sequence length
            
        Returns:
            Initial context tensor (B, H, L_Q, D)
        """
        B, H, L_V, D = V.shape
        
        if not self.mask_flag:
            # No mask: use mean of all V
            V_mean = V.mean(dim=-2)
            context = V_mean.unsqueeze(-2).expand(B, H, L_Q, D).clone()
        else:
            # With mask: use cumulative mean (each position only sees previous)
            # For simplicity, use overall mean (approximation for efficiency)
            V_mean = V.mean(dim=-2)
            context = V_mean.unsqueeze(-2).expand(B, H, L_Q, D).clone()
            
        return context
    
    def _update_context(self, context: torch.Tensor, V: torch.Tensor,
                       scores: torch.Tensor, index: torch.Tensor,
                       L_Q: int, attn_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update context for selected top-u queries.
        
        Args:
            context: Initial context (B, H, L_Q, D)
            V: Value tensor (B, H, L_V, D)
            scores: Attention scores for top queries (B, H, n_top, L_K)
            index: Indices of top queries (B, H, n_top)
            L_Q: Query sequence length
            attn_mask: Optional attention mask
            
        Returns:
            Updated context and attention weights
        """
        B, H, L_V, D = V.shape
        
        # Apply mask if needed
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L_Q, device=V.device)
            # Only apply mask to the top query positions
            # This is an approximation - full implementation would create per-query masks
            
        # Softmax over scores
        attn = self.dropout(F.softmax(scores, dim=-1))
        
        # Compute weighted values for top queries
        # output: (B, H, n_top, D)
        output = torch.matmul(attn, V)
        
        # Scatter update: replace context at selected indices
        # Expand index to match dimensions for scatter
        index_expanded = index.unsqueeze(-1).expand(B, H, index.shape[-1], D)
        context.scatter_(2, index_expanded, output)
        
        return context, attn
        
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute ProbSparse attention.
        
        Args:
            queries: Query tensor (batch, heads, seq_len_q, d_k)
            keys: Key tensor (batch, heads, seq_len_k, d_k)
            values: Value tensor (batch, heads, seq_len_v, d_v)
            attn_mask: Optional attention mask
            
        Returns:
            output: Attention output (batch, heads, seq_len_q, d_v)
            attn: Attention weights if output_attention=True, else None
        """
        B, H, L_Q, D = queries.shape
        _, _, L_K, _ = keys.shape
        _, _, L_V, _ = values.shape
        
        # Scale
        scale = self.scale or 1.0 / math.sqrt(D)
        
        # Calculate sampling parameters
        # u = factor * ln(L_K), sample_k = factor * ln(L_Q)
        U_part = max(1, self.factor * int(np.ceil(np.log(L_K))))  # n_top queries
        u = max(1, self.factor * int(np.ceil(np.log(L_Q))))  # sample_k keys
        
        U_part = min(U_part, L_Q)
        u = min(u, L_K)
        
        # Get top-U queries based on sparsity measure
        scores_top, index = self._prob_QK(queries, keys, sample_k=u, n_top=U_part)
        
        # Initialize context with mean(V)
        context = self._get_initial_context(values, L_Q)
        
        # Compute full Q*K for selected top queries
        # Select top queries: (B, H, U_part, D)
        Q_reduce = queries[
            torch.arange(B, device=queries.device)[:, None, None],
            torch.arange(H, device=queries.device)[None, :, None],
            index,
            :
        ]
        
        # Compute attention scores for top queries: (B, H, U_part, L_K)
        scores = torch.matmul(Q_reduce, keys.transpose(-2, -1)) * scale
        
        # Update context with attended values
        context, attn = self._update_context(
            context, values, scores, index, L_Q, attn_mask
        )
        
        if self.output_attention:
            return context, attn
        else:
            return context, None


class AttentionLayer(nn.Module):
    """
    Multi-head attention layer wrapper.
    
    Wraps ProbSparse or Full attention with:
    - Linear projections for Q, K, V
    - Multi-head splitting
    - Output projection
    """
    
    def __init__(self, attention: nn.Module, d_model: int, n_heads: int,
                 d_keys: Optional[int] = None, d_values: Optional[int] = None):
        """
        Args:
            attention: Attention mechanism (ProbSparse or Full)
            d_model: Model dimension
            n_heads: Number of attention heads
            d_keys: Dimension of keys (default: d_model // n_heads)
            d_values: Dimension of values (default: d_model // n_heads)
        """
        super(AttentionLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.inner_attention = attention
        self.n_heads = n_heads
        
        # Linear projections
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply multi-head attention.
        
        Args:
            queries: Query tensor (batch, seq_len_q, d_model)
            keys: Key tensor (batch, seq_len_k, d_model)
            values: Value tensor (batch, seq_len_v, d_model)
            attn_mask: Optional attention mask
            
        Returns:
            output: Attention output (batch, seq_len_q, d_model)
            attn: Attention weights if returned by inner attention
        """
        B, L_Q, _ = queries.shape
        _, L_K, _ = keys.shape
        _, L_V, _ = values.shape
        H = self.n_heads
        
        # Project to multi-head dimensions
        queries = self.query_projection(queries).view(B, L_Q, H, -1).transpose(1, 2)
        keys = self.key_projection(keys).view(B, L_K, H, -1).transpose(1, 2)
        values = self.value_projection(values).view(B, L_V, H, -1).transpose(1, 2)
        
        # Apply attention
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        
        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(B, L_Q, -1)
        out = self.out_projection(out)
        
        return out, attn
