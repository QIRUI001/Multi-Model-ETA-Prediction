"""
Utility functions for Informer model.

Includes:
- Triangular causal mask for decoder
- Probability mask for ProbSparse attention
- Helper functions for tensor operations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


def triangular_causal_mask(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Generate triangular causal mask for decoder self-attention.
    
    The mask prevents attention to future positions by setting them to -inf.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        device: Device to create tensor on
        
    Returns:
        Mask tensor of shape (batch_size, 1, seq_len, seq_len)
        True values indicate positions to mask (set to -inf)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    mask = mask.expand(batch_size, 1, seq_len, seq_len)
    return mask


def prob_mask(batch_size: int, heads: int, seq_len_q: int, seq_len_k: int, 
              top_u: int, device: torch.device) -> torch.Tensor:
    """
    Generate probability mask for ProbSparse attention.
    
    Args:
        batch_size: Batch size
        heads: Number of attention heads
        seq_len_q: Query sequence length
        seq_len_k: Key sequence length
        top_u: Number of top queries to select
        device: Device to create tensor on
        
    Returns:
        Mask indicating which query positions are active
    """
    mask = torch.ones(batch_size, heads, seq_len_q, seq_len_k, device=device)
    return mask


class TriangularCausalMask:
    """
    Triangular causal mask wrapper class.
    
    Provides a lazy-evaluated mask that is created when first accessed.
    """
    
    def __init__(self, batch_size: int, seq_len: int, device: torch.device = torch.device('cpu')):
        self._mask = None
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        
    @property
    def mask(self) -> torch.Tensor:
        if self._mask is None:
            self._mask = triangular_causal_mask(
                self.batch_size, self.seq_len, self.device
            )
        return self._mask


class ProbMask:
    """
    Probability mask for ProbSparse self-attention.
    
    Used to mask out non-selected queries in the sparse attention mechanism.
    """
    
    def __init__(self, batch_size: int, heads: int, seq_len_q: int, seq_len_k: int,
                 index: torch.Tensor, scores: torch.Tensor, 
                 device: torch.device = torch.device('cpu')):
        """
        Args:
            batch_size: Batch size
            heads: Number of attention heads  
            seq_len_q: Query sequence length
            seq_len_k: Key sequence length
            index: Indices of top-u queries
            scores: Attention scores for top-u queries
            device: Device to create tensor on
        """
        # Create mask initialized to -inf (masked out)
        _mask = torch.ones(seq_len_q, scores.shape[-1], dtype=torch.bool, device=device)
        _mask = _mask.triu(1)  # Upper triangular mask
        
        # Expand to batch and head dimensions
        _mask = _mask.unsqueeze(0).unsqueeze(0).expand(batch_size, heads, -1, -1)
        self._mask = _mask
        
    @property
    def mask(self) -> torch.Tensor:
        return self._mask


def get_activation(activation: str) -> nn.Module:
    """
    Get activation function by name.
    
    Args:
        activation: Name of activation function ('relu', 'gelu', 'elu')
        
    Returns:
        Activation module
    """
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'elu': nn.ELU(),
    }
    return activations.get(activation.lower(), nn.ReLU())


def clone_module(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Create n clones of a module.
    
    Args:
        module: Module to clone
        n: Number of clones
        
    Returns:
        ModuleList containing n clones
    """
    return nn.ModuleList([module for _ in range(n)])


def attention_mask_to_float(mask: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Convert boolean attention mask to float mask.
    
    True values become -inf (masked), False values become 0 (not masked).
    
    Args:
        mask: Boolean mask tensor
        dtype: Output dtype
        
    Returns:
        Float mask tensor
    """
    return mask.float().masked_fill(mask, float('-inf')).to(dtype)


def subsequent_mask(size: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Create a mask for subsequent positions (used in decoder).
    
    Args:
        size: Sequence length
        device: Device to create tensor on
        
    Returns:
        Lower triangular mask of shape (1, size, size)
    """
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    return mask.unsqueeze(0) == 0


def calculate_output_length(input_len: int, kernel_size: int, stride: int = 1, 
                           padding: int = 0, dilation: int = 1) -> int:
    """
    Calculate output length after convolution.
    
    Args:
        input_len: Input sequence length
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Padding size
        dilation: Dilation factor
        
    Returns:
        Output sequence length
    """
    return (input_len + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
