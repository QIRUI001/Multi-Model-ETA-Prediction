"""
Informer-TP: Informer-based Model for Ship Trajectory Prediction and ETA Estimation

This package implements the Informer model architecture for long-term ship trajectory
prediction using AIS (Automatic Identification System) data.

Key Components:
- ProbSparse Self-Attention: Efficient attention mechanism with O(L log L) complexity
- Encoder with Distilling: Multi-layer encoder with dimension reduction
- Decoder: Multi-head attention combining encoder output with masked predictions
- Data Embedding: Combines scalar projection, positional encoding, and temporal features

Reference:
    Xiong, C.; Shi, H.; Li, J.; Wu, X.; Gao, R. Informer-Based Model for Long-Term 
    Ship Trajectory Prediction. J. Mar. Sci. Eng. 2024, 12, 1269.
"""

from .model import Informer, InformerEncoder, InformerDecoder
from .embed import DataEmbedding, TokenEmbedding, PositionalEncoding, TemporalEmbedding
from .attention import ProbSparseAttention, FullAttention, AttentionLayer

__all__ = [
    'Informer',
    'InformerEncoder',
    'InformerDecoder',
    'DataEmbedding',
    'TokenEmbedding',
    'PositionalEncoding',
    'TemporalEmbedding',
    'ProbSparseAttention',
    'FullAttention',
    'AttentionLayer',
]

__version__ = '1.0.0'
