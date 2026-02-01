"""
Input Embedding Module for Informer.

The input representation combines:
1. TokenEmbedding (Scalar Projection): Projects input features to d_model dimensions via Conv1d
2. PositionalEncoding (Local Timestamps): Fixed sinusoidal position embeddings
3. TemporalEmbedding (Global Timestamps): Learnable embeddings for time features
   (minute, hour, weekday, day, month)

Reference: Section 2.2.1 of Informer-TP paper
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding.
    
    Uses sine and cosine functions of different frequencies to encode position:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    This allows the model to learn relative positions through linear projections.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length to pre-compute
            dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the division term
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but should be saved with model)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """
    Scalar projection via 1D convolution.
    
    Projects input features (e.g., lat, lon, sog, cog) to d_model dimensions
    using a 1D convolution layer. This serves as the initial feature transformation.
    """
    
    def __init__(self, input_dim: int, d_model: int, kernel_size: int = 3):
        """
        Args:
            input_dim: Number of input features (e.g., 4 for lat, lon, sog, cog)
            d_model: Model dimension
            kernel_size: Convolution kernel size (default 3 for local context)
        """
        super(TokenEmbedding, self).__init__()
        
        # Padding to maintain sequence length
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode='circular'  # Handle edge cases smoothly
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input features to d_model dimensions.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Projected tensor of shape (batch, seq_len, d_model)
        """
        # Conv1d expects (batch, channels, seq_len), so transpose
        x = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, d_model)
        return x


class FixedEmbedding(nn.Module):
    """
    Fixed (non-learnable) embedding using sinusoidal patterns.
    
    Similar to positional encoding but for categorical time features.
    """
    
    def __init__(self, num_embeddings: int, d_model: int):
        """
        Args:
            num_embeddings: Vocabulary size (e.g., 60 for minutes)
            d_model: Embedding dimension
        """
        super(FixedEmbedding, self).__init__()
        
        # Create fixed sinusoidal embedding
        w = torch.zeros(num_embeddings, d_model).float()
        position = torch.arange(0, num_embeddings).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        
        self.embedding = nn.Embedding(num_embeddings, d_model)
        self.embedding.weight = nn.Parameter(w, requires_grad=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Long tensor of indices, shape (batch, seq_len)
            
        Returns:
            Embedding tensor of shape (batch, seq_len, d_model)
        """
        return self.embedding(x.long())


class TemporalEmbedding(nn.Module):
    """
    Global timestamp embedding for time features.
    
    Embeds multiple time granularities:
    - Minute of hour (0-59)
    - Hour of day (0-23)
    - Day of week (0-6)
    - Day of month (1-31)
    - Month of year (1-12)
    
    Each granularity has its own embedding layer, and all are summed together.
    """
    
    def __init__(self, d_model: int, embed_type: str = 'fixed', freq: str = 't'):
        """
        Args:
            d_model: Model dimension
            embed_type: 'fixed' for sinusoidal, 'learned' for learnable embeddings
            freq: Time frequency ('t' for minutely, 'h' for hourly, etc.)
        """
        super(TemporalEmbedding, self).__init__()
        
        # Choose embedding type
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        
        # Minute embedding (for minutely data)
        if freq == 't':
            self.minute_embed = Embed(60, d_model)
        else:
            self.minute_embed = None
            
        self.hour_embed = Embed(24, d_model)
        self.weekday_embed = Embed(7, d_model)
        self.day_embed = Embed(32, d_model)  # 1-31 days
        self.month_embed = Embed(13, d_model)  # 1-12 months (0 unused)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal embeddings from time features.
        
        Args:
            x: Time feature tensor of shape (batch, seq_len, num_time_features)
               Expected features: [minute, hour, weekday, day, month] or subset
               
        Returns:
            Temporal embedding of shape (batch, seq_len, d_model)
        """
        # x should contain time features in columns
        # Assuming x has shape (batch, seq_len, num_features)
        # where features are in order: minute, hour, weekday, day, month
        
        x = x.long()
        
        # Sum all temporal embeddings
        embed = torch.zeros(x.shape[0], x.shape[1], self.hour_embed.embedding.weight.shape[1],
                           device=x.device, dtype=torch.float32)
        
        feature_idx = 0
        
        if self.minute_embed is not None and x.shape[-1] > feature_idx:
            embed = embed + self.minute_embed(x[:, :, feature_idx])
            feature_idx += 1
            
        if x.shape[-1] > feature_idx:
            embed = embed + self.hour_embed(x[:, :, feature_idx])
            feature_idx += 1
            
        if x.shape[-1] > feature_idx:
            embed = embed + self.weekday_embed(x[:, :, feature_idx])
            feature_idx += 1
            
        if x.shape[-1] > feature_idx:
            embed = embed + self.day_embed(x[:, :, feature_idx])
            feature_idx += 1
            
        if x.shape[-1] > feature_idx:
            embed = embed + self.month_embed(x[:, :, feature_idx])
            
        return embed


class TimeFeatureEmbedding(nn.Module):
    """
    Alternative temporal embedding using continuous time features.
    
    Instead of discrete embeddings, projects continuous time features
    (mapped to [-0.5, 0.5]) through a linear layer.
    """
    
    def __init__(self, d_model: int, num_features: int = 5, freq: str = 't'):
        """
        Args:
            d_model: Model dimension
            num_features: Number of time features (default 5: minute, hour, weekday, day, month)
            freq: Time frequency
        """
        super(TimeFeatureEmbedding, self).__init__()
        
        # Determine number of features based on frequency
        freq_map = {'t': 5, 'h': 4, 'd': 3, 'w': 2, 'm': 1}
        d_inp = freq_map.get(freq, num_features)
        
        self.embed = nn.Linear(d_inp, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project continuous time features to d_model dimensions.
        
        Args:
            x: Time features tensor of shape (batch, seq_len, num_features)
               Values should be normalized to [-0.5, 0.5]
               
        Returns:
            Embedding tensor of shape (batch, seq_len, d_model)
        """
        return self.embed(x)


class DataEmbedding(nn.Module):
    """
    Complete input embedding combining all components.
    
    Final representation = TokenEmbedding + PositionalEncoding + TemporalEmbedding
    
    This follows Equation (3) in the paper:
    X_feed[i] = α * u_i + PE(pos, i) + Σ SE[i]_p
    """
    
    def __init__(self, input_dim: int, d_model: int, embed_type: str = 'fixed',
                 freq: str = 't', dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            input_dim: Number of input features (e.g., 4 for lat, lon, sog, cog)
            d_model: Model dimension
            embed_type: 'fixed' for sinusoidal temporal, 'learned' for learnable
            freq: Time frequency for temporal embedding
            dropout: Dropout probability
            max_len: Maximum sequence length for positional encoding
        """
        super(DataEmbedding, self).__init__()
        
        # Token embedding (scalar projection)
        self.token_embedding = TokenEmbedding(input_dim, d_model)
        
        # Positional encoding (local timestamps)
        self.position_embedding = PositionalEncoding(d_model, max_len, dropout=0.0)
        
        # Temporal embedding (global timestamps)
        self.temporal_embedding = TimeFeatureEmbedding(d_model, freq=freq)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute complete input embedding.
        
        Args:
            x: Input features tensor of shape (batch, seq_len, input_dim)
            x_mark: Time features tensor of shape (batch, seq_len, num_time_features)
                   Optional - if None, only token and position embeddings are used
                   
        Returns:
            Embedded tensor of shape (batch, seq_len, d_model)
        """
        # Token embedding (scalar projection)
        token_embed = self.token_embedding(x)
        
        # Add positional encoding
        embed = self.position_embedding(token_embed)
        
        # Add temporal embedding if time features provided
        if x_mark is not None:
            embed = embed + self.temporal_embedding(x_mark)
            
        return self.dropout(embed)


class DataEmbedding_wo_pos(nn.Module):
    """
    Data embedding without positional encoding.
    
    Useful for ablation studies or when position information is not needed.
    """
    
    def __init__(self, input_dim: int, d_model: int, embed_type: str = 'fixed',
                 freq: str = 't', dropout: float = 0.1):
        """
        Args:
            input_dim: Number of input features
            d_model: Model dimension
            embed_type: Temporal embedding type
            freq: Time frequency
            dropout: Dropout probability
        """
        super(DataEmbedding_wo_pos, self).__init__()
        
        self.token_embedding = TokenEmbedding(input_dim, d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model, freq=freq)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input features tensor
            x_mark: Time features tensor (optional)
            
        Returns:
            Embedded tensor
        """
        embed = self.token_embedding(x)
        
        if x_mark is not None:
            embed = embed + self.temporal_embedding(x_mark)
            
        return self.dropout(embed)
