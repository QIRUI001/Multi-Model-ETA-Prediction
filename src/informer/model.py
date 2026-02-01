"""
Informer Model for Ship ETA Prediction.

Main architecture components:
1. Encoder: Multiple attention blocks with distilling (Conv1d + MaxPool)
2. Decoder: Multi-head attention combining encoder output with masked predictions
3. Informer: Complete model combining encoder and decoder

Reference: Informer-TP (Xiong et al., 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .attention import ProbSparseAttention, FullAttention, AttentionLayer
from .embed import DataEmbedding, DataEmbedding_wo_pos


class ConvLayer(nn.Module):
    """
    Convolutional layer for encoder distilling.
    
    Applies 1D convolution followed by ELU activation and max pooling
    to reduce sequence length by half while extracting features.
    
    Equation (8): X^t_{j+1} = MaxPool(ELU(Conv1d([X^t_j]_AB)))
    """
    
    def __init__(self, d_model: int):
        """
        Args:
            d_model: Model dimension (number of channels)
        """
        super(ConvLayer, self).__init__()
        
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode='circular'
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ELU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution, activation, and pooling.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Output tensor (batch, seq_len // 2, d_model)
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)  # (batch, d_model, seq_len)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len // 2, d_model)
        return x


class EncoderLayer(nn.Module):
    """
    Single encoder layer.
    
    Consists of:
    1. Multi-head ProbSparse Self-Attention
    2. Feedforward network (two linear layers with activation)
    3. Layer normalization and residual connections
    """
    
    def __init__(self, attention: AttentionLayer, d_model: int, d_ff: int = None,
                 dropout: float = 0.1, activation: str = 'relu'):
        """
        Args:
            attention: Attention layer (with ProbSparse or Full attention)
            d_model: Model dimension
            d_ff: Feedforward hidden dimension (default: 4 * d_model)
            dropout: Dropout probability
            activation: Activation function ('relu' or 'gelu')
        """
        super(EncoderLayer, self).__init__()
        
        d_ff = d_ff or 4 * d_model
        
        self.attention = attention
        
        # Feedforward network
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Activation
        self.activation = F.relu if activation == 'relu' else F.gelu
        
    def forward(self, x: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through encoder layer.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            attn_mask: Optional attention mask
            
        Returns:
            output: Encoded tensor (batch, seq_len, d_model)
            attn: Attention weights if available
        """
        # Self-attention with residual connection
        new_x, attn = self.attention(x, x, x, attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        
        # Feedforward with residual connection
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, -2))))
        y = self.dropout(self.conv2(y).transpose(-1, -2))
        
        output = self.norm2(x + y)
        
        return output, attn


class InformerEncoder(nn.Module):
    """
    Informer Encoder with Distilling.
    
    Stacks multiple encoder layers with optional convolution layers
    between them for dimension reduction (distilling operation).
    
    The distilling reduces sequence length by half at each level,
    creating a hierarchical structure that captures multi-scale features.
    """
    
    def __init__(self, attn_layers: List[EncoderLayer], conv_layers: Optional[List[ConvLayer]] = None,
                 norm_layer: Optional[nn.Module] = None):
        """
        Args:
            attn_layers: List of encoder layers
            conv_layers: List of convolution layers for distilling (one less than attn_layers)
            norm_layer: Final layer normalization
        """
        super(InformerEncoder, self).__init__()
        
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        
    def forward(self, x: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            attn_mask: Optional attention mask
            
        Returns:
            output: Encoded tensor (batch, reduced_seq_len, d_model)
            attns: List of attention weights from each layer
        """
        attns = []
        
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            # Last attention layer without conv
            x, attn = self.attn_layers[-1](x, attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask)
                attns.append(attn)
                
        if self.norm is not None:
            x = self.norm(x)
            
        return x, attns


class DecoderLayer(nn.Module):
    """
    Single decoder layer.
    
    Consists of:
    1. Masked Multi-head Self-Attention (ProbSparse)
    2. Multi-head Cross-Attention with encoder output
    3. Feedforward network
    4. Layer normalization and residual connections
    """
    
    def __init__(self, self_attention: AttentionLayer, cross_attention: AttentionLayer,
                 d_model: int, d_ff: int = None, dropout: float = 0.1, 
                 activation: str = 'relu'):
        """
        Args:
            self_attention: Self-attention layer
            cross_attention: Cross-attention layer
            d_model: Model dimension
            d_ff: Feedforward hidden dimension
            dropout: Dropout probability
            activation: Activation function
        """
        super(DecoderLayer, self).__init__()
        
        d_ff = d_ff or 4 * d_model
        
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        
        # Feedforward network
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.activation = F.relu if activation == 'relu' else F.gelu
        
    def forward(self, x: torch.Tensor, cross: torch.Tensor,
                x_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through decoder layer.
        
        Args:
            x: Decoder input (batch, seq_len_dec, d_model)
            cross: Encoder output (batch, seq_len_enc, d_model)
            x_mask: Self-attention mask (causal)
            cross_mask: Cross-attention mask
            
        Returns:
            output: Decoded tensor (batch, seq_len_dec, d_model)
        """
        # Masked self-attention
        x_attn, _ = self.self_attention(x, x, x, x_mask)
        x = x + self.dropout(x_attn)
        x = self.norm1(x)
        
        # Cross-attention with encoder output
        x_cross, _ = self.cross_attention(x, cross, cross, cross_mask)
        x = x + self.dropout(x_cross)
        x = self.norm2(x)
        
        # Feedforward
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, -2))))
        y = self.dropout(self.conv2(y).transpose(-1, -2))
        
        output = self.norm3(x + y)
        
        return output


class InformerDecoder(nn.Module):
    """
    Informer Decoder.
    
    Stacks multiple decoder layers to generate predictions.
    Takes encoder output and decoder input (start token + zero placeholders).
    """
    
    def __init__(self, layers: List[DecoderLayer], norm_layer: Optional[nn.Module] = None,
                 projection: Optional[nn.Module] = None):
        """
        Args:
            layers: List of decoder layers
            norm_layer: Final layer normalization
            projection: Output projection layer
        """
        super(InformerDecoder, self).__init__()
        
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        
    def forward(self, x: torch.Tensor, cross: torch.Tensor,
                x_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            x: Decoder input (batch, seq_len_dec, d_model)
            cross: Encoder output (batch, seq_len_enc, d_model)
            x_mask: Self-attention mask
            cross_mask: Cross-attention mask
            
        Returns:
            output: Decoded output (batch, seq_len_dec, output_dim)
        """
        for layer in self.layers:
            x = layer(x, cross, x_mask, cross_mask)
            
        if self.norm is not None:
            x = self.norm(x)
            
        if self.projection is not None:
            x = self.projection(x)
            
        return x


class Informer(nn.Module):
    """
    Informer Model for Ship ETA Prediction.
    
    Complete architecture combining:
    - Input embedding (scalar projection + positional + temporal)
    - Encoder with ProbSparse attention and distilling
    - Decoder with cross-attention
    - Output projection for prediction
    
    Input format:
    - x_enc: Encoder input (batch, seq_len_enc, enc_in) - historical trajectory
    - x_mark_enc: Encoder time features (batch, seq_len_enc, time_features)
    - x_dec: Decoder input (batch, seq_len_dec, dec_in) - start token + zeros
    - x_mark_dec: Decoder time features (batch, seq_len_dec, time_features)
    
    For ETA prediction:
    - enc_in = 4 (lat, lon, sog, cog)
    - dec_in = 4 (same features)
    - c_out = 1 (remaining time to arrival) or 4 (full trajectory prediction)
    """
    
    def __init__(self, enc_in: int = 4, dec_in: int = 4, c_out: int = 1,
                 seq_len: int = 96, label_len: int = 48, pred_len: int = 24,
                 d_model: int = 512, n_heads: int = 8, e_layers: int = 2,
                 d_layers: int = 1, d_ff: int = 2048, factor: int = 5,
                 dropout: float = 0.1, attn: str = 'prob', embed: str = 'fixed',
                 freq: str = 't', activation: str = 'gelu',
                 output_attention: bool = False, distil: bool = True,
                 device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        """
        Args:
            enc_in: Number of encoder input features (default 4: lat, lon, sog, cog)
            dec_in: Number of decoder input features
            c_out: Number of output features (1 for ETA, 4 for trajectory)
            seq_len: Encoder input sequence length
            label_len: Start token length (overlap between encoder and decoder)
            pred_len: Prediction sequence length
            d_model: Model dimension
            n_heads: Number of attention heads
            e_layers: Number of encoder layers
            d_layers: Number of decoder layers
            d_ff: Feedforward hidden dimension
            factor: ProbSparse sampling factor
            dropout: Dropout probability
            attn: Attention type ('prob' for ProbSparse, 'full' for standard)
            embed: Embedding type ('fixed' or 'learned')
            freq: Time frequency for temporal embedding
            activation: Activation function ('relu' or 'gelu')
            output_attention: Whether to output attention weights
            distil: Whether to use distilling in encoder
            device: Device to run model on
        """
        super(Informer, self).__init__()
        
        self.pred_len = pred_len
        self.label_len = label_len
        self.attn = attn
        self.output_attention = output_attention
        
        # Input Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        
        # Select attention mechanism
        if attn == 'prob':
            Attn = ProbSparseAttention
        else:
            Attn = FullAttention
            
        # Encoder
        self.encoder = InformerEncoder(
            attn_layers=[
                EncoderLayer(
                    AttentionLayer(
                        Attn(mask_flag=False, factor=factor, attention_dropout=dropout,
                             output_attention=output_attention),
                        d_model, n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            conv_layers=[ConvLayer(d_model) for _ in range(e_layers - 1)] if distil else None,
            norm_layer=nn.LayerNorm(d_model)
        )
        
        # Decoder
        self.decoder = InformerDecoder(
            layers=[
                DecoderLayer(
                    # Self-attention (masked)
                    AttentionLayer(
                        Attn(mask_flag=True, factor=factor, attention_dropout=dropout,
                             output_attention=False),
                        d_model, n_heads
                    ),
                    # Cross-attention (full attention to encoder)
                    AttentionLayer(
                        FullAttention(mask_flag=False, factor=factor, attention_dropout=dropout,
                                     output_attention=False),
                        d_model, n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )
        
        self.to(device)
        
    def forward(self, x_enc: torch.Tensor, x_mark_enc: Optional[torch.Tensor],
                x_dec: torch.Tensor, x_mark_dec: Optional[torch.Tensor],
                enc_self_mask: Optional[torch.Tensor] = None,
                dec_self_mask: Optional[torch.Tensor] = None,
                dec_enc_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Informer.
        
        Args:
            x_enc: Encoder input (batch, seq_len, enc_in)
            x_mark_enc: Encoder time features (batch, seq_len, time_features) or None
            x_dec: Decoder input (batch, label_len + pred_len, dec_in)
            x_mark_dec: Decoder time features (batch, label_len + pred_len, time_features) or None
            enc_self_mask: Encoder self-attention mask
            dec_self_mask: Decoder self-attention mask
            dec_enc_mask: Decoder-encoder cross-attention mask
            
        Returns:
            output: Predictions (batch, pred_len, c_out)
        """
        # Encode input sequence
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, enc_self_mask)
        
        # Decode to generate predictions
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, dec_self_mask, dec_enc_mask)
        
        # Return only the prediction part (last pred_len steps)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
    
    def predict(self, x_enc: torch.Tensor, x_mark_enc: Optional[torch.Tensor] = None,
                x_mark_dec: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Convenience method for inference.
        
        Automatically constructs decoder input with start token and zero padding.
        
        Args:
            x_enc: Encoder input (batch, seq_len, enc_in)
            x_mark_enc: Encoder time features (optional)
            x_mark_dec: Decoder time features (optional)
            
        Returns:
            Predictions (batch, pred_len, c_out)
        """
        batch_size = x_enc.shape[0]
        device = x_enc.device
        
        # Create decoder input: [start_token (last label_len of encoder), zeros (pred_len)]
        # Start token is the last label_len points from encoder input
        start_token = x_enc[:, -self.label_len:, :]
        
        # Zero padding for prediction positions
        zeros = torch.zeros(batch_size, self.pred_len, x_enc.shape[-1], device=device)
        
        # Concatenate: (batch, label_len + pred_len, enc_in)
        x_dec = torch.cat([start_token, zeros], dim=1)
        
        return self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)


class InformerForETA(Informer):
    """
    Informer specialized for ETA (Estimated Time of Arrival) prediction.
    
    Configured for:
    - Input: lat, lon, sog, cog (4 features)
    - Output: remaining time to arrival (1 feature)
    
    Uses MSE loss for training.
    """
    
    def __init__(self, seq_len: int = 96, label_len: int = 48, pred_len: int = 1,
                 d_model: int = 512, n_heads: int = 8, e_layers: int = 2,
                 d_layers: int = 1, d_ff: int = 2048, factor: int = 5,
                 dropout: float = 0.1, device: torch.device = None):
        """
        Args:
            seq_len: Historical sequence length
            label_len: Start token length
            pred_len: Prediction length (1 for single ETA, more for trajectory)
            d_model: Model dimension
            n_heads: Number of attention heads
            e_layers: Number of encoder layers
            d_layers: Number of decoder layers
            d_ff: Feedforward dimension
            factor: ProbSparse factor
            dropout: Dropout rate
            device: Compute device
        """
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
        super(InformerForETA, self).__init__(
            enc_in=4,  # lat, lon, sog, cog
            dec_in=4,
            c_out=1,   # ETA (remaining time)
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            factor=factor,
            dropout=dropout,
            attn='prob',
            embed='fixed',
            freq='t',
            activation='gelu',
            output_attention=False,
            distil=True,
            device=device
        )
        
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss for ETA prediction.
        
        Args:
            pred: Predicted ETA (batch, pred_len, 1)
            target: True ETA (batch, pred_len, 1)
            
        Returns:
            MSE loss value
        """
        return F.mse_loss(pred, target)


class InformerForTrajectory(Informer):
    """
    Informer specialized for trajectory prediction.
    
    Configured for:
    - Input: lat, lon, sog, cog (4 features)
    - Output: lat, lon, sog, cog (4 features) for future positions
    
    Uses MSE loss with optional Haversine distance loss for lat/lon.
    """
    
    def __init__(self, seq_len: int = 96, label_len: int = 48, pred_len: int = 24,
                 d_model: int = 512, n_heads: int = 8, e_layers: int = 2,
                 d_layers: int = 1, d_ff: int = 2048, factor: int = 5,
                 dropout: float = 0.1, device: torch.device = None):
        """
        Args:
            seq_len: Historical sequence length
            label_len: Start token length
            pred_len: Prediction horizon length
            d_model: Model dimension
            n_heads: Number of attention heads
            e_layers: Number of encoder layers
            d_layers: Number of decoder layers
            d_ff: Feedforward dimension
            factor: ProbSparse factor
            dropout: Dropout rate
            device: Compute device
        """
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
        super(InformerForTrajectory, self).__init__(
            enc_in=4,  # lat, lon, sog, cog
            dec_in=4,
            c_out=4,   # lat, lon, sog, cog predictions
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            factor=factor,
            dropout=dropout,
            attn='prob',
            embed='fixed',
            freq='t',
            activation='gelu',
            output_attention=False,
            distil=True,
            device=device
        )
        
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss for trajectory prediction.
        
        Args:
            pred: Predicted trajectory (batch, pred_len, 4)
            target: True trajectory (batch, pred_len, 4)
            
        Returns:
            MSE loss value
        """
        return F.mse_loss(pred, target)
    
    @staticmethod
    def haversine_distance(pred_lat: torch.Tensor, pred_lon: torch.Tensor,
                          true_lat: torch.Tensor, true_lon: torch.Tensor) -> torch.Tensor:
        """
        Compute Haversine distance between predicted and true positions.
        
        Args:
            pred_lat, pred_lon: Predicted coordinates (in degrees)
            true_lat, true_lon: True coordinates (in degrees)
            
        Returns:
            Distance in kilometers
        """
        R = 6371.0  # Earth's radius in km
        
        # Convert to radians
        lat1 = torch.deg2rad(pred_lat)
        lat2 = torch.deg2rad(true_lat)
        dlat = torch.deg2rad(true_lat - pred_lat)
        dlon = torch.deg2rad(true_lon - pred_lon)
        
        # Haversine formula
        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        
        return R * c
