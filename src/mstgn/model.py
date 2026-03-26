"""
Maritime Spatio-Temporal Graph Network (MSTGN).

Architecture:
1. Graph Branch: 2-layer GCN on maritime route graph → cell embeddings
2. Sequence Branch: GRU on trajectory augmented with cell embeddings
3. Fusion: attention-pooled sequence + last-step features + current cell → MLP → ETA

The route graph captures spatial structure of ocean shipping lanes,
while the GRU captures temporal dynamics of individual vessel trajectories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GCNLayer(nn.Module):
    """Graph Convolutional Layer: X' = σ(Â X W + b)."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, X, A_hat):
        """
        X: (N, in_features) node features
        A_hat: (N, N) normalized adjacency
        """
        return F.relu(A_hat @ self.linear(X))


class AttentionPooling(nn.Module):
    """Learnable attention-weighted pooling over sequence dimension."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """x: (batch, seq_len, hidden) → (batch, hidden)"""
        scores = self.query(x).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class MSTGN(nn.Module):
    """Maritime Spatio-Temporal Graph Network.

    Args:
        adj_matrix: (N, N) float32 numpy, normalized adjacency
        init_node_features: (N, F) float32 numpy, initial node features
        seq_feat_dim: input sequence feature dimension (default 11)
        seq_len: sequence length (default 48)
        gcn_hidden: GCN hidden dimension
        cell_emb_dim: cell embedding dimension (GCN output)
        gru_hidden: GRU hidden dimension
        gru_layers: number of GRU layers
        dropout: dropout rate
    """

    def __init__(self, adj_matrix, init_node_features,
                 seq_feat_dim=11, seq_len=48,
                 gcn_hidden=64, cell_emb_dim=32,
                 gru_hidden=256, gru_layers=2,
                 dropout=0.1):
        super().__init__()

        node_feat_dim = init_node_features.shape[1]
        self.num_nodes = init_node_features.shape[0]

        # === Graph branch ===
        self.register_buffer('adj', torch.from_numpy(adj_matrix).float())
        self.node_features = nn.Parameter(
            torch.from_numpy(init_node_features).float()
        )
        self.gcn1 = GCNLayer(node_feat_dim, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden, cell_emb_dim)
        self.gcn_drop = nn.Dropout(dropout)

        # === Sequence branch (augmented features) ===
        aug_dim = seq_feat_dim + cell_emb_dim
        self.gru = nn.GRU(
            aug_dim, gru_hidden, num_layers=gru_layers,
            batch_first=True, dropout=dropout if gru_layers > 1 else 0
        )
        self.attn_pool = AttentionPooling(gru_hidden)

        # === Fusion + prediction ===
        fusion_dim = gru_hidden + seq_feat_dim + cell_emb_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self._init_weights()

    def _init_weights(self):
        gcn_linears = {self.gcn1.linear, self.gcn2.linear}
        for m in self.modules():
            if isinstance(m, nn.Linear) and m not in gcn_linears:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, cell_ids):
        """
        x: (batch, seq_len, seq_feat_dim)
        cell_ids: (batch, seq_len) int64
        Returns: (batch,) predicted normalized ETA
        """
        # Graph: compute cell embeddings
        h = self.gcn1(self.node_features, self.adj)
        h = self.gcn_drop(h)
        cell_emb = self.gcn2(h, self.adj)  # (N, cell_emb_dim)

        # Look up embeddings for each trajectory step
        step_emb = cell_emb[cell_ids]  # (batch, seq_len, cell_emb_dim)

        # Sequence branch with augmented features
        x_aug = torch.cat([x, step_emb], dim=-1)
        gru_out, _ = self.gru(x_aug)
        seq_repr = self.attn_pool(gru_out)  # (batch, gru_hidden)

        # Global features from last timestep
        global_feat = x[:, -1, :]  # (batch, seq_feat_dim)
        current_cell_emb = cell_emb[cell_ids[:, -1]]  # (batch, cell_emb_dim)

        # Fusion
        fused = torch.cat([seq_repr, global_feat, current_cell_emb], dim=-1)
        return self.head(fused).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MSTGN_LateFusion(nn.Module):
    """MSTGN with late fusion: GRU processes raw features, graph context
    is fused only at the prediction layer."""

    def __init__(self, adj_matrix, init_node_features,
                 seq_feat_dim=11, seq_len=48,
                 gcn_hidden=64, cell_emb_dim=32,
                 gru_hidden=256, gru_layers=2,
                 dropout=0.1):
        super().__init__()

        node_feat_dim = init_node_features.shape[1]

        # Graph branch
        self.register_buffer('adj', torch.from_numpy(adj_matrix).float())
        self.node_features = nn.Parameter(
            torch.from_numpy(init_node_features).float()
        )
        self.gcn1 = GCNLayer(node_feat_dim, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden, cell_emb_dim)
        self.gcn_drop = nn.Dropout(dropout)

        # Sequence branch (original features)
        self.gru = nn.GRU(
            seq_feat_dim, gru_hidden, num_layers=gru_layers,
            batch_first=True, dropout=dropout if gru_layers > 1 else 0
        )
        self.attn_pool = AttentionPooling(gru_hidden)

        # Fusion: seq + route_context + current_cell + global_feat
        fusion_dim = gru_hidden + cell_emb_dim * 2 + seq_feat_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x, cell_ids):
        h = self.gcn1(self.node_features, self.adj)
        h = self.gcn_drop(h)
        cell_emb = self.gcn2(h, self.adj)

        gru_out, _ = self.gru(x)
        seq_repr = self.attn_pool(gru_out)

        route_context = cell_emb[cell_ids].mean(dim=1)
        current_cell = cell_emb[cell_ids[:, -1]]
        global_feat = x[:, -1, :]

        fused = torch.cat([seq_repr, route_context, current_cell, global_feat], dim=-1)
        return self.head(fused).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MSTGN_MLP(nn.Module):
    """Graph-Enhanced MLP: statistical features + graph cell embeddings."""

    def __init__(self, adj_matrix, init_node_features,
                 seq_feat_dim=11, seq_len=48,
                 gcn_hidden=64, cell_emb_dim=32,
                 dropout=0.1):
        super().__init__()

        node_feat_dim = init_node_features.shape[1]

        self.register_buffer('adj', torch.from_numpy(adj_matrix).float())
        self.node_features = nn.Parameter(
            torch.from_numpy(init_node_features).float()
        )
        self.gcn1 = GCNLayer(node_feat_dim, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden, cell_emb_dim)
        self.gcn_drop = nn.Dropout(dropout)

        stat_dim = seq_feat_dim * 4  # last, mean, std, diff
        total_dim = stat_dim + cell_emb_dim * 2

        self.head = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x, cell_ids):
        h = self.gcn1(self.node_features, self.adj)
        h = self.gcn_drop(h)
        cell_emb = self.gcn2(h, self.adj)

        last = x[:, -1, :]
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        diff = last - x[:, 0, :]
        stats = torch.cat([last, mean, std, diff], dim=-1)

        current_cell = cell_emb[cell_ids[:, -1]]
        route_context = cell_emb[cell_ids].mean(dim=1)

        combined = torch.cat([stats, current_cell, route_context], dim=-1)
        return self.head(combined).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
