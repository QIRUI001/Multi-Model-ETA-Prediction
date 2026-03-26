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


class StatMLP(nn.Module):
    """Statistical feature MLP without graph — ablation baseline.

    Same features as MSTGN_MLP (last, mean, std, diff) but no graph.
    """

    def __init__(self, seq_feat_dim=11, dropout=0.1, **kwargs):
        super().__init__()
        stat_dim = seq_feat_dim * 4
        self.head = nn.Sequential(
            nn.Linear(stat_dim, 512),
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
        last = x[:, -1, :]
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        diff = last - x[:, 0, :]
        stats = torch.cat([last, mean, std, diff], dim=-1)
        return self.head(stats).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MSTGN_Hybrid(nn.Module):
    """Hybrid model: GRU temporal + statistical features + graph context.

    Combines three complementary information sources:
    1. GRU: temporal patterns (acceleration, weather trends)
    2. Statistics: tabular summaries (like XGBoost features)
    3. Graph: spatial context from route graph
    """

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

        # Sequence branch (raw features)
        self.gru = nn.GRU(
            seq_feat_dim, gru_hidden, num_layers=gru_layers,
            batch_first=True, dropout=dropout if gru_layers > 1 else 0
        )
        self.attn_pool = AttentionPooling(gru_hidden)

        # Fusion: temporal(gru) + stats(4*feat) + graph(2*cell_emb)
        stat_dim = seq_feat_dim * 4
        fusion_dim = gru_hidden + stat_dim + cell_emb_dim * 2
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
        # Graph
        h = self.gcn1(self.node_features, self.adj)
        h = self.gcn_drop(h)
        cell_emb = self.gcn2(h, self.adj)

        # Temporal
        gru_out, _ = self.gru(x)
        seq_repr = self.attn_pool(gru_out)

        # Statistical
        last = x[:, -1, :]
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        diff = last - x[:, 0, :]
        stats = torch.cat([last, mean, std, diff], dim=-1)

        # Graph context
        current_cell = cell_emb[cell_ids[:, -1]]
        route_context = cell_emb[cell_ids].mean(dim=1)

        fused = torch.cat([seq_repr, stats, current_cell, route_context], dim=-1)
        return self.head(fused).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HybridNoGraph(nn.Module):
    """GRU temporal + statistical features, no graph — ablation baseline."""

    def __init__(self, seq_feat_dim=11, seq_len=48,
                 gru_hidden=256, gru_layers=2,
                 dropout=0.1, **kwargs):
        super().__init__()

        self.gru = nn.GRU(
            seq_feat_dim, gru_hidden, num_layers=gru_layers,
            batch_first=True, dropout=dropout if gru_layers > 1 else 0
        )
        self.attn_pool = AttentionPooling(gru_hidden)

        stat_dim = seq_feat_dim * 4
        fusion_dim = gru_hidden + stat_dim
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
        gru_out, _ = self.gru(x)
        seq_repr = self.attn_pool(gru_out)

        last = x[:, -1, :]
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        diff = last - x[:, 0, :]
        stats = torch.cat([last, mean, std, diff], dim=-1)

        fused = torch.cat([seq_repr, stats], dim=-1)
        return self.head(fused).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    """Pre-norm residual block: LN → Linear → SiLU → Drop → Linear + skip."""

    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.activation(x + self.net(x)))


class MSTGN_V2(nn.Module):
    """Enhanced MSTGN: rich statistical features + residual MLP + stronger GCN.

    Improvements over MSTGN_MLP:
    1. Rich features: 9 aggregations (last, first, mean, std, diff, min, max,
       recent_mean, recent_std) + explicit cross-features
    2. 3-layer GCN with residual connections, wider embeddings
    3. 3 cell contexts: current, origin, route-mean
    4. Residual MLP blocks with LayerNorm + SiLU
    """

    def __init__(self, adj_matrix, init_node_features,
                 seq_feat_dim=11, seq_len=48,
                 gcn_hidden=128, cell_emb_dim=64,
                 hidden_dim=512, num_blocks=3,
                 dropout=0.15):
        super().__init__()

        node_feat_dim = init_node_features.shape[1]

        # === Graph branch: 3-layer GCN with residual ===
        self.register_buffer('adj', torch.from_numpy(adj_matrix).float())
        self.node_features = nn.Parameter(
            torch.from_numpy(init_node_features).float()
        )
        self.gcn1 = GCNLayer(node_feat_dim, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_hidden)
        self.gcn3 = GCNLayer(gcn_hidden, cell_emb_dim)
        self.gcn_drop = nn.Dropout(dropout)

        # Feature dimensions:
        #   9 aggregations × 11 features = 99
        #   + 4 cross features
        #   + 3 × cell_emb_dim graph context
        stat_dim = seq_feat_dim * 9 + 4
        total_dim = stat_dim + cell_emb_dim * 3

        # === Residual MLP ===
        self.input_proj = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, 1)
        )

        self._init_weights()

    def _init_weights(self):
        gcn_linears = {self.gcn1.linear, self.gcn2.linear, self.gcn3.linear}
        for m in self.modules():
            if isinstance(m, nn.Linear) and m not in gcn_linears:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _extract_features(self, x):
        """Extract rich statistical features from sequence.

        x: (batch, 48, 11)
        Returns: (batch, 99 + 4) = (batch, 103)
        """
        last = x[:, -1, :]           # (B, 11)
        first = x[:, 0, :]           # (B, 11)
        mean = x.mean(dim=1)         # (B, 11)
        std = x.std(dim=1)           # (B, 11)
        diff = last - first           # (B, 11)
        xmin = x.min(dim=1).values   # (B, 11)
        xmax = x.max(dim=1).values   # (B, 11)

        # Recent window (last 12 steps)
        recent = x[:, -12:, :]
        recent_mean = recent.mean(dim=1)  # (B, 11)
        recent_std = recent.std(dim=1)    # (B, 11)

        # Cross features (indices: sog=2, dist=4, bearing_diff=5)
        sog_last = last[:, 2:3]
        dist_last = last[:, 4:5]
        bearing_last = last[:, 5:6]
        sog_x_dist = sog_last * dist_last
        dist_sq = dist_last * dist_last
        sog_x_bearing = sog_last * bearing_last
        sog_accel_x_dist = diff[:, 2:3] * dist_last

        return torch.cat([last, first, mean, std, diff, xmin, xmax,
                          recent_mean, recent_std,
                          sog_x_dist, dist_sq, sog_x_bearing,
                          sog_accel_x_dist], dim=-1)

    def forward(self, x, cell_ids):
        # Graph: 3-layer GCN with residual between layers 1 and 2
        h1 = self.gcn1(self.node_features, self.adj)
        h1 = self.gcn_drop(h1)
        h2 = self.gcn2(h1, self.adj) + h1  # residual
        h2 = self.gcn_drop(h2)
        cell_emb = self.gcn3(h2, self.adj)  # (N, cell_emb_dim)

        # Rich statistical features
        stats = self._extract_features(x)

        # 3 graph contexts: current position, origin, route average
        current_cell = cell_emb[cell_ids[:, -1]]
        origin_cell = cell_emb[cell_ids[:, 0]]
        route_mean = cell_emb[cell_ids].mean(dim=1)

        combined = torch.cat([stats, current_cell, origin_cell, route_mean],
                             dim=-1)

        # Residual MLP
        h = self.input_proj(combined)
        for block in self.res_blocks:
            h = block(h)
        return self.output_head(h).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
