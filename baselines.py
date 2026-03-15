#!/usr/bin/env python
"""
Baseline models for ETA prediction comparison.

Models:
1. LSTM
2. GRU
3. MLP (Feedforward)
4. XGBoost (gradient boosting on hand-crafted features)
5. Linear Regression (simplest baseline)
6. Vanilla Transformer (encoder-only)
7. TCN (Temporal Convolutional Network)
8. ConvTransformer (CNN + Transformer hybrid)
9. CNN-1D (pure convolutional)

All baselines use the same preprocessed data as the Informer model
(loaded from cached memmap arrays).
"""

import os
import sys
import gc
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# Utilities
# ============================================================

def inverse_normalize_target(normalized, target_mean, target_std):
    """Reverse log-standardization of target values."""
    log_target = normalized * target_std + target_mean
    return np.expm1(log_target)


def calculate_metrics(y_pred, y_true):
    """Compute MSE, MAE, RMSE, MAPE."""
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(mse)
    mask = y_true > 24
    mape = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0
    return {'MSE': mse, 'MAE_hours': mae, 'MAE_days': mae / 24, 'RMSE': rmse, 'MAPE': mape}


def load_data(cache_dir, norm_path, batch_size=4096, num_workers=8):
    """Load cached memmap data and normalization params."""
    cache_dir = Path(cache_dir)

    # Load actual counts
    counts = np.load(cache_dir / "actual_counts.npy", allow_pickle=True).item()
    train_n, val_n, test_n = counts['train'], counts['val'], counts['test']

    # Load normalization parameters
    norm = np.load(norm_path, allow_pickle=True)
    target_mean = float(norm['target_mean'])
    target_std = float(norm['target_std'])

    # Load data via memmap
    X_train = np.load(cache_dir / "X_train.npy", mmap_mode='r')[:train_n]
    y_train = np.load(cache_dir / "y_train.npy", mmap_mode='r')[:train_n]
    X_val = np.load(cache_dir / "X_val.npy", mmap_mode='r')[:val_n]
    y_val = np.load(cache_dir / "y_val.npy", mmap_mode='r')[:val_n]
    X_test = np.load(cache_dir / "X_test.npy", mmap_mode='r')[:test_n]
    y_test = np.load(cache_dir / "y_test.npy", mmap_mode='r')[:test_n]
    sd_test = np.load(cache_dir / "sd_test.npy", mmap_mode='r')[:test_n]

    print(f"Data loaded: train={train_n:,}, val={val_n:,}, test={test_n:,}")
    print(f"Feature shape: {X_train.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, sd_test, target_mean, target_std


# ============================================================
# Model Definitions
# ============================================================

class LSTMModel(nn.Module):
    """LSTM baseline for sequence-to-one regression."""
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, (h_n, _) = self.lstm(x)
        # Use last hidden state
        return self.fc(h_n[-1]).squeeze(-1)


class GRUModel(nn.Module):
    """GRU baseline for sequence-to-one regression."""
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, h_n = self.gru(x)
        return self.fc(h_n[-1]).squeeze(-1)


class MLPModel(nn.Module):
    """MLP baseline: flatten sequence then feedforward."""
    def __init__(self, seq_len, input_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        flat_dim = seq_len * input_dim
        layers = []
        prev = flat_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.1)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.reshape(x.size(0), -1)).squeeze(-1)


class TransformerModel(nn.Module):
    """Vanilla Transformer encoder for sequence-to-one regression."""
    def __init__(self, input_dim, seq_len=48, d_model=256, nhead=8, num_layers=3, d_ff=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_proj(x) + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)
        # Global average pooling over sequence
        x = x.mean(dim=1)
        return self.fc(x).squeeze(-1)


class TCNBlock(nn.Module):
    """Single TCN residual block with dilated causal convolution."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # x: (batch, channels, seq_len)
        residual = self.downsample(x)
        out = self.dropout(torch.relu(self.bn1(self.conv1(x)[:, :, :x.size(2)])))
        out = self.dropout(torch.relu(self.bn2(self.conv2(out)[:, :, :x.size(2)])))
        return torch.relu(out + residual)


class TCNModel(nn.Module):
    """Temporal Convolutional Network for sequence-to-one regression."""
    def __init__(self, input_dim, num_channels=None, kernel_size=3, dropout=0.1):
        super().__init__()
        if num_channels is None:
            num_channels = [128, 128, 256, 256]
        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels[-1], 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.network(x)
        return self.fc(x).squeeze(-1)


class ConvTransformerModel(nn.Module):
    """CNN feature extractor + Transformer encoder hybrid."""
    def __init__(self, input_dim, seq_len=48, d_model=256, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        # Conv feature extractor
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = self.conv(x)       # (batch, d_model, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, d_model)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x).squeeze(-1)


class CNN1DModel(nn.Module):
    """1D CNN for sequence-to-one regression."""
    def __init__(self, input_dim, seq_len=48):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4, 256),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        return self.fc(x).squeeze(-1)


# ============================================================
# Training & Evaluation
# ============================================================

def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-3, patience=3, model_name="model"):
    """Generic training loop for PyTorch models."""
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    criterion = nn.HuberLoss(delta=1.0)

    best_val = float('inf')
    no_improve = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{epochs}", leave=False):
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                pred = model(x)
                val_losses.append(criterion(pred, y).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"  [{model_name}] Epoch {epoch+1}/{epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}, LR={lr_now:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [{model_name}] Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model


def predict_model(model, loader, device):
    """Run inference and return predictions."""
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            pred = model(x)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds)


def run_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """XGBoost baseline using last-step features + statistical features."""
    try:
        import xgboost as xgb
    except ImportError:
        print("xgboost not installed, skipping. Install with: pip install xgboost")
        return None

    def extract_features(X):
        """Extract hand-crafted features from sequences."""
        # X: (N, seq_len, n_features)
        last = X[:, -1, :]           # last step features
        first = X[:, 0, :]          # first step features
        mean = X.mean(axis=1)       # mean over sequence
        std = X.std(axis=1)         # std over sequence
        diff = last - first         # change over sequence
        return np.hstack([last, mean, std, diff])

    print("  [XGBoost] Extracting features...")
    # Need to load into memory for feature extraction
    X_tr_feat = extract_features(np.array(X_train))
    X_val_feat = extract_features(np.array(X_val))
    X_te_feat = extract_features(np.array(X_test))

    print(f"  [XGBoost] Feature shape: {X_tr_feat.shape}")

    dtrain = xgb.DMatrix(X_tr_feat, label=np.array(y_train))
    dval = xgb.DMatrix(X_val_feat, label=np.array(y_val))
    dtest = xgb.DMatrix(X_te_feat)

    params = {
        'objective': 'reg:squarederror',
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'tree_method': 'hist',
        'device': 'cpu',
        'eval_metric': 'mae',
    }

    print("  [XGBoost] Training...")
    bst = xgb.train(
        params, dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=20,
        verbose_eval=50
    )

    y_pred_norm = bst.predict(dtest)
    return y_pred_norm


def run_linear_regression(X_train, y_train, X_test):
    """Simple linear regression on last-step features."""
    from sklearn.linear_model import Ridge

    # Use last step + mean features
    def extract(X):
        last = np.array(X[:, -1, :])
        mean = np.array(X.mean(axis=1))
        return np.hstack([last, mean])

    print("  [LinearReg] Extracting features...")
    X_tr = extract(X_train)
    X_te = extract(X_test)

    print(f"  [LinearReg] Feature shape: {X_tr.shape}")
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0)
    model.fit(X_tr, np.array(y_train))
    return model.predict(X_te)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Baseline models for ETA prediction")
    parser.add_argument('--cache_dir', type=str, default='./output/processed/cache')
    parser.add_argument('--norm_path', type=str, default='./output/norm_params.npz')
    parser.add_argument('--output_dir', type=str, default='./output/baselines')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--models', type=str, default='all',
                        help='Comma-separated: lstm,gru,mlp,xgboost,linear or all')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, sd_test, target_mean, target_std = \
        load_data(args.cache_dir, args.norm_path, args.batch_size, args.num_workers)

    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]

    models_to_run = args.models.split(',') if args.models != 'all' else ['lstm', 'gru', 'mlp', 'xgboost', 'linear', 'transformer', 'tcn', 'convtransformer', 'cnn1d']

    # Create simple dataloaders (sequence features + target only)
    train_ds = TensorDataset(torch.FloatTensor(np.array(X_train)), torch.FloatTensor(np.array(y_train)))
    val_ds = TensorDataset(torch.FloatTensor(np.array(X_val)), torch.FloatTensor(np.array(y_val)))
    test_ds = TensorDataset(torch.FloatTensor(np.array(X_test)), torch.FloatTensor(np.array(y_test)))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    results = {}

    # ---- LSTM ----
    if 'lstm' in models_to_run:
        print("\n" + "="*50)
        print("Training LSTM baseline...")
        print("="*50)
        lstm = LSTMModel(n_features, hidden_dim=256, num_layers=2, dropout=0.1)
        print(f"  Parameters: {sum(p.numel() for p in lstm.parameters()):,}")
        lstm = train_model(lstm, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, patience=args.patience, model_name="LSTM")
        y_pred_norm = predict_model(lstm, test_loader, device)
        y_pred = inverse_normalize_target(y_pred_norm, target_mean, target_std)
        y_true = inverse_normalize_target(np.array(y_test), target_mean, target_std)
        y_pred = np.maximum(y_pred, 0)
        metrics = calculate_metrics(y_pred, y_true)
        results['LSTM'] = metrics
        print(f"  LSTM Results: MAE={metrics['MAE_hours']:.2f}h, RMSE={metrics['RMSE']:.2f}h, MAPE={metrics['MAPE']:.2f}%")
        del lstm; gc.collect(); torch.cuda.empty_cache()

    # ---- GRU ----
    if 'gru' in models_to_run:
        print("\n" + "="*50)
        print("Training GRU baseline...")
        print("="*50)
        gru = GRUModel(n_features, hidden_dim=256, num_layers=2, dropout=0.1)
        print(f"  Parameters: {sum(p.numel() for p in gru.parameters()):,}")
        gru = train_model(gru, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, patience=args.patience, model_name="GRU")
        y_pred_norm = predict_model(gru, test_loader, device)
        y_pred = inverse_normalize_target(y_pred_norm, target_mean, target_std)
        y_true = inverse_normalize_target(np.array(y_test), target_mean, target_std)
        y_pred = np.maximum(y_pred, 0)
        metrics = calculate_metrics(y_pred, y_true)
        results['GRU'] = metrics
        print(f"  GRU Results: MAE={metrics['MAE_hours']:.2f}h, RMSE={metrics['RMSE']:.2f}h, MAPE={metrics['MAPE']:.2f}%")
        del gru; gc.collect(); torch.cuda.empty_cache()

    # ---- MLP ----
    if 'mlp' in models_to_run:
        print("\n" + "="*50)
        print("Training MLP baseline...")
        print("="*50)
        mlp = MLPModel(seq_len, n_features, hidden_dims=[512, 256, 128])
        print(f"  Parameters: {sum(p.numel() for p in mlp.parameters()):,}")
        mlp = train_model(mlp, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, patience=args.patience, model_name="MLP")
        y_pred_norm = predict_model(mlp, test_loader, device)
        y_pred = inverse_normalize_target(y_pred_norm, target_mean, target_std)
        y_true = inverse_normalize_target(np.array(y_test), target_mean, target_std)
        y_pred = np.maximum(y_pred, 0)
        metrics = calculate_metrics(y_pred, y_true)
        results['MLP'] = metrics
        print(f"  MLP Results: MAE={metrics['MAE_hours']:.2f}h, RMSE={metrics['RMSE']:.2f}h, MAPE={metrics['MAPE']:.2f}%")
        del mlp; gc.collect(); torch.cuda.empty_cache()

    # ---- XGBoost ----
    if 'xgboost' in models_to_run:
        print("\n" + "="*50)
        print("Training XGBoost baseline...")
        print("="*50)
        y_pred_norm = run_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)
        if y_pred_norm is not None:
            y_pred = inverse_normalize_target(y_pred_norm, target_mean, target_std)
            y_true = inverse_normalize_target(np.array(y_test), target_mean, target_std)
            y_pred = np.maximum(y_pred, 0)
            metrics = calculate_metrics(y_pred, y_true)
            results['XGBoost'] = metrics
            print(f"  XGBoost Results: MAE={metrics['MAE_hours']:.2f}h, RMSE={metrics['RMSE']:.2f}h, MAPE={metrics['MAPE']:.2f}%")

    # ---- Linear Regression ----
    if 'linear' in models_to_run:
        print("\n" + "="*50)
        print("Training Linear Regression baseline...")
        print("="*50)
        y_pred_norm = run_linear_regression(X_train, y_train, X_test)
        y_pred = inverse_normalize_target(y_pred_norm, target_mean, target_std)
        y_true = inverse_normalize_target(np.array(y_test), target_mean, target_std)
        y_pred = np.maximum(y_pred, 0)
        metrics = calculate_metrics(y_pred, y_true)
        results['Linear'] = metrics
        print(f"  Linear Results: MAE={metrics['MAE_hours']:.2f}h, RMSE={metrics['RMSE']:.2f}h, MAPE={metrics['MAPE']:.2f}%")

    # ---- Vanilla Transformer ----
    if 'transformer' in models_to_run:
        print("\n" + "="*50)
        print("Training Transformer baseline...")
        print("="*50)
        transformer = TransformerModel(n_features, seq_len=seq_len, d_model=256, nhead=8, num_layers=3, d_ff=512, dropout=0.1)
        print(f"  Parameters: {sum(p.numel() for p in transformer.parameters()):,}")
        transformer = train_model(transformer, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, patience=args.patience, model_name="Transformer")
        y_pred_norm = predict_model(transformer, test_loader, device)
        y_pred = inverse_normalize_target(y_pred_norm, target_mean, target_std)
        y_true = inverse_normalize_target(np.array(y_test), target_mean, target_std)
        y_pred = np.maximum(y_pred, 0)
        metrics = calculate_metrics(y_pred, y_true)
        results['Transformer'] = metrics
        print(f"  Transformer Results: MAE={metrics['MAE_hours']:.2f}h, RMSE={metrics['RMSE']:.2f}h, MAPE={metrics['MAPE']:.2f}%")
        del transformer; gc.collect(); torch.cuda.empty_cache()

    # ---- TCN ----
    if 'tcn' in models_to_run:
        print("\n" + "="*50)
        print("Training TCN baseline...")
        print("="*50)
        tcn = TCNModel(n_features, num_channels=[128, 128, 256, 256], kernel_size=3, dropout=0.1)
        print(f"  Parameters: {sum(p.numel() for p in tcn.parameters()):,}")
        tcn = train_model(tcn, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, patience=args.patience, model_name="TCN")
        y_pred_norm = predict_model(tcn, test_loader, device)
        y_pred = inverse_normalize_target(y_pred_norm, target_mean, target_std)
        y_true = inverse_normalize_target(np.array(y_test), target_mean, target_std)
        y_pred = np.maximum(y_pred, 0)
        metrics = calculate_metrics(y_pred, y_true)
        results['TCN'] = metrics
        print(f"  TCN Results: MAE={metrics['MAE_hours']:.2f}h, RMSE={metrics['RMSE']:.2f}h, MAPE={metrics['MAPE']:.2f}%")
        del tcn; gc.collect(); torch.cuda.empty_cache()

    # ---- ConvTransformer ----
    if 'convtransformer' in models_to_run:
        print("\n" + "="*50)
        print("Training ConvTransformer baseline...")
        print("="*50)
        convtf = ConvTransformerModel(n_features, seq_len=seq_len, d_model=256, nhead=8, num_layers=2, dropout=0.1)
        print(f"  Parameters: {sum(p.numel() for p in convtf.parameters()):,}")
        convtf = train_model(convtf, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, patience=args.patience, model_name="ConvTransformer")
        y_pred_norm = predict_model(convtf, test_loader, device)
        y_pred = inverse_normalize_target(y_pred_norm, target_mean, target_std)
        y_true = inverse_normalize_target(np.array(y_test), target_mean, target_std)
        y_pred = np.maximum(y_pred, 0)
        metrics = calculate_metrics(y_pred, y_true)
        results['ConvTransformer'] = metrics
        print(f"  ConvTransformer Results: MAE={metrics['MAE_hours']:.2f}h, RMSE={metrics['RMSE']:.2f}h, MAPE={metrics['MAPE']:.2f}%")
        del convtf; gc.collect(); torch.cuda.empty_cache()

    # ---- CNN-1D ----
    if 'cnn1d' in models_to_run:
        print("\n" + "="*50)
        print("Training CNN-1D baseline...")
        print("="*50)
        cnn = CNN1DModel(n_features, seq_len=seq_len)
        print(f"  Parameters: {sum(p.numel() for p in cnn.parameters()):,}")
        cnn = train_model(cnn, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, patience=args.patience, model_name="CNN-1D")
        y_pred_norm = predict_model(cnn, test_loader, device)
        y_pred = inverse_normalize_target(y_pred_norm, target_mean, target_std)
        y_true = inverse_normalize_target(np.array(y_test), target_mean, target_std)
        y_pred = np.maximum(y_pred, 0)
        metrics = calculate_metrics(y_pred, y_true)
        results['CNN-1D'] = metrics
        print(f"  CNN-1D Results: MAE={metrics['MAE_hours']:.2f}h, RMSE={metrics['RMSE']:.2f}h, MAPE={metrics['MAPE']:.2f}%")
        del cnn; gc.collect(); torch.cuda.empty_cache()

    # ===== Summary =====
    print("\n" + "="*60)
    print("BASELINE COMPARISON RESULTS")
    print("="*60)
    print(f"{'Model':<15} {'MAE(h)':<10} {'RMSE(h)':<10} {'MAPE(%)':<10} {'MAE(d)':<10}")
    print("-" * 55)
    for name, m in results.items():
        print(f"{name:<15} {m['MAE_hours']:<10.2f} {m['RMSE']:<10.2f} {m['MAPE']:<10.2f} {m['MAE_days']:<10.4f}")

    # Save results
    with open(os.path.join(args.output_dir, 'baseline_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/baseline_results.json")

    # Discord notification
    try:
        import urllib.request
        webhook_url = "https://discord.com/api/webhooks/1477180434474336266/HQSS_BKlo1Ib-rzwItazg8ay0To2l-GR1GprnTvlPVpLxCxD9xu6_iEPsD9aBd4iHNgX"
        lines = ["**Baseline Experiments Complete** ✅"]
        for name, m in results.items():
            lines.append(f"  {name}: MAE={m['MAE_hours']:.2f}h, RMSE={m['RMSE']:.2f}h, MAPE={m['MAPE']:.2f}%")
        payload = json.dumps({"content": "\n".join(lines)}).encode()
        req = urllib.request.Request(webhook_url, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"Discord notification failed: {e}")


if __name__ == '__main__':
    main()
