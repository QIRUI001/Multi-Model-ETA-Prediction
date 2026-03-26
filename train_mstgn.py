#!/usr/bin/env python
"""
Training script for MSTGN (Maritime Spatio-Temporal Graph Network).

Uses the same preprocessed data as baselines, plus:
- Maritime route graph (adj_normalized.npy, node_features.npy)
- Cell ID mappings (cell_ids_{split}.npy)

Usage:
    python train_mstgn.py
    python train_mstgn.py --epochs 15 --gru_hidden 256 --cell_emb_dim 32
"""

import os
import sys
import gc
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.mstgn.model import MSTGN, MSTGN_LateFusion, MSTGN_MLP, MSTGN_MLP2, MSTGN_MLP3, StatMLP, MSTGN_Hybrid, HybridNoGraph, MSTGN_V2, MSTGN_FTTransformer


# ============================================================
# Utilities
# ============================================================

def inverse_normalize_target(normalized, target_mean, target_std):
    log_target = normalized * target_std + target_mean
    return np.expm1(log_target)


def calculate_metrics(y_pred, y_true):
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(mse)
    mask = y_true > 24
    mape = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0
    return {'MSE': mse, 'MAE_hours': mae, 'MAE_days': mae / 24, 'RMSE': rmse, 'MAPE': mape}


def send_discord_notification(message):
    import urllib.request
    webhook_url = "https://discord.com/api/webhooks/1477180434474336266/HQSS_BKlo1Ib-rzwItazg8ay0To2l-GR1GprnTvlPVpLxCxD9aBd4iHNgX"
    data = json.dumps({"content": message}).encode('utf-8')
    req = urllib.request.Request(webhook_url, data=data,
                                headers={'Content-Type': 'application/json'})
    try:
        urllib.request.urlopen(req)
    except Exception:
        pass


# ============================================================
# Dataset
# ============================================================

class MSTGNDataset(Dataset):
    """Dataset that provides sequence features, cell IDs, and targets."""

    def __init__(self, X_path, cell_ids_path, y_path, actual_length=None,
                 soft_targets_path=None):
        self.X = np.load(X_path, mmap_mode='r')
        self.cell_ids = np.load(cell_ids_path, mmap_mode='r')
        self.y = np.load(y_path, mmap_mode='r')
        self.soft = np.load(soft_targets_path, mmap_mode='r') if soft_targets_path else None
        self.length = actual_length if actual_length is not None else len(self.y)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        items = (
            torch.from_numpy(self.X[idx].copy()).float(),
            torch.from_numpy(self.cell_ids[idx].copy()).long(),
            torch.tensor(self.y[idx]).float()
        )
        if self.soft is not None:
            return items + (torch.tensor(self.soft[idx]).float(),)
        return items


# ============================================================
# Training
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, epochs,
                    distill_alpha=0.0):
    model.train()
    losses = []
    pbar = tqdm(loader, desc=f"Train {epoch+1}/{epochs}", leave=False)
    for batch in pbar:
        x, cell_ids, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        optimizer.zero_grad()
        pred = model(x, cell_ids)
        loss_hard = criterion(pred, y)
        if distill_alpha > 0 and len(batch) > 3:
            y_soft = batch[3].to(device)
            loss_soft = nn.functional.mse_loss(pred, y_soft)
            loss = (1 - distill_alpha) * loss_hard + distill_alpha * loss_soft
        else:
            loss = loss_hard
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return np.mean(losses)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    preds, targets = [], []
    for batch in loader:
        x, cell_ids, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        pred = model(x, cell_ids)
        losses.append(criterion(pred, y).item())
        preds.append(pred.cpu().numpy())
        targets.append(y.cpu().numpy())
    return np.mean(losses), np.concatenate(preds), np.concatenate(targets)


def main():
    parser = argparse.ArgumentParser(description='Train MSTGN')
    parser.add_argument('--cache_dir', default='output/cache_sequences/seq48_label24_pred1_mv150000_ms50000000')
    parser.add_argument('--norm_path', default='output/norm_params.npz')
    parser.add_argument('--graph_dir', default='output/graph')
    parser.add_argument('--output_dir', default='output/mstgn')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=4)
    # Model hyperparameters
    parser.add_argument('--gcn_hidden', type=int, default=64)
    parser.add_argument('--cell_emb_dim', type=int, default=32)
    parser.add_argument('--gru_hidden', type=int, default=256)
    parser.add_argument('--gru_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--variant', type=str, default='gru',
                        choices=['gru', 'late_fusion', 'mlp', 'mlp2', 'mlp3', 'stat_mlp', 'hybrid', 'hybrid_no_graph', 'v2', 'ft_transformer'],
                        help='Model variant')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_blocks', type=int, default=3)
    # FT-Transformer hyperparameters
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--ffn_dim', type=int, default=256)
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine'],
                        help='LR scheduler type')
    parser.add_argument('--loss', type=str, default='huber',
                        choices=['huber', 'mse'],
                        help='Loss function')
    parser.add_argument('--swa', action='store_true',
                        help='Enable Stochastic Weight Averaging')
    parser.add_argument('--swa_start', type=int, default=5,
                        help='Epoch to start SWA')
    parser.add_argument('--swa_lr', type=float, default=1e-4,
                        help='SWA learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for AdamW')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    # Knowledge distillation
    parser.add_argument('--distill', action='store_true',
                        help='Enable knowledge distillation from ensemble soft targets')
    parser.add_argument('--distill_alpha', type=float, default=0.5,
                        help='Weight for soft target loss (0=hard only, 1=soft only)')
    parser.add_argument('--soft_targets_dir', type=str, default=None,
                        help='Directory containing y_soft_{split}.npy files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Device: {device}")
    print(f"Args: {vars(args)}")

    # ---- Load normalization params ----
    norm = np.load(args.norm_path, allow_pickle=True)
    target_mean = float(norm['target_mean'])
    target_std = float(norm['target_std'])

    # ---- Load graph ----
    graph_dir = Path(args.graph_dir)
    adj = np.load(graph_dir / 'adj_normalized.npy')
    node_features = np.load(graph_dir / 'node_features.npy')
    with open(graph_dir / 'graph_meta.json') as f:
        graph_meta = json.load(f)

    print(f"Graph: {graph_meta['num_nodes']} nodes, "
          f"{graph_meta['num_active_cells']} active cells, "
          f"cell_size={graph_meta['cell_size']}°")

    # ---- Load data ----
    cache_dir = Path(args.cache_dir)
    counts = np.load(cache_dir / 'actual_counts.npy', allow_pickle=True).item()

    # Soft targets for knowledge distillation
    soft_dir = None
    if args.distill:
        soft_dir = Path(args.soft_targets_dir) if args.soft_targets_dir else cache_dir / 'soft_targets'
        if not (soft_dir / 'y_soft_train.npy').exists():
            print(f"ERROR: Soft targets not found in {soft_dir}. Run generate_soft_targets.py first.")
            sys.exit(1)
        print(f"Knowledge distillation enabled: alpha={args.distill_alpha}, soft_dir={soft_dir}")

    train_ds = MSTGNDataset(
        cache_dir / 'X_train.npy', cache_dir / 'cell_ids_train.npy',
        cache_dir / 'y_train.npy', counts['train'],
        soft_targets_path=soft_dir / 'y_soft_train.npy' if soft_dir else None
    )
    val_ds = MSTGNDataset(
        cache_dir / 'X_val.npy', cache_dir / 'cell_ids_val.npy',
        cache_dir / 'y_val.npy', counts['val'],
        soft_targets_path=soft_dir / 'y_soft_val.npy' if soft_dir else None
    )
    test_ds = MSTGNDataset(
        cache_dir / 'X_test.npy', cache_dir / 'cell_ids_test.npy',
        cache_dir / 'y_test.npy', counts['test']
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             num_workers=args.num_workers, pin_memory=True)

    print(f"Data: train={counts['train']:,}, val={counts['val']:,}, test={counts['test']:,}")
    print(f"Sequence features: {train_ds.X.shape[-1]}, sequence length: {train_ds.X.shape[1]}")

    # ---- Create model ----
    model_cls = {
        'gru': MSTGN, 'late_fusion': MSTGN_LateFusion,
        'mlp': MSTGN_MLP, 'mlp2': MSTGN_MLP2, 'mlp3': MSTGN_MLP3, 'stat_mlp': StatMLP,
        'hybrid': MSTGN_Hybrid, 'hybrid_no_graph': HybridNoGraph,
        'v2': MSTGN_V2, 'ft_transformer': MSTGN_FTTransformer,
    }[args.variant]

    no_graph_variants = {'stat_mlp', 'hybrid_no_graph'}
    no_gru_variants = {'mlp', 'mlp2', 'mlp3', 'stat_mlp', 'v2', 'ft_transformer'}

    if args.variant in no_graph_variants:
        model_kwargs = dict(
            seq_feat_dim=train_ds.X.shape[-1],
            dropout=args.dropout,
        )
        if args.variant not in no_gru_variants:
            model_kwargs.update(gru_hidden=args.gru_hidden, gru_layers=args.gru_layers)
    elif args.variant == 'v2':
        model_kwargs = dict(
            adj_matrix=adj,
            init_node_features=node_features,
            seq_feat_dim=train_ds.X.shape[-1],
            seq_len=train_ds.X.shape[1],
            gcn_hidden=args.gcn_hidden,
            cell_emb_dim=args.cell_emb_dim,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
        )
    elif args.variant == 'ft_transformer':
        model_kwargs = dict(
            adj_matrix=adj,
            init_node_features=node_features,
            seq_feat_dim=train_ds.X.shape[-1],
            seq_len=train_ds.X.shape[1],
            gcn_hidden=args.gcn_hidden,
            cell_emb_dim=args.cell_emb_dim,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            ffn_dim=args.ffn_dim,
            dropout=args.dropout,
        )
    else:
        model_kwargs = dict(
            adj_matrix=adj,
            init_node_features=node_features,
            seq_feat_dim=train_ds.X.shape[-1],
            seq_len=train_ds.X.shape[1],
            gcn_hidden=args.gcn_hidden,
            cell_emb_dim=args.cell_emb_dim,
            dropout=args.dropout,
        )
        if args.variant not in no_gru_variants:
            model_kwargs.update(gru_hidden=args.gru_hidden, gru_layers=args.gru_layers)
    model = model_cls(**model_kwargs).to(device)

    variant_name = f"MSTGN-{args.variant}"
    print(f"{variant_name} parameters: {model.count_parameters():,}")

    # ---- Training ----
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        use_cosine = True
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
        use_cosine = False
    criterion = nn.HuberLoss(delta=1.0) if args.loss == 'huber' else nn.MSELoss()

    # SWA setup
    swa_model = None
    if args.swa:
        from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)
        print(f"SWA enabled: start_epoch={args.swa_start}, swa_lr={args.swa_lr}")

    best_val = float('inf')
    best_state = None
    no_improve = 0
    train_losses, val_losses = [], []

    print(f"\n{'='*60}")
    print(f"Training {variant_name} for {args.epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     device, epoch, args.epochs,
                                     distill_alpha=args.distill_alpha if args.distill else 0.0)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        if args.swa and epoch >= args.swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        elif use_cosine:
            scheduler.step()
        else:
            scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]['lr']
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        improved = ""
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            improved = " ★"
            torch.save(best_state, Path(args.output_dir) / 'best_mstgn.pth')
        else:
            no_improve += 1

        print(f"  Epoch {epoch+1}/{args.epochs}: "
              f"Train={train_loss:.4f}, Val={val_loss:.4f}, "
              f"LR={lr_now:.2e}{improved}")

        if no_improve >= args.patience and not (args.swa and epoch < args.epochs - 1):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # ---- Evaluate best model ----
    print(f"\n{'='*60}")
    print("Evaluating best model on test set")
    print(f"{'='*60}")

    if swa_model is not None:
        # Manual BN update for SWA (update_bn doesn't handle multi-input DataLoaders)
        print("  Updating SWA batch norm statistics...")
        swa_model.train()
        with torch.no_grad():
            for batch in tqdm(train_loader, desc='SWA BN', leave=False):
                x, cell_ids = batch[0].to(device), batch[1].to(device)
                swa_model(x, cell_ids)
        swa_model.eval()
        _, y_pred_norm, y_true_norm = evaluate(swa_model, test_loader, criterion, device)
        # Also evaluate non-SWA best for comparison
        model.load_state_dict(best_state)
        model.to(device)
        _, y_pred_base, _ = evaluate(model, test_loader, criterion, device)
        y_pred_base_h = inverse_normalize_target(y_pred_base, target_mean, target_std)
        y_true_h = inverse_normalize_target(y_true_norm, target_mean, target_std)
        base_mae = np.mean(np.abs(np.maximum(y_pred_base_h, 0) - y_true_h))
        print(f"  Best-epoch MAE: {base_mae:.2f}h")
    else:
        model.load_state_dict(best_state)
        model.to(device)
        _, y_pred_norm, y_true_norm = evaluate(model, test_loader, criterion, device)

    y_pred = inverse_normalize_target(y_pred_norm, target_mean, target_std)
    y_true = inverse_normalize_target(y_true_norm, target_mean, target_std)
    y_pred = np.maximum(y_pred, 0)

    metrics = calculate_metrics(y_pred, y_true)
    print(f"\n{variant_name} Test Results:")
    print(f"  MAE:  {metrics['MAE_hours']:.2f} hours ({metrics['MAE_days']:.2f} days)")
    print(f"  RMSE: {metrics['RMSE']:.2f} hours")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")

    # Save results
    results = {
        'model': variant_name,
        'metrics': {k: float(v) for k, v in metrics.items()},
        'args': vars(args),
        'graph_meta': graph_meta,
        'num_parameters': model.count_parameters(),
        'best_val_loss': float(best_val),
        'epochs_trained': len(train_losses),
        'timestamp': datetime.now().isoformat(),
    }
    with open(Path(args.output_dir) / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save training curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(train_losses)+1), train_losses, label='Train', marker='o')
    ax.plot(range(1, len(val_losses)+1), val_losses, label='Val', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Huber Loss')
    ax.set_title(f'{variant_name} Training Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(Path(args.output_dir) / f'training_curve_{args.variant}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Save predictions for analysis
    np.savez(Path(args.output_dir) / 'predictions.npz',
             y_pred=y_pred, y_true=y_true,
             y_pred_norm=y_pred_norm, y_true_norm=y_true_norm)

    # Discord notification
    msg = (f"🚢 {variant_name} Training Complete!\n"
           f"MAE: {metrics['MAE_hours']:.2f}h | "
           f"RMSE: {metrics['RMSE']:.2f}h | "
           f"MAPE: {metrics['MAPE']:.2f}%\n"
           f"Graph: {graph_meta['num_nodes']} nodes, "
           f"{graph_meta['cell_size']}° cells\n"
           f"Params: {model.count_parameters():,}")
    send_discord_notification(msg)

    print(f"\nResults saved to {args.output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
