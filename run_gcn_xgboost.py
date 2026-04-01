#!/usr/bin/env python
"""
Q3 Reviewer Ablation: Stats + GCN Embeddings + XGBoost

Motivation: The reviewer asks whether MSTGN's gains come from the graph
information itself or from the neural fusion head combined with KD strategy.

This script tests: XGBoost using the SAME features as MSTGN-MLP2's input
(81 statistical + 96 GCN cell embedding features = 177-dim), trained from the
GCN embeddings of a pre-trained MSTGN-MLP2.

If XGBoost + GCN ≈ MSTGN-MLP2 → the MLP head adds nothing beyond tree-level capacity
If XGBoost + GCN < MSTGN-MLP2 → neural fusion + KD provide meaningful additional value

Usage:
    python run_gcn_xgboost.py
    python run_gcn_xgboost.py --ckpt output/mstgn_mlp2_kd06/best_mstgn.pth
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path

from src.mstgn.model import MSTGN_MLP2


def inverse_normalize_target(normalized, target_mean, target_std):
    log_target = normalized * target_std + target_mean
    return np.expm1(log_target)


def calculate_metrics(y_pred, y_true):
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mask = y_true > 24
    mape = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0
    return {'MAE_hours': mae, 'RMSE': rmse, 'MAPE': mape}


def extract_stat_features(X):
    """Extract same 81 statistical + 4 cross features as MSTGN-MLP2.

    X: (N, 48, 11) float32
    Returns: (N, 85) feature array
    """
    last = X[:, -1, :]        # (N, 11)
    first = X[:, 0, :]        # (N, 11)
    mean = X.mean(axis=1)     # (N, 11)
    std = X.std(axis=1)       # (N, 11)
    diff = last - first       # (N, 11)
    xmin = X.min(axis=1)      # (N, 11)
    xmax = X.max(axis=1)      # (N, 11)
    recent_mean = X[:, -12:, :].mean(axis=1)  # (N, 11)

    # Cross features: sog=col2, dist=col4, bearing_diff=col5
    sog_x_dist = (last[:, 2] * last[:, 4]).reshape(-1, 1)
    dist_sq = (last[:, 4] ** 2).reshape(-1, 1)
    sog_x_bearing = (last[:, 2] * last[:, 5]).reshape(-1, 1)
    sog_accel_x_dist = (diff[:, 2] * last[:, 4]).reshape(-1, 1)

    return np.hstack([last, mean, std, diff, xmin, xmax, recent_mean,
                      sog_x_dist, dist_sq, sog_x_bearing, sog_accel_x_dist])


def extract_gcn_features(cell_emb_np, cell_ids):
    """Lookup GCN embeddings for each sample.

    cell_emb_np: (N_nodes, 32) numpy array
    cell_ids: (N_samples, 48) int numpy array
    Returns: (N_samples, 96) feature array
    """
    current = cell_emb_np[cell_ids[:, -1]]           # (N, 32)
    origin = cell_emb_np[cell_ids[:, 0]]             # (N, 32)
    route_mean = cell_emb_np[cell_ids].mean(axis=1)  # (N, 32)
    return np.hstack([current, origin, route_mean])


def main():
    parser = argparse.ArgumentParser(description='XGBoost + GCN embeddings ablation (Q3)')
    parser.add_argument('--ckpt', default='output/mstgn_mlp2_kd06/best_mstgn.pth',
                        help='Path to trained MSTGN-MLP2 checkpoint.')
    parser.add_argument('--graph_dir', default='output/graph')
    parser.add_argument('--cache_dir',
                        default='output/cache_sequences/seq48_label24_pred1_mv150000_ms50000000')
    parser.add_argument('--cell_ids_dir', default=None,
                        help='Directory with cell_ids_*.npy. Defaults to cache_dir.')
    parser.add_argument('--norm_path', default='output/norm_params.npz')
    parser.add_argument('--output_dir', default='output/xgb_gcn')
    parser.add_argument('--n_estimators', type=int, default=1000)
    parser.add_argument('--early_stopping', type=int, default=30)
    args = parser.parse_args()

    try:
        import xgboost as xgb
    except ImportError:
        print("xgboost not installed. Run: pip install xgboost")
        return

    import urllib.request

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cell_ids_dir = Path(args.cell_ids_dir) if args.cell_ids_dir else cache_dir
    graph_dir = Path(args.graph_dir)

    # ---- Load normalization ----
    norm = np.load(args.norm_path, allow_pickle=True)
    target_mean = float(norm['target_mean'])
    target_std = float(norm['target_std'])

    # ---- Load graph ----
    adj = np.load(graph_dir / 'adj_normalized.npy')
    node_features = np.load(graph_dir / 'node_features.npy')
    with open(graph_dir / 'graph_meta.json') as f:
        graph_meta = json.load(f)
    print(f"Graph: {graph_meta['num_nodes']} nodes, {graph_meta['cell_size']}° grid")

    # ---- Load MSTGN-MLP2 and extract GCN embeddings ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = MSTGN_MLP2(
        adj_matrix=adj,
        init_node_features=node_features,
        seq_feat_dim=11,
        gcn_hidden=64,
        cell_emb_dim=32,
    )
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    with torch.no_grad():
        h = model.gcn1(model.node_features, model.adj)
        h = model.gcn_drop(h)
        cell_emb = model.gcn2(h, model.adj)
    cell_emb_np = cell_emb.cpu().numpy()
    print(f"GCN cell embeddings: {cell_emb_np.shape}")

    # ---- Load sample counts ----
    counts = np.load(cache_dir / 'actual_counts.npy', allow_pickle=True).item()

    # ---- Feature extraction ----
    def get_features(split):
        n = counts[split]
        X = np.load(cache_dir / f'X_{split}.npy', mmap_mode='r')[:n]
        cell_ids = np.load(cell_ids_dir / f'cell_ids_{split}.npy')[:n]
        y = np.load(cache_dir / f'y_{split}.npy', mmap_mode='r')[:n]

        CHUNK = 200_000
        stat_chunks, gcn_chunks = [], []
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            X_chunk = np.array(X[start:end], dtype=np.float32)
            stat_chunks.append(extract_stat_features(X_chunk))
            gcn_chunks.append(extract_gcn_features(cell_emb_np, cell_ids[start:end]))

        stat_feats = np.vstack(stat_chunks)
        gcn_feats = np.vstack(gcn_chunks)
        combined = np.hstack([stat_feats, gcn_feats])
        return combined, np.array(y)

    print("Extracting features...")
    X_train, y_train = get_features('train')
    X_val, y_val = get_features('val')
    X_test, y_test = get_features('test')
    print(f"Feature shape: {X_train.shape}  (stat={85}, gcn={96})")

    # ---- Baseline: same stats without GCN ----
    print("\n--- Training XGBoost (Stats only, 85-dim) ---")
    dtrain_stat = xgb.DMatrix(X_train[:, :85], label=y_train)
    dval_stat = xgb.DMatrix(X_val[:, :85], label=y_val)
    dtest_stat = xgb.DMatrix(X_test[:, :85])

    params_base = {
        'objective': 'reg:squarederror',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'tree_method': 'hist',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'eval_metric': 'mae',
    }

    bst_stat = xgb.train(
        params_base, dtrain_stat,
        num_boost_round=args.n_estimators,
        evals=[(dtrain_stat, 'train'), (dval_stat, 'val')],
        early_stopping_rounds=args.early_stopping,
        verbose_eval=100,
    )
    y_pred_stat_norm = bst_stat.predict(dtest_stat)
    y_pred_stat = inverse_normalize_target(y_pred_stat_norm, target_mean, target_std)
    y_pred_stat = np.maximum(y_pred_stat, 0)
    y_true = inverse_normalize_target(y_test, target_mean, target_std)

    m_stat = calculate_metrics(y_pred_stat, y_true)
    print(f"\nXGBoost (Stats-only):   MAE={m_stat['MAE_hours']:.2f}h  RMSE={m_stat['RMSE']:.2f}h  MAPE={m_stat['MAPE']:.2f}%")

    # ---- Main: Stats + GCN ----
    print("\n--- Training XGBoost (Stats + GCN, 177-dim) ---")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    bst = xgb.train(
        params_base, dtrain,
        num_boost_round=args.n_estimators,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=args.early_stopping,
        verbose_eval=100,
    )
    y_pred_norm = bst.predict(dtest)
    y_pred = inverse_normalize_target(y_pred_norm, target_mean, target_std)
    y_pred = np.maximum(y_pred, 0)

    m = calculate_metrics(y_pred, y_true)
    print(f"\nXGBoost (Stats + GCN):  MAE={m['MAE_hours']:.2f}h  RMSE={m['RMSE']:.2f}h  MAPE={m['MAPE']:.2f}%")

    # ---- Save results ----
    results = {
        'xgb_stats_only': {k: float(v) for k, v in m_stat.items()},
        'xgb_stats_gcn': {k: float(v) for k, v in m.items()},
        'feature_dims': {'stats': 85, 'gcn': 96, 'combined': 181},
        'checkpoint': str(args.ckpt),
        'graph_meta': graph_meta,
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/results.json")

    summary = (
        f"📊 Q3 Ablation: XGBoost + GCN Embeddings\n"
        f"XGBoost (Stats-only, 85-d): MAE={m_stat['MAE_hours']:.2f}h\n"
        f"XGBoost (Stats+GCN, 177-d): MAE={m['MAE_hours']:.2f}h\n"
        f"(MSTGN-MLP2 w/ KD: 15.13h for reference)"
    )
    print(f"\n{summary}")

    webhook_url = "https://discord.com/api/webhooks/1477180434474336266/HQSS_BKlo1Ib-rzwItazg8ay0To2l-GR1GprnTvlPVpLxCxD9aBd4iHNgX"
    data = json.dumps({"content": summary}).encode('utf-8')
    req = urllib.request.Request(webhook_url, data=data,
                                 headers={'Content-Type': 'application/json'})
    try:
        urllib.request.urlopen(req)
    except Exception:
        pass


if __name__ == '__main__':
    main()
