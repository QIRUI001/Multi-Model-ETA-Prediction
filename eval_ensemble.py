#!/usr/bin/env python
"""Evaluate ensemble of MSTGN models by averaging predictions."""
import numpy as np
import json
import sys
from pathlib import Path


def calc_metrics(y_pred, y_true):
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mask = y_true > 24
    mape = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100
    return mae, rmse, mape


def main():
    ensemble_dir = Path('output/ensemble')

    # Auto-detect all trained seeds
    seed_dirs = sorted(ensemble_dir.glob('seed*'))
    seeds = []
    for d in seed_dirs:
        pred_path = d / 'predictions.npz'
        if pred_path.exists():
            seeds.append(int(d.name.replace('seed', '')))
    if not seeds:
        print("No trained models found in output/ensemble/seed*/predictions.npz")
        sys.exit(1)

    y_true = None
    preds = []
    val_losses = []
    for seed in seeds:
        pred_path = ensemble_dir / f'seed{seed}' / 'predictions.npz'
        d = np.load(pred_path)
        if y_true is None:
            y_true = d['y_true']
        preds.append(d['y_pred'])
        # Load val loss for weighting
        results_path = ensemble_dir / f'seed{seed}' / 'results.json'
        if results_path.exists():
            with open(results_path) as f:
                val_losses.append(json.load(f)['best_val_loss'])
        else:
            val_losses.append(None)

    preds = np.array(preds)  # (K, N)
    K = len(seeds)

    print(f"\n{'='*60}")
    print(f"Ensemble of {K} MSTGN-MLP2 models (MSE loss)")
    print(f"{'='*60}")

    # Individual results
    individual_maes = []
    for i, seed in enumerate(seeds):
        mae_i = np.mean(np.abs(preds[i] - y_true))
        individual_maes.append(mae_i)
        vl = f", val_loss={val_losses[i]:.4f}" if val_losses[i] else ""
        print(f"  Seed {seed}: MAE = {mae_i:.2f}h{vl}")

    # Mean ensemble
    y_mean = np.maximum(np.mean(preds, axis=0), 0)
    mae_mean, rmse_mean, mape_mean = calc_metrics(y_mean, y_true)
    print(f"\n  Mean Ensemble ({K} models):")
    print(f"    MAE:  {mae_mean:.2f} hours")
    print(f"    RMSE: {rmse_mean:.2f} hours")
    print(f"    MAPE: {mape_mean:.2f}%")

    # Median ensemble
    y_med = np.maximum(np.median(preds, axis=0), 0)
    mae_med, rmse_med, mape_med = calc_metrics(y_med, y_true)
    print(f"\n  Median Ensemble ({K} models):")
    print(f"    MAE:  {mae_med:.2f} hours")
    print(f"    RMSE: {rmse_med:.2f} hours")
    print(f"    MAPE: {mape_med:.2f}%")

    # Weighted ensemble (inverse val loss)
    if all(v is not None for v in val_losses):
        inv_losses = np.array([1.0 / v for v in val_losses])
        weights = inv_losses / inv_losses.sum()
        y_weighted = np.maximum(np.sum(preds * weights[:, None], axis=0), 0)
        mae_w, rmse_w, mape_w = calc_metrics(y_weighted, y_true)
        print(f"\n  Weighted Ensemble (inv val_loss, {K} models):")
        print(f"    MAE:  {mae_w:.2f} hours")
        print(f"    RMSE: {rmse_w:.2f} hours")
        print(f"    MAPE: {mape_w:.2f}%")
        print(f"    Weights: {', '.join(f'{w:.3f}' for w in weights)}")

    # Top-K ensemble (best models by individual MAE)
    for top_k in [5, 7, 10]:
        if top_k >= K:
            continue
        top_indices = np.argsort(individual_maes)[:top_k]
        y_topk = np.maximum(np.mean(preds[top_indices], axis=0), 0)
        mae_topk, rmse_topk, mape_topk = calc_metrics(y_topk, y_true)
        top_seeds = [seeds[i] for i in top_indices]
        print(f"\n  Top-{top_k} Mean Ensemble (seeds: {top_seeds}):")
        print(f"    MAE:  {mae_topk:.2f} hours")
        print(f"    RMSE: {rmse_topk:.2f} hours")
        print(f"    MAPE: {mape_topk:.2f}%")

    # Save
    np.savez(ensemble_dir / 'ensemble_predictions.npz',
             y_pred_mean=y_mean, y_pred_median=y_med, y_true=y_true)
    results = {
        'num_models': K,
        'mean_ensemble': {'MAE': float(mae_mean), 'RMSE': float(rmse_mean), 'MAPE': float(mape_mean)},
        'median_ensemble': {'MAE': float(mae_med), 'RMSE': float(rmse_med), 'MAPE': float(mape_med)},
        'individual_MAE': {str(s): float(individual_maes[i]) for i, s in enumerate(seeds)},
    }
    with open(ensemble_dir / 'ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {ensemble_dir}")


if __name__ == '__main__':
    main()
