#!/usr/bin/env python
"""Evaluate ensemble of MSTGN models by averaging predictions."""
import numpy as np
import json
import sys
from pathlib import Path


def main():
    ensemble_dir = Path('output/ensemble')
    seeds = [42, 43, 44, 45, 46]

    y_true = None
    preds = []
    for seed in seeds:
        pred_path = ensemble_dir / f'seed{seed}' / 'predictions.npz'
        if not pred_path.exists():
            print(f"Missing: {pred_path}")
            sys.exit(1)
        d = np.load(pred_path)
        if y_true is None:
            y_true = d['y_true']
        preds.append(d['y_pred'])

    preds = np.array(preds)  # (5, N)

    # Mean ensemble
    y_mean = np.maximum(np.mean(preds, axis=0), 0)
    mae_mean = np.mean(np.abs(y_mean - y_true))
    rmse_mean = np.sqrt(np.mean((y_mean - y_true) ** 2))
    mask = y_true > 24
    mape_mean = np.mean(np.abs((y_mean[mask] - y_true[mask]) / y_true[mask])) * 100

    # Median ensemble
    y_med = np.maximum(np.median(preds, axis=0), 0)
    mae_med = np.mean(np.abs(y_med - y_true))
    rmse_med = np.sqrt(np.mean((y_med - y_true) ** 2))
    mape_med = np.mean(np.abs((y_med[mask] - y_true[mask]) / y_true[mask])) * 100

    print(f"\n{'='*60}")
    print(f"Ensemble of {len(seeds)} MSTGN-MLP2 models (MSE loss)")
    print(f"{'='*60}")

    # Individual results
    for i, seed in enumerate(seeds):
        mae_i = np.mean(np.abs(preds[i] - y_true))
        print(f"  Seed {seed}: MAE = {mae_i:.2f}h")

    print(f"\n  Mean Ensemble:")
    print(f"    MAE:  {mae_mean:.2f} hours")
    print(f"    RMSE: {rmse_mean:.2f} hours")
    print(f"    MAPE: {mape_mean:.2f}%")

    print(f"\n  Median Ensemble:")
    print(f"    MAE:  {mae_med:.2f} hours")
    print(f"    RMSE: {rmse_med:.2f} hours")
    print(f"    MAPE: {mape_med:.2f}%")

    # Save
    np.savez(ensemble_dir / 'ensemble_predictions.npz',
             y_pred_mean=y_mean, y_pred_median=y_med, y_true=y_true)
    results = {
        'mean_ensemble': {'MAE': float(mae_mean), 'RMSE': float(rmse_mean), 'MAPE': float(mape_mean)},
        'median_ensemble': {'MAE': float(mae_med), 'RMSE': float(rmse_med), 'MAPE': float(mape_med)},
        'individual_MAE': {str(s): float(np.mean(np.abs(preds[i] - y_true))) for i, s in enumerate(seeds)},
    }
    with open(ensemble_dir / 'ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {ensemble_dir}")


if __name__ == '__main__':
    main()
