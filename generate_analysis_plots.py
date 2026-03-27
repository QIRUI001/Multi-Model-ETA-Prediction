#!/usr/bin/env python
"""Generate analysis plots from saved MSTGN-KD predictions.

Usage (on server):
    python generate_analysis_plots.py --predictions output/mstgn_kd06_avg3/predictions.npz
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def generate_plots(y_pred: np.ndarray, y_true: np.ndarray, save_dir: Path, model_name: str = "MSTGN-KD"):
    """Generate four complementary analysis panels.

    Panels:
      (a) Predicted vs Actual  – hexbin density scatter
      (b) MAE by Remaining Time – binned by true remaining hours
      (c) Error Distribution   – histogram with mean/median lines
      (d) Cumulative Accuracy  – % of predictions within X hours
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))

    # ── (a) Predicted vs Actual (density) ──
    ax = axes[0, 0]
    max_val = max(y_true.max(), y_pred.max()) * 1.02
    hb = ax.hexbin(y_true, y_pred, gridsize=60, cmap='Blues', mincnt=1,
                   extent=[0, max_val, 0, max_val])
    ax.plot([0, max_val], [0, max_val], color='#C44E52', linestyle='--',
            linewidth=1.5, label='Perfect prediction', zorder=5)
    ax.set_xlabel('Actual Remaining Time (hours)', fontsize=10)
    ax.set_ylabel('Predicted Remaining Time (hours)', fontsize=10)
    ax.set_title('(a) Predicted vs Actual', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect('equal', adjustable='box')
    fig.colorbar(hb, ax=ax, label='Count', shrink=0.8)

    # ── (b) MAE by Remaining Time (true) ──
    ax = axes[0, 1]
    # Bin by true remaining hours in 24h (1-day) bins, cap at 25 days
    max_hours_for_bin = min(y_true.max(), 600)  # ~25 days cap
    bin_edges = np.arange(0, max_hours_for_bin + 24, 24)
    bin_maes, bin_counts, bin_centers = [], [], []
    for i in range(len(bin_edges) - 1):
        mask = (y_true >= bin_edges[i]) & (y_true < bin_edges[i + 1])
        if mask.sum() > 100:
            bin_maes.append(np.mean(abs_errors[mask]))
            bin_counts.append(mask.sum())
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)

    # Convert to days for x-axis
    bin_centers_days = np.array(bin_centers) / 24
    bars = ax.bar(bin_centers_days, bin_maes, width=0.85, color='#4C72B0',
                  edgecolor='white', linewidth=0.6, alpha=0.85)
    ax.set_xlabel('True Remaining Time (days)', fontsize=10)
    ax.set_ylabel('MAE (hours)', fontsize=10)
    ax.set_title('(b) MAE by True Remaining Time', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    # ── (c) Error Distribution ──
    ax = axes[1, 0]
    clip_val = np.percentile(np.abs(errors), 99.5)
    display_errors = np.clip(errors, -clip_val, clip_val)
    ax.hist(display_errors, bins=100, color='#4C72B0', edgecolor='white',
            linewidth=0.2, alpha=0.85)
    ax.axvline(x=0, color='#C44E52', linestyle='--', linewidth=1.2, label='Zero')
    ax.axvline(x=np.mean(errors), color='#DD8452', linestyle='-', linewidth=1.5,
               label=f'Mean: {np.mean(errors):+.1f}h')
    ax.axvline(x=np.median(errors), color='#55A868', linestyle='-', linewidth=1.5,
               label=f'Median: {np.median(errors):+.1f}h')
    ax.set_xlabel('Prediction Error (hours)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('(c) Error Distribution', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8.5, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    # ── (d) Cumulative Accuracy ──
    ax = axes[1, 1]
    thresholds = np.arange(0, 101, 0.5)
    cum_pct = np.array([np.mean(abs_errors <= t) * 100 for t in thresholds])
    ax.plot(thresholds, cum_pct, color='#4C72B0', linewidth=2)
    ax.fill_between(thresholds, cum_pct, alpha=0.12, color='#4C72B0')
    # Mark key percentages
    for target_pct in [50, 80, 90, 95]:
        idx = np.searchsorted(cum_pct, target_pct)
        if idx < len(thresholds):
            th = thresholds[idx]
            ax.plot(th, target_pct, 'o', color='#C44E52', markersize=5, zorder=5)
            # Offset labels to avoid overlap
            offsets = {50: (5, -5), 80: (5, -5), 90: (5, -5), 95: (5, -5)}
            dx, dy = offsets[target_pct]
            ax.annotate(f'{target_pct}% within {th:.0f}h',
                        xy=(th, target_pct),
                        xytext=(th + dx, target_pct + dy),
                        fontsize=8, color='#333333',
                        arrowprops=dict(arrowstyle='->', color='#999999', lw=0.8))
    ax.set_xlabel('Absolute Error Threshold (hours)', fontsize=10)
    ax.set_ylabel('Predictions (%)', fontsize=10)
    ax.set_title(f'(d) Cumulative Accuracy (MAE = {mae:.2f}h)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.suptitle(f'{model_name} — Test Set Analysis (n = {len(y_pred):,})',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout(pad=1.5)

    # Save
    save_dir.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf']:
        path = save_dir / f'analysis_plots.{ext}'
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"Saved {path}")
    plt.close(fig)

    # Print summary stats
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"Test samples: {len(y_pred):,}")
    print(f"MAE:  {mae:.2f}h  ({mae/24:.2f} days)")
    print(f"RMSE: {rmse:.2f}h  ({rmse/24:.2f} days)")
    print(f"Mean error:   {np.mean(errors):+.2f}h")
    print(f"Median error: {np.median(errors):+.2f}h")
    print(f"|error| > 100h: {np.mean(abs_errors > 100)*100:.2f}%")
    print(f"50th pct: {np.percentile(abs_errors, 50):.1f}h")
    print(f"90th pct: {np.percentile(abs_errors, 90):.1f}h")
    print(f"95th pct: {np.percentile(abs_errors, 95):.1f}h")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description='Generate analysis plots')
    parser.add_argument('--predictions', type=str,
                        default='output/mstgn_kd06_avg3/predictions.npz',
                        help='Path to predictions.npz')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save plots')
    parser.add_argument('--model_name', type=str, default='MSTGN-KD',
                        help='Model name for title')
    args = parser.parse_args()

    data = np.load(args.predictions)
    y_pred = data['y_pred']
    y_true = data['y_true']

    generate_plots(y_pred, y_true, Path(args.output_dir), args.model_name)


if __name__ == '__main__':
    main()
