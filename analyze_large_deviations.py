"""Analyze large deviations in ETA predictions to characterize and reduce them."""
import numpy as np
import json
import sys
from pathlib import Path
from collections import defaultdict


def inverse_normalize_target(normalized, target_mean, target_std):
    log_target = normalized * target_std + target_mean
    return np.expm1(log_target)


def load_predictions(pred_dir, norm_path='output/norm_params.npz'):
    """Load predictions and ground truth in hours."""
    d = np.load(Path(pred_dir) / 'predictions.npz')
    y_pred = d['y_pred']  # already in hours
    y_true = d['y_true']
    return y_pred, y_true


def load_test_features(cache_dir):
    """Load test set raw features for analysis."""
    x_test = np.load(Path(cache_dir) / 'x_test.npy', mmap_mode='r')
    cell_test = np.load(Path(cache_dir) / 'cell_ids_test.npy', mmap_mode='r')
    return x_test, cell_test


def analyze_deviations(y_pred, y_true, x_test=None):
    """Comprehensive analysis of prediction errors."""
    error = y_pred - y_true  # positive = overestimate, negative = underestimate
    abs_error = np.abs(error)
    n = len(y_true)

    print("=" * 70)
    print("OVERALL TEST SET STATISTICS")
    print("=" * 70)
    print(f"  Total samples: {n:,}")
    print(f"  MAE: {abs_error.mean():.2f}h")
    print(f"  RMSE: {np.sqrt((error**2).mean()):.2f}h")
    print(f"  Median AE: {np.median(abs_error):.2f}h")
    print(f"  y_true range: [{y_true.min():.1f}, {y_true.max():.1f}]h")
    print(f"  y_pred range: [{y_pred.min():.1f}, {y_pred.max():.1f}]h")

    # Error percentiles
    print(f"\n  Error percentiles (absolute):")
    for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
        print(f"    P{p}: {np.percentile(abs_error, p):.2f}h")

    # Define "large deviation" thresholds
    print("\n" + "=" * 70)
    print("LARGE DEVIATION ANALYSIS")
    print("=" * 70)

    thresholds = [24, 48, 72, 100, 150, 200]
    for t in thresholds:
        mask = abs_error > t
        count = mask.sum()
        rate = count / n * 100
        if count > 0:
            mean_err = abs_error[mask].mean()
            mean_true = y_true[mask].mean()
            under = (error[mask] < 0).sum()
            over = (error[mask] >= 0).sum()
            print(f"\n  |error| > {t}h: {count:,} ({rate:.3f}%)")
            print(f"    Mean |error|: {mean_err:.1f}h, Mean true: {mean_true:.1f}h")
            print(f"    Underestimates: {under:,} ({under/count*100:.1f}%), Overestimates: {over:,} ({over/count*100:.1f}%)")
        else:
            print(f"\n  |error| > {t}h: 0 (0.000%)")

    # Relative error analysis (for voyages > 24h to avoid div-by-zero noise)
    print("\n" + "=" * 70)
    print("RELATIVE ERROR ANALYSIS (voyages > 24h)")
    print("=" * 70)
    long_mask = y_true > 24
    if long_mask.sum() > 0:
        rel_error = np.abs(error[long_mask]) / y_true[long_mask]
        print(f"  Samples: {long_mask.sum():,}")
        print(f"  Mean |relative error|: {rel_error.mean()*100:.2f}%")
        for t in [0.25, 0.5, 1.0]:
            cnt = (rel_error > t).sum()
            print(f"  |rel_error| > {t*100:.0f}%: {cnt:,} ({cnt/long_mask.sum()*100:.3f}%)")

    # Per-duration-bin analysis
    print("\n" + "=" * 70)
    print("PER-DURATION-BIN ANALYSIS")
    print("=" * 70)
    bins = [(0, 24), (24, 72), (72, 168), (168, 336), (336, 504), (504, 720)]
    bin_stats = {}
    for lo, hi in bins:
        m = (y_true >= lo) & (y_true < hi)
        if m.sum() > 0:
            ae = abs_error[m]
            err = error[m]
            true_bin = y_true[m]
            large_mask = ae > 100
            stats = {
                'count': int(m.sum()),
                'MAE': float(ae.mean()),
                'RMSE': float(np.sqrt((err**2).mean())),
                'P90': float(np.percentile(ae, 90)),
                'P99': float(np.percentile(ae, 99)),
                'large_dev_rate': float(large_mask.sum() / m.sum() * 100),
                'large_dev_count': int(large_mask.sum()),
                'mean_true': float(true_bin.mean()),
                'underest_rate': float((err < 0).mean() * 100),
            }
            bin_stats[f"{lo}-{hi}h"] = stats
            print(f"\n  [{lo}, {hi})h: n={stats['count']:,}, MAE={stats['MAE']:.2f}h, "
                  f"P90={stats['P90']:.2f}h, P99={stats['P99']:.2f}h")
            print(f"    |err|>100h: {stats['large_dev_count']:,} ({stats['large_dev_rate']:.3f}%), "
                  f"Underest: {stats['underest_rate']:.1f}%")

    # Characterize the worst cases
    print("\n" + "=" * 70)
    print("WORST 100 PREDICTIONS (|error| > 100h)")
    print("=" * 70)
    large_idx = np.where(abs_error > 100)[0]
    if len(large_idx) > 0:
        # Sort by absolute error descending
        sorted_idx = large_idx[np.argsort(-abs_error[large_idx])][:100]
        print(f"  Total large deviations (>100h): {len(large_idx):,}")

        # Analyze features of large-deviation cases
        if x_test is not None:
            # Feature indices: lat=0, lon=1, sog=2, cog=3, dist_to_dest=4,
            #                  bearing_diff=5, temp=6, wind_speed=7, wind_level=8,
            #                  prmsl=9, visibility=10
            feat_names = ['lat', 'lon', 'sog', 'cog', 'dist_to_dest', 'bearing_diff',
                          'temp', 'wind_speed', 'wind_level', 'prmsl', 'visibility']

            print(f"\n  Feature comparison (last timestep, min-max normalized):")
            print(f"  {'Feature':<15} {'Large Dev (mean)':<18} {'All (mean)':<15} {'Diff'}")
            print(f"  {'-'*60}")

            # Get last timestep features for large-dev vs all
            x_large = x_test[large_idx, -1, :]  # (N_large, 11)
            x_all_last = x_test[:, -1, :]  # all test, last timestep

            for i, name in enumerate(feat_names):
                large_mean = float(np.mean(x_large[:, i]))
                all_mean = float(np.mean(x_all_last[:, i]))
                diff = large_mean - all_mean
                print(f"  {name:<15} {large_mean:<18.4f} {all_mean:<15.4f} {diff:+.4f}")

            # Temporal patterns: SOG variance over sequence
            print(f"\n  Temporal SOG patterns:")
            sog_var_large = np.var(x_test[large_idx, :, 2], axis=1)
            sog_var_all = np.var(x_test[:, :, 2], axis=1)
            print(f"    SOG variance (large dev): mean={np.mean(sog_var_large):.6f}, median={np.median(sog_var_large):.6f}")
            print(f"    SOG variance (all):       mean={np.mean(sog_var_all):.6f}, median={np.median(sog_var_all):.6f}")

            # Distance patterns
            dist_large = x_test[large_idx, -1, 4]  # dist_to_dest at last step
            dist_all = x_test[:, -1, 4]
            print(f"\n  Distance to dest (last step, normalized):")
            print(f"    Large dev: mean={np.mean(dist_large):.4f}, std={np.std(dist_large):.4f}")
            print(f"    All:       mean={np.mean(dist_all):.4f}, std={np.std(dist_all):.4f}")

        # Sign analysis of large deviations
        large_err = error[large_idx]
        large_true = y_true[large_idx]
        large_pred = y_pred[large_idx]
        print(f"\n  Sign distribution of large deviations:")
        print(f"    Underestimates (pred < true): {(large_err < 0).sum():,} ({(large_err < 0).mean()*100:.1f}%)")
        print(f"    Overestimates  (pred > true): {(large_err >= 0).sum():,} ({(large_err >= 0).mean()*100:.1f}%)")

        # True value distribution of large deviations
        print(f"\n  True value distribution of large-deviation samples:")
        for lo, hi in [(0, 72), (72, 168), (168, 336), (336, 720)]:
            m = (large_true >= lo) & (large_true < hi)
            print(f"    [{lo}, {hi})h: {m.sum():,} ({m.sum()/len(large_true)*100:.1f}%)")

        # Prediction magnitude analysis
        print(f"\n  Large deviation predictions:")
        print(f"    True: mean={large_true.mean():.1f}h, P25={np.percentile(large_true, 25):.1f}h, "
              f"P50={np.median(large_true):.1f}h, P75={np.percentile(large_true, 75):.1f}h")
        print(f"    Pred: mean={large_pred.mean():.1f}h, P25={np.percentile(large_pred, 25):.1f}h, "
              f"P50={np.median(large_pred):.1f}h, P75={np.percentile(large_pred, 75):.1f}h")

    # Feature-based analysis: which sequences are hardest?
    if x_test is not None:
        print("\n" + "=" * 70)
        print("FEATURE-BASED ERROR ANALYSIS")
        print("=" * 70)

        # SOG at last timestep vs error
        sog_last = x_test[:, -1, 2]
        sog_bins = [(0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0)]
        print(f"\n  SOG (last step) bins:")
        for lo, hi in sog_bins:
            m = (sog_last >= lo) & (sog_last < hi)
            if m.sum() > 0:
                ae_bin = abs_error[m]
                large_in_bin = (ae_bin > 100).sum()
                print(f"    SOG [{lo:.2f}, {hi:.2f}): n={m.sum():,}, MAE={ae_bin.mean():.2f}h, "
                      f"|err|>100h: {large_in_bin:,} ({large_in_bin/m.sum()*100:.3f}%)")

        # Distance to destination bins
        dist_last = x_test[:, -1, 4]
        dist_bins = [(0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 1.0)]
        print(f"\n  Distance to dest (last step) bins:")
        for lo, hi in dist_bins:
            m = (dist_last >= lo) & (dist_last < hi)
            if m.sum() > 0:
                ae_bin = abs_error[m]
                large_in_bin = (ae_bin > 100).sum()
                print(f"    dist [{lo:.2f}, {hi:.2f}): n={m.sum():,}, MAE={ae_bin.mean():.2f}h, "
                      f"|err|>100h: {large_in_bin:,} ({large_in_bin/m.sum()*100:.3f}%)")

        # SOG sequence variance (stability indicator)
        sog_var = np.var(x_test[:, :, 2], axis=1)
        var_p = np.percentile(sog_var, [25, 50, 75, 90, 95])
        var_bins = [(0, var_p[0]), (var_p[0], var_p[1]), (var_p[1], var_p[2]),
                    (var_p[2], var_p[3]), (var_p[3], float('inf'))]
        print(f"\n  SOG variance bins (proxy for speed instability):")
        for i, (lo, hi) in enumerate(var_bins):
            m = (sog_var >= lo) & (sog_var < hi)
            if m.sum() > 0:
                ae_bin = abs_error[m]
                large_in_bin = (ae_bin > 100).sum()
                label = f"Q{i*25}-Q{min((i+1)*25, 100)}" if i < 4 else "Q90+"
                print(f"    {label} [{lo:.6f}, {hi:.6f}): n={m.sum():,}, MAE={ae_bin.mean():.2f}h, "
                      f"|err|>100h: {large_in_bin:,} ({large_in_bin/m.sum()*100:.3f}%)")

        # Speed change (deceleration/acceleration) at end of sequence
        sog_diff = x_test[:, -1, 2] - x_test[:, -6, 2]  # last - 6th from last
        diff_bins = [(-1, -0.1), (-0.1, -0.01), (-0.01, 0.01), (0.01, 0.1), (0.1, 1)]
        print(f"\n  SOG change (last vs 6-steps-ago):")
        for lo, hi in diff_bins:
            m = (sog_diff >= lo) & (sog_diff < hi)
            if m.sum() > 0:
                ae_bin = abs_error[m]
                large_in_bin = (ae_bin > 100).sum()
                print(f"    Δsog [{lo:+.2f}, {hi:+.2f}): n={m.sum():,}, MAE={ae_bin.mean():.2f}h, "
                      f"|err|>100h: {large_in_bin:,} ({large_in_bin/m.sum()*100:.3f}%)")

    # Return summary stats for further use
    return {
        'n_total': n,
        'mae': float(abs_error.mean()),
        'large_dev_100h_count': int((abs_error > 100).sum()),
        'large_dev_100h_rate': float((abs_error > 100).mean() * 100),
        'bin_stats': bin_stats,
    }


def main():
    pred_dir = sys.argv[1] if len(sys.argv) > 1 else 'output/mstgn_mlp2_kd06'
    cache_dir = 'output/cache_sequences/seq48_label24_pred1_mv150000_ms50000000'

    print(f"Loading predictions from {pred_dir}")
    y_pred, y_true = load_predictions(pred_dir)

    print(f"Loading test features from {cache_dir}")
    x_test, cell_test = load_test_features(cache_dir)
    print(f"  x_test shape: {x_test.shape}")

    stats = analyze_deviations(y_pred, y_true, x_test)

    # Save summary
    out_path = Path(pred_dir) / 'deviation_analysis.json'
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == '__main__':
    main()
