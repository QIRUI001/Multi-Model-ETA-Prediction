#!/usr/bin/env python
"""
从已有的 spill 桶文件重建 memmap 缓存。

用途：当 Pass C 中断后，无需重新跑 Step 1-3 的数据准备，
直接从 spill_dir 中的 bucket pkl 文件重建序列化的 memmap 文件。

支持 data_new:old 比例控制（默认 2:1）。

用法：
    python rebuild_memmap.py --new_ratio 2.0
    python rebuild_memmap.py --new_ratio 2.0 --max_seqs_per_bucket 500000
"""

import os
import sys
import gc
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import glob

sys.path.insert(0, os.path.dirname(__file__))
from train_eta import (
    VoyageETADataset, _compute_geom_features, _update_minmax,
    _update_welford, _finalize_welford, build_decoder_input,
    create_memmap_arrays
)


def main():
    parser = argparse.ArgumentParser(description="从spill桶重建memmap缓存")
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--seq_len', type=int, default=48)
    parser.add_argument('--label_len', type=int, default=24)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--max_voyages', type=int, default=150000)
    parser.add_argument('--max_sequences', type=int, default=75000000)
    parser.add_argument('--max_seqs_per_bucket', type=int, default=500000)
    parser.add_argument('--new_ratio', type=float, default=2.0,
                        help='data_new : old 的目标比例 (default: 2.0，即 new 是 old 的 2 倍)')
    parser.add_argument('--new_prefix', type=str, default='new_',
                        help='data_new 的 voyage_id 前缀')
    args = parser.parse_args()

    cache_tag = f"seq{args.seq_len}_label{args.label_len}_pred{args.pred_len}_mv{args.max_voyages}_ms{args.max_sequences}"
    cache_dir = Path(args.output_dir) / "cache_sequences" / cache_tag
    spill_dir = cache_dir / "step3_spill"

    bucket_files = sorted(spill_dir.glob("bucket_*_part_*.pkl"))
    if len(bucket_files) == 0:
        print(f"错误：{spill_dir} 中没有找到分桶文件")
        return
    print(f"找到 {len(bucket_files)} 个分桶文件")

    # ========== 按 bucket_id 分组（合并同一 bucket 的多个 part，修复拆分航程问题） ==========
    from collections import defaultdict
    bucket_groups = defaultdict(list)
    for bf in bucket_files:
        # 文件名格式: bucket_5_part_0.pkl
        bucket_id = int(bf.stem.split('_')[1])
        bucket_groups[bucket_id].append(bf)
    bucket_ids_sorted = sorted(bucket_groups.keys())
    print(f"合并为 {len(bucket_ids_sorted)} 个逻辑桶")

    def load_merged_bucket(bucket_id):
        """加载并合并同一 bucket 的所有 part 文件"""
        parts = [pd.read_pickle(f) for f in bucket_groups[bucket_id]]
        return pd.concat(parts, ignore_index=True)

    # 删除旧的 npy 文件
    for f in cache_dir.glob("*.npy"):
        f.unlink()
    print("已清理旧的 npy 文件")

    # ========== 收集所有 voyage_id ==========
    print("收集 voyage_id ...")
    all_voyage_ids = set()
    for bid in bucket_ids_sorted:
        df = load_merged_bucket(bid)
        all_voyage_ids.update(df['voyage_id'].unique())
        del df

    # 按 train_eta.py 相同的方式划分 split
    voyage_ids = np.array(sorted(all_voyage_ids))
    rng = np.random.default_rng(42)
    rng.shuffle(voyage_ids)
    n_total = len(voyage_ids)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    train_ids = set(voyage_ids[:n_train])
    val_ids = set(voyage_ids[n_train:n_train + n_val])
    test_ids = set(voyage_ids[n_train + n_val:])
    print(f"航程总数: {n_total}, train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # ========== Pass A: 归一化参数（流式） ==========
    print("\n=== Pass A: 统计归一化参数 ===")
    dataset = VoyageETADataset(seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len)
    feature_cols = ['lat', 'lon', 'sog', 'cog', 'dist_to_dest_km', 'bearing_diff', 'naive_eta_hours', 'cum_dist_km']

    feat_min = None
    feat_max = None
    count, mean, m2 = 0, 0.0, 0.0

    for bid in tqdm(bucket_ids_sorted, desc="Pass A"):
        df_bucket = load_merged_bucket(bid)
        df_bucket = df_bucket[df_bucket['voyage_id'].isin(train_ids)]
        if len(df_bucket) == 0:
            del df_bucket
            gc.collect()
            continue
        for _, group in df_bucket.groupby('voyage_id'):
            group = _compute_geom_features(group)
            feat_vals = group[feature_cols].values.astype(np.float32)
            feat_min, feat_max = _update_minmax(feat_min, feat_max, feat_vals)
            target_vals = np.log1p(group['remaining_hours'].values.astype(np.float32))
            count, mean, m2 = _update_welford(count, mean, m2, target_vals)
        del df_bucket
        gc.collect()

    dataset.feature_min = feat_min
    dataset.feature_max = feat_max
    dataset.target_mean, dataset.target_std = _finalize_welford(count, mean, m2)
    dataset.save_params(os.path.join(args.output_dir, 'norm_params.npz'))
    print(f"归一化参数已保存")

    # ========== Pass B: 统计各 split 中 new/old 的序列数 ==========
    print("\n=== Pass B: 统计序列数量（new vs old） ===")
    splits_info = {
        'train': {'ids': train_ids, 'new_seqs': 0, 'old_seqs': 0},
        'val':   {'ids': val_ids,   'new_seqs': 0, 'old_seqs': 0},
        'test':  {'ids': test_ids,  'new_seqs': 0, 'old_seqs': 0},
    }

    for bid in tqdm(bucket_ids_sorted, desc="Pass B"):
        df_bucket = load_merged_bucket(bid)
        for vid, group in df_bucket.groupby('voyage_id'):
            n_seq = max(0, len(group) - args.seq_len - args.pred_len + 1)
            if n_seq == 0:
                continue
            is_new = str(vid).startswith(args.new_prefix)
            for name, info in splits_info.items():
                if vid in info['ids']:
                    if is_new:
                        info['new_seqs'] += n_seq
                    else:
                        info['old_seqs'] += n_seq
                    break
        del df_bucket
        gc.collect()

    # 按比例计算目标序列数
    print("\n序列统计 (data_new:old 比例控制):")
    target_counts = {}
    for name, info in splits_info.items():
        new_s = info['new_seqs']
        old_s = info['old_seqs']
        # 全部保留 new，old 按比例采样
        target_old = int(new_s / args.new_ratio) if args.new_ratio > 0 else old_s
        target_old = min(target_old, old_s)  # 不能超过实际可用
        total = new_s + target_old
        target_counts[name] = {
            'new_target': new_s,
            'old_target': target_old,
            'total': total,
            'old_ratio': target_old / max(old_s, 1),  # old 的采样比例
        }
        print(f"  {name}: new={new_s:,} (全部), old={target_old:,}/{old_s:,} "
              f"(采样{target_counts[name]['old_ratio']:.1%}), total={total:,}")

    # ========== 创建 memmap 数组 ==========
    print("\n=== 创建 memmap 数组 ===")
    memmaps = {}
    for name in ['train', 'val', 'test']:
        n = target_counts[name]['total']
        arrays = create_memmap_arrays(cache_dir, n, args.seq_len, args.label_len, args.pred_len, prefix=name)
        memmaps[name] = arrays
        size_gb = n * (args.seq_len * 6 + args.seq_len * 5 + 25 * 6 + 25 * 5 + 1 + 1) * 4 / 1e9
        print(f"  {name}: {n:,} 序列, 预估 {size_gb:.1f} GB")

    # ========== Pass C: 生成序列（按比例写入 memmap） ==========
    print("\n=== Pass C: 生成序列 (memmap 增量写入) ===")
    idxs = {'train': 0, 'val': 0, 'test': 0}
    meta_list = []

    for bi, bid in enumerate(bucket_ids_sorted):
        df_bucket = load_merged_bucket(bid)
        if len(df_bucket) == 0:
            del df_bucket
            gc.collect()
            continue
        df_bucket['mmsi'] = df_bucket['mmsi'].astype('int32')

        for name in ['train', 'val', 'test']:
            ids = splits_info[name]['ids']
            tc = target_counts[name]
            mm = memmaps[name]  # X, X_mark_enc, X_dec, X_mark_dec, y, sd
            idx = idxs[name]
            max_total = tc['total']

            if idx >= max_total:
                continue

            split_df = df_bucket[df_bucket['voyage_id'].isin(ids)]
            if len(split_df) == 0:
                continue

            # 分别处理 new 和 old
            new_df = split_df[split_df['voyage_id'].apply(lambda v: str(v).startswith(args.new_prefix))]
            old_df = split_df[~split_df['voyage_id'].apply(lambda v: str(v).startswith(args.new_prefix))]

            for source_label, source_df, target_seqs in [
                ('new', new_df, tc['new_target']),
                ('old', old_df, tc['old_target']),
            ]:
                if len(source_df) == 0 or idx >= max_total:
                    continue

                # 计算这个 source 在这个 bucket 的最大序列数
                remaining = max_total - idx
                bucket_max = min(args.max_seqs_per_bucket, remaining)

                # 对 old 按比例限制
                if source_label == 'old':
                    bucket_max = min(bucket_max, int(args.max_seqs_per_bucket * tc['old_ratio']))
                    bucket_max = max(bucket_max, 1)

                if name == 'test':
                    Xs, Xmes, Xmds, ys, sds, meta = dataset.create_sequences(
                        source_df, max_sequences=bucket_max, fit=False, return_meta=True
                    )
                else:
                    Xs, Xmes, Xmds, ys, sds = dataset.create_sequences(
                        source_df, max_sequences=bucket_max, fit=False
                    )

                n_new = len(Xs)
                if idx + n_new > max_total:
                    n_new = max_total - idx
                    Xs = Xs[:n_new]
                    Xmes = Xmes[:n_new]
                    Xmds = Xmds[:n_new]
                    ys = ys[:n_new]
                    sds = sds[:n_new]
                    if name == 'test':
                        meta = {k: v[:n_new] for k, v in meta.items()}

                if n_new > 0:
                    mm[0][idx:idx+n_new] = Xs
                    mm[1][idx:idx+n_new] = Xmes
                    mm[2][idx:idx+n_new] = build_decoder_input(Xs, args.label_len, args.pred_len)
                    mm[3][idx:idx+n_new] = Xmds
                    mm[4][idx:idx+n_new] = ys
                    mm[5][idx:idx+n_new] = sds
                    if name == 'test':
                        meta_list.append(meta)
                    idx += n_new

                del Xs, Xmes, Xmds, ys, sds

            idxs[name] = idx

        del df_bucket
        gc.collect()

        if (bi + 1) % 4 == 0:
            print(f"  进度: {bi+1}/{len(bucket_ids_sorted)} buckets, "
                  f"train={idxs['train']:,}/{target_counts['train']['total']:,}, "
                  f"val={idxs['val']:,}/{target_counts['val']['total']:,}, "
                  f"test={idxs['test']:,}/{target_counts['test']['total']:,}")

    # Flush memmap
    for name in ['train', 'val', 'test']:
        for arr in memmaps[name]:
            del arr
    gc.collect()

    # 保存实际写入的序列数（解决 memmap 尾部零填充问题）
    actual_counts = {name: idxs[name] for name in ['train', 'val', 'test']}
    np.save(cache_dir / "actual_counts.npy", actual_counts, allow_pickle=True)
    print(f"\n实际写入序列数已保存: {actual_counts}")

    # 合并 meta
    if meta_list:
        test_meta = {
            'mmsi': np.concatenate([m['mmsi'] for m in meta_list]),
            'voyage_id': np.concatenate([m['voyage_id'] for m in meta_list]),
            'pred_time': np.concatenate([m['pred_time'] for m in meta_list]),
            'end_time': np.concatenate([m['end_time'] for m in meta_list]),
        }
    else:
        test_meta = {'mmsi': np.array([]), 'voyage_id': np.array([]),
                     'pred_time': np.array([]), 'end_time': np.array([])}
    np.save(cache_dir / "test_meta.npy", test_meta, allow_pickle=True)

    print(f"\n=== 完成 ===")
    print(f"最终序列: train={idxs['train']:,}, val={idxs['val']:,}, test={idxs['test']:,}")
    print(f"缓存目录: {cache_dir}")
    print(f"\n下一步: python train_eta.py --use_cache --max_voyages {args.max_voyages} --max_sequences {args.max_sequences}")


if __name__ == '__main__':
    main()
