#!/usr/bin/env python
"""
并行化数据预处理脚本（内存优化版）

功能：
1. 分块读取CSV，避免内存溢出
2. 按MMSI分组提取所有船舶的航程
3. 使用有限并行数控制内存
4. 流式合并结果

用法：
    python preprocess_data_parallel.py --data_dir ./data --output_dir ./output/processed
    python preprocess_data_parallel.py --workers 4  # 减少并行数以降低内存
    python preprocess_data_parallel.py --chunk_size 500000  # 调整分块大小
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Generator
from dataclasses import dataclass
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import gc
import shutil

warnings.filterwarnings('ignore')


@dataclass
class VoyageSegment:
    """航段信息"""
    mmsi: int
    segment_id: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_hours: float
    num_points: int
    avg_sog: float


@dataclass
class PortStop:
    """港口停靠信息"""
    mmsi: int
    arrival_time: pd.Timestamp
    departure_time: pd.Timestamp
    duration_hours: float
    lon: float
    lat: float
    region: str


# ============================================================
# 核心处理函数
# ============================================================

REQUIRED_COLS = ['mmsi', 'postime', 'eta', 'lon', 'lat', 'sog', 'cog']

# 停靠速度阈值（节）
STOP_SPEED_THRESHOLD = 0.3
# 最小航段时长（小时）
MIN_SEGMENT_HOURS = 6
# 最小停靠时长（小时）
MIN_STOP_HOURS = 2
# 最小数据密度（点/天）
MIN_POINTS_PER_DAY = 10
# 最小数据点数
MIN_POINTS = 50


def classify_region(lon: float, lat: float) -> str:
    """根据经纬度判断区域"""
    if 100 <= lon <= 135 and 20 <= lat <= 45:
        return '中国东部'
    if 103 <= lon <= 105 and 0 <= lat <= 3:
        return '新加坡'
    if 55 <= lon <= 80 and 10 <= lat <= 30:
        return '中东/印度'
    if 30 <= lon <= 55 and 25 <= lat <= 35:
        return '红海/苏伊士'
    if -130 <= lon <= -115 and 30 <= lat <= 50:
        return '美国西海岸'
    return '其他'


def process_single_ship(ship_df: pd.DataFrame, mmsi: int) -> Tuple[List[pd.DataFrame], List[dict]]:
    """处理单艘船的数据，提取所有有效航段"""
    
    if len(ship_df) < MIN_POINTS:
        return [], []
    
    ship_df = ship_df.sort_values('postime').copy()
    ship_df['postime'] = pd.to_datetime(ship_df['postime'])
    
    # 检查整体数据密度
    time_span = (ship_df['postime'].max() - ship_df['postime'].min()).total_seconds() / 86400
    if time_span > 0 and len(ship_df) / time_span < MIN_POINTS_PER_DAY:
        return [], []
    
    # 标记停靠状态
    ship_df['is_stopped'] = ship_df['sog'] < STOP_SPEED_THRESHOLD
    
    # 分割连续的航行/停靠段
    ship_df['segment_change'] = ship_df['is_stopped'].ne(ship_df['is_stopped'].shift()).cumsum()
    
    voyage_dfs = []
    stops = []
    segment_idx = 0
    
    for seg_id, group in ship_df.groupby('segment_change'):
        is_sailing = not group['is_stopped'].iloc[0]
        start_time = group['postime'].iloc[0]
        end_time = group['postime'].iloc[-1]
        duration_hours = (end_time - start_time).total_seconds() / 3600
        
        if is_sailing and duration_hours >= MIN_SEGMENT_HOURS and len(group) >= MIN_POINTS:
            # 检查数据密度
            duration_days = duration_hours / 24
            if duration_days > 0 and len(group) / duration_days >= MIN_POINTS_PER_DAY:
                # 有效航段
                seg_df = group.drop(columns=['is_stopped', 'segment_change']).copy()
                seg_df['mmsi'] = mmsi
                seg_df['voyage_id'] = f"{mmsi}_{segment_idx}"
                seg_df['voyage_duration_hours'] = duration_hours
                seg_df['remaining_hours'] = (end_time - seg_df['postime']).dt.total_seconds() / 3600
                
                voyage_dfs.append(seg_df)
                segment_idx += 1
        
        elif not is_sailing and duration_hours >= MIN_STOP_HOURS:
            # 港口停靠
            mean_lon = group['lon'].mean()
            mean_lat = group['lat'].mean()
            stops.append({
                'mmsi': mmsi,
                'arrival_time': start_time,
                'departure_time': end_time,
                'duration_hours': duration_hours,
                'lon': mean_lon,
                'lat': mean_lat,
                'region': classify_region(mean_lon, mean_lat)
            })
    
    return voyage_dfs, stops


def process_single_file(args) -> Tuple[str, int, int, int]:
    """处理单个CSV文件（分块读取，低内存）"""
    csv_file, output_dir, chunk_size = args
    
    file_stem = Path(csv_file).stem
    temp_dir = Path(output_dir) / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计变量
    n_ships = 0
    n_voyages = 0
    n_points = 0
    processed_mmsis = set()  # 记录已处理的MMSI
    
    try:
        # 分块读取并按MMSI聚合
        mmsi_data = {}  # mmsi -> list of chunks
        
        for chunk in pd.read_csv(csv_file, usecols=REQUIRED_COLS, 
                                  low_memory=False, chunksize=chunk_size):
            for mmsi, group in chunk.groupby('mmsi'):
                if mmsi in processed_mmsis:
                    continue  # 已处理过
                if mmsi not in mmsi_data:
                    mmsi_data[mmsi] = []
                mmsi_data[mmsi].append(group)
            
            # 检查内存，如果某个MMSI数据量够大就先处理
            for mmsi in list(mmsi_data.keys()):
                total_rows = sum(len(df) for df in mmsi_data[mmsi])
                if total_rows >= MIN_POINTS * 2:  # 有足够数据
                    # 合并并处理这艘船
                    ship_df = pd.concat(mmsi_data[mmsi], ignore_index=True)
                    del mmsi_data[mmsi]
                    processed_mmsis.add(mmsi)
                    
                    voyage_dfs, stops = process_single_ship(ship_df, mmsi)
                    
                    # 统计并写入临时文件
                    if voyage_dfs:
                        n_ships += 1
                        n_voyages += len(voyage_dfs)
                        for i, vdf in enumerate(voyage_dfs):
                            n_points += len(vdf)
                            vpath = temp_dir / f'{file_stem}_{mmsi}_{i}_voyage.parquet'
                            vdf.to_parquet(vpath, index=False)
                    
                    if stops:
                        spath = temp_dir / f'{file_stem}_{mmsi}_stops.parquet'
                        pd.DataFrame(stops).to_parquet(spath, index=False)
                    
                    del ship_df, voyage_dfs, stops
                    gc.collect()
        
        # 处理剩余的船舶数据
        for mmsi, chunks in mmsi_data.items():
            if not chunks:
                continue
            ship_df = pd.concat(chunks, ignore_index=True)
            voyage_dfs, stops = process_single_ship(ship_df, mmsi)
            
            if voyage_dfs:
                n_ships += 1
                n_voyages += len(voyage_dfs)
                for i, vdf in enumerate(voyage_dfs):
                    n_points += len(vdf)
                    vpath = temp_dir / f'{file_stem}_{mmsi}_{i}_voyage.parquet'
                    vdf.to_parquet(vpath, index=False)
            
            if stops:
                spath = temp_dir / f'{file_stem}_{mmsi}_stops.parquet'
                pd.DataFrame(stops).to_parquet(spath, index=False)
            
            del ship_df, voyage_dfs, stops
        
        del mmsi_data
        gc.collect()
        
        return (file_stem, n_ships, n_voyages, n_points)
    
    except Exception as e:
        print(f"Error processing {file_stem}: {e}")
        return (file_stem, 0, 0, 0)


def merge_results(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """流式合并所有临时文件（内存优化）"""
    temp_dir = output_dir / 'temp'
    
    # 合并航程数据 - 流式处理
    voyage_files = list(temp_dir.glob('*_voyage.parquet'))
    print(f"  合并 {len(voyage_files)} 个航程文件...")
    
    if voyage_files:
        # 分批合并，每次最多100个文件
        batch_size = 100
        merged_files = []
        
        for i in range(0, len(voyage_files), batch_size):
            batch = voyage_files[i:i+batch_size]
            batch_dfs = [pd.read_parquet(f) for f in batch]
            batch_combined = pd.concat(batch_dfs, ignore_index=True)
            
            # 保存中间结果
            merged_path = temp_dir / f'merged_voyage_{i//batch_size}.parquet'
            batch_combined.to_parquet(merged_path, index=False)
            merged_files.append(merged_path)
            
            del batch_dfs, batch_combined
            gc.collect()
        
        # 最终合并
        if len(merged_files) > 1:
            final_dfs = [pd.read_parquet(f) for f in merged_files]
            voyage_combined = pd.concat(final_dfs, ignore_index=True)
            del final_dfs
        else:
            voyage_combined = pd.read_parquet(merged_files[0])
        
        gc.collect()
    else:
        voyage_combined = pd.DataFrame()
    
    # 合并停靠数据
    stop_files = list(temp_dir.glob('*_stops.parquet'))
    print(f"  合并 {len(stop_files)} 个停靠文件...")
    
    if stop_files:
        stop_dfs = []
        for f in stop_files:
            stop_dfs.append(pd.read_parquet(f))
            if len(stop_dfs) >= 100:
                # 批量合并
                combined = pd.concat(stop_dfs, ignore_index=True)
                stop_dfs = [combined]
                gc.collect()
        
        stop_combined = pd.concat(stop_dfs, ignore_index=True) if stop_dfs else pd.DataFrame()
        del stop_dfs
        gc.collect()
    else:
        stop_combined = pd.DataFrame()
    
    return voyage_combined, stop_combined


def filter_data_quality(voyage_df: pd.DataFrame, stop_df: pd.DataFrame,
                        max_stop_hours: float = 168.0,
                        min_stop_hours: float = 4.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """数据质量过滤"""
    print("\n数据质量过滤...")
    
    original_voyages = len(voyage_df)
    original_voyage_count = voyage_df['voyage_id'].nunique() if len(voyage_df) > 0 else 0
    
    # 航程数据已经在process_single_ship中过滤过了，这里主要过滤停靠数据
    if len(stop_df) > 0:
        original_stops = len(stop_df)
        # 过滤异常停靠时间
        stop_df = stop_df[(stop_df['duration_hours'] >= min_stop_hours) & 
                          (stop_df['duration_hours'] <= max_stop_hours)].copy()
        print(f"  停靠数据: {original_stops:,} -> {len(stop_df):,}")
    
    final_voyage_count = voyage_df['voyage_id'].nunique() if len(voyage_df) > 0 else 0
    print(f"  航程数据: {original_voyages:,} 条, {final_voyage_count:,} 个航程")
    
    return voyage_df, stop_df


def print_summary(voyage_df: pd.DataFrame, stop_df: pd.DataFrame):
    """打印数据摘要"""
    print("\n" + "="*60)
    print("数据摘要")
    print("="*60)
    
    if len(voyage_df) > 0:
        n_voyages = voyage_df['voyage_id'].nunique()
        n_ships = voyage_df['mmsi'].nunique()
        
        print(f"\n【航程数据】")
        print(f"  总数据点: {len(voyage_df):,}")
        print(f"  航程数量: {n_voyages:,}")
        print(f"  船舶数量: {n_ships:,}")
        print(f"  剩余时间范围: {voyage_df['remaining_hours'].min():.1f} ~ {voyage_df['remaining_hours'].max():.1f} 小时")
        print(f"  速度范围: {voyage_df['sog'].min():.1f} ~ {voyage_df['sog'].max():.1f} 节")
        
        # 航程时长分布
        voyage_stats = voyage_df.groupby('voyage_id').agg({
            'voyage_duration_hours': 'first',
            'mmsi': 'count'
        }).rename(columns={'mmsi': 'points'})
        
        print(f"\n  航程时长分布:")
        duration_bins = [0, 6, 12, 24, 48, 96, 168, 336, float('inf')]
        labels = ['6-12h', '12-24h', '1-2d', '2-4d', '4-7d', '7-14d', '>14d']
        voyage_stats['duration_bin'] = pd.cut(voyage_stats['voyage_duration_hours'], 
                                               bins=duration_bins, labels=['<6h'] + labels)
        dist = voyage_stats['duration_bin'].value_counts().sort_index()
        for label, count in dist.items():
            if count > 0:
                print(f"    {label}: {count:,} 个航程")
    
    if len(stop_df) > 0:
        print(f"\n【港口停靠数据】")
        print(f"  总停靠次数: {len(stop_df):,}")
        print(f"  停靠时长: {stop_df['duration_hours'].min():.1f} ~ {stop_df['duration_hours'].max():.1f} 小时")
        
        print(f"\n  按区域统计:")
        region_stats = stop_df.groupby('region')['duration_hours'].agg(['mean', 'count'])
        for region, row in region_stats.iterrows():
            print(f"    {region}: 平均 {row['mean']:.1f}h, {int(row['count']):,} 次")


def main():
    parser = argparse.ArgumentParser(description='并行化AIS数据预处理（内存优化版）')
    parser.add_argument('--data_dir', type=str, default='./data', help='原始数据目录')
    parser.add_argument('--output_dir', type=str, default='./output/processed', help='输出目录')
    parser.add_argument('--workers', type=int, default=4, help='并行工作进程数（默认4，降低内存占用）')
    parser.add_argument('--max_files', type=int, default=None, help='最多处理文件数（用于测试）')
    parser.add_argument('--chunk_size', type=int, default=500000, help='CSV分块读取大小（默认50万行）')
    
    args = parser.parse_args()
    
    # 设置并行数 - 限制以控制内存
    n_workers = min(args.workers, cpu_count())
    
    print("="*60)
    print("并行化AIS数据预处理（内存优化版）")
    print("="*60)
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"并行进程数: {n_workers} (建议64GB内存使用4进程)")
    print(f"分块大小: {args.chunk_size:,} 行/块")
    print(f"CPU核心数: {cpu_count()}")
    
    # 获取文件列表
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_files = sorted([f for f in data_dir.glob('*-ais.csv') if not f.name.startswith('._')])
    if args.max_files:
        csv_files = csv_files[:args.max_files]
    
    print(f"\n找到 {len(csv_files)} 个AIS文件")
    
    # 准备并行任务 - 传递chunk_size参数
    tasks = [(str(f), str(output_dir), args.chunk_size) for f in csv_files]
    
    # 并行处理
    print("\n开始并行处理...")
    from tqdm import tqdm
    
    results = []
    with Pool(n_workers) as pool:
        for result in tqdm(pool.imap(process_single_file, tasks), total=len(tasks), desc='Processing'):
            results.append(result)
            gc.collect()  # 每完成一个文件强制GC
    
    # 打印每个文件的处理结果
    print("\n各文件处理结果:")
    total_ships = 0
    total_voyages = 0
    total_points = 0
    for file_stem, n_ships, n_voyages, n_points in results:
        if n_voyages > 0:
            print(f"  {file_stem}: {n_ships} 艘船, {n_voyages} 个航程, {n_points:,} 点")
            total_ships += n_ships
            total_voyages += n_voyages
            total_points += n_points
    
    print(f"\n处理完成: 共 {total_ships} 艘船, {total_voyages} 个航程, {total_points:,} 点")
    
    # 合并结果
    print("\n合并临时文件...")
    voyage_df, stop_df = merge_results(output_dir)
    
    if len(voyage_df) == 0:
        print("\n警告: 未提取到有效航程数据!")
        return
    
    # 数据质量过滤
    voyage_df, stop_df = filter_data_quality(voyage_df, stop_df)
    
    # 保存最终结果
    voyage_path = output_dir / 'processed_voyages.csv'
    voyage_df.to_csv(voyage_path, index=False)
    print(f"\n保存航程数据: {len(voyage_df):,} 条 -> {voyage_path}")
    
    if len(stop_df) > 0:
        stop_path = output_dir / 'port_stops.csv'
        stop_df.to_csv(stop_path, index=False)
        print(f"保存停靠数据: {len(stop_df):,} 条 -> {stop_path}")
    
    # 清理临时文件
    temp_dir = output_dir / 'temp'
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)
        print("清理临时文件完成")
    
    print_summary(voyage_df, stop_df)
    print("\n预处理完成!")


if __name__ == '__main__':
    main()
