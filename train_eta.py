#!/usr/bin/env python
"""
ETA预测模型训练脚本

双模型架构：
1. Informer-TP: 预测纯航行时间（使用预处理后的航程数据）
2. MLP: 预测港口停靠时间

最终ETA = 预测航行时间 + 预测停靠时间

用法：
    # 完整训练（包括停靠时间模型）
    python train_eta.py --epochs 3 --train_port_model
    
    # 仅训练航程模型
    python train_eta.py --epochs 3
    
    # 使用缓存的预处理数据
    python train_eta.py --use_cache --epochs 5
"""

import os
import sys
import gc
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import json
from concurrent.futures import ProcessPoolExecutor
from collections import deque

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.informer.model import Informer


def _basic_filter_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Step 3: 基础过滤（降低内存/提升质量）"""
    chunk = chunk.dropna()
    chunk = chunk[(chunk['remaining_hours'] >= 0) & (chunk['remaining_hours'] <= 720)]
    chunk = chunk[(chunk['sog'] >= 0) & (chunk['sog'] <= 30)]
    # 过滤超长航程（只保留航程时长 <= 30天 = 720小时的航程）
    if 'voyage_duration_hours' in chunk.columns:
        valid_voyages = chunk.groupby('voyage_id')['voyage_duration_hours'].first()
        valid_voyages = valid_voyages[valid_voyages <= 720].index
        chunk = chunk[chunk['voyage_id'].isin(valid_voyages)]
    return chunk


def _summarize_chunk_for_step3(chunk: pd.DataFrame) -> pd.DataFrame:
    """Step 3: 并行统计航程时间跨度（减少全局内存占用）"""
    chunk = _basic_filter_chunk(chunk)
    if len(chunk) == 0:
        return pd.DataFrame(columns=['t_min', 't_max', 'dur', 'rows'])
    chunk['postime_dt'] = pd.to_datetime(chunk['postime'])
    summary = chunk.groupby('voyage_id').agg(
        t_min=('postime_dt', 'min'),
        t_max=('postime_dt', 'max'),
        dur=('voyage_duration_hours', 'first'),
        rows=('voyage_id', 'size')
    )
    return summary


def _filter_and_select_chunk(chunk: pd.DataFrame, selected_voyage_ids: set) -> pd.DataFrame:
    """Step 3: 过滤并保留目标航程"""
    chunk = _basic_filter_chunk(chunk)
    if len(chunk) == 0:
        return chunk
    return chunk[chunk['voyage_id'].isin(selected_voyage_ids)]


def _update_minmax(current_min, current_max, values: np.ndarray):
    if current_min is None:
        return values.min(axis=0), values.max(axis=0)
    return np.minimum(current_min, values.min(axis=0)), np.maximum(current_max, values.max(axis=0))


def _update_welford(count, mean, m2, values: np.ndarray):
    # values: 1D array
    for x in values:
        count += 1
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        m2 += delta * delta2
    return count, mean, m2


def _finalize_welford(count, mean, m2):
    if count < 2:
        return mean, 1.0
    variance = m2 / (count - 1)
    return mean, np.sqrt(variance)


def _compute_geom_features(group: pd.DataFrame) -> pd.DataFrame:
    """为单个航程计算几何特征"""
    group = group.sort_values('postime')
    dest_lat = group['lat'].iloc[-1]
    dest_lon = group['lon'].iloc[-1]
    dist_km = VoyageETADataset._haversine_km(group['lat'].values, group['lon'].values, dest_lat, dest_lon)
    bearing_to_dest = VoyageETADataset._bearing_deg(group['lat'].values, group['lon'].values, dest_lat, dest_lon)
    bearing_diff = np.abs(((bearing_to_dest - group['cog'].values + 180) % 360) - 180)
    group = group.copy()
    group['dist_to_dest_km'] = dist_km
    group['bearing_diff'] = bearing_diff
    
    return group


def build_decoder_input(X, label_len, pred_len):
    """构建Informer解码器输入"""
    dec_len = label_len + pred_len
    X_dec = np.zeros((len(X), dec_len, X.shape[-1]), dtype=X.dtype)
    X_dec[:, :label_len, :] = X[:, -label_len:, :]
    return X_dec


class MemmapDataset(torch.utils.data.Dataset):
    """支持memmap的数据集，避免全部加载到内存"""
    
    def __init__(self, X_path, X_mark_enc_path, X_dec_path, X_mark_dec_path, y_path, sd_path, actual_length=None):
        # 使用memmap模式打开文件（只读，不加载到内存）
        self.X = np.load(X_path, mmap_mode='r')
        self.X_mark_enc = np.load(X_mark_enc_path, mmap_mode='r')
        self.X_dec = np.load(X_dec_path, mmap_mode='r')
        self.X_mark_dec = np.load(X_mark_dec_path, mmap_mode='r')
        self.y = np.load(y_path, mmap_mode='r')
        self.sd = np.load(sd_path, mmap_mode='r')
        # 使用实际写入的长度（避免memmap尾部零填充样本）
        self.length = actual_length if actual_length is not None else len(self.y)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 只有访问时才读取对应的数据
        return (
            torch.from_numpy(self.X[idx].copy()).float(),
            torch.from_numpy(self.X_mark_enc[idx].copy()).float(),
            torch.from_numpy(self.X_dec[idx].copy()).float(),
            torch.from_numpy(self.X_mark_dec[idx].copy()).float(),
            torch.tensor(self.y[idx]).float(),
            torch.tensor(self.sd[idx]).float()
        )


def create_memmap_arrays(cache_dir: Path, n_samples: int, seq_len: int, label_len: int, pred_len: int, n_features: int = 6, prefix: str = 'train'):
    """创建memmap数组用于增量写入序列
    
    n_features = 6: lat, lon, sog, cog, dist_to_dest_km, bearing_diff
    """
    dec_len = label_len + pred_len
    
    X = np.lib.format.open_memmap(
        cache_dir / f"X_{prefix}.npy", mode='w+', dtype=np.float32, shape=(n_samples, seq_len, n_features)
    )
    X_mark_enc = np.lib.format.open_memmap(
        cache_dir / f"X_mark_enc_{prefix}.npy", mode='w+', dtype=np.float32, shape=(n_samples, seq_len, 5)
    )
    X_dec = np.lib.format.open_memmap(
        cache_dir / f"X_dec_{prefix}.npy", mode='w+', dtype=np.float32, shape=(n_samples, dec_len, n_features)
    )
    X_mark_dec = np.lib.format.open_memmap(
        cache_dir / f"X_mark_dec_{prefix}.npy", mode='w+', dtype=np.float32, shape=(n_samples, dec_len, 5)
    )
    y = np.lib.format.open_memmap(
        cache_dir / f"y_{prefix}.npy", mode='w+', dtype=np.float32, shape=(n_samples,)
    )
    sd = np.lib.format.open_memmap(
        cache_dir / f"sd_{prefix}.npy", mode='w+', dtype=np.float32, shape=(n_samples,)
    )
    
    return X, X_mark_enc, X_dec, X_mark_dec, y, sd


# ============================================================
# Part 1: 港口停靠时间预测模型 (MLP)
# ============================================================

class PortStopPredictor(nn.Module):
    """港口停靠时间预测MLP"""
    
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class PortStopModel:
    """港口停靠时间预测模型管理器"""
    
    def __init__(self, model_dir: str = './output/port_model'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = StandardScaler()
        self.region_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.region_classes = []
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """准备特征"""
        features = df.copy()
        
        features['arrival_time'] = pd.to_datetime(features['arrival_time'])
        features['month'] = features['arrival_time'].dt.month
        features['weekday'] = features['arrival_time'].dt.weekday
        
        # 周期编码
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        features['weekday_sin'] = np.sin(2 * np.pi * features['weekday'] / 7)
        features['weekday_cos'] = np.cos(2 * np.pi * features['weekday'] / 7)
        
        feature_cols = ['lon', 'lat', 'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos']
        
        # 区域编码
        if fit:
            self.region_encoder.fit(features['region'])
            self.region_classes = list(self.region_encoder.classes_)
        
        region_encoded = self.region_encoder.transform(features['region'])
        num_regions = len(self.region_classes)
        region_onehot = np.zeros((len(features), num_regions))
        region_onehot[np.arange(len(features)), region_encoded] = 1
        
        numeric_features = features[feature_cols].values
        all_features = np.hstack([numeric_features, region_onehot])
        
        if fit:
            all_features = self.scaler.fit_transform(all_features)
        else:
            all_features = self.scaler.transform(all_features)
        
        return all_features
    
    def train(self, stop_df: pd.DataFrame, epochs: int = 100, batch_size: int = 32, lr: float = 0.001):
        """训练模型"""
        print(f"Training port stop predictor on {len(stop_df)} samples...")
        
        stop_df = stop_df[stop_df['duration_hours'] > 0].copy()
        stop_df = stop_df[stop_df['duration_hours'] < 200]
        
        if len(stop_df) < 10:
            print("Not enough data to train port model")
            return None
        
        X = self.prepare_features(stop_df, fit=True)
        y = np.log1p(stop_df['duration_hours'].values)
        
        # 划分
        n = len(X)
        indices = np.random.permutation(n)
        train_size = int(n * 0.8)
        train_idx, val_idx = indices[:train_size], indices[train_size:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        input_dim = X.shape[1]
        self.model = PortStopPredictor(input_dim).to(self.device)
        
        optimizer = Adam(self.model.parameters(), lr=lr)
        criterion = nn.HuberLoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
            
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    val_loss += criterion(self.model(batch_x), batch_y).item()
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save()
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Val Loss={val_loss:.4f}")
        
        # 最终评估
        self.model.eval()
        with torch.no_grad():
            pred_log = self.model(torch.FloatTensor(X_val).to(self.device)).cpu().numpy()
            pred_hours = np.expm1(pred_log)
            actual_hours = np.expm1(y_val)
            mae = np.mean(np.abs(pred_hours - actual_hours))
        
        print(f"  Port model MAE: {mae:.2f} hours")
        return {'mae': mae}
    
    def predict(self, stop_df: pd.DataFrame) -> np.ndarray:
        """预测停靠时间"""
        if self.model is None:
            self.load()
        
        X = self.prepare_features(stop_df, fit=False)
        self.model.eval()
        with torch.no_grad():
            pred_log = self.model(torch.FloatTensor(X).to(self.device)).cpu().numpy()
        return np.expm1(pred_log)
    
    def save(self):
        if self.model is None:
            return
        torch.save(self.model.state_dict(), self.model_dir / 'model.pth')
        np.save(self.model_dir / 'scaler_mean.npy', self.scaler.mean_)
        np.save(self.model_dir / 'scaler_scale.npy', self.scaler.scale_)
        config = {'region_classes': self.region_classes, 'input_dim': self.model.network[0].in_features}
        with open(self.model_dir / 'config.json', 'w') as f:
            json.dump(config, f)
    
    def load(self):
        with open(self.model_dir / 'config.json', 'r') as f:
            config = json.load(f)
        self.region_classes = config['region_classes']
        self.region_encoder.classes_ = np.array(self.region_classes)
        self.scaler.mean_ = np.load(self.model_dir / 'scaler_mean.npy')
        self.scaler.scale_ = np.load(self.model_dir / 'scaler_scale.npy')
        self.model = PortStopPredictor(config['input_dim']).to(self.device)
        self.model.load_state_dict(torch.load(self.model_dir / 'model.pth', map_location=self.device))


# ============================================================
# Part 2: 航程ETA数据处理
# ============================================================

class VoyageETADataset:
    """航程数据处理器"""
    
    def __init__(self, seq_len: int = 48, label_len: int = 24, pred_len: int = 1):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        
        self.feature_min = None
        self.feature_max = None
        self.target_mean = None
        self.target_std = None

    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2):
        """计算两点间距离（km）"""
        lat1 = np.deg2rad(lat1)
        lon1 = np.deg2rad(lon1)
        lat2 = np.deg2rad(lat2)
        lon2 = np.deg2rad(lon2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371.0 * c

    @staticmethod
    def _bearing_deg(lat1, lon1, lat2, lon2):
        """计算方位角（度）"""
        lat1 = np.deg2rad(lat1)
        lon1 = np.deg2rad(lon1)
        lat2 = np.deg2rad(lat2)
        lon2 = np.deg2rad(lon2)
        dlon = lon2 - lon1
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        bearing = np.rad2deg(np.arctan2(y, x))
        return (bearing + 360) % 360
    
    def normalize_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """Minmax归一化（带clipping防止测试集超出范围）"""
        if fit:
            self.feature_min = features.min(axis=0)
            self.feature_max = features.max(axis=0)
        
        range_vals = self.feature_max - self.feature_min
        range_vals[range_vals == 0] = 1.0
        normalized = (features - self.feature_min) / range_vals
        # 限制范围，防止测试集极端值导致模型输出爆炸
        return np.clip(normalized, -0.5, 1.5)
    
    def normalize_target(self, target: np.ndarray, fit: bool = False) -> np.ndarray:
        """Log + 标准化"""
        log_target = np.log1p(target)
        if fit:
            self.target_mean = log_target.mean()
            self.target_std = log_target.std()
        return (log_target - self.target_mean) / self.target_std
    
    def inverse_normalize_target(self, normalized: np.ndarray) -> np.ndarray:
        """反归一化目标值"""
        log_target = normalized * self.target_std + self.target_mean
        return np.expm1(log_target)
    
    def inverse_normalize_features(self, normalized: np.ndarray) -> np.ndarray:
        """特征反归一化"""
        if self.feature_min is None or self.feature_max is None:
            return normalized
        range_vals = self.feature_max - self.feature_min
        range_vals[range_vals == 0] = 1.0
        return normalized * range_vals + self.feature_min
    
    def create_sequences(self, voyage_df: pd.DataFrame, max_sequences: int = 50000000, fit: bool = True, return_meta: bool = False) -> tuple:
        """从航程数据创建序列（内存优化版 + 分层采样）
        
        优化策略：
        1. 先统计总序列数，预分配数组
        2. 使用float32减少内存
        3. 分层采样：确保高remaining_hours样本被充分采样
        """
        # 原始6个特征
        feature_cols = ['lat', 'lon', 'sog', 'cog', 'dist_to_dest_km', 'bearing_diff']
        group_col = 'voyage_id' if 'voyage_id' in voyage_df.columns else 'mmsi'
        
        # 第一遍：统计每个航程可产生的序列数
        print("  统计序列数量...")
        voyage_seq_counts = []
        voyage_ids = []
        voyage_max_remaining = []  # 记录每个航程的最大remaining_hours
        
        for vid, group in voyage_df.groupby(group_col):
            n_seq = max(0, len(group) - self.seq_len - self.pred_len + 1)
            if n_seq > 0:
                voyage_seq_counts.append(n_seq)
                voyage_ids.append(vid)
                voyage_max_remaining.append(group['remaining_hours'].max())
        
        total_possible = sum(voyage_seq_counts)
        print(f"  可生成序列: {total_possible:,}, 目标: {max_sequences:,}")
        
        # 分层采样：按航程时长分类，确保各类别都有足够样本
        voyage_max_remaining = np.array(voyage_max_remaining)
        
        # 定义时长分层
        strata_bins = [0, 100, 200, 400, 720]
        strata_labels = ['<100h', '100-200h', '200-400h', '>400h']
        strata_weights = [1.0, 1.5, 2.0, 3.0]  # 长航程权重更高
        
        # 计算每个航程的采样权重
        voyage_strata = np.digitize(voyage_max_remaining, strata_bins[1:])
        voyage_weights = np.array([strata_weights[min(s, len(strata_weights)-1)] for s in voyage_strata])
        
        # 基于权重计算采样数量
        if total_possible > max_sequences:
            base_ratio = max_sequences / total_possible
            samples_per_voyage = []
            for i, c in enumerate(voyage_seq_counts):
                # 权重调整的采样数
                weighted_sample = int(c * base_ratio * voyage_weights[i])
                samples_per_voyage.append(max(1, min(c, weighted_sample)))
            actual_total = sum(samples_per_voyage)
            
            # 如果超出目标，按比例缩减
            if actual_total > max_sequences * 1.2:
                scale = max_sequences / actual_total
                samples_per_voyage = [max(1, int(s * scale)) for s in samples_per_voyage]
                actual_total = sum(samples_per_voyage)
        else:
            samples_per_voyage = voyage_seq_counts
            actual_total = total_possible
        
        # 打印分层统计
        for i, label in enumerate(strata_labels):
            mask = (voyage_strata == i)
            n_voyages = mask.sum()
            n_samples = sum(samples_per_voyage[j] for j in range(len(voyage_ids)) if mask[j])
            print(f"  {label}: {n_voyages} 航程, {n_samples:,} 样本")
        
        print(f"  实际采样: {actual_total:,} 序列")
        
        # 预分配数组（float32节省内存）
        X = np.zeros((actual_total, self.seq_len, len(feature_cols)), dtype=np.float32)
        X_mark_enc = np.zeros((actual_total, self.seq_len, 5), dtype=np.float32)
        X_mark_dec = np.zeros((actual_total, self.label_len + self.pred_len, 5), dtype=np.float32)
        y = np.zeros(actual_total, dtype=np.float32)
        sailing_days = np.zeros(actual_total, dtype=np.float32)
        meta_mmsi = np.zeros(actual_total, dtype=np.int64)
        meta_voyage_id = np.empty(actual_total, dtype=object)
        meta_pred_time = np.empty(actual_total, dtype='datetime64[ns]')
        meta_end_time = np.empty(actual_total, dtype='datetime64[ns]')
        
        # 第二遍：填充数据
        print("  生成序列...")
        idx = 0
        for i, (vid, group) in enumerate(voyage_df.groupby(group_col)):
            if vid not in voyage_ids:
                continue
            
            vid_idx = voyage_ids.index(vid)
            n_to_sample = samples_per_voyage[vid_idx]
            
            group = group.sort_values('postime')
            # 使用统一的几何特征计算函数（包含航速增强特征）
            group = _compute_geom_features(group)

            features = group[feature_cols].values.astype(np.float32)
            targets = group['remaining_hours'].values.astype(np.float32)
            postime = pd.to_datetime(group['postime'])
            
            first_time = postime.iloc[0]
            sd_values = ((postime - first_time).dt.total_seconds() / 86400).values.astype(np.float32)
            
            # 时间特征
            time_marks = np.stack([
                postime.dt.month.values / 12,
                postime.dt.day.values / 31,
                postime.dt.weekday.values / 7,
                postime.dt.hour.values / 24,
                postime.dt.minute.values / 60
            ], axis=1).astype(np.float32)
            
            # 可用的起始位置
            max_start = len(features) - self.seq_len - self.pred_len + 1
            if max_start <= 0:
                continue
            
            # 均匀采样起始位置
            if n_to_sample >= max_start:
                starts = list(range(max_start))
            else:
                starts = np.linspace(0, max_start - 1, n_to_sample, dtype=int).tolist()
            
            for start in starts:
                if idx >= actual_total:
                    break
                X[idx] = features[start:start+self.seq_len]
                X_mark_enc[idx] = time_marks[start:start+self.seq_len]
                X_mark_dec[idx] = time_marks[start+self.seq_len-self.label_len:start+self.seq_len+self.pred_len]
                y[idx] = targets[start+self.seq_len]
                sailing_days[idx] = sd_values[start+self.seq_len]
                meta_mmsi[idx] = group['mmsi'].iloc[start+self.seq_len]
                meta_voyage_id[idx] = vid
                meta_pred_time[idx] = postime.iloc[start+self.seq_len].to_datetime64()
                meta_end_time[idx] = postime.iloc[-1].to_datetime64()
                idx += 1
            
            if idx >= actual_total:
                break
        
        # 截断到实际使用的大小
        X = X[:idx]
        X_mark_enc = X_mark_enc[:idx]
        X_mark_dec = X_mark_dec[:idx]
        y = y[:idx]
        sailing_days = sailing_days[:idx]
        meta_mmsi = meta_mmsi[:idx]
        meta_voyage_id = meta_voyage_id[:idx]
        meta_pred_time = meta_pred_time[:idx]
        meta_end_time = meta_end_time[:idx]
        
        print(f"  最终序列数: {idx:,}")
        
        # 归一化
        X_flat = X.reshape(-1, X.shape[-1])
        X_norm = self.normalize_features(X_flat, fit=fit).reshape(X.shape).astype(np.float32)
        y_norm = self.normalize_target(y, fit=fit).astype(np.float32)

        if return_meta:
            meta = {
                'mmsi': meta_mmsi,
                'voyage_id': meta_voyage_id,
                'pred_time': meta_pred_time,
                'end_time': meta_end_time
            }
            return X_norm, X_mark_enc, X_mark_dec, y_norm, sailing_days, meta

        return X_norm, X_mark_enc, X_mark_dec, y_norm, sailing_days
    
    def save_params(self, path: str):
        """保存归一化参数"""
        np.savez(path,
                 feature_min=self.feature_min,
                 feature_max=self.feature_max,
                 target_mean=self.target_mean,
                 target_std=self.target_std)
    
    def load_params(self, path: str):
        """加载归一化参数"""
        data = np.load(path)
        self.feature_min = data['feature_min']
        self.feature_max = data['feature_max']
        self.target_mean = data['target_mean']
        self.target_std = data['target_std']


def create_data_loaders(X, X_mark_enc, X_mark_dec, y, sailing_days, label_len, pred_len, batch_size=32):
    """创建数据加载器"""
    n = len(X)
    indices = np.random.permutation(n)
    
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]
    
    dec_len = label_len + pred_len
    
    def make_loader(idx, shuffle=True):
        X_batch = torch.FloatTensor(X[idx])
        X_mark_enc_batch = torch.FloatTensor(X_mark_enc[idx])
        X_mark_dec_batch = torch.FloatTensor(X_mark_dec[idx])
        y_batch = torch.FloatTensor(y[idx])
        sd_batch = torch.FloatTensor(sailing_days[idx])
        
        X_dec = torch.zeros(len(idx), dec_len, X.shape[-1])
        X_dec[:, :label_len, :] = X_batch[:, -label_len:, :]
        
        dataset = TensorDataset(X_batch, X_mark_enc_batch, X_dec, X_mark_dec_batch, y_batch, sd_batch)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return (make_loader(train_idx, True), make_loader(val_idx, False), make_loader(test_idx, False),
            sailing_days[test_idx], y[test_idx], test_idx)


# ============================================================
# Part 3: Informer训练器
# ============================================================

class InformerTrainer:
    """Informer模型训练器"""
    
    def __init__(self, model, device, lr=5e-6, scheduler_type='plateau', epochs=10, steps_per_epoch=None):
        """
        Args:
            scheduler_type: 学习率调度器类型
                - 'plateau': ReduceLROnPlateau (验证损失不下降时降低LR)
                - 'cosine': CosineAnnealingLR (余弦退火)
                - 'cosine_restart': CosineAnnealingWarmRestarts (带热重启的余弦退火)
                - 'onecycle': OneCycleLR (一周期策略，自动找最佳LR)
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.HuberLoss(delta=1.0)
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler_type = scheduler_type
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        # 根据类型选择学习率调度器
        if scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=lr * 0.01)
        elif scheduler_type == 'cosine_restart':
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=max(1, epochs // 3), T_mult=2, eta_min=lr * 0.01)
        elif scheduler_type == 'onecycle':
            if steps_per_epoch is None:
                raise ValueError("OneCycleLR requires steps_per_epoch")
            self.scheduler = OneCycleLR(self.optimizer, max_lr=lr * 10, epochs=epochs, steps_per_epoch=steps_per_epoch)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(loader, desc="Training", leave=False):
            x_enc, x_mark_enc, x_dec, x_mark_dec, y, _ = batch
            
            x_enc = x_enc.to(self.device)
            x_mark_enc = x_mark_enc.to(self.device)
            x_dec = x_dec.to(self.device)
            x_mark_dec = x_mark_dec.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            output = output.squeeze(-1).squeeze(-1)
            
            loss = self.criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # OneCycleLR 需要每个batch后更新
            if self.scheduler_type == 'onecycle':
                self.scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in loader:
                x_enc, x_mark_enc, x_dec, x_mark_dec, y, _ = batch
                
                x_enc = x_enc.to(self.device)
                x_mark_enc = x_mark_enc.to(self.device)
                x_dec = x_dec.to(self.device)
                x_mark_dec = x_mark_dec.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                output = output.squeeze(-1).squeeze(-1)
                loss = self.criterion(output, y)
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def predict(self, loader):
        self.model.eval()
        preds, trues, sailing_days_list = [], [], []
        
        with torch.no_grad():
            for batch in loader:
                x_enc, x_mark_enc, x_dec, x_mark_dec, y, sd = batch
                
                x_enc = x_enc.to(self.device)
                x_mark_enc = x_mark_enc.to(self.device)
                x_dec = x_dec.to(self.device)
                x_mark_dec = x_mark_dec.to(self.device)
                
                output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                output = output.squeeze(-1).squeeze(-1)
                
                preds.append(output.cpu().numpy())
                trues.append(y.numpy())
                sailing_days_list.append(sd.numpy())
        
        return np.concatenate(preds), np.concatenate(trues), np.concatenate(sailing_days_list)
    
    def step_scheduler(self, val_loss=None):
        """更新学习率调度器"""
        if self.scheduler_type == 'plateau':
            self.scheduler.step(val_loss)
        elif self.scheduler_type == 'onecycle':
            pass  # OneCycleLR在train_epoch中每个batch后更新
        else:
            self.scheduler.step()
    
    def save(self, path):
        """保存模型权重"""
        torch.save(self.model.state_dict(), path)
    
    def save_checkpoint(self, path, epoch, val_loss):
        """保存完整训练状态（用于继续训练）"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
            'best_val_loss': val_loss,
            'scheduler_type': self.scheduler_type
        }
        torch.save(checkpoint, path)
        print(f"  -> 保存检查点: epoch={epoch+1}, val_loss={val_loss:.4f}")
    
    def load(self, path):
        """加载模型权重"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
    
    def load_checkpoint(self, path):
        """加载完整训练状态（用于继续训练）"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] is not None and self.scheduler_type == checkpoint.get('scheduler_type'):
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except:
                print("  警告: 调度器状态加载失败，使用新调度器")
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"  从 epoch {self.start_epoch} 继续训练, best_val_loss={self.best_val_loss:.4f}")
        return self.start_epoch


# ============================================================
# Part 4: 评估和可视化
# ============================================================

def calculate_metrics(y_pred, y_true):
    """计算评估指标"""
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(mse)
    
    mask = y_true > 24
    mape = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0
    
    return {'MSE': mse, 'MAE_hours': mae, 'MAE_days': mae / 24, 'RMSE': rmse, 'MAPE': mape}


def analyze_bad_cases(y_pred, y_true, X_input, test_meta, dataset, save_dir, threshold=200, suffix=""):
    """分析误差较大的样本并保存"""
    errors = np.abs(y_pred - y_true)
    bad_indices = np.where(errors > threshold)[0]
    
    if len(bad_indices) == 0:
        return

    print(f"\n【Bad Cases 分析 ({suffix.strip('_')})】发现 {len(bad_indices)} 个样本误差超过 {threshold} 小时")
    
    # 反归一化特征 (取序列最后一步)
    # X_input: (N, seq_len, featuring_dim)
    last_step_features = X_input[bad_indices, -1, :]
    raw_features = dataset.inverse_normalize_features(last_step_features)
    
    # 11个特征: 原始6个 + 5个航速增强特征
    feature_cols = ['lat', 'lon', 'sog', 'cog', 'dist_to_dest_km', 'bearing_diff',
                    'sog_hist_mean', 'sog_hist_std', 'sog_deviation', 'voyage_progress', 'sog_short_avg']
    
    records = []
    for i, idx in enumerate(bad_indices):
        rec = {
            'mmsi': test_meta['mmsi'][idx],
            'voyage_id': test_meta['voyage_id'][idx],
            'pred_time': test_meta['pred_time'][idx],
            'end_time': test_meta['end_time'][idx],
            'true_hours': y_true[idx],
            'pred_hours': y_pred[idx],
            'error_hours': errors[idx],
            'error_ratio': (errors[idx] / y_true[idx]) if y_true[idx] > 1.0 else 0
        }
        
        # 添加特征
        for j, col in enumerate(feature_cols):
            if j < raw_features.shape[1]:
                rec[col] = raw_features[i, j]
            
        records.append(rec)
        
    df = pd.DataFrame(records)
    df = df.sort_values('error_hours', ascending=False)
    
    filename = f'bad_cases{suffix}.csv'
    save_path = os.path.join(save_dir, filename)
    df.to_csv(save_path, index=False)
    print(f"  已保存 Bad Cases 到: {save_path}")
    print(df[['voyage_id', 'true_hours', 'pred_hours', 'error_hours', 'dist_to_dest_km']].head().to_string())


def plot_results(y_pred, y_true, sailing_days, save_dir, suffix="", title_prefix=""):
    """绘制评估结果
    
    Args:
        suffix: 文件名后缀，如 '_sailing' 或 '_total'
        title_prefix: 图标题前缀，如 '单模型' 或 '双模型'
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 误差vs航行天数
    ax = axes[0, 0]
    scatter = ax.scatter(sailing_days, np.abs(y_pred - y_true), c=y_true, cmap='viridis', alpha=0.5, s=10)
    ax.set_xlabel('Sailing Days')
    ax.set_ylabel('Absolute Error (hours)')
    ax.set_title('Prediction Error vs Sailing Days')
    plt.colorbar(scatter, ax=ax, label='True Remaining Hours')
    
    # 2. 分箱MAE
    ax = axes[0, 1]
    bins = np.arange(0, sailing_days.max() + 1, 1)
    bin_indices = np.digitize(sailing_days, bins)
    bin_maes, bin_centers = [], []
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if mask.sum() > 10:
            bin_maes.append(np.mean(np.abs(y_pred[mask] - y_true[mask])))
            bin_centers.append((bins[i-1] + bins[i]) / 2)
    ax.bar(bin_centers, bin_maes, width=0.8, alpha=0.7, color='steelblue')
    ax.set_xlabel('Sailing Days')
    ax.set_ylabel('MAE (hours)')
    ax.set_title('MAE by Sailing Days')
    ax.grid(True, alpha=0.3)
    
    # 3. 预测vs实际 - 使用2D直方图展示密度
    ax = axes[1, 0]
    max_val = max(y_true.max(), y_pred.max())
    
    # 使用hexbin展示点密度
    hb = ax.hexbin(y_true, y_pred, gridsize=50, cmap='Blues', mincnt=1, 
                   extent=[0, max_val, 0, max_val])
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect')
    ax.set_xlabel('Actual Hours')
    ax.set_ylabel('Predicted Hours')
    ax.set_title('Predicted vs Actual (density)')
    ax.legend()
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    plt.colorbar(hb, ax=ax, label='Count')
    
    # 4. 误差分布
    ax = axes[1, 1]
    errors = y_pred - y_true
    ax.hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--')
    ax.axvline(x=np.mean(errors), color='orange', linestyle='-', label=f'Mean: {np.mean(errors):.1f}h')
    ax.set_xlabel('Prediction Error (hours)')
    ax.set_ylabel('Count')
    ax.set_title(f'{title_prefix}Error Distribution' if title_prefix else 'Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'analysis_plots{suffix}.png' if suffix else 'analysis_plots.png'
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close()
    print(f"Plots saved to {save_dir}/{filename}")


# ============================================================
# Part 5: 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='ETA预测模型训练')
    
    # 数据路径
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--processed_dir', type=str, default='./output/processed')
    
    # 模型参数
    parser.add_argument('--seq_len', type=int, default=48)
    parser.add_argument('--label_len', type=int, default=24)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.05)
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader并行加载线程数')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default='plateau', 
                        choices=['plateau', 'cosine', 'cosine_restart', 'onecycle'],
                        help='学习率调度器: plateau(默认), cosine, cosine_restart, onecycle')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='验证损失连续N个epoch未改善则停止训练（0=禁用）')
    parser.add_argument('--resume', action='store_true', help='从checkpoint继续训练')
    
    # 港口模型
    parser.add_argument('--train_port_model', action='store_true', help='训练港口停靠时间模型')
    parser.add_argument('--port_epochs', type=int, default=100)
    
    # 其他
    parser.add_argument('--use_cache', action='store_true', default=True, help='使用缓存的预处理数据')
    parser.add_argument('--max_files', type=int, default=None, help='处理文件数限制')
    parser.add_argument('--max_voyages', type=int, default=150000, help='最多使用的航程数（控制内存）')
    parser.add_argument('--max_sequences', type=int, default=50000000, help='最多生成的训练序列数')
    parser.add_argument('--max_seqs_per_bucket', type=int, default=500000, help='每个bucket最多处理的序列数（控制内存峰值）')
    parser.add_argument('--eval_only', action='store_true', help='仅评估和重绘图（跳过训练）')
    parser.add_argument('--step3_workers', type=int, default=os.cpu_count(), help='Step3并行工作进程数')
    parser.add_argument('--step3_chunk_size', type=int, default=500000, help='Step3读取CSV的chunk大小')
    parser.add_argument('--step3_spill', action='store_true', default=True, help='Step3将过滤数据分桶写入磁盘，降低内存峰值')
    parser.add_argument('--step3_bucket_count', type=int, default=32, help='Step3分桶数量（spill模式）')
    parser.add_argument('--step3_bucket_rows', type=int, default=1_000_000, help='Step3分桶缓存行数（spill模式）')
    parser.add_argument('--use_memmap', action='store_true', default=True, help='使用memmap存储序列，避免OOM')
    parser.add_argument('--no_use_memmap', action='store_true', help='禁用memmap模式')
    
    args = parser.parse_args()
    
    # 处理 --no_use_memmap
    if args.no_use_memmap:
        args.use_memmap = False
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ===== Step 1: 加载数据 =====
    print("\n" + "="*60)
    print("Step 1: 加载数据")
    print("="*60)
    
    voyage_path = os.path.join(args.processed_dir, 'processed_voyages.csv')
    stop_path = os.path.join(args.processed_dir, 'port_stops.csv')
    
    if not os.path.exists(voyage_path):
        print("预处理数据不存在，请先运行: python preprocess_data.py")
        return
    
    # 只统计行数，不加载整个文件
    print("统计数据量...")
    voyage_count = sum(1 for _ in open(voyage_path)) - 1  # 减去header
    stop_df = pd.read_csv(stop_path) if os.path.exists(stop_path) else pd.DataFrame()
    
    print(f"航程数据: {voyage_count:,} 条")
    print(f"停靠数据: {len(stop_df):,} 条")
    
    # ===== Step 2: 训练港口停靠模型（可选）=====
    if args.train_port_model and len(stop_df) > 10:
        print("\n" + "="*60)
        print("Step 2: 训练港口停靠时间预测模型")
        print("="*60)
        
        port_model = PortStopModel(os.path.join(args.output_dir, 'port_model'))
        port_model.train(stop_df, epochs=args.port_epochs)
        
        print("\n各区域平均停靠时间（baseline）:")
        for region, avg in stop_df.groupby('region')['duration_hours'].mean().items():
            print(f"  {region}: {avg:.2f} 小时")
    
    # ===== Step 3: 准备训练数据 =====
    print("\n" + "="*60)
    print("Step 3: 准备训练数据")
    print("="*60)
    
    # 采样参数 - 97M数据太大，采样部分航程训练
    max_voyages = args.max_voyages  # 使用命令行参数

    # 缓存路径（可跳过Step 3）
    cache_tag = f"seq{args.seq_len}_label{args.label_len}_pred{args.pred_len}_mv{args.max_voyages}_ms{args.max_sequences}"
    cache_dir = Path(args.output_dir) / "cache_sequences" / cache_tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_ready = (cache_dir / "X_train.npy").exists()
    use_memmap_mode = False  # 默认不使用memmap模式
    
    if args.use_cache and cache_ready:
        print(f"使用缓存: {cache_dir}")
        
        # 检查是否有X_dec文件（memmap模式生成的缓存）
        has_dec_cache = (cache_dir / "X_dec_train.npy").exists()
        
        if args.use_memmap and has_dec_cache:
            # 使用memmap模式加载（避免全部加载到内存）
            print("使用memmap模式加载缓存...")
            use_memmap_mode = True
            
            dataset = VoyageETADataset(seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len)
            dataset.load_params(os.path.join(args.output_dir, 'norm_params.npz'))
            
            # 加载实际写入的序列数（避免memmap尾部零填充样本）
            counts_path = cache_dir / "actual_counts.npy"
            actual_counts = None
            if counts_path.exists():
                actual_counts = np.load(counts_path, allow_pickle=True).item()
                print(f"使用实际序列数: train={actual_counts.get('train', '?'):,}, "
                      f"val={actual_counts.get('val', '?'):,}, test={actual_counts.get('test', '?'):,}")
            
            train_dataset = MemmapDataset(
                cache_dir / "X_train.npy", cache_dir / "X_mark_enc_train.npy",
                cache_dir / "X_dec_train.npy", cache_dir / "X_mark_dec_train.npy",
                cache_dir / "y_train.npy", cache_dir / "sd_train.npy",
                actual_length=actual_counts.get('train') if actual_counts else None
            )
            val_dataset = MemmapDataset(
                cache_dir / "X_val.npy", cache_dir / "X_mark_enc_val.npy",
                cache_dir / "X_dec_val.npy", cache_dir / "X_mark_dec_val.npy",
                cache_dir / "y_val.npy", cache_dir / "sd_val.npy",
                actual_length=actual_counts.get('val') if actual_counts else None
            )
            test_dataset = MemmapDataset(
                cache_dir / "X_test.npy", cache_dir / "X_mark_enc_test.npy",
                cache_dir / "X_dec_test.npy", cache_dir / "X_mark_dec_test.npy",
                cache_dir / "y_test.npy", cache_dir / "sd_test.npy",
                actual_length=actual_counts.get('test') if actual_counts else None
            )
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
            
            # 为评估阶段准备
            X_test = np.load(cache_dir / "X_test.npy", mmap_mode='r')
            y_test = np.load(cache_dir / "y_test.npy", mmap_mode='r')
            test_meta = np.load(cache_dir / "test_meta.npy", allow_pickle=True).item()
            
            print(f"训练集: {len(train_dataset):,}, 验证集: {len(val_dataset):,}, 测试集: {len(test_dataset):,}")
        else:
            # 传统方式加载缓存
            X_train = np.load(cache_dir / "X_train.npy", allow_pickle=True)
            X_mark_enc_train = np.load(cache_dir / "X_mark_enc_train.npy", allow_pickle=True)
            X_mark_dec_train = np.load(cache_dir / "X_mark_dec_train.npy", allow_pickle=True)
            y_train = np.load(cache_dir / "y_train.npy", allow_pickle=True)
            sd_train = np.load(cache_dir / "sd_train.npy", allow_pickle=True)

            X_val = np.load(cache_dir / "X_val.npy", allow_pickle=True)
            X_mark_enc_val = np.load(cache_dir / "X_mark_enc_val.npy", allow_pickle=True)
            X_mark_dec_val = np.load(cache_dir / "X_mark_dec_val.npy", allow_pickle=True)
            y_val = np.load(cache_dir / "y_val.npy", allow_pickle=True)
            sd_val = np.load(cache_dir / "sd_val.npy", allow_pickle=True)

            X_test = np.load(cache_dir / "X_test.npy", allow_pickle=True)
            X_mark_enc_test = np.load(cache_dir / "X_mark_enc_test.npy", allow_pickle=True)
            X_mark_dec_test = np.load(cache_dir / "X_mark_dec_test.npy", allow_pickle=True)
            y_test = np.load(cache_dir / "y_test.npy", allow_pickle=True)
            sd_test = np.load(cache_dir / "sd_test.npy", allow_pickle=True)
            test_meta = np.load(cache_dir / "test_meta.npy", allow_pickle=True).item()

            dataset = VoyageETADataset(seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len)
            dataset.load_params(os.path.join(args.output_dir, 'norm_params.npz'))
    else:
        # 分块读取CSV，只加载需要的列，使用float32
        print("分块读取航程数据...")
        use_cols = ['lat', 'lon', 'sog', 'cog', 'remaining_hours', 'mmsi', 'postime', 'voyage_id', 'voyage_duration_hours']
        dtype = {'lat': 'float32', 'lon': 'float32', 'sog': 'float32', 'cog': 'float32', 
                 'remaining_hours': 'float32', 'voyage_duration_hours': 'float32'}

        chunk_size = args.step3_chunk_size
        num_workers = max(1, args.step3_workers or 1)
        print(f"Step3并行: workers={num_workers}, chunk_size={chunk_size}")

        # Pass 1: 并行统计航程时间跨度（避免全量内存）
        print("统计航程时间跨度(并行)...")
        summaries = []
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = deque()
                for chunk in pd.read_csv(voyage_path, usecols=use_cols, dtype=dtype, chunksize=chunk_size, memory_map=True):
                    futures.append(executor.submit(_summarize_chunk_for_step3, chunk))
                    if len(futures) >= num_workers * 2:
                        summaries.append(futures.popleft().result())
                while futures:
                    summaries.append(futures.popleft().result())
        else:
            for chunk in pd.read_csv(voyage_path, usecols=use_cols, dtype=dtype, chunksize=chunk_size, memory_map=True):
                summaries.append(_summarize_chunk_for_step3(chunk))

        if len(summaries) == 0:
            print("数据量不足，请先处理更多文件")
            return

        summary_df = pd.concat(summaries).groupby(level=0).agg(
            t_min=('t_min', 'min'),
            t_max=('t_max', 'max'),
            dur=('dur', 'first'),
            rows=('rows', 'sum')
        )
        summary_df['actual_span'] = (summary_df['t_max'] - summary_df['t_min']).dt.total_seconds() / 3600
        summary_df['ratio'] = summary_df['actual_span'] / summary_df['dur'].clip(lower=1)
        valid_time_voyages = summary_df[summary_df['ratio'] <= 1.5].index

        # 采样航程
        rng = np.random.default_rng(42)
        voyage_ids = np.array(valid_time_voyages)
        if len(voyage_ids) > max_voyages:
            selected_voyage_ids = set(rng.choice(voyage_ids, size=max_voyages, replace=False))
        else:
            selected_voyage_ids = set(voyage_ids)

        print(f"候选航程: {len(valid_time_voyages):,}, 采样航程: {len(selected_voyage_ids):,}")

        # Pass 2: 并行过滤并收集训练数据
        print("收集训练数据(并行)...")
        total_rows = 0
        spill_dir = cache_dir / "step3_spill"
        if args.step3_spill:
            spill_dir.mkdir(parents=True, exist_ok=True)
        
        if args.step3_spill:
            bucket_count = max(4, args.step3_bucket_count)
            bucket_rows = max(100_000, args.step3_bucket_rows)
            bucket_buffers = {i: [] for i in range(bucket_count)}
            bucket_counts = {i: 0 for i in range(bucket_count)}
            bucket_parts = {i: 0 for i in range(bucket_count)}

            def flush_bucket(bucket_id: int):
                if bucket_counts[bucket_id] == 0:
                    return
                df_bucket = pd.concat(bucket_buffers[bucket_id], ignore_index=True, copy=False)
                part = bucket_parts[bucket_id]
                file_path = spill_dir / f"bucket_{bucket_id}_part_{part}.pkl"
                df_bucket.to_pickle(file_path)
                bucket_parts[bucket_id] += 1
                bucket_buffers[bucket_id].clear()
                bucket_counts[bucket_id] = 0

            def add_to_buckets(df: pd.DataFrame):
                nonlocal total_rows
                if len(df) == 0:
                    return
                total_rows += len(df)
                # 分桶写入
                bucket_ids = df['voyage_id'].apply(lambda x: hash(x) % bucket_count).values
                for b in range(bucket_count):
                    mask = bucket_ids == b
                    if mask.sum() == 0:
                        continue
                    bucket_buffers[b].append(df.loc[mask])
                    bucket_counts[b] += mask.sum()
                    if bucket_counts[b] >= bucket_rows:
                        flush_bucket(b)

            if num_workers > 1:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = deque()
                    stop_reading = False
                    for chunk in pd.read_csv(voyage_path, usecols=use_cols, dtype=dtype, chunksize=chunk_size, memory_map=True):
                        if stop_reading:
                            break
                        futures.append(executor.submit(_filter_and_select_chunk, chunk, selected_voyage_ids))
                        if len(futures) >= num_workers * 2:
                            result = futures.popleft().result()
                            add_to_buckets(result)
                    while futures:
                        result = futures.popleft().result()
                        add_to_buckets(result)
            else:
                for chunk in pd.read_csv(voyage_path, usecols=use_cols, dtype=dtype, chunksize=chunk_size, memory_map=True):
                    result = _filter_and_select_chunk(chunk, selected_voyage_ids)
                    add_to_buckets(result)

            # flush remaining
            for b in range(bucket_count):
                flush_bucket(b)

            print(f"已写入分桶数据至: {spill_dir}")
        else:
            chunks = []
            if num_workers > 1:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = deque()
                    stop_reading = False
                    for chunk in pd.read_csv(voyage_path, usecols=use_cols, dtype=dtype, chunksize=chunk_size, memory_map=True):
                        if stop_reading:
                            break
                        futures.append(executor.submit(_filter_and_select_chunk, chunk, selected_voyage_ids))
                        if len(futures) >= num_workers * 2:
                            result = futures.popleft().result()
                            if len(result) > 0:
                                chunks.append(result)
                                total_rows += len(result)
                    while futures:
                        result = futures.popleft().result()
                        if len(result) > 0:
                            chunks.append(result)
                            total_rows += len(result)
            else:
                for chunk in pd.read_csv(voyage_path, usecols=use_cols, dtype=dtype, chunksize=chunk_size, memory_map=True):
                    result = _filter_and_select_chunk(chunk, selected_voyage_ids)
                    if len(result) > 0:
                        chunks.append(result)
                        total_rows += len(result)

            training_df = pd.concat(chunks, ignore_index=True, copy=False)
            del chunks
            gc.collect()

            # 降低内存占用
            if 'mmsi' in training_df.columns:
                training_df['mmsi'] = training_df['mmsi'].astype('int32')
            training_df['voyage_id'] = training_df['voyage_id'].astype('category')

            print(f"采样航程数: {training_df['voyage_id'].nunique():,}")
            print(f"过滤后数据: {len(training_df):,} 条")

            if len(training_df) < 1000:
                print("数据量不足，请先处理更多文件")
                return

            dataset = VoyageETADataset(seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len)

            # 按航程划分训练/验证/测试，避免数据泄漏
            voyage_ids = training_df['voyage_id'].unique()
            rng = np.random.default_rng(42)
            rng.shuffle(voyage_ids)
            n_total = len(voyage_ids)
            n_train = int(n_total * 0.7)
            n_val = int(n_total * 0.15)
            train_ids = set(voyage_ids[:n_train])
            val_ids = set(voyage_ids[n_train:n_train + n_val])
            test_ids = set(voyage_ids[n_train + n_val:])

            train_df = training_df[training_df['voyage_id'].isin(train_ids)]
            val_df = training_df[training_df['voyage_id'].isin(val_ids)]
            test_df = training_df[training_df['voyage_id'].isin(test_ids)]

            # 生成序列（只用训练集拟合归一化参数）
            X_train, X_mark_enc_train, X_mark_dec_train, y_train, sd_train = dataset.create_sequences(
                train_df, max_sequences=args.max_sequences, fit=True
            )
            X_val, X_mark_enc_val, X_mark_dec_val, y_val, sd_val = dataset.create_sequences(
                val_df, max_sequences=max(1, args.max_sequences // 10), fit=False
            )
            X_test, X_mark_enc_test, X_mark_dec_test, y_test, sd_test, test_meta = dataset.create_sequences(
                test_df, max_sequences=max(1, args.max_sequences // 10), fit=False, return_meta=True
            )

            # 释放原始数据内存
            del training_df, train_df, val_df, test_df
            gc.collect()

        if args.step3_spill:
            # 使用分桶文件进行流式处理（减少内存峰值）
            bucket_files = sorted(spill_dir.glob("bucket_*_part_*.pkl"))
            if len(bucket_files) == 0:
                print("Spill数据为空，无法继续")
                return

            dataset = VoyageETADataset(seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len)

            # 按航程划分训练/验证/测试，避免数据泄漏
            voyage_ids = np.array(list(selected_voyage_ids))
            rng = np.random.default_rng(42)
            rng.shuffle(voyage_ids)
            n_total = len(voyage_ids)
            n_train = int(n_total * 0.7)
            n_val = int(n_total * 0.15)
            train_ids = set(voyage_ids[:n_train])
            val_ids = set(voyage_ids[n_train:n_train + n_val])
            test_ids = set(voyage_ids[n_train + n_val:])

            # Pass A: 仅用训练集统计归一化参数（流式）
            print("统计归一化参数(流式)...")
            feat_min = None
            feat_max = None
            count, mean, m2 = 0, 0.0, 0.0
            # 6个特征
            feature_cols = ['lat', 'lon', 'sog', 'cog', 'dist_to_dest_km', 'bearing_diff']
            for bf in bucket_files:
                df_bucket = pd.read_pickle(bf)
                df_bucket = df_bucket[df_bucket['voyage_id'].isin(train_ids)]
                if len(df_bucket) == 0:
                    continue
                for _, group in df_bucket.groupby('voyage_id'):
                    group = _compute_geom_features(group)
                    feat_vals = group[feature_cols].values.astype(np.float32)
                    feat_min, feat_max = _update_minmax(feat_min, feat_max, feat_vals)
                    target_vals = np.log1p(group['remaining_hours'].values.astype(np.float32))
                    count, mean, m2 = _update_welford(count, mean, m2, target_vals)

            dataset.feature_min = feat_min
            dataset.feature_max = feat_max
            dataset.target_mean, dataset.target_std = _finalize_welford(count, mean, m2)

            # Pass B: 统计每个split可用的总序列数
            print("统计各split序列数量...")
            train_seq_count = 0
            val_seq_count = 0
            test_seq_count = 0
            
            for bf in bucket_files:
                df_bucket = pd.read_pickle(bf)
                if len(df_bucket) == 0:
                    continue
                # 统计训练集
                train_df = df_bucket[df_bucket['voyage_id'].isin(train_ids)]
                for vid, group in train_df.groupby('voyage_id'):
                    train_seq_count += max(0, len(group) - args.seq_len - args.pred_len + 1)
                # 统计验证集
                val_df = df_bucket[df_bucket['voyage_id'].isin(val_ids)]
                for vid, group in val_df.groupby('voyage_id'):
                    val_seq_count += max(0, len(group) - args.seq_len - args.pred_len + 1)
                # 统计测试集
                test_df = df_bucket[df_bucket['voyage_id'].isin(test_ids)]
                for vid, group in test_df.groupby('voyage_id'):
                    test_seq_count += max(0, len(group) - args.seq_len - args.pred_len + 1)
                del df_bucket, train_df, val_df, test_df
                gc.collect()
            
            # 限制最大序列数
            train_seq_count = min(train_seq_count, args.max_sequences)
            val_seq_count = min(val_seq_count, args.max_sequences // 10)
            test_seq_count = min(test_seq_count, args.max_sequences // 10)
            
            print(f"  预估序列: train={train_seq_count:,}, val={val_seq_count:,}, test={test_seq_count:,}")
            
            # 创建memmap数组
            print("创建memmap数组...")
            if args.use_memmap:
                X_train_mm, X_mark_enc_train_mm, X_dec_train_mm, X_mark_dec_train_mm, y_train_mm, sd_train_mm = create_memmap_arrays(
                    cache_dir, train_seq_count, args.seq_len, args.label_len, args.pred_len, prefix='train'
                )
                X_val_mm, X_mark_enc_val_mm, X_dec_val_mm, X_mark_dec_val_mm, y_val_mm, sd_val_mm = create_memmap_arrays(
                    cache_dir, val_seq_count, args.seq_len, args.label_len, args.pred_len, prefix='val'
                )
                X_test_mm, X_mark_enc_test_mm, X_dec_test_mm, X_mark_dec_test_mm, y_test_mm, sd_test_mm = create_memmap_arrays(
                    cache_dir, test_seq_count, args.seq_len, args.label_len, args.pred_len, prefix='test'
                )
            
            # Pass C: 生成序列并增量写入memmap
            print("生成序列(memmap增量写入)...")
            train_idx = 0
            val_idx = 0
            test_idx = 0
            meta_list = []
            
            # 使用参数控制每个bucket处理的最大序列数，避免单次分配过大内存
            max_seqs_per_bucket = args.max_seqs_per_bucket
            
            for bf_idx, bf in enumerate(bucket_files):
                df_bucket = pd.read_pickle(bf)
                if len(df_bucket) == 0:
                    continue
                df_bucket['mmsi'] = df_bucket['mmsi'].astype('int32')
                
                # 处理训练集
                train_df = df_bucket[df_bucket['voyage_id'].isin(train_ids)]
                if len(train_df) > 0 and train_idx < train_seq_count:
                    # 限制单次调用的max_sequences，避免预分配过大数组
                    bucket_max = min(max_seqs_per_bucket, train_seq_count - train_idx)
                    Xt, Xme, Xmd, yt, sdt = dataset.create_sequences(
                        train_df, max_sequences=bucket_max, fit=False
                    )
                    n_new = len(Xt)
                    if train_idx + n_new > train_seq_count:
                        n_new = train_seq_count - train_idx
                        Xt, Xme, Xmd, yt, sdt = Xt[:n_new], Xme[:n_new], Xmd[:n_new], yt[:n_new], sdt[:n_new]
                    
                    if args.use_memmap:
                        X_train_mm[train_idx:train_idx+n_new] = Xt
                        X_mark_enc_train_mm[train_idx:train_idx+n_new] = Xme
                        X_dec_train_mm[train_idx:train_idx+n_new] = build_decoder_input(Xt, args.label_len, args.pred_len)
                        X_mark_dec_train_mm[train_idx:train_idx+n_new] = Xmd
                        y_train_mm[train_idx:train_idx+n_new] = yt
                        sd_train_mm[train_idx:train_idx+n_new] = sdt
                    train_idx += n_new
                    del Xt, Xme, Xmd, yt, sdt
                
                # 处理验证集
                val_df = df_bucket[df_bucket['voyage_id'].isin(val_ids)]
                if len(val_df) > 0 and val_idx < val_seq_count:
                    bucket_max_val = min(max_seqs_per_bucket, val_seq_count - val_idx)
                    Xv, Xmev, Xmdv, yv, sdv = dataset.create_sequences(
                        val_df, max_sequences=bucket_max_val, fit=False
                    )
                    n_new = len(Xv)
                    if val_idx + n_new > val_seq_count:
                        n_new = val_seq_count - val_idx
                        Xv, Xmev, Xmdv, yv, sdv = Xv[:n_new], Xmev[:n_new], Xmdv[:n_new], yv[:n_new], sdv[:n_new]
                    
                    if args.use_memmap:
                        X_val_mm[val_idx:val_idx+n_new] = Xv
                        X_mark_enc_val_mm[val_idx:val_idx+n_new] = Xmev
                        X_dec_val_mm[val_idx:val_idx+n_new] = build_decoder_input(Xv, args.label_len, args.pred_len)
                        X_mark_dec_val_mm[val_idx:val_idx+n_new] = Xmdv
                        y_val_mm[val_idx:val_idx+n_new] = yv
                        sd_val_mm[val_idx:val_idx+n_new] = sdv
                    val_idx += n_new
                    del Xv, Xmev, Xmdv, yv, sdv
                
                # 处理测试集
                test_df = df_bucket[df_bucket['voyage_id'].isin(test_ids)]
                if len(test_df) > 0 and test_idx < test_seq_count:
                    bucket_max_test = min(max_seqs_per_bucket, test_seq_count - test_idx)
                    Xte, Xmete, Xmdte, yte, sdte, meta = dataset.create_sequences(
                        test_df, max_sequences=bucket_max_test, fit=False, return_meta=True
                    )
                    n_new = len(Xte)
                    if test_idx + n_new > test_seq_count:
                        n_new = test_seq_count - test_idx
                        Xte, Xmete, Xmdte, yte, sdte = Xte[:n_new], Xmete[:n_new], Xmdte[:n_new], yte[:n_new], sdte[:n_new]
                        meta = {k: v[:n_new] for k, v in meta.items()}
                    
                    if args.use_memmap:
                        X_test_mm[test_idx:test_idx+n_new] = Xte
                        X_mark_enc_test_mm[test_idx:test_idx+n_new] = Xmete
                        X_dec_test_mm[test_idx:test_idx+n_new] = build_decoder_input(Xte, args.label_len, args.pred_len)
                        X_mark_dec_test_mm[test_idx:test_idx+n_new] = Xmdte
                        y_test_mm[test_idx:test_idx+n_new] = yte
                        sd_test_mm[test_idx:test_idx+n_new] = sdte
                    meta_list.append(meta)
                    test_idx += n_new
                    del Xte, Xmete, Xmdte, yte, sdte
                
                del df_bucket
                gc.collect()
                
                if (bf_idx + 1) % 8 == 0:
                    print(f"  处理进度: {bf_idx+1}/{len(bucket_files)} buckets, train={train_idx:,}, val={val_idx:,}, test={test_idx:,}")
            
            # Flush memmap to disk
            if args.use_memmap:
                del X_train_mm, X_mark_enc_train_mm, X_dec_train_mm, X_mark_dec_train_mm, y_train_mm, sd_train_mm
                del X_val_mm, X_mark_enc_val_mm, X_dec_val_mm, X_mark_dec_val_mm, y_val_mm, sd_val_mm
                del X_test_mm, X_mark_enc_test_mm, X_dec_test_mm, X_mark_dec_test_mm, y_test_mm, sd_test_mm
                gc.collect()
            
            # 合并meta
            if meta_list:
                test_meta = {
                    'mmsi': np.concatenate([m['mmsi'] for m in meta_list]),
                    'voyage_id': np.concatenate([m['voyage_id'] for m in meta_list]),
                    'pred_time': np.concatenate([m['pred_time'] for m in meta_list]),
                    'end_time': np.concatenate([m['end_time'] for m in meta_list])
                }
            else:
                test_meta = {'mmsi': np.array([]), 'voyage_id': np.array([]), 'pred_time': np.array([]), 'end_time': np.array([])}
            
            print(f"实际序列: train={train_idx:,}, val={val_idx:,}, test={test_idx:,}")
            
            # 保存归一化参数和meta
            dataset.save_params(os.path.join(args.output_dir, 'norm_params.npz'))
            np.save(cache_dir / "test_meta.npy", test_meta, allow_pickle=True)
            # 保存实际写入的序列数
            np.save(cache_dir / "actual_counts.npy", {'train': train_idx, 'val': val_idx, 'test': test_idx}, allow_pickle=True)
            
            # 使用memmap模式加载数据创建DataLoader
            print("创建DataLoader (memmap模式)...")
            train_dataset = MemmapDataset(
                cache_dir / "X_train.npy", cache_dir / "X_mark_enc_train.npy",
                cache_dir / "X_dec_train.npy", cache_dir / "X_mark_dec_train.npy",
                cache_dir / "y_train.npy", cache_dir / "sd_train.npy"
            )
            val_dataset = MemmapDataset(
                cache_dir / "X_val.npy", cache_dir / "X_mark_enc_val.npy",
                cache_dir / "X_dec_val.npy", cache_dir / "X_mark_dec_val.npy",
                cache_dir / "y_val.npy", cache_dir / "sd_val.npy"
            )
            test_dataset = MemmapDataset(
                cache_dir / "X_test.npy", cache_dir / "X_mark_enc_test.npy",
                cache_dir / "X_dec_test.npy", cache_dir / "X_mark_dec_test.npy",
                cache_dir / "y_test.npy", cache_dir / "sd_test.npy"
            )
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
            
            # 为评估阶段准备（需要完整加载test数据）
            X_test = np.load(cache_dir / "X_test.npy", mmap_mode='r')
            y_test = np.load(cache_dir / "y_test.npy", mmap_mode='r')
            
            print(f"训练集: {len(train_dataset):,}, 验证集: {len(val_dataset):,}, 测试集: {len(test_dataset):,}")
            
            # spill+memmap模式已经完成DataLoader创建，跳转到Step 4
            use_memmap_mode = True

    # 如果不是memmap模式，使用传统方式创建DataLoader
    if not use_memmap_mode:
        print(f"训练序列: {len(X_train):,}, 验证序列: {len(X_val):,}, 测试序列: {len(X_test):,}")
        print(f"训练形状: X={X_train.shape}, y={y_train.shape}")

        # 保存归一化参数
        dataset.save_params(os.path.join(args.output_dir, 'norm_params.npz'))

        # 保存缓存
        np.save(cache_dir / "X_train.npy", X_train)
        np.save(cache_dir / "X_mark_enc_train.npy", X_mark_enc_train)
        np.save(cache_dir / "X_mark_dec_train.npy", X_mark_dec_train)
        np.save(cache_dir / "y_train.npy", y_train)
        np.save(cache_dir / "sd_train.npy", sd_train)
        np.save(cache_dir / "X_val.npy", X_val)
        np.save(cache_dir / "X_mark_enc_val.npy", X_mark_enc_val)
        np.save(cache_dir / "X_mark_dec_val.npy", X_mark_dec_val)
        np.save(cache_dir / "y_val.npy", y_val)
        np.save(cache_dir / "sd_val.npy", sd_val)
        np.save(cache_dir / "X_test.npy", X_test)
        np.save(cache_dir / "X_mark_enc_test.npy", X_mark_enc_test)
        np.save(cache_dir / "X_mark_dec_test.npy", X_mark_dec_test)
        np.save(cache_dir / "y_test.npy", y_test)
        np.save(cache_dir / "sd_test.npy", sd_test)
        np.save(cache_dir / "test_meta.npy", test_meta, allow_pickle=True)

        X_dec_train = build_decoder_input(X_train, args.label_len, args.pred_len)
        X_dec_val = build_decoder_input(X_val, args.label_len, args.pred_len)
        X_dec_test = build_decoder_input(X_test, args.label_len, args.pred_len)

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(X_mark_enc_train),
            torch.FloatTensor(X_dec_train),
            torch.FloatTensor(X_mark_dec_train),
            torch.FloatTensor(y_train),
            torch.FloatTensor(sd_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(X_mark_enc_val),
            torch.FloatTensor(X_dec_val),
            torch.FloatTensor(X_mark_dec_val),
            torch.FloatTensor(y_val),
            torch.FloatTensor(sd_val)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(X_mark_enc_test),
            torch.FloatTensor(X_dec_test),
            torch.FloatTensor(X_mark_dec_test),
            torch.FloatTensor(y_test),
            torch.FloatTensor(sd_test)
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        print(f"训练集: {len(train_loader.dataset):,}, 验证集: {len(val_loader.dataset):,}, 测试集: {len(test_loader.dataset):,}")
    
    # ===== Step 4: 训练Informer =====
    print("\n" + "="*60)
    print("Step 4: 训练Informer模型")
    print("="*60)
    
    # 6个特征: lat, lon, sog, cog, dist_to_dest_km, bearing_diff
    n_features = 6
    model = Informer(
        enc_in=n_features, dec_in=n_features, c_out=1,
        seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
        d_model=args.d_model, n_heads=args.n_heads,
        e_layers=args.e_layers, d_layers=args.d_layers,
        d_ff=args.d_ff, dropout=args.dropout,
        attn='prob', embed='timeF', activation='gelu',
        output_attention=False, distil=True, device=device
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 初始化训练器
    steps_per_epoch = len(train_loader) if args.scheduler == 'onecycle' else None
    trainer = InformerTrainer(
        model, device, lr=args.lr, 
        scheduler_type=args.scheduler, 
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch
    )
    
    model_path = os.path.join(args.output_dir, 'best_informer.pth')
    checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pth')
    
    # 如果是eval_only模式，直接加载模型跳过训练
    if args.eval_only:
        if os.path.exists(model_path):
            print(f"加载已有模型: {model_path}")
            trainer.load(model_path)
        else:
            print(f"错误: 模型文件不存在 {model_path}")
            return
    else:
        start_epoch = 0
        best_val_loss = float('inf')
        
        # 从checkpoint继续训练
        if args.resume and os.path.exists(checkpoint_path):
            print(f"从checkpoint继续训练: {checkpoint_path}")
            start_epoch = trainer.load_checkpoint(checkpoint_path)
            best_val_loss = trainer.best_val_loss
        elif args.resume:
            print("未找到checkpoint，从头开始训练")
        
        print(f"\n使用学习率调度器: {args.scheduler}")
        print(f"初始学习率: {args.lr:.2e}")
        
        no_improve_count = 0
        for epoch in range(start_epoch, args.epochs):
            train_loss = trainer.train_epoch(train_loader)
            val_loss = trainer.validate(val_loader)
            trainer.step_scheduler(val_loss)
            
            lr = trainer.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{args.epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}, LR={lr:.2e}")
            
            # 保存检查点（用于继续训练）
            trainer.save_checkpoint(checkpoint_path, epoch, min(val_loss, best_val_loss))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trainer.save(model_path)
                print("  -> 保存最佳模型!")
                no_improve_count = 0
            else:
                no_improve_count += 1
                print(f"  验证损失未改善 ({no_improve_count}/{args.early_stopping_patience})")
                if args.early_stopping_patience > 0 and no_improve_count >= args.early_stopping_patience:
                    print(f"\n早停: 验证损失连续 {args.early_stopping_patience} 个 epoch 未改善，停止训练。")
                    break
        
        # 训练后加载最佳模型
        trainer.load(model_path)
    
    # ===== Step 5: 评估 =====
    print("\n" + "="*60)
    print("Step 5: 评估")
    print("="*60)
    
    y_pred_norm, y_true_norm, pred_sd = trainer.predict(test_loader)
    
    y_pred_sailing = dataset.inverse_normalize_target(y_pred_norm)
    y_true = dataset.inverse_normalize_target(y_true_norm)
    y_pred_sailing = np.maximum(y_pred_sailing, 0)
    
    print(f"【Informer 航行时间预测】")
    print(f"  预测范围: [{y_pred_sailing.min():.2f}, {y_pred_sailing.max():.2f}] 小时")
    print(f"  实际范围: [{y_true.min():.2f}, {y_true.max():.2f}] 小时")
    
    # 单模型指标
    metrics_sailing = calculate_metrics(y_pred_sailing, y_true)
    print("\n【单模型指标（仅航行时间）】")
    for name, value in metrics_sailing.items():
        print(f"  {name}: {value:.4f}")
    
    # ===== 双模型评估 =====
    # 加载港口停靠模型并预测停靠时间
    port_model_dir = os.path.join(args.output_dir, 'port_model')
    port_model_exists = os.path.exists(os.path.join(port_model_dir, 'model.pth'))
    
    # 先绘制单模型结果
    plot_results(y_pred_sailing, y_true, pred_sd, args.output_dir, 
                 suffix='_sailing', title_prefix='单模型(仅航行) ')
    
    # 分析单模型 Bad Cases (仅在eval时)
    analyze_bad_cases(y_pred_sailing, y_true, X_test, test_meta, dataset, args.output_dir, suffix='_sailing')
    
    if port_model_exists and os.path.exists(stop_path):
        print("\n【双模型评估】")
        print("加载港口停靠时间模型...")
        
        port_model = PortStopModel(port_model_dir)
        port_model.load()
        
        # 读取停靠数据
        stop_df = pd.read_csv(stop_path)
        if len(stop_df) == 0:
            print("停靠数据为空，跳过双模型评估")
            y_pred = y_pred_sailing
            metrics = metrics_sailing
        else:
            stop_df['arrival_time'] = pd.to_datetime(stop_df['arrival_time'])
            stop_df['mmsi'] = stop_df['mmsi'].astype('int64')
            
            # 取每个航程的mmsi和结束时间
            voyage_meta_df = pd.DataFrame({
                'voyage_id': test_meta['voyage_id'],
                'mmsi': test_meta['mmsi'],
                'end_time': test_meta['end_time']
            }).drop_duplicates('voyage_id')
            voyage_meta_df['end_time'] = pd.to_datetime(voyage_meta_df['end_time'])
            voyage_meta_df['mmsi'] = voyage_meta_df['mmsi'].astype('int64')
            
            # 为每个航程匹配下一次停靠（使用简单的向量化方法）
            print("  匹配航程与停靠记录...")
            
            # 构建每个mmsi的停靠记录索引
            stop_by_mmsi = stop_df.groupby('mmsi')
            
            matched_records = []
            tolerance_hours = 48
            
            for _, row in voyage_meta_df.iterrows():
                mmsi = row['mmsi']
                end_time = row['end_time']
                voyage_id = row['voyage_id']
                
                record = {'voyage_id': voyage_id, 'mmsi': mmsi, 'end_time': end_time}
                
                if mmsi in stop_by_mmsi.groups:
                    mmsi_stops = stop_by_mmsi.get_group(mmsi)
                    # 找航程结束后48小时内最近的停靠
                    future_stops = mmsi_stops[
                        (mmsi_stops['arrival_time'] >= end_time) & 
                        (mmsi_stops['arrival_time'] <= end_time + pd.Timedelta(hours=tolerance_hours))
                    ]
                    if len(future_stops) > 0:
                        nearest = future_stops.loc[future_stops['arrival_time'].idxmin()]
                        record['region'] = nearest['region']
                        record['duration_hours'] = nearest['duration_hours']
                        record['arrival_time'] = nearest['arrival_time']
                        record['lat'] = nearest.get('lat', nearest.get('port_lat', np.nan))
                        record['lon'] = nearest.get('lon', nearest.get('port_lon', np.nan))
                    else:
                        record['region'] = np.nan
                        record['duration_hours'] = np.nan
                        record['arrival_time'] = pd.NaT
                        record['lat'] = np.nan
                        record['lon'] = np.nan
                else:
                    record['region'] = np.nan
                    record['duration_hours'] = np.nan
                    record['arrival_time'] = pd.NaT
                    record['lat'] = np.nan
                    record['lon'] = np.nan
                
                matched_records.append(record)
            
            matched = pd.DataFrame(matched_records)
            
            matched_valid = matched[~matched['region'].isna()].copy()
            print(f"  匹配到停靠记录: {len(matched_valid):,} / {len(voyage_meta_df):,} 航程")
            
            if len(matched_valid) == 0:
                print("未匹配到任何停靠记录，跳过双模型评估")
                y_pred = y_pred_sailing
                metrics = metrics_sailing
            else:
                # 预测停靠时长
                stop_pred_hours = port_model.predict(matched_valid)
                matched_valid['pred_stop_hours'] = stop_pred_hours
                
                stop_pred_map = dict(zip(matched_valid['voyage_id'], matched_valid['pred_stop_hours']))
                y_pred_stop = np.array([stop_pred_map.get(vid, 0.0) for vid in test_meta['voyage_id']])
                
                # 最终ETA = 航行时间 + 停靠时间
                y_pred_total = y_pred_sailing + y_pred_stop
                
                print(f"\n  预测停靠时间: 平均 {y_pred_stop.mean():.1f}h")
                print(f"  最终ETA范围: [{y_pred_total.min():.2f}, {y_pred_total.max():.2f}] 小时")
                
                metrics_total = calculate_metrics(y_pred_total, y_true)
                print("\n【双模型指标（航行+停靠）】")
                for name, value in metrics_total.items():
                    print(f"  {name}: {value:.4f}")
                
                # 绘制双模型结果
                plot_results(y_pred_total, y_true, pred_sd, args.output_dir,
                             suffix='_total', title_prefix='双模型(航行+停靠) ')
                
                # 分析双模型 Bad Cases
                analyze_bad_cases(y_pred_total, y_true, X_test, test_meta, dataset, args.output_dir, suffix='_total')
                
                y_pred = y_pred_total
                metrics = metrics_total
    else:
        print("\n港口模型不存在，跳过双模型评估")
        print("可使用 --train_port_model 训练港口模型")
        y_pred = y_pred_sailing
        metrics = metrics_sailing
    
    # 保存结果
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"训练时间: {datetime.now().isoformat()}\n\n")
        f.write("配置:\n")
        for k, v in vars(args).items():
            f.write(f"  {k}: {v}\n")
        f.write("\n航程ETA指标:\n")
        for name, value in metrics.items():
            f.write(f"  {name}: {value:.4f}\n")
    
    print(f"\n训练完成! 结果保存至 {args.output_dir}")
    print("\n【最终ETA计算方式】")
    print("  最终ETA = Informer预测的航行时间 + 港口停靠时间模型预测的停靠时间")

    # Discord webhook 通知
    try:
        import urllib.request, json
        webhook_url = "https://discord.com/api/webhooks/1477180434474336266/HQSS_BKlo1Ib-rzwItazg8ay0To2l-GR1GprnTvlPVpLxCxD9xu6_iEPsD9aBd4iHNgX"
        msg_lines = [f"**ETA训练完成** ✅", f"时间: {datetime.now().isoformat()}"]
        for name, value in metrics.items():
            msg_lines.append(f"  {name}: {value:.4f}")
        payload = json.dumps({"content": "\n".join(msg_lines)}).encode('utf-8')
        req = urllib.request.Request(webhook_url, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        print("Discord 通知已发送")
    except Exception as e:
        print(f"Discord 通知发送失败: {e}")


if __name__ == '__main__':
    main()
