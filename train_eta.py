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

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.informer.model import Informer


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
    
    def normalize_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """Minmax归一化"""
        if fit:
            self.feature_min = features.min(axis=0)
            self.feature_max = features.max(axis=0)
        
        range_vals = self.feature_max - self.feature_min
        range_vals[range_vals == 0] = 1.0
        return (features - self.feature_min) / range_vals
    
    def normalize_target(self, target: np.ndarray, fit: bool = False) -> np.ndarray:
        """Log + 标准化"""
        log_target = np.log1p(target)
        if fit:
            self.target_mean = log_target.mean()
            self.target_std = log_target.std()
        return (log_target - self.target_mean) / self.target_std
    
    def inverse_normalize_target(self, normalized: np.ndarray) -> np.ndarray:
        """反归一化"""
        log_target = normalized * self.target_std + self.target_mean
        return np.expm1(log_target)
    
    def create_sequences(self, voyage_df: pd.DataFrame, max_sequences: int = 500000) -> tuple:
        """从航程数据创建序列（内存优化版 + 分层采样）
        
        优化策略：
        1. 先统计总序列数，预分配数组
        2. 使用float32减少内存
        3. 分层采样：确保高remaining_hours样本被充分采样
        """
        feature_cols = ['lat', 'lon', 'sog', 'cog']
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
        X = np.zeros((actual_total, self.seq_len, 4), dtype=np.float32)
        X_mark_enc = np.zeros((actual_total, self.seq_len, 5), dtype=np.float32)
        X_mark_dec = np.zeros((actual_total, self.label_len + self.pred_len, 5), dtype=np.float32)
        y = np.zeros(actual_total, dtype=np.float32)
        sailing_days = np.zeros(actual_total, dtype=np.float32)
        
        # 第二遍：填充数据
        print("  生成序列...")
        idx = 0
        for i, (vid, group) in enumerate(voyage_df.groupby(group_col)):
            if vid not in voyage_ids:
                continue
            
            vid_idx = voyage_ids.index(vid)
            n_to_sample = samples_per_voyage[vid_idx]
            
            group = group.sort_values('postime')
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
                idx += 1
            
            if idx >= actual_total:
                break
        
        # 截断到实际使用的大小
        X = X[:idx]
        X_mark_enc = X_mark_enc[:idx]
        X_mark_dec = X_mark_dec[:idx]
        y = y[:idx]
        sailing_days = sailing_days[:idx]
        
        print(f"  最终序列数: {idx:,}")
        
        # 归一化
        X_flat = X.reshape(-1, X.shape[-1])
        X_norm = self.normalize_features(X_flat, fit=True).reshape(X.shape).astype(np.float32)
        y_norm = self.normalize_target(y, fit=True).astype(np.float32)
        
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
    
    def __init__(self, model, device, lr=5e-6):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.HuberLoss(delta=1.0)
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)
    
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
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


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


def plot_results(y_pred, y_true, sailing_days, save_dir):
    """绘制结果图表"""
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
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'analysis_plots.png'), dpi=150)
    plt.close()
    print(f"Plots saved to {save_dir}")


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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-6)
    
    # 港口模型
    parser.add_argument('--train_port_model', action='store_true', help='训练港口停靠时间模型')
    parser.add_argument('--port_epochs', type=int, default=100)
    
    # 其他
    parser.add_argument('--use_cache', action='store_true', help='使用缓存的预处理数据')
    parser.add_argument('--max_files', type=int, default=None, help='处理文件数限制')
    parser.add_argument('--max_voyages', type=int, default=5000, help='最多使用的航程数（控制内存）')
    parser.add_argument('--max_sequences', type=int, default=500000, help='最多生成的训练序列数')
    parser.add_argument('--eval_only', action='store_true', help='仅评估和重绘图（跳过训练）')
    
    args = parser.parse_args()
    
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
    
    # 分块读取CSV，只加载需要的列，使用float32
    print("分块读取航程数据...")
    use_cols = ['lat', 'lon', 'sog', 'cog', 'remaining_hours', 'mmsi', 'postime', 'voyage_id', 'voyage_duration_hours']
    dtype = {'lat': 'float32', 'lon': 'float32', 'sog': 'float32', 'cog': 'float32', 
             'remaining_hours': 'float32', 'voyage_duration_hours': 'float32'}
    
    chunks = []
    voyage_ids_seen = set()
    total_rows = 0
    
    for chunk in pd.read_csv(voyage_path, usecols=use_cols, dtype=dtype, chunksize=500000):
        # 数据过滤
        chunk = chunk.dropna()
        chunk = chunk[(chunk['remaining_hours'] >= 0) & (chunk['remaining_hours'] <= 720)]
        chunk = chunk[(chunk['sog'] >= 0) & (chunk['sog'] <= 30)]
        
        # 过滤超长航程（只保留航程时长 <= 30天 = 720小时的航程）
        if 'voyage_duration_hours' in chunk.columns:
            valid_voyages = chunk.groupby('voyage_id')['voyage_duration_hours'].first()
            valid_voyages = valid_voyages[valid_voyages <= 720].index
            chunk = chunk[chunk['voyage_id'].isin(valid_voyages)]
        
        # 过滤时间跨度异常的航程（实际时间跨度不应超过标记时长的1.5倍）
        chunk['postime_dt'] = pd.to_datetime(chunk['postime'])
        voyage_time_check = chunk.groupby('voyage_id').agg({
            'postime_dt': ['min', 'max'],
            'voyage_duration_hours': 'first'
        })
        voyage_time_check.columns = ['t_min', 't_max', 'dur']
        voyage_time_check['actual_span'] = (voyage_time_check['t_max'] - voyage_time_check['t_min']).dt.total_seconds() / 3600
        voyage_time_check['ratio'] = voyage_time_check['actual_span'] / voyage_time_check['dur'].clip(lower=1)
        valid_time_voyages = voyage_time_check[voyage_time_check['ratio'] <= 1.5].index
        chunk = chunk[chunk['voyage_id'].isin(valid_time_voyages)]
        chunk = chunk.drop(columns=['postime_dt'])
        
        # 检查新航程
        new_voyage_ids = set(chunk['voyage_id'].unique()) - voyage_ids_seen
        
        if len(voyage_ids_seen) >= max_voyages:
            # 只保留已见过的航程数据
            chunk = chunk[chunk['voyage_id'].isin(voyage_ids_seen)]
        else:
            # 添加新航程直到达到上限
            remaining = max_voyages - len(voyage_ids_seen)
            new_to_add = list(new_voyage_ids)[:remaining]
            voyage_ids_seen.update(new_to_add)
            chunk = chunk[chunk['voyage_id'].isin(voyage_ids_seen)]
        
        if len(chunk) > 0:
            chunks.append(chunk)
            total_rows += len(chunk)
        
        if len(voyage_ids_seen) >= max_voyages and total_rows > 5000000:
            break  # 足够数据了
    
    training_df = pd.concat(chunks, ignore_index=True)
    del chunks
    import gc
    gc.collect()
    
    # 全局过滤：移除时间跨度异常的航程（跨多个chunk的情况）
    print("过滤时间跨度异常的航程...")
    training_df['postime_dt'] = pd.to_datetime(training_df['postime'])
    voyage_time_check = training_df.groupby('voyage_id').agg({
        'postime_dt': ['min', 'max'],
        'voyage_duration_hours': 'first'
    })
    voyage_time_check.columns = ['t_min', 't_max', 'dur']
    voyage_time_check['actual_span'] = (voyage_time_check['t_max'] - voyage_time_check['t_min']).dt.total_seconds() / 3600
    voyage_time_check['ratio'] = voyage_time_check['actual_span'] / voyage_time_check['dur'].clip(lower=1)
    
    bad_voyages = voyage_time_check[voyage_time_check['ratio'] > 1.5].index
    print(f"  移除 {len(bad_voyages)} 个时间跨度异常的航程")
    training_df = training_df[~training_df['voyage_id'].isin(bad_voyages)]
    training_df = training_df.drop(columns=['postime_dt'])
    gc.collect()
    
    print(f"采样航程数: {training_df['voyage_id'].nunique():,}")
    print(f"过滤后数据: {len(training_df):,} 条")
    
    if len(training_df) < 1000:
        print("数据量不足，请先处理更多文件")
        return
    
    dataset = VoyageETADataset(seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len)
    X, X_mark_enc, X_mark_dec, y, sailing_days = dataset.create_sequences(training_df, max_sequences=args.max_sequences)
    
    # 释放原始数据内存
    del training_df
    gc.collect()
    
    print(f"序列数量: {len(X):,}")
    print(f"序列形状: X={X.shape}, y={y.shape}")
    
    # 保存归一化参数
    dataset.save_params(os.path.join(args.output_dir, 'norm_params.npz'))
    
    train_loader, val_loader, test_loader, test_sd, test_y, test_idx = create_data_loaders(
        X, X_mark_enc, X_mark_dec, y, sailing_days,
        label_len=args.label_len, pred_len=args.pred_len, batch_size=args.batch_size
    )
    
    print(f"训练集: {len(train_loader.dataset):,}, 验证集: {len(val_loader.dataset):,}, 测试集: {len(test_loader.dataset):,}")
    
    # ===== Step 4: 训练Informer =====
    print("\n" + "="*60)
    print("Step 4: 训练Informer模型")
    print("="*60)
    
    model = Informer(
        enc_in=4, dec_in=4, c_out=1,
        seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
        d_model=args.d_model, n_heads=args.n_heads,
        e_layers=args.e_layers, d_layers=args.d_layers,
        d_ff=args.d_ff, dropout=args.dropout,
        attn='prob', embed='timeF', activation='gelu',
        output_attention=False, distil=True, device=device
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = InformerTrainer(model, device, lr=args.lr)
    
    # 如果是eval_only模式，直接加载模型跳过训练
    if args.eval_only:
        model_path = os.path.join(args.output_dir, 'best_informer.pth')
        if os.path.exists(model_path):
            print(f"加载已有模型: {model_path}")
            trainer.load(model_path)
        else:
            print(f"错误: 模型文件不存在 {model_path}")
            return
    else:
        best_val_loss = float('inf')
        for epoch in range(args.epochs):
            train_loss = trainer.train_epoch(train_loader)
            val_loss = trainer.validate(val_loader)
            trainer.scheduler.step(val_loss)
            
            lr = trainer.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{args.epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}, LR={lr:.2e}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trainer.save(os.path.join(args.output_dir, 'best_informer.pth'))
                print("  -> 保存最佳模型!")
        
        # 训练后加载最佳模型
        trainer.load(os.path.join(args.output_dir, 'best_informer.pth'))
    
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
    
    if port_model_exists and os.path.exists(stop_path):
        print("\n【双模型评估】")
        print("加载港口停靠时间模型...")
        
        port_model = PortStopModel(port_model_dir)
        port_model.load()
        
        # 读取停靠数据，计算各区域平均停靠时间
        stop_df = pd.read_csv(stop_path)
        region_avg_stop = stop_df.groupby('region')['duration_hours'].mean().to_dict()
        overall_avg_stop = stop_df['duration_hours'].mean()
        
        print(f"  各区域平均停靠时间:")
        for region, avg in region_avg_stop.items():
            print(f"    {region}: {avg:.1f}h")
        
        # 计算预测的停靠时间
        # 简化假设：每个航程结束后有一次停靠，停靠地点用目的地（美国西海岸）作为默认
        # 实际场景中，船舶知道会途径哪些港口
        default_region = '美国西海岸'  # 数据集主要是中国-美西航线
        avg_stop_hours = region_avg_stop.get(default_region, overall_avg_stop)
        
        # 对于每个预测，根据剩余航行时间估计还有多少次停靠
        # 简化：假设每次跨洋航行（>100小时）后有1次停靠
        num_stops = np.where(y_pred_sailing > 100, 1, 0)
        y_pred_stop = num_stops * avg_stop_hours
        
        # 最终ETA = 航行时间 + 停靠时间
        y_pred_total = y_pred_sailing + y_pred_stop
        
        print(f"\n  预测停靠时间: 平均 {y_pred_stop.mean():.1f}h")
        print(f"  最终ETA范围: [{y_pred_total.min():.2f}, {y_pred_total.max():.2f}] 小时")
        
        metrics_total = calculate_metrics(y_pred_total, y_true)
        print("\n【双模型指标（航行+停靠）】")
        for name, value in metrics_total.items():
            print(f"  {name}: {value:.4f}")
        
        # 使用双模型结果绘图
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
    
    plot_results(y_pred, y_true, pred_sd, args.output_dir)
    
    print(f"\n训练完成! 结果保存至 {args.output_dir}")
    print("\n【最终ETA计算方式】")
    print("  最终ETA = Informer预测的航行时间 + 港口停靠时间模型预测的停靠时间")


if __name__ == '__main__':
    main()
