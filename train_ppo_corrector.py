#!/usr/bin/env python
"""
PPO 残差修正器 — 在冻结的 Informer 基模型上训练一个小型 Actor-Critic 网络，
用 PPO 算法学习残差修正量，以进一步降低 MAE。

架构设计：
  1. 冻结 Informer 模型生成基础预测 ŷ_base（小时）
  2. 小型 MLP (Actor-Critic) 读取上下文状态 s，输出修正量 δ
  3. 最终预测 = ŷ_base + δ
  4. 奖励 = |error_base| − |error_corrected|（正=修正改善了预测）

状态空间 s (per sample):
  - ŷ_base_hours (1): Informer 基础预测值（小时）
  - last_step_features (6): 编码器最后一步的 [lat, lon, sog, cog, dist_to_dest, bearing_diff]
  - seq_mean_features (6): 整条序列的特征均值
  - sailing_days (1): 已航行天数
  → 总计 14 维

动作空间 (continuous):
  - δ ∈ [-max_correction, +max_correction]（小时），建模为 Gaussian(μ, σ)

奖励:
  - improvement = |base_error| − |corrected_error|
  - 加小幅正则化 penalty 防止修正量过大

用法:
    python train_ppo_corrector.py --epochs 30 --max_correction 50
    python train_ppo_corrector.py --eval_only
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.informer.model import Informer


# ============================================================
# 1. Actor-Critic 网络
# ============================================================

class ResidualActorCritic(nn.Module):
    """Actor-Critic for residual correction.
    
    Actor: outputs μ, log_σ for Gaussian policy → correction δ
    Critic: outputs state value V(s)
    """
    
    def __init__(self, state_dim: int = 14, hidden_dims: list = [128, 64],
                 max_correction: float = 50.0):
        super().__init__()
        self.max_correction = max_correction
        
        # 共享特征提取
        shared_layers = []
        prev_dim = state_dim
        for hd in hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hd),
                nn.LayerNorm(hd),
                nn.GELU(),
            ])
            prev_dim = hd
        self.shared = nn.Sequential(*shared_layers)
        
        # Actor head: μ and log_σ
        self.actor_mu = nn.Linear(prev_dim, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1))  # learnable scalar log_σ
        
        # Critic head: V(s)
        self.critic = nn.Linear(prev_dim, 1)
        
        # 初始化：让 actor 初始输出接近零修正
        nn.init.zeros_(self.actor_mu.weight)
        nn.init.zeros_(self.actor_mu.bias)
    
    def forward(self, state):
        """返回 (μ, σ, V)"""
        feat = self.shared(state)
        mu = self.actor_mu(feat).squeeze(-1)  # (B,)
        # 用 tanh 约束 μ 在 [-max_correction, max_correction]
        mu = torch.tanh(mu) * self.max_correction
        std = torch.exp(self.actor_log_std).expand_as(mu)
        value = self.critic(feat).squeeze(-1)  # (B,)
        return mu, std, value
    
    def get_action(self, state, deterministic=False):
        """采样动作并返回 log_prob"""
        mu, std, value = self.forward(state)
        if deterministic:
            action = mu
            log_prob = torch.zeros_like(mu)
        else:
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            action = torch.clamp(action, -self.max_correction, self.max_correction)
            log_prob = dist.log_prob(action)
        return action, log_prob, value
    
    def evaluate_action(self, state, action):
        """评估已有动作的 log_prob 和 entropy（PPO 更新时用）"""
        mu, std, value = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value


# ============================================================
# 2. Informer 基础预测器（冻结）
# ============================================================

class FrozenInformer:
    """加载已训练好的 Informer 并生成基础预测"""
    
    def __init__(self, model_path: str, norm_params_path: str, device: torch.device,
                 seq_len=48, label_len=24, pred_len=1,
                 d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
                 dropout=0.05, n_features=6):
        
        self.device = device
        self.n_features = n_features
        
        # 加载归一化参数
        params = np.load(norm_params_path)
        self.target_mean = float(params['target_mean'])
        self.target_std = float(params['target_std'])
        self.feature_min = params['feature_min']
        self.feature_max = params['feature_max']
        
        # 构建并加载 Informer
        self.model = Informer(
            enc_in=n_features, dec_in=n_features, c_out=1,
            seq_len=seq_len, label_len=label_len, pred_len=pred_len,
            d_model=d_model, n_heads=n_heads,
            e_layers=e_layers, d_layers=d_layers,
            d_ff=d_ff, dropout=dropout,
            attn='prob', embed='timeF', activation='gelu',
            output_attention=False, distil=True, device=device
        )
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        
        # 冻结所有参数
        for p in self.model.parameters():
            p.requires_grad = False
        
        print(f"Informer 基模型已加载 (frozen): {model_path}")
        print(f"  target_mean={self.target_mean:.6f}, target_std={self.target_std:.6f}")
    
    def inverse_normalize_target(self, normalized: np.ndarray) -> np.ndarray:
        """反归一化: normalized → hours"""
        log_target = normalized * self.target_std + self.target_mean
        return np.expm1(log_target)
    
    def inverse_normalize_features(self, normalized: np.ndarray) -> np.ndarray:
        """特征反归一化"""
        range_vals = self.feature_max - self.feature_min
        range_vals[range_vals == 0] = 1.0
        return normalized * range_vals + self.feature_min
    
    @torch.no_grad()
    def predict_batch(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """返回 normalized 预测值 (B,)"""
        output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return output.squeeze(-1).squeeze(-1)


# ============================================================
# 3. PPO 数据集 — 使用 memmap
# ============================================================

class PPODataset(Dataset):
    """从 memmap 缓存加载数据，每个样本返回 Informer 输入 + 目标"""
    
    def __init__(self, cache_dir: Path, split: str = 'train', actual_length: int = None):
        self.X = np.load(cache_dir / f"X_{split}.npy", mmap_mode='r')
        self.X_mark_enc = np.load(cache_dir / f"X_mark_enc_{split}.npy", mmap_mode='r')
        self.X_dec = np.load(cache_dir / f"X_dec_{split}.npy", mmap_mode='r')
        self.X_mark_dec = np.load(cache_dir / f"X_mark_dec_{split}.npy", mmap_mode='r')
        self.y = np.load(cache_dir / f"y_{split}.npy", mmap_mode='r')
        self.sd = np.load(cache_dir / f"sd_{split}.npy", mmap_mode='r')
        self.length = actual_length if actual_length is not None else len(self.y)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx].copy()).float(),
            torch.from_numpy(self.X_mark_enc[idx].copy()).float(),
            torch.from_numpy(self.X_dec[idx].copy()).float(),
            torch.from_numpy(self.X_mark_dec[idx].copy()).float(),
            torch.tensor(self.y[idx]).float(),
            torch.tensor(self.sd[idx]).float()
        )


# ============================================================
# 4. 构建状态向量
# ============================================================

def build_state(x_enc_np, pred_hours, sd, feature_min, feature_max):
    """构造 PPO 状态向量
    
    Args:
        x_enc_np: (B, seq_len, 6) normalized features
        pred_hours: (B,) Informer 基础预测（小时）
        sd: (B,) sailing days
        feature_min, feature_max: 归一化参数
        
    Returns:
        state: (B, 14) tensor
    """
    B = x_enc_np.shape[0]
    
    # 反归一化特征
    range_vals = feature_max - feature_min
    range_vals[range_vals == 0] = 1.0
    
    # last step features (6)
    last_step = x_enc_np[:, -1, :]  # (B, 6) normalized
    last_step_raw = last_step * range_vals + feature_min  # 反归一化
    
    # sequence mean features (6)
    seq_mean = x_enc_np.mean(axis=1)  # (B, 6) normalized
    seq_mean_raw = seq_mean * range_vals + feature_min
    
    # 标准化各维度到合理范围
    # lat: /90, lon: /180, sog: /20, cog: /360, dist: /20000, bearing_diff: /180
    scale = np.array([90.0, 180.0, 20.0, 360.0, 20000.0, 180.0], dtype=np.float32)
    last_step_scaled = last_step_raw / scale
    seq_mean_scaled = seq_mean_raw / scale
    
    # pred_hours: log-scale normalization
    pred_hours_feat = np.log1p(np.maximum(pred_hours, 0)) / 10.0  # log1p/10
    
    # sailing_days: /30
    sd_feat = sd / 30.0
    
    # concat: [pred_hours(1), last_step(6), seq_mean(6), sd(1)] = 14
    state = np.column_stack([
        pred_hours_feat.reshape(-1, 1),
        last_step_scaled,
        seq_mean_scaled,
        sd_feat.reshape(-1, 1),
    ]).astype(np.float32)
    
    return state


# ============================================================
# 5. PPO 训练
# ============================================================

class PPOTrainer:
    """PPO with clipped objective for residual correction"""
    
    def __init__(self, actor_critic: ResidualActorCritic, device: torch.device,
                 lr: float = 3e-4, gamma: float = 1.0,  # gamma=1 因为是单步
                 clip_eps: float = 0.2, entropy_coef: float = 0.01,
                 value_coef: float = 0.5, max_grad_norm: float = 0.5,
                 correction_penalty: float = 0.001):
        
        self.ac = actor_critic.to(device)
        self.device = device
        self.optimizer = Adam(self.ac.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.correction_penalty = correction_penalty
    
    def compute_rewards(self, pred_base_hours, corrections, true_hours):
        """计算奖励
        
        reward = |base_error| - |corrected_error| - penalty * |correction|
        正奖励 = 修正改善了预测
        """
        base_error = np.abs(pred_base_hours - true_hours)
        corrected = pred_base_hours + corrections
        corrected_error = np.abs(corrected - true_hours)
        
        improvement = base_error - corrected_error  # 正 = 改善
        penalty = self.correction_penalty * np.abs(corrections)
        
        return (improvement - penalty).astype(np.float32)
    
    def collect_rollout(self, loader, frozen_informer, batch_limit=None):
        """收集一个 epoch 的经验

        Returns:
            states, actions, log_probs, rewards, values (all numpy)
            base_mae, corrected_mae (float)
        """
        all_states = []
        all_actions = []
        all_log_probs = []
        all_rewards = []
        all_values = []
        all_base_errors = []
        all_corrected_errors = []
        
        self.ac.eval()
        
        for i, batch in enumerate(tqdm(loader, desc="Collecting rollouts", leave=False)):
            if batch_limit and i >= batch_limit:
                break
                
            x_enc, x_mark_enc, x_dec, x_mark_dec, y_norm, sd = batch
            
            x_enc_dev = x_enc.to(self.device)
            x_mark_enc_dev = x_mark_enc.to(self.device)
            x_dec_dev = x_dec.to(self.device)
            x_mark_dec_dev = x_mark_dec.to(self.device)
            
            # 1. Informer 基础预测 (normalized → hours)
            pred_norm = frozen_informer.predict_batch(
                x_enc_dev, x_mark_enc_dev, x_dec_dev, x_mark_dec_dev
            ).cpu().numpy()
            pred_hours = frozen_informer.inverse_normalize_target(pred_norm)
            pred_hours = np.maximum(pred_hours, 0)
            
            true_hours = frozen_informer.inverse_normalize_target(y_norm.numpy())
            sd_np = sd.numpy()
            x_enc_np = x_enc.numpy()
            
            # 2. 构建状态
            state = build_state(
                x_enc_np, pred_hours, sd_np,
                frozen_informer.feature_min, frozen_informer.feature_max
            )
            state_t = torch.from_numpy(state).to(self.device)
            
            # 3. Actor-Critic 采样修正量
            with torch.no_grad():
                action, log_prob, value = self.ac.get_action(state_t)
            
            corrections = action.cpu().numpy()
            log_probs_np = log_prob.cpu().numpy()
            values_np = value.cpu().numpy()
            
            # 4. 计算奖励
            rewards = self.compute_rewards(pred_hours, corrections, true_hours)
            
            all_states.append(state)
            all_actions.append(corrections)
            all_log_probs.append(log_probs_np)
            all_rewards.append(rewards)
            all_values.append(values_np)
            
            base_err = np.abs(pred_hours - true_hours)
            corrected_err = np.abs(pred_hours + corrections - true_hours)
            all_base_errors.append(base_err)
            all_corrected_errors.append(corrected_err)
        
        states = np.concatenate(all_states)
        actions = np.concatenate(all_actions)
        log_probs = np.concatenate(all_log_probs)
        rewards = np.concatenate(all_rewards)
        values = np.concatenate(all_values)
        
        base_mae = np.concatenate(all_base_errors).mean()
        corrected_mae = np.concatenate(all_corrected_errors).mean()
        
        return states, actions, log_probs, rewards, values, base_mae, corrected_mae
    
    def ppo_update(self, states, actions, old_log_probs, rewards, old_values,
                   n_epochs: int = 4, mini_batch_size: int = 4096):
        """PPO 策略更新
        
        单步 episode → advantage = reward - V(s)
        """
        self.ac.train()
        
        N = len(states)
        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        old_log_probs_t = torch.from_numpy(old_log_probs).to(self.device)
        
        # Advantage = R - V (单步 episode, no discounting needed)
        advantages = rewards - old_values
        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages_norm = (advantages - adv_mean) / adv_std
        advantages_t = torch.from_numpy(advantages_norm.astype(np.float32)).to(self.device)
        returns_t = torch.from_numpy(rewards.astype(np.float32)).to(self.device)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for _ in range(n_epochs):
            indices = np.random.permutation(N)
            for start in range(0, N, mini_batch_size):
                end = min(start + mini_batch_size, N)
                idx = indices[start:end]
                
                mb_states = states_t[idx]
                mb_actions = actions_t[idx]
                mb_old_log_probs = old_log_probs_t[idx]
                mb_advantages = advantages_t[idx]
                mb_returns = returns_t[idx]
                
                # Evaluate
                new_log_probs, entropy, new_values = self.ac.evaluate_action(mb_states, mb_actions)
                
                # Policy loss (clipped)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values, mb_returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
        }
    
    def evaluate(self, loader, frozen_informer, deterministic=True):
        """在 test/val 集上评估（deterministic=True 时使用 μ 不加噪声）"""
        self.ac.eval()
        
        all_base_hours = []
        all_corrected_hours = []
        all_true_hours = []
        all_corrections = []
        
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            x_enc, x_mark_enc, x_dec, x_mark_dec, y_norm, sd = batch
            
            x_enc_dev = x_enc.to(self.device)
            x_mark_enc_dev = x_mark_enc.to(self.device)
            x_dec_dev = x_dec.to(self.device)
            x_mark_dec_dev = x_mark_dec.to(self.device)
            
            pred_norm = frozen_informer.predict_batch(
                x_enc_dev, x_mark_enc_dev, x_dec_dev, x_mark_dec_dev
            ).cpu().numpy()
            pred_hours = frozen_informer.inverse_normalize_target(pred_norm)
            pred_hours = np.maximum(pred_hours, 0)
            
            true_hours = frozen_informer.inverse_normalize_target(y_norm.numpy())
            sd_np = sd.numpy()
            x_enc_np = x_enc.numpy()
            
            state = build_state(
                x_enc_np, pred_hours, sd_np,
                frozen_informer.feature_min, frozen_informer.feature_max
            )
            state_t = torch.from_numpy(state).to(self.device)
            
            with torch.no_grad():
                action, _, _ = self.ac.get_action(state_t, deterministic=deterministic)
            
            corrections = action.cpu().numpy()
            corrected = np.maximum(pred_hours + corrections, 0)
            
            all_base_hours.append(pred_hours)
            all_corrected_hours.append(corrected)
            all_true_hours.append(true_hours)
            all_corrections.append(corrections)
        
        base_hours = np.concatenate(all_base_hours)
        corrected_hours = np.concatenate(all_corrected_hours)
        true_hours = np.concatenate(all_true_hours)
        corrections = np.concatenate(all_corrections)
        
        base_mae = np.mean(np.abs(base_hours - true_hours))
        corrected_mae = np.mean(np.abs(corrected_hours - true_hours))
        
        base_rmse = np.sqrt(np.mean((base_hours - true_hours) ** 2))
        corrected_rmse = np.sqrt(np.mean((corrected_hours - true_hours) ** 2))
        
        # MAPE (filter > 24h)
        mask = true_hours > 24
        base_mape = np.mean(np.abs((base_hours[mask] - true_hours[mask]) / true_hours[mask])) * 100 if mask.sum() > 0 else 0
        corrected_mape = np.mean(np.abs((corrected_hours[mask] - true_hours[mask]) / true_hours[mask])) * 100 if mask.sum() > 0 else 0
        
        return {
            'base_mae': base_mae,
            'corrected_mae': corrected_mae,
            'base_rmse': base_rmse,
            'corrected_rmse': corrected_rmse,
            'base_mape': base_mape,
            'corrected_mape': corrected_mape,
            'mean_correction': np.mean(corrections),
            'std_correction': np.std(corrections),
            'median_abs_correction': np.median(np.abs(corrections)),
            'improvement': base_mae - corrected_mae,
        }
    
    def save(self, path):
        torch.save({
            'model_state_dict': self.ac.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])


# ============================================================
# 6. Discord 通知
# ============================================================

def send_discord_notification(message: str):
    """发送训练完成通知到 Discord"""
    import urllib.request
    webhook_url = "https://discord.com/api/webhooks/1477180434474336266/HQSS_BKlo1Ib-rzwItazg8ay0To2l-GR1GprnTvlPVpLxCxD9aBd4iHNgX"
    data = json.dumps({"content": message}).encode('utf-8')
    req = urllib.request.Request(webhook_url, data=data, headers={'Content-Type': 'application/json'})
    try:
        urllib.request.urlopen(req)
    except Exception as e:
        print(f"Discord通知发送失败: {e}")


# ============================================================
# 7. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PPO Residual Corrector for ETA")
    
    # 路径
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--model_path', type=str, default=None,
                        help="Informer模型路径，默认 output/best_informer.pth")
    
    # PPO 超参数
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--ppo_epochs', type=int, default=4, help="每个rollout的PPO更新轮数")
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--rollout_batches', type=int, default=200,
                        help="每个epoch收集多少批数据做rollout (0=全量)")
    parser.add_argument('--max_correction', type=float, default=50.0,
                        help="最大修正量（小时）")
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--correction_penalty', type=float, default=0.001,
                        help="修正量正则化惩罚系数")
    parser.add_argument('--hidden_dims', type=str, default='128,64',
                        help="Actor-Critic隐藏层维度")
    
    # Informer 配置（需匹配基模型）
    parser.add_argument('--seq_len', type=int, default=48)
    parser.add_argument('--label_len', type=int, default=24)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--max_sequences', type=int, default=50000000)
    
    # 模式
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    
    # ===== 加载 Informer 基模型 =====
    model_path = args.model_path or os.path.join(args.output_dir, 'best_informer.pth')
    norm_path = os.path.join(args.output_dir, 'norm_params.npz')
    
    frozen_informer = FrozenInformer(
        model_path=model_path,
        norm_params_path=norm_path,
        device=device,
        seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
        d_model=args.d_model, n_heads=args.n_heads,
        e_layers=args.e_layers, d_layers=args.d_layers,
        d_ff=args.d_ff, dropout=args.dropout,
    )
    
    # ===== 加载数据 =====
    cache_tag = f"seq{args.seq_len}_label{args.label_len}_pred{args.pred_len}_mv150000_ms{args.max_sequences}"
    cache_dir = Path(args.output_dir) / "cache_sequences" / cache_tag
    
    counts_path = cache_dir / "actual_counts.npy"
    actual_counts = None
    if counts_path.exists():
        actual_counts = np.load(counts_path, allow_pickle=True).item()
        print(f"数据量: train={actual_counts.get('train', '?'):,}, "
              f"val={actual_counts.get('val', '?'):,}, test={actual_counts.get('test', '?'):,}")
    
    train_dataset = PPODataset(cache_dir, 'train',
                                actual_length=actual_counts.get('train') if actual_counts else None)
    val_dataset = PPODataset(cache_dir, 'val',
                              actual_length=actual_counts.get('val') if actual_counts else None)
    test_dataset = PPODataset(cache_dir, 'test',
                               actual_length=actual_counts.get('test') if actual_counts else None)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, pin_memory=True)
    
    print(f"Train: {len(train_dataset):,}, Val: {len(val_dataset):,}, Test: {len(test_dataset):,}")
    
    # ===== Actor-Critic =====
    state_dim = 14  # pred_hours(1) + last_step(6) + seq_mean(6) + sd(1)
    ac = ResidualActorCritic(
        state_dim=state_dim,
        hidden_dims=hidden_dims,
        max_correction=args.max_correction,
    )
    print(f"Actor-Critic 参数量: {sum(p.numel() for p in ac.parameters()):,}")
    
    ppo = PPOTrainer(
        ac, device, lr=args.lr,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        correction_penalty=args.correction_penalty,
    )
    
    corrector_path = os.path.join(args.output_dir, 'ppo_corrector.pth')
    
    # ===== Eval Only =====
    if args.eval_only:
        if os.path.exists(corrector_path):
            ppo.load(corrector_path)
            print(f"加载 PPO corrector: {corrector_path}")
        else:
            print(f"错误: {corrector_path} 不存在")
            return
        
        print("\n=== 测试集评估 (deterministic) ===")
        metrics = ppo.evaluate(test_loader, frozen_informer, deterministic=True)
        print(f"  Base MAE:      {metrics['base_mae']:.2f}h")
        print(f"  Corrected MAE: {metrics['corrected_mae']:.2f}h")
        print(f"  Improvement:   {metrics['improvement']:.2f}h")
        print(f"  Base RMSE:     {metrics['base_rmse']:.2f}h")
        print(f"  Corrected RMSE:{metrics['corrected_rmse']:.2f}h")
        print(f"  Mean correction: {metrics['mean_correction']:.2f}h")
        print(f"  Std correction:  {metrics['std_correction']:.2f}h")
        print(f"  Median |corr|:   {metrics['median_abs_correction']:.2f}h")
        return
    
    # ===== 训练 =====
    print(f"\n{'='*60}")
    print(f"PPO 残差修正器训练")
    print(f"  max_correction: ±{args.max_correction}h")
    print(f"  hidden_dims: {hidden_dims}")
    print(f"  lr: {args.lr}")
    print(f"  PPO epochs per rollout: {args.ppo_epochs}")
    print(f"  rollout_batches: {args.rollout_batches or 'all'}")
    print(f"  correction_penalty: {args.correction_penalty}")
    print(f"{'='*60}\n")
    
    best_val_mae = float('inf')
    best_improvement = 0
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # 1. Collect rollout from training data
        batch_limit = args.rollout_batches if args.rollout_batches > 0 else None
        states, actions, log_probs, rewards, values, train_base_mae, train_corrected_mae = \
            ppo.collect_rollout(train_loader, frozen_informer, batch_limit=batch_limit)
        
        print(f"  Rollout: {len(states):,} samples, "
              f"base_MAE={train_base_mae:.2f}h, corrected_MAE={train_corrected_mae:.2f}h, "
              f"Δ={train_base_mae - train_corrected_mae:+.2f}h")
        print(f"  Corrections: mean={actions.mean():.2f}h, std={actions.std():.2f}h, "
              f"median|δ|={np.median(np.abs(actions)):.2f}h")
        print(f"  Rewards: mean={rewards.mean():.4f}, std={rewards.std():.4f}")
        
        # 2. PPO update
        update_info = ppo.ppo_update(
            states, actions, log_probs, rewards, values,
            n_epochs=args.ppo_epochs,
            mini_batch_size=args.batch_size,
        )
        print(f"  PPO Update: policy_loss={update_info['policy_loss']:.4f}, "
              f"value_loss={update_info['value_loss']:.4f}, "
              f"entropy={update_info['entropy']:.4f}")
        
        # 3. Validate
        val_metrics = ppo.evaluate(val_loader, frozen_informer, deterministic=True)
        print(f"  Val: base_MAE={val_metrics['base_mae']:.2f}h, "
              f"corrected_MAE={val_metrics['corrected_mae']:.2f}h, "
              f"Δ={val_metrics['improvement']:+.2f}h")
        
        # 保存最佳模型（基于 corrected MAE 最低）
        if val_metrics['corrected_mae'] < best_val_mae:
            best_val_mae = val_metrics['corrected_mae']
            best_improvement = val_metrics['improvement']
            ppo.save(corrector_path)
            print(f"  -> 保存最佳 corrector! MAE={best_val_mae:.2f}h (Δ={best_improvement:+.2f}h)")
        
        # Actor 标准差
        with torch.no_grad():
            log_std = ppo.ac.actor_log_std.item()
        print(f"  Actor σ: {np.exp(log_std):.4f}")
    
    # ===== 最终测试集评估 =====
    print(f"\n{'='*60}")
    print("最终测试集评估")
    print(f"{'='*60}")
    
    ppo.load(corrector_path)
    test_metrics = ppo.evaluate(test_loader, frozen_informer, deterministic=True)
    
    print(f"\n【测试集结果】")
    print(f"  Base Informer MAE:      {test_metrics['base_mae']:.2f}h")
    print(f"  PPO Corrected MAE:      {test_metrics['corrected_mae']:.2f}h")
    print(f"  Improvement (Δ):        {test_metrics['improvement']:+.2f}h")
    print(f"  Base RMSE:              {test_metrics['base_rmse']:.2f}h")
    print(f"  PPO Corrected RMSE:     {test_metrics['corrected_rmse']:.2f}h")
    print(f"  Base MAPE:              {test_metrics['base_mape']:.2f}%")
    print(f"  PPO Corrected MAPE:     {test_metrics['corrected_mape']:.2f}%")
    print(f"  Mean correction:        {test_metrics['mean_correction']:.2f}h")
    print(f"  Std correction:         {test_metrics['std_correction']:.2f}h")
    print(f"  Median |correction|:    {test_metrics['median_abs_correction']:.2f}h")
    
    # Discord 通知
    msg = (f"🎯 PPO残差修正器训练完成\n"
           f"Base MAE: {test_metrics['base_mae']:.2f}h → Corrected MAE: {test_metrics['corrected_mae']:.2f}h\n"
           f"Improvement: {test_metrics['improvement']:+.2f}h\n"
           f"RMSE: {test_metrics['base_rmse']:.2f} → {test_metrics['corrected_rmse']:.2f}\n"
           f"MAPE: {test_metrics['base_mape']:.2f}% → {test_metrics['corrected_mape']:.2f}%")
    send_discord_notification(msg)
    print("\n训练完成！通知已发送。")


if __name__ == '__main__':
    main()
