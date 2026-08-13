import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ============================================================
# MetaSTC-C: 静态聚类 + 显式端到端聚类损失
# ------------------------------------------------------------
# 消融目的: 隔离"端到端"机制的贡献
#   - 聚类保持静态 (h_pooled, 固定 centroids, 单一 α)
#   - 加入 cluster_loss: L_compact + L_sep + L_balance
#   - 训练用 L_pred + λ·L_cluster, 联合优化聚类与预测
#   - forward 返回 (prediction, alpha, centroids, h_pooled)
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()

# ==========================================
# 1. Traffic2Vec (不变, 只返回 h_t)
# ==========================================
class Traffic2Vec(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Traffic2Vec, self).__init__()
        self.w_d = nn.Parameter(torch.tensor(2 * math.pi / 24.0))
        self.w_w = nn.Parameter(torch.tensor(2 * math.pi / (24.0 * 7.0)))
        self.phi_1 = nn.Parameter(torch.randn(1))
        self.phi_2 = nn.Parameter(torch.randn(1))
        self.w_0 = nn.Parameter(torch.randn(1))
        self.b_0 = nn.Parameter(torch.zeros(1))
        self.dense = nn.Linear(input_dim + 3, hidden_dim)

    def forward(self, x, t_norm):
        linear_term = self.w_0 * t_norm + self.b_0
        daily_term = torch.sin(self.w_d * t_norm + self.phi_1)
        weekly_term = torch.sin(self.w_w * t_norm + self.phi_2)
        combined = torch.cat([x, linear_term, daily_term, weekly_term], dim=-1)
        h_t = F.relu(self.dense(combined))
        return h_t

# ==========================================
# 2. Spatial Encoder (不变)
# ==========================================
class SpatialEncoder(nn.Module):
    def __init__(self, num_roads, num_levels, num_lanes, embed_dim):
        super(SpatialEncoder, self).__init__()
        self.embed_road = nn.Embedding(num_roads, embed_dim)
        self.embed_level = nn.Embedding(num_levels, embed_dim)
        self.embed_lane = nn.Embedding(num_lanes, embed_dim)
        self.out_dim = 3 * embed_dim

    def forward(self, road_idx, road_level, lane_num):
        emb_r = self.embed_road(road_idx)
        emb_l = self.embed_level(road_level)
        emb_n = self.embed_lane(lane_num)
        h_s = torch.cat([emb_r, emb_l, emb_n], dim=-1)
        return h_s

# ==========================================
# 3. Spatio-Temporal Gating Unit (不变, 返回 h_seq + h_pooled)
# ==========================================
class SpatioTemporalGating(nn.Module):
    def __init__(self, spatial_dim, temporal_dim):
        super(SpatioTemporalGating, self).__init__()
        self.W_ss = nn.Linear(spatial_dim, spatial_dim)
        self.W_ts = nn.Linear(temporal_dim, spatial_dim)
        self.W_st = nn.Linear(spatial_dim, temporal_dim)
        self.W_tt = nn.Linear(temporal_dim, temporal_dim)
        self.final_dim = spatial_dim + temporal_dim

    def forward(self, h_s, h_t):
        seq_len = h_t.size(1)
        h_s_expanded = h_s.unsqueeze(1).repeat(1, seq_len, 1)
        g_s = torch.sigmoid(self.W_ss(h_s_expanded) + self.W_ts(h_t))
        g_t = torch.sigmoid(self.W_st(h_s_expanded) + self.W_tt(h_t))
        fused_s = g_s * h_s_expanded
        fused_t = g_t * h_t
        h = torch.cat([fused_s, fused_t], dim=-1)
        h_pooled = torch.mean(h, dim=1)
        return h, h_pooled

# ==========================================
# 4. Cluster-Aware Memory Unit (静态, 不变)
# ==========================================
class ClusterMemoryUnit(nn.Module):
    def __init__(self, hidden_dim, num_clusters, mem_dim, temperature=0.5):
        super(ClusterMemoryUnit, self).__init__()
        self.num_clusters = num_clusters
        self.mem_dim = mem_dim
        self.temperature = temperature

        self.centroids = nn.Parameter(torch.randn(num_clusters, hidden_dim))
        self.memory = nn.Parameter(torch.randn(num_clusters, mem_dim))

        self.W_r = nn.Linear(mem_dim, mem_dim)
        self.W_u = nn.Linear(mem_dim, mem_dim)
        self.W_c = nn.Linear(mem_dim, mem_dim)

    def get_similarity(self, h_j):
        scores = torch.matmul(h_j, self.centroids.t())
        alpha = F.softmax(scores / self.temperature, dim=1)
        return alpha

    def forward(self, h_j):
        alpha = self.get_similarity(h_j)
        M_j_c = torch.matmul(alpha, self.memory)

        r_t = torch.sigmoid(self.W_r(M_j_c))
        u_t = torch.sigmoid(self.W_u(M_j_c))
        c_tilde = torch.tanh(self.W_c(r_t * M_j_c))
        current_M_c = (1 - u_t) * M_j_c + u_t * c_tilde
        return current_M_c, alpha

# ==========================================
# 5. Prediction Backbone (不变)
# ==========================================
class MetaBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, mem_dim, num_layers=3):
        super(MetaBackbone, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.hidden_dim = hidden_dim
        self.final_predictor = nn.Sequential(
            nn.Linear(hidden_dim + mem_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x_seq, memory_context):
        lstm_out, _ = self.lstm(x_seq)
        last_hidden = lstm_out[:, -1, :]
        combined = torch.cat([last_hidden, memory_context], dim=-1)
        prediction = self.final_predictor(combined)
        return prediction

# ==========================================
# 6. Main MetaSTC Framework (Static + End-to-End)
# ==========================================
class MetaSTC(nn.Module):
    def __init__(self,
                 num_roads, num_levels, num_lanes,
                 input_flow_dim=1,
                 spatial_embed_dim=16,
                 temporal_hidden_dim=12,
                 num_clusters=5,
                 mem_dim=64,
                 output_seq_len=6,
                 temperature=0.5,
                 num_layers=3):
        super(MetaSTC, self).__init__()
        self.num_clusters = num_clusters

        self.traffic2vec = Traffic2Vec(input_dim=input_flow_dim, hidden_dim=temporal_hidden_dim)
        self.spatial_encoder = SpatialEncoder(num_roads, num_levels, num_lanes, spatial_embed_dim)
        spatial_out_dim = spatial_embed_dim * 3

        self.gating = SpatioTemporalGating(spatial_dim=spatial_out_dim, temporal_dim=temporal_hidden_dim)

        fused_dim = spatial_out_dim + temporal_hidden_dim
        self.memory_unit = ClusterMemoryUnit(
            hidden_dim=fused_dim,
            num_clusters=num_clusters,
            mem_dim=mem_dim,
            temperature=temperature
        )

        self.backbone = MetaBackbone(
            input_dim=fused_dim,
            hidden_dim=64,
            output_dim=output_seq_len,
            mem_dim=mem_dim,
            num_layers=num_layers
        )

    def forward(self, x_flow, x_time, road_idx, road_level, lane_num):
        h_t = self.traffic2vec(x_flow, x_time)
        h_s = self.spatial_encoder(road_idx, road_level, lane_num)
        h_seq, h_pooled = self.gating(h_s, h_t)
        memory_context, alpha = self.memory_unit(h_pooled)
        prediction = self.backbone(h_seq, memory_context)
        # 返回聚类损失所需量: alpha, centroids, h_pooled
        centroids = self.memory_unit.centroids  # [K, D] 静态参数
        return prediction, alpha, centroids, h_pooled

# ==========================================
# 7. End-to-End Clustering Loss (静态版)
#    h_pooled: [B, D], alpha: [B, K], centroids: [K, D]
# ==========================================
def cluster_loss(h_pooled, alpha, centroids, num_clusters,
                 lambda_compact=1.0, lambda_sep=1.0, lambda_balance=0.1):
    """[修复] 端到端聚类损失 —— 全部分量有界，防止梯度爆炸
    - L_compact: 用 cosine 相似度（有界 [-1,1]），最大化 → 取负
    - L_sep:     用 tanh 压缩质心距离（有界 [0,1]），最大化 → 取负
    - L_balance: p_k 加 detach，只调节 α 分布不反传到质心
    """
    # (1) 紧致性: 最大化样本与所属质心的余弦相似度
    h_norm = F.normalize(h_pooled, dim=-1)                              # [B, D]
    c_norm = F.normalize(centroids, dim=-1)                             # [K, D]
    sim = torch.matmul(h_norm, c_norm.t())                              # [B, K] ∈ [-1,1]
    L_compact = -(alpha * sim).sum(-1).mean()                            # ∈ [-1, 1]

    # (2) 分离性: 最大化质心间余弦距离（1 - 余弦相似度）/2 ∈ [0,1]
    c_sim = torch.matmul(c_norm, c_norm.t())                            # [K, K] ∈ [-1,1]
    c_dist = (1.0 - c_sim) / 2.0                                        # [K, K] ∈ [0,1]
    mask = torch.triu(torch.ones(num_clusters, num_clusters, device=centroids.device), diagonal=1).bool()
    L_sep = -c_dist[mask].mean()                                        # ∈ [-1, 0]

    # (3) 均衡性: 最大化簇分布熵（防塌缩）；p_k detach 防止与 L_sep 冲突
    p_k = alpha.detach().mean(dim=0)                                    # [K], stop-grad
    entropy = -(p_k * torch.log(p_k + 1e-10)).sum()
    L_balance = -entropy                                                # ∈ [-log K, 0]

    total = lambda_compact * L_compact + lambda_sep * L_sep + lambda_balance * L_balance
    return total, L_compact, L_sep, L_balance

# ==========================================
# 8. Data Loading (不变)
# ==========================================
class TrafficDataset(Dataset):
    def __init__(self,
                 pkl_path, feature_path,
                 look_back=12, look_forward=6,
                 time_range=None, max_flow_override=None):
        if not os.path.exists(pkl_path): raise FileNotFoundError(f"{pkl_path} not found")

        with open(pkl_path, 'rb') as f: raw_data = pickle.load(f)
        feature_df = pd.read_csv(feature_path)
        csv_ids = set(feature_df['link_ID'].unique())
        valid_data = [d for d in raw_data if d['id'] in csv_ids]
        id_list = [d['id'] for d in valid_data]
        data_flow = np.array([d['flow'] for d in valid_data])
        original_time_len = data_flow.shape[1]

        feature_df = feature_df.set_index('link_ID').loc[id_list].reset_index()

        if max_flow_override is not None:
            self.max_flow = max_flow_override
        else:
            self.max_flow = data_flow[:, time_range[0]:time_range[1]].max() if time_range else data_flow.max()

        data_flow = data_flow / (self.max_flow + 1e-10)
        if time_range: data_flow = data_flow[:, time_range[0]:time_range[1]]

        level_categories = sorted(feature_df['Kind'].unique().tolist())
        self.level_map = {l: i for i, l in enumerate(level_categories)}
        lane_categories = sorted(feature_df['LaneNum'].unique().tolist())
        self.lane_map = {l: i for i, l in enumerate(lane_categories)}
        self.road_map = {rid: i for i, rid in enumerate(id_list)}

        self.num_roads = len(id_list)
        self.num_levels = len(level_categories)
        self.num_lanes = len(lane_categories)

        self.data_flow = torch.from_numpy(data_flow).float().unsqueeze(-1)
        full_time_indices = np.linspace(0, 1, original_time_len)
        if time_range:
            self.time_indices = torch.from_numpy(full_time_indices[time_range[0]:time_range[1]]).float().unsqueeze(-1)
        else:
            self.time_indices = torch.from_numpy(full_time_indices).float().unsqueeze(-1)

        self.road_ids = torch.LongTensor([self.road_map[rid] for rid in id_list])
        self.levels = torch.LongTensor([self.level_map[feature_df.iloc[i]['Kind']] for i in range(len(id_list))])
        self.lanes = torch.LongTensor([self.lane_map[feature_df.iloc[i]['LaneNum']] for i in range(len(id_list))])

        self.look_back = look_back
        self.look_forward = look_forward
        self.time_len = self.data_flow.shape[1]
        self.samples_per_road = self.time_len - look_back - look_forward
        self.total_samples = self.num_roads * self.samples_per_road

    def __len__(self): return self.total_samples

    def __getitem__(self, idx):
        road_idx = idx // self.samples_per_road
        time_start = idx % self.samples_per_road

        x_f = self.data_flow[road_idx, time_start : time_start + self.look_back]
        x_t = self.time_indices[time_start : time_start + self.look_back]
        tgt = self.data_flow[road_idx, time_start + self.look_back : time_start + self.look_back + self.look_forward].squeeze(-1)

        r_id = self.road_ids[road_idx]
        lvl = self.levels[road_idx]
        ln = self.lanes[road_idx]

        return x_f, x_t, r_id, lvl, ln, tgt

# ============================================================
# 9. 训练 + 可视化 (C: 静态聚类 + 显式端到端聚类损失)
# ------------------------------------------------------------
# 关键诊断: cluster_loss 各分量 (compact/sep/balance) 的演化
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_model(model, train_loader, val_loader, config, log_prefix='C'):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []
    cls_losses = {'compact': [], 'sep': [], 'balance': []}
    best_val = float('inf')
    patience_counter = 0
    os.makedirs('figure', exist_ok=True); os.makedirs('log', exist_ok=True)
    log_file = open(f'log/train_{log_prefix}.txt', 'w')

    for epoch in range(config['epochs']):
        t0 = time.time(); model.train()
        epoch_loss = 0.0; n_batches = 0
        ep_compact = ep_sep = ep_balance = 0.0
        for i, (x_f, x_t, r_id, r_lvl, ln, tgt) in enumerate(train_loader):
            x_f, x_t, tgt = x_f.to(device), x_t.to(device), tgt.to(device)
            r_id, r_lvl, ln = r_id.to(device), r_lvl.to(device), ln.to(device)
            optimizer.zero_grad()
            # C: 4-tuple (pred, alpha, centroids, h_pooled)
            pred, alpha, centroids, h_pooled = model(x_f, x_t, r_id, r_lvl, ln)
            loss_pred = criterion(pred, tgt)
            loss_cls, lc, ls, lb = cluster_loss(h_pooled, alpha, centroids,
                                                config['num_clusters'])
            loss = loss_pred + config['lambda_cls'] * loss_cls
            loss.backward(); optimizer.step()
            epoch_loss += loss_pred.item(); n_batches += 1
            ep_compact += lc.item(); ep_sep += ls.item(); ep_balance += lb.item()
        avg_train = epoch_loss / max(n_batches, 1)
        cls_losses['compact'].append(ep_compact / max(n_batches, 1))
        cls_losses['sep'].append(ep_sep / max(n_batches, 1))
        cls_losses['balance'].append(ep_balance / max(n_batches, 1))

        model.eval(); val_loss = 0.0; n_val = 0
        with torch.no_grad():
            for i, (x_f, x_t, r_id, r_lvl, ln, tgt) in enumerate(val_loader):
                x_f, x_t, tgt = x_f.to(device), x_t.to(device), tgt.to(device)
                r_id, r_lvl, ln = r_id.to(device), r_lvl.to(device), ln.to(device)
                pred, _, _, _ = model(x_f, x_t, r_id, r_lvl, ln)
                val_loss += criterion(pred, tgt).item(); n_val += 1
        avg_val = val_loss / max(n_val, 1)
        train_losses.append(avg_train); val_losses.append(avg_val)
        msg = (f"Epoch {epoch+1}/{config['epochs']} | Train {avg_train:.6f} | Val {avg_val:.6f} "
               f"| compact {cls_losses['compact'][-1]:.4f} | sep {cls_losses['sep'][-1]:.4f} "
               f"| bal {cls_losses['balance'][-1]:.4f} | {time.time()-t0:.1f}s")
        print(msg); log_file.write(msg+'\n'); log_file.flush()
        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), f'param/metastc_{log_prefix}_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                msg = f"Early stopping at epoch {epoch+1} (patience={config['patience']})"
                print(msg); log_file.write(msg+'\n'); log_file.flush()
                break
    log_file.close()
    return train_losses, val_losses, cls_losses

def plot_loss(train_losses, val_losses, cls_losses, save_path='figure/C_loss.png'):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    # 左: 预测损失
    axes[0].plot(train_losses, label='Train', color='#4C72B0')
    axes[0].plot(val_losses, label='Val', color='#C44E52', linestyle='--')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('MetaSTC-C Prediction Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
    # 右: 聚类损失分量
    axes[1].plot(cls_losses['compact'], label='L_compact (↓)', color='#4C72B0')
    axes[1].plot(cls_losses['sep'], label='L_sep (↑→-)', color='#55A868')
    axes[1].plot(cls_losses['balance'], label='L_balance (↑→-)', color='#C44E52')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Cluster Loss')
    axes[1].set_title('Cluster Loss Components'); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"Loss curves saved to {save_path}")

def plot_prediction(model, val_loader, max_flow, save_path='figure/C_prediction.png', max_batches=5):
    model.eval(); all_preds, all_targets = [], []
    with torch.no_grad():
        for i, (x_f, x_t, r_id, r_lvl, ln, tgt) in enumerate(val_loader):
            if i >= max_batches: break
            x_f, x_t = x_f.to(device), x_t.to(device)
            r_id, r_lvl, ln = r_id.to(device), r_lvl.to(device), ln.to(device)
            pred, _, _, _ = model(x_f, x_t, r_id, r_lvl, ln)
            all_preds.append(pred.cpu().numpy()); all_targets.append(tgt.numpy())
    y_pred = np.concatenate(all_preds, axis=0).flatten() * max_flow
    y_true = np.concatenate(all_targets, axis=0).flatten() * max_flow
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    plt.figure(figsize=(15, 6))
    n_show = min(len(y_true), 300)
    plt.plot(y_true[:n_show], label='Ground Truth', color='#4C72B0')
    plt.plot(y_pred[:n_show], label='Prediction', color='#55A868', linestyle='--')
    plt.title(f'MetaSTC-C Prediction | MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}')
    plt.xlabel('Sample'); plt.ylabel('Traffic Flow')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"Prediction plot saved to {save_path} | MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}")
    return mae, rmse, r2

def plot_cluster_alpha(model, val_loader, save_path='figure/C_cluster_alpha.png', max_batches=3):
    """静态版: alpha [B, K]"""
    model.eval(); all_alpha = []
    with torch.no_grad():
        for i, (x_f, x_t, r_id, r_lvl, ln, tgt) in enumerate(val_loader):
            if i >= max_batches: break
            x_f, x_t = x_f.to(device), x_t.to(device)
            r_id, r_lvl, ln = r_id.to(device), r_lvl.to(device), ln.to(device)
            _, alpha, _, _ = model(x_f, x_t, r_id, r_lvl, ln)  # [B, K]
            all_alpha.append(alpha.cpu().numpy())
    alpha_mat = np.concatenate(all_alpha, axis=0)  # [N, K]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    n_show = min(alpha_mat.shape[0], 100)
    im = axes[0].imshow(alpha_mat[:n_show].T, aspect='auto', cmap='YlOrRd')
    axes[0].set_xlabel('Sample'); axes[0].set_ylabel('Cluster k')
    axes[0].set_title('Cluster Membership α (per sample)')
    plt.colorbar(im, ax=axes[0])
    labels = np.argmax(alpha_mat, axis=1)
    counts = np.bincount(labels, minlength=alpha_mat.shape[1])
    axes[1].bar(range(len(counts)), counts, color='#4C72B0')
    axes[1].set_xlabel('Cluster k'); axes[1].set_ylabel('Sample Count')
    axes[1].set_title('Cluster Size Distribution (argmax α)')
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"Cluster α heatmap saved to {save_path}")
    p_k = alpha_mat.mean(axis=0)
    entropy = -np.sum(p_k * np.log(p_k + 1e-10))
    print(f"Cluster distribution entropy: {entropy:.4f} (max={np.log(alpha_mat.shape[1]):.4f})")

def run_training(look_back=12):
    tag = f'C_L{look_back}'
    config = {
        'batch_size': 64, 'epochs': 1000, 'lr': 0.001, 'patience': 20,
        'pkl_path': 'data/traffic_flow/1/20230306/part-00000.pkl',
        'feature_path': 'data/link_feature.csv', 'split_ratio': 0.8,
        'num_clusters': 5, 'lambda_cls': 0.01, 'look_back': look_back,
    }
    os.makedirs('param', exist_ok=True)
    with open(config['pkl_path'], 'rb') as f:
        raw = pickle.load(f); time_len = len(raw[0]['flow'])
    train_end = int(time_len * config['split_ratio'])
    train_ds = TrafficDataset(config['pkl_path'], config['feature_path'],
                              look_back=config['look_back'], time_range=(0, train_end))
    max_flow = train_ds.max_flow
    val_ds = TrafficDataset(config['pkl_path'], config['feature_path'],
                            look_back=config['look_back'],
                            time_range=(train_end, time_len), max_flow_override=max_flow)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
    model = MetaSTC(
        num_roads=train_ds.num_roads, num_levels=train_ds.num_levels, num_lanes=train_ds.num_lanes,
        num_clusters=config['num_clusters']
    ).to(device)
    print(f">>> Training MetaSTC-C (Static + end-to-end cluster loss, L={look_back})")
    train_losses, val_losses, cls_losses = train_model(model, train_loader, val_loader, config, log_prefix=tag)
    model.load_state_dict(torch.load(f'param/metastc_{tag}_best.pth', weights_only=True))
    plot_loss(train_losses, val_losses, cls_losses, f'figure/{tag}_loss.png')
    plot_prediction(model, val_loader, max_flow, f'figure/{tag}_prediction.png')
    plot_cluster_alpha(model, val_loader, f'figure/{tag}_cluster_alpha.png')
    print(f"\n>>> Done. Outputs: figure/{tag}_loss.png, figure/{tag}_prediction.png, figure/{tag}_cluster_alpha.png")

if __name__ == "__main__":
    for L in [12, 24]:
        run_training(look_back=L)
