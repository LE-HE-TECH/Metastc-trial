import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import math
import pickle
import os
import time
import copy
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans

# ============================================================
# MetaSTC Full: 论文完整实现
# ------------------------------------------------------------
# 架构: Traffic2Vec + SpatialEncoder + Gating + ClusterMemoryUnit + Hypernetwork
# 聚类: K-Means++ 在时空表示 h 上 (论文 Algorithm 1)
# 训练: MAML (FOMAML) + Support/Query 划分 (论文 Eq.13-16)
# 推理: 簇感知 (每簇独立微调模型)
# ============================================================

# ==========================================
# Device Configuration
# ==========================================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()

# ==========================================
# 1. Traffic2Vec (Temporal Encoder) - Eq.1-3
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
# 2. Spatial Encoder - Eq.4
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
# 3. Spatio-Temporal Gating Unit - Eq.5-7
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
# 4. Cluster-Aware Memory Unit - Eq.8-12
# τ=0.5 (项目硬约束)
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

    def forward(self, h_j, prev_M_c=None):
        if prev_M_c is None:
            prev_M_c = self.memory
        alpha = self.get_similarity(h_j)
        M_j_c = torch.matmul(alpha, prev_M_c)

        r_t = torch.sigmoid(self.W_r(M_j_c))
        u_t = torch.sigmoid(self.W_u(M_j_c))
        c_tilde = torch.tanh(self.W_c(r_t * M_j_c))
        current_M_c = (1 - u_t) * M_j_c + u_t * c_tilde
        return current_M_c, alpha

# ==========================================
# 5. Prediction Backbone (Hypernetwork)
# Memory → 预测层专属权重 (论文: distinct parameters)
# ==========================================
class MetaBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, mem_dim):
        super(MetaBackbone, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.weight_gen = nn.Linear(mem_dim, hidden_dim * output_dim)
        self.bias_gen = nn.Linear(mem_dim, output_dim)

    def forward(self, x_seq, memory_context):
        lstm_out, _ = self.lstm(x_seq)
        last_hidden = lstm_out[:, -1, :]

        generated_weights = self.weight_gen(memory_context)
        generated_bias = self.bias_gen(memory_context)
        weights = generated_weights.view(-1, self.hidden_dim, self.output_dim)

        prediction = torch.bmm(last_hidden.unsqueeze(1), weights).squeeze(1) + generated_bias
        return prediction

# ==========================================
# 6. Main MetaSTC Framework
# ==========================================
class MetaSTC(nn.Module):
    def __init__(self,
                 num_roads, num_levels, num_lanes,
                 input_flow_dim=1,
                 spatial_embed_dim=16,
                 temporal_hidden_dim=32,
                 num_clusters=5,
                 mem_dim=64,
                 output_seq_len=6):
        super(MetaSTC, self).__init__()

        self.traffic2vec = Traffic2Vec(input_dim=input_flow_dim, hidden_dim=temporal_hidden_dim)
        self.spatial_encoder = SpatialEncoder(num_roads, num_levels, num_lanes, spatial_embed_dim)
        spatial_out_dim = spatial_embed_dim * 3

        self.gating = SpatioTemporalGating(spatial_dim=spatial_out_dim, temporal_dim=temporal_hidden_dim)

        fused_dim = spatial_out_dim + temporal_hidden_dim
        self.memory_unit = ClusterMemoryUnit(hidden_dim=fused_dim, num_clusters=num_clusters,
                                             mem_dim=mem_dim, temperature=0.5)

        self.backbone = MetaBackbone(input_dim=fused_dim, hidden_dim=64,
                                     output_dim=output_seq_len, mem_dim=mem_dim)

    def forward(self, x_flow, x_time, road_idx, road_level, lane_num):
        h_t = self.traffic2vec(x_flow, x_time)
        h_s = self.spatial_encoder(road_idx, road_level, lane_num)
        h_seq, h_pooled = self.gating(h_s, h_t)
        memory_context, cluster_attn = self.memory_unit(h_pooled)
        prediction = self.backbone(h_seq, memory_context)
        return prediction, cluster_attn, h_pooled

    def get_representation(self, x_flow, x_time, road_idx, road_level, lane_num):
        """仅提取 h_pooled (用于 K-Means++ 聚类)"""
        h_t = self.traffic2vec(x_flow, x_time)
        h_s = self.spatial_encoder(road_idx, road_level, lane_num)
        _, h_pooled = self.gating(h_s, h_t)
        return h_pooled

# ==========================================
# 7. Data Loading (增加 road_indices 供簇划分)
# ==========================================
class TrafficDataset(Dataset):
    def __init__(self,
                 pkl_path, feature_path,
                 look_back=12, look_forward=6,
                 time_range=None, max_flow_override=None):
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"{pkl_path} not found")

        with open(pkl_path, 'rb') as f:
            raw_data = pickle.load(f)
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
        if time_range:
            data_flow = data_flow[:, time_range[0]:time_range[1]]

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
        # 预计算 road_indices (供簇划分)
        self.road_indices = np.arange(self.total_samples) // self.samples_per_road

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        road_idx = idx // self.samples_per_road
        time_start = idx % self.samples_per_road

        x_f = self.data_flow[road_idx, time_start: time_start + self.look_back]
        x_t = self.time_indices[time_start: time_start + self.look_back]
        tgt = self.data_flow[road_idx, time_start + self.look_back: time_start + self.look_back + self.look_forward].squeeze(-1)

        r_id = self.road_ids[road_idx]
        lvl = self.levels[road_idx]
        ln = self.lanes[road_idx]

        return x_f, x_t, r_id, lvl, ln, tgt

# ============================================================
# 8. K-Means++ 聚类 (在时空表示 h 上)
# 论文 Algorithm 1: 聚类输入是 h, 不是原始特征
# ============================================================
def pretrain_encoder(model, train_loader, epochs=10, lr=0.001):
    """预训练编码器以获得有意义的 h 表示"""
    print(">>> 预训练编码器 (用于提取 h 进行聚类)...")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss, count = 0, 0
        for i, (x_f, x_t, r_id, r_lvl, ln, tgt) in enumerate(train_loader):
            if i >= 50:
                break
            x_f, x_t, tgt = x_f.to(device), x_t.to(device), tgt.to(device)
            r_id, r_lvl, ln = r_id.to(device), r_lvl.to(device), ln.to(device)

            optimizer.zero_grad()
            pred, _, _ = model(x_f, x_t, r_id, r_lvl, ln)
            loss = criterion(pred, tgt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1
        print(f"  Pretrain Epoch {epoch+1}/{epochs} | Loss: {total_loss/max(count,1):.6f}")


def extract_representations(model, dataset, batch_size=128):
    """提取每条路的 h_pooled 表示 (对时间维度取平均)"""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 收集所有样本的 h_pooled 和 road_idx
    road_h_map = {}  # road_idx -> list of h_pooled

    with torch.no_grad():
        for x_f, x_t, r_id, r_lvl, ln, tgt in loader:
            x_f, x_t = x_f.to(device), x_t.to(device)
            r_id, r_lvl, ln = r_id.to(device), r_lvl.to(device), ln.to(device)
            h_pooled = model.get_representation(x_f, x_t, r_id, r_lvl, ln)

            for i in range(r_id.size(0)):
                rid = r_id[i].item()
                if rid not in road_h_map:
                    road_h_map[rid] = []
                road_h_map[rid].append(h_pooled[i].cpu())

    # 对每条路的 h_pooled 取平均
    num_roads = dataset.num_roads
    h_dim = next(iter(road_h_map.values()))[0].shape[0]
    h_matrix = np.zeros((num_roads, h_dim))

    for rid, h_list in road_h_map.items():
        h_mean = torch.stack(h_list).mean(dim=0)
        h_matrix[rid] = h_mean.numpy()

    print(f"  提取了 {num_roads} 条路的 h 表示, 维度: {h_dim}")
    return h_matrix


def perform_clustering_on_h(h_matrix, k=5):
    """K-Means++ 聚类在 h 上 (sklearn 默认 init='k-means++')"""
    print(f">>> K-Means++ 聚类 (在 h 上, K={k})...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(h_matrix)
    print(f"  聚类完成。各类别数量: {np.bincount(labels)}")
    return labels  # road_index -> cluster_label

# ============================================================
# 9. MAML 训练 (FOMAML + Support/Query + 参数恢复)
# 论文 Eq.13-16
# ============================================================
def split_support_query(dataset, road_cluster_labels, road_indices, num_clusters, batch_size=64):
    """
    按簇划分数据, 每簇内 50:50 分为 Support 和 Query
    返回: {cluster_k: (support_loader, query_loader)}
    """
    cluster_loaders = {}

    for k in range(num_clusters):
        # 找出属于簇 k 的样本索引
        indices = [i for i, r_idx in enumerate(road_indices)
                   if road_cluster_labels[r_idx] == k]

        if len(indices) < 2:
            print(f"  Warning: Cluster {k} has only {len(indices)} samples, skipping.")
            continue

        # 50:50 划分
        np.random.shuffle(indices)
        split_point = len(indices) // 2
        support_indices = indices[:split_point]
        query_indices = indices[split_point:]

        support_subset = Subset(dataset, support_indices)
        query_subset = Subset(dataset, query_indices)

        support_loader = DataLoader(support_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        query_loader = DataLoader(query_subset, batch_size=batch_size, shuffle=True, num_workers=0)

        cluster_loaders[k] = (support_loader, query_loader)
        print(f"  Cluster {k}: Support {len(support_indices)} | Query {len(query_indices)}")

    return cluster_loaders


def maml_step(model, support_batch, query_batch, optimizer, lr_inner, criterion):
    """
    FOMAML 单步训练:
    1. 保存原始参数 θ
    2. Inner loop: θ' = θ - β·∇L(D_s) (FOMAML, create_graph=False)
    3. Outer loop: 计算 L(D_q; θ'), 反向传播
    4. 恢复 θ, 应用 ∇_θ' L(D_q) ≈ ∇_θ L(D_q) (FOMAML 近似)
    """
    x_f_s, x_t_s, r_id_s, r_lvl_s, ln_s, tgt_s = support_batch
    x_f_q, x_t_q, r_id_q, r_lvl_q, ln_q, tgt_q = query_batch

    x_f_s, x_t_s, tgt_s = x_f_s.to(device), x_t_s.to(device), tgt_s.to(device)
    r_id_s, r_lvl_s, ln_s = r_id_s.to(device), r_lvl_s.to(device), ln_s.to(device)
    x_f_q, x_t_q, tgt_q = x_f_q.to(device), x_t_q.to(device), tgt_q.to(device)
    r_id_q, r_lvl_q, ln_q = r_id_q.to(device), r_lvl_q.to(device), ln_q.to(device)

    # 1. 保存原始参数
    original_params = {name: p.data.clone() for name, p in model.named_parameters()}

    # 2. Inner loop: FOMAML 更新 (在 Support set 上)
    pred_s, _, _ = model(x_f_s, x_t_s, r_id_s, r_lvl_s, ln_s)
    loss_s = criterion(pred_s, tgt_s)
    grads = torch.autograd.grad(loss_s, model.parameters(), create_graph=False)
    with torch.no_grad():
        for p, g in zip(model.parameters(), grads):
            p.data -= lr_inner * g

    # 3. Outer loop: 在 Query set 上计算损失并反向传播
    pred_q, _, _ = model(x_f_q, x_t_q, r_id_q, r_lvl_q, ln_q)
    loss_q = criterion(pred_q, tgt_q)

    # 4. 反向传播 (梯度 w.r.t. inner-updated params ≈ w.r.t. original, FOMAML)
    optimizer.zero_grad()
    loss_q.backward()

    # 5. 恢复原始参数 (p.data = θ, p.grad = ∇_θ' L(D_q))
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.data.copy_(original_params[name])

    # 6. 应用梯度到原始参数 (θ ← θ - γ·∇_θ' L(D_q))
    optimizer.step()

    return loss_q.item()


def maml_train(model, cluster_loaders, val_loader, config, log_prefix='Full'):
    """
    MAML 训练循环 (论文 Eq.13-16)
    - 每轮遍历所有簇
    - 每簇: Support set 做 inner loop, Query set 做 outer loop
    - 参数恢复确保簇间独立
    """
    print(f"\n>>> MAML Training (FOMAML, Support/Query Split)")
    optimizer = optim.Adam(model.parameters(), lr=config['lr_outer'])
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val = float('inf')
    patience_counter = 0
    log_file = open(f'log/train_{log_prefix}_maml.txt', 'w')

    for epoch in range(config['epochs']):
        t0 = time.time()
        model.train()
        epoch_loss, step_count = 0, 0

        for cluster_k, (support_loader, query_loader) in cluster_loaders.items():
            support_iter = iter(support_loader)
            query_iter = iter(query_loader)

            for step in range(config['max_batches_per_cluster']):
                try:
                    support_batch = next(support_iter)
                except StopIteration:
                    support_iter = iter(support_loader)
                    support_batch = next(support_iter)
                try:
                    query_batch = next(query_iter)
                except StopIteration:
                    query_iter = iter(query_loader)
                    query_batch = next(query_iter)

                loss = maml_step(model, support_batch, query_batch,
                                 optimizer, config['lr_inner'], criterion)
                epoch_loss += loss
                step_count += 1

        avg_train = epoch_loss / max(step_count, 1)

        # 验证
        model.eval()
        val_loss, v_count = 0, 0
        with torch.no_grad():
            for i, (x_f, x_t, r_id, r_lvl, ln, tgt) in enumerate(val_loader):
                if i >= config['max_batches_per_cluster']:
                    break
                x_f, x_t, tgt = x_f.to(device), x_t.to(device), tgt.to(device)
                r_id, r_lvl, ln = r_id.to(device), r_lvl.to(device), ln.to(device)
                pred, _, _ = model(x_f, x_t, r_id, r_lvl, ln)
                val_loss += criterion(pred, tgt).item()
                v_count += 1
        avg_val = val_loss / max(v_count, 1)

        train_losses.append(avg_train)
        val_losses.append(avg_val)

        msg = (f"Epoch {epoch+1}/{config['epochs']} | Train {avg_train:.6f} | "
               f"Val {avg_val:.6f} | {time.time()-t0:.1f}s")
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config['model_dir'], 'maml_base_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch+1} (patience={config['patience']})")
                break

    log_file.close()
    return train_losses, val_losses

# ============================================================
# 10. 簇微调 + 簇感知推理
# ============================================================
def fine_tune_clusters_maml(model, base_weights, train_dataset, road_cluster_labels,
                            road_indices, config):
    """
    簇微调: 对每个簇, 用 inner loop (FOMAML) 在簇数据上微调
    保存每簇的微调模型
    """
    print("\n>>> Cluster Fine-Tuning (MAML Inner Loop)")

    for k in range(config['num_clusters']):
        indices = [i for i, r_idx in enumerate(road_indices)
                   if road_cluster_labels[r_idx] == k]

        if len(indices) == 0:
            print(f"  Warning: No samples for cluster {k}, using base model.")
            m = copy.deepcopy(model)
            m.load_state_dict(base_weights)
            m.eval()
            torch.save(m.state_dict(), os.path.join(config['model_dir'], f'maml_model_cluster_{k}.pth'))
            continue

        cluster_subset = Subset(train_dataset, indices)
        cluster_loader = DataLoader(cluster_subset, batch_size=config['batch_size'], shuffle=True, num_workers=0)

        cluster_model = copy.deepcopy(model)
        cluster_model.load_state_dict(base_weights)

        # FOMAML inner loop 微调
        criterion = nn.MSELoss()
        cluster_model.train()
        for epoch in range(config['ft_epochs']):
            for x_f, x_t, r_id, r_lvl, ln, tgt in cluster_loader:
                x_f, x_t, tgt = x_f.to(device), x_t.to(device), tgt.to(device)
                r_id, r_lvl, ln = r_id.to(device), r_lvl.to(device), ln.to(device)

                pred, _, _ = cluster_model(x_f, x_t, r_id, r_lvl, ln)
                loss = criterion(pred, tgt)

                grads = torch.autograd.grad(loss, cluster_model.parameters(), create_graph=False)
                with torch.no_grad():
                    for p, g in zip(cluster_model.parameters(), grads):
                        p.data -= config['lr_inner'] * g

        torch.save(cluster_model.state_dict(),
                   os.path.join(config['model_dir'], f'maml_model_cluster_{k}.pth'))
        print(f"  Cluster {k} fine-tuned ({len(indices)} samples)")

    print("  All clusters fine-tuned.")


def test_cluster_aware(model, val_loader, train_dataset, road_cluster_labels, max_flow, config, log_prefix='Full'):
    """簇感知推理: 按簇路由到对应微调模型"""
    print(f"\n>>> Cluster-Aware Inference ({log_prefix})")

    cluster_models = {}
    for k in range(config['num_clusters']):
        path = os.path.join(config['model_dir'], f'maml_model_cluster_{k}.pth')
        m = copy.deepcopy(model)
        if os.path.exists(path):
            m.load_state_dict(torch.load(path, weights_only=True))
        else:
            m.load_state_dict(torch.load(os.path.join(config['model_dir'], 'maml_base_model.pth'), weights_only=True))
        m.eval()
        cluster_models[k] = m

    # road_index → cluster 映射
    idx_to_cluster = torch.tensor(road_cluster_labels, device=device, dtype=torch.long)

    all_preds, all_targets = [], []
    start_time = time.time()

    with torch.no_grad():
        for x_f, x_t, r_id, r_lvl, ln, tgt in val_loader:
            x_f, x_t = x_f.to(device), x_t.to(device)
            r_id, r_lvl, ln = r_id.to(device), r_lvl.to(device), ln.to(device)
            target = tgt.numpy()

            batch_clusters = idx_to_cluster[r_id]
            batch_preds = torch.zeros(x_f.size(0), config['look_forward'], device=device)

            for k in torch.unique(batch_clusters):
                k_val = k.item()
                mask = (batch_clusters == k_val)
                if mask.sum() == 0:
                    continue
                pred_k, _, _ = cluster_models[k_val](x_f[mask], x_t[mask],
                                                      r_id[mask], r_lvl[mask], ln[mask])
                batch_preds[mask] = pred_k

            all_preds.append((batch_preds * max_flow).cpu().numpy())
            all_targets.append(target * max_flow)

    test_duration = time.time() - start_time

    y_pred = np.concatenate(all_preds, axis=0).flatten()
    y_true = np.concatenate(all_targets, axis=0).flatten()

    # MAPE 过滤
    mask = y_true != 0
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean))
    r2 = r2_score(y_true_clean, y_pred_clean)

    print("-" * 30)
    print(f"MetaSTC-{log_prefix} Results (Full Paper Implementation):")
    print(f"Test Time:  {test_duration:.4f}s")
    print(f"Test MAE:   {mae:.4f}")
    print(f"Test RMSE:  {rmse:.4f}")
    print(f"Test MAPE:  {mape:.4f}")
    print(f"Test R2:    {r2:.4f}")
    print("-" * 30)

    metrics = {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}
    return y_true, y_pred, metrics

# ============================================================
# 11. 可视化
# ============================================================
def plot_loss(train_losses, val_losses, save_path='figure/Full_loss.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='#4C72B0')
    plt.plot(val_losses, label='Val Loss', color='#C44E52', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('MetaSTC Full (MAML) Training Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to {save_path}")


def plot_prediction(y_true, y_pred, metrics, save_path='figure/Full_prediction.png'):
    plt.figure(figsize=(15, 6))
    n_show = min(len(y_true), 300)
    plt.plot(y_true[:n_show], label='Ground Truth', color='#4C72B0')
    plt.plot(y_pred[:n_show], label='Prediction', color='#55A868', linestyle='--')
    plt.title(f"MetaSTC Full (MAML+K-Means++) | MAE={metrics['mae']:.4f} "
              f"RMSE={metrics['rmse']:.4f} MAPE={metrics['mape']:.4f} R2={metrics['r2']:.4f}")
    plt.xlabel('Sample')
    plt.ylabel('Traffic Flow')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Prediction plot saved to {save_path}")


def plot_cluster_alpha(model, val_loader, save_path='figure/Full_cluster_alpha.png', max_batches=3):
    model.eval()
    all_alpha = []
    with torch.no_grad():
        for i, (x_f, x_t, r_id, r_lvl, ln, tgt) in enumerate(val_loader):
            if i >= max_batches:
                break
            x_f, x_t = x_f.to(device), x_t.to(device)
            r_id, r_lvl, ln = r_id.to(device), r_lvl.to(device), ln.to(device)
            _, alpha, _ = model(x_f, x_t, r_id, r_lvl, ln)
            all_alpha.append(alpha.cpu().numpy())

    alpha_mat = np.concatenate(all_alpha, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    n_show = min(alpha_mat.shape[0], 100)
    im = axes[0].imshow(alpha_mat[:n_show].T, aspect='auto', cmap='YlOrRd')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Cluster k')
    axes[0].set_title('Cluster Membership α (MAML)')
    plt.colorbar(im, ax=axes[0])

    labels = np.argmax(alpha_mat, axis=1)
    counts = np.bincount(labels, minlength=alpha_mat.shape[1])
    axes[1].bar(range(len(counts)), counts, color='#4C72B0')
    axes[1].set_xlabel('Cluster k')
    axes[1].set_ylabel('Sample Count')
    axes[1].set_title('Cluster Size Distribution')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Cluster α heatmap saved to {save_path}")

# ============================================================
# 12. 主入口: 论文完整复现
# ============================================================
def run_paper_full(look_back=12):
    """论文完整实现: K-Means++ (on h) + MAML + Cluster-Aware Inference"""
    tag = f'Full_L{look_back}'
    config = {
        'batch_size': 64,                 # 论文配置
        'epochs': 50,                     # 实际轮数 (论文 1000, 此处用 50 + 早停)
        'ft_epochs': 3,                   # 簇微调轮数
        'lr_outer': 0.001,                # 全局学习率 γ (论文)
        'lr_inner': 0.01,                 # 内循环学习率 β (论文)
        'patience': 5,                    # 早停
        'split_ratio': 0.8,               # 8:2 (项目硬约束)
        'pkl_path': 'data/traffic_flow/1/20230306/part-00000.pkl',
        'feature_path': 'data/link_feature.csv',
        'max_batches_per_cluster': 40,    # 每簇每轮 batch 数 (5簇×40=200, 项目硬约束)
        'pretrain_epochs': 10,            # 预训练轮数 (为提取 h)
        'model_dir': f'param/metastc_{tag}',
        'num_clusters': 5,                # K=5 (项目硬约束)
        'look_back': look_back,
        'look_forward': 6,
    }
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs('figure', exist_ok=True)
    os.makedirs('log', exist_ok=True)

    # (1) 数据
    print(f"\n{'='*60}")
    print(f">>> MetaSTC-{tag} (Full Paper Implementation)")
    print(f"    K-Means++ (on h) + MAML (FOMAML) + Cluster-Aware Inference")
    print(f"{'='*60}")

    with open(config['pkl_path'], 'rb') as f:
        raw_data = pickle.load(f)
        time_len = len(raw_data[0]['flow'])
    train_end = int(time_len * config['split_ratio'])

    train_ds = TrafficDataset(config['pkl_path'], config['feature_path'],
                              look_back=config['look_back'], look_forward=config['look_forward'],
                              time_range=(0, train_end))
    max_flow = train_ds.max_flow
    val_ds = TrafficDataset(config['pkl_path'], config['feature_path'],
                            look_back=config['look_back'], look_forward=config['look_forward'],
                            time_range=(train_end, time_len), max_flow_override=max_flow)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    # (2) 模型
    model = MetaSTC(
        num_roads=train_ds.num_roads,
        num_levels=train_ds.num_levels,
        num_lanes=train_ds.num_lanes,
        input_flow_dim=1,
        spatial_embed_dim=16,
        temporal_hidden_dim=32,
        num_clusters=config['num_clusters'],
        mem_dim=64,
        output_seq_len=config['look_forward']
    ).to(device)

    # (3) Step 1: 预训练编码器 → 提取 h → K-Means++ 聚类
    pretrain_encoder(model, train_loader, epochs=config['pretrain_epochs'], lr=config['lr_outer'])
    h_matrix = extract_representations(model, train_ds)
    road_cluster_labels = perform_clustering_on_h(h_matrix, k=config['num_clusters'])

    # (4) Step 2: Support/Query 划分
    print("\n>>> Support/Query Split (per cluster, 50:50)")
    cluster_loaders = split_support_query(train_ds, road_cluster_labels,
                                           train_ds.road_indices,
                                           config['num_clusters'],
                                           config['batch_size'])

    # (5) Step 3: MAML 训练
    train_losses, val_losses = maml_train(model, cluster_loaders, val_loader, config, log_prefix=tag)

    # (6) Step 4: 簇微调 (MAML inner loop per cluster)
    base_weights = torch.load(os.path.join(config['model_dir'], 'maml_base_model.pth'), weights_only=True)
    fine_tune_clusters_maml(model, base_weights, train_ds, road_cluster_labels,
                            train_ds.road_indices, config)

    # (7) Step 5: 簇感知推理
    y_true, y_pred, metrics = test_cluster_aware(model, val_loader, train_ds,
                                                  road_cluster_labels, max_flow, config, log_prefix=tag)

    # (8) 可视化
    plot_loss(train_losses, val_losses, f'figure/{tag}_loss.png')
    plot_prediction(y_true, y_pred, metrics, f'figure/{tag}_prediction.png')
    model.load_state_dict(base_weights)
    plot_cluster_alpha(model, val_loader, f'figure/{tag}_cluster_alpha.png')

    print(f"\n>>> Done. Outputs: figure/{tag}_*.png")
    return metrics

if __name__ == "__main__":
    for L in [12, 24]:
        run_paper_full(look_back=L)
