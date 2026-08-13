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
# MetaSTC Full-C: 论文完整实现 + 显式端到端聚类损失
# ------------------------------------------------------------
# 架构: Traffic2Vec + SpatialEncoder + Gating + ClusterMemoryUnit + Hypernetwork
# 聚类: K-Means++ 在时空表示 h 上 (论文 Algorithm 1)
# 训练: MAML (FOMAML) + Support/Query 划分 + cluster_loss (L_compact + L_sep + L_balance)
#       - inner loop:  L = L_pred(D_s) + λ·L_cluster(D_s)
#       - outer loop:  L = L_pred(D_q) + λ·L_cluster(D_q)
# 推理: 簇感知 (每簇独立微调模型)
# 消融目的: 在 Full 基础上加入端到端聚类损失, 验证聚类损失对 MAML 训练的影响
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
# 6. Main MetaSTC Framework (Full-C: 4-tuple forward)
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
        # [Full-C 修改] 返回 centroids 供 cluster_loss 使用
        centroids = self.memory_unit.centroids  # [K, D]
        return prediction, cluster_attn, centroids, h_pooled

    def get_representation(self, x_flow, x_time, road_idx, road_level, lane_num):
        """仅提取 h_pooled (用于 K-Means++ 聚类)"""
        h_t = self.traffic2vec(x_flow, x_time)
        h_s = self.spatial_encoder(road_idx, road_level, lane_num)
        _, h_pooled = self.gating(h_s, h_t)
        return h_pooled

# ==========================================
# 7. End-to-End Clustering Loss (从 C 版本移植)
#    h_pooled: [B, D], alpha: [B, K], centroids: [K, D]
# ==========================================
def cluster_loss(h_pooled, alpha, centroids, num_clusters,
                 lambda_compact=1.0, lambda_sep=1.0, lambda_balance=0.1):
    """端到端聚类损失 —— 全部分量有界，防止梯度爆炸
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

    # (3) 均衡性: 最大化簇分布熵（防塌缩）
    # [修复] 不再 detach α，让 L_balance 真正反传，对抗 L_compact 的塌缩倾向
    # lambda_balance=0.1 << lambda_compact=1.0，仅作弱约束防极端塌缩
    p_k = alpha.detach().mean(dim=0)                                             # [K], 保留梯度
    entropy = -(p_k * torch.log(p_k + 1e-10)).sum()
    L_balance = -entropy                                                # ∈ [-log K, 0]

    total = lambda_compact * L_compact + lambda_sep * L_sep + lambda_balance * L_balance
    return total, L_compact, L_sep, L_balance

# ==========================================
# 8. Data Loading (增加 road_indices 供簇划分)
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
# 9. K-Means++ 聚类 (在时空表示 h 上)
# 论文 Algorithm 1: 聚类输入是 h, 不是原始特征
# ============================================================
def pretrain_encoder(model, train_loader, epochs=10, lr=0.001):
    """预训练编码器以获得有意义的 h 表示 (预训练不加 cluster_loss, 因为聚类还没做)"""
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
            pred, _, _, _ = model(x_f, x_t, r_id, r_lvl, ln)  # [Full-C] 4-tuple
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

    road_h_map = {}

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
    return labels

# ============================================================
# 10. MAML 训练 (FOMAML + Support/Query + 参数恢复 + cluster_loss)
# 论文 Eq.13-16
# ============================================================
def split_support_query(dataset, road_cluster_labels, road_indices, num_clusters, batch_size=64):
    """按簇划分数据, 每簇内 50:50 分为 Support 和 Query"""
    cluster_loaders = {}

    for k in range(num_clusters):
        indices = [i for i, r_idx in enumerate(road_indices)
                   if road_cluster_labels[r_idx] == k]

        if len(indices) < 2:
            print(f"  Warning: Cluster {k} has only {len(indices)} samples, skipping.")
            continue

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


def maml_step(model, support_batch, query_batch, optimizer, lr_inner, criterion,
              num_clusters, lambda_cls):
    """[Full-C 修改] FOMAML 单步训练 + cluster_loss
    1. 保存原始参数 θ
    2. Inner loop: θ' = θ - β·∇[L_pred(D_s) + λ·L_cluster(D_s)]  (FOMAML)
    3. Outer loop: 计算 [L_pred(D_q) + λ·L_cluster(D_q)] w.r.t. θ', 反向传播
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

    # 2. Inner loop: FOMAML 更新 (Support set, 含 cluster_loss)
    pred_s, alpha_s, centroids_s, h_s = model(x_f_s, x_t_s, r_id_s, r_lvl_s, ln_s)
    loss_pred_s = criterion(pred_s, tgt_s)
    loss_cls_s, _, _, _ = cluster_loss(h_s, alpha_s, centroids_s, num_clusters)
    loss_s = loss_pred_s + lambda_cls * loss_cls_s  # [Full-C] 联合损失

    grads = torch.autograd.grad(loss_s, model.parameters(), create_graph=False)
    with torch.no_grad():
        for p, g in zip(model.parameters(), grads):
            p.data -= lr_inner * g

    # 3. Outer loop: Query set (含 cluster_loss)
    pred_q, alpha_q, centroids_q, h_q = model(x_f_q, x_t_q, r_id_q, r_lvl_q, ln_q)
    loss_pred_q = criterion(pred_q, tgt_q)
    loss_cls_q, lc_q, ls_q, lb_q = cluster_loss(h_q, alpha_q, centroids_q, num_clusters)
    loss_q = loss_pred_q + lambda_cls * loss_cls_q  # [Full-C] 联合损失

    # 4. 反向传播
    optimizer.zero_grad()
    loss_q.backward()

    # 5. 恢复原始参数
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.data.copy_(original_params[name])

    # 6. 应用梯度到原始参数
    optimizer.step()

    return loss_pred_q.item(), lc_q.item(), ls_q.item(), lb_q.item()


def maml_train(model, cluster_loaders, val_loader, config, log_prefix='FullC'):
    """[Full-C 修改] MAML 训练循环 + cluster_loss 分量记录"""
    print(f"\n>>> MAML Training (FOMAML + cluster_loss, Support/Query Split)")
    optimizer = optim.Adam(model.parameters(), lr=config['lr_outer'])
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    cls_losses = {'compact': [], 'sep': [], 'balance': []}
    best_val = float('inf')
    patience_counter = 0
    os.makedirs('log', exist_ok=True)
    log_file = open(f'log/train_{log_prefix}_maml.txt', 'w')

    for epoch in range(config['epochs']):
        t0 = time.time()
        model.train()
        epoch_loss, step_count = 0, 0
        ep_compact = ep_sep = ep_balance = 0.0

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

                loss_pred, lc, ls, lb = maml_step(
                    model, support_batch, query_batch,
                    optimizer, config['lr_inner'], criterion,
                    config['num_clusters'], config['lambda_cls']
                )
                epoch_loss += loss_pred
                ep_compact += lc; ep_sep += ls; ep_balance += lb
                step_count += 1

        avg_train = epoch_loss / max(step_count, 1)
        avg_compact = ep_compact / max(step_count, 1)
        avg_sep = ep_sep / max(step_count, 1)
        avg_balance = ep_balance / max(step_count, 1)

        # 验证
        model.eval()
        val_loss, v_count = 0, 0
        with torch.no_grad():
            for i, (x_f, x_t, r_id, r_lvl, ln, tgt) in enumerate(val_loader):
                if i >= config['max_batches_per_cluster']:
                    break
                x_f, x_t, tgt = x_f.to(device), x_t.to(device), tgt.to(device)
                r_id, r_lvl, ln = r_id.to(device), r_lvl.to(device), ln.to(device)
                pred, _, _, _ = model(x_f, x_t, r_id, r_lvl, ln)  # [Full-C] 4-tuple
                val_loss += criterion(pred, tgt).item()
                v_count += 1
        avg_val = val_loss / max(v_count, 1)

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        cls_losses['compact'].append(avg_compact)
        cls_losses['sep'].append(avg_sep)
        cls_losses['balance'].append(avg_balance)

        msg = (f"Epoch {epoch+1}/{config['epochs']} | Train {avg_train:.6f} | "
               f"Val {avg_val:.6f} | compact {avg_compact:.4f} | sep {avg_sep:.4f} | "
               f"bal {avg_balance:.4f} | {time.time()-t0:.1f}s")
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
    return train_losses, val_losses, cls_losses

# ============================================================
# 11. 簇微调 (含 cluster_loss) + 簇感知推理
# ============================================================
def fine_tune_clusters_maml(model, base_weights, train_dataset, road_cluster_labels,
                            road_indices, config):
    """[Full-C 修改] 簇微调: FOMAML inner loop + cluster_loss"""
    print("\n>>> Cluster Fine-Tuning (MAML Inner Loop + cluster_loss)")

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

        criterion = nn.MSELoss()
        cluster_model.train()
        for epoch in range(config['ft_epochs']):
            for x_f, x_t, r_id, r_lvl, ln, tgt in cluster_loader:
                x_f, x_t, tgt = x_f.to(device), x_t.to(device), tgt.to(device)
                r_id, r_lvl, ln = r_id.to(device), r_lvl.to(device), ln.to(device)

                pred, alpha, centroids, h_pooled = cluster_model(x_f, x_t, r_id, r_lvl, ln)
                loss_pred = criterion(pred, tgt)
                loss_cls, _, _, _ = cluster_loss(h_pooled, alpha, centroids, config['num_clusters'])
                loss = loss_pred + config['lambda_cls'] * loss_cls  # [Full-C] 联合损失

                grads = torch.autograd.grad(loss, cluster_model.parameters(), create_graph=False)
                with torch.no_grad():
                    for p, g in zip(cluster_model.parameters(), grads):
                        p.data -= config['lr_inner'] * g

        torch.save(cluster_model.state_dict(),
                   os.path.join(config['model_dir'], f'maml_model_cluster_{k}.pth'))
        print(f"  Cluster {k} fine-tuned ({len(indices)} samples)")

    print("  All clusters fine-tuned.")


def test_cluster_aware(model, val_loader, train_dataset, road_cluster_labels, max_flow, config, log_prefix='FullC'):
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
                pred_k, _, _, _ = cluster_models[k_val](x_f[mask], x_t[mask],
                                                         r_id[mask], r_lvl[mask], ln[mask])  # [Full-C] 4-tuple
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
    print(f"MetaSTC-{log_prefix} Results (Full + cluster_loss):")
    print(f"Test Time:  {test_duration:.4f}s")
    print(f"Test MAE:   {mae:.4f}")
    print(f"Test RMSE:  {rmse:.4f}")
    print(f"Test MAPE:  {mape:.4f}")
    print(f"Test R2:    {r2:.4f}")
    print("-" * 30)

    metrics = {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}
    return y_true, y_pred, metrics

# ============================================================
# 12. 可视化
# ============================================================
def plot_loss(train_losses, val_losses, cls_losses, save_path='figure/FullC_loss.png'):
    """[Full-C] 双图: 预测损失 + 聚类损失分量"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    # 左: 预测损失
    axes[0].plot(train_losses, label='Train', color='#4C72B0')
    axes[0].plot(val_losses, label='Val', color='#C44E52', linestyle='--')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('MetaSTC Full-C (MAML) Prediction Loss')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    # 右: 聚类损失分量
    axes[1].plot(cls_losses['compact'], label='L_compact (↓)', color='#4C72B0')
    axes[1].plot(cls_losses['sep'], label='L_sep (↑→-)', color='#55A868')
    axes[1].plot(cls_losses['balance'], label='L_balance (↑→-)', color='#C44E52')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Cluster Loss')
    axes[1].set_title('Cluster Loss Components (MAML outer loop)')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"Loss curves saved to {save_path}")


def plot_prediction(y_true, y_pred, metrics, save_path='figure/FullC_prediction.png'):
    plt.figure(figsize=(15, 6))
    n_show = min(len(y_true), 300)
    plt.plot(y_true[:n_show], label='Ground Truth', color='#4C72B0')
    plt.plot(y_pred[:n_show], label='Prediction', color='#55A868', linestyle='--')
    plt.title(f"MetaSTC Full-C (MAML+K-Means++ + cluster_loss) | "
              f"MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} "
              f"MAPE={metrics['mape']:.4f} R2={metrics['r2']:.4f}")
    plt.xlabel('Sample')
    plt.ylabel('Traffic Flow')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Prediction plot saved to {save_path}")


def plot_cluster_alpha(model, val_loader, save_path='figure/FullC_cluster_alpha.png', max_batches=3):
    model.eval()
    all_alpha = []
    with torch.no_grad():
        for i, (x_f, x_t, r_id, r_lvl, ln, tgt) in enumerate(val_loader):
            if i >= max_batches:
                break
            x_f, x_t = x_f.to(device), x_t.to(device)
            r_id, r_lvl, ln = r_id.to(device), r_lvl.to(device), ln.to(device)
            _, alpha, _, _ = model(x_f, x_t, r_id, r_lvl, ln)  # [Full-C] 4-tuple
            all_alpha.append(alpha.cpu().numpy())

    alpha_mat = np.concatenate(all_alpha, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    n_show = min(alpha_mat.shape[0], 100)
    im = axes[0].imshow(alpha_mat[:n_show].T, aspect='auto', cmap='YlOrRd')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Cluster k')
    axes[0].set_title('Cluster Membership α (MAML + cluster_loss)')
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

    p_k = alpha_mat.mean(axis=0)
    entropy = -np.sum(p_k * np.log(p_k + 1e-10))
    print(f"Cluster distribution entropy: {entropy:.4f} (max={np.log(alpha_mat.shape[1]):.4f})")

# ============================================================
# 13. 主入口: 论文完整复现 + 端到端聚类损失
# ============================================================
def run_paper_full_c(look_back=12):
    """Full-C: K-Means++ (on h) + MAML + cluster_loss + Cluster-Aware Inference"""
    tag = f'FullC_L{look_back}'
    config = {
        'batch_size': 64,
        'epochs': 50,
        'ft_epochs': 3,
        'lr_outer': 0.001,
        'lr_inner': 0.01,
        'patience': 5,
        'split_ratio': 0.8,
        'pkl_path': 'data/traffic_flow/1/20230306/part-00000.pkl',
        'feature_path': 'data/link_feature.csv',
        'max_batches_per_cluster': 40,    # 5簇×40=200 batches/epoch (项目硬约束)
        'pretrain_epochs': 10,
        'model_dir': f'param/metastc_{tag}',
        'num_clusters': 5,                # K=5 (项目硬约束)
        'lambda_cls': 0.01,               # cluster_loss 权重 (从 C 版本移植)
        'look_back': look_back,
        'look_forward': 6,
    }
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs('figure', exist_ok=True)
    os.makedirs('log', exist_ok=True)

    # (1) 数据
    print(f"\n{'='*60}")
    print(f">>> MetaSTC-{tag} (Full + cluster_loss)")
    print(f"    K-Means++ (on h) + MAML (FOMAML) + cluster_loss + Cluster-Aware Inference")
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

    # (5) Step 3: MAML 训练 (含 cluster_loss)
    train_losses, val_losses, cls_losses = maml_train(
        model, cluster_loaders, val_loader, config, log_prefix=tag
    )

    # (6) Step 4: 簇微调 (MAML inner loop per cluster, 含 cluster_loss)
    base_weights = torch.load(os.path.join(config['model_dir'], 'maml_base_model.pth'), weights_only=True)
    fine_tune_clusters_maml(model, base_weights, train_ds, road_cluster_labels,
                            train_ds.road_indices, config)

    # (7) Step 5: 簇感知推理
    y_true, y_pred, metrics = test_cluster_aware(model, val_loader, train_ds,
                                                  road_cluster_labels, max_flow, config, log_prefix=tag)

    # (8) 可视化
    plot_loss(train_losses, val_losses, cls_losses, f'figure/{tag}_loss.png')
    plot_prediction(y_true, y_pred, metrics, f'figure/{tag}_prediction.png')
    model.load_state_dict(base_weights)
    plot_cluster_alpha(model, val_loader, f'figure/{tag}_cluster_alpha.png')

    print(f"\n>>> Done. Outputs: figure/{tag}_*.png")
    return metrics

if __name__ == "__main__":
    for L in [12, 24]:
        run_paper_full_c(look_back=L)
