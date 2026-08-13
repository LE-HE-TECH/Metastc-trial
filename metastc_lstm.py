import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

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
# 1. Traffic2Vec (Temporal Encoder)
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
# 2. Spatial Encoder
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
# 3. Spatio-Temporal Gating Unit
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
# 4. Cluster-Aware Memory Unit (Revised)
# ==========================================
class ClusterMemoryUnit(nn.Module):
    def __init__(self, hidden_dim, num_clusters, mem_dim, temperature=0.5):
        super(ClusterMemoryUnit, self).__init__()
        self.num_clusters = num_clusters
        self.mem_dim = mem_dim
        self.temperature = temperature  # [修改] 控制聚类的尖锐程度
        
        self.centroids = nn.Parameter(torch.randn(num_clusters, hidden_dim))
        self.memory = nn.Parameter(torch.randn(num_clusters, mem_dim))
        
        self.W_r = nn.Linear(mem_dim, mem_dim)
        self.W_u = nn.Linear(mem_dim, mem_dim)
        self.W_c = nn.Linear(mem_dim, mem_dim)

    def get_similarity(self, h_j):
        scores = torch.matmul(h_j, self.centroids.t())
        # [修改] 使用 temperature 锐化 softmax，使其更接近硬聚类
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
# 5. Prediction Backbone (Major Fix)
# ==========================================
class MetaBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, mem_dim):
        super(MetaBackbone, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # [关键修改] 移除不稳定的 Hypernetwork (Weight Generation)
        # 改为 Context Concatenation 策略，更加稳定
        self.hidden_dim = hidden_dim
        
        # 输入：LSTM输出 (hidden_dim) + Memory Context (mem_dim)
        # 输出：预测值 (output_dim)
        self.final_predictor = nn.Sequential(
            nn.Linear(hidden_dim + mem_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x_seq, memory_context):
        # 1. Standard LSTM
        lstm_out, _ = self.lstm(x_seq)
        last_hidden = lstm_out[:, -1, :] # [Batch, Hidden]
        
        # 2. Concat Context (Explicit Conditioning)
        # memory_context: [Batch, Mem_Dim]
        combined = torch.cat([last_hidden, memory_context], dim=-1)
        
        # 3. Predict
        prediction = self.final_predictor(combined)
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
        self.memory_unit = ClusterMemoryUnit(hidden_dim=fused_dim, num_clusters=num_clusters, mem_dim=mem_dim)
        
        # 使用修正后的 Backbone
        self.backbone = MetaBackbone(input_dim=fused_dim, hidden_dim=64, output_dim=output_seq_len, mem_dim=mem_dim)

    def forward(self, x_flow, x_time, road_idx, road_level, lane_num):
        h_t = self.traffic2vec(x_flow, x_time) 
        h_s = self.spatial_encoder(road_idx, road_level, lane_num)
        h_seq, h_pooled = self.gating(h_s, h_t)
        memory_context, cluster_attn = self.memory_unit(h_pooled)
        prediction = self.backbone(h_seq, memory_context)
        return prediction, cluster_attn

# ==========================================
# 7. Data Loading (保持不变，确保完整性)
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