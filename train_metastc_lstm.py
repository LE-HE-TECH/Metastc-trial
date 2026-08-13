import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import os
import time
import pickle
import copy
import matplotlib.pyplot as plt
import argparse

# 导入模型
from metastc_lstm import MetaSTC, TrafficDataset, get_device

# ==========================================
# 0. 解析参数
# ==========================================
parser = argparse.ArgumentParser(description='MetaSTC with Explicit Clustering')
parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='Operation mode')
args = parser.parse_args()

# ==========================================
# 1. 配置参数
# ==========================================
CONFIG = {
    "batch_size": 128,
    "epochs": 50,           # Global training epochs
    "ft_epochs": 10,        # Fine-tuning epochs per cluster
    "lr": 0.001,
    "ft_lr": 0.0005,        # Fine-tuning learning rate (smaller)
    "patience": 20,
    "split_ratio": 0.8,
    "device": get_device(),
    "pkl_path": "data/traffic_flow/1/20230306/part-00000.pkl",
    "feature_path": "data/link_feature.csv",
    "max_batches": 50,      # 增加 batch 数以确保覆盖足够多的路段
    "model_dir": "param/metastc_hybrid",
    "num_clusters": 5,      # K=5
    "look_back": 12,
    "look_forward": 6
}

os.makedirs(CONFIG["model_dir"], exist_ok=True)
os.makedirs('figure', exist_ok=True)

# ==========================================
# 2. 辅助函数：K-Means 聚类 (源自 meta-LSTM.py)
# ==========================================
def perform_clustering(pkl_path, feature_path, k=5):
    print("正在执行 K-Means 聚类...")
    # 1. 加载流量数据
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    id_list = [d['id'] for d in data]
    data_flow = np.array([d['flow'] for d in data])
    
    # 2. 加载特征数据
    feature = pd.read_csv(feature_path)
    
    # 3. 对齐数据
    valid_indices = [i for i, x in enumerate(id_list) if x in feature['link_ID'].values]
    data_flow = data_flow[valid_indices]
    id_list = [id_list[i] for i in valid_indices]
    
    feature_road = feature[feature['link_ID'].isin(id_list)]
    feature_road = feature_road.set_index('link_ID').loc[id_list].reset_index()
    
    # 4. 准备聚类特征
    # (a) 静态特征归一化
    feature_used = feature_road.drop(columns=['link_ID', 'Kind', 'geometry'], errors='ignore')
    # 处理可能存在的非数值列
    feature_used = feature_used.select_dtypes(include=[np.number])
    feature_used = feature_used.fillna(0).values
    
    for i in range(feature_used.shape[1]):
        max_val = np.max(feature_used[:, i])
        if max_val > 0:
            feature_used[:, i] = feature_used[:, i] / max_val
            
    # (b) 动态特征 (历史均值)
    # 按 meta-LSTM 逻辑：计算每个小时(或其他周期)的均值，这里简化为每 look_back 步的均值
    # 为了保持一致性，我们取 data_flow 的前 80% 计算均值
    train_len = int(data_flow.shape[1] * 0.8)
    train_flow = data_flow[:, :train_len]
    
    # 将时间轴压缩为 12 个特征 (模拟 meta-LSTM 的处理)
    # meta-LSTM 中是: np.stack([np.mean(..., axis=1) for j in range(12)]).T
    # 这里我们简化处理：将一天分为 12 个时段取均值
    steps_per_slot = train_flow.shape[1] // 12
    flow_features = []
    for i in range(12):
        start = i * steps_per_slot
        end = (i + 1) * steps_per_slot
        if start >= train_flow.shape[1]: break
        flow_features.append(np.mean(train_flow[:, start:end], axis=1))
    
    if len(flow_features) > 0:
        flow_features = np.stack(flow_features).T
        # 归一化
        flow_features = flow_features / (np.max(flow_features) + 1e-10)
        combined_features = np.concatenate((feature_used, flow_features), axis=1)
    else:
        combined_features = feature_used

    # 5. K-Means
    # 简单的手写 K-Means 或使用 sklearn (这里用 sklearn 以保证稳定性)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(combined_features)
    
    # 建立映射: road_id -> cluster_label
    road_cluster_map = {rid: label for rid, label in zip(id_list, labels)}
    
    print(f"聚类完成。各类别数量: {np.bincount(labels)}")
    return road_cluster_map

# ==========================================
# 3. 准备数据与模型
# ==========================================
# 获取聚类标签
road_cluster_map = perform_clustering(CONFIG["pkl_path"], CONFIG["feature_path"], k=CONFIG["num_clusters"])

with open(CONFIG["pkl_path"], 'rb') as f:
    raw_data = pickle.load(f)
    time_len = len(raw_data[0]['flow'])

train_end_idx = int(time_len * CONFIG["split_ratio"])

# 加载完整数据集
full_train_dataset = TrafficDataset(CONFIG["pkl_path"], CONFIG["feature_path"], time_range=(0, train_end_idx))
max_flow = full_train_dataset.max_flow
full_val_dataset = TrafficDataset(CONFIG["pkl_path"], CONFIG["feature_path"], time_range=(train_end_idx, time_len), max_flow_override=max_flow)

# 创建 DataLoader
train_loader = DataLoader(full_train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
# 验证集不 shuffle，方便后续可视化
val_loader = DataLoader(full_val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

device = CONFIG["device"]
# 初始化 Base Model
model = MetaSTC(
    num_roads=full_train_dataset.num_roads,
    num_levels=full_train_dataset.num_levels,
    num_lanes=full_train_dataset.num_lanes,
    input_flow_dim=1,
    spatial_embed_dim=16,
    temporal_hidden_dim=32,
    num_clusters=CONFIG["num_clusters"], # 这里仍保留内部 Soft Cluster 能力
    mem_dim=64,
    output_seq_len=6
).to(device)

criterion = nn.MSELoss()
# Global Optimizer
optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

# ==========================================
# 4. 阶段一：Global Meta-Training
# ==========================================
def train_global():
    print("\n>>> Phase 1: Global Meta-Training (Base Model)")
    best_loss = float('inf')
    
    # 如果已经有训练好的模型，可以选择跳过或加载继续训练
    # model.load_state_dict(torch.load(os.path.join(CONFIG["model_dir"], "base_model.pth")))
    
    for epoch in range(CONFIG["epochs"]):
        start_time = time.time()
        model.train()
        total_loss = 0
        count = 0
        
        for i, (x_f, x_t, r_id, r_lvl, ln, target) in enumerate(train_loader):
            if i >= CONFIG["max_batches"]: break
            x_f, x_t, target = x_f.to(device), x_t.to(device), target.to(device)
            r_id, r_lvl, ln = r_id.to(device), r_lvl.to(device), ln.to(device)
            
            optimizer.zero_grad()
            # MAML 风格：这里我们简化为标准预训练，因为后续有 Explicit Fine-tuning
            # 如果想保留 MAML，可以这里加 inner loop，但 Explicit Fine-tuning 通常足够强
            pred, _ = model(x_f, x_t, r_id, r_lvl, ln)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            count += 1
            
        avg_loss = total_loss / count
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {avg_loss:.6f} | Time: {time.time()-start_time:.1f}s")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(CONFIG["model_dir"], "base_model.pth"))

# ==========================================
# 5. 阶段二：Cluster-Specific Fine-Tuning
# ==========================================
def fine_tune_clusters():
    print("\n>>> Phase 2: Cluster-Specific Fine-Tuning (Freezing Encoders)")
    
    # 1. 加载最佳 Base Model 并准备映射
    base_weights = torch.load(os.path.join(CONFIG["model_dir"], "base_model.pth"), weights_only=True)
    
    # 建立 road_index -> cluster 的映射
    road_idx_to_cluster = np.zeros(full_train_dataset.num_roads, dtype=int)
    for rid, idx in full_train_dataset.road_map.items():
        road_idx_to_cluster[idx] = road_cluster_map.get(rid, 0)
    
    for k in range(CONFIG["num_clusters"]):
        print(f"--- Fine-tuning Cluster {k} ---")
        
        # (a) 准备该 Cluster 的数据
        # 找出 dataset 中所有属于 cluster k 的样本索引
        indices = [i for i, r_idx in enumerate(full_train_dataset.road_indices) 
                   if road_idx_to_cluster[r_idx] == k]
        
        if len(indices) == 0:
            print(f"Warning: No samples found for cluster {k}, skipping.")
            continue
            
        cluster_subset = Subset(full_train_dataset, indices)
        cluster_loader = DataLoader(cluster_subset, batch_size=CONFIG["batch_size"], shuffle=True)
        
        # (b) 初始化模型
        cluster_model = copy.deepcopy(model)
        cluster_model.load_state_dict(base_weights)
        
        # [关键修改] 冻结底层参数 (Feature Extractors & Memory)
        for name, param in cluster_model.named_parameters():
            param.requires_grad = False # 默认全锁
            
        # [关键修改] 只解锁最后的输出层 (Backbone 中的 predictor)
        # 根据 metastc_lstm.py 的结构，最后的层在 backbone.final_predictor
        for name, param in cluster_model.backbone.final_predictor.named_parameters():
            param.requires_grad = True
            # print(f"Unfrozen: {name}") # 调试用
            
        # 也可以选择解锁 LSTM 部分，视情况而定：
        # for name, param in cluster_model.backbone.lstm.named_parameters():
        #     param.requires_grad = True

        # [关键修改] 优化器只更新 requires_grad=True 的参数
        # 且使用更小的学习率 (1e-4 或 5e-5)，避免破坏
        cluster_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, cluster_model.parameters()), 
            lr=CONFIG["ft_lr"]
        )
        
        # (c) 微调 (减少 Epoch，例如改为 3-5 轮足够了)
        cluster_model.train()
        for epoch in range(3): # 降低轮数
            total_loss = 0
            steps = 0
            for x_f, x_t, r_id, r_lvl, ln, target in cluster_loader:
                x_f, x_t, target = x_f.to(device), x_t.to(device), target.to(device)
                r_id, r_lvl, ln = r_id.to(device), r_lvl.to(device), ln.to(device)
                
                cluster_optimizer.zero_grad()
                pred, _ = cluster_model(x_f, x_t, r_id, r_lvl, ln)
                loss = criterion(pred, target)
                loss.backward()
                cluster_optimizer.step()
                total_loss += loss.item()
                steps += 1
            # print(f"  Ep {epoch+1}: Loss {total_loss/steps:.6f}")
            
        torch.save(cluster_model.state_dict(), os.path.join(CONFIG["model_dir"], f"model_cluster_{k}.pth"))

# ==========================================
# 6. 阶段三：Cluster-Aware Inference
# ==========================================
def test_hybrid():
    print("\n>>> Phase 3: Cluster-Aware Inference")
    
    # 1. 预加载所有 K 个模型到内存 (或显存，如果够大)
    cluster_models = {}
    for k in range(CONFIG["num_clusters"]):
        path = os.path.join(CONFIG["model_dir"], f"model_cluster_{k}.pth")
        if os.path.exists(path):
            m = copy.deepcopy(model)
            m.load_state_dict(torch.load(path, weights_only=True))
            m.eval()
            cluster_models[k] = m
        else:
            print(f"Warning: Model for cluster {k} not found, using Base Model.")
            m = copy.deepcopy(model)
            m.load_state_dict(torch.load(os.path.join(CONFIG["model_dir"], "base_model.pth"), weights_only=True))
            m.eval()
            cluster_models[k] = m

    all_preds = []
    all_targets = []
    
    start_time = time.time()
    
    # 2. 遍历测试集
    # 为了效率，我们按 Batch 处理，但一个 Batch 可能包含不同 Cluster 的路
    # 简单策略：将 Batch 按 Cluster 拆分，分别送入对应模型，再拼回来
    # 高效策略：Test Loader 最好也是 shuffle=False，且最好按 road 排序
    
    with torch.no_grad():
        for batch_idx, (x_f, x_t, r_id, r_lvl, ln, target) in enumerate(val_loader):
            if batch_idx >= CONFIG["max_batches"]: break # 测试部分样本
            
            x_f, x_t = x_f.to(device), x_t.to(device)
            r_id, r_lvl, ln = r_id.to(device), r_lvl.to(device), ln.to(device)
            target = target.cpu().numpy()
            
            # 获取当前 batch 中每个样本的真实 Road ID (从 r_id 映射回来)
            # Dataset 中 r_id 已经是映射过的 0~N index
            # 我们需要查 road_map 知道原始 ID，再查 road_cluster_map 知道类别
            # 这种反查太慢。
            # 优化：在 Dataset 初始化时，建立一个 index -> cluster 的 tensor
            
            # 这里做个临时处理：
            # 实际上 r_id 就是 road index。我们可以预先建立 road_index -> cluster_label 的数组
            if batch_idx == 0:
                # 建立 road_index -> cluster map (只做一次)
                # full_train_dataset.road_map: {'link_id': index}
                # road_cluster_map: {'link_id': cluster}
                idx_to_cluster = np.zeros(len(full_train_dataset.road_map), dtype=int)
                for rid, idx in full_train_dataset.road_map.items():
                    if rid in road_cluster_map:
                        idx_to_cluster[idx] = road_cluster_map[rid]
                idx_to_cluster = torch.tensor(idx_to_cluster, device=device)

            batch_clusters = idx_to_cluster[r_id] # [Batch]
            
            # 初始化预测容器
            batch_preds = torch.zeros(x_f.size(0), 6, device=device)
            
            # 对每个存在的 Cluster 进行 Mask 操作
            unique_clusters = torch.unique(batch_clusters)
            for k in unique_clusters:
                k = k.item()
                mask = (batch_clusters == k)
                if mask.sum() == 0: continue
                
                # 选出属于 Cluster k 的子集
                sub_x_f = x_f[mask]
                sub_x_t = x_t[mask]
                sub_r_id = r_id[mask]
                sub_r_lvl = r_lvl[mask]
                sub_ln = ln[mask]
                
                # 使用模型 k 预测
                pred_k, _ = cluster_models[k](sub_x_f, sub_x_t, sub_r_id, sub_r_lvl, sub_ln)
                
                # 填回
                batch_preds[mask] = pred_k

            all_preds.append((batch_preds * max_flow).cpu().numpy())
            all_targets.append(target * max_flow)
            
    test_duration = time.time() - start_time
    
    # 3. 评估
    y_pred = np.concatenate(all_preds, axis=0).flatten()
    y_true = np.concatenate(all_targets, axis=0).flatten()
    
    # 去除0值防止MAPE爆炸
    mask = y_true > 1.0
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    print("-" * 30)
    print(f"Hybrid MetaSTC (Explicit Clustering) Results:")
    print(f"Test Time:  {test_duration:.4f}s")
    print(f"Test MAE:   {mean_absolute_error(y_true_clean, y_pred_clean):.4f}")
    print(f"Test RMSE:  {np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)):.4f}")
    print(f"Test MAPE:  {np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)):.4f}")
    print(f"Test R2:    {r2_score(y_true_clean, y_pred_clean):.4f}")
    print("-" * 30)
    
    # 4. 绘图 (选一段连续的)
    plt.figure(figsize=(15, 6))
    look_forward = 6
    # 取前 288 个点 (大约一天)
    display_len = min(len(y_true), 288 * look_forward)
    indices = np.arange(0, display_len, look_forward) # 降采样，避免太密集
    
    plt.plot(y_true[indices], label='Ground Truth', color='#4C72B0')
    plt.plot(y_pred[indices], label='Hybrid Prediction', color='#55A868', linestyle='--')
    plt.title('Hybrid MetaSTC Prediction (Explicit Clustering)')
    plt.legend()
    plt.savefig('figure/hybrid_prediction.png')
    print("Figure saved to figure/hybrid_prediction.png")

# ==========================================
# 主流程
# ==========================================
if __name__ == "__main__":
    if args.mode == 'train':
        train_global()
        fine_tune_clusters()
        # 训练完顺便测一下
        test_hybrid()
    else:
        # 单独测试模式
        test_hybrid()