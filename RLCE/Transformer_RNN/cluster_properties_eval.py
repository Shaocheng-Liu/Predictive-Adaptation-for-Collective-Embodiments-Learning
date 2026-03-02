import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    homogeneity_score,
)
from scipy.optimize import linear_sum_assignment
import torch
from sklearn.metrics.cluster import contingency_matrix
import json

# Evaluation of the cluster properties for thesis

metadata_path = "metadata/task_embedding/roberta_small/metaworld-all.json"
#path_own_loss = 'Transformer_RNN/embedding_log/emb_own.pth'
path_own_loss = 'Transformer_RNN/embedding_log_rob_0.5/emb.pth'
path_std_loss = 'Transformer_RNN/embedding_log/emb_std.pth'
path_Rnn = 'Transformer_RNN/embedding_log/rnn_emb.pth'

# Function to calculate accuracy using the Hungarian algorithm
def calculate_acc(y_true, y_pred):
    """
    Match predicted clusters to ground truth labels and calculate accuracy.

    Parameters:
    - y_true: Ground truth labels
    - y_pred: Predicted cluster labels

    Returns:
    - accuracy: Accuracy of clustering
    """
    contingency_matrix = np.zeros((max(y_true) + 1, max(y_pred) + 1))
    for i, j in zip(y_true, y_pred):
        contingency_matrix[i, j] += 1
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    return sum(contingency_matrix[row, col] for row, col in zip(row_ind, col_ind)) / len(y_true)

def calculate_acc_multi_assignments(y_true, y_pred):
    """
    Calculate Accuracy (ACC) with multiple assignments allowed per cluster.

    Parameters:
    - y_true: Ground truth labels.
    - y_pred: Predicted cluster labels.

    Returns:
    - accuracy: Accuracy considering multiple assignments.
    """
    # Build contingency matrix
    cont_matrix = contingency_matrix(y_true, y_pred)
    
    # Assign clusters to labels maximizing accuracy
    # Each cluster contributes its best-matching class count
    max_assignments = cont_matrix.max(axis=0)  # Max overlap for each predicted cluster
    accuracy = max_assignments.sum() / len(y_true)  # Normalize by total samples
    
    return accuracy

# Example dataset
# Replace these with your own dataset and labels
seed=5
np.random.seed(seed)
n_clusters = 10

def Rnn():
    datafile = torch.load(path_own_loss)
    data = datafile['state_emb']
    env_idx = datafile['task']
    unique_env_idx = np.unique(env_idx)
    value_to_rank = {value: rank-1 for rank, value in enumerate(unique_env_idx, start=1)}
    replaced_env_idx = np.vectorize(value_to_rank.get)(env_idx)

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    y_pred = kmeans.fit_predict(data)

    # Calculate metrics
    acc_adjusted = calculate_acc_multi_assignments(replaced_env_idx, y_pred)
    silhouette = silhouette_score(data, y_pred)
    homogeneity = homogeneity_score(replaced_env_idx, y_pred)

    # Print results
    print(f"Adjusted Accuracy (ACC): {acc_adjusted:.4f}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Homogeneity Score: {homogeneity:.4f}")

def transformer_eval(path):
    print(f"\n{'='*20} Evaluating: {path} {'='*20}")
    # 1. 加载数据
    # 注意：确保你的 save_path 里保存了 'task_arm' (在 train.py 的 embedding_valdation 函数里)
    try:
        datafile = torch.load(path, map_location='cpu', weights_only=False) # 防止 GPU/CPU 不匹配
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return

    data = datafile['tra_emb']
    with open(metadata_path) as f:
        metadata = json.load(f)
    task_names = list(metadata.keys())
    if isinstance(data, torch.Tensor):
        data = data.numpy() # 转为 numpy 用于 sklearn
        
    # === Evaluation 1: Task Clustering (任务聚类) ===
    env_idx = datafile['task']
    unique_env_idx = np.unique(env_idx)
    n_task_clusters = len(unique_env_idx)
    
    print(f"\n[Task Clustering] Ground Truth Clusters: {n_task_clusters}")
    
    # 映射 Label 到 0~N-1
    value_to_rank_task = {value: rank for rank, value in enumerate(unique_env_idx)}
    y_true_task = np.vectorize(value_to_rank_task.get)(env_idx)

    # K-Means for Task
    kmeans_task = KMeans(n_clusters=n_task_clusters, random_state=seed, n_init=10)
    y_pred_task = kmeans_task.fit_predict(data)

    # Metrics for Task
    acc_task = calculate_acc_multi_assignments(y_true_task, y_pred_task)
    sil_task = silhouette_score(data, y_pred_task)
    hom_task = homogeneity_score(y_true_task, y_pred_task)

    print(f"Task ACC: {acc_task:.4f}")
    print(f"Task Silhouette: {sil_task:.4f}")
    print(f"Task Homogeneity: {hom_task:.4f}")

    # === Evaluation 2: Robot Clustering (机器人聚类) ===
    # 检查是否存在 robot label (task_arm)
    if 'task_arm' in datafile:
        robot_idx = datafile['task_arm']
        unique_robot_idx = np.unique(robot_idx)
        n_robot_clusters = len(unique_robot_idx)
        
        print(f"\n[Robot Clustering] Ground Truth Clusters: {n_robot_clusters} (IDs: {unique_robot_idx})")
        
        # 映射 Label 到 0~N-1
        value_to_rank_robot = {value: rank for rank, value in enumerate(unique_robot_idx)}
        y_true_robot = np.vectorize(value_to_rank_robot.get)(robot_idx)
        
        # K-Means for Robot (注意：这里 K 变成了机器人的数量，通常是 3)
        kmeans_robot = KMeans(n_clusters=n_robot_clusters, random_state=seed, n_init=10)
        y_pred_robot = kmeans_robot.fit_predict(data)
        
        # Metrics for Robot
        acc_robot = calculate_acc_multi_assignments(y_true_robot, y_pred_robot)
        sil_robot = silhouette_score(data, y_pred_robot) # 这里的 Silhouette 可能不如 Task 高，因为 Robot 只是 Embedding 的一个子特征
        hom_robot = homogeneity_score(y_true_robot, y_pred_robot)
        
        print(f"Robot ACC: {acc_robot:.4f}")
        print(f"Robot Silhouette: {sil_robot:.4f}")
        print(f"Robot Homogeneity: {hom_robot:.4f}")
        
    else:
        print("\n[Robot Clustering] Skipped ('task_arm' not found in .pth file)")

def calculate_acc_multi_assignments(y_true, y_pred):
    from sklearn.metrics.cluster import contingency_matrix
    from scipy.optimize import linear_sum_assignment
    
    # 使用匈牙利算法计算最佳匹配 Accuracy
    # 这是一个更严谨的计算聚类 ACC 的方法
    cm = contingency_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    return cm[row_ind, col_ind].sum() / np.sum(cm)

def hierarchical_eval(path):
    print(f"\n{'='*20} Hierarchical Evaluation: {path} {'='*20}")
    try:
        datafile = torch.load(path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return

    data = datafile['tra_emb']
    if isinstance(data, torch.Tensor):
        data = data.numpy()

    # 加载任务名称映射
    with open(metadata_path) as f:
        metadata = json.load(f)
    task_names = list(metadata.keys())
        
    task_idx = datafile['task']
    robot_idx = datafile['task_arm']
    
    unique_tasks = np.unique(task_idx)
    unique_robots = np.unique(robot_idx)
    
    print(f"Total Tasks: {len(unique_tasks)}")
    print(f"Total Robots: {len(unique_robots)}")
    
    # 存储每个任务内部的 Robot Clustering 分数
    task_robot_accs = []
    task_robot_sils = []
    
    print("\n--- Intra-Task Robot Clustering Results ---")
    # 调整了表头宽度以适应长 Task Name
    print(f"{'Task ID':<10} | {'Task Name':<25} | {'ACC':<10} | {'Silhouette':<10}")
    print("-" * 65)
    
    # 【修正点 1】：直接遍历 unique_tasks 中的实际 ID 值
    # 不需要 enumerate 的 index 来做筛选
    for t in unique_tasks:
        # 【修正点 2】：使用实际的 task ID (t) 来创建 mask，而不是循环的索引
        indices = (task_idx == t)
        sub_data = data[indices]
        sub_labels = robot_idx[indices]
        
        # 获取任务名称 (确保转为 int 索引)
        t_int = int(t)
        current_task_name = task_names[t_int] if t_int < len(task_names) else f"Unknown-{t_int}"
        
        # 确保当前任务包含多个 Robot 数据
        if len(np.unique(sub_labels)) < 2:
            print(f"{t_int:<10} | {current_task_name:<25} | Skipped (Not enough classes)")
            continue
            
        # 2. 在当前任务内部做 Robot 聚类 (K=3)
        kmeans = KMeans(n_clusters=len(unique_robots), random_state=42, n_init=10)
        y_pred = kmeans.fit_predict(sub_data)
        
        # 3. 计算指标
        acc = calculate_acc_multi_assignments(sub_labels, y_pred)
        
        if len(sub_data) > len(unique_robots):
            sil = silhouette_score(sub_data, y_pred)
        else:
            sil = 0
            
        task_robot_accs.append(acc)
        task_robot_sils.append(sil)
        
        # 【修正点 3】：打印时使用实际 ID 和对应的名称
        print(f"{t_int:<10} | {current_task_name:<25} | {acc:.4f}     | {sil:.4f}")

    print("-" * 65)
    if task_robot_accs:
        print(f"Average Intra-Task Robot ACC: {np.mean(task_robot_accs):.4f}")
        print(f"Average Intra-Task Robot Silhouette: {np.mean(task_robot_sils):.4f}")
    else:
        print("No valid tasks for evaluation.")

hierarchical_eval(path_own_loss)
# print(f"--- Evaluation stdandard loss with {n_clusters} clusters ---")
# transformer(path_std_loss)
# print(f"--- Evaluation Rnn with {n_clusters} clusters ---")
# Rnn()