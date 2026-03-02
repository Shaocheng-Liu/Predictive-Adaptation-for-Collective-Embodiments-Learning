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
    # 1. Load data
    # Note: Ensure your save_path contains 'task_arm' (saved in the embedding_validation function of train.py)
    try:
        datafile = torch.load(path, map_location='cpu', weights_only=False) # Prevent GPU/CPU mismatch
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return

    data = datafile['tra_emb']
    with open(metadata_path) as f:
        metadata = json.load(f)
    task_names = list(metadata.keys())
    if isinstance(data, torch.Tensor):
        data = data.numpy() # Convert to numpy for sklearn
        
    # === Evaluation 1: Task Clustering ===
    env_idx = datafile['task']
    unique_env_idx = np.unique(env_idx)
    n_task_clusters = len(unique_env_idx)
    
    print(f"\n[Task Clustering] Ground Truth Clusters: {n_task_clusters}")
    
    # Map labels to 0~N-1
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

    # === Evaluation 2: Robot Clustering ===
    # Check if robot label (task_arm) exists
    if 'task_arm' in datafile:
        robot_idx = datafile['task_arm']
        unique_robot_idx = np.unique(robot_idx)
        n_robot_clusters = len(unique_robot_idx)
        
        print(f"\n[Robot Clustering] Ground Truth Clusters: {n_robot_clusters} (IDs: {unique_robot_idx})")
        
        # Map labels to 0~N-1
        value_to_rank_robot = {value: rank for rank, value in enumerate(unique_robot_idx)}
        y_true_robot = np.vectorize(value_to_rank_robot.get)(robot_idx)
        
        # K-Means for Robot (Note: K is set to the number of robots, typically 3)
        kmeans_robot = KMeans(n_clusters=n_robot_clusters, random_state=seed, n_init=10)
        y_pred_robot = kmeans_robot.fit_predict(data)
        
        # Metrics for Robot
        acc_robot = calculate_acc_multi_assignments(y_true_robot, y_pred_robot)
        sil_robot = silhouette_score(data, y_pred_robot) # Silhouette may be lower than Task since Robot is only a sub-feature of the Embedding
        hom_robot = homogeneity_score(y_true_robot, y_pred_robot)
        
        print(f"Robot ACC: {acc_robot:.4f}")
        print(f"Robot Silhouette: {sil_robot:.4f}")
        print(f"Robot Homogeneity: {hom_robot:.4f}")
        
    else:
        print("\n[Robot Clustering] Skipped ('task_arm' not found in .pth file)")

def calculate_acc_multi_assignments(y_true, y_pred):
    from sklearn.metrics.cluster import contingency_matrix
    from scipy.optimize import linear_sum_assignment
    
    # Use the Hungarian algorithm to compute optimal matching Accuracy
    # This is a more rigorous method for calculating clustering ACC
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

    # Load task name mapping
    with open(metadata_path) as f:
        metadata = json.load(f)
    task_names = list(metadata.keys())
        
    task_idx = datafile['task']
    robot_idx = datafile['task_arm']
    
    unique_tasks = np.unique(task_idx)
    unique_robots = np.unique(robot_idx)
    
    print(f"Total Tasks: {len(unique_tasks)}")
    print(f"Total Robots: {len(unique_robots)}")
    
    # Store intra-task Robot Clustering scores
    task_robot_accs = []
    task_robot_sils = []
    
    print("\n--- Intra-Task Robot Clustering Results ---")
    # Adjusted header widths to accommodate long Task Names
    print(f"{'Task ID':<10} | {'Task Name':<25} | {'ACC':<10} | {'Silhouette':<10}")
    print("-" * 65)
    
    # [Fix 1]: Iterate directly over actual ID values in unique_tasks
    # No need to use enumerate index for filtering
    for t in unique_tasks:
        # [Fix 2]: Use the actual task ID (t) to create the mask, not the loop index
        indices = (task_idx == t)
        sub_data = data[indices]
        sub_labels = robot_idx[indices]
        
        # Get task name (ensure conversion to int index)
        t_int = int(t)
        current_task_name = task_names[t_int] if t_int < len(task_names) else f"Unknown-{t_int}"
        
        # Ensure the current task contains multiple Robot data entries
        if len(np.unique(sub_labels)) < 2:
            print(f"{t_int:<10} | {current_task_name:<25} | Skipped (Not enough classes)")
            continue
            
        # 2. Perform Robot clustering within the current task (K=3)
        kmeans = KMeans(n_clusters=len(unique_robots), random_state=42, n_init=10)
        y_pred = kmeans.fit_predict(sub_data)
        
        # 3. Compute metrics
        acc = calculate_acc_multi_assignments(sub_labels, y_pred)
        
        if len(sub_data) > len(unique_robots):
            sil = silhouette_score(sub_data, y_pred)
        else:
            sil = 0
            
        task_robot_accs.append(acc)
        task_robot_sils.append(sil)
        
        # [Fix 3]: Use the actual ID and corresponding name when printing
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