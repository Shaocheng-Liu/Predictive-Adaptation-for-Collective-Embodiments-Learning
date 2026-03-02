import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import os

metadata_path = "metadata/task_embedding/roberta_small/metaworld-all.json"
#path = 'Transformer_RNN/embedding_log/emb_own.pth'
#path = 'Transformer_RNN/embedding_log/emb_std.pth'
path = 'Transformer_RNN/embedding_log_task_true_seed_3/emb.pth'
cluster_path = 'Transformer_RNN/bnpy_save/data/latent_samples_end.npz'
rnn_path = 'Transformer_RNN/embedding_log/rnn_emb.pth'
safe_arm = False

def multiple_samples():
    # Generate or load your data as a numpy array
    datafile = torch.load(path)
    state_embedding = datafile['state_embedding']
    task_obs = datafile['task'].flatten()
    tra_num = datafile['tra_num'].flatten()
    rewards = datafile['rewards'].flatten()
    rewards = rewards / np.max(rewards)
    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(task_obs))))

    # Create mapping ids to names
    with open(metadata_path) as f:
        metadata = json.load(f)
    task_names = list(metadata.keys())

    # Initialize t-SNE object
    tsne = TSNE(n_components=2, random_state=42)

    # Fit and transform your data
    data_tsne = tsne.fit_transform(state_embedding)

    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(task_obs))))

    # Plot the result
    plt.figure(figsize=(16, 12))

    for i, label in enumerate(np.unique(task_obs)):
        mask = (task_obs == label)
        plt.scatter(data_tsne[mask, 0], data_tsne[mask, 1], color=colors[i], label=task_names[label], alpha=0.5)

    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # Create legend
    legend_dict = {label: color for label, color in zip(task_names, colors)}
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=key, markerfacecolor=value, markersize=10) for key, value in legend_dict.items()]
    plt.legend(handles=handles, title='Class')

    plt.savefig('Transformer_RNN/graphics/tSNE_plot_state_emb.png', bbox_inches='tight')

    plt.show()

    # trajacetory pos plot
    plt.figure(figsize=(16, 12))

    colors = plt.cm.coolwarm(tra_num) #plt.cm.tab10(np.linspace(0, 1, len(np.unique(tra_num))))

    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], color=colors)

    plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=1)), label='trajacetory pos')

    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    plt.savefig('Transformer_RNN/graphics/tSNE_state_emb.png', bbox_inches='tight')

    plt.show()

def single_samples():
    # Generate or load your data as a numpy array
    datafile = torch.load(path)
    #data = datafile['state_emb']
    data = datafile['tra_emb']
    task_obs = datafile['task'].flatten()
    rewards = datafile['rewards'].flatten()
    task_arm = datafile['task_arm'].flatten()
    rewards = rewards / np.max(rewards)
    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(task_obs))))

    # Create mapping ids to names
    with open(metadata_path) as f:
        metadata = json.load(f)
    task_names = list(metadata.keys())

    # Initialize t-SNE object
    tsne = TSNE(n_components=2, random_state=42)

    # Fit and transform your data
    data_tsne = tsne.fit_transform(data)

    # Plot the result
    plt.figure(figsize=(16, 12))

    for i, label in enumerate(np.unique(task_obs)):
        mask = (task_obs == label)
        plt.scatter(data_tsne[mask, 0], data_tsne[mask, 1], color=colors[i], label=task_names[label], alpha=0.5)

    plt.title('t-SNE Cluster Visualization', fontsize=22)
    plt.xlabel('t-SNE Component 1', fontsize=19)
    plt.ylabel('t-SNE Component 2', fontsize=19)

    # Create legend
    legend_dict = {task_names[np.unique(task_obs)[i]]: colors[i] for i in range(len(np.unique(task_obs)))}
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=key, markerfacecolor=value, markersize=10) for key, value in legend_dict.items()]
    plt.legend(handles=handles, title='Task', fontsize=13, title_fontsize=14)

    plt.savefig('Transformer_RNN/graphics/tSNE_plot.png', bbox_inches='tight')

    plt.show()

    # reward plot
    plt.figure(figsize=(16, 12))

    colors = plt.cm.coolwarm(rewards)

    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], color=colors)

    plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=1)), label='Reward')

    plt.title('Reward Visualization', fontsize=22)
    plt.xlabel('t-SNE Component 1', fontsize=19)
    plt.ylabel('t-SNE Component 2', fontsize=19)

    plt.savefig('Transformer_RNN/graphics/tSNE_reward_plot.png', bbox_inches='tight')

    plt.show()

    # arm config plot
    if safe_arm:
        plt.figure(figsize=(16, 12))

        colors = plt.cm.tab20(np.linspace(0, 1, len(np.unique(task_arm))))

        for i in range(len(np.unique(task_arm))):
            mask = (task_arm == i)
            plt.scatter(data_tsne[mask, 0], data_tsne[mask, 1], color=colors[i], alpha=0.5)

        plt.title('Arm Config Visualization')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')

        # Create legend
        legend_dict = {label: color for label, color in zip(np.unique(task_arm), colors)}
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=key, markerfacecolor=value, markersize=10) for key, value in legend_dict.items()]
        plt.legend(handles=handles, title='Class')

        plt.savefig('Transformer_RNN/graphics/tSNE_arm_config_plot.png', bbox_inches='tight')

        plt.show()

def plot_trajectory_emb():
    # Generate or load your data as a numpy array
    datafile = torch.load(path, weights_only=False)
    data = datafile['tra_emb']
    task_obs = datafile['task'].flatten()
    task_arm = datafile['task_arm'].flatten()
    rewards = datafile['rewards'].flatten()
    timesteps = datafile['timesteps'].flatten()
    timesteps = timesteps / np.max(timesteps)
    rewards = rewards / np.max(rewards)
    
    print('全局 arm 分布: ', np.unique(task_arm, return_counts=True))

    print('—— 每个 task 下的 arm 计数 ——')
    for t in sorted(np.unique(task_obs)):
        ids, cnts = np.unique(task_arm[task_obs==t], return_counts=True)
        print(f'task {int(t)}:', dict(zip(ids.astype(int), cnts)))

    # Create mapping ids to names
    with open(metadata_path) as f:
        metadata = json.load(f)
    task_names = list(metadata.keys())

    # Initialize t-SNE object
    tsne = TSNE(n_components=2, random_state=42)

    # Fit and transform your data
    data_tsne = tsne.fit_transform(data)

    task_ids = np.unique(task_obs)
    arm_ids = np.unique(task_arm)
    colors = plt.cm.tab10(np.linspace(0, 1, len(task_ids)))
    markers = ['o', '^', 's', 'D', 'P', 'X', '*', 'v']  
    assert len(arm_ids) <= len(markers), "please extend the markers list."

    # Plot the result
    plt.figure(figsize=(16, 12))

    task_handles = {}
    for ti, t in enumerate(task_ids):
        color = colors[ti]
        for ai, a in enumerate(arm_ids):
            m = markers[ai]
            mask = (task_obs == t) & (task_arm == a)
            if not np.any(mask):
                continue
            plt.scatter(
                data_tsne[mask, 0], data_tsne[mask, 1],
                color=[color], marker=m, alpha=0.6, s=18, linewidths=0
            )
        # 增大了图例中圆点的大小 (markersize=15)
        task_handles[t] = plt.Line2D([0], [0], marker='o', color='w',
                                     label=task_names[t],
                                     markerfacecolor=colors[ti], markersize=15)

    plt.title('t-SNE Cluster Visualization', fontsize=22)
    plt.xlabel('t-SNE Component 1', fontsize=19)
    plt.ylabel('t-SNE Component 2', fontsize=19)

    # 将图例移至右侧外部，并放大字体 (bbox_to_anchor 定位)
    plt.legend(handles=list(task_handles.values()),
               title='Task', fontsize=16, title_fontsize=18,
               bbox_to_anchor=(1.02, 0.5), loc='center left', frameon=True)

    os.makedirs('Transformer_RNN/graphics', exist_ok=True)
    
    # 使用 bbox_inches='tight' 确保外部图例不会被裁剪，并保存为 PDF
    plt.savefig('Transformer_RNN/graphics/tSNE_plot_tra.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    # reward plot
    plt.figure(figsize=(16, 12))
    colors_reward = plt.cm.coolwarm(rewards)
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], color=colors_reward)
    plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=1)), label='Reward')
    plt.title('Reward Visualization', fontsize=22)
    plt.xlabel('t-SNE Component 1', fontsize=19)
    plt.ylabel('t-SNE Component 2', fontsize=19)
    # 同步修改为 PDF 输出
    plt.savefig('Transformer_RNN/graphics/tSNE_reward_plot_tra.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    # timesteps plot
    plt.figure(figsize=(16, 12))
    colors_time = plt.cm.coolwarm(timesteps)
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], color=colors_time)
    plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=1)), label='Timesteps')
    plt.title('Timesteps Visualization', fontsize=22)
    plt.xlabel('t-SNE Component 1', fontsize=19)
    plt.ylabel('t-SNE Component 2', fontsize=19)
    # 同步修改为 PDF 输出
    plt.savefig('Transformer_RNN/graphics/tSNE_timesteps_plot_tra.pdf', format='pdf', bbox_inches='tight')
    plt.show()



#multiple_samples()
#single_samples()
plot_trajectory_emb()
#cluster_rnn_assign()
#cluster_assign()
#analysis()
