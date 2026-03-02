import torch
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def inspect_dimension_noise(source_dir):
    source_path = Path(source_dir)
    pt_files = sorted([p for p in source_path.glob("*.pt") if "_" in p.stem], 
                      key=lambda p: int(p.stem.split('_')[0]))
    
    if not pt_files:
        print("❌ No files found.")
        return

    print(f"🔍 Analyzing dimensions in: {source_dir}")

    all_log_stds = []
    LOG_STD_IDX = 7  # 确保这是你之前确认正确的 Log Std Index

    for p in pt_files:
        try:
            payload = torch.load(p, weights_only=False)
            if len(payload) <= LOG_STD_IDX: continue
            
            chunk = payload[LOG_STD_IDX]
            if isinstance(chunk, torch.Tensor):
                chunk = chunk.detach().cpu().numpy()
            all_log_stds.append(chunk)
        except:
            pass

    if not all_log_stds:
        return

    # Shape: [Total_Steps, 4]
    full_log_std = np.concatenate(all_log_stds, axis=0)
    full_sigma = np.exp(full_log_std)
    
    # === 核心修改：按维度统计 ===
    # mean(axis=0) 表示跨时间求平均，保留维度差异
    mean_sigma_per_dim = full_sigma.mean(axis=0) 
    min_sigma_per_dim = full_sigma.min(axis=0)
    max_sigma_per_dim = full_sigma.max(axis=0)

    print("\n📊 === Per-Dimension Noise Analysis ===")
    dims = ["X-axis", "Y-axis", "Z-axis", "Gripper"]
    
    print(f"{'Dim':<10} | {'Mean Sigma':<12} | {'Min':<8} | {'Max':<8}")
    print("-" * 50)
    
    for i in range(4):
        name = dims[i] if i < 4 else f"Dim {i}"
        print(f"{name:<10} | {mean_sigma_per_dim[i]:.4f}       | {min_sigma_per_dim[i]:.2f}     | {max_sigma_per_dim[i]:.2f}")

    # === 可视化维度差异 ===
    plt.figure(figsize=(12, 6))
    for i in range(4):
        # 降采样绘制曲线
        plt.plot(full_sigma[:, i][::1000], label=f"Dim {i} ({dims[i] if i<4 else ''})", alpha=0.7)
    
    plt.title("Noise Level per Dimension over Training")
    plt.ylabel("Sigma")
    plt.xlabel("Steps (x1000)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('dim_noise_analysis.png')
    print("\n📈 Plot saved to 'dim_noise_analysis.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    inspect_dimension_noise(args.source)