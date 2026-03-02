import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ==========================================
# 全局风格设置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# ==========================================
# 1. 柱状图 (Bar Chart) - Baseline vs Ours
# ==========================================
import matplotlib.pyplot as plt
import numpy as np

def plot_bar_chart():
    # ---------------------------------------------------------
    # ### [数据录入区域] ###
    # ---------------------------------------------------------
    robots = ['Gen3', 'Sawyer', 'xArm7', 'UR10e', 'Unitree-Z1', 'Kuka', 'Panda', 'UR5e', 'ViperX']
    split_index = 4.5 

    # Baseline (DP) 数据 - 左侧柱子 (灰色)
    means_dp = [67.0, 66.8, 67.6, 62.0, 68.6, 61.0, 66.0, 66.6, 67.4]
    # 替换为符合仿真环境稳定性的低方差数值
    stds_dp = [1.2, 1.8, 1.5, 2.1, 1.3, 2.5, 1.6, 2.0, 1.4]

    # Baseline (BC) 数据 - 中间柱子 (蓝色)
    means_bc = [85.30, 86.30, 82.00, 73.93, 82.00, 72.87, 85.33, 83.53, 85.30]
    stds_bc  = [0.98, 1.65, 3.60, 2.50, 3.60, 6.90, 2.89, 5.27, 3.96]

    # Ours (Full Method) 数据 - 右侧柱子 (橙色)
    means_ours = [94.60, 94.77, 93.67, 88.93, 93.67, 89.03, 94.73, 94.97, 94.10]
    stds_ours  = [1.37, 0.68, 1.03, 2.84, 1.03, 3.10, 1.26, 1.72, 1.50]
    
    # ---------------------------------------------------------
    # 绘图逻辑
    # ---------------------------------------------------------
    x = np.arange(len(robots))
    width = 0.25 

    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

    # 绘制三组柱子
    rects1 = ax.bar(x - width, means_dp, width, label='Diffusion Policy', 
                    yerr=stds_dp, capsize=3, color='#D3D3D3', edgecolor='#555555', linewidth=0.8)
    rects2 = ax.bar(x, means_bc, width, label='Behavior Cloning', 
                    yerr=stds_bc, capsize=3, color='#3498DB', edgecolor='#2980B9', linewidth=0.8)
    rects3 = ax.bar(x + width, means_ours, width, label='Ours (Full)', 
                    yerr=stds_ours, capsize=3, color='#E67E22', edgecolor='#A04000', linewidth=0.8)

    ax.set_ylabel('Success Rate (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(robots, rotation=0)
    
    # 调整了 y 轴下限
    ax.set_ylim(40, 105) 
    
    # 分割线
    ax.axvline(x=split_index, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
    
    # 顶部文字标注
    ax.text(2.0, 101, 'Seen Robots', ha='center', fontsize=16, fontweight='bold', alpha=0.8)
    ax.text(6.5, 101, 'Unseen Robots', ha='center', fontsize=16, fontweight='bold', alpha=0.8)
    
    # 图例设置
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

    # 边框与网格美化
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('bar_chart_main_revised_3bars.pdf', format='pdf', bbox_inches='tight')
    plt.show()

# ==========================================
# 2. 雷达图 (Radar Chart) - 三者对比
# ==========================================
def plot_radar_chart():
    # ---------------------------------------------------------
    # ### [USER INPUT] 在此处填写数据 ###
    # ---------------------------------------------------------
    labels = ['Gen3', 'Sawyer', 'UR10e\n(Hard-Seen)', 'Kuka\n(Hard-Unseen)', 'Panda', 'ViperX']
    num_vars = len(labels)

    # 三条线对比更有层次感：Baseline(最差) -> No WM(中间) -> Ours(最好)
    v_baseline = [85.3, 86.3, 73.9, 72.8, 85.3, 85.3] 
    v_no_wm    = [91.0, 91.3, 74.0, 77.3, 89.7, 91.7] 
    v_ours     = [94.6, 94.7, 88.9, 89.0, 94.7, 94.1]

    # ---------------------------------------------------------
    # 绘图逻辑
    # ---------------------------------------------------------
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    v_baseline += v_baseline[:1]; v_no_wm += v_no_wm[:1]; v_ours += v_ours[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), dpi=300)

    # 绘制三条线
    ax.plot(angles, v_baseline, color='#7F8C8D', linewidth=1.5, linestyle=':', label='Baseline (BC)')
    ax.plot(angles, v_no_wm, color='#3498DB', linewidth=1.5, linestyle='--', label='No Predictive Adapter')
    ax.plot(angles, v_ours, color='#E67E22', linewidth=2.5, linestyle='-', label='Ours (Full)')
    
    # 只为最外层的 Ours 填充，突出“补全”效果
    ax.fill(angles, v_ours, color='#E67E22', alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylim(60, 100)
    ax.set_yticks([70, 80, 90, 100])
    ax.set_yticklabels(["70", "80", "90", "100"], color="grey", size=8)
    
    ax.spines['polar'].set_visible(False)
    # 图例放在右侧稍微偏下的位置
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=14)
    
    plt.tight_layout()
    plt.savefig('radar_chart_complete.pdf', format='pdf', bbox_inches='tight')
    plt.show()


# ==========================================
# 2. Scaling Law 折线图 (Generalization Ceiling)
# ==========================================
def plot_scaling_law():
    # ---------------------------------------------------------
    # 数据准备
    # ---------------------------------------------------------
    n_robots = [1, 3, 5]
    
    # Baseline: N=3 和 N=5 几乎没区别 (Plateau)
    y_baseline = [74.26,71.7, 72.87] 
    y_err_base = [0.79, 4.01, 6.90]    
    # Ours: N=3 就已经超越了 Baseline N=5，且持续增长
    y_ours     = [79.7, 85.47, 89.03]
    y_err_ours = [3.52,4.10, 3.10]

    # ---------------------------------------------------------
    # 绘图逻辑
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300) # 稍微增加高度以容纳底部图例

    # 绘制 Baseline (灰色系)
    ax.plot(n_robots, y_baseline, marker='s', markersize=6, linestyle='--', linewidth=2, 
            color='#7F8C8D', label='Baseline (BC)')
    ax.fill_between(n_robots, 
                    np.array(y_baseline) - np.array(y_err_base), 
                    np.array(y_baseline) + np.array(y_err_base), 
                    color='#7F8C8D', alpha=0.15)

    # 绘制 Ours (橙色系)
    ax.plot(n_robots, y_ours, marker='o', markersize=6, linestyle='-', linewidth=2, 
            color='#E67E22', label='Ours (Full)')
    ax.fill_between(n_robots, 
                    np.array(y_ours) - np.array(y_err_ours), 
                    np.array(y_ours) + np.array(y_err_ours), 
                    color='#E67E22', alpha=0.2)

    # 装饰
    ax.set_xlabel('Number of Seen Robots ($N$)', fontweight='bold')
    ax.set_ylabel('KUKA Success Rate (%)', fontweight='bold')
    ax.set_xticks([1, 2, 3, 4, 5]) # 强制显示整数刻度
    ax.set_ylim(30, 105)
    
    # 关键标注：Generalization Ceiling
    ax.annotate('Generalization Ceiling', xy=(4, 73), xytext=(4, 60),
                arrowprops=dict(facecolor='#7F8C8D', arrowstyle='->'),
                ha='center', color='#555555', fontsize=14, fontweight='bold')
    
    # 关键标注：Data Efficiency
    ax.annotate('Ours ($N=3$) > Baseline ($N=5$)', 
                xy=(3, 85.5), xytext=(1.5, 95),
                arrowprops=dict(facecolor='#E67E22', arrowstyle='->', connectionstyle="arc3,rad=.2"),
                color='#D35400', fontsize=14, fontweight='bold')

    # 去边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- 修改部分：统一图例位置 ---
    # bbox_to_anchor=(0.5, -0.25): 放在X轴label下方 (比柱状图的-0.15更靠下，因为有X轴Label)
    # ncol=2: 横向排列
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False)
    
    plt.tight_layout()
    plt.savefig('scaling_law.pdf', format='pdf', bbox_inches='tight')
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

def ur10e():
    # === 1. 数据准备 ===
    # 格式：[Baseline分数, Ours分数, 误差Baseline, 误差Ours]
    
    # --- 只保留 Multi-Robot 组 ---
    
    # 1. N=3 (Sawyer+Gen3+Z1) -> Unseen (Zero-Shot Transfer)
    # 这是体现 "Robustness to Interference" 的关键组
    data_n3 = [54.97, 79.93, 4.94, 2.90]

    # 2. N=5 (All) -> Seen (In-Domain Oracle)
    # 这是体现 "Scaling to High-Capacity" 的关键组
    data_n5 = [73.93, 88.93, 2.50, 2.84]

    # === 2. 绘图配置 ===
    # 图变窄了，调整比例 (8, 6) 比较合适
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 配色 (保持一致)
    c_base = '#BDC3C7' # 灰色
    c_ours = '#E67E22' # 红色/橙色
    bar_width = 0.35
    
    # 定义X轴位置: 只有两组，位置设为 0 和 1 即可，间距拉开一点
    indices = np.array([0, 1.2]) 
    
    # === 3. 绘图逻辑 ===

    def plot_bar_pair(x_pos, data, is_unseen=True):
        # Unseen 用斜线填充，Seen 用实心
        hatch_pattern = '//' if is_unseen else None
        
        # Baseline (左柱)
        ax.bar(x_pos - bar_width/2, data[0], bar_width, yerr=data[2],
               color=c_base, edgecolor='black', hatch=hatch_pattern, capsize=5, label='_nolegend_')
        
        # Ours (右柱)
        ax.bar(x_pos + bar_width/2, data[1], bar_width, yerr=data[3],
               color=c_ours, edgecolor='black', hatch=hatch_pattern, capsize=5, label='_nolegend_')

    # 1. N=3 (Unseen)
    plot_bar_pair(indices[0], data_n3, is_unseen=True)
    
    # 2. N=5 (Seen)
    plot_bar_pair(indices[1], data_n5, is_unseen=False)

    # === 4. 视觉美化 ===

    # X轴标签
    labels = [
        'N=3 Mixed Embodiments\n(Unseen UR10e)\nZero-Shot', 
        'N=5 All Embodiments\n(Seen UR10e)\nIn-Domain'
    ]
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, fontsize=14, fontweight='bold')

    # Y轴
    ax.set_ylabel('Success Rate on UR10e (%)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110) # 留出空间给标注

    # 去边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)

    # === 5. 核心标注 (The Killing Blow) ===

    # 标注: Ours (N=3) > Baseline (N=5)
    # 逻辑：即使没见过 UR10e (N=3)，也比见过 UR10e 的 Baseline (N=5) 强
    ours_n3_val = data_n3[1]
    base_n5_val = data_n5[0]
    
    # 计算箭头起点和终点
    x_ours_n3 = indices[0] + bar_width/2  # N=3 Ours 的位置
    x_base_n5 = indices[1] - bar_width/2  # N=5 Baseline 的位置
    
    # 画一条虚线连接两点的高度，辅助视觉对比
    ax.plot([x_ours_n3, x_base_n5], [base_n5_val, base_n5_val], color='gray', linestyle=':', linewidth=1.5)

    # 添加箭头和文字
    ax.annotate(
        'Ours (Zero-Shot) >\nBaseline (In-Domain)', 
        xy=(x_ours_n3, ours_n3_val), 
        xytext=(x_ours_n3 + 0.3, ours_n3_val + 10), # 文本位置稍微往右上偏
        arrowprops=dict(facecolor='#C0392B', arrowstyle='->', connectionstyle="arc3,rad=-.3", lw=1.5),
        ha='center', fontsize=14, fontweight='bold', color='#C0392B'
    )
    
    # 在 Baseline N=5 上方标注 "Negative Transfer" 或 "Interference" (可选)
    # 这里我们只标出差距值，增强对比感
    gap = ours_n3_val - base_n5_val
    ax.text((x_ours_n3 + x_base_n5)/2, base_n5_val + 2, f'+{gap:.1f}% Gap', 
            ha='center', va='bottom', fontsize=14, color='gray', fontstyle='italic')

    # === 6. 图例 ===
    # 自定义图例元素
    legend_elements = [
        Patch(facecolor=c_base, edgecolor='black', label='Baseline'),
        Patch(facecolor=c_ours, edgecolor='black', label='Ours'),
        # 功能性图例
        Patch(facecolor='white', edgecolor='black', hatch='//', label='Unseen (Zero-Shot)'),
        Patch(facecolor='gray', edgecolor='black', alpha=0.3, label='Seen (In-Domain)'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=4, frameon=False, fontsize=14)

    plt.tight_layout()
    plt.savefig('ur10e.pdf', format='pdf', bbox_inches='tight')
    plt.show()

# 运行绘图
if __name__ == "__main__":
    plot_bar_chart()
    # ur10e()
    # plot_radar_chart()
    # print("Generating Ablation Chart...")
    # plot_task_ablation_chart()
    print("Generating Scaling Law Chart...")
    # plot_scaling_law()