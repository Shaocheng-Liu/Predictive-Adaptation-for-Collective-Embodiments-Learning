#!/bin/bash

# 自动生成9个机械臂 × 10个任务的视频
# 使用方法: bash generate_all_videos.sh

set -e  # 命令失败时退出

# 加载 run.sh 中定义的函数
source run.sh

# 定义9个机械臂 "sawyer" "ur10e" "gen3" "xarm7" "unitree_z1" "panda" "kuka" "ur5e" "viperx"
ROBOTS=("sawyer" "ur10e" "gen3" "xarm7" "unitree_z1" "panda")

# 定义10个任务"reach-v2" "push-v2" "pick-place-v2" "door-open-v2" "drawer-open-v2" 
    #    "faucet-open-v2" "button-press-topdown-v2" "peg-insert-side-v2" 
    #    "window-open-v2" "window-close-v2"
TASKS=("faucet-open-v2")

# 计算总数
TOTAL=$((${#ROBOTS[@]} * ${#TASKS[@]}))
CURRENT=0

# 记录开始时间
START_TIME=$(date +%s)

echo "======================================"
echo "开始生成视频"
echo "机械臂数: ${#ROBOTS[@]}"
echo "任务数: ${#TASKS[@]}"
echo "总视频数: $TOTAL"
echo "======================================"
echo ""

# 双层循环
for robot in "${ROBOTS[@]}"; do
    for task in "${TASKS[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        # 计算进度百分比
        PROGRESS=$((CURRENT * 100 / TOTAL))
        ELAPSED=$(($(date +%s) - START_TIME))
        
        echo "[$PROGRESS%] [$CURRENT/$TOTAL] 正在生成: $robot - $task (耗时: ${ELAPSED}s)"
        
        # 运行 evaluate_col_agent 命令（隐藏所有输出）
        evaluate_col_agent "$robot" "$task" > /dev/null 2>&1 || true
        
        echo "  ✓ 完成: $robot - $task"
        echo ""
    done
done

# 计算总耗时
TOTAL_TIME=$(($(date +%s) - START_TIME))
MINUTES=$((TOTAL_TIME / 60))
SECONDS=$((TOTAL_TIME % 60))

echo "======================================"
echo "✓ 所有视频生成完成！"
echo "总耗时: ${MINUTES}m${SECONDS}s"
echo "video文件夹位置: outputs/*/video/"
echo "======================================"
