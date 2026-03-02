#!/bin/bash

# 视频生成统计和验证脚本

echo ""
echo "========================================="
echo "📊 视频生成统计报告"
echo "========================================="
echo ""

# 定义机械臂和任务
ROBOTS=("sawyer" "gen3" "unitree_z1" "ur10e" "xarm7" "panda" "kuka" "ur5e" "viperx")
TASKS=("reach-v2" "push-v2" "pick-place-v2" "door-open-v2" "drawer-open-v2" 
       "drawer-close-v2" "button-press-topdown-v2" "peg-insert-side-v2" 
       "window-open-v2" "window-close-v2")

# 统计已生成的视频
TOTAL_VIDEOS=0
TOTAL_SIZE=0

echo "按机械臂统计："
echo "─────────────────────────────────────"

for robot in "${ROBOTS[@]}"; do
    robot_videos=$(find outputs/*/video -name "${robot}_*.mp4" 2>/dev/null | wc -l)
    robot_size=$(find outputs/*/video -name "${robot}_*.mp4" 2>/dev/null -exec du -b {} + 2>/dev/null | awk '{sum+=$1} END {printf "%.2f", sum/1024/1024/1024}')
    
    if [ "$robot_videos" -gt 0 ]; then
        printf "%-15s %3d/10 个视频   %5s GB\n" "$robot" "$robot_videos" "$robot_size"
    else
        printf "%-15s   0/10 个视频   - \n" "$robot"
    fi
    
    TOTAL_VIDEOS=$((TOTAL_VIDEOS + robot_videos))
    TOTAL_SIZE=$(awk "BEGIN {print $TOTAL_SIZE + $robot_size}")
done

echo "─────────────────────────────────────"
echo ""

# 整体统计
echo "整体统计："
echo "─────────────────────────────────────"
printf "已生成: %d / 90 个视频 (%.1f%%)\n" "$TOTAL_VIDEOS" "$(awk "BEGIN {printf \"%.1f\", $TOTAL_VIDEOS * 100 / 90}")"
printf "总大小: %.2f GB\n" "$TOTAL_SIZE"
echo "─────────────────────────────────────"
echo ""

# 按任务统计
echo "按任务统计："
echo "─────────────────────────────────────"

for task in "${TASKS[@]}"; do
    task_videos=$(find outputs/*/video -name "*_${task}_*.mp4" 2>/dev/null | wc -l)
    if [ "$task_videos" -gt 0 ]; then
        printf "%-35s %d/9 个机械臂\n" "$task" "$task_videos"
    fi
done

echo "─────────────────────────────────────"
echo ""

# 缺失视频
echo "缺失视频（未生成）："
echo "─────────────────────────────────────"

missing_count=0
for robot in "${ROBOTS[@]}"; do
    for task in "${TASKS[@]}"; do
        if ! find outputs/*/video -name "${robot}_*_${task}_*.mp4" 2>/dev/null | grep -q .; then
            echo "  • $robot - $task"
            missing_count=$((missing_count + 1))
        fi
    done
done

if [ "$missing_count" -eq 0 ]; then
    echo "✓ 所有视频已生成！"
else
    echo "─────────────────────────────────────"
    printf "总计缺失: %d 个视频\n" "$missing_count"
fi

echo ""
echo "========================================="

# 最近生成的5个视频
echo ""
echo "📺 最近生成的视频："
echo "─────────────────────────────────────"
find outputs/*/video -name "*.mp4" -type f 2>/dev/null -printf '%T@ %p\n' | sort -rn | head -5 | while read timestamp path; do
    filename=$(basename "$path")
    size=$(du -h "$path" | awk '{print $1}')
    echo "  $size - $filename"
done

echo "========================================="
echo ""

# 生成缺失视频的快速命令
if [ "$missing_count" -gt 0 ]; then
    echo "💡 快速恢复缺失视频："
    echo "bash generate_all_videos_advanced.sh --skip-existing"
    echo ""
fi
