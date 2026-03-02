#!/bin/bash

# 高级版本：支持并行、断点续传、跳过已有视频
# 使用方法: bash generate_all_videos_advanced.sh [--parallel N] [--skip-existing]

PARALLEL_JOBS=1
SKIP_EXISTING=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

set -e

# 加载 run.sh 中定义的函数
source run.sh

# 定义9个机械臂
ROBOTS=("sawyer" "gen3" "unitree_z1" "ur10e" "xarm7" "panda" "kuka" "ur5e" "viperx")

# 定义10个任务
TASKS=("reach-v2" "push-v2" "pick-place-v2" "door-open-v2" "drawer-open-v2" 
       "drawer-close-v2" "button-press-topdown-v2" "peg-insert-side-v2" 
       "window-open-v2" "window-close-v2")

# 计算总数
TOTAL=$((${#ROBOTS[@]} * ${#TASKS[@]}))
CURRENT=0
SKIPPED=0

# 记录开始时间
START_TIME=$(date +%s)

echo "======================================"
echo "开始生成视频（高级模式）"
echo "机械臂数: ${#ROBOTS[@]}"
echo "任务数: ${#TASKS[@]}"
echo "总视频数: $TOTAL"
echo "并行任务数: $PARALLEL_JOBS"
echo "跳过已有视频: $SKIP_EXISTING"
echo "======================================"
echo ""

# 创建临时文件存储待生成任务
TASK_QUEUE=$(mktemp)

# 填充任务队列
for robot in "${ROBOTS[@]}"; do
    for task in "${TASKS[@]}"; do
        echo "$robot $task" >> "$TASK_QUEUE"
    done
done

# 处理任务的函数
process_task() {
    local robot="$1"
    local task="$2"
    local job_num="$3"
    local total="$4"
    
    # 检查视频是否已存在
    VIDEO_PATTERN="outputs/*/video/*${robot}*${task}*success*.mp4"
    if [ "$SKIP_EXISTING" = true ] && ls $VIDEO_PATTERN 2>/dev/null | grep -q .; then
        echo "[$job_num/$total] ⊘ 跳过（已有）: $robot - $task"
        return 0
    fi
    
    # 运行生成 - 在子shell中source run.sh以获得evaluate_col_agent（隐藏所有输出）
    (
        source run.sh
        evaluate_col_agent "$robot" "$task" > /dev/null 2>&1
    ) || {
        echo "[$job_num/$total] ✗ 失败: $robot - $task"
        return 1
    }
    
    echo "[$job_num/$total] ✓ 完成: $robot - $task"
}

export -f process_task
export SKIP_EXISTING

# 使用 GNU Parallel（如果可用）或回退到 xargs
if command -v parallel &> /dev/null; then
    cat "$TASK_QUEUE" | parallel --jobs "$PARALLEL_JOBS" --line-buffer \
        'job_id=$(($(cat '"$TASK_QUEUE"' | grep -n "^$1 $2$" | cut -d: -f1)))
         process_task {1} {2} '$job_id' '"$TOTAL"''
elif command -v xargs &> /dev/null; then
    # 简单的 xargs 并行 (不支持进度显示)
    cat "$TASK_QUEUE" | xargs -P "$PARALLEL_JOBS" -I {} bash -c 'process_task {} '$TOTAL''
else
    # 顺序处理
    CURRENT=0
    while IFS=' ' read -r robot task; do
        CURRENT=$((CURRENT + 1))
        PROGRESS=$((CURRENT * 100 / TOTAL))
        echo "[$PROGRESS%] [$CURRENT/$TOTAL]"
        process_task "$robot" "$task" "$CURRENT" "$TOTAL"
    done < "$TASK_QUEUE"
fi

# 清理临时文件
rm "$TASK_QUEUE"

# 计算总耗时
TOTAL_TIME=$(($(date +%s) - START_TIME))
MINUTES=$((TOTAL_TIME / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "======================================"
echo "✓ 视频生成完成！"
echo "总耗时: ${MINUTES}m${SECONDS}s"
echo "视频位置: outputs/*/video/"
echo "查看文件: ls -lh outputs/*/video/*.mp4 | wc -l"
echo "======================================"
