

SCRIPT_EXTRA_ARGS=("$@")

evaluate_col_agent(){
    local robot_type="$1"
    local task_name="$2"
    local run_suffix="$3"
    if [[ -z "$task_name" ]]; then
        echo "Usage: $0 <task-name>"
        return 1
    fi

    echo "Evaluating col_network for task: $task_name"
    local log_dir="${PROJECT_ROOT}/logs/results/col/${task_name}"
    mkdir -p "$log_dir"

    if [[ -n "$run_suffix" ]]; then
        local log_file="${log_dir}/eval_${robot_type}_${run_suffix}.log"
    else
        # 自动加时间戳，格式: eval_gen3_20240120_1230.log
        local timestamp=$(date +%Y%m%d_%H%M%S)
        local log_file="${log_dir}/eval_${robot_type}_${timestamp}.log"
    fi
    python3 -u main.py \
        setup=metaworld \
        env=metaworld-mt1 \
        worker.multitask.num_envs=1 \
        experiment.robot_type="${robot_type}" \
        experiment.mode=evaluate_collective_transformer \
        env.benchmark.env_name="$task_name" \
        experiment.evaluate_transformer="collective_network" \
        "${SCRIPT_EXTRA_ARGS[@]}" > "$log_file" 2>&1
}





evaluate_col_agent viperx reach-v2
evaluate_col_agent viperx push-v2
evaluate_col_agent viperx peg-insert-side-v2
evaluate_col_agent viperx door-open-v2
evaluate_col_agent viperx window-open-v2
evaluate_col_agent viperx window-close-v2
evaluate_col_agent viperx drawer-open-v2
evaluate_col_agent viperx button-press-topdown-v2
evaluate_col_agent viperx faucet-open-v2
evaluate_col_agent viperx pick-place-v2