

evaluate_col_agent(){
    local robot_type="$1"
    local task_name="$2"
    if [[ -z "$task_name" ]]; then
        echo "Usage: $0 <task-name>"
        return 1
    fi

    echo "Evaluating col_network for task: $task_name"
    local result_path="${PROJECT_ROOT}/logs/results/col/$task_name"
    python3 -u main.py \
        setup=metaworld \
        env=metaworld-mt1 \
        worker.multitask.num_envs=1 \
        experiment.robot_type="${robot_type}" \
        experiment.mode=evaluate_collective_transformer \
        env.benchmark.env_name="$task_name" \
        experiment.evaluate_transformer="collective_network"  | tee -a $result_path
}

evaluate_real(){
    local task_name="$1"
    python3 main.py \
    setup=metaworld \
    env=metaworld-mt1 \
    experiment.mode=evaluate_real \
    env.benchmark.env_name=$task_name \
    experiment.real_robot_episodes=1 \
    experiment.real_robot_max_steps=200
}

evaluate_sim(){
    local task_name="$1"
    python3 main.py \
    setup=metaworld \
    env=metaworld-mt1 \
    experiment.mode=evaluate_sim \
    env.benchmark.env_name=$task_name \
    experiment.sim_episodes=10 \
    experiment.sim_max_steps=200
}


