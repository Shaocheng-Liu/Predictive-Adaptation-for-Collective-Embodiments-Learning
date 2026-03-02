

SCRIPT_EXTRA_ARGS=("$@")

print_results(){
    echo
    echo "RESULTS expert:"
    grep -r Evaluation ${PROJECT_ROOT}/logs/results/worker | grep Evaluation
    echo "RESULTS collective network:"
    grep -r Evaluation ${PROJECT_ROOT}/logs/results/col  | grep Evaluation
    echo "RESULTS student:"
    grep -r Evaluation ${PROJECT_ROOT}/logs/results/student | grep Evaluation
    echo
}

evaluate_col_agent(){
    local robot_type="$1"
    local task_name="$2"
    if [[ -z "$task_name" ]]; then
        echo "Usage: $0 <task-name>"
        return 1
    fi

    echo "Evaluating col_network for task: $task_name"
    local log_dir="${PROJECT_ROOT}/logs/results/col/${task_name}"
    mkdir -p "$log_dir"
    local log_file="${log_dir}/eval_${robot_type}.log"
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

# evaluate_col_agent sawyer reach-v2
# evaluate_col_agent sawyer push-v2
# evaluate_col_agent sawyer peg-insert-side-v2
# evaluate_col_agent sawyer door-open-v2
# evaluate_col_agent sawyer window-open-v2
# evaluate_col_agent sawyer window-close-v2
# evaluate_col_agent sawyer drawer-open-v2
# evaluate_col_agent sawyer button-press-topdown-v2
# evaluate_col_agent sawyer button-press-v2
# evaluate_col_agent sawyer pick-place-v2

# evaluate_col_agent ur10e reach-v2
# evaluate_col_agent ur10e push-v2
# evaluate_col_agent ur10e peg-insert-side-v2
# evaluate_col_agent ur10e door-open-v2
# evaluate_col_agent ur10e window-open-v2
# evaluate_col_agent ur10e window-close-v2
# evaluate_col_agent ur10e drawer-open-v2
# evaluate_col_agent ur10e button-press-topdown-v2
# evaluate_col_agent ur10e button-press-v2
# evaluate_col_agent ur10e pick-place-v2

# evaluate_col_agent ur5e reach-v2
# evaluate_col_agent ur5e push-v2
# evaluate_col_agent ur5e peg-insert-side-v2
# evaluate_col_agent ur5e door-open-v2
# evaluate_col_agent ur5e window-open-v2
# evaluate_col_agent ur5e window-close-v2
# evaluate_col_agent ur5e drawer-open-v2
# evaluate_col_agent ur5e button-press-topdown-v2
# evaluate_col_agent ur5e button-press-v2
# evaluate_col_agent ur5e pick-place-v2


# evaluate_col_agent kuka reach-v2
# evaluate_col_agent kuka push-v2
# evaluate_col_agent kuka peg-insert-side-v2
# evaluate_col_agent kuka door-open-v2
# evaluate_col_agent kuka window-open-v2
# evaluate_col_agent kuka window-close-v2
# evaluate_col_agent kuka drawer-open-v2
# evaluate_col_agent kuka button-press-topdown-v2
# evaluate_col_agent kuka button-press-v2
# evaluate_col_agent kuka pick-place-v2

# evaluate_task() {
#     local robot_type="$1"
#     local task_name="$2"
#     if [[ -z "$task_name" ]]; then
#         echo "Usage: evaluate_task <task-name>"
#         return 1
#     fi

#     echo "Evaluating expert for task: $task_name"

#     rm ${PROJECT_ROOT}/logs/experiment_test/evaluation_models/*
#     cp ${PROJECT_ROOT}/logs/experiment_test/model_dir/model_${robot_type}_${task_name}_seed_5/* ${PROJECT_ROOT}/logs/experiment_test/evaluation_models/

#     local result_path="${PROJECT_ROOT}/logs/results/worker/$task_name"

#     python3 -u main.py \
#         setup=metaworld \
#         env=metaworld-mt1 \
#         worker.multitask.num_envs=1 \
#         experiment.mode=evaluate_collective_transformer \
#         experiment.robot_type="${robot_type}" \
#         env.benchmark.env_name=${task_name} \
#         experiment.evaluate_transformer="scripted" | tee -a $result_path
# }

# evaluate_task sawyer reach-v2
# evaluate_task sawyer push-v2
# evaluate_task sawyer peg-insert-side-v2
# evaluate_task sawyer door-open-v2
# evaluate_task sawyer window-open-v2
# evaluate_task sawyer window-close-v2
# evaluate_task sawyer drawer-open-v2
# evaluate_task sawyer button-press-topdown-v2
# evaluate_task sawyer faucet-open-v2
# evaluate_task sawyer pick-place-v2

# evaluate_task ur10e reach-v2
# evaluate_task ur10e push-v2
# evaluate_task ur10e peg-insert-side-v2
# evaluate_task ur10e door-open-v2
# evaluate_task ur10e window-open-v2          
# evaluate_task ur10e window-close-v2
# evaluate_task ur10e drawer-open-v2
# evaluate_task ur10e button-press-topdown-v2
# evaluate_task ur10e faucet-open-v2
# evaluate_task ur10e pick-place-v2

# evaluate_task panda reach-v2
# evaluate_task panda push-v2
# evaluate_task panda peg-insert-side-v2
# evaluate_task panda door-open-v2
# evaluate_task panda window-open-v2      
# evaluate_task panda window-close-v2
# evaluate_task panda drawer-open-v2
# evaluate_task panda button-press-topdown-v2
# evaluate_task panda faucet-open-v2
# evaluate_task panda pick-place-v2

# evaluate_task xarm7 reach-v2
# evaluate_task xarm7 push-v2
# evaluate_task xarm7 peg-insert-side-v2  
# evaluate_task xarm7 door-open-v2
# evaluate_task xarm7 window-open-v2      
# evaluate_task xarm7 window-close-v2
# evaluate_task xarm7 drawer-open-v2
# evaluate_task xarm7 button-press-topdown-v2
# evaluate_task xarm7 faucet-open-v2
# evaluate_task xarm7 pick-place-v2

# evaluate_task unitree_z1 reach-v2
# evaluate_task unitree_z1 push-v2    
# evaluate_task unitree_z1 peg-insert-side-v2
# evaluate_task unitree_z1 door-open-v2
# evaluate_task unitree_z1 window-open-v2      
# evaluate_task unitree_z1 window-close-v2
# evaluate_task unitree_z1 drawer-open-v2
# evaluate_task unitree_z1 button-press-topdown-v2
# evaluate_task unitree_z1 faucet-open-v2
# evaluate_task unitree_z1 pick-place-v2

# evaluate_task gen3 reach-v2
# evaluate_task gen3 push-v2    
# evaluate_task gen3 peg-insert-side-v2
# evaluate_task gen3 door-open-v2     
# evaluate_task gen3 window-open-v2      
# evaluate_task gen3 window-close-v2
# evaluate_task gen3 drawer-open-v2
# evaluate_task gen3 button-press-topdown-v2
# evaluate_task gen3 faucet-open-v2
# evaluate_task gen3 pick-place-v2
