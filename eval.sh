

ROBOT="$1"
SEED="$2"      
EXP_NAME="$3"   

evaluate_col_agent(){
    local task_name="$1"
    if [[ -z "$task_name" ]]; then
        echo "Usage: $0 <task-name>"
        return 1
    fi

    echo "Evaluating col_network for task: $task_name"
    local log_dir="${PROJECT_ROOT}/logs/results/col/${task_name}_${EXP_NAME}"
    mkdir -p "$log_dir"

    local log_file="${log_dir}/eval_${ROBOT}_seed_${SEED}.log"
    python3 -u main.py \
        setup=metaworld \
        env=metaworld-mt1 \
        worker.multitask.num_envs=1 \
        experiment.robot_type="${ROBOT}" \
        experiment.mode=evaluate_collective_transformer \
        env.benchmark.env_name="$task_name" \
        experiment.evaluate_transformer="collective_network" \
        experiment.experiment="${EXP_NAME}"  \
        transformer_collective_network.transformer_encoder.representation_transformer.model_path="/root/bayes-tmp/sherma/RLCE/Transformer_RNN/checkpoints_${EXP_NAME}_seed_${SEED}/representation_cls_transformer_checkpoint.pth" \
        transformer_collective_network.transformer_encoder.prediction_head_cls.model_path="/root/bayes-tmp/sherma/RLCE/Transformer_RNN/checkpoints_${EXP_NAME}_seed_${SEED}/representation_cls_transformer_checkpoint.pth" \
        setup.seed=${SEED} > "$log_file" 2>&1
}


evaluate_col_agent reach-v2
evaluate_col_agent push-v2
evaluate_col_agent peg-insert-side-v2
evaluate_col_agent door-open-v2
evaluate_col_agent window-open-v2
evaluate_col_agent window-close-v2
evaluate_col_agent drawer-open-v2
evaluate_col_agent button-press-topdown-v2
evaluate_col_agent faucet-open-v2
evaluate_col_agent pick-place-v2