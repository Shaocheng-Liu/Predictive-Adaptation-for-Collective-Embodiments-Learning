
# for seed in 3 4 5
# do
#     echo "正在启动 Seed $seed (日志写入 log_seed_${seed}.txt)..."
    
#     # 重点：
#     # 1. > log_seed_${seed}.txt 2>&1 : 把标准输出和错误都导向单独的文件
#     # 2. python3 -u : 禁用缓存，让你能实时看到 log 更新，不用等缓冲区满
#     # 3. & : 后台运行
    
#     python3 -u main.py \
#         setup=metaworld \
#         env=metaworld-mt1 \
#         worker.multitask.num_envs=1 \
#         experiment.mode=train_world_model \
#         experiment.experiment="task_true" \
#         transformer_collective_network.world_model.load_on_init=False \
#         transformer_collective_network.transformer_encoder.representation_transformer.model_path="/root/bayes-tmp/sherma/RLCE/Transformer_RNN/checkpoints_task_true_seed_${seed}/representation_cls_transformer_checkpoint.pth" \
#         transformer_collective_network.transformer_encoder.prediction_head_cls.model_path="/root/bayes-tmp/sherma/RLCE/Transformer_RNN/checkpoints_task_true_seed_${seed}/representation_cls_transformer_checkpoint.pth" \
#         logger.use_tb=True \
#         setup.seed=$seed > log_wm_seed_task_true_${seed}.txt 2>&1 &

#     sleep 5

# done
# wait

for seed in 3 4 5
do
    echo "正在启动 Seed $seed (日志写入 log_seed_${seed}.txt)..."
    
    # 重点：
    # 1. > log_seed_${seed}.txt 2>&1 : 把标准输出和错误都导向单独的文件
    # 2. python3 -u : 禁用缓存，让你能实时看到 log 更新，不用等缓冲区满
    # 3. & : 后台运行
    
    python3 -u main.py \
        setup=metaworld \
        env=metaworld-mt1 \
        worker.multitask.num_envs=1 \
        experiment.mode=distill_collective_transformer \
        experiment.experiment="baseline_true" \
        transformer_collective_network.transformer_encoder.representation_transformer.model_path="/root/bayes-tmp/sherma/RLCE/Transformer_RNN/checkpoints_baseline_true_seed_${seed}/representation_cls_transformer_checkpoint.pth" \
        transformer_collective_network.transformer_encoder.prediction_head_cls.model_path="/root/bayes-tmp/sherma/RLCE/Transformer_RNN/checkpoints_baseline_true_seed_${seed}/representation_cls_transformer_checkpoint.pth" \
        logger.use_tb=True \
        setup.seed=$seed > log_col_baseline_true_seed_${seed}.txt 2>&1 &

    sleep 5

done

wait


for seed in 3 4 5
do
    echo "   -> Group A: 正在评估 Seed $seed (所有机械臂)..."
    bash eval.sh sawyer $seed baseline_true &
    bash eval.sh xarm7 $seed baseline_true &
    bash eval.sh panda $seed baseline_true &
    bash eval.sh ur10e $seed baseline_true &
    bash eval.sh gen3 $seed baseline_true &
    bash eval.sh unitree_z1 $seed baseline_true &
    bash eval.sh viperx $seed baseline_true &
    bash eval.sh ur5e $seed baseline_true &
    bash eval.sh kuka $seed baseline_true &
    
    # 这里的 wait 只会等待 Group A 内部的这 9 个 eval 任务
    wait 
    echo "   -> Group A: Seed $seed 评估完成。"
done
echo "✅ [Group A] 全部完成。"


# sed -i '/transformer_col_replay_buffer:/,/batch_size:/ s/batch_size: 256/batch_size: 1024/' config/replay_buffer/mtrl.yaml
# mv Transformer_RNN/checkpoints_task_dyn_true_256_seed_5 Transformer_RNN/checkpoints_task_dyn_true_seed_5
# sed -i 's/use_world_model: False/use_world_model: True/' config/transformer_collective_network/transformer_collective_sac.yaml
# sed -i 's/use_zeros: True/use_zeros: False/' config/transformer_collective_network/transformer_collective_sac.yaml
# sed -i 's/use_cls_prediction_head: False/use_cls_prediction_head: True/' config/transformer_collective_network/transformer_collective_sac.yaml
# for seed in 3 4 5
# do
#     echo "正在启动 Seed $seed (日志写入 log_seed_${seed}.txt)..."
    
#     # 重点：
#     # 1. > log_seed_${seed}.txt 2>&1 : 把标准输出和错误都导向单独的文件
#     # 2. python3 -u : 禁用缓存，让你能实时看到 log 更新，不用等缓冲区满
#     # 3. & : 后台运行
    
#     python3 -u main.py \
#         setup=metaworld \
#         env=metaworld-mt1 \
#         worker.multitask.num_envs=1 \
#         experiment.mode=train_world_model \
#         experiment.experiment="task_dyn_true_3" \
#         transformer_collective_network.world_model.load_on_init=False \
#         transformer_collective_network.transformer_encoder.representation_transformer.model_path="/root/bayes-tmp/sherma/RLCE/Transformer_RNN/checkpoints_task_dyn_true_3_seed_${seed}/representation_cls_transformer_checkpoint.pth" \
#         transformer_collective_network.transformer_encoder.prediction_head_cls.model_path="/root/bayes-tmp/sherma/RLCE/Transformer_RNN/checkpoints_task_dyn_true_3_seed_${seed}/representation_cls_transformer_checkpoint.pth" \
#         logger.use_tb=True \
#         setup.seed=$seed > log_wm_seed_task_dyn_true_3_${seed}.txt 2>&1 &

#     sleep 5

# done
# wait

# for seed in 3 4
# do
#     echo "正在启动 Seed $seed (日志写入 log_seed_${seed}.txt)..."
    
#     # 重点：
#     # 1. > log_seed_${seed}.txt 2>&1 : 把标准输出和错误都导向单独的文件
#     # 2. python3 -u : 禁用缓存，让你能实时看到 log 更新，不用等缓冲区满
#     # 3. & : 后台运行
    
#     python3 -u main.py \
#         setup=metaworld \
#         env=metaworld-mt1 \
#         worker.multitask.num_envs=1 \
#         experiment.mode=distill_collective_transformer \
#         experiment.experiment="task_dyn_true_3" \
#         transformer_collective_network.transformer_encoder.representation_transformer.model_path="/root/bayes-tmp/sherma/RLCE/Transformer_RNN/checkpoints_task_dyn_true_3_seed_${seed}/representation_cls_transformer_checkpoint.pth" \
#         transformer_collective_network.transformer_encoder.prediction_head_cls.model_path="/root/bayes-tmp/sherma/RLCE/Transformer_RNN/checkpoints_task_dyn_true_3_seed_${seed}/representation_cls_transformer_checkpoint.pth" \
#         logger.use_tb=True \
#         setup.seed=$seed > log_col_task_dyn_true_3_seed_${seed}.txt 2>&1 &

#     sleep 5

# done

# wait


# for seed in 3 4
# do
#     echo "   -> Group A: 正在评估 Seed $seed (所有机械臂)..."
#     bash eval.sh sawyer $seed task_dyn_true_3 &
#     # bash eval.sh xarm7 $seed task_dyn_true_3 &
#     bash eval.sh panda $seed task_dyn_true_3 &
#     bash eval.sh ur10e $seed task_dyn_true_3 &
#     bash eval.sh gen3 $seed task_dyn_true_3 &
#     bash eval.sh unitree_z1 $seed task_dyn_true_3 &
#     bash eval.sh viperx $seed task_dyn_true_3 &
#     bash eval.sh ur5e $seed task_dyn_true_3 &
#     bash eval.sh kuka $seed task_dyn_true_3 &
    
#     wait 
#     echo "   -> Group A: Seed $seed 评估完成。"
# done
# echo "✅ [Group A] 全部完成。"

# bash eval_kuka.sh kuka 5 task_dyn_true_3 &
# bash eval.sh ur10e 5 task_dyn_true_3 &
# wait

