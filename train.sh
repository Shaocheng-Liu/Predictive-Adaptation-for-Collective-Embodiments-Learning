
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
#         transformer_collective_network.world_model.load_on_init=False \
#         logger.use_tb=True \
#         setup.seed=$seed \
        
#         > log_seed_wm_200:1_${seed}.txt 2>&1 &

#     sleep 5

# done
# wait

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
#         transformer_collective_network.world_model.load_on_init=False \
#         experiment.experiment="no_wm" \
#         logger.use_tb=True \
#         setup.seed=$seed \
        
#         > log_no_wm_seed_${seed}.txt 2>&1 &

#     sleep 5

# done

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
        logger.use_tb=True \
        experiment.experiment="no_wm" \
        setup.seed=$seed > log_seed_col_no_wm_seed_${seed}.txt 2>&1 &

    sleep 5

done

wait

for seed in 3 4 5
do
    bash eval.sh sawyer $seed no_wm &
    bash eval.sh xarm7 $seed no_wm &
    bash eval.sh panda $seed no_wm &
    bash eval.sh ur10e $seed no_wm &
    bash eval.sh gen3 $seed no_wm &
    bash eval.sh unitree_z1 $seed no_wm &
    bash eval.sh viperx $seed no_wm &
    bash eval.sh ur5e $seed no_wm &
    bash eval.sh kuka $seed no_wm &
    wait

done

