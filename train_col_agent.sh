
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
        experiment.experiment="wm_200:1_random_tt" \
        setup.seed=$seed > log_seed_col_200:1_seed_${seed}.txt 2>&1 &

    sleep 5

done

wait

for seed in 3 4 5
do
    bash eval.sh sawyer $seed wm_200:1_random_tt &
    bash eval.sh xarm7 $seed wm_200:1_random_tt &
    bash eval.sh panda $seed wm_200:1_random_tt &
    bash eval.sh ur10e $seed wm_200:1_random_tt &
    bash eval.sh gen3 $seed wm_200:1_random_tt &
    bash eval.sh unitree_z1 $seed wm_200:1_random_tt &
    bash eval.sh viperx $seed wm_200:1_random_tt &
    bash eval.sh ur5e $seed wm_200:1_random_tt &
    bash eval.sh kuka $seed wm_200:1_random_tt &
    wait

done

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
        experiment.experiment="wm_1:1_random_tt" \
        setup.seed=$seed > log_seed_col_1:1_seed_${seed}.txt 2>&1 &

    sleep 5
done

wait

for seed in 3 4 5
do
    bash eval.sh sawyer $seed wm_1:1_random_tt &
    bash eval.sh xarm7 $seed wm_1:1_random_tt &
    bash eval.sh panda $seed wm_1:1_random_tt &
    bash eval.sh ur10e $seed wm_1:1_random_tt &
    bash eval.sh gen3 $seed wm_1:1_random_tt &
    bash eval.sh unitree_z1 $seed wm_1:1_random_tt &
    bash eval.sh viperx $seed wm_1:1_random_tt &
    bash eval.sh ur5e $seed wm_1:1_random_tt &
    bash eval.sh kuka $seed wm_1:1_random_tt &
    wait

done
