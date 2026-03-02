

# generate_distill_data() {
#     # 读取参数
#     local robot_type=$1
#     local task_name=$2
#     local gen_steps=$3      # 总生成步数 (Push: 160000, Pick-Place: 200000)
#     local sigma_start=$4    # 起始噪声 (e.g., 0.5)
#     local sigma_end=$5      # 结束噪声 (e.g., 0.25)

#     # 检查必要参数
#     if [[ -z "$task_name" || -z "$gen_steps" ]]; then
#         echo "Usage: generate_distill_data <robot> <task> <steps> <sigma_start> <sigma_end> [seed]"
#         return 1
#     fi

#     echo "========================================================"
#     echo " Generating Distill Data (Noise Injected Script Policy)"
#     echo " Robot: ${robot_type} | Task: ${task_name}"
#     echo " Steps: ${gen_steps} | Noise Sigma Start: ${sigma_start} | Noise Sigma End: ${sigma_end}"
#     echo " Mode: generate_distill_data"
#     echo "========================================================"

#     # 运行 Python 主程序
#     # 注意：这里我们注入了 experiment.mode=generate_distill_data
#     # 以及我们之前商定的噪声参数
#     python main.py \
#         setup=metaworld \
#         env=metaworld-mt1 \
#         worker.multitask.num_envs=1 \
#         experiment.mode=generate_distill_data \
#         experiment.robot_type=${robot_type} \
#         env.benchmark.env_name="${task_name}" \
#         experiment.num_distill_gen_steps=${gen_steps} \
#         experiment.distill_noise_sigma_start=${sigma_start} \
#         experiment.distill_noise_sigma_end=${sigma_end} \
#         # 确保其他参数（如 save_dir）正确，这里假设在 config 里有默认值
#         # 如果需要 override save_dir，可以在这里添加 setup.save_dir=...
# }

# generate_distill_data sawyer pick-place-v2 240000 0.5 0.2
# generate_distill_data ur5e pick-place-v2 240000 0.9 0.25
# generate_distill_data ur10e pick-place-v2 240000 1.0 0.3

# python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_world_model transformer_collective_network.world_model.load_on_init=False setup.seed=5 logger.use_tb=True
# python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=distill_collective_transformer setup.seed=5 logger.use_tb=True 

# bash eval_sawyer.sh setup.seed=5 &
# bash eval_xarm7.sh setup.seed=5 &
# bash eval_gen3.sh setup.seed=5 &
# wait
# bash eval_ur10e.sh setup.seed=5 &
# bash eval_ur5e.sh setup.seed=5 &
# bash eval_kuka.sh setup.seed=5 &
# wait
# bash eval_unitree_z1.sh setup.seed=5 &
# bash eval_panda.sh setup.seed=5 &
# bash eval_viperx.sh setup.seed=5 &
# wait

# mv logs/experiment_test/model_dir/model_col logs/experiment_test/model_dir/model_col_seed_5_200:1
# mv logs/experiment_test/model_dir/model_world logs/experiment_test/model_dir/model_world_seed_5_200:1

# python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_world_model transformer_collective_network.world_model.load_on_init=False setup.seed=4 logger.use_tb=True
# python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=distill_collective_transformer setup.seed=4 logger.use_tb=True 

# bash eval_sawyer.sh setup.seed=4 &
# bash eval_xarm7.sh setup.seed=4 &
# bash eval_gen3.sh setup.seed=4 &
# wait
# bash eval_ur10e.sh setup.seed=4 &
# bash eval_ur5e.sh setup.seed=4 &
# bash eval_kuka.sh setup.seed=4 &
# wait
# bash eval_unitree_z1.sh setup.seed=4 &
# bash eval_panda.sh setup.seed=4 &
# bash eval_viperx.sh setup.seed=4 &
# wait

# mv logs/experiment_test/model_dir/model_col logs/experiment_test/model_dir/model_col_seed_4_200:1
# mv logs/experiment_test/model_dir/model_world logs/experiment_test/model_dir/model_world_seed_4_200:1

# python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_world_model transformer_collective_network.world_model.load_on_init=False setup.seed=3 logger.use_tb=True
# python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=distill_collective_transformer setup.seed=3 logger.use_tb=True 

# bash eval_sawyer.sh setup.seed=3 &
# bash eval_xarm7.sh setup.seed=3 &
# bash eval_gen3.sh setup.seed=3 &
# wait
# bash eval_ur10e.sh setup.seed=3 &
# bash eval_ur5e.sh setup.seed=3 &
# bash eval_kuka.sh setup.seed=3 &
# wait
bash eval_unitree_z1.sh setup.seed=3 &
bash eval_panda.sh setup.seed=3 &
bash eval_viperx.sh setup.seed=3 &
wait

mv logs/experiment_test/model_dir/model_col logs/experiment_test/model_dir/model_col_seed_3_200:1
mv logs/experiment_test/model_dir/model_world logs/experiment_test/model_dir/model_world_seed_3_200:1

mv logs/experiment_test/model_dir/model_world_seed_3_200:1_4096 logs/experiment_test/model_dir/model_world

python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=distill_collective_transformer setup.seed=3 logger.use_tb=True 

bash eval_sawyer.sh setup.seed=3 &
bash eval_xarm7.sh setup.seed=3 &
bash eval_gen3.sh setup.seed=3 &
wait
bash eval_ur10e.sh setup.seed=3 &
bash eval_ur5e.sh setup.seed=3 &
bash eval_kuka.sh setup.seed=3 &
wait
bash eval_unitree_z1.sh setup.seed=3 &
bash eval_panda.sh setup.seed=3 &
bash eval_viperx.sh setup.seed=3 &
wait

mv logs/experiment_test/model_dir/model_col logs/experiment_test/model_dir/model_col_seed_3_200:1_4096
mv logs/experiment_test/model_dir/model_world logs/experiment_test/model_dir/model_world_seed_3_200:1_4096




# python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=distill_collective_transformer setup.seed=5 logger.use_tb=True transformer_collective_network.builder.use_world_model=False
# bash eval_sawyer.sh setup.seed=5 transformer_collective_network.builder.use_world_model=False &
# bash eval_xarm7.sh setup.seed=5 transformer_collective_network.builder.use_world_model=False &
# wait
# bash eval_gen3.sh setup.seed=5 transformer_collective_network.builder.use_world_model=False &
# bash eval_ur10e.sh setup.seed=5 transformer_collective_network.builder.use_world_model=False &
# wait
# bash eval_ur5e.sh setup.seed=5 transformer_collective_network.builder.use_world_model=False &
# bash eval_kuka.sh setup.seed=5 transformer_collective_network.builder.use_world_model=False &
# wait
# bash eval_unitree_z1.sh setup.seed=5 transformer_collective_network.builder.use_world_model=False &
# bash eval_panda.sh setup.seed=5 transformer_collective_network.builder.use_world_model=False &
# wait

# mv logs/experiment_test/model_dir/model_col logs/experiment_test/model_dir/model_col_seed_5_tt_con


# python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_world_model transformer_collective_network.world_model.load_on_init=False setup.seed=5 logger.use_tb=True
# python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=distill_collective_transformer setup.seed=5 logger.use_tb=True 

# bash eval_sawyer.sh setup.seed=5 &
# bash eval_xarm7.sh setup.seed=5 &
# wait
# bash eval_gen3.sh setup.seed=5 &
# bash eval_ur10e.sh setup.seed=5 &
# wait
# bash eval_ur5e.sh setup.seed=5 &
# bash eval_kuka.sh setup.seed=5 &
# wait
# bash eval_unitree_z1.sh setup.seed=5 &
# bash eval_panda.sh setup.seed=5 &
# wait

# mv logs/experiment_test/model_dir/model_col logs/experiment_test/model_dir/model_col_seed_5_1:1_tt_con
# mv logs/experiment_test/model_dir/model_world logs/experiment_test/model_dir/model_world_seed_5_1:1_tt_con