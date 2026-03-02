
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""`Experiment` class manages the lifecycle of a multi-task model."""
import random

import time
from typing import Dict, List, Tuple
# import rospy
import hydra
import numpy as np
import torch
import mujoco

from mtrl.agent import utils as agent_utils
from mtrl.env.types import EnvType
from mtrl.env.vec_env import VecEnv  # type: ignore
from mtrl.experiment import collective_experiment
from mtrl.replay_buffer import ReplayBufferSample
from mtrl.utils.types import ConfigType, EnvMetaDataType, EnvsDictType, ListConfigType

import torch.distributions as D
from scipy.stats import norm
import math


class Experiment(collective_experiment.Experiment):
    def __init__(self, config: ConfigType, experiment_id: str = "0"):
        """Experiment Class to manage the lifecycle of a collective learning model.

        Args:
            config (ConfigType):
            experiment_id (str, optional): Defaults to "0".
        """
        super().__init__(config, experiment_id)
        self.eval_modes_to_env_ids = self.create_eval_modes_to_env_ids()
        self.should_reset_env_manually = False
        self.metrics_to_track = {
            x[0] for x in self.config.metrics["train"] if not x[0].endswith("_")
        }
        # Early stopping:
        from collections import deque
        self.success_history = deque(maxlen=self.config.experiment.early_stopping_window)

    def _get_robot_prefix(self) -> str:
        """Get robot type prefix for filenames.
        
        Returns:
            str: Robot type prefix with underscore (e.g., 'ur5e_') or empty string if not available.
        """
        return f"{self.robot_type}_" if hasattr(self, 'robot_type') and self.robot_type else ""

    def build_envs(self) -> Tuple[EnvsDictType, EnvMetaDataType]:
        """Build environments and return env-related metadata"""
        if "dmcontrol" not in self.config.env.name:
            raise NotImplementedError
        envs: EnvsDictType = {}
        mode = "train"
        env_id_list = self.config.env[mode]
        num_envs = len(env_id_list)
        seed_list = list(range(1, num_envs + 1))
        mode_list = [mode for _ in range(num_envs)]

        envs[mode] = hydra.utils.instantiate(
            self.config.env.builder,
            env_id_list=env_id_list,
            seed_list=seed_list,
            mode_list=mode_list,
        )
        envs["eval"] = self._create_dmcontrol_vec_envs_for_eval()
        metadata = self.get_env_metadata(env=envs["train"])
        return envs, metadata

    def create_eval_modes_to_env_ids(self) -> Dict[str, List[int]]:
        """Map each eval mode to a list of environment index.

            The eval modes are of the form `eval_xyz` where `xyz` specifies
            the specific type of evaluation. For example. `eval_interpolation`
            means that we are using interpolation environments for evaluation.
            The eval moe can also be set to just `eval`.

        Returns:
            Dict[str, List[int]]: dictionary with different eval modes as
                keys and list of environment index as values.
        """
        eval_modes_to_env_ids: Dict[str, List[int]] = {}
        eval_modes = [
            key for key in self.config.metrics.keys() if not key.startswith("train")
        ]
        for mode in eval_modes:
            if "_" in mode:
                _mode, _submode = mode.split("_")
                env_ids = self.config.env[_mode][_submode]
                eval_modes_to_env_ids[mode] = env_ids
            elif mode != "eval":
                raise ValueError(f"eval mode = `{mode}`` is not supported.")
        return eval_modes_to_env_ids

    def _create_dmcontrol_vec_envs_for_eval(self) -> EnvType:
        """Method to create the vec env with multiple copies of the same
        environment. It is useful when evaluating the agent multiple times
        in the same env.

        The vec env is organized as follows - number of modes x number of tasks per mode x number of episodes per task

        """

        env_id_list: List[str] = []
        seed_list: List[int] = []
        mode_list: List[str] = []
        num_episodes_per_env = self.config.experiment.num_eval_episodes
        for mode in self.config.metrics.keys():
            if mode == "train":
                continue

            if "_" in mode:
                _mode, _submode = mode.split("_")
                if _mode != "eval":
                    raise ValueError("`mode` does not start with `eval_`")
                if not isinstance(self.config.env.eval, ConfigType):
                    raise ValueError(
                        f"""`self.config.env.eval` should either be a DictConfig.
                        Detected type is {type(self.config.env.eval)}"""
                    )
                if _submode in self.config.env.eval:
                    for _id in self.config.env[_mode][_submode]:
                        env_id_list += [_id for _ in range(num_episodes_per_env)]
                        seed_list += list(range(1, num_episodes_per_env + 1))
                        mode_list += [_submode for _ in range(num_episodes_per_env)]
            elif mode == "eval":
                if isinstance(self.config.env.eval, ListConfigType):
                    for _id in self.config.env[mode]:
                        env_id_list += [_id for _ in range(num_episodes_per_env)]
                        seed_list += list(range(1, num_episodes_per_env + 1))
                        mode_list += [mode for _ in range(num_episodes_per_env)]
            else:
                raise ValueError(f"eval mode = `{mode}` is not supported.")
        env = hydra.utils.instantiate(
            self.config.env.builder,
            env_id_list=env_id_list,
            seed_list=seed_list,
            mode_list=mode_list,
        )

        return env
    
    def run(self):
        if self.config.experiment.mode == 'evaluate_collective_transformer':
            self.evaluate_collective_transformer()
        elif self.config.experiment.mode == 'online_distill_collective_transformer':
            self.run_online_distillation()
        elif self.config.experiment.mode == 'evaluate_real':
            self.evaluate_real()
        elif self.config.experiment.mode == 'evaluate_sim':
            self.evaluate_sim()
        else:
            raise NotImplemented

    def save_all_buffers_and_models(self, step):
        print("Saving all buffers and models ...")
        exp_config = self.config.experiment
        for i in range(self.num_envs):
            if exp_config.save.model:
                self.agent[i].save(
                    self.model_dir[i],
                    step=step,
                    retain_last_n=exp_config.save.model.retain_last_n,
                )
            if exp_config.save.buffer.should_save:
                self.replay_buffer[i].save(
                    self.buffer_dir[i],
                    size_per_chunk=exp_config.save.buffer.size_per_chunk,
                    num_samples_to_save=exp_config.save.buffer.num_samples_to_save,
                )
                self.replay_buffer_distill_tmp[i].save(
                    self.buffer_dir_distill_tmp[i],
                    size_per_chunk=exp_config.save.buffer.size_per_chunk,
                    num_samples_to_save=exp_config.save.buffer.num_samples_to_save,
                )


    def evaluate_collective_transformer(self):
        """Run the experiment."""
        exp_config = self.config.experiment

        assert self.col_start_step >= 0
        episode = self.col_start_step // self.max_episode_steps
        start_time = time.time()
        count = np.zeros(self.config.worker.multitask.num_envs)

        for step in range(exp_config.evaluate_episodes):
            # evaluate agent periodically
            self.logger.log("train/duration", time.time() - start_time, step)
            start_time = time.time()
            print(exp_config.evaluate_transformer)
            if exp_config.evaluate_transformer=="collective_network":
                result = self.evaluate_transformer(
                    agent=self.col_agent, vec_env=self.envs["eval"], step=step, episode=episode, seq_len=self.seq_len, sample_actions=(step==0), reward_unscaled=exp_config.use_unscaled
                )
            elif exp_config.evaluate_transformer=="agent":
                result = self.evaluate_vec_env_of_tasks(
                        vec_env=self.envs["eval"], step=step, episode=episode, agent="worker"
                    )
            else:
                result = self.evaluate_vec_env_of_tasks(
                        vec_env=self.envs["eval"], step=step, episode=episode, agent="scripted"
                    )
            count += result
            self.reset_goal_locations_in_order(env="eval")

        if exp_config.evaluate_transformer=="collective_network":
            # for step in range(exp_config.evaluate_critic_rounds):
            #     self.col_agent.evaluate_critic(self.replay_buffer_distill, self.logger, step, seq_len=self.seq_len, tb_log=True)
                self.logger.log("train/duration", time.time() - start_time, step)
                start_time = time.time()
                self.logger.dump(step)

        if self.config.experiment.save_video:
            for i in range(exp_config.recording_eval_episodes):
                print('start recording videos ...')
                if exp_config.evaluate_transformer=="collective_network":
                    self.record_videos_for_transformer(
                        self.col_agent, 
                        tag=f"sample_{i}", 
                        seq_len=self.seq_len,
                        camera_azimuth=350 ,     # ← 添加这一行即可看到正左侧视角
                        camera_elevation=-20.0
                    )
                elif exp_config.evaluate_transformer=="agent":
                    self.record_videos(eval_agent="worker", tag=f"sample_{i}")
                else:
                    self.record_videos(eval_agent="scripted", tag=f"sample_{i}")

                print('video recording finished. Check folder:{}'.format(self.video.dir_name))

        self.close_envs()

        print("finished learning")
        print(f"Evaluation: robot {self.robot_type} {count} from {exp_config.evaluate_episodes}")

    def evaluate_real(self):
        """
        实机评估模式 (Real Robot Evaluation)
        
        直接在 KUKA 真实机器人上运行 collective transformer 模型。
        每个任务的传感器类型、跟踪模式、偏移参数都在此函数中配置。
        
        支持的任务:
            reach-v2         - 颜色相机追踪目标
            push-v2          - 颜色相机追踪物体，手动输入目标
            pick-place-v2    - 颜色相机追踪物体，手动输入目标
            drawer-open-v2   - Apriltag 追踪把手，自动计算目标
            window-close-v2  - Apriltag 追踪把手，自动计算目标
            window-open-v2   - Apriltag 追踪把手，自动计算目标
            button-press-topdown-v2 - Apriltag 追踪按钮，自动计算目标
        
        使用方法 (run.sh):
            evaluate_real reach-v2
            evaluate_real push-v2
            ...
        """
        import rospy

        print("=" * 60)
        print("Starting Real Robot Evaluation (evaluate_real mode)")
        print("=" * 60)
        
        exp_config = self.config.experiment
        
        # ==================== 基本信息 ====================
        task_name = self.task_names[0] if self.task_names else "reach-v2"
        task_num = self.task_num[0] if len(self.task_num) > 0 else 0
        num_episodes = exp_config.get("real_robot_episodes", 1)
        max_steps = exp_config.get("real_robot_max_steps", 200)
        
        print(f"Task: {task_name}")
        print(f"Task number: {task_num}")
        print(f"Model loaded from step: {self.col_start_step}")
        print(f"Episodes: {num_episodes}, Max steps: {max_steps}")
        
        # ==================== 环境导入 ====================
        from mtrl.env.real_envs.reach_env import KukaReachEnv
        from mtrl.env.real_envs.push_env import KukaPushEnv
        from mtrl.env.real_envs.pick_place_env import KukaPickPlaceEnv
        from mtrl.env.real_envs.drawer_open_env import KukaDrawerOpenEnv
        from mtrl.env.real_envs.window_close_env import KukaWindowCloseEnv
        from mtrl.env.real_envs.window_open_env import KukaWindowOpenEnv
        from mtrl.env.real_envs.button_press_topdown_env import KukaButtonPressTopdownEnv
        
        # ==================================================================
        #                   ★★★ 任务配置 (PER-TASK SETUP) ★★★
        # ==================================================================
        # 
        # 每个任务需要配置:
        #   1. observation_offset [x,y,z] (仿真坐标系) - 将真机坐标映射到训练坐标范围
        #   2. camera_offset [x,y,z] (真实坐标系) - 相机系统性校正
        #   3. apriltag_offset [x,y,z] (真实坐标系) - Apriltag 标记到操作点的偏移
        #   4. 物体/目标位置 (真实坐标系)
        #   5. ee_init_position [x,y,z] (真实坐标系) - 机械臂初始位置
        #
        # 所有标记 "TODO" 的参数需要根据实际实验环境修改！
        # ==================================================================
        
        if task_name == "reach-v2":
            # ==================== Reach ====================
            # 传感器: 颜色相机
            # 追踪: track_goal_realtime=True (实时追踪目标颜色标记)
            # 不需要追踪物体 (没有可操作物体)
            real_env = KukaReachEnv(
                max_episode_steps=max_steps,
                use_camera_apriltag=True,
                track_object_realtime=False,
                track_goal_realtime=True,           # 实时追踪目标
                object_tag_id=1,                 # TODO: 目标标记颜色，根据实际颜色修改
                observation_offset=[0, 0.1, 0.0],   # TODO: 仿真坐标系偏移
                camera_offset=[0, -0.12, 0],             # TODO: 相机校正偏移 (真实坐标系)
            )
            # 初始位置
            real_env.set_ee_init_position(np.array([0.5, 0.0, 0.2]))  # TODO: 真实坐标系
            # 目标位置: 由相机实时追踪，也可以手动设置:
            # real_env.set_goal(np.array([0.65, -0.1, 0.2]))  # TODO: 如不用相机，取消注释
            print(f"[Reach] object_tag_id=1, track_goal_realtime=True")
        
        elif task_name == "push-v2":
            # ==================== Push ====================
            # 传感器: 颜色相机
            # 追踪: track_object_realtime=True (物体被推动，位置变化)
            # 目标: 手动输入 (固定位置)
            real_env = KukaPushEnv(
                max_episode_steps=max_steps,
                use_camera_apriltag=True,
                track_object_realtime=True,          # 实时追踪物体
                track_goal_realtime=False,            # 目标固定
                object_tag_id=1,                   # TODO: 物体标记颜色
                observation_offset=[0, 0.1, 0.0],     # TODO: 仿真坐标系偏移
                camera_offset=[0, -0.15, 0.02],               # TODO: 相机校正偏移 (真实坐标系)
            )
            real_env.set_ee_init_position(np.array([0.5, 0.0, 0.2]))  # TODO: 真实坐标系
            real_env.set_goal(np.array([0.75, 0.1, 0.02]))             # TODO: 目标位置 (真实坐标系)
            # 物体初始位置由相机在 reset() 时自动检测
            print(f"[Push] object_tag_id=1, goal manually set")
        
        elif task_name == "pick-place-v2":
            # ==================== Pick-Place ====================
            # 传感器: 颜色相机
            # 追踪: track_object_realtime=True (物体被抓取后位置变化)
            # 目标: 手动输入 (固定空中位置)
            real_env = KukaPickPlaceEnv(
                max_episode_steps=max_steps,
                use_camera_apriltag=True,
                track_object_realtime=True,
                track_goal_realtime=False,
                object_tag_id=1,                   # TODO: 物体标记颜色
                observation_offset=[0, 0.1, 0.0],     # TODO: 仿真坐标系偏移
                camera_offset=[0, -0.15, 0.02],               # TODO: 相机校正偏移 (真实坐标系)
            )
            real_env.set_ee_init_position(np.array([0.5, 0.0, 0.2]))  # TODO: 真实坐标系
            real_env.set_goal(np.array([0.7, -0.08, 0.2]))              # TODO: 目标位置 (真实坐标系, Z>0.1表示空中)
            print(f"[PickPlace] object_tag_id=1, goal manually set")
        
        elif task_name == "drawer-open-v2":
            # ==================== Drawer Open ====================
            # 传感器: Apriltag 相机
            # 追踪: track_object_realtime=True (把手随抽屉移动)
            # 目标: 从把手位置自动计算 (handle + [0, -0.20, 0.09])
            real_env = KukaDrawerOpenEnv(
                max_episode_steps=max_steps,
                use_camera_apriltag=True,
                track_object_realtime=True,
                track_goal_realtime=False,
                object_tag_id=1,                      # TODO: 把手上的 Apriltag ID
                apriltag_offset=[0.02, -0.17, 0],             # TODO: Apriltag 标记到把手操作点的偏移 (真实坐标系)
                observation_offset=[0, 0.1, 0.03],      # TODO: 仿真坐标系偏移
                camera_offset=[0, -0.1, 0],                # TODO: 相机校正偏移 (真实坐标系)
            )
            real_env.set_ee_init_position(np.array([0.5, 0.0, 0.2]))  # TODO: 真实坐标系
            # 使用 Apriltag 自动检测把手位置并计算 goal:
            real_env.set_handle_from_apriltag(tag_id=1)  # TODO: Apriltag ID
            # 或手动设置: real_env.set_handle_position(np.array([0.7, 0.0, 0.15]))
            print(f"[DrawerOpen] Apriltag tag_id=0, goal auto-computed from handle")
        
        elif task_name == "window-close-v2":
            # ==================== Window Close ====================
            # 传感器: Apriltag 相机
            # 追踪: track_object_realtime=True (把手随窗户滑动)
            # 目标: 从把手位置自动计算 (handle - [0.2, 0, 0])
            real_env = KukaWindowCloseEnv(
                max_episode_steps=max_steps,
                use_camera_apriltag=True,
                track_object_realtime=False,
                track_goal_realtime=False,
                object_tag_id=1,                      # TODO: 把手上的 Apriltag ID
                apriltag_offset=[0, 0, 0],             # TODO: Apriltag 偏移 (真实坐标系)
                observation_offset=[0, 0.1, 0.0],      # TODO: 仿真坐标系偏移
                camera_offset=[0, -0.13, 0.04],                # TODO: 相机校正偏移 (真实坐标系)
            )
            real_env.set_ee_init_position(np.array([0.5, 0.0, 0.2]))  # TODO: 真实坐标系
            real_env.set_handle_from_apriltag(tag_id=1)  # TODO: Apriltag ID
            print(f"[WindowClose] Apriltag tag_id=0, goal auto-computed from handle")
        
        elif task_name == "window-open-v2":
            # ==================== Window Open ====================
            # 传感器: Apriltag 相机
            # 追踪: track_object_realtime=True (把手随窗户滑动)
            # 目标: 从把手位置自动计算 (handle + [0.2, 0, 0])
            real_env = KukaWindowOpenEnv(
                max_episode_steps=max_steps,
                use_camera_apriltag=True,
                track_object_realtime=False,
                track_goal_realtime=False,
                object_tag_id=1,                      # TODO: 把手上的 Apriltag ID
                apriltag_offset=[0, 0, 0],             # TODO: Apriltag 偏移 (真实坐标系)
                observation_offset=[0, 0.1, 0.0],      # TODO: 仿真坐标系偏移
                camera_offset=[0, -0.12, 0.03],                # TODO: 相机校正偏移 (真实坐标系)
            )
            real_env.set_ee_init_position(np.array([0.5, 0.0, 0.2]))  # TODO: 真实坐标系
            real_env.set_handle_from_apriltag(tag_id=1)  # TODO: Apriltag ID
            print(f"[WindowOpen] Apriltag tag_id=0, goal auto-computed from handle")
        
        elif task_name == "button-press-topdown-v2":
            # ==================== Button Press Topdown ====================
            # 传感器: Apriltag 相机
            # 追踪: track_object_realtime=True (按钮 Z 值在按下过程中变化)
            # 目标: 从按钮位置自动计算 (button_Z - press_depth)
            real_env = KukaButtonPressTopdownEnv(
                max_episode_steps=max_steps,
                use_camera_apriltag=True,
                track_object_realtime=True,
                track_goal_realtime=False,
                object_tag_id=1,                      # TODO: 按钮上的 Apriltag ID
                apriltag_offset=[0.055, -0.06, 0],             # TODO: Apriltag 到按钮顶面中心的偏移 (真实坐标系)
                observation_offset=[0, 0.1, 0.09],      # TODO: 仿真坐标系偏移
                camera_offset=[0, -0.08, 0.03],                # TODO: 相机校正偏移 (真实坐标系)
            )
            real_env.set_ee_init_position(np.array([0.4, 0.0, 0.2]))  # TODO: 真实坐标系
            real_env.set_button_from_apriltag(tag_id=1)  # TODO: Apriltag ID
            print(f"[ButtonPress] Apriltag tag_id=0, goal auto-computed from button")
        
        else:
            raise ValueError(
                f"未知任务: {task_name}. "
                f"支持的任务: reach-v2, push-v2, pick-place-v2, drawer-open-v2, "
                f"window-close-v2, window-open-v2, button-press-topdown-v2"
            )
        
        print(f"Environment created: {real_env.__class__.__name__}")
        print(f"Object position: {getattr(real_env, 'object_position', 'N/A')}")
        print(f"Goal position: {getattr(real_env, 'goal', 'N/A')}")
        
        # ==================== 推理循环 (Inference Loop) ====================
        # Rolling buffer 方式与 evaluate_sim 完全一致
        T = self.seq_len
        B = 1
        action_dim = 4
        device = self.device
        
        total_success = 0
        total_rewards = []
        
        for episode in range(num_episodes):
            print(f"\n{'='*40} Episode {episode + 1}/{num_episodes} {'='*40}")
            
            # === 1. Reset 环境 ===
            obs = real_env.reset()
            rospy.sleep(1.0)

            # ============================== 若无法人为设定ee_init_pos，替代做法 ==========================
            # rospy.sleep(1)
            # print('current position: ', real_env.ee_current_position)
            # while not rospy.is_shutdown():
            #     diff = ([0.5,0,0.2] - real_env.ee_current_position)*0.1
            #     s_target_position = real_env.ee_current_position + diff
            #     if np.linalg.norm([0.5,0,0.2] - real_env.ee_current_position) > 0.005:
            #         step_pos = real_env._set_ee_pose(s_target_position[0],s_target_position[1],s_target_position[2])
            #         real_env._robot_pub.publish(step_pos)
            #         real_env._rate.sleep()
            #     else:
            #         print("reach to: ", real_env.ee_current_position)
            #         break
            # rospy.sleep(3)
            
            # === 2. 初始化 rolling buffer ===
            first_state_np = real_env.get_state_for_transformer()
            first_state = torch.from_numpy(first_state_np).float().to(device)
            
            # [B, T, state_dim]: 用初始状态填满整个序列
            state_buffer = first_state.unsqueeze(0).unsqueeze(1).repeat(1, T, 1)
            action_buffer = torch.zeros((B, T, action_dim), device=device)
            reward_buffer = torch.zeros((B, T, 1), device=device)
            
            task_ids_tensor = torch.tensor([task_num], device=device)
            
            episode_reward = 0.0
            success = False
            step = 0
            
            for step in range(max_steps):
                print("state_buffer: ", state_buffer)
                # print("action buffer: ", action_buffer)

                # === 3. 模型推理 ===
                with agent_utils.eval_mode(self.col_agent):
                    action_out = self.col_agent.select_action(
                        states=state_buffer,               # [B, T, 21]
                        actions=action_buffer[:, 1:, :],   # [B, T-1, 4]
                        rewards=reward_buffer[:, 1:, :],   # [B, T-1, 1]
                        task_ids=task_ids_tensor
                    )
                
                # === 4. 动作后处理 ===
                if isinstance(action_out, torch.Tensor):
                    action_np = action_out.detach().cpu().numpy()
                else:
                    action_np = action_out
                if action_np.ndim == 1:
                    action_np = action_np[np.newaxis, :]
                
                # === 5. 环境交互 ===
                obs, reward, done, info = real_env.step(action_np[0])
                
                # === 6. Rolling buffer 更新 ===
                state_buffer = torch.roll(state_buffer, shifts=-1, dims=1)
                action_buffer = torch.roll(action_buffer, shifts=-1, dims=1)
                reward_buffer = torch.roll(reward_buffer, shifts=-1, dims=1)
                
                new_state = torch.from_numpy(
                    real_env.get_state_for_transformer()
                ).float().to(device)
                
                if isinstance(reward, np.ndarray):
                    new_reward = torch.from_numpy(reward).float().to(device)
                else:
                    new_reward = torch.tensor(reward, dtype=torch.float32, device=device)
                
                state_buffer[:, -1, :] = new_state
                action_buffer[:, -1, :] = torch.from_numpy(action_np).float().to(device)
                reward_buffer[:, -1, :] = new_reward.reshape(B, 1)
                
                # === 7. 统计 ===
                episode_reward += float(reward)
                success = bool(info.get('success', False))
                
                if step % 20 == 0:
                    print(f"  Step {step}: reward={float(reward):.3f}, "
                          f"ee={real_env.ee_current_position}, "
                          f"obj={real_env.object_position}")
                
                if done or success:
                    break
            
            print(f"Episode {episode + 1}: steps={step+1}, reward={episode_reward:.3f}, success={success}")
            total_rewards.append(episode_reward)
            if success:
                total_success += 1
        
        # ==================== 结果汇总 ====================
        print("\n" + "=" * 60)
        print(f"Real Robot Evaluation Complete: {task_name}")
        print(f"Success rate: {total_success}/{num_episodes} ({100*total_success/max(num_episodes,1):.1f}%)")
        if total_rewards:
            print(f"Average reward: {np.mean(total_rewards):.3f} (+/- {np.std(total_rewards):.3f})")
        print("=" * 60)
    


    def evaluate_sim(self):
        """
        Simulation-based evaluation mode.
        
        This method runs the collective transformer model in MuJoCo simulation,
        using the real_envs reward function and task logic. The simulation
        wrapper "tricks" the real_envs classes into thinking they're controlling
        real hardware, but actually routes all commands to MuJoCo.
        
        Key features:
        - Uses SimulatedRealEnv wrapper for MuJoCo-backed execution
        - Preserves real_envs reward functions and success criteria
        - Supports hardcoded goal_position and object_position
        - Real-time MuJoCo rendering visualization
        
        Configuration options (via config.experiment):
        - sim_episodes: Number of evaluation episodes (default: 10)
        - sim_max_steps: Max steps per episode (default: 200)
        - sim_render: Enable/disable rendering (default: True)
        - sim_goal_position: [x, y, z] goal override (optional)
        - sim_object_position: [x, y, z] object override (optional)
        """
        print("=" * 60)
        print("Starting Simulation Evaluation (evaluate_sim mode)")
        print("=" * 60)
        
        exp_config = self.config.experiment
        
        # Get task info
        task_name = self.task_names[0] if self.task_names else "reach-v2"
        task_num = self.task_num[0] if len(self.task_num) > 0 else 0
        
        print(f"Task: {task_name}")
        print(f"Task number: {task_num}")
        print(f"Model loaded from step: {self.col_start_step}")
        print(f"Max episode steps: {self.max_episode_steps}")
        print(f"Device: {self.device}")
        
        # Import simulated environment wrapper
        try:
            from mtrl.env.real_envs.sim_wrapper import SimulatedRealEnv, create_simulated_real_env
        except ImportError as e:
            print(f"[evaluate_sim] Error: Could not import sim_wrapper: {e}")
            print("[evaluate_sim] Please ensure MetaWorld is properly installed.")
            return
        
        # Get simulation configuration
        num_episodes = exp_config.get("sim_episodes", 10)
        max_steps = exp_config.get("sim_max_steps", 200)
        render_enabled = exp_config.get("sim_render", True)
        
        # Get position overrides from config
        goal_position = None
        object_position = None
        handle_position = None
        
        # For push/pick-place: set goal_position and object_position
        if hasattr(exp_config, "sim_goal_position") and exp_config.sim_goal_position:
            goal_position = np.array(exp_config.sim_goal_position)
            print(f"[evaluate_sim] Using custom goal position: {goal_position}")
        
        if hasattr(exp_config, "sim_object_position") and exp_config.sim_object_position:
            object_position = np.array(exp_config.sim_object_position)
            print(f"[evaluate_sim] Using custom object position: {object_position}")
        
        # For fixture tasks (drawer/window): set handle_position
        if hasattr(exp_config, "sim_handle_position") and exp_config.sim_handle_position:
            handle_position = np.array(exp_config.sim_handle_position)
            print(f"[evaluate_sim] Using custom handle position: {handle_position}")
        
        print(f"\nRunning {num_episodes} simulation episode(s), max {max_steps} steps each")
        print(f"Rendering: {render_enabled}")
        
        # Create simulated environment
        sim_env = create_simulated_real_env(
            task_name=task_name,
            goal_position=goal_position,
            object_position=object_position,
            handle_position=handle_position,
            max_episode_steps=max_steps,
            observation_offset=[0, 0.1, 0],
            render_mode="human" if render_enabled else None,
        )
        sim_env.set_ee_start_position(np.array([0.0, 0.4, 0.4]))
        
        # Buffer dimensions
        T = self.seq_len
        B = 1  # Batch size for simulation
        state_dim = 21
        action_dim = 4
        device = self.device
        
        total_success = 0
        total_rewards = []
        
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            
            # === 1. Reset and initialize buffers ===
            obs = sim_env.reset()
            
            # Get initial state (21-dim)
            first_state_np = sim_env.get_state_for_transformer()
            first_state = torch.from_numpy(first_state_np).float().to(device)
            
            # [State Buffer]: [B, T, state_dim]
            state_buffer = first_state.unsqueeze(0).unsqueeze(1).repeat(1, T, 1)
            
            # [Action/Reward Buffer]: Fill with zeros [B, T, dim]
            action_buffer = torch.zeros((B, T, action_dim), device=device)
            reward_buffer = torch.zeros((B, T, 1), device=device)
            
            # Cache task IDs
            task_ids_tensor = torch.tensor([task_num], device=device)
            
            episode_reward = 0.0
            success = False
            step = 0
            
            for step in range(max_steps):
                # === 2. Inference ===
                with agent_utils.eval_mode(self.col_agent):
                    active_states = state_buffer
                    active_actions = action_buffer[:, 1:, :]
                    active_rewards = reward_buffer[:, 1:, :]

                    action_out = self.col_agent.select_action(
                        states=active_states,
                        actions=active_actions,
                        rewards=active_rewards,
                        task_ids=task_ids_tensor
                    )
                
                # === 3. Data type conversion ===
                if isinstance(action_out, torch.Tensor):
                    action_np = action_out.detach().cpu().numpy()
                else:
                    action_np = action_out
                
                if action_np.ndim == 1:
                    action_np = action_np[np.newaxis, :]
                
                # === 4. Environment interaction ===
                obs, reward, done, info = sim_env.step(action_np[0])
                
                # === 5. Rolling buffer update ===
                state_buffer = torch.roll(state_buffer, shifts=-1, dims=1)
                action_buffer = torch.roll(action_buffer, shifts=-1, dims=1)
                reward_buffer = torch.roll(reward_buffer, shifts=-1, dims=1)
                
                # Prepare new data
                new_state_np = sim_env.get_state_for_transformer()
                new_state = torch.from_numpy(new_state_np).float().to(device)
                
                if isinstance(reward, np.ndarray):
                    new_reward = torch.from_numpy(reward).float().to(device)
                else:
                    new_reward = torch.tensor(reward, dtype=torch.float32, device=device)
                
                new_action = torch.from_numpy(action_np).float().to(device)
                
                # Fill buffer tail
                state_buffer[:, -1, :] = new_state
                action_buffer[:, -1, :] = new_action
                reward_buffer[:, -1, :] = new_reward.reshape(B, 1)
                
                # === 6. Statistics ===
                episode_reward += float(reward)
                success = bool(info.get('success', False))
                
                if done or success:
                    break
            
            print(f"Episode {episode + 1}: steps={step+1}, reward={episode_reward:.3f}, success={success}")
            total_rewards.append(episode_reward)
            
            if success:
                total_success += 1
        
        print("\n" + "=" * 60)
        print(f"Simulation Evaluation Complete")
        print(f"Success rate: {total_success}/{num_episodes} ({100*total_success/num_episodes:.1f}%)")
        print(f"Average reward: {np.mean(total_rewards):.3f} (+/- {np.std(total_rewards):.3f})")
        print("=" * 60)
        
        sim_env.close()

    def _test_model_inference(self, task_num):
        """
        Test model inference without real robot.
        Useful for verifying model loading and action selection.
        """
        print("Testing model inference with dummy inputs...")
        
        # Create dummy trajectory data
        batch_size = 1
        seq_len = self.seq_len
        state_dim = 21
        action_dim = 4
        
        # Dummy states: [batch, seq_len, state_dim]
        states = torch.zeros(batch_size, seq_len, state_dim).to(self.device)
        # Dummy actions: [batch, seq_len-1, action_dim]
        actions = torch.zeros(batch_size, seq_len - 1, action_dim).to(self.device)
        # Dummy rewards: [batch, seq_len-1, 1]
        rewards = torch.zeros(batch_size, seq_len - 1, 1).to(self.device)
        # Task ID
        task_ids = torch.tensor([task_num]).to(self.device)
        
        print(f"Input shapes:")
        print(f"  states: {states.shape}")
        print(f"  actions: {actions.shape}")
        print(f"  rewards: {rewards.shape}")
        print(f"  task_ids: {task_ids.shape}")
        
        # Run inference
        with agent_utils.eval_mode(self.col_agent):
            action = self.col_agent.select_action(
                states=states,
                actions=actions,
                rewards=rewards,
                task_ids=task_ids
            )
        
        print(f"\nOutput action: {action}")
        print(f"Action shape: {action.shape}")
        print("\nModel inference test PASSED!")
    
    def _test_model_inference(self, task_num):
        """
        Test model inference without real robot.
        Useful for verifying model loading and action selection.
        """
        print("Testing model inference with dummy inputs...")
        
        # Create dummy trajectory data
        batch_size = 1
        seq_len = self.seq_len
        state_dim = 21
        action_dim = 4
        
        # Dummy states: [batch, seq_len, state_dim]
        states = torch.zeros(batch_size, seq_len, state_dim).to(self.device)
        # Dummy actions: [batch, seq_len-1, action_dim]
        actions = torch.zeros(batch_size, seq_len - 1, action_dim).to(self.device)
        # Dummy rewards: [batch, seq_len-1, 1]
        rewards = torch.zeros(batch_size, seq_len - 1, 1).to(self.device)
        # Task ID
        task_ids = torch.tensor([task_num]).to(self.device)
        
        print(f"Input shapes:")
        print(f"  states: {states.shape}")
        print(f"  actions: {actions.shape}")
        print(f"  rewards: {rewards.shape}")
        print(f"  task_ids: {task_ids.shape}")
        
        # Run inference
        with agent_utils.eval_mode(self.col_agent):
            action = self.col_agent.select_action(
                states=states,
                actions=actions,
                rewards=rewards,
                task_ids=task_ids
            )
        
        print(f"\nOutput action: {action}")
        print(f"Action shape: {action.shape}")
        print("\nModel inference test PASSED!")

    def _render_high_res_frame(self, env, camera_azimuth=140.0, camera_elevation=-40.0, 
                               camera_lookat=None, camera_distance=2.0):
        """Render high-resolution frame using MuJoCo Renderer API with custom camera.
        
        Args:
            env: Environment to render from
            camera_azimuth: Camera azimuth angle in degrees
            camera_elevation: Camera elevation angle in degrees
            camera_lookat: Camera lookat position [x, y, z]
            camera_distance: Camera distance from target
            
        Returns:
            RGB image array of shape (height, width, 3)
        """
        try:
            raw_env = env.unwrapped
            if not hasattr(raw_env, 'model') or not hasattr(raw_env, 'data'):
                return env.render()
            
            # Set rendering resolution in model
            raw_env.model.vis.global_.offwidth = self.video.width
            raw_env.model.vis.global_.offheight = self.video.height
            
            # Create MuJoCo renderer with specified resolution
            renderer = mujoco.Renderer(raw_env.model, height=self.video.height, width=self.video.width)
            
            # Create a free camera with custom parameters
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.azimuth = camera_azimuth
            cam.elevation = camera_elevation
            cam.distance = camera_distance
            if camera_lookat is not None:
                cam.lookat = np.array(camera_lookat, dtype=np.float64)
            
            # Update scene with our custom camera object
            renderer.update_scene(raw_env.data, camera=cam)
            
            # Render image (no flip needed)
            img = renderer.render()
            return img
            
        except Exception as e:
            print(f"Warning: High-res render failed ({e}), falling back to default render")
            try:
                return env.render()
            except:
                return np.zeros((self.video.height, self.video.width, 3), dtype=np.uint8)

    def record_videos(self, eval_agent: str = "worker", tag="result", num_eps_per_task_to_record=1,
                     camera_azimuth=140.0, camera_elevation=-40.0, camera_lookat=None, camera_distance=2.0):
        """
        Record videos of all envs, each env performs one episodes.
        
        Args:
            eval_agent (str): Agent type ("worker", "scripted", or "student")
            tag (str): Tag for filename
            num_eps_per_task_to_record (int): Number of episodes to record
            camera_azimuth (float): Camera azimuth angle in degrees (0=front, 90=left, 180=back, 270=right). Defaults to 140.0 (upper-right view)
            camera_elevation (float): Camera elevation angle in degrees. Defaults to -40.0.
            camera_lookat (array-like): Point the camera is looking at [x, y, z]. Defaults to [0, 0.5, 0.0].
            camera_distance (float): Distance from camera to the center. Defaults to 2.0.
        """
        if camera_lookat is None:
            camera_lookat = np.array([0, 0.5, 0.0])
        else:
            camera_lookat = np.array(camera_lookat)
            
        num_record = num_eps_per_task_to_record
        video_envs = self.list_envs
        
        for env_idx in self.env_indices_i:
            print(f'start recording env {self.env_indices[env_idx]} ...')
            self.reset_goal_locations_for_listenv(env_idx, video_envs[self.env_indices[env_idx]].env.name)
            self.video.reset()
            episode_step = 0           
            env_obs = []
            success = 0.0

            obs = video_envs[self.env_indices[env_idx]].reset()[0]
            env_obs.append(obs)
            multitask_obs = {"env_obs": torch.tensor(env_obs), "task_obs": torch.tensor([self.env_indices[env_idx]])}

            # Initial frame recording with high resolution
            frame = self._render_high_res_frame(video_envs[self.env_indices[env_idx]], camera_azimuth, camera_elevation, camera_lookat, camera_distance)
            self.video.record(frame=frame)

            episode_reward = 0

            for episode_step in range(self.max_episode_steps * num_record):

                # record for env_idx env with high resolution
                frame = self._render_high_res_frame(video_envs[self.env_indices[env_idx]], camera_azimuth, camera_elevation, camera_lookat, camera_distance)
                self.video.record(frame=frame)
                
                # agent select action
                if eval_agent=="worker":
                    with agent_utils.eval_mode(self.agent[env_idx]):
                        action = self.agent[env_idx].select_action(
                            multitask_obs=multitask_obs, modes=["eval"]
                        )
                elif eval_agent=="scripted":
                    action = np.clip(self.scripted_policies[env_idx].get_action(self.scripted_policies[env_idx], multitask_obs["env_obs"][0].cpu().numpy()), -1, 1)[None]
                else:
                    with agent_utils.eval_mode(self.student):
                        action = self.student.select_action(
                            multitask_obs=multitask_obs, modes=["eval"]
                        )
                
                # interactive with envs get new obs
                env_obs = []
                obs, reward, _, _, info = video_envs[self.env_indices[env_idx]].step(action[0])

                episode_reward += reward
                
                if (episode_step+1) % self.max_episode_steps == 0:
                    obs = video_envs[self.env_indices[env_idx]].reset()[0]
                    frame = self._render_high_res_frame(self.list_envs[self.env_indices[env_idx]], camera_azimuth, camera_elevation, camera_lookat, camera_distance)
                    self.video.record(frame=frame)
                env_obs.append(obs)
                success += info['success']

                multitask_obs = {"env_obs": torch.tensor(env_obs), "task_obs": torch.tensor([self.env_indices[env_idx]])}
                episode_step += 1

            success = float(success > 0)
            # Get task name from environment
            task_name = video_envs[self.env_indices[env_idx]].env.name
            # Include robot_type in video filename for easier identification
            robot_str = self._get_robot_prefix()
            self.video.save(file_name=f'{robot_str}{eval_agent}_env{self.env_indices[env_idx]}_{task_name}_success_{success}_reward_{int(episode_reward)}_{tag}')
            #video_envs[self.env_indices[env_idx]].close()

    def record_videos_for_transformer(self, agent, tag="result", seq_len=16, num_eps_per_task_to_record=1,
                                     camera_azimuth=140.0, camera_elevation=-40.0, camera_lookat=None, camera_distance=2.0):
        """
        Record videos of all envs, each env performs one episodes.
        
        Args:
            agent: The transformer agent to evaluate
            tag (str): Tag for filename
            seq_len (int): Sequence length for the transformer
            num_eps_per_task_to_record (int): Number of episodes to record
            camera_azimuth (float): Camera azimuth angle in degrees (0=front, 90=left, 180=back, 270=right). Defaults to 140.0 (upper-right view)
            camera_elevation (float): Camera elevation angle in degrees. Defaults to -40.0.
            camera_lookat (array-like): Point the camera is looking at [x, y, z]. Defaults to [0, 0.5, 0.0].
            camera_distance (float): Distance from camera to the center. Defaults to 2.0.
        """
        if camera_lookat is None:
            camera_lookat = np.array([0, 0.5, 0.0])
        else:
            camera_lookat = np.array(camera_lookat)
            
        num_record = num_eps_per_task_to_record
        video_envs = self.list_envs
        
        for env_idx in self.env_indices_i:
            print(f'start recording env {self.env_indices[env_idx]} ...')
            self.reset_goal_locations_for_listenv(env_idx, video_envs[self.env_indices[env_idx]].env.name)
            self.video.reset()
            episode_step = 0           
            env_obs = []
            success = 0.0

            obs = video_envs[self.env_indices[env_idx]].reset()[0]
            env_obs.append(obs)
            multitask_obs = {"env_obs": torch.tensor(env_obs), "task_obs": torch.tensor([self.env_indices[env_idx]])}
            states = torch.cat((multitask_obs['env_obs'][:,:18], multitask_obs['env_obs'][:,36:]), dim=1).float()[:, None]
            actions = torch.empty(1, 0, 4)
            rewards = torch.empty(1, 0, 1)

            # Initial frame recording with high resolution
            frame = self._render_high_res_frame(video_envs[self.env_indices[env_idx]], camera_azimuth, camera_elevation, camera_lookat, camera_distance)
            self.video.record(frame=frame)

            episode_reward = 0

            for episode_step in range(self.max_episode_steps * num_record):

                # record for env_idx env with high resolution
                frame = self._render_high_res_frame(video_envs[self.env_indices[env_idx]], camera_azimuth, camera_elevation, camera_lookat, camera_distance)
                self.video.record(frame=frame)
                
                # agent select action
                with agent_utils.eval_mode(agent):
                    action = agent.select_action(
                        states=states,
                        actions=actions,
                        rewards=rewards,
                        task_ids=torch.tensor(self.task_num[self.env_indices_i]),
                    )
                
                # interactive with envs get new obs
                env_obs = []
                obs, reward, _, _, info = video_envs[self.env_indices[env_idx]].step(action[0])

                new_state = torch.cat((multitask_obs['env_obs'][:,:18], multitask_obs['env_obs'][:,36:]), dim=1).float()[:, None]
                new_action = torch.tensor(action).float()[:, None]
                new_reward = torch.tensor([reward]).float()[:,None,None]
                
                states = torch.cat((states[:, -seq_len+1:], new_state), dim=1)
                actions = torch.cat((actions[:, -seq_len+2:], new_action), dim=1)
                rewards = torch.cat((rewards[:, -seq_len+2:], new_reward), dim=1)

                episode_reward += reward
                
                if (episode_step+1) % self.max_episode_steps == 0:
                    obs = video_envs[self.env_indices[env_idx]].reset()[0]
                    frame = self._render_high_res_frame(self.list_envs[self.env_indices[env_idx]], camera_azimuth, camera_elevation, camera_lookat, camera_distance)
                    self.video.record(frame=frame)
                env_obs.append(obs)
                success += info['success']

                multitask_obs = {"env_obs": torch.tensor(env_obs), "task_obs": torch.tensor([self.env_indices[env_idx]])}
                episode_step += 1

            success = float(success > 0)
            # Get task name from environment
            task_name = video_envs[self.env_indices[env_idx]].env.name
            # Include robot_type in video filename for easier identification
            robot_str = self._get_robot_prefix()
            self.video.save(file_name=f'{robot_str}transformer_env{self.env_indices[env_idx]}_{task_name}_success_{success}_reward_{int(episode_reward)}_{tag}')
