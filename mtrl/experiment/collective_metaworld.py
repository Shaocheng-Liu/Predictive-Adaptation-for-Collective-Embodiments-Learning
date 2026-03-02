# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Class to interface with an Experiment"""

from typing import Dict, Tuple

import hydra
import numpy as np
import torch
import random

from mtrl.agent import utils as agent_utils
from mtrl.env import builder as env_builder
from mtrl.env.vec_env import VecEnv  # type: ignore[attr-defined]
from mtrl.experiment import collective_learning
from mtrl.utils.types import ConfigType, EnvMetaDataType, EnvsDictType

from collections import defaultdict

class Experiment(collective_learning.Experiment):
    """Experiment Class"""

    def __init__(self, config: ConfigType, experiment_id: str = "0"):
        super().__init__(config, experiment_id)
        self.should_reset_env_manually = True

    def create_eval_modes_to_env_ids(self):
        eval_modes_to_env_ids = {}
        eval_modes = [
            key for key in self.config.metrics.keys() if not key.startswith("train")
        ]
        for mode in eval_modes:
            if self.config.env.benchmark._target_ in [
                "metaworld.ML1",
                "metaworld.MT1",
                "metaworld.MT10",
                "metaworld.MT50",
                "metaworld.COL_MT5"
            ]:
                eval_modes_to_env_ids[mode] = list(range(self.config.env.num_envs))
            else:
                raise ValueError(
                    f"`{self.config.env.benchmark._target_}` env is not supported by metaworld experiment."
                )
        return eval_modes_to_env_ids

    def build_envs(self):
        benchmark = hydra.utils.instantiate(self.config.env.benchmark)
        
        self.env_name_task_dict = defaultdict(list)
        train_tasks = benchmark._train_tasks

        for task in train_tasks:
            self.env_name_task_dict[task.env_name].append(task)

        if "COL" in self.config.env.benchmark._target_:
            build_vec_func = env_builder.build_metaworld_vec_env_col
            build_vec_func_list = env_builder.build_metaworld_env_list_col
        else:
            build_vec_func = env_builder.build_metaworld_vec_env
            build_vec_func_list = env_builder.build_metaworld_env_list

        envs = {}
        mode = "train"
        envs[mode], env_id_to_task_map = build_vec_func(
            config=self.config, benchmark=benchmark, mode=mode, env_id_to_task_map=None
        )
        self.env_id_to_task_map = env_id_to_task_map 
        mode = "eval"
        envs[mode], env_id_to_task_map = build_vec_func(
            config=self.config,
            benchmark=benchmark,
            mode="train",
            env_id_to_task_map=env_id_to_task_map,
        )
        # In MT10 and MT50, the tasks are always sampled in the train mode.
        # For more details, refer https://github.com/rlworkgroup/metaworld

        max_episode_steps = 400
        # hardcoding the steps as different environments return different
        # values for max_path_length. MetaWorld uses 150 as the max length.
        metadata = self.get_env_metadata(
            env=envs["train"],
            max_episode_steps=max_episode_steps,
            ordered_task_list=list(env_id_to_task_map.keys()),
        )

        ### andi
        self.list_envs, self.env_id_to_task_map_recording = build_vec_func_list(
            config=self.config,
            benchmark=benchmark,
            mode="train",
            env_id_to_task_map=self.env_id_to_task_map ,
        )
        ### andi

        return envs, metadata
    
    def create_env_id_to_index_map(self) -> Dict[str, int]:
        env_id_to_index_map: Dict[str, int] = {}
        current_id = 0
        for env in self.envs.values():
            assert isinstance(env, VecEnv)
            for env_name in env.ids:
                if env_name not in env_id_to_index_map:
                    env_id_to_index_map[env_name] = current_id
                    current_id += 1
        return env_id_to_index_map

    def evaluate_vec_env_of_tasks(self, vec_env: VecEnv, step: int, episode: int, agent: str = "worker"):
        """Evaluate the agent's performance on the different environments,
        vectorized as a single instance of vectorized environment.

        Since we are evaluating on multiple tasks, we track additional metadata
        to track which metric corresponds to which task.

        Args:
            vec_env (VecEnv): vectorized environment.
            step (int): step for tracking the training of the agent.
            episode (int): episode for tracking the training of the agent.
        """
        import time  # 确保导入 time

        episode_step = 0
        self.logger.log(f"eval/episode", episode, step)

        episode_reward, mask, done, success = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 1.0, False, 0.0]
        ]  # (num_envs, 1)
        
        multitask_obs = vec_env.reset()  # (num_envs, 9, 84, 84)

        while episode_step < self.max_episode_steps:
            # --- 计时起点 1: 策略/推理阶段 (CPU Prep + GPU Inference) ---
            t0 = time.time()

            action = np.full(shape=(vec_env.num_envs, np.prod(self.action_space.shape)), fill_value=0.)
            if agent=="worker":
                for i in self.env_indices_i:
                    env_obs_i = {
                        key: value[self.env_indices[i]] for key, value in multitask_obs.items()
                    }
                    with agent_utils.eval_mode(self.agent[i]):
                        action[self.env_indices[i]] = self.agent[i].select_action(
                            multitask_obs=env_obs_i, modes=["eval"]
                        )
            elif agent=="scripted":
                for i in self.env_indices_i:
                    env_obs_i = {
                        key: value[self.env_indices[i]] for key, value in multitask_obs.items()
                    }
                    action[self.env_indices[i]] = np.clip(self.scripted_policies[i].get_action(self.scripted_policies[i], env_obs_i["env_obs"].cpu().numpy()), -1, 1)
            else: # student
                env_obs_i = {
                    key: value[self.env_indices] for key, value in multitask_obs.items()
                }
                with agent_utils.eval_mode(self.student):
                    action[self.env_indices] = self.student.select_action(
                        multitask_obs=env_obs_i, modes=["eval"]
                    )

            multitask_obs, reward, done, info = vec_env.step(action)

            success += info['success']
            mask = mask * (1 - done.astype(int))
            episode_reward += reward * mask
            episode_step += 1
        
        success = (success > 0).astype("float")

        success = success[self.env_indices]
        # if success==0:
            # self.record_videos(eval_agent="scripted", tag=f"sample_{i}")
        episode_reward = episode_reward[self.env_indices]
        
        self.logger.log(
            f"eval/episode_reward",
            episode_reward.mean(),
            step,
        )
        self.logger.log(
            f"eval/success",
            success.mean(),
            step,
        )

        for _current_env_index, _current_env_id in enumerate(self.env_indices):
            self.logger.log(
                f"eval/episode_reward_env_index_{_current_env_id}",
                episode_reward[_current_env_index].mean(),
                step,
            )
            self.logger.log(
                f"eval/success_env_index_{_current_env_id}",
                success[_current_env_index].mean(),
                step,
            )

        self.logger.dump(step)
        return success.sum()

    def evaluate_transformer(self, agent, vec_env: VecEnv, step: int, episode: int, seq_len: int, sample_actions: bool, reward_unscaled: bool):
        """Evaluate the agent's performance on the different environments,
        vectorized as a single instance of vectorized environment.
        """
        
        # 获取设备 (假设 agent 已经在正确的设备上)
        device = agent.device if hasattr(agent, "device") else torch.device("cpu")

        episode_step = 0
        self.logger.log(f"col_eval/episode", episode, step)

        # 初始化统计变量 (保持原版 Numpy 格式)
        episode_reward, mask, done, success = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 1.0, False, 0.0]
        ] 
        
        # === 1. 初始化 & Buffer 预分配 (逻辑替换原版 left_pad_batch) ===
        multitask_obs = vec_env.reset()
        
        # 处理初始 State (兼容 Tensor 和 Numpy 输入)
        obs_data = multitask_obs['env_obs']
        if isinstance(obs_data, torch.Tensor):
            obs_tensor = obs_data
        else:
            obs_tensor = torch.from_numpy(obs_data)
        
        # 提取 p1, p2 并拼接 (对应原版: multitask_obs['env_obs'][:,:18] + [:,36:])
        first_state = torch.cat((obs_tensor[:, :18], obs_tensor[:, 36:]), dim=1).float().to(device)
        
        # 自动获取维度
        B = vec_env.num_envs
        obs_dim = first_state.shape[-1]
        action_dim = np.prod(self.action_space.shape)

        # [State Buffer]: [Num_Envs, T, Obs_Dim]
        # 逻辑：直接用第一帧填满整个序列，等效于 left_pad_batch 在 t=0 时的效果
        state_buffer = first_state.unsqueeze(1).repeat(1, seq_len, 1)

        # [Action/Reward Buffer]: 填 0
        action_buffer = torch.zeros((B, seq_len, action_dim), device=device)
        reward_buffer = torch.zeros((B, seq_len, 1), device=device)

        # 缓存 Task IDs
        task_ids_tensor = torch.tensor(self.task_num[self.env_indices_i], device=device)

        while episode_step < self.max_episode_steps:
            # 初始化当前步的 Action Numpy 数组 (保持原版逻辑)
            action_np = np.full(shape=(B, action_dim), fill_value=0.)
            
            # === 2. 推理 (使用 Buffer 切片) ===
            with agent_utils.eval_mode(agent):
                # 切片：State 取全长，Action/Reward 取后 T-1 个 (对应原版 T-1)
                active_states = state_buffer[self.env_indices]
                active_actions = action_buffer[self.env_indices, 1:, :]
                active_rewards = reward_buffer[self.env_indices, 1:, :]

                if sample_actions:
                    action_out = agent.sample_action(
                        states=active_states,
                        actions=active_actions, 
                        rewards=active_rewards,
                        task_ids=task_ids_tensor
                    )
                else:
                    action_out = agent.select_action(
                        states=active_states,
                        actions=active_actions,
                        rewards=active_rewards,
                        task_ids=task_ids_tensor
                    )
            
            # === 3. 数据类型转换 (Tensor -> Numpy) ===
            # 确保传给 env.step 的是 numpy，且处理部分 env active 的情况
            if isinstance(action_out, torch.Tensor):
                action_subset_np = action_out.detach().cpu().numpy()
            else:
                action_subset_np = action_out
            
            action_np[self.env_indices] = action_subset_np

            # === 4. 环境交互 ===
            print("obs is:", multitask_obs)
            multitask_obs, reward, done, info = vec_env.step(action_np)
            if reward_unscaled:
                reward = info['unscaled_reward']

            # === 5. 滚动更新 Buffer (替换原版 torch.cat) ===
            # A. 整体左移 (丢弃最旧的一帧)
            state_buffer = torch.roll(state_buffer, shifts=-1, dims=1)
            action_buffer = torch.roll(action_buffer, shifts=-1, dims=1)
            reward_buffer = torch.roll(reward_buffer, shifts=-1, dims=1)

            # B. 准备新数据
            # 处理 New State
            obs_data_next = multitask_obs['env_obs']
            if isinstance(obs_data_next, torch.Tensor):
                obs_next_tensor = obs_data_next
            else:
                obs_next_tensor = torch.from_numpy(obs_data_next)
            
            new_state = torch.cat((obs_next_tensor[:, :18], obs_next_tensor[:, 36:]), dim=1).float().to(device)
            
            # 处理 New Reward
            if isinstance(reward, torch.Tensor):
                reward_tensor = reward
            else:
                reward_tensor = torch.from_numpy(reward)
            new_reward = reward_tensor.float().to(device).unsqueeze(-1) # [B, 1]

            # 处理 Action (需要存入 Buffer，所以转回 Tensor)
            new_action = torch.from_numpy(action_np).float().to(device) # [B, A]

            # C. 填入 Buffer 末尾
            state_buffer[:, -1, :] = new_state
            action_buffer[:, -1, :] = new_action
            reward_buffer[:, -1, :] = new_reward

            # === 6. 统计 (保持原版逻辑) ===
            success += info['success']
            mask = mask * (1 - done.astype(int))
            episode_reward += reward * mask
            episode_step += 1
        
        success = (success > 0).astype("float")
        success = success[self.env_indices]
        episode_reward = episode_reward[self.env_indices]
        
        self.logger.log(f"col_eval/episode_reward", episode_reward.mean(), step)
        self.logger.log(f"col_eval/success", success.mean(), step)

        for _current_env_index, _current_env_id in enumerate(self.env_indices):
            self.logger.log(f"col_eval/episode_reward_env_index_{_current_env_id}", episode_reward[_current_env_index].mean(), step)
            self.logger.log(f"col_eval/success_env_index_{_current_env_id}", success[_current_env_index].mean(), step)

        self.logger.dump(step)
        # 保持和源代码一致的返回值
        return success

    def evaluate_decision_tf(self, vec_env: VecEnv, step: int, episode: int, seq_len: int, reward_unscaled: bool):
        """Evaluate the agent's performance on the different environments,
        vectorized as a single instance of vectorized environment.

        Since we are evaluating on multiple tasks, we track additional metadata
        to track which metric corresponds to which task.

        Args:
            vec_env (VecEnv): vectorized environment.
            step (int): step for tracking the training of the agent.
            episode (int): episode for tracking the training of the agent.
        """
        episode_step = 0
        #prefix = "" if agent=="worker" or agent=="student" else "col_"
        self.logger.log(f"col_eval/episode", episode, step)

        episode_reward, mask, done, success = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 1.0, False, 0.0]
        ]  # (num_envs, 1)
        multitask_obs = vec_env.reset()  # (num_envs, 9, 84, 84)

        state_src = multitask_obs['env_obs'][:,None].float()
        action_src = torch.zeros((vec_env.num_envs, 1, self.action_shape))
        rewards_src = torch.zeros((vec_env.num_envs, 1, 1))

        #TODO: improve evaluate_unused
        action = np.full(shape=(vec_env.num_envs, np.prod(self.action_space.shape)), fill_value=0.)
        while episode_step < self.max_episode_steps:

            with agent_utils.eval_mode(self.col_agent):
                action[self.env_indices] = self.col_agent.select_action(
                    state_src, action_src, rewards_src
                )
            
            multitask_obs, reward, done, info = vec_env.step(action)
            if reward_unscaled:
                reward = info['unscaled_reward']

            state_src = torch.cat((state_src[:, -seq_len+1:], multitask_obs['env_obs'][:,None].float()), dim=1)
            action_src = torch.cat((action_src[:, -seq_len+1:], torch.from_numpy(action)[:,None].float()), dim=1)
            rewards_src = torch.cat((rewards_src[:, -seq_len+1:], torch.tensor(reward)[:,None,None].float()), dim=1)

            success += info['success']
            mask = mask * (1 - done.astype(int))
            episode_reward += reward * mask
            episode_step += 1
        success = (success > 0).astype("float")

        success = success[self.env_indices]
        episode_reward = episode_reward[self.env_indices]
        
        self.logger.log(
            f"col_eval/episode_reward",
            episode_reward.mean(),
            step,
        )
        self.logger.log(
            f"col_eval/success",
            success.mean(),
            step,
        )

        for _current_env_index, _current_env_id in enumerate(self.env_indices):
            self.logger.log(
                f"col_eval/episode_reward_env_index_{_current_env_id}",
                episode_reward[_current_env_index].mean(),
                step,
            )
            self.logger.log(
                f"col_eval/success_env_index_{_current_env_id}",
                success[_current_env_index].mean(),
                step,
            )

        self.logger.dump(step)

    def collect_trajectory(self, vec_env: VecEnv, num_steps: int) -> None:
        multitask_obs = vec_env.reset()  # (num_envs, 9, 84, 84)
        env_indices = multitask_obs["task_obs"]
        episode_reward, episode_step, done = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 0, True]
        ]  # (num_envs, 1)

        for _ in range(num_steps):
            with agent_utils.eval_mode(self.agent):
                action = self.agent.sample_action(
                    multitask_obs=multitask_obs, mode="train"
                )  # (num_envs, action_dim)
            next_multitask_obs, reward, done, info = vec_env.step(action)
            if self.should_reset_env_manually:
                if (episode_step[0] + 1) % self.max_episode_steps == 0:
                    # we do a +2 because we started the counting from 0 and episode_step is incremented after updating the buffer
                    next_multitask_obs = vec_env.reset()
            episode_reward += reward

            # allow infinite bootstrap
            for index, env_index in enumerate(env_indices):
                done_bool = (
                    0
                    if episode_step[index] + 1 == self.max_episode_steps
                    else float(done[index])
                )
                self.replay_buffer.add(
                    multitask_obs["env_obs"][index],
                    action[index],
                    reward[index],
                    next_multitask_obs["env_obs"][index],
                    done_bool,
                    env_index=env_index,
                )

            multitask_obs = next_multitask_obs
            episode_step += 1

    def build_video_envs(self):
        
        benchmark = hydra.utils.instantiate(self.config.env.benchmark)
        #benchmark = hydra.utils.instantiate(self.config.env.kuka_benchmark)

        list_envs, env_id_to_task_map_recording = env_builder.build_metaworld_env_list(
            config=self.config,
            benchmark=benchmark,
            mode="train",
            env_id_to_task_map=self.env_id_to_task_map ,
        )
        self.list_envs = list_envs
        self.env_id_to_task_map_recording = env_id_to_task_map_recording
        
        return list_envs
    