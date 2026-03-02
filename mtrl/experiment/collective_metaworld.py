# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Class to interface with an Experiment"""

from typing import Dict, Tuple

import hydra
import numpy as np
import torch
import random
import time

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

    def _build_minimal_metadata(self) -> EnvMetaDataType:
        """Override to create minimal metadata with metaworld-specific attributes."""
        # Call parent method to get base metadata
        metadata = super()._build_minimal_metadata()
        
        # Create benchmark to get task information (needed for env_name_task_dict)
        import hydra
        from collections import defaultdict
        
        benchmark = hydra.utils.instantiate(self.config.env.benchmark)
        
        # Initialize env_name_task_dict for modes that don't create envs
        self.env_name_task_dict = defaultdict(list)
        train_tasks = benchmark._train_tasks
        
        for task in train_tasks:
            self.env_name_task_dict[task.env_name].append(task)
        
        # Initialize env_id_to_task_map (empty for modes without envs)
        self.env_id_to_task_map = {}
        
        # Initialize list_envs (empty for modes without envs)
        self.list_envs = []
        
        return metadata
    
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
        episode_step = 0
        self.logger.log(f"eval/episode", episode, step)

        episode_reward, mask, done, success = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 1.0, False, 0.0]
        ]  # (num_envs, 1)
        
        multitask_obs = vec_env.reset()  # (num_envs, 9, 84, 84)

        while episode_step < self.max_episode_steps:

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
        episode_step = 0
        self.logger.log(f"col_eval/episode", episode, step)

        episode_reward, mask, done, success = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 1.0, False, 0.0]
        ] 
        
        # === 1. 初始化 & GPU 预分配 (Pre-fill) ===
        multitask_obs = vec_env.reset()
        
        # [Fix 1] 处理初始 State (兼容 Tensor 和 Numpy)
        # 既然你说 multitask_obs['env_obs'] 已经是 Tensor，这里直接切片即可
        obs_data = multitask_obs['env_obs']
        if isinstance(obs_data, torch.Tensor):
            p1 = obs_data[:, :18]
            p2 = obs_data[:, 36:]
        else:
            p1 = torch.from_numpy(obs_data[:, :18])
            p2 = torch.from_numpy(obs_data[:, 36:])
            
        first_state = torch.cat((p1, p2), dim=1).float().to(self.device)
        
        # 自动获取维度
        obs_dim = first_state.shape[-1]

        # [State Buffer]: [Num_Envs, T, Obs_Dim]
        ctx_states = torch.empty((vec_env.num_envs, seq_len, obs_dim), device=self.device)
        ctx_states[:, :, :] = first_state.unsqueeze(1) # Pre-fill

        # [Action/Reward Buffer]
        ctx_actions = torch.zeros((vec_env.num_envs, seq_len, 4), device=self.device)
        ctx_rewards = torch.zeros((vec_env.num_envs, seq_len, 1), device=self.device)

        task_ids_tensor = torch.tensor(self.task_num[self.env_indices_i], device=self.device)
        action_np = np.full(shape=(vec_env.num_envs, np.prod(self.action_space.shape)), fill_value=0.)

        total_policy_time = 0
        total_env_time = 0
        step_count = 0
        while episode_step < self.max_episode_steps:
            t0 = time.time()
            # === 2. 推理 ===
            with agent_utils.eval_mode(agent):
                # 保持切片逻辑：State全量，Action/Reward少一帧
                active_states = ctx_states[self.env_indices]
                active_actions = ctx_actions[self.env_indices, 1:, :]
                active_rewards = ctx_rewards[self.env_indices, 1:, :]

                if sample_actions:
                    action_tensor = agent.sample_action(
                        states=active_states,
                        actions=active_actions, 
                        rewards=active_rewards,
                        task_ids=task_ids_tensor,
                        return_tensor=True
                    )
                else:
                    action_tensor = agent.select_action(
                        states=active_states,
                        actions=active_actions,
                        rewards=active_rewards,
                        task_ids=task_ids_tensor,
                        return_tensor=True
                    )
            
            # [Fix 2] 处理 Action 返回值 (兼容 Tensor 和 Numpy)
            # 解决 AttributeError: 'numpy.ndarray' object has no attribute 'detach'
            action_np_subset = action_tensor.detach().cpu().numpy()
            action_np[self.env_indices] = action_np_subset
            t1 = time.time()
            # === 3. 环境交互 ===
            multitask_obs, reward, done, info = vec_env.step(action_np)
            t2 = time.time()
            if reward_unscaled:
                reward = info['unscaled_reward']

            # === 4. 滚动更新 ===
            # A. 整体左移
            ctx_states = torch.roll(ctx_states, shifts=-1, dims=1)
            ctx_actions = torch.roll(ctx_actions, shifts=-1, dims=1)
            ctx_rewards = torch.roll(ctx_rewards, shifts=-1, dims=1)

            # B. 准备新数据 (兼容 Tensor 和 Numpy)
            
            # 处理 New State
            obs_data_next = multitask_obs['env_obs']
            if isinstance(obs_data_next, torch.Tensor):
                p1_next = obs_data_next[:, :18]
                p2_next = obs_data_next[:, 36:]
            else:
                p1_next = torch.from_numpy(obs_data_next[:, :18])
                p2_next = torch.from_numpy(obs_data_next[:, 36:])
            
            new_state = torch.cat((p1_next, p2_next), dim=1).float().to(self.device)
            
            # 处理 New Reward (兼容 Tensor 和 Numpy)
            if isinstance(reward, torch.Tensor):
                reward_tensor = reward
            else:
                reward_tensor = torch.from_numpy(reward)
            
            new_reward = reward_tensor.float().to(self.device).unsqueeze(-1)

            # C. 填入 Buffer
            ctx_states[:, -1, :] = new_state
            ctx_actions[self.env_indices, -1, :] = action_tensor
            ctx_rewards[:, -1, :] = new_reward

            # === 5. 统计 ===
            success += info['success']
            mask = mask * (1 - done.astype(int))
            episode_reward += reward * mask
            episode_step += 1
            total_policy_time += (t1 - t0)
            total_env_time += (t2 - t1)
            step_count += 1
        
        success = (success > 0).astype("float")
        success = success[self.env_indices]
        episode_reward = episode_reward[self.env_indices]
        
        self.logger.log(f"col_eval/episode_reward", episode_reward.mean(), step)
        self.logger.log(f"col_eval/success", success.mean(), step)
        avg_policy = (total_policy_time / step_count) * 1000
        avg_env = (total_env_time / step_count) * 1000
        print(f"\n[Perf Diagnosis] Steps: {step_count}")
        print(f"  🤖 Policy Time (GPU): {avg_policy:.2f} ms/step")
        print(f"  🌍 Env Time (CPU):    {avg_env:.2f} ms/step")
        
        if avg_env > avg_policy:
            print("  👉 瓶颈在 CPU/环境模拟！(服务器主频太低)")
        else:
            print("  👉 瓶颈在 GPU/代码逻辑！(数据传输或模型太大)")

        for _current_env_index, _current_env_id in enumerate(self.env_indices):
            self.logger.log(f"col_eval/episode_reward_env_index_{_current_env_id}", episode_reward[_current_env_index].mean(), step)
            self.logger.log(f"col_eval/success_env_index_{_current_env_id}", success[_current_env_index].mean(), step)

        self.logger.dump(step)
        return success.sum()

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
    