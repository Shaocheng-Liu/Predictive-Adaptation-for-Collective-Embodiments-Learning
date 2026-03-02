# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""`Experiment` class manages the lifecycle of a multi-task model."""
import random

import time
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch

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
        if self.config.experiment.mode == 'train_worker':
            self.run_training_worker()
        elif self.config.experiment.mode == 'record':
            self.run_record()
        elif self.config.experiment.mode == 'distill_collective_transformer':
            self.distill_collective_transformer()
        elif self.config.experiment.mode == 'evaluate_collective_transformer':
            self.evaluate_collective_transformer()
        elif self.config.experiment.mode == 'online_distill_collective_transformer':
            self.run_online_distillation()
        elif self.config.experiment.mode == 'train_student':
            self.run_distilling_student_online()
        elif self.config.experiment.mode == 'train_student_finetuning':
            self.run_train_student_finetuning()
        elif self.config.experiment.mode == 'distill_policy':
            self.distill_policy()
        elif self.config.experiment.mode == 'train_world_model':
            self.run_train_world_model()
        elif self.config.experiment.mode == 'evaluate_world_model':
            self.evaluate_world_model()
        elif self.config.experiment.mode == 'generate_distill_data':
            self.generate_noise_injected_data()
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


    def run_training_worker(self):
        """Run the experiment."""
        exp_config = self.config.experiment

        vec_env = self.envs["train"]

        episode_reward, episode_step, done = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 0, True]
        ]  # (num_envs, 1)

        if "success" in self.metrics_to_track:
            success = np.full(shape=vec_env.num_envs, fill_value=0.0)

        info = {}

        assert exp_config.col_sampling_freq >= exp_config.col_training_samples
        assert self.start_step >= 0
        assert exp_config.col_sampling_freq % self.max_episode_steps == 0
        episode = self.start_step // self.max_episode_steps

        start_time = time.time()

        # -------------------------------------------------------------------------
        # [Warmup Debug] 只有在这里才能看出脚本到底是死是活
        # -------------------------------------------------------------------------
        print("[DEBUG] Warmup: Resetting...", flush=True)
        col_obs = vec_env.reset()
        
        num_warmup_episodes = exp_config.num_warmup_episodes
        
        if num_warmup_episodes > 0 and self.start_step == 0:
            print(f"--- [Warmup] STARTING ({num_warmup_episodes} eps) ---", flush=True)
            print("!!! DEBUG MODE: NO NOISE, NO TRY-EXCEPT !!!", flush=True)
            
            warmup_ep_counts = np.zeros(vec_env.num_envs, dtype=int)
            warmup_episode_step = np.zeros(vec_env.num_envs, dtype=int) 
            warmup_running_rewards = np.zeros(vec_env.num_envs, dtype=float)

            while np.min(warmup_ep_counts) < num_warmup_episodes:
                action = np.zeros((vec_env.num_envs, self.action_space.shape[0]), dtype=np.float32)
                
                for i in self.env_indices_i:
                    env_idx = self.env_indices[i]
                    if warmup_ep_counts[env_idx] < num_warmup_episodes:
                        env_obs_i = {key: value[env_idx] for key, value in col_obs.items()}
                        curr_obs_np = env_obs_i["env_obs"].cpu().numpy()
                        
                        script_act = self.scripted_policies[i].get_action(self.scripted_policies[i], curr_obs_np)
                        
                        # script_act += np.random.normal(0, 0.05, size=script_act.shape) 
                        
                        action[env_idx] = np.clip(script_act, -1, 1)

                next_col_obs, reward, done, info = vec_env.step(action)
                
                final_reward = reward
                if exp_config.use_unscaled:
                    if isinstance(info, dict):
                        final_reward = info.get('unscaled_reward', reward)
                    elif isinstance(info, list) and len(info) > 0 and isinstance(info[0], dict):
                         safe_rewards = []
                         for idx, inf in enumerate(info):
                             safe_rewards.append(inf.get('unscaled_reward', reward[idx]))
                         final_reward = np.array(safe_rewards)
                
                warmup_running_rewards += final_reward

                # 5. 存入 Buffer
                for index in self.env_indices_i:
                    env_idx = self.env_indices[index]
                    
                    if warmup_ep_counts[env_idx] < num_warmup_episodes:
                        is_success = 0.0
                        if isinstance(info, dict):
                            is_success = info.get('success', [0])[env_idx]
                        elif isinstance(info, list):
                            is_success = info[env_idx].get('success', 0.0)

                        is_time_limit = (warmup_episode_step[env_idx] + 1) == self.max_episode_steps
                        done_bool = 0.0 if is_time_limit else float(done[env_idx])
                        
                        self.replay_buffer[index].add(
                            col_obs["env_obs"][env_idx],
                            action[env_idx],
                            final_reward[env_idx],
                            next_col_obs["env_obs"][env_idx],
                            done_bool,
                            task_obs=self.env_indices[index],
                        )
                        
                        if is_time_limit or done[env_idx]:
                            warmup_ep_counts[env_idx] += 1
                            
                            print(f"[Warmup] Env {env_idx} Done. Reward: {warmup_running_rewards[env_idx]:.2f} | Success: {is_success} | Steps: {warmup_episode_step[env_idx]+1}", flush=True)
                            self.reset_goal_locations()
                            next_col_obs =vec_env.reset()
                            warmup_episode_step[env_idx] = 0
                            warmup_running_rewards[env_idx] = 0.0
                        else:
                            warmup_episode_step[env_idx] += 1

                col_obs = next_col_obs
                
                if np.min(warmup_ep_counts) >= num_warmup_episodes:
                    break
            
            print("--- [Warmup] FINISHED. Saving... ---", flush=True)
            self.save_all_buffers_and_models(0) 
            
            # 重置环境
            col_obs = vec_env.reset()
            episode_reward.fill(0.0)
            episode_step.fill(0)
            if "success" in self.metrics_to_track:
                success.fill(0.0)

        action = np.asarray([self.action_space.sample() for _ in range(vec_env.num_envs)])

        train_mode = ["train" for _ in range(vec_env.num_envs)]
        
        step = exp_config.num_train_step
        for step in range(self.start_step, exp_config.num_train_steps):
            if step < exp_config.init_steps:
                action = np.asarray(
                    [self.action_space.sample() for _ in range(vec_env.num_envs)]
                )  # (num_envs, action_dim)
            elif exp_config.follow_scripted:
                for i in self.env_indices_i:
                    env_obs_i = {
                        key: value[self.env_indices[i]] for key, value in col_obs.items()
                    }
                    action[self.env_indices[i]] = np.clip(self.scripted_policies[i].get_action(self.scripted_policies[i], env_obs_i["env_obs"].cpu().numpy()), -1, 1)
            else:
                for i in self.env_indices_i:
                    with agent_utils.eval_mode(self.agent[i]):
                        env_obs_i = { #watch: correct value
                            key: value[self.env_indices[i]] for key, value in col_obs.items()
                        }
                        action[self.env_indices[i]] = self.agent[i].sample_action(
                            multitask_obs=env_obs_i,
                            modes=[
                                train_mode,
                            ],
                        )

            # run training update
            if step >= exp_config.init_steps and exp_config.train_worker:
                num_updates = (
                    exp_config.init_steps if step == exp_config.init_steps else 1
                )
                for _ in range(num_updates):
                    for i in self.env_indices_i:
                        self.agent[i].update(self.replay_buffer[i], self.logger, step)

            # copy latest transitions
            if step % exp_config.col_sampling_freq == 0 and step > exp_config.col_training_warmup:
                for i in self.env_indices_i:
                    idx = self.replay_buffer[i].idx
                    env_obs_arr = self.replay_buffer[i].env_obses[idx-exp_config.col_training_samples:idx]
                    actions_arr = self.replay_buffer[i].actions[idx-exp_config.col_training_samples:idx]
                    rewards_arr = self.replay_buffer[i].rewards[idx-exp_config.col_training_samples:idx]
                    next_env_obs_arr = self.replay_buffer[i].next_env_obses[idx-exp_config.col_training_samples:idx]
                    not_done_arr = np.logical_not(self.replay_buffer[i].not_dones[idx-exp_config.col_training_samples:idx])
                    task_obs_arr = self.replay_buffer[i].task_obs[idx-exp_config.col_training_samples:idx]
                    length = len(env_obs_arr)
                    if length < exp_config.col_training_samples:
                        env_obs_arr = np.concatenate((self.replay_buffer[i].env_obses[-exp_config.col_training_samples-length:], env_obs_arr))
                        actions_arr = np.concatenate((self.replay_buffer[i].actions[-exp_config.col_training_samples-length:], actions_arr))
                        rewards_arr = np.concatenate((self.replay_buffer[i].rewards[-exp_config.col_training_samples-length:], rewards_arr))
                        next_env_obs_arr = np.concatenate((self.replay_buffer[i].next_env_obses[-exp_config.col_training_samples-length:], next_env_obs_arr))
                        not_done_arr = np.concatenate((np.logical_not(self.replay_buffer[i].not_dones[-exp_config.col_training_samples-length:]), not_done_arr))
                        task_obs_arr = np.concatenate((self.replay_buffer[i].task_obs[-exp_config.col_training_samples-length:], task_obs_arr))
                    self.replay_buffer_distill_tmp[i].add_array(
                        env_obs_arr,
                        actions_arr,
                        rewards_arr,
                        next_env_obs_arr,
                        not_done_arr,
                        task_obs_arr,
                        exp_config.col_training_samples,
                    )
                    print(f"----- Distill tmp Idx {self.replay_buffer_distill_tmp[i].idx} -----")

            next_col_obs, reward, done, info = vec_env.step(action)
            if exp_config.use_unscaled:
                reward = info['unscaled_reward']

            episode_reward += reward
            if "success" in self.metrics_to_track:
                success += info['success']

            if self.should_reset_env_manually:
                if (episode_step[0] + 1) % self.max_episode_steps == 0:
                    # we do a +1 because we started the counting from 0 and episode_step is incremented after updating the buffer
                    if exp_config.reset_goal_state:
                        if exp_config.reset_goal_state_in_order:
                            self.reset_goal_locations_in_order()
                        else:
                            self.reset_goal_locations()
                    next_col_obs = vec_env.reset()
                    
            # allow infinite bootstrap
            for index in self.env_indices_i:
                done_bool = (
                    0
                    if episode_step[self.env_indices[index]] + 1 == self.max_episode_steps
                    else float(done[self.env_indices[index]])
                )
                self.replay_buffer[index].add(
                    col_obs["env_obs"][self.env_indices[index]],
                    action[self.env_indices[index]],
                    reward[self.env_indices[index]],
                    next_col_obs["env_obs"][self.env_indices[index]],
                    done_bool,
                    task_obs=self.env_indices[index],
                )

            if step % self.max_episode_steps == 0:
                self.logger.log("train/episode", episode, step) # moved up here
                if step > 0:
                    if "success" in self.metrics_to_track:
                        success = (success > 0).astype("float")
                        for index in self.env_indices:
                            self.logger.log(
                                f"train/success_env_index_{index}",
                                success[index],
                                step,
                            )
                        success = success[self.env_indices]
                        self.logger.log("train/success", success.mean(), step)
                    for index in self.env_indices:
                        self.logger.log(
                            f"train/episode_reward_env_index_{index}",
                            episode_reward[index],
                            step,
                        )

                    self.logger.log("train/duration", time.time() - start_time, step)
                    start_time = time.time()
                    self.logger.dump(step)

                    # EARLY STOPPING: stop training if success rate is good enough
                    self.success_history.append(success.mean())
                    if len(self.success_history) == exp_config.early_stopping_window and np.mean(self.success_history) >= exp_config.early_stopping_threshold:
                        print(f"Early stopping triggered at step {step} with avg success over {exp_config.early_stopping_window} evals: {np.mean(self.success_history):.2f}")
                        break


                # evaluate agent periodically
                if step % exp_config.eval_freq == 0:
                    self.evaluate_vec_env_of_tasks(
                        vec_env=self.envs["eval"], step=step, episode=episode, agent="worker"
                    )

                episode += 1
                episode_reward = np.full(shape=vec_env.num_envs, fill_value=0.0)
                if "success" in self.metrics_to_track:
                    success = np.full(shape=vec_env.num_envs, fill_value=0.0)
                
                #self.logger.log("train/episode", episode, step)  

                if step % exp_config.save_freq == 0:
                    self.save_all_buffers_and_models(step)
                if step % self.config.experiment.save_video_freq == 0 and self.config.experiment.save_video and step > 0:
                    print('Recording videos ...')
                    self.record_videos(eval_agent="worker", tag=f"step{step}")

            col_obs = next_col_obs
            episode_step += 1

        print(f"End eval")

        # Evaluation of the policy
        for i in range(5):
            self.evaluate_vec_env_of_tasks(
                vec_env=self.envs["eval"], step=exp_config.num_train_steps+1+i, episode=episode, agent="worker"
            )
        if step != None:
            self.save_all_buffers_and_models(step)

        print(f"Creating the DistilledReplayBuffer - {self.replay_buffer_distill_tmp[0].idx} samples collected")

        for i in range(self.num_envs):
            for idx in range(0, self.replay_buffer_distill_tmp[i].idx, exp_config.col_training_samples):
                size = np.min((exp_config.col_training_samples, self.replay_buffer_distill_tmp[i].idx-idx))
                batch = ReplayBufferSample(
                    torch.as_tensor(self.replay_buffer_distill_tmp[i].env_obses[idx:idx+size], device=self.device).float(), 
                    torch.as_tensor(self.replay_buffer_distill_tmp[i].actions[idx:idx+size], device=self.device),
                    torch.as_tensor(self.replay_buffer_distill_tmp[i].rewards[idx:idx+size], device=self.device),
                    torch.as_tensor(self.replay_buffer_distill_tmp[i].next_env_obses[idx:idx+size], device=self.device),
                    torch.as_tensor(self.replay_buffer_distill_tmp[i].not_dones[idx:idx+size], device=self.device),
                    torch.as_tensor(self.replay_buffer_distill_tmp[i].task_obs[idx:idx+size], device=self.device), 
                    None, #idxs
                )
                q_targets, teacher_mu, teacher_log_std = self.agent[i].compute_q_target_and_policy_density_from_batch(batch)
                self.replay_buffer_distill[i].add_array(
                    self.replay_buffer_distill_tmp[i].env_obses[idx:idx+size],
                    self.replay_buffer_distill_tmp[i].actions[idx:idx+size],
                    self.replay_buffer_distill_tmp[i].rewards[idx:idx+size],
                    self.replay_buffer_distill_tmp[i].next_env_obses[idx:idx+size],
                    np.logical_not(self.replay_buffer_distill_tmp[i].not_dones[idx:idx+size]),
                    np.array([[self.task_num[i]] for _ in range(size)]),
                    q_targets.cpu().numpy(),
                    teacher_mu.cpu().numpy(), 
                    teacher_log_std.cpu().numpy(),
                    size,
                )

        for i in range(self.num_envs):
            self.replay_buffer_distill[i].save(
                    self.buffer_dir_distill[i],
                    size_per_chunk=exp_config.save.buffer.size_per_chunk,
                    num_samples_to_save=exp_config.save.buffer.num_samples_to_save,
                )

        if self.config.experiment.save_video:
            print('start recording videos ...')
            self.record_videos(eval_agent="worker")
            print('video recording finished. Check folder:{}'.format(self.video.dir_name))

        for i in range(self.num_envs):
            print(f"[{i}/{self.num_envs}] Replaybuffer-distill: {self.replay_buffer_distill[i].idx} | Replaybuffer: {self.replay_buffer[i].idx} \
                  | Replaybuffer-distill-tmp: {self.replay_buffer_distill_tmp[i].idx}")

        if self.config.experiment.save.buffer.delete_replaybuffer:
            for i in range(self.num_envs):
                self.replay_buffer[i].delete_from_filesystem(self.buffer_dir[i])
                self.replay_buffer_distill_tmp[i].delete_from_filesystem(self.buffer_dir_distill_tmp[i])
                self.replay_buffer_distill[i].delete_from_filesystem(self.buffer_dir_distill[i])

        self.close_envs()

        print("finished learning")

    def run_record(self):
        """Run the experiment."""
        exp_config = self.config.experiment

        vec_env = self.envs["train"]

        episode_reward, episode_step, done = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 0, True]
        ]  # (num_envs, 1)

        if "success" in self.metrics_to_track:
            success = np.full(shape=vec_env.num_envs, fill_value=0.0)

        info = {}

        assert self.start_step >= 0
        episode = self.start_step // self.max_episode_steps

        start_time = time.time()

        col_obs = vec_env.reset()  # (num_envs, 9, 84, 84)

        mu = action = np.asarray([self.action_space.sample() for _ in range(vec_env.num_envs)])

        for step in range(self.start_step, exp_config.num_recording_steps):
            # sample action
            for i in self.env_indices_i:
                env_obs_i = {
                    key: value[self.env_indices[i]] for key, value in col_obs.items()
                }
                mu[self.env_indices[i]] = np.clip(self.scripted_policies[i].get_action(self.scripted_policies[i], env_obs_i["env_obs"].cpu().numpy()), -1, 1)
                if (np.random.rand(1) > exp_config.scripted_porb and exp_config.init_record_step < step) or \
                    step > exp_config.only_scripted_step:
                    action[self.env_indices[i]] = mu[self.env_indices[i]]
                else:
                    action[self.env_indices[i]] = self.action_space.sample()

            next_col_obs, reward, done, info = vec_env.step(action)
            if exp_config.use_unscaled:
                reward = info['unscaled_reward']

            episode_reward += reward
            if "success" in self.metrics_to_track:
                success += info['success']

            # allow infinite bootstrap
            for index in self.env_indices_i:
                done_bool = (
                    0
                    if episode_step[self.env_indices[index]] + 1 == self.max_episode_steps
                    else float(done[self.env_indices[index]])
                )
                
                self.replay_buffer_distill[index].add(
                    col_obs["env_obs"][self.env_indices[index]],
                    action[self.env_indices[index]],
                    reward[self.env_indices[index]],
                    next_col_obs["env_obs"][self.env_indices[index]],
                    done_bool,
                    task_obs=self.task_num[self.env_indices[index]],
                    #task_obs=self.env_indices[index],
                    q_value=0., 
                    mu=mu[self.env_indices[i]], 
                    log_std=np.random.rand(4),
                )

            if self.should_reset_env_manually:
                if (episode_step[0] + 1) % self.max_episode_steps == 0:
                    # we do a +1 because we started the counting from 0 and episode_step is incremented after updating the buffer
                    if exp_config.reset_goal_state:
                        self.reset_goal_locations_in_order()
                    next_col_obs = vec_env.reset()

            if step % self.max_episode_steps == 0 and step > 0:
                if step > 0:
                    if "success" in self.metrics_to_track:
                        success = (success > 0).astype("float")
                        for index in self.env_indices:
                            self.logger.log(
                                f"train/success_env_index_{index}",
                                success[index],
                                step,
                            )
                        success = success[self.env_indices]
                        self.logger.log("train/success", success.mean(), step)
                    for index in self.env_indices:
                        self.logger.log(
                            f"train/episode_reward_env_index_{index}",
                            episode_reward[index],
                            step,
                        )

                    self.logger.log("train/duration", time.time() - start_time, step)
                    start_time = time.time()
                    self.logger.dump(step)

                episode += 1
                episode_reward = np.full(shape=vec_env.num_envs, fill_value=0.0)
                if "success" in self.metrics_to_track:
                    success = np.full(shape=vec_env.num_envs, fill_value=0.0)

                self.logger.log("train/episode", episode, step)

                if step % exp_config.save_freq == 0:
                    if exp_config.save.buffer.should_save:
                        for i in range(self.num_envs):
                            self.replay_buffer_distill[i].save(
                                self.buffer_dir_distill[i],
                                size_per_chunk=exp_config.save.buffer.size_per_chunk,
                                num_samples_to_save=exp_config.save.buffer.num_samples_to_save,
                            )
                    
            col_obs = next_col_obs
            episode_step += 1

        print(f"End eval")
        print(f"Lengths of replaybuffers: {[x.idx for x in self.replay_buffer_distill]}")

        if exp_config.save.buffer.should_save:
            for i in range(self.num_envs):
                self.replay_buffer_distill[i].save(
                    self.buffer_dir_distill[i],
                    size_per_chunk=exp_config.save.buffer.size_per_chunk,
                    num_samples_to_save=exp_config.save.buffer.num_samples_to_save,
                )

        if self.config.experiment.save.buffer.delete_replaybuffer:
            for i in range(self.num_envs):
                self.replay_buffer_distill[i].delete_from_filesystem(self.buffer_dir_distill[i])

        self.close_envs()

        print("finished learning")

    def run_online_distillation(self):
        """Run training of student experiment."""
        exp_config = self.config.experiment

        vec_env = self.envs["train"]

        assert self.start_step >= 0


        action = np.asarray([self.action_space.sample() for _ in range(vec_env.num_envs)])

        print(f"Starting online training")

        episode_reward, episode_step, done = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 0, True]
        ]  # (num_envs, 1)

        if "success" in self.metrics_to_track:
            success = np.full(shape=vec_env.num_envs, fill_value=0.0)

        info = {}
        episode = self.start_step // self.max_episode_steps

        start_time = time.time()

        col_obs = vec_env.reset()
        states = torch.cat((col_obs['env_obs'][:,:18], col_obs['env_obs'][:,36:]), dim=1).float()[:, None]
        actions = torch.empty(vec_env.num_envs, 0, 4)
        rewards = torch.empty(vec_env.num_envs, 0, 1)

        action = np.asarray([self.action_space.sample() for _ in range(vec_env.num_envs)])
        eval_mode = ["eval" for _ in range(vec_env.num_envs)]

        for step in range(self.start_step, exp_config.num_online_train_step):
            if step % self.max_episode_steps == 0:
                if step > 0:
                    if "success" in self.metrics_to_track:
                        success = (success > 0).astype("float")
                        for index in self.env_indices:
                            self.logger.log(
                                f"train/success_env_index_{index}",
                                success[index],
                                step,
                            )
                        success = success[self.env_indices]
                        self.logger.log("train/success", success.mean(), step)
                    for index in self.env_indices:
                        self.logger.log(
                            f"train/episode_reward_env_index_{index}",
                            episode_reward[index],
                            step,
                        )

                    self.logger.log("train/duration", time.time() - start_time, step)
                    start_time = time.time()
                    self.logger.dump(step)

                # evaluate agent periodically
                if step % exp_config.eval_freq == 0:
                    self.evaluate_transformer(
                        agent=self.col_agent, vec_env=self.envs["eval"], step=step, episode=episode, seq_len=self.seq_len, sample_actions=False, reward_unscaled=exp_config.use_unscaled
                    )

                episode += 1
                episode_reward = np.full(shape=vec_env.num_envs, fill_value=0.0)
                if "success" in self.metrics_to_track:
                    success = np.full(shape=vec_env.num_envs, fill_value=0.0)

                self.logger.log("train/episode", episode, step)

                if step % exp_config.save_freq == 0:
                    if exp_config.save.buffer.should_save:
                        self.replay_buffer.save(
                            self.buffer_dir,
                            size_per_chunk=exp_config.save.buffer.size_per_chunk,
                            num_samples_to_save=exp_config.save.buffer.num_samples_to_save,
                        )

                if step % self.config.experiment.save_video_freq == 0 and self.config.experiment.save_video and step > 0:
                    print('Recording videos ...')
                    self.record_videos_for_transformer(self.col_agent, seq_len=self.seq_len)
            
            if step < exp_config.init_steps_online_tf and not exp_config.follow_col_network:
                action = np.asarray(
                    [self.action_space.sample() for _ in range(vec_env.num_envs)]
                )  # (num_envs, action_dim)
            elif step > exp_config.stu_expert_steps:
                for i in self.env_indices_i:
                    with agent_utils.eval_mode(self.expert[i]):
                        env_obs_i = {
                            key: value[self.env_indices[i]] for key, value in col_obs.items()
                        }
                        if exp_config.distill_scripted_policy:
                            action[self.env_indices[i]] = np.clip(self.scripted_policies[i].get_action(self.scripted_policies[i], env_obs_i["env_obs"].cpu().numpy()), -1, 1)
                        else:
                            action[self.env_indices[i]] = self.expert[i].sample_action(
                                multitask_obs=env_obs_i,
                                modes=[eval_mode,],
                            )
            else:
                with agent_utils.eval_mode(self.col_agent):
                    action[self.env_indices] = self.col_agent.sample_action(
                    #action[self.env_indices] = self.col_agent.select_action(
                        states=states[self.env_indices],
                        actions=actions[self.env_indices],
                        rewards=rewards[self.env_indices],
                        task_ids=torch.tensor(self.task_num[self.env_indices_i])
                    )

            # calculate the task encoding with random initialized TT    
            encoding = self.col_agent.calculate_task_encoding(
                states=states[self.env_indices],
                actions=actions[self.env_indices],
                rewards=rewards[self.env_indices],
                task_ids=torch.tensor(self.task_num[self.env_indices_i])
            )

            # run training update
            if step >= exp_config.init_steps_online_tf and step % exp_config.actor_update_freq == 0 and step < exp_config.stu_expert_steps:
                num_updates = (
                    exp_config.init_steps_online_tf if step == exp_config.init_steps_online_tf else 1
                )
                for _ in range(num_updates):
                    self.col_agent.distill_actor(self.replay_buffer, self.logger, step, cls_token=self.cls_token, tb_log=True)

            next_col_obs, reward, done, info = vec_env.step(action)

            if exp_config.use_unscaled:
                reward = info['unscaled_reward']

            episode_reward += reward
            if "success" in self.metrics_to_track:
                success += info['success']

            new_state = torch.cat((next_col_obs['env_obs'][:,:18], next_col_obs['env_obs'][:,36:]), dim=1).float()[:, None]
            new_action = torch.tensor(action).float()[:, None]
            new_reward = torch.tensor(reward).float()[:, None, None]

            states = torch.cat((states[:, -self.seq_len+1:], new_state), dim=1)
            actions = torch.cat((actions[:, -self.seq_len+2:], new_action), dim=1)
            rewards = torch.cat((rewards[:, -self.seq_len+2:], new_reward), dim=1)

            if self.should_reset_env_manually:
                if (episode_step[0] + 1) % self.max_episode_steps == 0:
                    if exp_config.reset_goal_state:
                        self.reset_goal_locations_in_order()
                    next_col_obs = vec_env.reset()
                    states = torch.cat((next_col_obs['env_obs'][:,:18], next_col_obs['env_obs'][:,36:]), dim=1).float()[:, None]
                    actions = torch.empty(vec_env.num_envs, 0, 4)
                    rewards = torch.empty(vec_env.num_envs, 0, 1)
                    
            # allow infinite bootstrap
            for index in self.env_indices_i:
                q_target, mu, log_std = self.expert[index].compute_q_target_and_policy_density(
                                            env_obs=col_obs["env_obs"][self.env_indices[index]][None].float().to(self.device),
                                            next_env_obs=next_col_obs["env_obs"][self.env_indices[index]][None].float().to(self.device),
                                            task_obs=torch.tensor(self.env_indices[index]).to(self.device)
                                        )
                if exp_config.distill_scripted_policy:
                    mu = np.clip(self.scripted_policies[index].get_action(self.scripted_policies[index], col_obs["env_obs"][self.env_indices[index]].cpu().numpy()), -1, 1)
                else:
                    mu=mu.cpu().numpy()
                done_bool = (
                    0
                    if episode_step[self.env_indices[index]] + 1 == self.max_episode_steps
                    else float(done[self.env_indices[index]])
                )
                self.replay_buffer.add(
                    col_obs["env_obs"][self.env_indices[index]],
                    next_col_obs["env_obs"][self.env_indices[index]],
                    action[self.env_indices[index]],
                    reward[self.env_indices[index]],
                    done=done_bool,
                    task_obs=self.task_num[index],
                    encoding=encoding[index].cpu().numpy(),
                    q_value=q_target.cpu().numpy(),
                    mu=mu,
                    log_std=log_std.cpu().numpy(),
                )

            col_obs = next_col_obs
            episode_step += 1

        print(f"Finished online training")

        if self.config.experiment.save_video:
            print('start recording videos ...')
            self.record_videos_for_transformer(self.col_agent, seq_len=self.seq_len)
            print('video recording finished. Check folder:{}'.format(self.video.dir_name))

        if self.config.experiment.save.buffer.delete_replaybuffer:
            self.replay_buffer.delete_from_filesystem(self.buffer_dir)

        self.close_envs()

        print("Finished online transformer learning")

    def run_train_student_finetuning(self):
        """Run training of student experiment."""
        exp_config = self.config.experiment

        vec_env = self.envs["train"]

        episode_reward, episode_step, done = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 0, True]
        ]

        if "success" in self.metrics_to_track:
            success = np.full(shape=vec_env.num_envs, fill_value=0.0)

        info = {}

        assert self.start_step >= 0
        episode = self.start_step // self.max_episode_steps

        start_time = time.time()

        col_obs = vec_env.reset()
        states = torch.cat((col_obs['env_obs'][:,:18], col_obs['env_obs'][:,36:]), dim=1).float()[:, None]
        actions = torch.empty(vec_env.num_envs, 0, 4)
        rewards = torch.empty(vec_env.num_envs, 0, 1)

        action = np.asarray([self.action_space.sample() for _ in range(vec_env.num_envs)])

        train_mode = ["train" for _ in range(vec_env.num_envs)]

        print("Start finetuning ... ")

        for step in range(self.start_step, exp_config.num_student_online_trainsteps):
            if step % self.max_episode_steps == 0:
                if step > 0:
                    if "success" in self.metrics_to_track:
                        success = (success > 0).astype("float")
                        for index in self.env_indices:
                            self.logger.log(
                                f"train/success_env_index_{index}",
                                success[index],
                                step,
                            )
                        success = success[self.env_indices]
                        self.logger.log("train/success", success.mean(), step)
                    for index in self.env_indices:
                        self.logger.log(
                            f"train/episode_reward_env_index_{index}",
                            episode_reward[index],
                            step,
                        )

                    self.logger.log("train/duration", time.time() - start_time, step)
                    start_time = time.time()
                    self.logger.dump(step)

                # evaluate agent periodically
                if step % exp_config.eval_freq == 0:
                    self.evaluate_transformer(
                        agent=self.student, vec_env=self.envs["eval"], step=step, episode=episode, seq_len=self.seq_len, sample_actions=False, reward_unscaled=exp_config.use_unscaled
                    )

                episode += 1
                episode_reward = np.full(shape=vec_env.num_envs, fill_value=0.0)
                if "success" in self.metrics_to_track:
                    success = np.full(shape=vec_env.num_envs, fill_value=0.0)

                self.logger.log("train/episode", episode, step)

                if step % exp_config.save_freq == 0:
                    if exp_config.save.model:
                        self.student.save(
                            self.student_model_dir,
                            step=step,
                            retain_last_n=exp_config.save.model.retain_last_n,
                        )
                    if exp_config.save.buffer.should_save:
                        self.replay_buffer.save(
                            self.buffer_dir,
                            size_per_chunk=exp_config.save.buffer.size_per_chunk,
                            num_samples_to_save=exp_config.save.buffer.num_samples_to_save,
                        )

                if step % self.config.experiment.save_video_freq == 0 and self.config.experiment.save_video and step == 0:
                    print('Recording videos ...')
                    self.record_videos_for_transformer(self.student, seq_len=self.seq_len)

            # safe the encoding for later evalutation
            encoding = self.student.calculate_task_encoding(
                states=states[self.env_indices],
                actions=actions[self.env_indices],
                rewards=rewards[self.env_indices],
                task_ids=torch.tensor(self.task_num[self.env_indices_i])
            )

            with agent_utils.eval_mode(self.student):
                action[self.env_indices] = self.student.sample_action(
                    states=states[self.env_indices],
                    actions=actions[self.env_indices],
                    rewards=rewards[self.env_indices],
                    task_ids=torch.tensor(self.task_num[self.env_indices_i])
                )

            # run training update
            if step >= exp_config.init_steps_stu:
                num_updates = (
                    exp_config.init_steps_stu if step == exp_config.init_steps_stu else 1
                )
                for _ in range(num_updates):
                   self.student.update(self.replay_buffer, self.logger, step)

            next_col_obs, reward, done, info = vec_env.step(action)

            if exp_config.use_unscaled:
                reward = info['unscaled_reward']

            episode_reward += reward
            if "success" in self.metrics_to_track:
                success += info['success']

            new_state = torch.cat((next_col_obs['env_obs'][:,:18], next_col_obs['env_obs'][:,36:]), dim=1).float()[:, None]
            new_action = torch.tensor(action).float()[:, None]
            new_reward = torch.tensor(reward).float()[:, None, None]

            states = torch.cat((states[:, -self.seq_len+1:], new_state), dim=1)
            actions = torch.cat((actions[:, -self.seq_len+2:], new_action), dim=1)
            rewards = torch.cat((rewards[:, -self.seq_len+2:], new_reward), dim=1)

            if self.should_reset_env_manually:
                if (episode_step[0] + 1) % self.max_episode_steps == 0:
                    next_col_obs = vec_env.reset()
                    states = torch.cat((next_col_obs['env_obs'][:,:18], next_col_obs['env_obs'][:,36:]), dim=1).float()[:, None]
                    actions = torch.empty(vec_env.num_envs, 0, 4)
                    rewards = torch.empty(vec_env.num_envs, 0, 1)
                    
            # allow infinite bootstrap
            for index in self.env_indices_i:
                done_bool = (
                    0
                    if episode_step[self.env_indices[index]] + 1 == self.max_episode_steps
                    else float(done[self.env_indices[index]])
                )
                self.replay_buffer.add(
                    col_obs["env_obs"][self.env_indices[index]],
                    next_col_obs["env_obs"][self.env_indices[index]],
                    action[self.env_indices[index]],
                    reward[self.env_indices[index]],
                    done=done_bool,
                    task_obs=self.env_indices[index],
                    encoding=encoding.detach().cpu().numpy(),
                    q_value=0,
                    mu=[0,0,0,0],
                    log_std=[0,0,0,0],
                )

            col_obs = next_col_obs
            episode_step += 1

        print(f"Finished online training")

        if self.config.experiment.save_video:
            print('start recording videos ...')
            self.record_videos_for_transformer(self.student, tag=f"result", seq_len=self.seq_len)
            print('video recording finished. Check folder:{}'.format(self.video.dir_name))

        if self.config.experiment.save.buffer.delete_replaybuffer:
            self.replay_buffer.delete_from_filesystem(self.buffer_dir)

        self.close_envs()

        print("Finished student learning")

    def run_distilling_student_online(self):
        """Run training of student experiment."""
        exp_config = self.config.experiment

        vec_env = self.envs["train"]

        episode_reward, episode_step, done = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 0, True]
        ]  # (num_envs, 1)

        if "success" in self.metrics_to_track:
            success = np.full(shape=vec_env.num_envs, fill_value=0.0)

        info = {}

        assert self.start_step >= 0
        episode = self.start_step // self.max_episode_steps

        col_obs = vec_env.reset()

        action = np.asarray([self.action_space.sample() for _ in range(vec_env.num_envs)])

        train_mode = ["train" for _ in range(vec_env.num_envs)]
        
        print(f"Starting online training")

        episode_reward, episode_step, done = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 0, True]
        ]  # (num_envs, 1)

        if "success" in self.metrics_to_track:
            success = np.full(shape=vec_env.num_envs, fill_value=0.0)

        info = {}
        episode = self.start_step // self.max_episode_steps

        start_time = time.time()

        col_obs = vec_env.reset()
        states = torch.cat((col_obs['env_obs'][:,:18], col_obs['env_obs'][:,36:]), dim=1).float()[:, None]
        actions = torch.empty(vec_env.num_envs, 0, 4)
        rewards = torch.empty(vec_env.num_envs, 0, 1)

        action = np.asarray([self.action_space.sample() for _ in range(vec_env.num_envs)])

        self.evaluate_transformer(
            agent=self.col_agent, vec_env=self.envs["eval"], step=0, episode=episode, seq_len=self.seq_len, sample_actions=False, reward_unscaled=exp_config.use_unscaled
        )
        self.logger.dump(0)

        for step in range(self.start_step, exp_config.num_student_online_trainsteps2):
            if step % self.max_episode_steps == 0:
                if step > 0:
                    if "success" in self.metrics_to_track:
                        success = (success > 0).astype("float")
                        for index in self.env_indices:
                            self.logger.log(
                                f"train/success_env_index_{index}",
                                success[index],
                                step,
                            )
                        success = success[self.env_indices]
                        self.logger.log("train/success", success.mean(), step)
                    for index in self.env_indices:
                        self.logger.log(
                            f"train/episode_reward_env_index_{index}",
                            episode_reward[index],
                            step,
                        )

                    self.logger.log("train/duration", time.time() - start_time, step)
                    start_time = time.time()
                    self.logger.dump(step)

                # evaluate agent periodically
                if step % exp_config.eval_freq == 0:
                    self.evaluate_vec_env_of_tasks(
                        vec_env=self.envs["eval"], step=step, episode=episode, agent="student"
                    )

                episode += 1
                episode_reward = np.full(shape=vec_env.num_envs, fill_value=0.0)
                if "success" in self.metrics_to_track:
                    success = np.full(shape=vec_env.num_envs, fill_value=0.0)

                self.logger.log("train/episode", episode, step)

                if step % exp_config.save_freq == 0:
                    if exp_config.save.model:
                        self.student.save(
                            self.student_model_dir,
                            step=step,
                            retain_last_n=exp_config.save.model.retain_last_n,
                        )
                    if exp_config.save.buffer.should_save:
                        self.replay_buffer.save(
                            self.buffer_dir,
                            size_per_chunk=exp_config.save.buffer.size_per_chunk,
                            num_samples_to_save=exp_config.save.buffer.num_samples_to_save,
                        )

                if step % self.config.experiment.save_video_freq == 0 and self.config.experiment.save_video and step > 0:
                    print('Recording videos ...')
                    self.record_videos(eval_agent="student", tag=f"student_step{step}")

            if step < exp_config.init_steps_stu2:
                action = np.asarray(
                    [self.action_space.sample() for _ in range(vec_env.num_envs)]
                )
            else:
                with agent_utils.eval_mode(self.student):
                    env_obs_i = { 
                        key: value[self.env_indices] for key, value in col_obs.items()
                    }
                    action[self.env_indices] = self.student.sample_action(
                        multitask_obs=env_obs_i,
                        modes=[
                            train_mode,
                        ],
                    )

            # calculate q_target and best action
            if step < exp_config.expert_train_step:
                if not exp_config.use_scripted_mu:
                    mu, log_std, _ = self.col_agent.calculate_q_targets_expert_action(
                                                states=states[self.env_indices],
                                                actions=actions[self.env_indices],
                                                rewards=rewards[self.env_indices],
                                                task_ids=torch.tensor(self.task_num[self.env_indices_i])
                                            )
                else:
                    env_obs_i = {
                        key: value[0] for key, value in col_obs.items()
                    }
                    mu = np.clip(self.scripted_policies[0].get_action(self.scripted_policies[0], env_obs_i["env_obs"].cpu().numpy()), -1, 1)
                    log_std = 1e-1
                    
                expert_sigma = np.clip(np.exp(log_std), a_min=1e-4, a_max=None)
                diff = action - mu
                normalization_term = (1 / (expert_sigma * np.sqrt(2 * math.pi))).prod(axis=1)
                likelihood = ((1 / (expert_sigma * np.sqrt(2 * math.pi))) * np.exp(-0.5 * (diff / expert_sigma) ** 2)).prod(axis=1)
                q_target = likelihood / normalization_term
                
                # safe the encoding for later evalutation
                encoding = self.col_agent.calculate_task_encoding(
                    states=states[self.env_indices],
                    actions=actions[self.env_indices],
                    rewards=rewards[self.env_indices],
                    task_ids=torch.tensor(self.task_num[self.env_indices_i])
                )
            else:
                mu, log_std, q_target = 0., 0., 0.
                encoding = torch.zeros((1,6))

            # run training update
            if step >= exp_config.init_steps_stu2:
                num_updates = (
                    exp_config.init_steps_stu2 if step == exp_config.init_steps_stu2 else 1
                )
                for _ in range(num_updates):
                    if step < exp_config.expert_train_step:
                        expert_weight = exp_config.start_weight * np.clip((1-((step-exp_config.init_steps_stu2-exp_config.constant_expert_weight) /
                                                                            (exp_config.expert_train_step-exp_config.init_steps_stu2-exp_config.constant_expert_weight))), 0., 1.)
                        self.student.update_with_reward_shapening(self.replay_buffer, self.logger, step, expert_weight)
                    else:
                        self.student.update(self.replay_buffer, self.logger, step)

            next_col_obs, reward, done, info = vec_env.step(action)

            if exp_config.use_unscaled:
                reward = info['unscaled_reward']

            episode_reward += reward
            if "success" in self.metrics_to_track:
                success += info['success']

            new_state = torch.cat((next_col_obs['env_obs'][:,:18], next_col_obs['env_obs'][:,36:]), dim=1).float()[:, None]
            new_action = torch.tensor(action).float()[:, None]
            new_reward = torch.tensor(reward).float()[:, None, None]

            states = torch.cat((states[:, -self.seq_len+1:], new_state), dim=1)
            actions = torch.cat((actions[:, -self.seq_len+2:], new_action), dim=1)
            rewards = torch.cat((rewards[:, -self.seq_len+2:], new_reward), dim=1)

            if self.should_reset_env_manually:
                if (episode_step[0] + 1) % self.max_episode_steps == 0:
                    if exp_config.reset_goal_state:
                        self.reset_goal_locations_in_order()
                    next_col_obs = vec_env.reset()
                    states = torch.cat((next_col_obs['env_obs'][:,:18], next_col_obs['env_obs'][:,36:]), dim=1).float()[:, None]
                    actions = torch.empty(vec_env.num_envs, 0, 4)
                    rewards = torch.empty(vec_env.num_envs, 0, 1)
                    
            # allow infinite bootstrap
            for index in self.env_indices_i:
                done_bool = (
                    0
                    if episode_step[self.env_indices[index]] + 1 == self.max_episode_steps
                    else float(done[self.env_indices[index]])
                )
                self.replay_buffer.add(
                    col_obs["env_obs"][self.env_indices[index]],
                    next_col_obs["env_obs"][self.env_indices[index]],
                    action[self.env_indices[index]],
                    reward[self.env_indices[index]],
                    done=done_bool,
                    task_obs=self.env_indices[index],
                    encoding=encoding.detach().cpu().numpy(),
                    q_value=q_target,
                    mu=mu,
                    log_std=log_std,
                )

            col_obs = next_col_obs
            episode_step += 1

        print(f"Finished online training")

        if self.config.experiment.save_video:
            print('start recording videos ...')
            self.record_videos(eval_agent="student", tag="student_result")
            print('video recording finished. Check folder:{}'.format(self.video.dir_name))

        if self.config.experiment.save.buffer.delete_replaybuffer:
            self.replay_buffer.delete_from_filesystem(self.buffer_dir)

        self.close_envs()

        print("Finished student learning")

    def distill_collective_transformer(self):
        """Run the experiment."""
        exp_config = self.config.experiment

        assert self.col_start_step >= 0
        episode = self.col_start_step // self.max_episode_steps
        start_time = time.time()

        if self.col_start_step < exp_config.num_actor_train_step:
            for step in range(self.col_start_step, exp_config.num_actor_train_step):
                # evaluate agent periodically
                if step % exp_config.col_eval_freq == 0:
                    self.logger.log("train/duration", time.time() - start_time, step)
                    start_time = time.time()
                    if step % exp_config.col_eval_freq * 3 == 0:
                        self.evaluate_transformer(
                            agent=self.col_agent, vec_env=self.envs["eval"], step=step, episode=episode, seq_len=self.seq_len, sample_actions=False, reward_unscaled=exp_config.use_unscaled
                        )

                # run distillation updates
                self.col_agent.distill_actor(self.replay_buffer_distill, self.logger, step, tb_log=True)
                self.col_start_step += 1

                if exp_config.save.model and step % exp_config.save_freq_transformer == 0:
                    self.col_agent.save(
                        self.col_model_dir,
                        step=step,
                        retain_last_n=exp_config.save.model.retain_last_n,
                    )

        if self.col_start_step < exp_config.num_actor_train_step+exp_config.num_critic_train_step:
            for step in range(self.col_start_step, exp_config.num_actor_train_step+exp_config.num_critic_train_step):
                # evaluate agent periodically
                if step % exp_config.col_eval_freq == 0:
                    self.logger.log("train/duration", time.time() - start_time, step)
                    start_time = time.time()
                    self.logger.dump(step)

                # run distillation updates
                self.col_agent.distill_critic(self.replay_buffer_distill, self.logger, step, tb_log=True, action_space=self.action_space)
                self.col_start_step += 1

                if exp_config.save.model and step % exp_config.save_freq_transformer == 0:
                    self.col_agent.save(
                        self.col_model_dir,
                        step=step,
                        retain_last_n=exp_config.save.model.retain_last_n,
                    )

        if self.config.experiment.save_video:
            print('start recording videos ...')
            self.record_videos_for_transformer(self.col_agent, seq_len=self.seq_len)
            print('video recording finished. Check folder:{}'.format(self.video.dir_name))

        if exp_config.save.model:
            self.col_agent.save(
                self.col_model_dir,
                step=exp_config.num_actor_train_step+exp_config.num_critic_train_step,
                retain_last_n=exp_config.save.model.retain_last_n,
            )

        self.close_envs()

        print("finished learning")

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
                    self.record_videos_for_transformer(self.col_agent, tag=f"sample_{i}", seq_len=self.seq_len)
                elif exp_config.evaluate_transformer=="agent":
                    self.record_videos(eval_agent="worker", tag=f"sample_{i}")
                else:
                    self.record_videos(eval_agent="scripted", tag=f"sample_{i}")

                print('video recording finished. Check folder:{}'.format(self.video.dir_name))

        self.close_envs()

        print("finished learning")
        print(f"Evaluation: robot {self.robot_type} {count} from {exp_config.evaluate_episodes}")

    def record_videos(self, eval_agent: str = "worker", tag="result", num_eps_per_task_to_record=1):
        """
        Record videos of all envs, each env performs one episodes.
        """
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

            self.video.record(frame=video_envs[self.env_indices[env_idx]].render())
            
            video_envs[self.env_indices[env_idx]].env.mujoco_renderer.viewer.cam.azimuth = 140.0
            video_envs[self.env_indices[env_idx]].env.mujoco_renderer.viewer.cam.elevation = -40.0
            video_envs[self.env_indices[env_idx]].env.mujoco_renderer.viewer.cam.lookat = np.array([0, 0.5, 0.0])
            video_envs[self.env_indices[env_idx]].env.mujoco_renderer.viewer.cam.distance = 2.0

            episode_reward = 0

            for episode_step in range(self.max_episode_steps * num_record):

                # record for env_idx env
                self.video.record(frame=video_envs[self.env_indices[env_idx]].render())
                
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
                    self.video.record(frame=self.list_envs[self.env_indices[env_idx]].render())
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

    def record_videos_for_transformer(self, agent, tag="result", seq_len=16, num_eps_per_task_to_record=1):
        """
        Record videos of all envs, each env performs one episodes.
        """
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

            self.video.record(frame=video_envs[self.env_indices[env_idx]].render())
            
            video_envs[self.env_indices[env_idx]].env.mujoco_renderer.viewer.cam.azimuth = 140.0
            video_envs[self.env_indices[env_idx]].env.mujoco_renderer.viewer.cam.elevation = -40.0
            video_envs[self.env_indices[env_idx]].env.mujoco_renderer.viewer.cam.lookat = np.array([0, 0.5, 0.0])
            video_envs[self.env_indices[env_idx]].env.mujoco_renderer.viewer.cam.distance = 2.0

            episode_reward = 0

            for episode_step in range(self.max_episode_steps * num_record):

                # record for env_idx env
                self.video.record(frame=video_envs[self.env_indices[env_idx]].render())
                
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
                    self.video.record(frame=self.list_envs[self.env_indices[env_idx]].render())
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


    def collect_trajectory(self, vec_env: VecEnv, num_steps: int) -> None:
        """Collect some trajectories, by unrolling the policy (in train mode),
        and update the replay buffer.
        Args:
            vec_env (VecEnv): environment to collect data from.
            num_steps (int): number of steps to collect data for.

        """
        raise NotImplementedError

    def generate_data_from_col_agent(self, tasks: List[str], num_episodes_per_task: int = 5) -> None:
        """Generate rollouts by running the collective agent policy for each task, using numpy sequences."""
        seq_len = self.seq_len
        action_dim = self.action_shape

        for task_idx, task_name in enumerate(tasks):
            task_data = random.choice(self.env_name_task_dict[task_name])
            self.envs['train'].call('set_task', task_data)

            result = self.evaluate_transformer(
                agent=self.col_agent, vec_env=self.envs["eval"], step=step, episode=episode, seq_len=self.seq_len, sample_actions=(step==0), reward_unscaled=exp_config.use_unscaled
            )

            for ep in range(num_episodes_per_task):
                obs = self.envs['train'].reset()
                done = False
                # initialize numpy buffers
                state_buf = []  # list of np arrays shape (state_dim,)
                action_buf = []  # list of np arrays shape (action_dim,)
                reward_buf = []  # list of floats

                while not done:
                    # get raw obs numpy
                    raw = obs['env_obs']
                    if hasattr(raw, 'cpu'):
                        raw_np = raw[0].cpu().numpy()
                    else:
                        raw_np = raw[0]
                    # slice state dims
                    s = np.concatenate([raw_np[:18], raw_np[36:]], axis=-1)

                    # append zero for first step
                    if not action_buf:
                        action_buf.append(np.zeros(action_dim, dtype=np.float32))
                        reward_buf.append(0.0)
                    state_buf.append(s)

                                        # build fixed-length numpy sequences
                    # determine state dimension from raw slice
                    state_dim = s.shape[0]
                    seq_s = np.zeros((seq_len, state_dim), dtype=np.float32)
                    seq_a = np.zeros((seq_len, action_dim), dtype=np.float32)
                    seq_r = np.zeros((seq_len, 1), dtype=np.float32)
                    buf_len = len(state_buf)
                    if buf_len >= seq_len:
                        seq_s[:] = np.stack(state_buf[-seq_len:], axis=0)
                        seq_a[:] = np.stack(action_buf[-seq_len:], axis=0)
                        seq_r[:] = np.array(reward_buf[-seq_len:], dtype=np.float32).reshape(seq_len,1)
                    else:
                        seq_s[-buf_len:] = np.stack(state_buf, axis=0)
                        seq_a[-buf_len:] = np.stack(action_buf, axis=0)
                        seq_r[-buf_len:] = np.array(reward_buf, dtype=np.float32).reshape(buf_len,1)

                    # convert sequences to torch once
                    s_batch = torch.from_numpy(seq_s).unsqueeze(0).to(self.device)
                    a_batch = torch.from_numpy(seq_a).unsqueeze(0).to(self.device)
                    r_batch = torch.from_numpy(seq_r).unsqueeze(0).to(self.device)
                    t_batch = torch.tensor([[task_idx]], device=self.device)

                    with agent_utils.eval_mode(self.col_agent):
                        mu, log_std, q_val = self.col_agent.calculate_q_targets_expert_action(
                            states=s_batch, actions=a_batch, rewards=r_batch, task_ids=t_batch
                        )
                    action_np = mu[0].cpu().numpy()

                    next_obs, rewards, dones, infos = self.envs['train'].step(action_np)
                    reward_val = float(rewards[0])
                    done = bool(dones[0])

                    # get next obs
                    raw_next = next_obs['env_obs']
                    if hasattr(raw_next, 'cpu'):
                        next_np = raw_next[0].cpu().numpy()
                    else:
                        next_np = raw_next[0]
                    # slice next state
                    s_next = np.concatenate([next_np[:18], next_np[36:]], axis=-1)

                    # add to buffer
                    self.replay_buffer.add(
                        env_obs=s,
                        action=action_np[0],
                        reward=np.array([reward_val], dtype=np.float32),
                        next_env_obs=s_next,
                        done=done,
                        task_obs=np.array([task_idx], dtype=np.int64)
                    )

                    # update numpy buffers
                    state_buf.append(s_next)
                    action_buf.append(action_np[0])
                    reward_buf.append(reward_val)
                    obs = next_obs
        print("Data generation complete.")


    def distill_policy(self):
        """Run distillation from col_agent to student using KL divergence on logits."""
        exp_config = self.config.experiment
        #assert hasattr(self, "col_agent") and hasattr(self, "student_agent")

        self.student_agent.train()  # Only the student needs to train

        optimizer = self.student_agent._optimizers["actor"]

        distill_steps = exp_config.distill_steps
        batch_size = exp_config.batch_size
        temperature = exp_config.temperature
        start_time = time.time()

        if self.replay_buffer.is_empty():
            self.generate_data_from_col_agent(["reach-v2"],num_episodes_per_task=5)


        for step in range(distill_steps):
            print(f"Buffer size: {len(self.replay_buffer)}, batch_size: {batch_size}")
            if len(self.replay_buffer) < batch_size:
                print("Buffer not sufficiently filled. Abort or wait.")
                return

            sample: ReplayBufferSample = self.replay_buffer.sample(batch_size)
            obs = sample.observation.to(self.device)

            # Get teacher logits (e.g., Q-values or action logits)
            with torch.no_grad():
                teacher_logits = self.col_agent.get_action_logits(obs) / temperature
                teacher_probs = torch.softmax(teacher_logits, dim=-1)

            # Get student logits
            student_logits = self.student_agent.get_action_logits(obs) / temperature
            student_log_probs = torch.log_softmax(student_logits, dim=-1)

            # KL Divergence loss
            loss = torch.nn.functional.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean',
                log_target=False
            ) * (temperature ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % exp_config.log_interval == 0:
                self.logger.log("distill/loss", loss.item(), step)
                self.logger.log("distill/duration", time.time() - start_time, step)
                self.logger.dump(step)
                print(f"[Distill Step {step}] Loss: {loss.item():.4f}")
                start_time = time.time()

            if exp_config.save.model and step % exp_config.save_freq == 0:
                self.student_agent.save(
                    self.student_model_dir,
                    step=step,
                    retain_last_n=exp_config.save.model.retain_last_n,
                )

        self.close_envs()
        print("Finished policy distillation.")

    def run_train_world_model(self):
        """
        用 TransformerAgent 的 train_world_model 在离线 buffer 上训练 world model。
        包含周期性评估和早停 (Early Stopping) 功能。
        """
        exp_config = self.config.experiment
        
        # ================= [新增] 早停配置初始化 =================
        # 从配置中读取，如果未配置则默认 patience=10, min_delta=0.0
        use_early_stopping = getattr(exp_config, "early_stopping", False)
        
        # 1. 读取 patience，如果为 None 则设为默认值 10
        es_patience = getattr(exp_config, "es_patience", 10)
        if es_patience is None:
            es_patience = 10
            
        # 2. 读取 min_delta，如果为 None 则设为默认值 0.0
        es_min_delta = getattr(exp_config, "es_min_delta", 0.0)
        if es_min_delta is None:
            es_min_delta = 0.0
            
        # 内部状态变量
        es_counter = 0
        es_best_loss = float('inf')
        early_stop_triggered = False
        # ========================================================

        print("\n" + "="*50)
        print(">>> Starting World Model Training & Evaluation <<<")
        print(f"Total training steps: {exp_config.num_wm_train_step}")
        print(f"Periodic evaluation frequency: {exp_config.wm_eval_freq} steps")
        
        if use_early_stopping:
            print(f"Early Stopping ENABLED: patience={es_patience}, min_delta={es_min_delta}")
        else:
            print("Early Stopping DISABLED")
        print("="*50 + "\n")

        # 检查验证集是否存在
        has_validation_buffer = hasattr(self, "replay_buffer_val") and len(self.replay_buffer_val) > 0
        if not has_validation_buffer:
            print("[WARN] Validation buffer (`replay_buffer_val`) not found or is empty. Periodic evaluation & Early Stopping will be SKIPPED.")
            use_early_stopping = False # 强制关闭早停

        start_time = time.time()
        assert self.wm_start_step >= 0

        # --- 主训练循环 ---
        for step in range(self.wm_start_step, exp_config.num_wm_train_step):
            
            # === 1. 核心训练步骤 ===
            self.col_agent.train_world_model(
                self.replay_buffer_distill, self.logger, step, tb_log=True
            )
            self.wm_start_step += 1

            # === 2. 周期性评估与早停逻辑 ===
            if step > 0 and step % exp_config.wm_eval_freq == 0 and has_validation_buffer:
                print(f"\n--- Running periodic WM evaluation at step {step} ---")
                
                # 执行评估 (注意：evaluate_world_model 内部必须处理好 model.eval() 和 model.train() 的切换)
                avg_val_losses = self.col_agent.evaluate_world_model(
                    validation_buffer=self.replay_buffer_val,
                    batch_size=self.config.replay_buffer.transformer_col_replay_buffer.batch_size
                )
                
                if avg_val_losses:
                    # 日志记录 (加上 eval/ 前缀)
                    for loss_name, loss_value in avg_val_losses.items():
                        self.logger.log(f"eval/{loss_name}", loss_value, step)

                    # --- [新增] 早停核心逻辑 ---
                    if use_early_stopping:
                        # 必须确保 evaluate_world_model 返回的 key 是 "avg_total_loss"
                        monitor_metric = avg_val_losses.get("avg_total_loss", None)
                        
                        if monitor_metric is not None:
                            # 检查是否显著提升 (Loss 降低)
                            if monitor_metric < (es_best_loss - es_min_delta):
                                print(f"✅ [EarlyStop] Validation loss improved: {es_best_loss:.5f} -> {monitor_metric:.5f}")
                                es_best_loss = monitor_metric
                                es_counter = 0 # 重置计数器
                                
                                # 保存最佳模型 (Best Checkpoint)
                                # 使用 "best" 作为 step 参数，保存为 world_model_best.pt
                                self.col_agent.save_only_world_model(
                                    self.col_model_dir, step="best", retain_last_n=1
                                )
                                print(f"   Saved best model checkpoint to {self.col_model_dir}")
                                
                            else:
                                es_counter += 1
                                print(f"⚠️ [EarlyStop] No significant improvement. Counter: {es_counter}/{es_patience}")
                                
                                if es_counter >= es_patience:
                                    print(f"\n🛑 [EarlyStop] Patience exhausted! Stopping training at step {step}.")
                                    early_stop_triggered = True
                        else:
                            print(f"[WARN] Metric 'avg_total_loss' not found in evaluation results. Cannot update early stopping status.")
                    # ---------------------------

            # === 3. 周期性保存 (Regular Checkpoint) 和日志打印 ===
            if step % exp_config.col_eval_freq == 0:
                self.logger.log("train/duration", time.time() - start_time, step)
                start_time = time.time()
                self.logger.dump(step)

            # 定期保存 (按原有逻辑，比如每 10000 步存一次)
            if exp_config.save.model and step % exp_config.save_freq_transformer == 0 and step > 0:
                self.col_agent.save_only_world_model(
                    self.col_model_dir,
                    step=step,
                    retain_last_n=exp_config.save.model.retain_last_n,
                )
                self.col_agent.save_metadata(self.col_model_dir, step=step)
                self.col_agent.save_world_model_optimizer(self.col_model_dir,
                    step=step,
                    retain_last_n=exp_config.save.model.retain_last_n)

            # 如果触发了早停，跳出循环
            if early_stop_triggered:
                break

        # --- 训练结束后的处理 ---
        
        # 1. 只有在非早停情况下 (正常跑完 num_wm_train_step)，才保存 Final Checkpoint
        #    如果是早停退出的，Best Checkpoint 已经存好了，没必要存一个性能差的 Final。
        #    或者你为了完整性，也可以强行存一个。这里保留原逻辑，但加上条件判断。
        if not early_stop_triggered and exp_config.save.model:
            print("Saving final model checkpoint...")
            self.col_agent.save_only_world_model(
                self.col_model_dir,
                step=self.wm_start_step, # 或者 exp_config.num_wm_train_step
                retain_last_n=exp_config.save.model.retain_last_n,
            )
            self.col_agent.save_metadata(self.col_model_dir, step=self.wm_start_step)
            self.col_agent.save_world_model_optimizer(self.col_model_dir,
                step=self.wm_start_step,
                retain_last_n=exp_config.save.model.retain_last_n)

        self.close_envs()
        print("finished world model training")
        
        if use_early_stopping:
            print(f"Training finished. Best validation loss: {es_best_loss:.5f}")

    def evaluate_world_model(self):
        """Evaluate the pre-trained world model on the validation dataset."""
        print("\n" + "="*50)
        print(">>> Starting World Model Evaluation <<<")
        print("="*50 + "\n")
        
        # 1. 安全检查，确保所有需要的组件都已就绪
        if not hasattr(self, "col_agent") or not self.col_agent:
            print("[ERROR] Collective agent (col_agent) was not initialized. Check your setup.")
            return
        if not self.col_agent.use_world_model:
            print("[ERROR] `use_world_model` is False. Cannot evaluate world model.")
            return
        if not hasattr(self, "replay_buffer_val") or len(self.replay_buffer_val) == 0:
            print("[ERROR] Validation replay buffer (`replay_buffer_val`) not found or is empty.")
            return

        # 2. 将评估任务委托给 agent 对象处理，这是良好的代码实践
        #    我们假设 agent 中会有一个 evaluate_world_model 方法
        avg_losses = self.col_agent.evaluate_world_model(self.replay_buffer_val)

        # 3. 打印和记录结果
        print("\n" + "--- World Model Evaluation Results ---")
        if not avg_losses:
            print("Evaluation did not return any results.")
        else:
            for loss_name, loss_value in avg_losses.items():
                print(f"  -> {loss_name}: {loss_value:.4f}")
                # 将结果记录到日志中，方便后续分析
                self.logger.log(f"eval/{loss_name}", loss_value, 0) # 记录在 step 0
        
        self.logger.dump(0) # 强制将日志写入文件
        print("--- Results logged. Evaluation finished. ---\n")
        
        self.close_envs()

    def generate_noise_injected_data(self):
        """
        Generate offline distillation data using a Script Policy with Noise Injection.
        Configuration is controlled via hydra config (self.config.experiment).
        """
        exp_config = self.config.experiment
        vec_env = self.envs["train"]

        # === 1. 从 Config 读取参数 ===
        total_steps = exp_config.num_distill_gen_steps
        sigma_high = exp_config.distill_noise_sigma_start 
        sigma_low = exp_config.distill_noise_sigma_end   
        
        # 定义 Curriculum 阶段
        phase_1_end = int(0.2 * total_steps) # Exploration
        phase_2_end = int(0.7 * total_steps) # Refinement

        print(f"Starting Noise-Injected Data Generation for {total_steps} steps...")

        # === 2. 初始化 ===
        episode_reward, episode_step, done = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 0, True]
        ]
        
        if "success" in self.metrics_to_track:
            success = np.full(shape=vec_env.num_envs, fill_value=0.0)

        # [修复 Bug]: 初始化 episode 变量
        # 这一行必须在循环开始前存在，否则会报 UnboundLocalError
        episode = self.start_step // self.max_episode_steps 

        start_time = time.time()
        col_obs = vec_env.reset()
        
        # 按照 run_training_worker 的风格初始化 mu 和 action
        mu = np.asarray([self.action_space.sample() for _ in range(vec_env.num_envs)])
        action = np.zeros_like(mu)

        # === 3. 主循环 ===
        target_step = self.start_step + total_steps
        
        for step in range(self.start_step, target_step):
            
            # --- A. 计算噪声 Sigma (Curriculum) ---
            relative_step = step - self.start_step
            
            if relative_step < phase_1_end:
                current_sigma = sigma_high
            elif relative_step < phase_2_end:
                # 线性衰减
                progress = (relative_step - phase_1_end) / (phase_2_end - phase_1_end)
                current_sigma = sigma_high - progress * (sigma_high - sigma_low)
            else:
                current_sigma = 0.0

            # --- B. 动作生成 ---
            for i in self.env_indices_i:
                env_obs_i = {
                    key: value[self.env_indices[i]] for key, value in col_obs.items()
                }
                
                # 1. 获取专家动作 (Label)
                raw_expert_act = self.scripted_policies[i].get_action(
                    self.scripted_policies[i], 
                    env_obs_i["env_obs"].cpu().numpy()
                )
                mu[self.env_indices[i]] = np.clip(raw_expert_act, -1, 1)
                
                # 2. 注入噪声 (Input)
                if current_sigma > 0:
                    noise = np.random.normal(0, current_sigma, size=raw_expert_act.shape)
                    noisy_act = raw_expert_act + noise
                else:
                    noisy_act = raw_expert_act
                
                # 3. Clip 实际执行动作
                action[self.env_indices[i]] = np.clip(noisy_act, -1, 1)

            # --- C. 环境交互 ---
            next_col_obs, reward, done, info = vec_env.step(action)
            
            if exp_config.use_unscaled:
                reward = info['unscaled_reward']

            episode_reward += reward
            if "success" in self.metrics_to_track:
                success += info['success']

            # --- D. 存入 Replay Buffer ---
            for index in self.env_indices_i:
                done_bool = (
                    0
                    if episode_step[self.env_indices[index]] + 1 == self.max_episode_steps
                    else float(done[self.env_indices[index]])
                )
                
                self.replay_buffer_distill[index].add(
                    col_obs["env_obs"][self.env_indices[index]],
                    action[self.env_indices[index]],
                    reward[self.env_indices[index]],
                    next_col_obs["env_obs"][self.env_indices[index]],
                    done_bool,
                    task_obs=self.task_num[self.env_indices[index]],
                    q_value=0.0, 
                    mu=mu[self.env_indices[index]], 
                    log_std=np.zeros_like(mu[self.env_indices[index]]) 
                )

            # --- E. Reset 逻辑 ---
            if self.should_reset_env_manually:
                if (episode_step[0] + 1) % self.max_episode_steps == 0:
                    if exp_config.reset_goal_state:
                        self.reset_goal_locations_in_order()
                    next_col_obs = vec_env.reset()

            # --- F. Logging & Saving (对齐 run_training_worker) ---
            # 这里的逻辑是：每完成一个 max_episode_steps (例如500步)，记录一次 Log 并增加 episode 计数
            if step % self.max_episode_steps == 0 and step > self.start_step:
                
                # 1. 记录 Episode 数 (修复了 UnboundLocalError)
                self.logger.log("train/episode", episode, step)
                
                # 2. 记录 Success 和 Reward
                if "success" in self.metrics_to_track:
                    success_mask = (success > 0).astype("float")
                    for index in self.env_indices:
                        self.logger.log(f"train/success_env_index_{index}", success_mask[index], step)
                    self.logger.log("train/success", success_mask.mean(), step)
                
                for index in self.env_indices:
                    self.logger.log(f"train/episode_reward_env_index_{index}", episode_reward[index], step)

                self.logger.log("train/duration", time.time() - start_time, step)
                start_time = time.time()
                self.logger.dump(step)

                # 3. 更新变量为下一个 Episode 做准备
                episode += 1
                episode_reward.fill(0.0)
                if "success" in self.metrics_to_track:
                    success.fill(0.0)

                # 4. 定期保存 Buffer
                if step % exp_config.save_freq == 0:
                    if exp_config.save.buffer.should_save:
                        for i in range(self.num_envs):
                            self.replay_buffer_distill[i].save(
                                self.buffer_dir_distill[i],
                                size_per_chunk=exp_config.save.buffer.size_per_chunk,
                                num_samples_to_save=exp_config.save.buffer.num_samples_to_save,
                            )
            
            col_obs = next_col_obs
            episode_step += 1

        # --- 结束处理 ---
        print(f"Finished generating {total_steps} steps of Noise-Injected Distill Data.")
        
        if exp_config.save.buffer.should_save:
            for i in range(self.num_envs):
                self.replay_buffer_distill[i].save(
                    self.buffer_dir_distill[i],
                    size_per_chunk=exp_config.save.buffer.size_per_chunk,
                    num_samples_to_save=exp_config.save.buffer.num_samples_to_save,
                )
                
        if self.config.experiment.save.buffer.delete_replaybuffer:
            for i in range(self.num_envs):
                self.replay_buffer_distill[i].delete_from_filesystem(self.buffer_dir_distill[i])
        
        self.close_envs()