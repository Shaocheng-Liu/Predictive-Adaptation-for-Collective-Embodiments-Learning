"""Dataset utilities for the diffusion policy baseline."""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset


class DiffusionPolicyDataset(Dataset):
    """Wraps the multi-robot buffer data for diffusion-policy training."""

    def __init__(
        self,
        data_path: str,
        obs_horizon: int = 20,
        pred_horizon: int = 4,
        max_episode_len: int = 400,
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.max_episode_len = max_episode_len

        data = {
            "states": [],
            "actions": [],
            "rewards": [],
            "policy_mu": [],
        }
        i = 0
        while os.path.isfile(os.path.join(data_path, f"data_chunk_{i}")):
            chunk = torch.load(
                os.path.join(data_path, f"data_chunk_{i}"),
                weights_only=False,
            )
            data["states"].extend(chunk["states"])
            data["actions"].extend(chunk["actions"])
            data["rewards"].extend(chunk["rewards"])
            if "policy_mu" in chunk:
                data["policy_mu"].extend(chunk["policy_mu"])
            i += 1

        if i == 0:
            raise FileNotFoundError(
                f"No data files found in {data_path}"
            )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.states = torch.tensor(np.stack(data["states"]), dtype=torch.float32).to(device)
        self.actions = torch.tensor(np.stack(data["actions"]), dtype=torch.float32).to(device)
        self.rewards = torch.tensor(np.stack(data["rewards"]), dtype=torch.float32).to(device)
        self.policy_mu = None
        if data["policy_mu"]:
            self.policy_mu = torch.tensor(np.stack(data["policy_mu"]), dtype=torch.float32).to(device)

        self.num_episodes = self.states.shape[0]
        self.valid_start_max = self.max_episode_len - self.obs_horizon - self.pred_horizon + 1
        assert self.valid_start_max > 0, (
            "Episode length too short for the given obs_horizon + pred_horizon"
        )

    def __len__(self) -> int:
        return self.num_episodes * self.valid_start_max

    def __getitem__(self, idx: int):
        ep_idx = idx // self.valid_start_max
        start = idx % self.valid_start_max

        obs_end = start + self.obs_horizon
        act_end = obs_end + self.pred_horizon

        obs_states = self.states[ep_idx, start:obs_end]
        obs_actions = self.actions[ep_idx, start:obs_end - 1]
        obs_rewards = self.rewards[ep_idx, start:obs_end - 1]
        target_actions = self.actions[ep_idx, obs_end:act_end]

        return obs_states, obs_actions, obs_rewards, target_actions