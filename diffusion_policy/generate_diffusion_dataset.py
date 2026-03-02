"""Generate the diffusion-policy dataset from raw experience buffers.

This script reads the raw ``.pt`` chunk files produced by the existing
training pipeline (stored under ``collective_buffer`` or ``buffer_distill``
directories) and serialises them into a format optimised for the diffusion
policy ``DiffusionPolicyDataset``.

Two buffer formats are supported:

* **TransformerReplayBuffer** (10-element payload — primary source for
  collective_buffer data)::

      [env_obses, next_env_obses, actions, rewards, not_dones,
       task_encodings, task_obs, policy_mu, policy_log_std, q_target]

  ``env_obses`` may already be compressed to 21-dim when
  ``compressed_state=True``.
  ``policy_mu`` is at **index 7**.

* **DistilledReplayBuffer** (9-element payload)::

      [env_obses, next_env_obses, actions, rewards, not_dones,
       task_obs, policy_mu, policy_log_std, q_target]

  ``env_obses`` is always raw 39-dim.
  ``policy_mu`` is at **index 6**.

This script extracts **both** ``actions`` (index 2) and ``policy_mu``
(teacher's intended action).  The diffusion policy can train on either
target via the ``--train_target`` flag in ``train.py``.

Each array is shaped ``(N, feature_dim)`` and represents *flat* timesteps
collected at 400-step episode boundaries.  This script:

    1. Loads every ``.pt`` chunk found under ``--buffer_dir``.
    2. Auto-detects the observation dimension (21 or 39).
    3. Reshapes the flat arrays into ``(-1, 400, dim)`` episodes.
    4. Compresses 39-dim observations to 21-dim if needed.
    5. Saves the result as ``data_chunk_*`` files consumable by
       ``DiffusionPolicyDataset``, including ``policy_mu``.

Usage
-----
    python diffusion_policy/generate_diffusion_dataset.py \\
        --buffer_dir logs/experiment_test/buffer/collective_buffer/train \\
        --output_dir diffusion_policy/data/train

    # Then train with teacher mu (recommended):
    python diffusion_policy/train.py --data_path diffusion_policy/data/train --train_target mu

    # Or train with raw actions (original approach):
    python diffusion_policy/train.py --data_path diffusion_policy/data/train --train_target action
"""

import argparse
import gc
import os
import re
import sys

import numpy as np
import torch


# ── Helpers ─────────────────────────────────────────────────────────────────

EPISODE_LEN = 400
OBS_COMPRESSED_DIM = 21
OBS_RAW_DIM = 39
ACTION_DIM = 4


def _compress_states(raw: np.ndarray) -> np.ndarray:
    """Compress raw 39-dim obs to 21-dim: ``state[:18] ++ state[36:39]``."""
    return np.concatenate([raw[..., :18], raw[..., 36:]], axis=-1)


def load_single_buffer_dir(buffer_path: str):
    """Load one buffer directory that may contain several ``start_end.pt`` chunks.

    Supports both TransformerReplayBuffer (10-element) and
    DistilledReplayBuffer (9-element) payload formats.  The obs dimension
    is auto-detected: 39-dim observations are compressed to 21-dim.

    In addition to (states, actions, rewards), this function also extracts
    ``policy_mu`` — the teacher's intended action at each state.  The index
    depends on the buffer format:

    * TransformerReplayBuffer (10 elements): policy_mu at index 7
    * DistilledReplayBuffer  ( 9 elements): policy_mu at index 6

    Returns ``(states, actions, rewards, policy_mu)`` arrays of shape
    ``(num_episodes, 400, dim)``, or ``None`` if the directory is empty.
    """
    if not os.path.isdir(buffer_path):
        return None

    pt_files = sorted(
        [f for f in os.listdir(buffer_path) if f.endswith(".pt")],
        key=lambda x: int(x.split("_")[0]),
    )
    if not pt_files:
        return None

    all_obs, all_actions, all_rewards, all_mu = [], [], [], []
    for fname in pt_files:
        path = os.path.join(buffer_path, fname)
        try:
            # weights_only=False is required because the replay buffers
            # save lists of numpy arrays / tensors, not pure state dicts.
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  ⚠️  Skipping {path}: {e}")
            continue

        # Validate payload structure:
        #   TransformerReplayBuffer → 10 elements
        #   DistilledReplayBuffer   →  9 elements
        # In both formats: index 0 = env_obses, 2 = actions, 3 = rewards.
        if not isinstance(payload, list) or len(payload) not in (9, 10):
            print(f"  ⚠️  Skipping {path}: unexpected payload format "
                  f"(expected 9 or 10 elements, got "
                  f"{len(payload) if isinstance(payload, list) else type(payload).__name__})")
            continue

        env_obses = np.asarray(payload[0])   # (N, obs_dim)  — 21 or 39
        actions = np.asarray(payload[2])     # (N, 4)
        rewards = np.asarray(payload[3])     # (N, 1)

        # policy_mu: teacher's intended action at each state
        #   TransformerReplayBuffer (10 elements): index 7
        #   DistilledReplayBuffer   (9 elements): index 6
        mu_idx = 7 if len(payload) == 10 else 6
        policy_mu = np.asarray(payload[mu_idx])  # (N, 4)

        all_obs.append(env_obses)
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_mu.append(policy_mu)

    if not all_obs:
        return None

    obs = np.concatenate(all_obs, axis=0)
    act = np.concatenate(all_actions, axis=0)
    rew = np.concatenate(all_rewards, axis=0)
    mu = np.concatenate(all_mu, axis=0)

    # Auto-detect obs dimension
    obs_dim = obs.shape[-1]

    # Trim to a multiple of EPISODE_LEN
    usable = (obs.shape[0] // EPISODE_LEN) * EPISODE_LEN
    if usable == 0:
        print(f"  ⚠️  Buffer {buffer_path} has < {EPISODE_LEN} transitions, skipping.")
        return None

    obs = obs[:usable].reshape(-1, EPISODE_LEN, obs_dim)
    act = act[:usable].reshape(-1, EPISODE_LEN, ACTION_DIM)
    rew = rew[:usable].reshape(-1, EPISODE_LEN, 1)
    mu = mu[:usable].reshape(-1, EPISODE_LEN, ACTION_DIM)

    # Compress observations if they are raw 39-dim
    if obs_dim == OBS_RAW_DIM:
        obs = _compress_states(obs)  # (num_ep, 400, 21)
        print(f"    Compressed observations: {obs_dim}-dim → {OBS_COMPRESSED_DIM}-dim")
    elif obs_dim == OBS_COMPRESSED_DIM:
        print(f"    Observations already compressed: {obs_dim}-dim")
    else:
        print(f"    ⚠️  Unexpected obs dimension: {obs_dim} (expected 21 or 39)")

    return obs, act, rew, mu


# ── Main ────────────────────────────────────────────────────────────────────

def generate(args):
    buffer_root = args.buffer_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"📂  Buffer root : {buffer_root}")
    print(f"📂  Output dir  : {output_dir}")

    all_states, all_actions, all_rewards, all_mu = [], [], [], []

    # The buffer_root may either:
    #   (a) Directly contain .pt files (single buffer), or
    #   (b) Contain sub-directories, each being a separate buffer.
    # We handle both cases.

    subdirs = sorted([
        d for d in os.listdir(buffer_root)
        if os.path.isdir(os.path.join(buffer_root, d))
    ]) if os.path.isdir(buffer_root) else []

    if subdirs:
        # Case (b): iterate sub-directories
        for sub in subdirs:
            full_path = os.path.join(buffer_root, sub)
            result = load_single_buffer_dir(full_path)
            if result is not None:
                s, a, r, m = result
                all_states.append(s)
                all_actions.append(a)
                all_rewards.append(r)
                all_mu.append(m)
                print(f"  ✅  {sub}: {s.shape[0]} episodes")
    else:
        # Case (a): buffer_root itself contains .pt files
        result = load_single_buffer_dir(buffer_root)
        if result is not None:
            s, a, r, m = result
            all_states.append(s)
            all_actions.append(a)
            all_rewards.append(r)
            all_mu.append(m)
            print(f"  ✅  {buffer_root}: {s.shape[0]} episodes")

    if not all_states:
        print("❌  No data loaded. Check --buffer_dir.")
        sys.exit(1)

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)
    policy_mu = np.concatenate(all_mu, axis=0)
    print(f"\n  Total episodes: {states.shape[0]}")
    print(f"  states    : {states.shape}")
    print(f"  actions   : {actions.shape}")
    print(f"  rewards   : {rewards.shape}")
    print(f"  policy_mu : {policy_mu.shape}")

    # ── Save in chunk format expected by DiffusionPolicyDataset ──────────
    chunk_size = args.chunk_size
    num_chunks = (states.shape[0] + chunk_size - 1) // chunk_size

    for ci in range(num_chunks):
        start = ci * chunk_size
        end = min((ci + 1) * chunk_size, states.shape[0])
        chunk_path = os.path.join(output_dir, f"data_chunk_{ci}")
        torch.save({
            "states": list(states[start:end]),
            "actions": list(actions[start:end]),
            "rewards": list(rewards[start:end]),
            "policy_mu": list(policy_mu[start:end]),
        }, chunk_path)
        print(f"  💾  Saved {chunk_path}  ({end - start} episodes)")
        gc.collect()

    print(f"\n✅  Done — {num_chunks} chunk(s) written to {output_dir}")


def main():
    p = argparse.ArgumentParser(
        description="Generate diffusion-policy dataset from raw experience buffers."
    )
    p.add_argument(
        "--buffer_dir", type=str, required=True,
        help="Root directory of raw buffer files. Can be a single buffer dir "
             "containing .pt files, or a parent dir with sub-directories "
             "(e.g. logs/experiment_test/buffer/collective_buffer/train).",
    )
    p.add_argument(
        "--output_dir", type=str, default="diffusion_policy/data/train",
        help="Where to write the data_chunk_* files.",
    )
    p.add_argument(
        "--chunk_size", type=int, default=2500,
        help="Number of episodes per output chunk.",
    )
    args = p.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
