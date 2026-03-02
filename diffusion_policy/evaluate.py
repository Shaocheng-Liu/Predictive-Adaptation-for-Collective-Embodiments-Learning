"""Evaluation script for the diffusion policy baseline (GPU-accelerated).

Mirrors the evaluation protocol of the original two-stage framework:
    * 400-step episodes in MetaWorld
    * Same compressed observation (state[:18] ++ state[36:39] → 21-dim)
    * 20-step history sliding window as condition (GPU tensors + torch.roll)
    * Records per-task success rate
    * All diffusion schedule constants pre-computed on GPU once
    * Supports DDPM (full chain) and DDIM (fast deterministic) samplers
    * Supports receding horizon (execute k < pred_horizon actions, re-plan)

Usage
-----
    # Fast evaluation with DDIM (default — 10x faster than DDPM)
    python diffusion_policy/evaluate.py \
        --checkpoint diffusion_policy/checkpoints \
        --arch_type advanced \
        --robot_type sawyer \
        --tasks reach-v2 push-v2

    # With receding horizon (execute 4 of 8 predicted actions, then re-plan)
    python diffusion_policy/evaluate.py \
        --checkpoint diffusion_policy/checkpoints \
        --arch_type advanced \
        --robot_type sawyer \
        --action_horizon 4 \
        --tasks reach-v2

    # Legacy DDPM sampling (slower, matches training exactly)
    python diffusion_policy/evaluate.py \
        --checkpoint diffusion_policy/checkpoints \
        --sampler ddpm \
        --robot_type sawyer \
        --tasks reach-v2
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

# Ensure project root is on the path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from model import DiffusionPolicy, VanillaDiffusionPolicy


# ============================================================================
# DDPM noise schedule helpers
# ============================================================================

def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def build_diffusion_constants(betas: torch.Tensor, device: torch.device):
    """Pre-compute all diffusion schedule constants on GPU once.

    Returns a dict of GPU tensors used by ddpm_sample() and ddim_sample().
    """
    betas = betas.to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]])

    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_recip_alphas": 1.0 / torch.sqrt(alphas),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "posterior_variance": betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
    }


@torch.no_grad()
def ddpm_sample(model, obs_states, obs_actions, obs_rewards, dc, device):
    """Run the full reverse diffusion chain to produce action predictions.

    All computation stays on GPU. Only the final result is moved to CPU.

    Parameters
    ----------
    model       : trained DiffusionPolicy (on device)
    obs_states  : (1, obs_horizon, 21)   GPU tensor
    obs_actions : (1, obs_horizon-1, 4)  GPU tensor
    obs_rewards : (1, obs_horizon-1, 1)  GPU tensor
    dc          : dict of pre-computed diffusion constants (GPU)
    device      : torch.device

    Returns
    -------
    actions : (pred_horizon, action_dim)  GPU tensor
    """
    T = dc["betas"].shape[0]
    pred_horizon = model.pred_horizon
    action_dim = model.action_dim
    x = torch.randn(1, action_dim, pred_horizon, device=device)

    betas = dc["betas"]
    sqrt_recip_alphas = dc["sqrt_recip_alphas"]
    sqrt_omc = dc["sqrt_one_minus_alphas_cumprod"]
    post_var = dc["posterior_variance"]

    for i in reversed(range(T)):
        t_batch = torch.full((1,), i, device=device, dtype=torch.long)
        noise_pred = model(x, t_batch, obs_states, obs_actions, obs_rewards)
        coeff = betas[i] / sqrt_omc[i]
        x = sqrt_recip_alphas[i] * (x - coeff * noise_pred)
        if i > 0:
            x = x + torch.sqrt(post_var[i]) * torch.randn_like(x)

    # x: (1, action_dim, pred_horizon) → (pred_horizon, action_dim)
    return x.squeeze(0).permute(1, 0)


@torch.no_grad()
def ddim_sample(model, obs_states, obs_actions, obs_rewards, dc, device,
                ddim_steps: int = 10):
    """Run DDIM (deterministic, eta=0) reverse diffusion for fast inference.

    Uses a subset of ``ddim_steps`` evenly-spaced timesteps from the full
    schedule, yielding ~T/ddim_steps speedup with similar quality.

    Parameters
    ----------
    model       : trained DiffusionPolicy (on device)
    obs_states  : (1, obs_horizon, 21)   GPU tensor
    obs_actions : (1, obs_horizon-1, 4)  GPU tensor
    obs_rewards : (1, obs_horizon-1, 1)  GPU tensor or None
    dc          : dict of pre-computed diffusion constants (GPU)
    device      : torch.device
    ddim_steps  : int — number of denoising steps (default 10)

    Returns
    -------
    actions : (pred_horizon, action_dim)  GPU tensor
    """
    T = dc["betas"].shape[0]
    pred_horizon = model.pred_horizon
    action_dim = model.action_dim
    x = torch.randn(1, action_dim, pred_horizon, device=device)

    # Evenly-spaced timestep sub-sequence (descending)
    timesteps = torch.linspace(T - 1, 0, ddim_steps + 1, device=device).long()

    sqrt_ac = dc["sqrt_alphas_cumprod"]
    sqrt_omc = dc["sqrt_one_minus_alphas_cumprod"]

    for i in range(len(timesteps) - 1):
        t_cur = timesteps[i].item()
        t_next = timesteps[i + 1].item()

        t_batch = torch.full((1,), t_cur, device=device, dtype=torch.long)
        noise_pred = model(x, t_batch, obs_states, obs_actions, obs_rewards)

        # Predict x_0
        x0_pred = (x - sqrt_omc[t_cur] * noise_pred) / sqrt_ac[t_cur]

        # DDIM deterministic step → x_{t_next}
        x = sqrt_ac[t_next] * x0_pred + sqrt_omc[t_next] * noise_pred

    # x: (1, action_dim, pred_horizon) → (pred_horizon, action_dim)
    return x.squeeze(0).permute(1, 0)


# ============================================================================
# Helpers — environment construction
# ============================================================================

def _compress_obs_torch(obs_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """Compress raw 39-dim obs to 21-dim and return as a GPU tensor."""
    compressed = np.concatenate([obs_np[:18], obs_np[36:39]], axis=-1)
    return torch.as_tensor(compressed, dtype=torch.float32, device=device)


def _make_env(task_name: str, robot_type: str, seed: int = 42):
    """Create a single MetaWorld environment for the given task and robot."""
    from Metaworld.metaworld.envs.robot_utils import set_robot_type, patch_all_metaworld_envs
    set_robot_type(robot_type)
    patch_all_metaworld_envs()

    sys.path.insert(0, os.path.join(PROJECT_ROOT, "Metaworld"))
    import metaworld

    mt1 = metaworld.MT1(task_name, seed=seed)
    env_cls = mt1.train_classes[task_name]
    env = env_cls()
    env.set_task(mt1.train_tasks[0])
    return env, mt1.train_tasks


# ============================================================================
# GPU-accelerated evaluation loop — mirrors evaluate_transformer()
# ============================================================================

def evaluate_task(
    model,
    task_name: str,
    robot_type: str,
    dc: dict,
    device: torch.device,
    num_episodes: int = 100,
    max_episode_steps: int = 400,
    obs_horizon: int = 20,
    sampler: str = "ddim",
    ddim_steps: int = 10,
    action_horizon: int = None,
):
    """Evaluate the diffusion policy on a single task (GPU-accelerated).

    Sliding-window buffers (states, actions, rewards) are GPU tensors.
    Uses ``torch.roll()`` for window updates — identical to the original
    ``evaluate_transformer()`` in ``collective_metaworld.py``.

    Parameters
    ----------
    sampler        : "ddpm" or "ddim" — inference sampler (default: ddim)
    ddim_steps     : int — number of DDIM denoising steps (default: 10)
    action_horizon : int or None — how many actions to execute per chunk
                     before re-planning (receding horizon).  None or 0
                     means execute all ``pred_horizon`` actions.

    Returns
    -------
    success_rate : float  in [0, 1]
    """
    env, all_tasks = _make_env(task_name, robot_type)

    pred_horizon = model.pred_horizon
    exec_horizon = action_horizon if action_horizon and action_horizon > 0 else pred_horizon
    use_reward = getattr(model, 'use_reward', True)
    successes = []
    total_policy_time = 0.0
    total_env_time = 0.0
    total_steps = 0

    for ep_idx in range(num_episodes):
        ep_start_time = time.time()
        task = all_tasks[ep_idx % len(all_tasks)]
        env.set_task(task)
        raw_obs, _ = env.reset()
        first_state = _compress_obs_torch(raw_obs, device)

        # GPU sliding-window buffers — pre-fill states with first obs
        ctx_states = first_state.unsqueeze(0).expand(obs_horizon, -1).clone()
        ctx_actions = torch.zeros(obs_horizon - 1, 4, device=device)
        ctx_rewards = torch.zeros(obs_horizon - 1, 1, device=device) if use_reward else None

        ep_success = 0.0
        ep_reward = 0.0
        action_queue = []

        for step_i in range(max_episode_steps):
            if len(action_queue) == 0:
                t0 = time.time()
                s_t = ctx_states.unsqueeze(0)
                a_t = ctx_actions.unsqueeze(0)
                r_t = ctx_rewards.unsqueeze(0) if ctx_rewards is not None else None

                if sampler == "ddim":
                    pred = ddim_sample(model, s_t, a_t, r_t, dc, device,
                                       ddim_steps=ddim_steps)
                else:
                    pred = ddpm_sample(model, s_t, a_t, r_t, dc, device)

                # Receding horizon: only keep first exec_horizon actions
                all_actions = list(pred.cpu().numpy())
                action_queue = all_actions[:exec_horizon]
                total_policy_time += time.time() - t0

            action = np.clip(action_queue.pop(0), -1.0, 1.0)

            t1 = time.time()
            raw_obs, reward, terminated, truncated, info = env.step(action)
            total_env_time += time.time() - t1

            new_state = _compress_obs_torch(raw_obs, device)
            action_t = torch.as_tensor(action, dtype=torch.float32, device=device)

            # Roll sliding window (GPU) — same as evaluate_transformer()
            ctx_states = torch.roll(ctx_states, shifts=-1, dims=0)
            ctx_states[-1] = new_state
            ctx_actions = torch.roll(ctx_actions, shifts=-1, dims=0)
            ctx_actions[-1] = action_t
            if ctx_rewards is not None:
                reward_t = torch.tensor([reward], dtype=torch.float32, device=device)
                ctx_rewards = torch.roll(ctx_rewards, shifts=-1, dims=0)
                ctx_rewards[-1] = reward_t

            ep_success += info.get("success", 0.0)
            ep_reward += reward
            total_steps += 1

        ep_time = time.time() - ep_start_time
        ep_success_binary = float(ep_success > 0)
        successes.append(ep_success_binary)
        running_sr = float(np.mean(successes)) * 100

        print(f"    [{task_name}] ep {ep_idx + 1}/{num_episodes}  "
              f"success={'YES' if ep_success_binary else 'NO ':>3s}  "
              f"reward={ep_reward:7.2f}  "
              f"time={ep_time:.1f}s  "
              f"running_SR={running_sr:.1f}%", flush=True)

    env.close()

    if total_steps > 0:
        avg_policy_ms = (total_policy_time / total_steps) * 1000
        avg_env_ms = (total_env_time / total_steps) * 1000
        print(f"    [Perf] policy={avg_policy_ms:.2f}ms/step  env={avg_env_ms:.2f}ms/step")

    return float(np.mean(successes))


# ============================================================================
# Entry point
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Diffusion Policy Evaluation")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint root dir (e.g. diffusion_policy/checkpoints) "
                        "or a single .pt file when used with --single_checkpoint")
    p.add_argument("--robot_type", type=str, default="sawyer",
                   help="Robot embodiment (sawyer, ur5e, ur10e, panda, xarm7, gen3, ...)")
    p.add_argument("--tasks", type=str, nargs="+",
                   default=["reach-v2", "push-v2", "pick-place-v2", "door-open-v2",
                            "drawer-open-v2", "button-press-topdown-v2",
                            "peg-insert-side-v2", "window-open-v2", "window-close-v2",
                            "faucet-open-v2"],
                   help="MetaWorld task names to evaluate")
    p.add_argument("--num_episodes", type=int, default=100,
                   help="Number of evaluation episodes per task")
    p.add_argument("--max_episode_steps", type=int, default=400,
                   help="Maximum steps per episode")
    p.add_argument("--obs_horizon", type=int, default=20,
                   help="History sliding-window length")
    p.add_argument("--pred_horizon", type=int, default=8,
                   help="Predicted action chunk length")

    # Model architecture (must match training — auto-loaded from hparams.json
    # when available, CLI values serve as fallback defaults)
    p.add_argument("--arch_type", type=str, default="advanced",
                   choices=["vanilla", "advanced"],
                   help="Architecture type: must match the training config")
    p.add_argument("--model_version", type=str, default="v2",
                   choices=["v1", "v2"],
                   help="U-Net architecture version: 'v2' (reference-aligned, default) "
                        "or 'v1' (legacy). Auto-detected from hparams.json.")
    p.add_argument("--encoder_type", type=str, default="mlp",
                   choices=["mlp", "conv1d"],
                   help="Observation encoder type")
    p.add_argument("--use_reward", action="store_true", default=False,
                   help="Include reward signals in conditioning")
    p.add_argument("--use_film", action="store_true", default=True,
                   help="Use FiLM conditioning in U-Net")
    p.add_argument("--no_film", dest="use_film", action="store_false",
                   help="Disable FiLM conditioning")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Dropout rate (should match training)")
    p.add_argument("--obs_state_dim", type=int, default=21)
    p.add_argument("--obs_action_dim", type=int, default=4)
    p.add_argument("--cond_dim", type=int, default=256)
    p.add_argument("--diffusion_step_embed_dim", type=int, default=256)
    p.add_argument("--down_dims", type=int, nargs="+", default=[256, 512, 1024])
    p.add_argument("--kernel_size", type=int, default=5)
    p.add_argument("--n_groups", type=int, default=8)
    p.add_argument("--vanilla_hidden_dim", type=int, default=256)
    p.add_argument("--vanilla_n_layers", type=int, default=3)

    # Diffusion
    p.add_argument("--num_diffusion_steps", type=int, default=100)
    p.add_argument("--noise_schedule", type=str, default="cosine",
                   choices=["cosine", "linear"])

    # Inference sampler
    p.add_argument("--sampler", type=str, default="ddim",
                   choices=["ddpm", "ddim"],
                   help="Inference sampler: 'ddpm' (full reverse chain) or "
                        "'ddim' (fast deterministic, default)")
    p.add_argument("--ddim_steps", type=int, default=10,
                   help="Number of DDIM denoising steps (default 10, "
                        "only used when --sampler ddim)")
    p.add_argument("--action_horizon", type=int, default=None,
                   help="Receding horizon: execute only the first k actions "
                        "from each predicted chunk before re-planning.  "
                        "None (default) = execute all pred_horizon actions.  "
                        "Recommended: 4 or pred_horizon//2.")

    # Multi-seed evaluation
    p.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5],
                   help="Seeds to evaluate. When set, --checkpoint is treated as the "
                        "save_dir root (e.g. diffusion_policy/checkpoints) and the "
                        "script looks for <arch_type>_seed_<s>/<model_filename> for each seed.")
    p.add_argument("--model_filename", type=str, default="best_model.pt",
                   help="Which checkpoint file to load from each seed directory "
                        "(e.g. best_model.pt, model_epoch_40.pt, final_model.pt)")
    p.add_argument("--single_checkpoint", action="store_true",
                   help="If set, treat --checkpoint as a single .pt file instead of "
                        "iterating over seeds.")

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--result_dir", type=str, default="diffusion_policy/results",
                   help="Directory to save evaluation results")
    return p.parse_args()


def _override_args_from_hparams(args, hparams_path: str):
    """Load hparams.json and override architecture-related args.

    This ensures the model architecture at eval time exactly matches what
    was used during training, even if the user doesn't re-specify every
    hyperparameter on the CLI.
    """
    if not os.path.isfile(hparams_path):
        return args

    with open(hparams_path, "r") as f:
        hp = json.load(f)

    # Keys that define the model architecture — override from saved hparams
    arch_keys = [
        "arch_type", "model_version", "obs_state_dim", "obs_action_dim", "cond_dim",
        "diffusion_step_embed_dim", "down_dims", "kernel_size", "n_groups",
        "vanilla_hidden_dim", "vanilla_n_layers", "obs_horizon", "pred_horizon",
        "num_diffusion_steps", "noise_schedule",
        "encoder_type", "use_reward", "use_film", "dropout",
    ]
    overridden = []

    # Legacy inference for old hparams.json without new feature flags.
    # Defaults here must match commit 578b178 (the v1 training code).
    if "model_version" not in hp:
        setattr(args, "model_version", "v1")
        overridden.append("model_version: (legacy inferred → v1)")
    if "use_reward" not in hp:
        setattr(args, "use_reward", False)
        overridden.append("use_reward: (legacy inferred → False)")
    if "encoder_type" not in hp:
        setattr(args, "encoder_type", "mlp")
        overridden.append("encoder_type: (legacy inferred → mlp)")
    if "use_film" not in hp:
        setattr(args, "use_film", True)
        overridden.append("use_film: (legacy inferred → True)")
    if "dropout" not in hp:
        setattr(args, "dropout", 0.0)
    if "kernel_size" not in hp:
        setattr(args, "kernel_size", 3)
        overridden.append("kernel_size: (legacy inferred → 3)")
    if "n_groups" not in hp:
        setattr(args, "n_groups", 8)

    for key in arch_keys:
        if key in hp:
            old_val = getattr(args, key, None)
            new_val = hp[key]
            if old_val != new_val:
                setattr(args, key, new_val)
                overridden.append(f"{key}: {old_val} → {new_val}")

    if overridden:
        print(f"  ℹ️  Auto-loaded hparams from {hparams_path}:")
        for msg in overridden:
            print(f"      {msg}")

    return args


def _build_model(args, device: torch.device):
    """Instantiate the correct architecture based on --arch_type."""
    if args.arch_type == "advanced":
        return DiffusionPolicy(
            obs_state_dim=args.obs_state_dim,
            obs_action_dim=args.obs_action_dim,
            obs_reward_dim=1,
            action_dim=args.obs_action_dim,
            pred_horizon=args.pred_horizon,
            cond_dim=args.cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=tuple(args.down_dims),
            kernel_size=args.kernel_size,
            n_groups=args.n_groups,
            obs_horizon=args.obs_horizon,
            encoder_type=args.encoder_type,
            use_reward=args.use_reward,
            use_film=args.use_film,
            dropout=args.dropout,
            model_version=args.model_version,
        ).to(device)
    else:
        return VanillaDiffusionPolicy(
            obs_state_dim=args.obs_state_dim,
            obs_action_dim=args.obs_action_dim,
            obs_reward_dim=1,
            obs_horizon=args.obs_horizon,
            action_dim=args.obs_action_dim,
            pred_horizon=args.pred_horizon,
            cond_dim=args.cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            hidden_dim=args.vanilla_hidden_dim,
            n_layers=args.vanilla_n_layers,
            use_reward=args.use_reward,
            dropout=args.dropout,
        ).to(device)


def _load_model(args, ckpt_path: str, device: torch.device):
    """Load a model checkpoint, auto-detecting hparams from the same directory."""
    # Try to load hparams.json from the checkpoint directory
    ckpt_dir = os.path.dirname(ckpt_path)
    hparams_path = os.path.join(ckpt_dir, "hparams.json")
    args = _override_args_from_hparams(args, hparams_path)

    model = _build_model(args, device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.result_dir, exist_ok=True)

    # Determine checkpoints to evaluate
    if args.single_checkpoint:
        ckpt_list = [("single", args.checkpoint)]
    else:
        ckpt_list = []
        for s in args.seeds:
            path = os.path.join(
                args.checkpoint, f"{args.arch_type}_seed_{s}", args.model_filename
            )
            if os.path.isfile(path):
                ckpt_list.append((str(s), path))
            else:
                print(f"  ⚠️  Checkpoint not found for seed {s}: {path}")

    if not ckpt_list:
        print("No checkpoints found. Exiting.")
        return

    all_results = {}
    for seed_label, ckpt_path in ckpt_list:
        print(f"\n{'='*60}")
        print(f"  Evaluating seed={seed_label}  |  robot={args.robot_type}")
        print(f"  Checkpoint: {ckpt_path}")
        exec_h = args.action_horizon if args.action_horizon else "all (pred_horizon)"
        print(f"  Sampler: {args.sampler}"
              f"{f' ({args.ddim_steps} steps)' if args.sampler == 'ddim' else ''}"
              f"  |  action_horizon={exec_h}")
        print(f"{'='*60}", flush=True)

        model = _load_model(args, ckpt_path, device)

        # Build diffusion schedule AFTER hparams override (num_diffusion_steps
        # may have changed)
        if args.noise_schedule == "cosine":
            betas = cosine_beta_schedule(args.num_diffusion_steps)
        else:
            betas = linear_beta_schedule(args.num_diffusion_steps)
        dc = build_diffusion_constants(betas, device)

        seed_results = {}

        for task in args.tasks:
            print(f"\n  ▶ Starting task: {task}", flush=True)
            sr = evaluate_task(
                model=model,
                task_name=task,
                robot_type=args.robot_type,
                dc=dc,
                device=device,
                num_episodes=args.num_episodes,
                max_episode_steps=args.max_episode_steps,
                obs_horizon=args.obs_horizon,
                sampler=args.sampler,
                ddim_steps=args.ddim_steps,
                action_horizon=args.action_horizon,
            )
            seed_results[task] = sr
            print(f"  ✅ {task:<30s}  success_rate = {sr*100:.1f}%", flush=True)

        avg = float(np.mean(list(seed_results.values())))
        seed_results["__average__"] = avg
        print(f"\n  {'AVERAGE':<30s}  success_rate = {avg*100:.1f}%")
        all_results[f"seed_{seed_label}"] = seed_results

    # Aggregate across seeds
    if len(ckpt_list) > 1:
        print(f"\n{'='*60}")
        print("  Aggregated results across seeds")
        print(f"{'='*60}")
        for task in args.tasks:
            rates = [all_results[k][task] for k in all_results if task in all_results[k]]
            mean = float(np.mean(rates))
            std = float(np.std(rates))
            print(f"  {task:<30s}  {mean*100:.1f}% ± {std*100:.1f}%")

    # Save results
    result_file = os.path.join(
        args.result_dir, f"eval_{args.robot_type}.json"
    )
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {result_file}")


if __name__ == "__main__":
    main()
