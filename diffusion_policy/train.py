"""Training script for the diffusion policy baseline."""

import argparse
import os
import sys
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(__file__))

from model import DiffusionPolicy, VanillaDiffusionPolicy, EMAModel
from dataset import DiffusionPolicyDataset


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def build_model(args):
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
        )
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
        )


def train_one_seed(args, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Training diffusion policy  |  seed={seed}  |  device={device}")
    print(f"{'='*60}")

    dataset = DiffusionPolicyDataset(
        data_path=args.data_path,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        max_episode_len=args.max_episode_len,
    )

    # 2. Validation data: use separate dataset if --val_path is given,
    #    otherwise fall back to random split from the training data.
    val_dataset = None
    if args.val_path:
        val_dataset = DiffusionPolicyDataset(
            data_path=args.val_path,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            max_episode_len=args.max_episode_len,
        )
        train_indices = torch.randperm(len(dataset), device=device)
        val_indices = torch.randperm(len(val_dataset), device=device)
        print(f"  Train samples: {len(dataset)} (from {args.data_path})")
        print(f"  Val   samples: {len(val_dataset)} (from {args.val_path})")
    else:
        num_total = len(dataset)
        indices = torch.randperm(num_total, device=device)
        val_split_size = int(num_total * args.val_ratio)
        train_indices = indices[val_split_size:]
        val_indices = indices[:val_split_size]
        print(f"  Train samples: {train_indices.shape[0]}  |  Val samples: {val_indices.shape[0]} (random split, val_ratio={args.val_ratio})")

    train_size = train_indices.shape[0]
    val_size = val_indices.shape[0]

    # Log training target info
    has_mu = dataset.policy_mu is not None
    if args.train_target == "mu" and has_mu:
        print(f"  Training target: policy_mu (teacher's intended action)")
    elif args.train_target == "mu" and not has_mu:
        print(f"  ⚠️  --train_target=mu but policy_mu not found in data (may be old dataset format); falling back to actions")
    else:
        print(f"  Training target: actions (actually-executed action)")

    # 3. 初始化模型、优化器、TensorBoard
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=tuple(args.adam_betas), weight_decay=args.weight_decay)

    # Learning rate schedule: step-based linear warmup + cosine annealing
    # (matching Chi et al. 2023 — scheduler steps every optimizer step)
    steps_per_epoch = (train_size + args.batch_size - 1) // args.batch_size
    total_training_steps = steps_per_epoch * args.epochs
    warmup_steps = args.lr_warmup_steps

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if warmup_steps > 0:
        print(f"  LR warmup: {warmup_steps} steps → cosine annealing ({total_training_steps} total steps)")

    # EMA (standard for diffusion models)
    ema = None
    if args.use_ema:
        ema = EMAModel(model, decay=args.ema_decay)
        print(f"  EMA enabled (decay={args.ema_decay})")
    
    save_dir = os.path.join(args.save_dir, f"{args.arch_type}_seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb_logs"))

    num_diffusion_steps = args.num_diffusion_steps
    if args.noise_schedule == "cosine":
        betas = cosine_beta_schedule(num_diffusion_steps)
    else:
        betas = linear_beta_schedule(num_diffusion_steps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    best_val_loss = float("inf")
    log_records = []

    def get_batch_from_indices(indices, ds):
        B = indices.shape[0]
        v_max = ds.valid_start_max
        ep_idx = indices // v_max
        start = indices % v_max
        
        obs_steps = torch.arange(ds.obs_horizon, device=device)
        obs_idx_map = start.unsqueeze(1) + obs_steps
        
        pred_steps = torch.arange(ds.pred_horizon, device=device)
        pred_idx_map = (start + ds.obs_horizon).unsqueeze(1) + pred_steps
        
        obs_s = ds.states[ep_idx.unsqueeze(1), obs_idx_map]
        obs_a = ds.actions[ep_idx.unsqueeze(1), obs_idx_map[:, :-1]]

        # Select training target: policy_mu (teacher's intended action) or
        # actions (actually-executed action).  policy_mu is cleaner and
        # matches what B2/B3 train on.
        if args.train_target == "mu" and ds.policy_mu is not None:
            target_a = ds.policy_mu[ep_idx.unsqueeze(1), pred_idx_map]
        else:
            target_a = ds.actions[ep_idx.unsqueeze(1), pred_idx_map]

        obs_r = None
        if args.use_reward:
            obs_r = ds.rewards[ep_idx.unsqueeze(1), obs_idx_map[:, :-1]]
        
        return obs_s, obs_a, obs_r, target_a

    # 5. 训练循环
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_start = time.time()
        
        # 每个 epoch 开始前随机打乱训练索引
        perm = torch.randperm(train_size, device=device)
        curr_train_indices = train_indices[perm]

        for i in range(0, train_size, args.batch_size):
            batch_idx = curr_train_indices[i : i + args.batch_size]
            
            obs_s, obs_a, obs_r, target_a = get_batch_from_indices(batch_idx, dataset)

            # Observation noise augmentation (reduces overfitting from
            # highly-overlapping sliding windows)
            if args.obs_noise_std > 0:
                obs_s = obs_s + args.obs_noise_std * torch.randn_like(obs_s)

            # --- 计算逻辑 ---
            B = target_a.shape[0]
            t = torch.randint(0, num_diffusion_steps, (B,), device=device)
            noise = torch.randn_like(target_a)

            sqrt_ac = sqrt_alphas_cumprod[t].view(B, 1, 1)
            sqrt_omac = sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)
            noisy_action = sqrt_ac * target_a + sqrt_omac * noise

            noise_pred = model(noisy_action.permute(0, 2, 1), t, obs_s, obs_a, obs_r)
            loss = nn.functional.mse_loss(noise_pred.permute(0, 2, 1), noise)
            
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()  # step-based (per optimizer step, matching reference)

            # Update EMA weights
            if ema is not None:
                ema.update(model)

            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / n_batches

        # Validation — use EMA model if available
        eval_model = ema.shadow if ema is not None else model
        eval_model.eval()
        val_ds = val_dataset if val_dataset is not None else dataset
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for i in range(0, val_size, args.batch_size):
                v_batch_idx = val_indices[i : i + args.batch_size]
                obs_s, obs_a, obs_r, target_a = get_batch_from_indices(v_batch_idx, val_ds)
                
                B = target_a.shape[0]
                t = torch.randint(0, num_diffusion_steps, (B,), device=device)
                noise = torch.randn_like(target_a)
                noisy_action = sqrt_alphas_cumprod[t].view(B, 1, 1) * target_a + \
                               sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1) * noise
                
                noise_pred = eval_model(noisy_action.permute(0, 2, 1), t, obs_s, obs_a, obs_r)
                val_loss += nn.functional.mse_loss(noise_pred.permute(0, 2, 1), noise).item()
                val_batches += 1
        
        avg_val = val_loss / val_batches
        elapsed = time.time() - t_start
        
        # TensorBoard & Logging
        
        writer.add_scalar("Loss/Train", avg_train, epoch)
        writer.add_scalar("Loss/Val", avg_val, epoch)
        log_records.append({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val})

        # 1. 检查点保存逻辑
        # 保存验证集表现最好的模型
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_sd = ema.state_dict() if ema is not None else model.state_dict()
            torch.save(save_sd, os.path.join(save_dir, "best_model.pt"))
            print(f"  [Seed {seed}] New best model saved at epoch {epoch} (val_loss={avg_val:.6f})"
                  f"{' [EMA]' if ema is not None else ''}")

        # 每隔 save_freq 个 epoch 保存一个备份
        if epoch % args.save_freq == 0:
            ckpt_sd = ema.state_dict() if ema is not None else model.state_dict()
            torch.save(ckpt_sd, os.path.join(save_dir, f"model_epoch_{epoch}.pt"))

        # 2. 终端日志输出
        if epoch % args.log_freq == 0 or epoch == 1:
            print(f"  [Seed {seed}] Epoch {epoch:>3} | train={avg_train:.6f} | val={avg_val:.6f} | lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

    # 3. 训练结束后的最终保存
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pt"))
    if ema is not None:
        torch.save(ema.state_dict(), os.path.join(save_dir, "final_ema_model.pt"))
    
    # 保存训练曲线数据
    with open(os.path.join(save_dir, "train_log.json"), "w") as f:
        json.dump(log_records, f, indent=2)

    # 保存本次训练的超参数
    with open(os.path.join(save_dir, "hparams.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    writer.close()

    print(f"  [Seed {seed}] Training complete.  Best val loss: {best_val_loss:.6f}")
    print(f"  Models saved to {save_dir}/")


def parse_args():
    p = argparse.ArgumentParser(description="Diffusion Policy Training")
    p.add_argument("--data_path", type=str, required=True,
                    help="Path to directory with training data files")
    p.add_argument("--val_path", type=str, default=None,
                    help="Path to directory with validation data files. "
                         "When provided, validation is done on this separate dataset "
                         "instead of a random split from --data_path.")
    p.add_argument("--obs_horizon", type=int, default=20,
                    help="History sliding-window length")
    p.add_argument("--pred_horizon", type=int, default=8,
                    help="Number of future action steps to predict (8 or 16 recommended for U-Net)")
    p.add_argument("--max_episode_len", type=int, default=400,
                    help="Maximum episode length in the buffer data")
    p.add_argument("--obs_state_dim", type=int, default=21,
                    help="State dimension after compression")
    p.add_argument("--obs_action_dim", type=int, default=4,
                    help="Action dimension")
    p.add_argument("--arch_type", type=str, default="advanced",
                    choices=["vanilla", "advanced"],
                    help="Architecture type")
    p.add_argument("--model_version", type=str, default="v2",
                    choices=["v1", "v2"],
                    help="U-Net architecture version: 'v2' (reference-aligned, default) "
                         "or 'v1' (legacy, for backward compatibility)")
    p.add_argument("--encoder_type", type=str, default="mlp",
                    choices=["mlp", "conv1d"],
                    help="Observation encoder type (mlp=standard diffusion policy, conv1d=legacy)")
    p.add_argument("--use_reward", action="store_true", default=False,
                    help="Include reward signals in conditioning (off by default for standard IL)")
    p.add_argument("--use_film", action="store_true", default=True,
                    help="Use FiLM conditioning in U-Net residual blocks (standard practice)")
    p.add_argument("--no_film", dest="use_film", action="store_false",
                    help="Disable FiLM conditioning (use additive instead)")
    p.add_argument("--use_ema", action="store_true", default=True,
                    help="Use EMA model for validation and checkpoint saving (standard practice)")
    p.add_argument("--no_ema", dest="use_ema", action="store_false",
                    help="Disable EMA")
    p.add_argument("--ema_decay", type=float, default=0.995,
                    help="EMA decay rate")
    p.add_argument("--dropout", type=float, default=0.0,
                    help="Dropout rate for regularization (0.0 = no dropout)")
    p.add_argument("--obs_noise_std", type=float, default=0.0,
                    help="Gaussian noise std added to observation states during training "
                         "(data augmentation to combat overlapping-window overfitting, "
                         "try 0.01-0.05)")
    p.add_argument("--cond_dim", type=int, default=256,
                    help="Condition vector dimension (256 matches reference)")
    p.add_argument("--diffusion_step_embed_dim", type=int, default=256,
                    help="Diffusion timestep embedding dimension (256 matches reference)")
    p.add_argument("--down_dims", type=int, nargs="+", default=[256, 512, 1024],
                    help="Channel widths for U-Net ([256,512,1024] matches reference)")
    p.add_argument("--kernel_size", type=int, default=5,
                    help="Conv kernel size for U-Net residual blocks (5 matches reference)")
    p.add_argument("--n_groups", type=int, default=8,
                    help="GroupNorm groups (8 matches reference)")
    p.add_argument("--vanilla_hidden_dim", type=int, default=256,
                    help="Hidden layer width")
    p.add_argument("--vanilla_n_layers", type=int, default=3,
                    help="Number of hidden layers")
    p.add_argument("--num_diffusion_steps", type=int, default=100,
                    help="Number of diffusion steps")
    p.add_argument("--noise_schedule", type=str, default="cosine",
                    choices=["cosine", "linear"],
                    help="Noise schedule type")
    p.add_argument("--epochs", type=int, default=100,
                    help="Number of training epochs (Chi et al. 2023 uses 100)")
    p.add_argument("--batch_size", type=int, default=256,
                    help="Batch size (256 matches Chi et al. 2023; use 2048 for GPU speed with --lr 3e-4)")
    p.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate (1e-4 matches Chi et al. 2023)")
    p.add_argument("--lr_warmup_steps", type=int, default=500,
                    help="Linear LR warmup optimizer STEPS (500 matches Chi et al. 2023 "
                         "for CNN-based; 1000 for Transformer-based)")
    p.add_argument("--adam_betas", type=float, nargs=2, default=[0.95, 0.999],
                    help="AdamW beta parameters ([0.95, 0.999] matches Chi et al. 2023)")
    p.add_argument("--train_target", type=str, default="mu",
                    choices=["mu", "action"],
                    help="Training target: 'mu' = teacher's intended action (cleaner, "
                         "matches B2/B3 targets); 'action' = actually-executed action "
                         "(noisier, original diffusion policy approach)")
    p.add_argument("--weight_decay", type=float, default=1e-6,
                    help="Weight decay (1e-6 matches Chi et al. 2023)")
    p.add_argument("--grad_clip", type=float, default=1.0,
                    help="Gradient clipping norm")
    p.add_argument("--val_ratio", type=float, default=0.1,
                    help="Validation fraction")
    p.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5],
                    help="List of random seeds")
    p.add_argument("--save_dir", type=str, default="diffusion_policy/checkpoints",
                    help="Directory to save models and logs")
    p.add_argument("--log_freq", type=int, default=10,
                    help="Print log frequency")
    p.add_argument("--save_freq", type=int, default=20,
                    help="Save checkpoint frequency (epochs)")
    p.add_argument("--device", type=str, default="cuda:0",
                    help="Torch device")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("  Diffusion Policy Baseline — Training")
    print("=" * 60)
    print(f"  Architecture: {args.arch_type} ({args.model_version})  |  encoder={args.encoder_type}  |  FiLM={args.use_film}  |  EMA={args.use_ema}")
    print(f"  use_reward={args.use_reward}  |  dropout={args.dropout}  |  obs_noise={args.obs_noise_std}  |  train_target={args.train_target}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Train data : {args.data_path}")
    if args.val_path:
        print(f"  Val data   : {args.val_path}")
    else:
        print(f"  Val data   : random split (val_ratio={args.val_ratio})")
    print(f"  epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}  warmup_steps={args.lr_warmup_steps}  wd={args.weight_decay}")
    print(f"  adam_betas={args.adam_betas}  kernel_size={args.kernel_size}  down_dims={args.down_dims}")

    for seed in args.seeds:
        train_one_seed(args, seed)

    print("\n✅  All seeds finished.")


if __name__ == "__main__":
    main()