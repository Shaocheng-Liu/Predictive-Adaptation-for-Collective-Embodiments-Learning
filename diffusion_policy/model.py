"""Conditional 1D U-Net for diffusion-based action prediction.

The network predicts the noise added to an action sequence, conditioned on a
history-state sliding window (states: [B, T, 21], actions: [B, T-1, 4],
and optionally rewards: [B, T-1, 1]).

Feature toggles (controlled via CLI flags):
    --use_reward    : Include reward signals in conditioning (default: off)
    --encoder_type  : Observation encoder type: 'mlp' (default) or 'conv1d'
    --use_film      : FiLM conditioning in U-Net residual blocks (default: on)
    --dropout       : Dropout rate for regularization (default: 0.0)
"""

import copy
import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding used in DDPM."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditionalResidualBlock1D(nn.Module):
    """1-D residual block conditioned on a global feature vector.

    Matches the architecture from Chi et al., 2023 (Diffusion Policy):
    - Additive (use_film=False): ``out = out + proj(cond)``
    - FiLM (use_film=True): ``out = scale * out + bias``
    """

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int,
                 kernel_size: int = 5, n_groups: int = 8,
                 use_film: bool = True, dropout: float = 0.0):
        super().__init__()
        self.use_film = use_film
        self.out_channels = out_channels
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(n_groups, out_channels),
                nn.Mish(),
            ),
            nn.Sequential(
                nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(n_groups, out_channels),
                nn.Mish(),
            ),
        ])
        cond_channels = out_channels * 2 if use_film else out_channels
        # Reference: Mish activation before linear projection
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0]
            bias = embed[:, 1]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = self.dropout(out)
        return out + self.residual_conv(x)


class LegacyConditionalResidualBlock1D(nn.Module):
    """Legacy (v1) residual block — for loading pre-alignment checkpoints.

    Matches the exact architecture at commit 578b178:
    - ``cond_proj`` (plain ``nn.Linear``, no Mish prefix)
    - kernel_size=3 (hardcoded), n_groups=8 (hardcoded)
    - FiLM formula: ``out = out * (1 + scale) + shift``

    State-dict key names (``cond_proj.weight``) differ from v2
    (``cond_encoder.1.weight``).
    """

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int,
                 kernel_size: int = 3, n_groups: int = 8,
                 use_film: bool = True, dropout: float = 0.0):
        super().__init__()
        self.use_film = use_film
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(n_groups, out_channels),
                nn.Mish(),
            ),
            nn.Sequential(
                nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(n_groups, out_channels),
                nn.Mish(),
            ),
        ])
        # v1 name: cond_proj (NOT cond_encoder) — must match checkpoint keys
        if use_film:
            self.cond_proj = nn.Linear(cond_dim, out_channels * 2)
        else:
            self.cond_proj = nn.Linear(cond_dim, out_channels)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        if self.use_film:
            cond_out = self.cond_proj(cond)
            scale, shift = cond_out.chunk(2, dim=-1)
            out = out * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        else:
            out = out + self.cond_proj(cond).unsqueeze(-1)
        out = self.blocks[1](out)
        out = self.dropout(out)
        return out + self.residual_conv(x)


class LegacyConditionalUNet1D(nn.Module):
    """Legacy (v1) conditional 1-D U-Net — for loading pre-alignment checkpoints.

    Matches the exact architecture at commit 578b178:
    - Uses ``LegacyConditionalResidualBlock1D`` (``cond_proj`` Linear,
      v1 FiLM formula ``(1+scale)*out + shift``)
    - ``time_mlp`` (v1 name, NOT ``diffusion_step_encoder``)
    - 1 residual block per encoder/decoder stage (not 2)
    - 1 mid block (not 2)
    - Downsample on ALL encoder stages (not non-last only)
    - Upsample before skip-concat (not after)
    - 2× timestep MLP expansion (not 4×)
    - 1×1 final conv only (no extra Conv1dBlock)

    This class exists solely for backward compatibility — new training
    should use ``model_version="v2"`` (the default).
    """

    def __init__(
        self,
        action_dim: int = 4,
        cond_dim: int = 128,
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple = (128, 256, 512),
        kernel_size: int = 3,
        n_groups: int = 8,
        use_film: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        dsed = diffusion_step_embed_dim
        # v1 name: time_mlp (must match checkpoint state_dict keys)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 2),
            nn.Mish(),
            nn.Linear(dsed * 2, dsed),
        )
        global_cond_dim = dsed + cond_dim

        in_ch = action_dim
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        for out_ch in down_dims:
            self.down_blocks.append(LegacyConditionalResidualBlock1D(
                in_ch, out_ch, global_cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                use_film=use_film, dropout=dropout))
            self.downsamplers.append(nn.Conv1d(out_ch, out_ch, 3, 2, 1))
            in_ch = out_ch

        self.mid_block = LegacyConditionalResidualBlock1D(
            in_ch, in_ch, global_cond_dim,
            kernel_size=kernel_size, n_groups=n_groups,
            use_film=use_film, dropout=dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for out_ch in reversed(down_dims):
            self.upsamplers.append(nn.ConvTranspose1d(in_ch, in_ch, 4, 2, 1))
            self.up_blocks.append(LegacyConditionalResidualBlock1D(
                in_ch + out_ch, out_ch, global_cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                use_film=use_film, dropout=dropout))
            in_ch = out_ch

        self.final_conv = nn.Conv1d(in_ch, action_dim, 1)

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_mlp(timestep.flatten())
        global_feat = torch.cat([t_emb, global_cond], dim=-1)

        skips = []
        x = noisy_action
        for block, downsample in zip(self.down_blocks, self.downsamplers):
            x = block(x, global_feat)
            skips.append(x)
            x = downsample(x)

        x = self.mid_block(x, global_feat)

        for block, upsample in zip(self.up_blocks, self.upsamplers):
            x = upsample(x)
            skip = skips.pop()
            if x.shape[2] != skip.shape[2]:
                x = nn.functional.interpolate(x, size=skip.shape[2], mode='nearest')
            x = torch.cat([x, skip], dim=1)
            x = block(x, global_feat)

        return self.final_conv(x)


class ConditionalUNet1D(nn.Module):
    """Conditional 1-D U-Net matching Chi et al. 2023 (Diffusion Policy).

    Architecture faithful to the reference implementation (v2):
    - 2 ConditionalResidualBlock1D per encoder/decoder stage
    - 2 mid blocks
    - Downsample (stride-2 Conv1d) on non-last encoder stages only
    - Upsample (stride-2 ConvTranspose1d) on non-last decoder stages only
    - kernel_size=5, n_groups=8 (reference defaults)
    - 4× expansion in timestep MLP
    - Extra Conv1dBlock before final 1×1 projection

    Parameters
    ----------
    action_dim : int
        Dimensionality of each action vector (default 4).
    cond_dim : int
        Dimensionality of the global condition vector.
    diffusion_step_embed_dim : int
        Embedding dimension for the diffusion timestep.
    down_dims : tuple of int
        Channel widths for U-Net encoder stages.
    kernel_size : int
        Conv kernel size (5 in reference).
    n_groups : int
        GroupNorm groups (8 in reference).
    """

    def __init__(
        self,
        action_dim: int = 4,
        cond_dim: int = 256,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        use_film: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        dsed = diffusion_step_embed_dim
        # Timestep embedding: 4× expansion (matching reference)
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        global_cond_dim = dsed + cond_dim

        all_dims = [action_dim] + list(down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        start_dim = down_dims[0]

        # Encoder: 2 residual blocks + downsample per stage
        # (downsample only on non-last stages, matching reference)
        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, global_cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    use_film=use_film, dropout=dropout),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, global_cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    use_film=use_film, dropout=dropout),
                nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity()
            ]))

        # Mid: 2 residual blocks (matching reference)
        mid_dim = down_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, global_cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                use_film=use_film, dropout=dropout),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, global_cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                use_film=use_film, dropout=dropout),
        ])

        # Decoder: 2 residual blocks + upsample per stage
        # (iterates over reversed(in_out[1:]), matching reference)
        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, global_cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    use_film=use_film, dropout=dropout),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, global_cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    use_film=use_film, dropout=dropout),
                nn.ConvTranspose1d(dim_in, dim_in, 4, 2, 1) if not is_last else nn.Identity()
            ]))

        # Final conv: Conv1dBlock + 1×1 projection (matching reference)
        self.final_conv = nn.Sequential(
            nn.Conv1d(start_dim, start_dim, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, start_dim),
            nn.Mish(),
            nn.Conv1d(start_dim, action_dim, 1),
        )

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        noisy_action : (B, action_dim, T_action)
        timestep     : (B,) or (B, 1)  diffusion step index
        global_cond  : (B, cond_dim)   observation-derived condition

        Returns
        -------
        noise_pred : (B, action_dim, T_action)
        """
        global_feature = self.diffusion_step_encoder(timestep.flatten())
        global_feature = torch.cat([global_feature, global_cond], dim=-1)

        # Encoder — skip connections stored after resnet2, before downsample
        h = []
        x = noisy_action
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # Mid
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Decoder — concat skip, apply 2 res blocks, upsample
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat([x, h.pop()], dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        return self.final_conv(x)


# ---------------------------------------------------------------------------
# Observation encoders
# ---------------------------------------------------------------------------

class MLPObservationEncoder(nn.Module):
    """MLP-based observation encoder (standard for diffusion policy).

    Flattens the history window of states and actions into a single 1D
    vector, then processes through an MLP with Mish activations.
    """

    def __init__(
        self,
        state_dim: int = 21,
        action_dim: int = 4,
        obs_horizon: int = 20,
        output_dim: int = 128,
        hidden_dim: int = 256,
        use_reward: bool = False,
        reward_dim: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_reward = use_reward
        flat_dim = obs_horizon * state_dim + (obs_horizon - 1) * action_dim
        if use_reward:
            flat_dim += (obs_horizon - 1) * reward_dim
        layers = [nn.Linear(flat_dim, hidden_dim), nn.Mish()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Mish()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, states, actions, rewards=None):
        B = states.shape[0]
        parts = [states.reshape(B, -1), actions.reshape(B, -1)]
        if self.use_reward and rewards is not None:
            parts.append(rewards.reshape(B, -1))
        return self.net(torch.cat(parts, dim=-1))


class Conv1DObservationEncoder(nn.Module):
    """1-D convolutional observation encoder (legacy architecture).

    Concatenates states, actions (and optionally rewards) along the feature
    axis per timestep, processes with a small 1-D conv stack, then
    average-pools into a single vector.
    """

    def __init__(
        self,
        state_dim: int = 21,
        action_dim: int = 4,
        hidden_dim: int = 128,
        output_dim: int = 128,
        use_reward: bool = False,
        reward_dim: int = 1,
    ):
        super().__init__()
        self.use_reward = use_reward
        in_dim = state_dim + action_dim
        if use_reward:
            in_dim += reward_dim
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.Mish(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.Mish(),
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, states, actions, rewards=None):
        B, T, _ = states.shape
        act_pad = torch.zeros(B, 1, actions.shape[-1], device=states.device)
        actions_full = torch.cat([actions, act_pad], dim=1)
        parts = [states, actions_full]
        if self.use_reward and rewards is not None:
            rew_pad = torch.zeros(B, 1, rewards.shape[-1], device=states.device)
            rewards_full = torch.cat([rewards, rew_pad], dim=1)
            parts.append(rewards_full)
        x = torch.cat(parts, dim=-1).permute(0, 2, 1)
        x = self.net(x)
        x = x.mean(dim=-1)
        return self.out_proj(x)


# ---------------------------------------------------------------------------
# Full Diffusion Policy wrapper (advanced — U-Net)
# ---------------------------------------------------------------------------

class DiffusionPolicy(nn.Module):
    """Combines the observation encoder and the conditional U-Net.

    Feature toggles:
        model_version : 'v2' (default, reference-aligned) or 'v1' (legacy)
        encoder_type  : 'mlp' (default) or 'conv1d'
        use_reward    : whether to include reward signals (default False)
        use_film      : FiLM conditioning in residual blocks (default True)
        dropout       : dropout rate (default 0.0)
    """

    def __init__(
        self,
        obs_state_dim: int = 21,
        obs_action_dim: int = 4,
        obs_reward_dim: int = 1,
        action_dim: int = 4,
        pred_horizon: int = 8,
        cond_dim: int = 256,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        obs_horizon: int = 20,
        encoder_type: str = "mlp",
        use_reward: bool = False,
        use_film: bool = True,
        dropout: float = 0.0,
        model_version: str = "v2",
    ):
        super().__init__()
        self.use_reward = use_reward

        if encoder_type == "mlp":
            self.obs_encoder = MLPObservationEncoder(
                state_dim=obs_state_dim,
                action_dim=obs_action_dim,
                obs_horizon=obs_horizon,
                output_dim=cond_dim,
                hidden_dim=cond_dim * 2,
                use_reward=use_reward,
                reward_dim=obs_reward_dim,
                dropout=dropout,
            )
        else:
            self.obs_encoder = Conv1DObservationEncoder(
                state_dim=obs_state_dim,
                action_dim=obs_action_dim,
                hidden_dim=cond_dim,
                output_dim=cond_dim,
                use_reward=use_reward,
                reward_dim=obs_reward_dim,
            )

        unet_cls = LegacyConditionalUNet1D if model_version == "v1" else ConditionalUNet1D
        self.noise_pred_net = unet_cls(
            action_dim=action_dim,
            cond_dim=cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            use_film=use_film,
            dropout=dropout,
        )
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor = None,
    ) -> torch.Tensor:
        cond = self.obs_encoder(states, actions, rewards)
        return self.noise_pred_net(noisy_action, timestep, cond)


# ---------------------------------------------------------------------------
# Vanilla (MLP-based) noise predictor — minimal baseline
# ---------------------------------------------------------------------------

class MLPNoisePredictor(nn.Module):
    """A simple MLP that predicts noise, conditioned on timestep + obs.

    The noisy action sequence is flattened, concatenated with the global
    condition vector and the timestep embedding, then passed through a
    small feedforward network.

    Parameters
    ----------
    action_dim : int
        Dimensionality of each action vector (default 4).
    pred_horizon : int
        Length of the predicted action chunk.
    cond_dim : int
        Dimensionality of the condition vector from the obs encoder.
    diffusion_step_embed_dim : int
        Embedding dimension for the diffusion timestep.
    hidden_dim : int
        Width of hidden layers.
    n_layers : int
        Number of hidden layers.
    """

    def __init__(
        self,
        action_dim: int = 4,
        pred_horizon: int = 4,
        cond_dim: int = 64,
        diffusion_step_embed_dim: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim),
            nn.Mish(),
        )

        input_dim = action_dim * pred_horizon + diffusion_step_embed_dim + cond_dim
        layers = []
        in_d = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_d, hidden_dim), nn.Mish()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, action_dim * pred_horizon))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        noisy_action : (B, action_dim, pred_horizon)
        timestep     : (B,)
        global_cond  : (B, cond_dim)

        Returns
        -------
        noise_pred : (B, action_dim, pred_horizon)
        """
        B = noisy_action.shape[0]
        flat_action = noisy_action.reshape(B, -1)          # (B, action_dim * pred_horizon)
        t_emb = self.time_mlp(timestep.flatten())           # (B, embed_dim)
        x = torch.cat([flat_action, t_emb, global_cond], dim=-1)
        out = self.net(x)                                    # (B, action_dim * pred_horizon)
        return out.reshape(B, self.action_dim, self.pred_horizon)


class VanillaObservationEncoder(nn.Module):
    """Minimal MLP encoder for the history sliding-window.

    Flattens the concatenated (states, actions, and optionally rewards) and
    passes them through a small MLP — no convolutions.
    """

    def __init__(
        self,
        state_dim: int = 21,
        action_dim: int = 4,
        obs_horizon: int = 20,
        output_dim: int = 64,
        hidden_dim: int = 128,
        use_reward: bool = False,
        reward_dim: int = 1,
    ):
        super().__init__()
        self.use_reward = use_reward
        flat_dim = obs_horizon * state_dim + (obs_horizon - 1) * action_dim
        if use_reward:
            flat_dim += (obs_horizon - 1) * reward_dim
        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor = None,
    ) -> torch.Tensor:
        B = states.shape[0]
        parts = [states.reshape(B, -1), actions.reshape(B, -1)]
        if self.use_reward and rewards is not None:
            parts.append(rewards.reshape(B, -1))
        return self.net(torch.cat(parts, dim=-1))


class VanillaDiffusionPolicy(nn.Module):
    """Minimal MLP-based diffusion policy — vanilla fallback baseline.

    Uses ``VanillaObservationEncoder`` + ``MLPNoisePredictor`` instead of
    the heavier conv-based encoder and 1-D U-Net used in ``DiffusionPolicy``.
    """

    def __init__(
        self,
        obs_state_dim: int = 21,
        obs_action_dim: int = 4,
        obs_reward_dim: int = 1,
        obs_horizon: int = 20,
        action_dim: int = 4,
        pred_horizon: int = 4,
        cond_dim: int = 64,
        diffusion_step_embed_dim: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 3,
        use_reward: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_reward = use_reward
        self.obs_encoder = VanillaObservationEncoder(
            state_dim=obs_state_dim,
            action_dim=obs_action_dim,
            obs_horizon=obs_horizon,
            output_dim=cond_dim,
            hidden_dim=hidden_dim,
            use_reward=use_reward,
            reward_dim=obs_reward_dim,
        )
        self.noise_pred_net = MLPNoisePredictor(
            action_dim=action_dim,
            pred_horizon=pred_horizon,
            cond_dim=cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor = None,
    ) -> torch.Tensor:
        cond = self.obs_encoder(states, actions, rewards)
        return self.noise_pred_net(noisy_action, timestep, cond)


# ---------------------------------------------------------------------------
# EMA wrapper
# ---------------------------------------------------------------------------

class EMAModel:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of the model that is updated with a running
    average of the training weights.  The EMA model is used for validation
    and checkpoint saving (standard practice in diffusion models).
    """

    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def eval(self):
        return self.shadow.eval()

    def __call__(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)
