import torch
import torch.nn.functional as F

@torch.jit.script
def symlog(x: torch.Tensor) -> torch.Tensor:
    # sign(x) * log(1 + |x|)
    x = x.to(torch.float32)
    return torch.sign(x) * torch.log1p(torch.abs(x))

@torch.jit.script
def symexp(x: torch.Tensor) -> torch.Tensor:
    # sign(x) * (exp(|x|) - 1)
    x = x.to(torch.float32)
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

@torch.jit.script
def log_std(x: torch.Tensor, low: torch.Tensor, dif: torch.Tensor) -> torch.Tensor:
    # low + 0.5 * dif * (tanh(x) + 1)
    return low + 0.5 * dif * (torch.tanh(x) + 1.0)

@torch.jit.script
def gaussian_logprob(eps: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    # sum_i [ -0.5 eps_i^2 - log_std_i - 0.5*log(2*pi) ]
    residual = -0.5 * eps.pow(2) - log_std
    log_prob = residual - 0.9189385332046727  # 0.5*ln(2*pi)
    return log_prob.sum(-1, keepdim=True)

@torch.jit.script
def squash(mu: torch.Tensor, pi: torch.Tensor, log_pi: torch.Tensor):
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    # log det |Jacobian| = \sum_i log(1 - tanh(pi_i)^2)
    log_pi = log_pi - torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi

class DRegCfg:
    """Configuration for discrete regression (two-hot), equivalent to the num_bins/vmin/vmax/bin_size part of the TD-MPC2 config."""
    __slots__ = ("num_bins", "vmin", "vmax", "bin_size")
    def __init__(self, num_bins: int, vmin: float, vmax: float):
        self.num_bins = int(num_bins)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.bin_size = (self.vmax - self.vmin) / max(self.num_bins - 1, 1)

@torch.jit.script
def two_hot(x: torch.Tensor, num_bins: int, vmin: float, vmax: float, bin_size: float) -> torch.Tensor:
    """Map scalar regression target x (shape [B,1] or [B]) to a soft two-hot vector [B, num_bins]."""
    if num_bins == 0:
        return x
    if num_bins == 1:
        return symlog(x)
    x = torch.clamp(symlog(x).squeeze(-1), vmin, vmax)      # [B]
    f = (x - vmin) / bin_size                               # continuous position
    left = torch.floor(f).to(torch.long)                    # [B]
    frac = (f - left.to(f.dtype)).unsqueeze(-1)             # [B,1], right bin weight

    B = x.shape[0]
    oh = torch.zeros(B, num_bins, device=x.device, dtype=x.dtype)

    # left bin
    left_idx = left.clamp(0, num_bins - 1).unsqueeze(-1)    # [B,1]
    oh.scatter_(1, left_idx, 1.0 - frac)

    # right bin (no wrapping, clamp to the rightmost bin)
    right = (left + 1).clamp(max=num_bins - 1).unsqueeze(-1)
    oh.scatter_(1, right, frac)
    return oh

@torch.jit.script
def two_hot_inv(x: torch.Tensor, num_bins: int, vmin: float, vmax: float) -> torch.Tensor:
    """Convert a two-hot probability distribution back to a scalar (apply symexp after computing the value in log domain)."""
    if num_bins == 0:
        return x
    if num_bins == 1:
        return symexp(x)
    bins = torch.linspace(vmin, vmax, num_bins, device=x.device, dtype=x.dtype)
    x = F.softmax(x, dim=-1)
    val = torch.sum(x * bins, dim=-1, keepdim=True)
    return symexp(val)

def soft_ce(pred_logits: torch.Tensor, target_scalar: torch.Tensor, dreg: DRegCfg) -> torch.Tensor:
    """Soft cross-entropy for discrete regression (same as TD-MPC2)."""
    # pred_logits: [B, num_bins]
    # target_scalar: [B, 1] (real-valued)
    logp = F.log_softmax(pred_logits, dim=-1)
    tgt = two_hot(target_scalar, dreg.num_bins, dreg.vmin, dreg.vmax, dreg.bin_size)  # [B, num_bins]
    return -(tgt * logp).sum(-1, keepdim=True)  # [B,1]