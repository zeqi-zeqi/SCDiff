from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ema_pytorch import EMA
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from models.train_utils import DiffusionModel

from .loss import get_loss


# -----------------------------------------------------------------------------
# Diffusion schedule helpers
# -----------------------------------------------------------------------------


def space_indices(num_steps: int, count: int) -> List[int]:
    """Return evenly spaced integer step indices."""
    assert count <= num_steps

    frac_stride = 1 if count <= 1 else (num_steps - 1) / (count - 1)
    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride
    return taken_steps


def extract(a: Tensor, t: Tensor, x_shape: Tuple[int, ...]) -> Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def unsqueeze_xdim(z: Tensor, xdim: Tuple[int, ...]) -> Tensor:
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


# -----------------------------------------------------------------------------
# Local surface geometry
# -----------------------------------------------------------------------------


def _pairwise_sq_dists(x: Tensor) -> Tensor:
    xx = (x**2).sum(dim=-1, keepdim=True)
    dist = xx + xx.transpose(1, 2) - 2.0 * torch.bmm(x, x.transpose(1, 2))
    return dist.clamp_min_(0.0)


def _knn_indices(x: Tensor, k: int) -> Tensor:
    B, N, _ = x.shape
    k = min(max(1, k), N - 1) if N > 1 else 1

    dist = _pairwise_sq_dists(x)
    eye = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
    dist = dist.masked_fill(eye, float("inf"))
    return dist.topk(k=k, dim=-1, largest=False).indices


def _gather_neighbors(x: Tensor, idx: Tensor) -> Tensor:
    B, N, C = x.shape
    k = idx.shape[-1]

    idx_expand = idx.unsqueeze(-1).expand(B, N, k, C)
    x_expand = x.unsqueeze(1).expand(B, N, N, C)
    return torch.gather(x_expand, 2, idx_expand)


def _estimate_normals_and_reliability(ref_xyz: Tensor, k: int = 16, eps: float = 1e-8) -> Tuple[Tensor, Tensor]:
    """Estimate local PCA normals and a planarity-based reliability score."""
    idx = _knn_indices(ref_xyz, k=k)
    nbrs = _gather_neighbors(ref_xyz, idx)

    center = nbrs.mean(dim=2, keepdim=True)
    nbrs_centered = nbrs - center
    cov = torch.matmul(nbrs_centered.transpose(-1, -2), nbrs_centered)
    cov = cov / max(nbrs_centered.shape[2], 1)

    evals, evecs = torch.linalg.eigh(cov)
    normals = evecs[..., 0]

    lam0 = evals[..., 0]
    lam1 = evals[..., 1]
    lam2 = evals[..., 2]

    reliability = ((lam1 - lam0) / (lam2 + eps)).clamp(min=0.0)
    rel_mean = reliability.mean(dim=1, keepdim=True).clamp_min(eps)
    reliability = (reliability / rel_mean).clamp(0.0, 3.0)
    return normals, reliability


def _smoothstep01(x: float) -> float:
    x = min(max(float(x), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _warmup_scale(current_step: Optional[int], start_step: int, warmup_steps: int) -> float:
    if current_step is None:
        return 1.0

    current_step = int(current_step)
    start_step = max(0, int(start_step))
    warmup_steps = max(0, int(warmup_steps))

    if current_step < start_step:
        return 0.0
    if warmup_steps <= 0:
        return 1.0
    return _smoothstep01((current_step - start_step) / float(warmup_steps))


# -----------------------------------------------------------------------------
# Bridge coefficients
# -----------------------------------------------------------------------------


def compute_gaussian_product_coef(sigma1: float, sigma2: float):
    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var


def make_beta_schedule(n_timestep: int = 1000, linear_start: float = 1e-4, linear_end: float = 2e-2):
    scale = 1000 / n_timestep
    linear_start *= scale
    linear_end *= scale
    betas = torch.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64) ** 2
    return betas.numpy()


class SCDiff(DiffusionModel):
    """Surface-constrained diffusion model for point-cloud denoising."""

    def __init__(self, cfg: Dict, model: torch.nn.Module):
        super().__init__()

        raw_device = cfg.gpu if cfg.gpu is not None else "cuda"
        self.device = torch.device(f"cuda:{raw_device}" if isinstance(raw_device, int) else raw_device)
        self.timesteps = cfg.diffusion.timesteps
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.ot_ode = cfg.diffusion.ot_ode
        self.cfg = cfg
        self.cond_x1 = cfg.diffusion.cond_x1 if "cond_x1" in cfg.diffusion else False
        self.add_x1_noise = cfg.diffusion.add_x1_noise if "add_x1_noise" in cfg.diffusion else False
        self.objective = cfg.diffusion.objective if "objective" in cfg.diffusion else "pred_noise"
        self.weight_loss = cfg.diffusion.weight_loss if "weight_loss" in cfg.diffusion else False
        self.symmetric = cfg.diffusion.symmetric if "symmetric" in cfg.diffusion else True
        self.loss_multiplier = cfg.diffusion.loss_multiplier if "loss_multiplier" in cfg.diffusion else 1.0
        snr_clip = cfg.diffusion.snr_clip if "snr_clip" in cfg.diffusion else False

        # Denoising backbone and EMA teacher.
        self.model = model.to(self.device)
        ema_decay = cfg.model.EMA.decay if "EMA" in cfg.model and "decay" in cfg.model.EMA else 0.999
        self.ema = EMA(self.model, beta=ema_decay) if cfg.model.ema else None

        # Bridge schedule buffers.
        betas = make_beta_schedule(
            n_timestep=cfg.diffusion.timesteps,
            linear_start=cfg.diffusion.beta_start,
            linear_end=cfg.diffusion.beta_end,
        )
        if self.symmetric:
            betas = np.concatenate(
                [
                    betas[: cfg.diffusion.timesteps // 2],
                    np.flip(betas[: cfg.diffusion.timesteps // 2]),
                ]
            )

        self.noise_levels = (
            torch.linspace(
                cfg.diffusion.t0,
                cfg.diffusion.T,
                cfg.diffusion.timesteps,
                dtype=torch.float32,
            ).to(self.device)
            * cfg.diffusion.timesteps
        )

        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas).to(self.device)
        self.std_fwd = to_torch(std_fwd).to(self.device)
        self.std_bwd = to_torch(std_bwd).to(self.device)
        self.std_sb = to_torch(std_sb).to(self.device)
        self.mu_x0 = to_torch(mu_x0).to(self.device)
        self.mu_x1 = to_torch(mu_x1).to(self.device)
        self.calculate_loss = get_loss(cfg.diffusion.get("loss_type", "mse"))

        # Optional SNR-balanced diffusion loss.
        alphas_cumprod = np.cumprod(1 - betas)
        snr = torch.from_numpy(alphas_cumprod / (1 - alphas_cumprod)).to(self.device)
        maybe_clipped_snr = snr.clone()
        if snr_clip:
            maybe_clipped_snr.clamp_(max=5.0)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32), persistent=False)
        if self.objective == "pred_noise":
            register_buffer("loss_weight", maybe_clipped_snr / snr)
        elif self.objective == "pred_x0":
            register_buffer("loss_weight", maybe_clipped_snr)

    def get_std_fwd(self, step: Tensor, xdim: Tuple[int, ...] = None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def compute_pred_x0_from_eps(self, step: Tensor, xt: Tensor, net_out: Tensor, clip_denoise: bool = False):
        std_fwd = self.get_std_fwd(step, xdim=xt.shape[1:])

        if std_fwd.ndim != net_out.ndim:
            std_fwd = std_fwd.squeeze(-1)

        pred_x0 = xt - std_fwd * net_out
        if clip_denoise:
            pred_x0.clamp_(-3.0, 3.0)
        return pred_x0

    def compute_gt(self, step: Tensor, x0: Tensor, xt: Tensor) -> Tensor:
        if self.objective == "pred_noise":
            std_fwd = self.get_std_fwd(step, xdim=x0.shape[1:])
            return ((xt - x0) / std_fwd).detach()
        if self.objective == "pred_x0":
            return x0.detach()
        raise NotImplementedError(self.objective)

    def q_sample(self, step: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        assert x0.shape == x1.shape
        _, *xdim = x0.shape

        mu_x0 = unsqueeze_xdim(self.mu_x0[step], xdim)
        mu_x1 = unsqueeze_xdim(self.mu_x1[step], xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

        xt = mu_x0 * x0 + mu_x1 * x1
        if not self.ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)
        return xt.detach()

    # Surface-guided training losses.

    def anisotropic_surface_loss(
        self,
        pred_x0: Tensor,
        ref_points: Tensor,
        steps: Tensor,
        *,
        k: int = 16,
        gate_ratio: float = 0.5,
        gate_mode: str = "low_noise",
        normal_weight: float = 1.0,
        tangent_weight: float = 0.05,
        robust_delta: float = 0.01,
        reliability_power: float = 1.0,
        detach_ref: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """Surface regularizer with stronger normal than tangent penalties."""
        device = pred_x0.device
        B = pred_x0.shape[0]

        if detach_ref:
            ref_points = ref_points.detach()

        pred_xyz = pred_x0.transpose(1, 2).contiguous()
        ref_xyz = ref_points.transpose(1, 2).contiguous()

        normals, reliability = _estimate_normals_and_reliability(ref_xyz, k=k)
        residual = pred_xyz - ref_xyz

        normal_scalar = (residual * normals).sum(dim=-1, keepdim=True)
        normal_vec = normal_scalar * normals
        tangent_vec = residual - normal_vec

        normal_energy = torch.sqrt((normal_vec**2).sum(dim=-1) + robust_delta**2) - robust_delta
        tangent_energy = torch.sqrt((tangent_vec**2).sum(dim=-1) + robust_delta**2) - robust_delta

        point_weight = reliability.clamp_min(1e-6).pow(reliability_power)
        per_point = point_weight * (normal_weight * normal_energy + tangent_weight * tangent_energy)

        per_sample = per_point.mean(dim=-1)
        raw = per_sample.mean()

        Tm1 = float(self.timesteps - 1)
        if gate_ratio <= 0:
            gate = torch.ones((B,), device=device, dtype=torch.float32)
        else:
            threshold = gate_ratio * Tm1
            if gate_mode == "low_noise":
                gate = (steps.float() <= threshold).float()
            elif gate_mode == "high_noise":
                gate = (steps.float() >= threshold).float()
            else:
                gate = torch.ones((B,), device=device, dtype=torch.float32)

        denom = gate.sum().clamp(min=1.0)
        gated = (per_sample * gate).sum() / denom
        return raw, gated

    def local_uniformity_loss(self, pred_x0: Tensor, k: int = 4) -> Tensor:
        """Discourage local point collapse during late denoising steps."""
        pred_xyz = pred_x0.transpose(1, 2).contiguous()
        _, N, _ = pred_xyz.shape
        if N < 2 or k <= 0:
            return pred_x0.new_zeros(())

        idx = _knn_indices(pred_xyz, k=min(k, N - 1))
        nbrs = _gather_neighbors(pred_xyz, idx)
        dists = torch.norm(nbrs - pred_xyz.unsqueeze(2), dim=-1).clamp_min(1e-6)
        local_scale = dists.mean(dim=-1, keepdim=True).detach().clamp_min(1e-6)
        penalty = torch.relu(local_scale - dists) ** 2 / (local_scale**2)
        return penalty.mean()

    def _curriculum_scale(self, current_step: Optional[int]) -> float:
        cur_cfg = self.cfg.diffusion.get("curriculum", {}) if "curriculum" in self.cfg.diffusion else {}
        if not bool(cur_cfg.get("enable", False)):
            return 1.0

        surface_start = int(cur_cfg.get("surface_start_step", 0))
        surface_warmup = int(cur_cfg.get("surface_warmup_steps", 0))

        return _warmup_scale(current_step, surface_start, surface_warmup)

    def _late_step_mask(self, steps: Tensor, cfg: Dict, default_ratio: float = 0.25) -> Tensor:
        if cfg.get("late_steps", None) is not None:
            late_t_max = min(self.timesteps - 1, max(0, int(cfg.get("late_steps"))))
        else:
            late_ratio = float(cfg.get("late_ratio", default_ratio))
            late_t_max = min(self.timesteps - 1, max(0, int((self.timesteps - 1) * late_ratio)))
        return steps <= late_t_max

    def _select_base_phys_ref(self, x1: Tensor, x_cond: Optional[Tensor]) -> Tensor:
        """Use conditional coordinates as the surface reference when available."""
        if x_cond is not None and x_cond.shape == x1.shape:
            return x_cond
        return x1

    # Dynamic reference surface.

    def _predict_x0_from_model_output(self, steps: Tensor, xt: Tensor, out: Tensor) -> Tensor:
        if self.objective == "pred_noise":
            return self.compute_pred_x0_from_eps(steps, xt, out, clip_denoise=False)
        if self.objective == "pred_x0":
            return out
        raise NotImplementedError(self.objective)

    @torch.no_grad()
    def _predict_ema_x0(self, xt: Tensor, steps: Tensor, noise_levels: Tensor, x_cond: Optional[Tensor]) -> Optional[Tensor]:
        if self.ema is None:
            return None

        was_training = self.ema.training
        self.ema.eval()
        out = self.ema(xt, noise_levels, x_cond=x_cond)
        if was_training:
            self.ema.train()
        return self._predict_x0_from_model_output(steps, xt, out).detach()

    def _refinement_timestep_gate(self, steps: Tensor, ratio: float, mode: str) -> Tensor:
        Tm1 = float(max(self.timesteps - 1, 1))
        step_norm = (steps.float() / Tm1).clamp(0.0, 1.0)
        ratio = min(max(float(ratio), 1e-6), 1.0)

        if mode == "low_noise":
            gate = (1.0 - step_norm / ratio).clamp(0.0, 1.0)
        elif mode == "high_noise":
            gate = ((step_norm - (1.0 - ratio)) / ratio).clamp(0.0, 1.0)
        else:
            gate = torch.ones_like(step_norm)
        return gate * gate * (3.0 - 2.0 * gate)

    def build_dynamic_phys_ref(
        self,
        *,
        base_ref: Tensor,
        pred_x0: Tensor,
        xt: Tensor,
        steps: Tensor,
        noise_levels: Tensor,
        x_cond_model: Optional[Tensor],
        cfg: Dict,
    ) -> Tuple[Tensor, Tensor]:
        """Blend the input surface with current/EMA restoration estimates."""
        ref_cfg = cfg.get("ref_update", {}) if "ref_update" in cfg else {}
        if not bool(ref_cfg.get("enable", False)):
            return base_ref, base_ref.new_zeros(())

        source = str(ref_cfg.get("source", "hybrid")).lower()
        current_x0 = pred_x0.detach()
        refined = current_x0

        ema_x0 = None
        if source in ("ema", "hybrid"):
            ema_x0 = self._predict_ema_x0(xt, steps, noise_levels, x_cond_model)

        if source == "ema":
            refined = ema_x0 if ema_x0 is not None else current_x0
        elif source == "hybrid" and ema_x0 is not None:
            ema_mix = min(max(float(ref_cfg.get("ema_mix", 0.5)), 0.0), 1.0)
            refined = (1.0 - ema_mix) * current_x0 + ema_mix * ema_x0

        if refined.shape != base_ref.shape:
            return base_ref, base_ref.new_zeros(())

        base_xyz = base_ref.transpose(1, 2).contiguous()
        ref_k = int(ref_cfg.get("reliability_k", cfg.get("k", 16)))
        _, reliability = _estimate_normals_and_reliability(base_xyz, k=ref_k)

        rel_scale = max(float(ref_cfg.get("reliability_scale", 1.0)), 1e-6)
        rel_power = float(ref_cfg.get("reliability_power", cfg.get("reliability_power", 1.0)))
        rel_gate = (reliability / rel_scale).clamp(0.0, 1.0).pow(rel_power)

        gate_ratio = float(ref_cfg.get("gate_ratio", cfg.get("gate_ratio", 0.5)))
        gate_mode = str(ref_cfg.get("gate_mode", cfg.get("gate_mode", "low_noise"))).lower()
        time_gate = self._refinement_timestep_gate(steps, gate_ratio, gate_mode).view(-1, 1)

        blend_min = min(max(float(ref_cfg.get("blend_min", 0.0)), 0.0), 1.0)
        blend_max = min(max(float(ref_cfg.get("blend_max", 0.5)), blend_min), 1.0)
        gamma = blend_min + (blend_max - blend_min) * time_gate * rel_gate
        gamma = gamma.unsqueeze(1)

        dynamic_ref = (1.0 - gamma) * base_ref + gamma * refined
        if bool(ref_cfg.get("detach", True)):
            dynamic_ref = dynamic_ref.detach()
        return dynamic_ref, gamma.mean().detach()

    # Reverse diffusion sampling.

    def p_posterior(self, nprev: int, n: int, x_n: Tensor, x0: Tensor) -> Tensor:
        """Calculate p(x_{t-1} | x_t, x_0)."""
        assert nprev < n, f"{nprev=}, {n=}"

        std_n = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)
        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not self.ot_ode and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)
        return xt_prev

    def sample_ddpm(
        self,
        steps: List[int],
        pred_x0_fn: callable,
        x1: Tensor,
        x_cond: Tensor = None,
        log_steps: List[int] = None,
        verbose: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """Run DDPM reverse sampling."""
        if self.add_x1_noise:
            x1 = x1 + torch.randn_like(x1)

        xt = x1.detach().to(self.device)
        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]
        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc="DDPM sampling", total=len(steps) - 1) if verbose else pair_steps

        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"
            pred_x0 = pred_x0_fn(xt, step, x1=x1, x_cond=x_cond)
            xt = self.p_posterior(prev_step, step, xt, pred_x0)

            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach())
                xs.append(xt.detach())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)

    @torch.no_grad()
    def ddpm_sampling(
        self,
        x1: Tensor,
        x_cond: Tensor = None,
        clip_denoise: bool = False,
        sampling_steps: int = None,
        log_count: int = 10,
        verbose: bool = True,
        use_ema: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Sample from the bridge using the configured DDPM schedule."""
        sampling_steps = sampling_steps or self.timesteps - 1
        assert 0 < sampling_steps < self.timesteps == len(self.betas)
        steps = space_indices(self.timesteps, sampling_steps + 1)

        log_count = min(len(steps) - 1, log_count)
        log_steps = [steps[i] for i in space_indices(len(steps) - 1, log_count)]
        assert log_steps[0] == 0

        if verbose:
            logger.info(f"[DDPM Sampling] T={self.timesteps}, {sampling_steps=}!")

        self.model.eval()

        def pred_x0_fn(xt, step, x1, x_cond=None):
            step = torch.full((xt.shape[0],), step, device=self.device, dtype=torch.long)
            noise_levels = self.noise_levels[step].detach()

            if self.cond_x1:
                x_cond = torch.cat([x1, x_cond], dim=1) if x_cond is not None else x1

            if use_ema and self.ema is not None:
                out = self.ema(xt, noise_levels, x_cond=x_cond)
            else:
                out = self.model(xt, noise_levels, x_cond=x_cond)

            if self.objective == "pred_noise":
                return self.compute_pred_x0_from_eps(step, xt, out, clip_denoise=clip_denoise)
            if self.objective == "pred_x0":
                return out
            raise NotImplementedError(self.objective)

        xs, pred_x0 = self.sample_ddpm(
            steps,
            pred_x0_fn,
            x1,
            x_cond=x_cond,
            log_steps=log_steps,
            verbose=verbose,
        )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        self.model.train()
        return xs, pred_x0

    @torch.no_grad()
    def sample(
        self,
        x_cond: Optional[Tensor] = None,
        x_start: Optional[Tensor] = None,
        clip: bool = False,
        use_ema: bool = False,
        verbose: bool = True,
        log_count: int = 10,
        steps: int = None,
    ) -> Dict:
        if self.cfg.diffusion.sampling_strategy != "DDPM":
            raise NotImplementedError(self.cfg.diffusion.sampling_strategy)

        xs, _ = self.ddpm_sampling(
            x1=x_start,
            x_cond=x_cond,
            clip_denoise=clip,
            sampling_steps=self.cfg.diffusion.sampling_timesteps if steps is None else steps,
            verbose=verbose,
            use_ema=use_ema,
            log_count=log_count,
        )
        return {
            "x_chain": xs,
            "x_pred": xs[:, 0, ...],
            "x_start": x_start,
        }

    def loss(self, pred: Tensor, gt: Tensor) -> Tensor:
        pred = pred.to(self.device)
        gt = gt.to(self.device)
        return self.calculate_loss(pred, gt).mean()

    # Training objective.

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        x_cond: Optional[Tensor] = None,
        current_step: Optional[int] = None,
    ) -> Tensor:
        steps = torch.randint(0, self.timesteps, (x0.shape[0],), device=self.device)

        if self.add_x1_noise:
            x1 = x1 + torch.randn_like(x1)

        xt = self.q_sample(steps, x0, x1)
        gt = self.compute_gt(steps, x0, xt)

        x_cond_phys = x_cond
        x_cond_model = x_cond
        if self.cond_x1:
            x_cond_model = torch.cat([x1, x_cond_model], dim=1) if x_cond_model is not None else x1

        noise_levels = self.noise_levels[steps].detach()
        pred = self.model(xt, noise_levels, x_cond=x_cond_model)

        base_loss = self.calculate_loss(pred, gt)
        if self.weight_loss:
            base_loss = base_loss * extract(self.loss_weight, steps, base_loss.shape)
        base_loss = base_loss.mean()

        pred_x0 = self._predict_x0_from_model_output(steps, xt, pred)

        diff_cfg = self.cfg.diffusion
        phys_cfg = diff_cfg.get("phys_consistency", {}) if "phys_consistency" in diff_cfg else {}

        phys_enable = bool(phys_cfg.get("enable", True))
        phys_weight = float(phys_cfg.get("weight", 0.0))
        phys_scale = self._curriculum_scale(current_step)

        phys_k = int(phys_cfg.get("k", 16))
        phys_gate_ratio = float(phys_cfg.get("gate_ratio", 0.5))
        phys_gate_mode = str(phys_cfg.get("gate_mode", "low_noise"))
        phys_normal_weight = float(phys_cfg.get("normal_weight", 1.0))
        phys_tangent_weight = float(phys_cfg.get("tangent_weight", 0.05))
        phys_robust_delta = float(phys_cfg.get("robust_delta", 0.01))
        phys_reliability_power = float(phys_cfg.get("reliability_power", 1.0))
        phys_detach_ref = bool(phys_cfg.get("detach_ref", True))

        self._log_t_mean = steps.float().mean().detach()

        phys_raw = torch.tensor(0.0, device=self.device)
        phys_loss = torch.tensor(0.0, device=self.device)
        uniformity_loss = torch.tensor(0.0, device=self.device)
        ref_gamma_mean = torch.tensor(0.0, device=self.device)

        if phys_enable and phys_weight > 0 and phys_scale > 0.0:
            base_phys_ref = self._select_base_phys_ref(x1=x1, x_cond=x_cond_phys)
            x_phys_ref, ref_gamma_mean = self.build_dynamic_phys_ref(
                base_ref=base_phys_ref,
                pred_x0=pred_x0,
                xt=xt,
                steps=steps,
                noise_levels=noise_levels,
                x_cond_model=x_cond_model,
                cfg=phys_cfg,
            )
            phys_raw, phys_gated = self.anisotropic_surface_loss(
                pred_x0=pred_x0,
                ref_points=x_phys_ref,
                steps=steps,
                k=phys_k,
                gate_ratio=phys_gate_ratio,
                gate_mode=phys_gate_mode,
                normal_weight=phys_normal_weight,
                tangent_weight=phys_tangent_weight,
                robust_delta=phys_robust_delta,
                reliability_power=phys_reliability_power,
                detach_ref=phys_detach_ref,
            )

            phys_loss_uncapped = phys_gated * phys_weight * base_loss.new_tensor(phys_scale)
            phys_cap = base_loss.detach() * float(phys_cfg.get("max_phys_ratio", 0.3))
            phys_loss = torch.minimum(phys_loss_uncapped, phys_cap)

        uniformity_weight = float(phys_cfg.get("uniformity_weight", 0.0))
        if phys_enable and uniformity_weight > 0.0 and phys_scale > 0.0:
            late_ratio = float(phys_cfg.get("uniformity_late_ratio", 0.25))
            late_mask = self._late_step_mask(steps, phys_cfg, default_ratio=late_ratio)
            if bool(late_mask.any().item()):
                uniformity_raw = self.local_uniformity_loss(
                    pred_x0=pred_x0[late_mask],
                    k=int(phys_cfg.get("uniformity_k", 4)),
                )
                uniformity_uncapped = uniformity_raw * uniformity_weight * base_loss.new_tensor(phys_scale)
                uniformity_cap = base_loss.detach() * float(phys_cfg.get("max_uniformity_ratio", 0.05))
                uniformity_loss = torch.minimum(uniformity_uncapped, uniformity_cap)

        phys_loss = phys_loss + uniformity_loss

        self._log_base_loss = base_loss.detach()
        self._log_phys_raw = phys_raw.detach()
        self._log_phys_weighted = phys_loss.detach()
        self._log_curriculum_phys_scale = base_loss.new_tensor(phys_scale).detach()
        self._log_ref_gamma_mean = ref_gamma_mean.detach()

        loss = base_loss + phys_loss
        return loss * self.loss_multiplier
