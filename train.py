import os
import sys

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import wandb
from dataloaders.dataloader import get_dataloader, save_iter
from dataloaders.punet import get_alignment_clean
from metrics.emd_assignment import emd_module
from models.evaluation import evaluate
from models.model_loader import load_diffusion, load_optim_sched
from models.train_utils import get_data_batch, getGradNorm, set_seed, setup_output_subdirs, to_cuda
from utils.args import parse_args

import json
import subprocess
import datetime
from pathlib import Path
import traceback
def init_processes(rank: int | str, size: int, fn: callable, args: DictConfig) -> None:
    """Initialize the distributed environment.

    Args:
        rank (int): Rank of the current process.
        size (int): Total number of processes.
        fn (function): Function to run.
        args (DictConfig): Configuration.
    """
    torch.cuda.set_device(rank)
    args.local_rank = rank
    args.global_rank = rank
    args.global_size = size
    args.gpu = rank

    os.environ["MASTER_ADDR"] = args.master_address
    os.environ["MASTER_PORT"] = args.master_port
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=size)

    fn(args)

    dist.barrier()
    dist.destroy_process_group()

def _safe_mkdir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def save_run_artifacts(cfg: DictConfig, output_dir: str, argv: list[str]) -> None:
    """
    Save:
      - resolved config
      - raw command line
      - environment/package versions
      - git info (if available)
    """
    _safe_mkdir(output_dir)

    cmd_path = Path(output_dir) / "cmd.txt"
    with open(cmd_path, "w", encoding="utf-8") as f:
        f.write(" ".join(argv) + "\n")

    resolved_cfg_path = Path(output_dir) / "config_resolved.yaml"
    OmegaConf.save(cfg, resolved_cfg_path)

    env = {
        "timestamp": datetime.datetime.now().isoformat(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_count": torch.cuda.device_count(),
    }
    try:
        import numpy as np
        env["numpy"] = np.__version__
    except Exception:
        pass
    try:
        import wandb as _wandb
        env["wandb"] = _wandb.__version__
    except Exception:
        pass

    env_path = Path(output_dir) / "env.json"
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)

    git_info = {}
    try:
        git_info["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        git_info["branch"] = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
        git_info["status"] = subprocess.check_output(["git", "status", "--porcelain"]).decode()
    except Exception:
        git_info["error"] = "git info not available"

    git_path = Path(output_dir) / "git.json"
    with open(git_path, "w", encoding="utf-8") as f:
        json.dump(git_info, f, indent=2)


def wandb_is_ready(use_wandb: bool) -> bool:
    """True if wandb is enabled AND already initialized."""
    if not use_wandb:
        return False
    try:
        return wandb.run is not None
    except Exception:
        return False


def safe_wandb_log(data: dict, step: int, use_wandb: bool) -> None:
    """Never crash training/eval due to wandb logging."""
    if not wandb_is_ready(use_wandb):
        return
    try:
        wandb.log(data, step=step)
    except Exception:
        return

def train(cfg: DictConfig) -> None:
    is_main_process = cfg.local_rank == 0

    logger.remove()

    if is_main_process:
        _safe_mkdir(cfg.output_dir)

    if is_main_process:
        (outf_syn,) = setup_output_subdirs(cfg.output_dir, "output")
        cfg.outf_syn = outf_syn

        fmt = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                + "<level>{level: <8}</level> | "
                + "<level>{message}</level>"
        )

        logger.add(sys.stdout, level="INFO", format=fmt)

        enable_file = True
        log_filename = "train.log"
        if "logging" in cfg:
            enable_file = bool(cfg.logging.get("enable_file", True))
            log_filename = str(cfg.logging.get("filename", "train.log"))

        if enable_file:
            log_path = os.path.join(cfg.output_dir, log_filename)
            logger.add(log_path, level="INFO", format=fmt, enqueue=True, backtrace=True, diagnose=False)
            logger.info("Logging to file: {}", log_path)

        try:
            save_run_artifacts(cfg, cfg.output_dir, sys.argv)
            logger.info("Saved run artifacts under {}", cfg.output_dir)
        except Exception as e:
            logger.warning("Could not save run artifacts: {}", e)

    set_seed(cfg)
    torch.cuda.empty_cache()

    train_loader, val_loader, train_sampler, val_sampler = get_dataloader(cfg)

    model, ckpt = load_diffusion(cfg)

    optimizer, lr_scheduler = load_optim_sched(cfg, model, ckpt)
    logger.info("Training with config {}", cfg.config)

    if cfg.data.dataset == "PUNet":
        aligner = emd_module.emdModule()
        emd_align = get_alignment_clean(aligner)

        @torch.no_grad()
        def align_fn(noisy, clean):
            align_idxs = emd_align(noisy, clean).detach().long()
            align_idxs = align_idxs.unsqueeze(1).expand(-1, 3, -1)
            clean = torch.gather(clean, -1, align_idxs)
            return clean

    else:
        align_fn = None

    use_wandb = False
    wandb_mode = "online"
    wandb_project = getattr(cfg, "wandb_project", None)
    wandb_entity = getattr(cfg, "wandb_entity", None)
    wandb_name = None
    wandb_tags = None
    wandb_watch = False
    wandb_watch_freq = cfg.training.log_interval * 10

    if "wandb" in cfg:
        use_wandb = bool(cfg.wandb.get("enable", False))
        wandb_mode = str(cfg.wandb.get("mode", "online")).lower()
        wandb_project = cfg.wandb.get("project", wandb_project)
        wandb_entity = cfg.wandb.get("entity", wandb_entity)
        wandb_name = cfg.wandb.get("name", None) or None
        wandb_tags = cfg.wandb.get("tags", None)
        wandb_watch = bool(cfg.wandb.get("watch", False))
        wandb_watch_freq = int(cfg.wandb.get("watch_freq", wandb_watch_freq))
    if wandb_mode == "disabled":
        use_wandb = False

    if is_main_process and use_wandb:
        try:
            if wandb_mode in ("offline", "disabled"):
                os.environ["WANDB_MODE"] = wandb_mode

            wandb.login()
            wandb.init(
                project=wandb_project,
                config=OmegaConf.to_container(cfg, resolve=True),
                entity=wandb_entity,
                name=wandb_name,
                tags=wandb_tags,
            )

            if wandb_watch:
                try:
                    wandb.watch(model, log="all", log_freq=wandb_watch_freq)
                except Exception as e:
                    logger.warning("Could not watch model. Skipping.")
                    logger.warning(e)

            logger.info("WandB enabled (mode={}, project={})", wandb_mode, wandb_project)
        except Exception as e:
            logger.warning("WandB init failed; continuing without wandb. Error: {}", e)
            use_wandb = False

    ampscaler = torch.cuda.amp.GradScaler(enabled=cfg.training.amp)

    train_iter = save_iter(train_loader, train_sampler)
    torch.cuda.empty_cache()
    logger.info("Setup training and evaluation iterators.")

    for step in range(cfg.start_step, cfg.training.steps):
        optimizer.zero_grad()

        if cfg.distribution_type == "multi":
            train_sampler.set_epoch(step // len(train_loader))

        loss_accum = torch.tensor(0.0, dtype=torch.float32, device=cfg.local_rank)

        for accum_iter in range(cfg.training.accumulation_steps):
            next_batch = next(train_iter)
            next_batch = to_cuda(next_batch, cfg.local_rank)

            data = next_batch
            data_batch = get_data_batch(batch=data, cfg=cfg, align_fn=align_fn)
            x_gt = data_batch["x_gt"]
            x_cond = data_batch["x_cond"]
            x_start = data_batch["x_start"]

            loss = model(x_gt, x1=x_start, x_cond=x_cond, current_step=step)
            loss /= cfg.training.accumulation_steps
            loss_accum += loss.detach()

            ampscaler.scale(loss).backward()

        ampscaler.unscale_(optimizer)
        if cfg.training.grad_clip.enabled:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip.value)

        ampscaler.step(optimizer)
        ampscaler.update()
        lr_scheduler.step()

        if model.ema is not None:
            model.ema.update()

        if cfg.distribution_type == "multi":
            dist.all_reduce(loss_accum)

        if step % cfg.training.log_interval == 0 and is_main_process:
            loss_accum /= cfg.global_size
            loss_accum = loss_accum.item()
            netpNorm, netgradNorm = getGradNorm(model.model)

            base_loss = getattr(model, "_log_base_loss", None)
            phys_raw = getattr(model, "_log_phys_raw", None)
            phys_weighted = getattr(model, "_log_phys_weighted", None)
            phys_scale = getattr(model, "_log_curriculum_phys_scale", None)
            t_mean = getattr(model, "_log_t_mean", None)
            ref_gamma = getattr(model, "_log_ref_gamma_mean", None)

            base_loss = base_loss.item() if base_loss is not None else None
            phys_raw = phys_raw.item() if phys_raw is not None else None
            phys_weighted = phys_weighted.item() if phys_weighted is not None else None
            phys_scale = phys_scale.item() if phys_scale is not None else None
            t_mean = t_mean.item() if t_mean is not None else None
            ref_gamma = ref_gamma.item() if ref_gamma is not None else None

            if (base_loss is not None) and (base_loss > 1e-12):
                phys_ratio = (phys_weighted / base_loss) if (phys_weighted is not None) else 0.0
                aux_ratio = phys_ratio
            else:
                phys_ratio = 0.0
                aux_ratio = 0.0
            current_lr = float(optimizer.param_groups[0]["lr"])

            logger.info(
                "[{:>4d}/{:>4d}] "
                "loss_total={:.6e} | loss_base={:.6e} | "
                "loss_phys_raw={:.6e} | loss_phys_w={:.6e} | "
                "phys_scale={:.4f} | "
                "lr={:.6e} | t_mean={:.1f} | ref_gamma={:.4f} | "
                "netpNorm={:.2f} | netgradNorm={:.4f} | "
                "phys_ratio={:.4e} | aux_ratio={:.4e}",
                step,
                cfg.training.steps,
                loss_accum,
                base_loss if base_loss is not None else 0.0,
                phys_raw if phys_raw is not None else 0.0,
                phys_weighted if phys_weighted is not None else 0.0,
                phys_scale if phys_scale is not None else 1.0,
                current_lr,
                t_mean if t_mean is not None else -1.0,
                ref_gamma if ref_gamma is not None else 0.0,
                netpNorm,
                netgradNorm,
                phys_ratio,
                aux_ratio,
            )

            safe_wandb_log(
                {
                    "loss/total": loss_accum,
                    "loss/base": base_loss if base_loss is not None else 0.0,
                    "loss/phys_raw": phys_raw if phys_raw is not None else 0.0,
                    "loss/phys_weighted": phys_weighted if phys_weighted is not None else 0.0,
                    "ratio/phys_to_base": phys_ratio,
                    "ratio/aux_to_base": aux_ratio,
                    "curriculum/phys_scale": phys_scale if phys_scale is not None else 1.0,
                    "diffusion/t_mean": t_mean if t_mean is not None else -1.0,
                    "phys/ref_gamma_mean": ref_gamma if ref_gamma is not None else 0.0,
                    "netpNorm": netpNorm,
                    "netgradNorm": netgradNorm,
                    "lr": current_lr,
                },
                step=step,
                use_wandb=use_wandb,
            )

        if (step + 1) % cfg.training.save_interval == 0:
            if is_main_process:
                save_dict = {
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }
                torch.save(save_dict, "%s/step_%d.pth" % (cfg.output_dir, step + 1))
                logger.info("Saved checkpoint to {}", cfg.output_dir)

            if cfg.distribution_type == "multi":
                dist.barrier()
                map_location = {"cuda:%d" % 0: "cuda:%d" % cfg.local_rank}
                model.load_state_dict(
                    torch.load(
                        "%s/step_%d.pth" % (cfg.output_dir, step + 1),
                        map_location=map_location,
                    )["model_state"]
                )



        if (step + 1) % cfg.training.viz_interval == 0:
            if cfg.distribution_type == "multi":
                dist.barrier()

            model.eval()
            if is_main_process:
                prev_wandb_cfg = None
                try:
                    if "wandb" in cfg:
                        prev_wandb_cfg = OmegaConf.to_container(cfg.wandb, resolve=True)
                        cfg.wandb.enable = False
                except Exception:
                    pass

                try:
                    metrics = evaluate(model, val_loader, cfg, step + 1)

                    if isinstance(metrics, dict):
                        diff_cfg = cfg.get("diffusion", {})
                        phys_cfg = diff_cfg.get("phys_consistency", {}) if "phys_consistency" in diff_cfg else {}
                        ref_cfg = phys_cfg.get("ref_update", {}) if "ref_update" in phys_cfg else {}
                        cur_cfg = diff_cfg.get("curriculum", {}) if "curriculum" in diff_cfg else {}

                        cfg_snapshot = {
                            "diffusion/weight_loss": bool(diff_cfg.get("weight_loss", False)),
                            "diffusion/snr_clip": bool(diff_cfg.get("snr_clip", False)),
                            "phys/enable": bool(phys_cfg.get("enable", False)),
                            "phys/weight": float(phys_cfg.get("weight", 0.0)),
                            "phys/gate_ratio": float(phys_cfg.get("gate_ratio", 0.0)),
                            "phys/max_ratio": float(phys_cfg.get("max_phys_ratio", 0.0)),
                            "phys/ref_update_enable": bool(ref_cfg.get("enable", False)),
                            "phys/ref_blend_max": float(ref_cfg.get("blend_max", 0.0)),
                            "curriculum/enable": bool(cur_cfg.get("enable", False)),
                            "curriculum/surface_start_step": int(cur_cfg.get("surface_start_step", 0) or 0),
                            "curriculum/surface_warmup_steps": int(cur_cfg.get("surface_warmup_steps", 0) or 0),
                        }

                        base_loss = getattr(model, "_log_base_loss", None)
                        phys_weighted = getattr(model, "_log_phys_weighted", None)
                        phys_scale = getattr(model, "_log_curriculum_phys_scale", None)
                        t_mean = getattr(model, "_log_t_mean", None)
                        ref_gamma = getattr(model, "_log_ref_gamma_mean", None)

                        base_loss = base_loss.item() if base_loss is not None else 0.0
                        phys_weighted = phys_weighted.item() if phys_weighted is not None else 0.0
                        phys_scale = phys_scale.item() if phys_scale is not None else 1.0
                        t_mean = t_mean.item() if t_mean is not None else -1.0
                        ref_gamma = ref_gamma.item() if ref_gamma is not None else 0.0

                        if base_loss > 1e-12:
                            phys_ratio = phys_weighted / base_loss
                            aux_ratio = phys_ratio
                        else:
                            phys_ratio = aux_ratio = 0.0

                        logger.info("Eval metrics @step {}: {}", step + 1, metrics)
                        logger.info(
                            "Eval loss-config @step {}: "
                            "phys(enable={}, w={}, gate={}, max_ratio={}, ref_update={}, ref_blend_max={}); "
                            "curriculum(enable={}, surface_start={})",
                            step + 1,
                            cfg_snapshot["phys/enable"],
                            cfg_snapshot["phys/weight"],
                            cfg_snapshot["phys/gate_ratio"],
                            cfg_snapshot["phys/max_ratio"],
                            cfg_snapshot["phys/ref_update_enable"],
                            cfg_snapshot["phys/ref_blend_max"],
                            cfg_snapshot["curriculum/enable"],
                            cfg_snapshot["curriculum/surface_start_step"],
                        )
                        logger.info(
                            "Eval cached-loss @step {}: base={:.3e}, phys_w={:.3e}, "
                            "phys_ratio={:.3e}, aux_ratio={:.3e}, "
                            "phys_scale={:.3f}, t_mean={:.1f}, ref_gamma={:.4f}",
                            step + 1,
                            base_loss,
                            phys_weighted,
                            phys_ratio,
                            aux_ratio,
                            phys_scale,
                            t_mean,
                            ref_gamma,
                        )

                        wb = {}
                        wb.update({f"eval/{k}": v for k, v in metrics.items()})
                        wb.update(cfg_snapshot)
                        wb.update(
                            {
                                "eval_cached/base_loss": base_loss,
                                "eval_cached/phys_weighted": phys_weighted,
                                "eval_cached/phys_ratio": phys_ratio,
                                "eval_cached/aux_ratio": aux_ratio,
                                "eval_cached/phys_scale": phys_scale,
                                "eval_cached/t_mean": t_mean,
                                "eval_cached/ref_gamma_mean": ref_gamma,
                            }
                        )
                        safe_wandb_log(wb, step=step + 1, use_wandb=use_wandb)

                except Exception as e:
                    logger.warning("Could not evaluate model. Skipping.")
                    logger.warning("Exception: {}", e)
                    logger.warning("Traceback:\n{}", traceback.format_exc())

                finally:
                    try:
                        if prev_wandb_cfg is not None:
                            cfg.wandb = OmegaConf.create(prev_wandb_cfg)
                    except Exception:
                        pass

            torch.cuda.empty_cache()
            model.train()

    if is_main_process and use_wandb:
        wandb.finish()


if __name__ == "__main__":
    opt = parse_args()

    save_data = DictConfig({})
    save_data.data = opt.data
    save_data.diffusion = opt.diffusion
    save_data.model = opt.model
    save_data.sampling = opt.sampling
    save_data.training = opt.training
    OmegaConf.save(save_data, os.path.join(opt.output_dir, "opt.yaml"))

    opt.ngpus_per_node = torch.cuda.device_count()

    torch.set_float32_matmul_precision("high")

    if opt.distribution_type == "multi":
        opt.world_size = opt.ngpus_per_node * opt.world_size
        opt.training.bs = int(opt.training.bs / opt.ngpus_per_node)
        opt.sampling.bs = opt.training.bs

        mp.spawn(init_processes, nprocs=opt.world_size, args=(opt.world_size, train, opt))
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        opt.gpu = 0
        train(opt)
