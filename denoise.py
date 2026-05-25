import argparse
import os

import numpy as np
import omegaconf
import open3d as o3d
import torch
from omegaconf import DictConfig

from evaluate import patch_based_denoise
from models.model_loader import load_diffusion
from models.train_utils import set_seed
from utils.utils import NormalizeUnitSphere, write_array_to_xyz


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the point cloud.")
    parser.add_argument("--save_path", type=str, required=True, help="Output root directory.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--k", type=int, default=3, help="Patch oversampling factor.")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA model for prediction.")
    parser.add_argument("--gpu", type=str, default="cuda:0", help="GPU to use.")
    parser.add_argument("--steps", type=int, default=5, help="Number of steps for the diffusion.")
    parser.add_argument("--niters", type=int, default=1, help="Number of outer denoising iterations.")
    parser.add_argument("--save_intermediate", action="store_true", help="Save intermediate sampling outputs.")
    parser.add_argument("--outlier_k", type=int, default=16, help="kNN size for mild outlier suppression.")
    parser.add_argument("--outlier_alpha", type=float, default=2.75, help="Std-based threshold for outlier repair.")
    parser.add_argument("--disable_outlier_suppression", action="store_true", help="Disable mild outlier repair.")
    parser.add_argument("--distribution_type", default="none")
    args = parser.parse_args()

    cfg_path = os.path.join(os.path.dirname(args.model_path), "opt.yaml")
    cfg = omegaconf.OmegaConf.load(cfg_path)
    cfg = omegaconf.OmegaConf.merge(cfg, omegaconf.OmegaConf.create(vars(args)))
    cfg.restart = False
    cfg.local_rank = 0
    return cfg


@torch.no_grad()
def sample(
    cfg: DictConfig,
) -> None:
    """
    Sample from the model.

    Args:
        cfg (DictConfig): The configuration dictionary.
    """
    set_seed(cfg)
    torch.cuda.manual_seed_all(cfg.training.seed)
    torch.backends.cudnn.benchmark = True

    model, _ = load_diffusion(cfg)
    model.eval()

    input_ext = os.path.splitext(cfg.data_path)[1].lower()
    output_ext = os.path.splitext(cfg.save_path)[1].lower()

    if input_ext == ".ply":
        pcl_noisy = o3d.io.read_point_cloud(cfg.data_path)
        pcl_noisy = torch.FloatTensor(np.asarray(pcl_noisy.points))
    else:
        pcl_noisy = torch.FloatTensor(np.loadtxt(cfg.data_path))

    pcl_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_noisy)
    data = {"pcl_noisy": pcl_noisy, "center": center, "scale": scale}

    pcl_noisy = data["pcl_noisy"].cuda()
    with torch.no_grad():
        model.eval()
        pcl_next = pcl_noisy
        all_iter_steps = []
        for iter_idx in range(max(1, int(cfg.niters))):
            pcl_next, pcl_next_steps = patch_based_denoise(
                model=model,
                pcl_noisy=pcl_next,
                patch_size=2048,
                seed_k=cfg.k,
                cfg=cfg,
                save_intermediate=cfg.save_intermediate,
            )
            if cfg.save_intermediate and pcl_next_steps is not None:
                all_iter_steps.append((iter_idx, pcl_next_steps))

    pcl_denoised = pcl_next.cpu()
    pcl_denoised = pcl_denoised * data["scale"] + data["center"]

    if output_ext != ".xyz":
        raise NotImplementedError("Only .xyz output files are supported for now.")

    os.makedirs(os.path.dirname(os.path.abspath(cfg.save_path)), exist_ok=True)
    write_array_to_xyz(cfg.save_path, pcl_denoised.numpy())

    if cfg.save_intermediate and all_iter_steps:
        save_dir = os.path.dirname(os.path.abspath(cfg.save_path))
        save_stem = os.path.splitext(os.path.basename(cfg.save_path))[0]
        steps_root = os.path.join(save_dir, f"{save_stem}_steps")
        single_iter = len(all_iter_steps) == 1

        for iter_idx, iter_steps in all_iter_steps:
            for step, item in enumerate(iter_steps):
                step_points = item.cpu()
                step_points = step_points * data["scale"] + data["center"]
                file_name = f"{save_stem}_{step}.xyz"
                if single_iter:
                    out_path = os.path.join(steps_root, file_name)
                else:
                    out_path = os.path.join(steps_root, f"iter_{iter_idx + 1}", file_name)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                write_array_to_xyz(out_path, step_points.numpy())


def main():
    opt = parse_args()
    sample(opt)


if __name__ == "__main__":
    main()
