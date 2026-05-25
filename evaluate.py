import argparse
import os
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import omegaconf
import pandas as pd
import pytorch3d.ops
import torch
import torch.utils.data
from loguru import logger
from omegaconf import DictConfig

from models.evaluation import Evaluator, farthest_point_sampling
from models.model_loader import load_diffusion
from models.train_utils import set_seed
from utils.utils import NormalizeUnitSphere, write_array_to_xyz


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/objects/examples/", help="Path to noisy point clouds.")
    parser.add_argument("--output_root", type=str, default="./output_objects", help="Output root directory.")
    parser.add_argument("--dataset_root", type=str, default="./data/objects/", help="Path to the dataset root.")
    parser.add_argument(
        "--model_path", type=str, default="./pretrained/PVDS_PUNet/latest.pth", help="Path to the model."
    )
    parser.add_argument("--dataset", type=str, default="PUNet", help="Dataset name.", choices=["PUNet", "PCNet"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--k", type=int, default=3, help="Patch oversampling factor.")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA model for prediction.")
    parser.add_argument("--save_intermediate", action="store_true", help="Save intermediate steps.")
    parser.add_argument("--gpu", type=str, default="cuda:0", help="GPU to use.")
    parser.add_argument("--steps", type=int, default=5, help="Number of steps for the diffusion.")
    parser.add_argument("--niters", type=int, default=1, help="Number of outer denoising iterations.")
    parser.add_argument(
        "--niters_noise_levels",
        type=float,
        nargs="*",
        default=None,
        help="Only apply niters > 1 to these noise levels, for example: --niters_noise_levels 0.03",
    )
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


def input_iter(input_dir: str) -> Generator[Any, Any, Any]:
    """
    Iterate over the input directory and yield the point cloud data.

    Args:
        input_dir (str): The input directory.

    Yields:
        Any: The point cloud data.
    """

    for fn in os.listdir(input_dir):
        if fn[-3:] != "xyz":
            continue
        pcl_noisy = torch.FloatTensor(np.loadtxt(os.path.join(input_dir, fn)))
        pcl_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_noisy)
        yield {"pcl_noisy": pcl_noisy, "name": fn[:-4], "center": center, "scale": scale}


def _select_niters_for_noise(noise: float, cfg: DictConfig) -> int:
    niters = max(1, int(cfg.get("niters", 1)))
    target_noises = cfg.get("niters_noise_levels", None)
    if niters <= 1 or not target_noises:
        return niters

    noise = float(noise)
    for target in target_noises:
        if abs(noise - float(target)) < 1e-8:
            return niters
    return 1


def _write_combined_summary(summary_records: List[Dict[str, Any]], dataset: str, out_root: str) -> Optional[str]:
    rows = []
    summary_name = f"Summary_{dataset}.csv"

    for record in summary_records:
        summary_path = os.path.join(record["output_dir"], summary_name)
        if not os.path.exists(summary_path):
            logger.warning(f"Summary file not found, skipped: {summary_path}")
            continue

        df = pd.read_csv(summary_path)
        if df.empty:
            continue

        if "method" not in df.columns:
            df = df.rename(columns={df.columns[0]: "method"})

        for _, row in df.iterrows():
            item = {
                "dataset": dataset,
                "resolution": record["resolution"],
                "noise": record["noise"],
                "experiment": record["run_title"],
                "exp_dir": os.path.basename(record["output_dir"]),
            }
            item.update(row.to_dict())
            item["summary_path"] = summary_path
            rows.append(item)

    if not rows:
        logger.warning("No summary rows were collected; combined summary was not written.")
        return None

    combined = pd.DataFrame(rows)
    preferred_cols = [
        "dataset",
        "resolution",
        "noise",
        "experiment",
        "exp_dir",
        "method",
        "cd_sph(mean)",
        "p2f(mean)",
        "summary_path",
    ]
    ordered_cols = [col for col in preferred_cols if col in combined.columns]
    ordered_cols.extend([col for col in combined.columns if col not in ordered_cols])
    combined = combined[ordered_cols]

    combined_path = os.path.join(out_root, f"Summary_{dataset}_all.csv")
    os.makedirs(os.path.dirname(combined_path), exist_ok=True)
    combined.to_csv(combined_path, index=False, float_format="%.12f")
    logger.info(f"Combined summary saved to {combined_path}")
    return combined_path


def _extract_patches(
    points: torch.Tensor,
    seed_points: torch.Tensor,
    patch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_points = points.shape[0]
    patch_size = min(patch_size, num_points)
    point_batch = points.unsqueeze(0)
    seed_batch = seed_points.unsqueeze(0)
    _, patch_indices, patches = pytorch3d.ops.knn_points(seed_batch, point_batch, K=patch_size, return_nn=True)
    return patches[0], patch_indices[0]


def _compute_patch_weights(patches: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    dist2 = ((patches - centers) ** 2).sum(dim=-1)
    sigma2 = dist2.mean(dim=1, keepdim=True).clamp_min(1e-8)
    return torch.exp(-dist2 / (2.0 * sigma2))


def _fuse_patch_predictions(
    patch_predictions: torch.Tensor,
    patch_indices: torch.Tensor,
    patch_weights: torch.Tensor,
    reference_points: torch.Tensor,
) -> torch.Tensor:
    num_points = reference_points.shape[0]
    accum_xyz = reference_points.new_zeros((num_points, 3))
    accum_w = reference_points.new_zeros((num_points, 1))

    flat_indices = patch_indices.reshape(-1)
    flat_weights = patch_weights.reshape(-1, 1)
    flat_predictions = patch_predictions.reshape(-1, 3)

    accum_xyz.index_add_(0, flat_indices, flat_predictions * flat_weights)
    accum_w.index_add_(0, flat_indices, flat_weights)

    fused = reference_points.clone()
    valid = accum_w.squeeze(-1) > 0
    fused[valid] = accum_xyz[valid] / accum_w[valid].clamp_min(1e-8)
    return fused


def _project_to_local_plane(point: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
    if neighbors.shape[0] < 3:
        return point

    center = neighbors.mean(dim=0)
    centered = neighbors - center
    cov = centered.transpose(0, 1) @ centered
    cov = cov / max(neighbors.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)

    if not bool(torch.isfinite(eigvals).all()) or eigvals[-1] <= 1e-8:
        return point
    if bool((eigvals[0] / eigvals[1].clamp_min(1e-8)) >= 0.3):
        return point

    normal = eigvecs[:, 0]
    return point - torch.dot(point - center, normal) * normal


def _suppress_outliers(points: torch.Tensor, k: int = 16, alpha: float = 2.75) -> torch.Tensor:
    num_points = points.shape[0]
    if num_points < 4 or k <= 0:
        return points

    query_k = min(num_points, k + 1)
    if query_k <= 1:
        return points

    _, neighbor_indices, neighbor_points = pytorch3d.ops.knn_points(
        points.unsqueeze(0), points.unsqueeze(0), K=query_k, return_nn=True
    )
    neighbor_indices = neighbor_indices[0, :, 1:]
    neighbor_points = neighbor_points[0, :, 1:, :]
    if neighbor_points.shape[1] < 3:
        return points

    mean_dist = torch.norm(neighbor_points - points.unsqueeze(1), dim=-1).mean(dim=-1)
    threshold = mean_dist.mean() + alpha * mean_dist.std(unbiased=False)
    outlier_mask = mean_dist > threshold
    if not bool(outlier_mask.any()):
        return points

    filtered = points.clone()
    for point_idx in torch.nonzero(outlier_mask, as_tuple=False).flatten():
        local_indices = neighbor_indices[point_idx]
        valid_mask = ~outlier_mask[local_indices]
        local_indices = local_indices[valid_mask]
        if local_indices.numel() < 3:
            continue

        neighbors = points[local_indices]
        anchor = points[point_idx]
        dist2 = ((neighbors - anchor) ** 2).sum(dim=-1)
        sigma2 = dist2.mean().clamp_min(1e-8)
        weights = torch.exp(-dist2 / (2.0 * sigma2))
        replacement = (neighbors * weights.unsqueeze(-1)).sum(dim=0) / weights.sum().clamp_min(1e-8)
        filtered[point_idx] = _project_to_local_plane(replacement, neighbors)

    return filtered


@torch.no_grad()
def patch_based_denoise(
    model: torch.nn.Module,
    pcl_noisy: torch.Tensor,
    patch_size: int,
    seed_k: int = 3,
    cfg: omegaconf.DictConfig = None,
    save_intermediate: bool = False,
):
    """

    Args:
        model (torch.nn.Module): The model to use for denoising.
        pcl_noisy (torch.Tensor): The noisy point cloud.
        patch_size (int): The size of the patches.
        seed_k (int): The oversampling factor for the seed points.
        cfg (omegaconf.DictConfig): The configuration dictionary.
        save_intermediate (bool): Whether to save intermediate steps.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: The denoised point cloud and the intermediate steps.

    """
    assert pcl_noisy.dim() == 2, "The shape of input point cloud must be (N, 3)."
    num_points = pcl_noisy.shape[0]
    patch_size = min(patch_size, num_points)

    pcl_noisy_batch = pcl_noisy.unsqueeze(0)
    num_seeds = max(1, int(seed_k * num_points / max(patch_size, 1)))
    seed_points, _ = farthest_point_sampling(pcl_noisy_batch, num_seeds)
    seed_points = seed_points[0]

    patches, patch_indices = _extract_patches(pcl_noisy, seed_points, patch_size)

    model.eval()

    centers = patches.mean(dim=1, keepdim=True)
    patches_centered = patches - centers
    scale = torch.norm(patches_centered, dim=-1).amax(dim=1, keepdim=True).clamp_min(1e-6)
    patches_normed = patches_centered / scale.unsqueeze(-1)
    out = model.sample(
        x_start=patches_normed.transpose(1, 2),
        use_ema=cfg.use_ema,
        steps=cfg.steps,
        log_count=cfg.steps,
        verbose=False,
    )

    patch_weights = _compute_patch_weights(patches, centers)
    patches_denoised = out["x_pred"].transpose(1, 2)

    patches_denoised = patches_denoised * scale.unsqueeze(-1) + centers
    pcl_denoised = _fuse_patch_predictions(patches_denoised, patch_indices, patch_weights, pcl_noisy)

    if cfg is None or not cfg.get("disable_outlier_suppression", False):
        pcl_denoised = _suppress_outliers(
            pcl_denoised,
            k=int(cfg.get("outlier_k", 16)) if cfg is not None else 16,
            alpha=float(cfg.get("outlier_alpha", 2.75)) if cfg is not None else 2.75,
        )

    if save_intermediate:
        patches_steps = out["x_chain"].transpose(-2, -1)
        patches_steps = patches_steps * scale[:, None, :, None] + centers.unsqueeze(1)
        pcl_steps_denoised = [
            _fuse_patch_predictions(patches_steps[:, step_idx], patch_indices, patch_weights, pcl_noisy)
            for step_idx in range(patches_steps.shape[1])
        ]
        pcl_steps_denoised = torch.stack(pcl_steps_denoised, dim=0)
    else:
        pcl_steps_denoised = None

    return pcl_denoised, pcl_steps_denoised


@torch.no_grad()
def sample(
    cfg: DictConfig,
    resolutions: List[int] = [10000, 50000],
    noises: List[float] = [0.01, 0.02, 0.03],
    save_title: str = "SCDiff",
) -> None:
    """
    Sample from the model.

    Args:
        cfg (DictConfig): The configuration dictionary.
        resolutions (List[int]): The resolutions to sample.
        noises (List[float]): The noise levels to sample.
        save_title (str): Title used in output directory names.
    """
    set_seed(cfg)
    torch.cuda.manual_seed_all(cfg.training.seed)
    torch.backends.cudnn.benchmark = True

    model, _ = load_diffusion(cfg)
    model.eval()

    out_root = os.path.join(cfg.output_root, cfg.dataset)

    if cfg.use_ema:
        save_title += "_ema"

    summary_records = []

    for res in resolutions:
        for noise in noises:
            current_niters = _select_niters_for_noise(noise, cfg)
            run_title = save_title
            if current_niters > 1:
                run_title += f"_niters_{current_niters}_steps_{cfg.steps}"
            else:
                run_title += f"_steps_{cfg.steps}"

            input_dir = os.path.join(cfg.data_path, "%s_%s_poisson_%s" % (cfg.dataset, res, noise))
            output_dir = os.path.join(out_root, f"{run_title}_{res}_{noise}")

            for data in input_iter(input_dir):
                logger.info(f"Processing {data['name']}")
                pcl_noisy = data["pcl_noisy"].cuda()
                model.eval()
                pcl_next = pcl_noisy
                all_iter_steps = []

                for iter_idx in range(current_niters):
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

                save_path = os.path.join(output_dir, "pcl", data["name"] + ".xyz")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                write_array_to_xyz(save_path, pcl_denoised.numpy())

                if cfg.save_intermediate and all_iter_steps:
                    single_iter = len(all_iter_steps) == 1
                    for iter_idx, iter_steps in all_iter_steps:
                        for step, item in enumerate(iter_steps):
                            step_points = item.cpu()
                            step_points = step_points * data["scale"] + data["center"]
                            file_name = data["name"] + f"_{step}.xyz"
                            if single_iter:
                                out_path = os.path.join(output_dir, "steps", data["name"], file_name)
                            else:
                                out_path = os.path.join(
                                    output_dir,
                                    "steps",
                                    data["name"],
                                    f"iter_{iter_idx + 1}",
                                    file_name,
                                )
                            os.makedirs(os.path.dirname(out_path), exist_ok=True)
                            write_array_to_xyz(out_path, step_points.numpy())

            evaluator = Evaluator(
                output_pcl_dir=os.path.join(output_dir, "pcl"),
                dataset_root=cfg.dataset_root,
                dataset=cfg.dataset,
                summary_dir=output_dir,
                experiment_name=run_title,
                device="cuda",
                res_gts=f"{res}_poisson",
            )
            evaluator.run()
            summary_records.append(
                {
                    "resolution": res,
                    "noise": noise,
                    "run_title": run_title,
                    "output_dir": output_dir,
                }
            )

    _write_combined_summary(summary_records, cfg.dataset, out_root)


def main():
    opt = parse_args()
    sample(opt)


if __name__ == "__main__":
    main()
