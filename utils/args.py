import argparse
import json
import os

from omegaconf import OmegaConf


def args_to_string(args):
    args_dict = OmegaConf.to_container(args, resolve=True)
    args_str = json.dumps(args_dict, indent=4)
    return args_str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file.")
    parser.add_argument("--name", type=str, default="", help="Name of the experiment.")
    parser.add_argument("--save_dir", default="outputs", help="path to save models")
    parser.add_argument("--wandb_project", type=str, default="SCDiff", help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="", help="wandb entity name")
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="path to model (to continue training)",
    )
    parser.add_argument("--restart", action="store_true", help="restart training from scratch")

    parser.add_argument("--world_size", default=1, type=int, help="Number of distributed nodes.")
    parser.add_argument(
        "--master_address",
        default="localhost",
        type=str,
        help="Address of master node.",
    )
    parser.add_argument("--master_port", default="6021", type=str, help="Port of master node.")
    parser.add_argument("--dist_backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument(
        "--distribution_type",
        default="single",
        choices=["multi", "single", None],
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    parser.add_argument("--node_rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=False,
        help="Use exponential moving average of model parameters.",
    )

    args, remaining_argv = parser.parse_known_args()
    if args.model_path != "" and args.save_dir == "outputs":
        args.save_dir = os.path.dirname(args.model_path)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.config is not None:
        cfg = OmegaConf.load(args.config)
    elif args.model_path != "":
        try:
            cfg = OmegaConf.load(os.path.join(os.path.dirname(args.model_path), "opt.yaml"))
        except FileNotFoundError:
            cfg = OmegaConf.load(args.config)
    else:
        raise ValueError("config file must be specified or model path must be specified")

    opt = OmegaConf.merge(cfg, OmegaConf.create(vars(args)))

    if remaining_argv:
        for i in range(0, len(remaining_argv), 2):
            key = remaining_argv[i].lstrip("--")
            value = remaining_argv[i + 1].strip()

            try:
                if value in ["True", "False", "true", "false"]:
                    value = value.capitalize() == "True"
                elif "." in remaining_argv[i + 1] or "e" in remaining_argv[i + 1]:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass

            OmegaConf.update(opt, key, value, merge=False)

    if opt.name == "" and opt.config is not None:
        opt.name = os.path.splitext(os.path.basename(opt.config))[0]

    if opt.model_path != "":
        if "timesteps_clip" not in opt.diffusion:
            opt.diffusion.timesteps_clip = opt.diffusion.timesteps

        if "clip" not in opt.diffusion:
            opt.diffusion.clip = False

        if "dynamic_threshold" not in opt.diffusion:
            opt.diffusion.dynamic_threshold = False

        model_name = opt.model_path.split("/")[-1].split(".")[0].split("_")[-1]
        steps = min(opt.diffusion.sampling_timesteps, opt.diffusion.timesteps_clip)
        scheduler_info = f"{opt.diffusion.sampling_strategy}(T={str(steps)})"

        if opt.diffusion.timesteps_clip < opt.diffusion.timesteps:
            scheduler_info += f"_ts_clip{str(opt.diffusion.timesteps_clip)}"

        if opt.diffusion.clip:
            if opt.diffusion.dynamic_threshold:
                clip = "_clip_dynamic"
            else:
                clip = "_clip"
        else:
            clip = ""
        scheduler_info += clip

        if args.use_ema:
            scheduler_info += "_ema"

        opt.out_sampling = os.path.join(os.path.dirname(opt.model_path), "sampling", model_name, scheduler_info)

    output_dir = os.path.join(opt.save_dir, opt.name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    opt.output_dir = output_dir

    opt.training.max_epochs = 1000

    return opt
