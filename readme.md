<p align="center">
  <h1 align="center">SCDiff: Surface-Consistent Diffusion for High-Fidelity Point Cloud Denoising</h1>
  <p align="center">
    Canyu Cai and Bo Jiang
  </p>
</p>

**SCDiff** restores clean surfaces from noisy high-density point clouds with a surface-constrained diffusion model. It combines dynamic reference surfaces, reliability-aware anisotropic constraints, and weighted patch fusion for stable denoising.

## Requirements

The code was developed with Python 3.10, PyTorch, CUDA, PyTorch3D, and several custom CUDA extensions. A typical setup is:

```bash
conda create -n scdiff python=3.10
conda activate scdiff
```

Install PyTorch according to your CUDA version. For CUDA 11.8, one working setup is:

```bash
conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=11.8 -c pytorch -c nvidia --yes
```

Then install PyTorch3D and TorchCluster following their official instructions. Finally install the remaining dependencies and compile the custom operators:

```bash
pip install -r requirements.txt
sh install.sh
```

## Object Datasets

Download the PU-Net and PC-Net object datasets from [ScoreDenoise](https://github.com/luost26/score-denoise), then extract them under `data/objects`:

```text
data/objects/
  examples/
  PCNet/
  PUNet/
```

Use `--data_path` and `--dataset_root` if your dataset is stored elsewhere.

## Training

Run the main PU-Net experiment:

```bash
python train.py --config configs/opt.yaml --save_dir outputs --name scdiff_punet
```

Resume from a checkpoint:

```bash
python train.py --model_path outputs/scdiff_punet/step_20000.pth
```

Use distributed training on one node:

```bash
python train.py --config configs/opt.yaml --save_dir outputs --name scdiff_punet --distribution_type multi
```

Training writes the following artifacts to `outputs/<name>`:

```text
opt.yaml
config_resolved.yaml
cmd.txt
env.json
step_*.pth
```

`opt.yaml` is required by the evaluation and denoising scripts because it stores the model and diffusion configuration used by the checkpoint.

## Pretrained Models

Place trained or pretrained checkpoints under `pretrained`:

```text
pretrained/
  SCD_PUNet/
    opt.yaml
    latest.pth
```

The checkpoint directory must contain both the weight file and the matching `opt.yaml`.

## Evaluation

Evaluate on the PU-Net object benchmark:

```bash
python evaluate.py --model_path ./pretrained/SCD_PUNet/latest.pth --dataset PUNet --steps 5 --use_ema
```

Evaluate on PC-Net:

```bash
python evaluate.py --model_path ./pretrained/SCD_PUNet/latest.pth --dataset PCNet --steps 5 --use_ema
```

Useful options:

```bash
--steps 5                         Number of DDPM sampling steps
--use_ema                         Use EMA weights from the checkpoint
--k 3                             Patch oversampling factor
--niters 2                        Run repeated outer denoising iterations
--niters_noise_levels 0.03        Apply repeated iterations only to selected noise levels
--save_intermediate               Save intermediate sampling outputs
--disable_outlier_suppression     Disable mild outlier repair
```

Outputs and metrics are saved to:

```text
output_objects/<dataset>/
```

Each evaluated resolution/noise pair gets its own folder and summary CSV. A combined summary is written as `Summary_<dataset>_all.csv`.

## Denoise Your Own Data

To denoise a single `.xyz` point cloud:

```bash
python denoise.py \
  --data_path path/to/noisy.xyz \
  --save_path path/to/denoised.xyz \
  --model_path ./pretrained/SCD_PUNet/latest.pth \
  --steps 5 \
  --k 3 \
  --use_ema
```

The input is normalized to the unit sphere before denoising and transformed back to the original coordinate scale before saving.
