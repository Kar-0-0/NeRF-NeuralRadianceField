text
# NeRF from Scratch (PyTorch, Lego Scene)

A from‑scratch implementation of a vanilla Neural Radiance Field (NeRF) trained on the Blender **lego** scene using pure PyTorch and minimal dependencies. The goal is readability and learning, not speed.

## Features

- End‑to‑end NeRF pipeline:
  - Blender lego data loader (images, poses, intrinsics)  
  - Ray generation in world coordinates  
  - Stratified sampling along rays with configurable `n_samples`  
  - Positional encodings for positions and view directions  
  - Density MLP (σ + features) and view‑dependent color MLP  
  - Volume rendering with numerically stable transmittance/weights  
- Simple training script:
  - Random ray batching across all views  
  - MSE loss on RGB  
  - Periodic checkpoints and visualizations  
- Side‑by‑side comparison images:
  - Left: ground‑truth lego image  
  - Right: current NeRF render from the same camera  

## Project Structure

- `nerf.py`  
  - `get_points_on_ray`: samples 3D points \(x = o + t d\) for each ray  
  - `PositionalEncoding`: sinusoidal encoding for \(\mathbb{R}^3\) inputs  
  - `MainMLP`: outputs density σ and a feature vector per 3D sample  
  - `ColorMLP`: predicts RGB given features + encoded view direction  
  - `volume_renderer`: converts σ and RGB along each ray into final pixel colors  
  - `NeRF`: wraps everything into a single model with a clean `forward` and a `debug_single_ray` helper  
- `train_nerf.py`  
  - `load_blender_nerf_scene`: loads lego images, poses, and camera intrinsics  
  - `get_world_rays`: turns intrinsics + camera‑to‑world transforms into per‑pixel rays  
  - `get_batch`: randomly samples rays and target colors from all images  
  - Training loop: optimizes NeRF with Adam, logs loss, saves checkpoints and comparison PNGs  

## Requirements

- Python 3.10+  
- PyTorch (MPS/CPU/GPU)  
- `torchvision`  
- `numpy`, `Pillow`  
- `datasets` (for loading the local lego dataset directory) [web:6]

## Install with:
``` bash
pip install torch torchvision numpy pillow datasets
```


## Data

This code expects the **Blender synthetic lego** scene in:
``` text
nerf/nerf_synthetic/lego/
transforms_train.json
transforms_val.json
*.png / *.jpg
```

The intrinsics are derived from `camera_angle_x` in the transforms file, as in the original NeRF paper.

## Usage

### Training

Run:
```bash
python src/train_nerf.py
```


Key hyperparameters (set in `train_nerf.py`):

- `L_pos = 10`, `L_dir = 4`  
- `in_channels = 3 + 6 * L_pos`  
- `hidden_dim = 256`  
- `n_samples = 128` (points per ray)  
- `batch_size = 2048` (rays per step)  
- `epochs = 20000` (training iterations)  

The script:

- Randomly samples ray batches from all training images  
- Computes predicted colors with `NeRF.forward`  
- Minimizes MSE against ground‑truth RGB  
- Saves model weights every 1000 steps as `nerf_epochXXXX.pth`  

### Visualization

Every 1000 steps, the training script:

1. Renders a full image from a chosen training camera using `render_image`.  
2. Builds a grid `[ground truth | NeRF render]` with `torchvision.utils.make_grid`.  
3. Saves it as `comparison_epochXXXX.png`.  

This lets you see the reconstruction quality improve over time. [web:6]

## Debugging and Learning Tools

The `NeRF.debug_single_ray` method lets you inspect what happens along one ray:

- Sample depths `t_vals`  
- Weights per depth (should be ≥ 0 and sum ≤ 1)  
- Densities σ and per‑sample RGB  

This is useful for understanding:

- Where along the ray the model thinks the object is  
- Whether near/far bounds and sampling are reasonable  
- How volume rendering behaves numerically  

## Roadmap / Extensions

Some natural next steps:

- Hierarchical (coarse + fine) sampling with a second MLP  
- Stratified / jittered sampling instead of pure linspace  
- Multi‑scene training or LLFF (forward‑facing) datasets  
- Acceleration with CUDA kernels or libraries like nerfacc