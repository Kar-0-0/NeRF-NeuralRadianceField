import torch 
import torch.nn.functional as F
import os
import json
import numpy as np
from PIL import Image
from datasets import load_dataset
from nerf import NeRF

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

ds = load_dataset("./nerf/nerf_synthetic/lego") # each image is # (800, 800, 3)
def load_blender_nerf_scene(basedir='nerf/nerf_synthetic/lego/', split='train'):
    # Load metadata
    with open(os.path.join(basedir, f'transforms_{split}.json'), 'r') as fp:
        meta = json.load(fp)

    imgs = []
    poses = []

    for frame in meta['frames']:
        # Image path
        fname = os.path.join(basedir, frame['file_path'] + '.png')
        if not os.path.isfile(fname):
            fname = os.path.join(basedir, frame['file_path'] + '.jpg')

        img = np.array(Image.open(fname)) / 255.0  # [H, W, 4] or [H, W, 3]
        if img.shape[-1] == 4:
            img = img[..., :3]  # drop alpha
        imgs.append(img.astype(np.float32))

        # Pose (4x4 c2w)
        pose = np.array(frame['transform_matrix'], dtype=np.float32)
        poses.append(pose)

    images = np.stack(imgs, axis=0)   # [N, H, W, 3]
    poses = np.stack(poses, axis=0)   # [N, 4, 4]

    # Get intrinsics from camera_angle_x and image width
    H, W = images.shape[1:3]
    camera_angle_x = meta['camera_angle_x']  # horizontal FOV in radians
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)  # fx = fy = focal

    fx = fy = focal
    cx = W * 0.5
    cy = H * 0.5

    intrinsics = {
        'H': H,
        'W': W,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
    }

    return torch.tensor(images, device=device), torch.tensor(poses, device=device), intrinsics

def get_world_rays(images, poses, intrinsics):
    H = intrinsics['H']
    W = intrinsics['W']
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']

    N = len(images)

    u, v = torch.meshgrid(
        torch.arange(H),
        torch.arange(W),
        indexing='xy'
    )

    xn = (u - cx) / fx
    yn = (v - cy) / fy 

    d_cam = torch.stack([xn, yn, torch.ones_like(xn)], dim=-1).to(device) # (H, W, 3)
    d_cam = d_cam / torch.norm(d_cam, dim=-1, keepdim=True)

    rays_d = torch.zeros((N, H, W, 3), device=device)
    rays_o = torch.zeros((N, H, W, 3), device=device)

    for i in range(N):
        pose = poses[i]
        R = pose[:3, :3] # (3, 3)
        o = pose[:3, 3] # (3,)

        d_world = d_cam @ R.T # (H, W, 3)
        d_world = d_world / torch.norm(d_world, dim=-1, keepdim=True)

        rays_d[i] = d_world
        rays_o[i] = o

    return rays_o, rays_d

def get_batch(images, rays_o, rays_d, batch_size):
    # Shape for all is (N, H, W, 3)
    N, H, W, _ = images.shape
    v, u = torch.randint(0, H, (batch_size,)).to(device), torch.randint(0, W, (batch_size,)).to(device)
    img_ix = torch.randint(0, N, (batch_size,), device=device)
    batch_rays_o = rays_o[img_ix, v, u]
    batch_rays_d = rays_d[img_ix, v, u]
    batch_rays_rgb = images[img_ix, v, u]

    return batch_rays_o, batch_rays_d, batch_rays_rgb

import torchvision
from torchvision.utils import make_grid

def render_image(model, pose, intrinsics, chunk_size=4096):
    H, W = intrinsics['H'], intrinsics['W']
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']

    # Rays for single pose
    u, v = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='xy'
    )
    xn = (u - cx) / fx
    yn = (v - cy) / fy
    d_cam = torch.stack([xn, yn, torch.ones_like(xn)], dim=-1)
    d_cam = d_cam / torch.norm(d_cam, dim=-1, keepdim=True)

    R = pose[:3, :3]
    o = pose[:3, 3]
    d_world = d_cam @ R.T
    d_world = d_world / torch.norm(d_world, dim=-1, keepdim=True)

    rays_d = d_world.view(-1, 3)
    rays_o = o.expand(H * W, -1)

    model.eval()
    rgb_chunks = []
    with torch.no_grad():
        for i in range(0, H * W, chunk_size):
            rays_o_chunk = rays_o[i:i + chunk_size]
            rays_d_chunk = rays_d[i:i + chunk_size]
            rgb_chunk = model(rays_o_chunk, rays_d_chunk)
            rgb_chunks.append(rgb_chunk)

    rgb_map = torch.cat(rgb_chunks, dim=0).view(H, W, 3).permute(2, 0, 1)  # (3, H, W)
    rgb_map = rgb_map.clamp(0, 1).cpu()

    return rgb_map


# Model Hyperparameters
L_pos = 10
L_dir = 4
in_channels = 3 + 6*L_pos
out_channels = 256
lr = 5e-4

# Batch Hyperparameters
batch_size = 2048

# Training Hyeprparameters
epochs = 10_000

model = NeRF(L_pos, L_dir, in_channels, out_channels)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

images, poses, intrinsics = load_blender_nerf_scene()
rays_o, rays_d = get_world_rays(images, poses, intrinsics)

val_poses, _, _ = load_blender_nerf_scene(split='val')
val_idx = 0
val_pose = val_poses[val_idx]
gt_img = images[val_idx].permute(2, 0, 1).cpu()

for epoch in range(epochs):
    batch_rays_o, batch_rays_d, batch_rays_rgb = get_batch(images, rays_o, rays_d, batch_size)
    rgb_map = model(batch_rays_o, batch_rays_d)
    loss = F.mse_loss(rgb_map, batch_rays_rgb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0 or epoch == 0:
        e = epoch+1
        print(f"Epoch {e} Loss: {loss.item()}")
    if (epoch+1) % 1000 == 0:
        torch.save(model.state_dict(), f"nerf_epoch{epoch+1}.pth")
    
        pred_img = render_image(model, val_pose, intrinsics)  # (3, H, W)

        grid = make_grid(
            torch.stack([gt_img, pred_img], dim=0),  # (2, 3, H, W)
            nrow=2,
            padding=2
        )  # (3, H, 2W + padding)
        # save grid
        torchvision.utils.save_image(grid, f"comparison_epoch{epoch+1}.png")
        print(f"Saved comparison_epoch{epoch+1}.png")

