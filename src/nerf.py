import torch 
import torch.nn as nn 
import torch.nn.functional as F
import os
import json
import numpy as np
from PIL import Image
from datasets import load_dataset

ds = load_dataset("./nerf/nerf_synthetic/lego") # each image is # (800, 800, 3)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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

def get_points_on_ray(batch_rays_o, batch_rays_d, low, high, n_samples):
    B, num_points = batch_rays_o.shape
    ray_slices = torch.linspace(low, high, n_samples, device=device)
    points = batch_rays_o.view(B, -1, num_points) + ray_slices.view(1, n_samples, 1) * batch_rays_d.view(B, -1, num_points) # (B, 1, 3) + ((64, 1)@(B, 1, 3))

    return ray_slices, points # (n_samples,), (B, N, 3)


class PositionalEncoding(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.l = l

    def forward(self, x):
        B, n_samples, n_points = x.shape
        
        k = torch.arange(0, self.l, 1)
        k = k.view(1, 1, 1, -1) # (1, 1, 1, L)

        freqs = torch.pow(2, k) * torch.pi # (1, 1, 1, L)

        x_new = x.view(B, n_samples, n_points, 1) # (B, N, 3, 1)

        sin_features = torch.sin(freqs * x_new) # (B, N, 3, L)
        cos_features = torch.cos(freqs * x_new) # (B, N, 3, L)

        sin_features = sin_features.view(B, n_samples, -1) # (B, n_samples, 3L)
        cos_features = cos_features.view(B, n_samples, -1) # (B, n_samples, 3L)

        out = torch.cat([x, sin_features, cos_features], dim=-1) # (B, n_samples, 3 + 6L)

        return out # (B, n_samples, 3 + 6L)


class MainMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
        )
        self.net2 = nn.Sequential(
            nn.Linear(out_channels + in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU()
        )

        self.sigma = nn.Linear(out_channels, 1)

    
    def forward(self, x):
        x_enc = x

        h = self.net1(x_enc)
        h = torch.cat([x_enc, h], dim=-1)
        feature_vector = self.net2(h)


        sigma = self.sigma(feature_vector)


        return sigma, feature_vector


class ColorMLP(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.l1 = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(out_channels, 3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x_color, dir_enc):
        x = torch.cat([x_color, dir_enc], dim=-1)
        color = self.sigmoid(self.l2(self.relu(self.l1(x))))

        return color
    

def volume_renderer(sigma, rgb, t_vals, eps=1e-10):
    # sigma ---> (B, N, 1)
    # rgb ---> (B, N, 3)
    # t_vals ---> (N,)

    dists = t_vals[1:] - t_vals[:-1] # (N-1)
    last = dists[-1:]
    dists = torch.cat([dists, last], dim=0) # (N,)
    dists = dists.view(1, dists.size(0), 1) # (1, N, 1)
    alpha = 1 - torch.exp(-sigma * dists) # (B, N, 1)
    T = torch.cumprod(1 - alpha + eps, dim=1)
    T = torch.cat(
        [torch.ones_like(T[:, :1, :]), T[:, :-1, :]],
        dim=1
    ) # (B, N, 1)
    weights = T * alpha
    rgb_map = (weights * rgb).sum(dim=1)

    return rgb_map # (B, 3)
