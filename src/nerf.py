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

images, poses, intrinsics = load_blender_nerf_scene()

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

    xn = u - cx / fx
    yn = v - cy / fy 

    d_cam = torch.stack([xn, yn, torch.ones_like(xn)], dim=-1).to(device) # (H, W, 3)
    d_cam = d_cam / torch.norm(d_cam, dim=-1, keepdim=True)

    rays_d = torch.zeros((N, H, W, 3))
    rays_o = torch.zeros((N, H, W, 3))

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
    v, u = torch.randint(0, H, (batch_size,)), torch.randint(0, W, (batch_size,))
    img_ix = torch.randint(0, N, (batch_size,))
    batch_rays_o = rays_o[img_ix, v, u]
    batch_rays_d = rays_d[img_ix, v, u]
    batch_rays_rgb = images[img_ix, v, u]

    return batch_rays_o, batch_rays_d, batch_rays_rgb



rays_o, rays_d = get_world_rays(images, poses, intrinsics)
batch_rays_o, batch_rays_d, batch_rays_rgb = get_batch(images, rays_o, rays_d, 32)

print(batch_rays_o.shape, batch_rays_d.shape, batch_rays_rgb.shape)
