import torch 
import torch.nn as nn 

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def get_points_on_ray(batch_rays_o, batch_rays_d, low=2, high=6, n_samples=128):
    B, _ = batch_rays_o.shape
    t_vals = torch.linspace(low, high, n_samples, device=batch_rays_o.device)  # (N,)

    # (B, 1, 3) + (1, N, 1) * (B, 1, 3) -> (B, N, 3)
    rays_o_exp = batch_rays_o[:, None, :] # (B, 1, 3)
    rays_d_exp = batch_rays_d[:, None, :] # (B, 1, 3)
    t_exp = t_vals[None, :, None] # (1, N, 1)
    points = rays_o_exp + t_exp * rays_d_exp # (B, N, 3)

    return t_vals, points


class PositionalEncoding(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.l = l

    def forward(self, x):
        B, n_samples, n_points = x.shape
        
        k = torch.arange(0, self.l, 1, device=device)
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
    
    def forward(self, x):
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
    alpha = torch.clamp(alpha, 0.0, 1.0)
    trans = torch.clamp(1 - alpha, min=0.0)
    T = torch.cumprod(trans + eps, dim=1)
    T = torch.cat(
        [torch.ones_like(T[:, :1, :]), T[:, :-1, :]],
        dim=1
    ) # (B, N, 1)
    weights = T * alpha
    weights = torch.clamp(weights, min=0.0)
    rgb_map = (weights * rgb).sum(dim=1)

    return rgb_map # (B, 3)


class NeRF(nn.Module):
    def __init__(
            self,
            L_pos,
            L_dir, 
            in_channels, 
            out_channels
    ):
        super().__init__()
        self.xyz_enc = PositionalEncoding(L_pos)
        self.main_mlp = MainMLP(in_channels, out_channels)
        self.dir_enc = PositionalEncoding(L_dir)
        self.color_mlp = ColorMLP(out_channels + (3 + 6*L_dir))
    
    def forward(self, batch_rays_o, batch_rays_d):
        t_val, points = get_points_on_ray(batch_rays_o, batch_rays_d) # (N,), (B, N, 3)
        B, N, _ = points.shape
        xyz_enc = self.xyz_enc(points) # (B, N, 3 + 6L)
        sigma, feature_map = self.main_mlp(xyz_enc)
        dirs = batch_rays_d[:, None, :].expand(B, N, 3)
        dir_enc = self.dir_enc(dirs)
        color_inp = torch.cat([dir_enc, feature_map], dim=-1)
        color = self.color_mlp(color_inp)
        rgb_map = volume_renderer(sigma, color, t_val)

        return rgb_map # B, 3
    
    # in nerf.py, inside NeRF class
    def debug_single_ray(self, ray_o, ray_d):
        # 1. sample points along ray
        t_vals, points = get_points_on_ray(ray_o, ray_d, low=2.0, high=6.0, n_samples=64)
        # t_vals: (N,), points: (1, N, 3)

        # 2. encode positions
        xyz_enc = self.xyz_enc(points)                 # (1, N, C_pos)
        sigma, feat = self.main_mlp(xyz_enc)          # sigma: (1, N, 1), feat: (1, N, F)

        # 3. encode direction (repeat per sample)
        dirs = ray_d.unsqueeze(1).repeat(1, t_vals.numel(), 1)  # (1, N, 3)
        dir_enc = self.dir_enc(dirs)                            # (1, N, C_dir)

        color_inp = torch.cat([feat, dir_enc], dim=-1)          # (1, N, F+C_dir)
        rgb = self.color_mlp(color_inp)                         # (1, N, 3)

        # 4. compute weights using your volume_renderer, but also capture weights
        # copy of your volume renderer math
        dists = t_vals[1:] - t_vals[:-1]
        dists = torch.cat([dists, dists[-1:]], dim=0)           # (N,)
        dists = dists.view(1, -1, 1)                            # (1, N, 1)

        alpha = 1.0 - torch.exp(-sigma * dists) # (1, N, 1)
        alpha = torch.clamp(alpha, 0.0, 1.0)
        trans = torch.clamp(1 - alpha, min=0.0)
        T = torch.cumprod(trans + 1e-10, dim=1)
        T = torch.cat(
            [torch.ones_like(T[:, :1, :]), T[:, :-1, :]],
            dim=1
        ) # (B, N, 1)
        weights = (T * alpha)[0, :, 0]                          # (N,)
        weights = torch.clamp(weights, min=0.0)
        return (
            t_vals.detach().cpu(),
            weights.detach().cpu(),
            rgb[0].detach().cpu(),
            sigma[0, :, 0].detach().cpu(),
        )