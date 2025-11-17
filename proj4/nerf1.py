import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import imageio.v2 as imageio 
from PIL import Image
import viser
import time

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(output_dir, exist_ok=True)
SNAPSHOT_ITERS = [0, 200, 400, 600, 1000]
VAL_EVERY = 100
PROG_SAVE_EVERY = 200

def transform(c2w, x_c):
    """
    Transform points from camera space to world space.
    
    Args:
        c2w: Camera-to-world transformation matrix of shape [4, 4] or [N, 4, 4]
        x_c: Points in camera space of shape [3], [N, 3], or [M, N, 3]
    
    Returns:
        x_w: Points in world space with same shape as x_c
    """
    if isinstance(c2w, torch.Tensor):
        lib = torch
    else:
        lib = np
    
    original_shape = x_c.shape
    
    if len(x_c.shape) == 1:
        if lib == np:
            x_c_homogeneous = np.concatenate([x_c, np.ones(1)])
        else:
            x_c_homogeneous = torch.cat([x_c, torch.ones(1, device=x_c.device, dtype=x_c.dtype)])
        x_c_homogeneous = x_c_homogeneous.reshape(4, 1)
    else:
        ones_shape = list(x_c.shape[:-1]) + [1]
        if lib == np:
            ones = np.ones(ones_shape)
            x_c_homogeneous = np.concatenate([x_c, ones], axis=-1)
        else:
            ones = torch.ones(ones_shape, device=x_c.device, dtype=x_c.dtype)
            x_c_homogeneous = torch.cat([x_c, ones], dim=-1)
    
    if len(c2w.shape) == 2:
        if len(original_shape) == 1:
            x_w_homogeneous = c2w @ x_c_homogeneous
            x_w = x_w_homogeneous[:3, 0]
        else:
            x_w_homogeneous = (c2w @ x_c_homogeneous[..., None])[..., 0]
            x_w = x_w_homogeneous[..., :3]
    else:
        x_w_homogeneous = (c2w @ x_c_homogeneous[..., None])[..., 0]
        x_w = x_w_homogeneous[..., :3]
    
    return x_w


def pixel_to_camera(K, uv, s):
    """
    Transform points from pixel coordinates to camera space.
    
    Args:
        K: Camera intrinsic matrix of shape [3, 3]
        uv: Pixel coordinates of shape [2], [N, 2], or [M, N, 2]
        s: Depth values of shape [], [N], or [M, N]
    
    Returns:
        x_c: Points in camera space of shape [3], [N, 3], or [M, N, 3]
    """
    if isinstance(K, torch.Tensor):
        lib = torch
    else:
        lib = np
    
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    if len(uv.shape) == 1:
        u = uv[0]
        v = uv[1]
    else:
        u = uv[..., 0]
        v = uv[..., 1]
    
    x_c = (u - cx) * s / fx
    y_c = (v - cy) * s / fy
    z_c = s
    
    if len(uv.shape) == 1:
        if lib == np:
            x_c_out = np.array([x_c, y_c, z_c])
        else:
            x_c_out = torch.stack([x_c, y_c, z_c])
    else:
        if lib == np:
            x_c_out = np.stack([x_c, y_c, z_c], axis=-1)
        else:
            x_c_out = torch.stack([x_c, y_c, z_c], dim=-1)
    
    return x_c_out


def pixel_to_ray(K, c2w, uv):
    """
    Convert pixel coordinates to rays with origin and direction.
    
    Args:
        K: Camera intrinsic matrix of shape [3, 3]
        c2w: Camera-to-world transformation matrix of shape [4, 4]
        uv: Pixel coordinates of shape [2], [N, 2], or [M, N, 2]
    
    Returns:
        ray_o: Ray origins in world space of shape [3], [N, 3], or [M, N, 3]
        ray_d: Ray directions (normalized) in world space of shape [3], [N, 3], or [M, N, 3]
    """
    if isinstance(K, torch.Tensor):
        lib = torch
    else:
        lib = np
    
    ray_o = c2w[:3, 3]
    
    if len(uv.shape) == 1:
        s = 1.0
    else:
        ones_shape = list(uv.shape[:-1])
        if lib == np:
            s = np.ones(ones_shape)
        else:
            s = torch.ones(ones_shape, device=uv.device, dtype=uv.dtype)
    
    x_c = pixel_to_camera(K, uv, s)
    x_w = transform(c2w, x_c)
    
    ray_d_unnormalized = x_w - ray_o
    
    if lib == np:
        if len(ray_d_unnormalized.shape) == 1:
            norm = np.linalg.norm(ray_d_unnormalized)
        else:
            norm = np.linalg.norm(ray_d_unnormalized, axis=-1, keepdims=True)
    else:
        if len(ray_d_unnormalized.shape) == 1:
            norm = torch.norm(ray_d_unnormalized)
        else:
            norm = torch.norm(ray_d_unnormalized, dim=-1, keepdim=True)
    
    ray_d = ray_d_unnormalized / norm
    
    if len(uv.shape) > 1:
        expand_shape = list(ray_d.shape)
        expand_dims = len(expand_shape) - 1
        for _ in range(expand_dims):
            if lib == np:
                ray_o = np.expand_dims(ray_o, 0)
            else:
                ray_o = ray_o.unsqueeze(0)
        
        if lib == np:
            ray_o = np.broadcast_to(ray_o, ray_d.shape)
        else:
            ray_o = ray_o.expand(ray_d.shape)
    
    return ray_o, ray_d


class RaysData:
    """
    Dataloader for rays from multi-view images.
    """
    def __init__(self, images, K, c2ws):
        """
        Args:
            images: Training images of shape [N, H, W, 3]
            K: Camera intrinsic matrix of shape [3, 3]
            c2ws: Camera-to-world matrices of shape [N, 4, 4]
        """
        self.images = images
        self.K = K
        self.c2ws = c2ws
        
        N, H, W, _ = images.shape
        
        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        u = u + 0.5
        v = v + 0.5
        uv_single = np.stack([u.flatten(), v.flatten()], axis=-1)
        
        self.uvs = np.tile(uv_single, (N, 1))
        
        self.pixels = images.reshape(-1, 3)
        all_rays_o = []
        all_rays_d = []
        
        for i in range(N):
            ray_o, ray_d = pixel_to_ray(K, c2ws[i], uv_single)
            all_rays_o.append(ray_o.reshape(-1, 3))
            all_rays_d.append(ray_d.reshape(-1, 3))
        
        self.rays_o = np.concatenate(all_rays_o, axis=0)
        self.rays_d = np.concatenate(all_rays_d, axis=0)
        
        self.num_rays = len(self.rays_o)
        print(f"Created dataset with {self.num_rays:,} rays from {N} images of size {H}x{W}")
    
    def sample_rays(self, batch_size):
        """
        Sample a batch of rays randomly.
        
        Args:
            batch_size: Number of rays to sample
        
        Returns:
            rays_o: Ray origins of shape [batch_size, 3]
            rays_d: Ray directions of shape [batch_size, 3]
            pixels: Corresponding pixel colors of shape [batch_size, 3]
        """
        indices = np.random.choice(self.num_rays, size=batch_size, replace=False)
        
        return self.rays_o[indices], self.rays_d[indices], self.pixels[indices]


def sample_along_rays(rays_o, rays_d, near, far, num_samples, random=True):
    """
    Sample points along rays between near and far planes.
    
    Args:
        rays_o: Ray origins of shape [N, 3]
        rays_d: Ray directions of shape [N, 3]  
        near: Near plane distance
        far: Far plane distance
        num_samples: Number of samples per ray
        random: Whether to add random perturbation
    
    Returns:
        points: Sampled 3D points of shape [N, num_samples, 3]
    """
    if isinstance(rays_o, torch.Tensor):
        lib = torch
        device = rays_o.device
    else:
        lib = np
        device = None
    
    N = rays_o.shape[0]
    
    if lib == np:
        t_vals = np.linspace(near, far, num_samples, dtype=rays_o.dtype)
        t_vals = np.broadcast_to(t_vals, (N, num_samples))
    else:
        t_vals = torch.linspace(near, far, num_samples, device=device, dtype=rays_o.dtype)
        t_vals = t_vals.expand(N, num_samples)
    
    if random:
        bin_size = (far - near) / num_samples
        if lib == np:
            perturbation = np.random.uniform(0, bin_size, (N, num_samples))
            t_vals = t_vals + perturbation
        else:
            perturbation = torch.rand(N, num_samples, device=device, dtype=rays_o.dtype) * bin_size
            t_vals = t_vals + perturbation
    
    if lib == np:
        rays_o_expanded = np.expand_dims(rays_o, 1)  
        rays_d_expanded = np.expand_dims(rays_d, 1)  
        t_vals_expanded = np.expand_dims(t_vals, 2)  
        points = rays_o_expanded + t_vals_expanded * rays_d_expanded
    else:
        rays_o_expanded = rays_o.unsqueeze(1) 
        rays_d_expanded = rays_d.unsqueeze(1)  
        t_vals_expanded = t_vals.unsqueeze(2) 
        points = rays_o_expanded + t_vals_expanded * rays_d_expanded
    
    return points


class PositionalEncoding(nn.Module):
    """
    Positional encoding for NeRF as described in the paper.
    Encodes input coordinates using sinusoidal functions.
    """
    def __init__(self, input_dim, L):
        super(PositionalEncoding, self).__init__()
        self.input_dim = input_dim
        self.L = L
        self.output_dim = input_dim * (2 * L + 1)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (..., input_dim)
        Returns:
            Encoded tensor of shape (..., input_dim * (2*L + 1))
        """
        batch_shape = x.shape[:-1]
        x = x.view(-1, self.input_dim)
        
        encoded = [x]
        
        for i in range(self.L):
            freq = 2.0 ** i
            encoded.append(torch.sin(freq * np.pi * x))
            encoded.append(torch.cos(freq * np.pi * x))
            
        encoded = torch.cat(encoded, dim=-1)
        return encoded.view(*batch_shape, -1)


class NeRFNetwork(nn.Module):
    """
    Neural Radiance Field MLP implementation following the architecture from the paper.
    """
    def __init__(self, coord_L=10, dir_L=4, hidden_dim=256):
        super(NeRFNetwork, self).__init__()
        
        self.coord_pe = PositionalEncoding(3, coord_L)  
        self.dir_pe = PositionalEncoding(3, dir_L)    
        
        self.coord_input_dim = self.coord_pe.output_dim 
        self.dir_input_dim = self.dir_pe.output_dim     
        
        self.coord_layers = nn.ModuleList()
        
        self.coord_layers.append(nn.Linear(self.coord_input_dim, hidden_dim))
        
        for i in range(3):
            self.coord_layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.coord_layers.append(nn.Linear(hidden_dim + self.coord_input_dim, hidden_dim))
        
        for i in range(3):
            self.coord_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.density_head = nn.Linear(hidden_dim, 1)
        
        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)
        

        self.dir_layer1 = nn.Linear(hidden_dim + self.dir_input_dim, 128)
        self.dir_layer2 = nn.Linear(128, 3)  
        
    def forward(self, coords, directions):
        """
        Args:
            coords: 3D coordinates tensor of shape (..., 3)
            directions: 3D direction vectors tensor of shape (..., 3)
        Returns:
            rgb: RGB colors tensor of shape (..., 3)
            density: Density values tensor of shape (..., 1)
        """
        coords_encoded = self.coord_pe(coords)
        dirs_encoded = self.dir_pe(directions)
        
        x = coords_encoded
        
        for i in range(4):
            x = F.relu(self.coord_layers[i](x))
        
        x_skip = torch.cat([x, coords_encoded], dim=-1)
        x = F.relu(self.coord_layers[4](x_skip))
        
        for i in range(5, 8):
            x = F.relu(self.coord_layers[i](x))
        
        density = F.relu(self.density_head(x))
        
        features = self.feature_layer(x)
        
        dir_input = torch.cat([features, dirs_encoded], dim=-1)
        rgb_features = F.relu(self.dir_layer1(dir_input))
        rgb = torch.sigmoid(self.dir_layer2(rgb_features))  
        
        return rgb, density


def volrend(sigmas, rgbs, step_size):
    """
    Volume rendering equation implementation.
    
    Args:
        sigmas: Density values of shape [N, num_samples, 1]
        rgbs: RGB colors of shape [N, num_samples, 3]
        step_size: Step size along the ray
        
    Returns:
        rendered_colors: Rendered RGB colors of shape [N, 3]
    """
    N, num_samples, _ = rgbs.shape
    
    alphas = 1.0 - torch.exp(-sigmas.squeeze(-1) * step_size) 
    eps = 1e-10
    cumprod = torch.cumprod(1.0 - alphas + eps, dim=1)
    ones = torch.ones(N, 1, device=alphas.device, dtype=alphas.dtype)
    transmittance = torch.cat([ones, cumprod[:, :-1]], dim=1)  

    weights = transmittance * alphas  
    rendered_colors = torch.sum(weights.unsqueeze(-1) * rgbs, dim=1) 
    
    return rendered_colors


def render_rays(model, rays_o, rays_d, near, far, num_samples, random=True):
    """
    Render rays using the NeRF model.
    
    Args:
        model: NeRF network
        rays_o: Ray origins of shape [N, 3]
        rays_d: Ray directions of shape [N, 3]
        near: Near plane distance
        far: Far plane distance
        num_samples: Number of samples per ray
        random: Whether to use random sampling
        
    Returns:
        colors: Rendered colors of shape [N, 3]
    """
    points = sample_along_rays(rays_o, rays_d, near, far, num_samples, random) 
    
    N, num_samples, _ = points.shape
    directions = rays_d.unsqueeze(1).expand(-1, num_samples, -1) 
    
    points_flat = points.reshape(-1, 3)  
    directions_flat = directions.reshape(-1, 3)  
    
    rgb_flat, density_flat = model(points_flat, directions_flat)

    rgb = rgb_flat.reshape(N, num_samples, 3)  
    density = density_flat.reshape(N, num_samples, 1)  

    step_size = (far - near) / num_samples
    colors = volrend(density, rgb, step_size)
    
    return colors


def compute_psnr(pred, gt):
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Args:
        pred: Predicted image
        gt: Ground truth image
        
    Returns:
        PSNR value
    """
    mse = torch.mean((pred - gt) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def visualize_rays_and_cameras(dataset, c2ws, num_rays=100,
                               near=2.0, far=6.0, num_samples=64):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    rays_o, rays_d, _ = dataset.sample_rays(num_rays)

    cam_pos = c2ws[:, :3, 3]
    ax.scatter(cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2],
               c="k", s=20, marker="^", label="Cameras")

    ax.scatter(rays_o[:, 0], rays_o[:, 1], rays_o[:, 2],
               c="tab:blue", s=10, alpha=0.8, label="Ray origins")

    import torch
    rays_o_t = torch.from_numpy(rays_o).float()
    rays_d_t = torch.from_numpy(rays_d).float()
    pts = sample_along_rays(rays_o_t, rays_d_t, near, far, num_samples, random=False)
    pts = pts.numpy()        

    for i in range(min(20, num_rays)):
        start = rays_o[i]
        end = start + 2.0 * rays_d[i]
        ax.plot([start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                "gray", linewidth=0.5, alpha=0.5)

        ax.scatter(pts[i, :, 0], pts[i, :, 1], pts[i, :, 2],
                   c="orange", s=3, alpha=0.6)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Camera Positions, Rays, and Samples ({num_rays} rays)")
    ax.legend()

    return fig

def train_nerf(images_train, images_val, c2ws_train, c2ws_val, focal, 
               num_iterations=1000, batch_size=10000, lr=5e-4):
    """
    Train the NeRF model.
    """
    train_losses = []
    train_psnrs  = []
    val_psnrs    = []
    val_iters    = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    H, W = images_train.shape[1], images_train.shape[2]
    K = np.array([
        [focal, 0, W/2],
        [0, focal, H/2],
        [0, 0, 1]
    ])
    
    prog_dir = os.path.join(output_dir, "train_progress")
    os.makedirs(prog_dir, exist_ok=True)

    monitor_c2w = c2ws_val[0]

    train_dataset = RaysData(images_train, K, c2ws_train)
    
    val_rays_o_list = []
    val_rays_d_list = []
    val_pixels_list = []
    
    for i in range(len(images_val)):
        H, W = images_val.shape[1], images_val.shape[2]
        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        u = u + 0.5
        v = v + 0.5
        uv = np.stack([u.flatten(), v.flatten()], axis=-1)
        
        rays_o, rays_d = pixel_to_ray(K, c2ws_val[i], uv)
        val_rays_o_list.append(rays_o)
        val_rays_d_list.append(rays_d)
        val_pixels_list.append(images_val[i].reshape(-1, 3))
    
    val_rays_o = np.concatenate(val_rays_o_list, axis=0)
    val_rays_d = np.concatenate(val_rays_d_list, axis=0)
    val_pixels = np.concatenate(val_pixels_list, axis=0)

    model = NeRFNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    near, far = 2.0, 6.0
    num_samples = 64
    
    for iteration in tqdm(range(num_iterations + 1)): 
        model.train()
        
        rays_o, rays_d, target_pixels = train_dataset.sample_rays(batch_size)
        
        rays_o = torch.tensor(rays_o, dtype=torch.float32, device=device)
        rays_d = torch.tensor(rays_d, dtype=torch.float32, device=device)
        target_pixels = torch.tensor(target_pixels, dtype=torch.float32, device=device)
        
        predicted_colors = render_rays(model, rays_o, rays_d, near, far, num_samples, random=True)
        
        loss = F.mse_loss(predicted_colors, target_pixels)
        train_losses.append(loss.item())
        train_psnr = compute_psnr(predicted_colors.detach(), target_pixels).item()
        train_psnrs.append(train_psnr)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % VAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                val_indices = np.random.choice(len(val_rays_o), size=1000, replace=False)
                val_rays_o_batch = torch.tensor(
                    val_rays_o[val_indices], dtype=torch.float32, device=device
                )
                val_rays_d_batch = torch.tensor(
                    val_rays_d[val_indices], dtype=torch.float32, device=device
                )
                val_pixels_batch = torch.tensor(
                    val_pixels[val_indices], dtype=torch.float32, device=device
                )
                
                val_colors = render_rays(
                    model, val_rays_o_batch, val_rays_d_batch,
                    near, far, num_samples, random=False
                )
                val_psnr = compute_psnr(val_colors, val_pixels_batch).item()
                val_psnrs.append(val_psnr)
                val_iters.append(iteration)
                
                print(
                    f"Iteration {iteration}: "
                    f"Train Loss = {loss.item():.6f}, "
                    f"Train PSNR = {train_psnr:.2f}, "
                    f"Val PSNR = {val_psnr:.2f}"
                )
        
        if iteration in SNAPSHOT_ITERS:
            model.eval()
            with torch.no_grad():
                render_img = render_novel_view(
                    model, monitor_c2w, K, H, W,
                    near=near, far=far, num_samples=num_samples
                )
            img8 = (np.clip(render_img, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(img8).save(
                os.path.join(prog_dir, f"iter_{iteration:04d}.png")
            )
            model.train()

    return model, train_losses, train_psnrs, val_psnrs, val_iters

def render_novel_view(model, c2w, K, H, W, near=2.0, far=6.0, num_samples=64):
    """
    Render a novel view from the trained model.
    """
    device = next(model.parameters()).device
    model.eval()
    
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    uv = np.stack([u.flatten(), v.flatten()], axis=-1)
    
    rays_o, rays_d = pixel_to_ray(K, c2w, uv)
    
    chunk_size = 1024
    rendered_pixels = []
    
    with torch.no_grad():
        for i in range(0, len(rays_o), chunk_size):
            rays_o_chunk = torch.tensor(rays_o[i:i+chunk_size], dtype=torch.float32, device=device)
            rays_d_chunk = torch.tensor(rays_d[i:i+chunk_size], dtype=torch.float32, device=device)
            
            colors_chunk = render_rays(model, rays_o_chunk, rays_d_chunk, near, far, num_samples, random=False)
            rendered_pixels.append(colors_chunk.cpu().numpy())
    
    rendered_pixels = np.concatenate(rendered_pixels, axis=0)
    rendered_image = rendered_pixels.reshape(H, W, 3)
    
    return rendered_image

def render_spherical_video(model, c2ws_test, K, H, W, out_dir, filename="lego_spherical.mp4",
                           near=2.0, far=6.0, num_samples=64, fps=30):
    """
    Render a video over the test camera trajectory.
    Saves individual frames and an mp4.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()

    frames = []

    with torch.no_grad():
        for i in tqdm(range(len(c2ws_test)), desc="Rendering spherical video"):
            c2w = c2ws_test[i]

            img = render_novel_view(model, c2w, K, H, W, near=near, far=far, num_samples=num_samples)
            img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

            frame_path = os.path.join(out_dir, f"frame_{i:04d}.png")
            Image.fromarray(img8).save(frame_path)

            frames.append(img8)

    video_path = os.path.join(out_dir, filename)
    imageio.mimsave(video_path, frames, fps=fps)
    print(f"✓ Saved spherical video to {video_path}")


if __name__ == "__main__":
    print("Testing volume rendering function...")
    torch.manual_seed(42)
    sigmas = torch.rand((10, 64, 1))
    rgbs = torch.rand((10, 64, 3))
    step_size = (6.0 - 2.0) / 64
    rendered_colors = volrend(sigmas, rgbs, step_size)

    correct = torch.tensor([
        [0.5006, 0.3728, 0.4728],
        [0.4322, 0.3559, 0.4134],
        [0.4027, 0.4394, 0.4610],
        [0.4514, 0.3829, 0.4196],
        [0.4002, 0.4599, 0.4103],
        [0.4471, 0.4044, 0.4069],
        [0.4285, 0.4072, 0.3777],
        [0.4152, 0.4190, 0.4361],
        [0.4051, 0.3651, 0.3969],
        [0.3253, 0.3587, 0.4215]
    ])
    assert torch.allclose(rendered_colors, correct, rtol=1e-4, atol=1e-4)
    print("✓ Volume rendering test passed!")
    
    data = np.load('lego_200x200.npz')
    images_train = data['images_train'].astype(np.float32)
    images_val   = data['images_val'].astype(np.float32)
    
    if images_train.max() > 1.0 or images_val.max() > 1.0:
        images_train /= 255.0
        images_val   /= 255.0
    if images_train.shape[-1] == 4:
        images_train = images_train[..., :3]
        images_val   = images_val[..., :3]
    print("Train images shape:", images_train.shape)
    print("Train min/max:", images_train.min(), images_train.max())

    c2ws_train = data['c2ws_train'].astype(np.float32)
    c2ws_val = data['c2ws_val'].astype(np.float32)
    c2ws_test = data['c2ws_test'].astype(np.float32)
    focal = float(data['focal'])
    
    print(f"Loaded dataset:")
    print(f"  Training images: {images_train.shape}")
    print(f"  Validation images: {images_val.shape}")
    print(f"  Test cameras: {c2ws_test.shape}")
    print(f"  Focal length: {focal}")
    
    H, W = images_train.shape[1], images_train.shape[2]
    K = np.array([
        [focal, 0, W/2],
        [0, focal, H/2],
        [0, 0, 1]
    ])
    
    train_dataset = RaysData(images_train, K, c2ws_train)
    uvs_start = 0
    uvs_end = min(40_000, len(train_dataset.uvs))
    sample_uvs = train_dataset.uvs[uvs_start:uvs_end]
    sample_uvs_int = (sample_uvs - 0.5).astype(int)
    assert np.allclose(
        images_train[0, sample_uvs_int[:,1], sample_uvs_int[:,0]], 
        train_dataset.pixels[uvs_start:uvs_end],
        atol=1e-5
    ), "UV coordinates are flipped!"
    print("✓ UV coordinates verified correctly")

    print("\nGenerating visualization of rays and cameras...")
    num_rays_to_show = 100
    ray_indices = np.random.choice(train_dataset.num_rays, num_rays_to_show, replace=False)
    rays_o_vis = train_dataset.rays_o[ray_indices]
    rays_d_vis = train_dataset.rays_d[ray_indices]

    near_vis = 2.0
    far_vis = 6.0
    num_samples_vis = 64

    server = viser.ViserServer(share=True)
    print("✓ Viser server started. Check the URL above to view the visualization.")
    H, W = images_train.shape[1], images_train.shape[2]
    for i, (image, c2w) in enumerate(zip(images_train, c2ws_train)):
        img_rgb = (image * 255).astype(np.uint8)
    
        server.scene.add_camera_frustum(
            f"/cameras/train/{i}",
            fov=2 * np.arctan2(H / 2, focal),
            aspect=W / H,
            scale=0.15,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            image=img_rgb
        )
    for i, (image, c2w) in enumerate(zip(images_val, c2ws_val)):
        img_rgb = (image * 255).astype(np.uint8)
    
        server.scene.add_camera_frustum(
            f"/cameras/val/{i}",
            fov=2 * np.arctan2(H / 2, focal),
            aspect=W / H,
            scale=0.15,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            image=img_rgb
        )
    for i, (ray_o, ray_d) in enumerate(zip(rays_o_vis, rays_d_vis)):
        ray_end = ray_o + ray_d * far_vis
    
        server.scene.add_spline_catmull_rom(
            f"/rays/{i}",
            positions=np.stack([ray_o, ray_end], axis=0),
            color=(0, 255, 0),
            line_width=1.0
        )
    
    all_sample_points = []
    for ray_o, ray_d in zip(rays_o_vis, rays_d_vis):
        t_vals = np.linspace(near_vis, far_vis, num_samples_vis)
        points = ray_o[None, :] + ray_d[None, :] * t_vals[:, None]
        all_sample_points.append(points)

    all_sample_points = np.concatenate(all_sample_points, axis=0)
    server.scene.add_point_cloud(
        "/sample_points",
        points=all_sample_points,
        colors=np.tile([255, 0, 0], (len(all_sample_points), 1)),  
        point_size=0.002
    )

    print("\nVisualization is ready! Keep this script running to view in browser.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n✓ Closing visualization and continuing...")

    NUM_ITERS = 1000
    model, train_losses, train_psnrs, val_psnrs, val_iters = train_nerf(
        images_train, images_val, c2ws_train, c2ws_val, focal,
        num_iterations=NUM_ITERS, batch_size = 10000, lr=5e-4
    )

    print("\nGenerating training curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Train Loss')
    ax1.set_title('Training Loss')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    if val_psnrs:
        ax2.plot(val_iters, val_psnrs, 'o-', markersize=4)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('PSNR (dB)')
        ax2.set_title('Validation PSNR')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    curves_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training curves to {curves_path}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(train_psnrs, label='Training PSNR', alpha=0.7, linewidth=0.5)
    ax.plot(val_iters, val_psnrs, 'o-', label='Validation PSNR', 
            linewidth=2, markersize=6)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('Training and Validation PSNR', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    combined_psnr_path = os.path.join(output_dir, "combined_psnr.png")
    plt.savefig(combined_psnr_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved combined PSNR curve to {combined_psnr_path}")
    
    print("\nCreating progression figure for Val 0...")
    prog_dir = os.path.join(output_dir, "train_progress")
    iters_to_show = SNAPSHOT_ITERS
    print("Progress iters to show:", iters_to_show)

    gt0 = images_val[0]

    prog_images = []
    for it in iters_to_show:
        frame_path = os.path.join(prog_dir, f"iter_{it:04d}.png")
        if not os.path.exists(frame_path):
            print(f"  [warn] {frame_path} not found, skipping")
            continue
        img = imageio.imread(frame_path) / 255.0
        prog_images.append((it, img))

    num_cols = 1 + len(prog_images)
    fig, axes = plt.subplots(1, num_cols, figsize=(3 * num_cols, 3))

    axes[0].imshow(gt0)
    axes[0].set_title("GT Val 0")
    axes[0].axis("off")

    for j, (it, img) in enumerate(prog_images):
        axes[j + 1].imshow(img)
        axes[j + 1].set_title(f"Iter {it}")
        axes[j + 1].axis("off")

    plt.tight_layout()
    prog_fig_path = os.path.join(output_dir, "val0_progression.png")
    plt.savefig(prog_fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Val 0 progression figure to {prog_fig_path}")

    print("\nRendering validation images...")
    val_images_rendered = []
    for i in range(len(images_val)):
        rendered_img = render_novel_view(model, c2ws_val[i], K, H, W)
        val_images_rendered.append(rendered_img)
    
    fig, axes = plt.subplots(2, len(images_val), figsize=(3*len(images_val), 6))
    for i in range(len(images_val)):
        axes[0, i].imshow(images_val[i])
        axes[0, i].set_title(f'GT Val {i}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(np.clip(val_images_rendered[i], 0, 1))
        axes[1, i].set_title(f'Rendered Val {i}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    val_comp_path = os.path.join(output_dir, "validation_comparison.png")
    plt.savefig(val_comp_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved validation comparison to {val_comp_path}")
    
    if val_psnrs:
        final_psnr = val_psnrs[-1]
        print(f"\nFinal validation PSNR: {final_psnr:.2f} dB")
        if final_psnr >= 23.0:
            print("✓ Achieved target PSNR of 23+ dB!")
        else:
            print("Target PSNR not reached. Consider training longer or adjusting hyperparameters.")

    model_path = os.path.join(output_dir, "nerf_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✓ Saved trained model to {model_path}")

    print("\nRendering spherical video...")
    spherical_dir = os.path.join(output_dir, "spherical")
    render_spherical_video(
        model, c2ws_test, K, H, W, 
        out_dir=spherical_dir, 
        filename="lego_spherical.mp4",
        near=2.0, far=6.0, num_samples=64, fps=30
    )
    
