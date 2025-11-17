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
import cv2
import glob

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

output_dir = "outputs_object_nerf_fixed"
os.makedirs(output_dir, exist_ok=True)
SNAPSHOT_ITERS = [0, 500, 1000, 2000, 3000, 4000, 5000]  
VAL_EVERY = 100  
PROG_SAVE_EVERY = 500


def load_image(image_path):
    """Load an image from various formats including HEIC."""
    try:
        if HEIC_SUPPORT and image_path.lower().endswith((".heic", ".heif")):
            pil_image = Image.open(image_path)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return image
        else:
            return cv2.imread(image_path)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


def transform(c2w, x_c):
    """Transform points from camera space to world space."""
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
    """Transform points from pixel coordinates to camera space."""
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
    """Convert pixel coordinates to rays with origin and direction."""
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
    """Dataloader for rays from multi-view images."""
    def __init__(self, images, K, c2ws):
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
            all_rays_o.append(ray_o)
            all_rays_d.append(ray_d)
        
        self.rays_o = np.concatenate(all_rays_o, axis=0)
        self.rays_d = np.concatenate(all_rays_d, axis=0)
        
    def __len__(self):
        return len(self.pixels)


class NeRFModel(nn.Module):
    """NeRF MLP with skip connection."""
    def __init__(self, pos_enc_L=6, hidden_dim=128):
        super().__init__()
        self.pos_enc_L = pos_enc_L
        input_dim = 3 + 3 * 2 * pos_enc_L
        
        self.layers1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.layers2 = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
        
    def positional_encoding(self, x):
        """Apply positional encoding."""
        encoded = [x]
        for l in range(self.pos_enc_L):
            encoded.append(torch.sin(2**l * np.pi * x))
            encoded.append(torch.cos(2**l * np.pi * x))
        return torch.cat(encoded, dim=-1)
    
    def forward(self, x):
        x_encoded = self.positional_encoding(x)
        h = self.layers1(x_encoded)
        h = torch.cat([h, x_encoded], dim=-1)
        output = self.layers2(h)
        rgb = torch.sigmoid(output[..., :3])
        sigma = F.relu(output[..., 3:4])
        return rgb, sigma


def volrend(sigmas, rgbs, step_size):
    """Volume rendering."""
    deltas = step_size
    alphas = 1.0 - torch.exp(-sigmas * deltas)
    
    T = torch.cumprod(1.0 - alphas + 1e-10, dim=-2)
    T = torch.cat([torch.ones_like(T[..., :1, :]), T[..., :-1, :]], dim=-2)
    
    weights = alphas * T
    rendered_colors = torch.sum(weights * rgbs, dim=-2)
    
    return rendered_colors


def render_rays(model, rays_o, rays_d, near, far, num_samples=64):
    """Render colors for a batch of rays."""
    device = rays_o.device
    batch_size = rays_o.shape[0]
    
    t_vals = torch.linspace(near, far, num_samples, device=device)
    t_vals = t_vals.expand(batch_size, num_samples)
    
    pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_vals.unsqueeze(2)
    
    pts_flat = pts.reshape(-1, 3)
    rgbs_flat, sigmas_flat = model(pts_flat)
    
    rgbs = rgbs_flat.reshape(batch_size, num_samples, 3)
    sigmas = sigmas_flat.reshape(batch_size, num_samples, 1)
    
    step_size = (far - near) / num_samples
    rendered_colors = volrend(sigmas, rgbs, step_size)
    
    return rendered_colors


def train_nerf(images_train, images_val, c2ws_train, c2ws_val, K, 
               num_iterations=5000, batch_size=4096, lr=5e-4, 
               near=0.05, far=0.5, num_samples=192):
    """Train NeRF model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = NeRFModel(pos_enc_L=6, hidden_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    
    train_dataset = RaysData(images_train, K, c2ws_train)
    
    train_losses = []
    train_psnrs = []
    val_psnrs = []
    val_iters = []
    
    H, W = images_train.shape[1], images_train.shape[2]
    prog_dir = os.path.join(output_dir, "train_progress")
    os.makedirs(prog_dir, exist_ok=True)
    
    best_val_psnr = 0
    best_model_state = None
    
    pbar = tqdm(range(num_iterations), desc="Training NeRF")
    for iteration in pbar:
        model.train()
        
        indices = np.random.choice(len(train_dataset), batch_size, replace=False)
        rays_o_batch = torch.from_numpy(train_dataset.rays_o[indices]).float().to(device)
        rays_d_batch = torch.from_numpy(train_dataset.rays_d[indices]).float().to(device)
        pixels_batch = torch.from_numpy(train_dataset.pixels[indices]).float().to(device)
        
        rendered_colors = render_rays(model, rays_o_batch, rays_d_batch, near, far, num_samples)
        
        loss = F.mse_loss(rendered_colors, pixels_batch)
        l2_reg = 0.0
        for param in model.parameters():
          l2_reg += torch.norm(param)
          loss = loss + 1e-5 * l2_reg
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_losses.append(loss.item())
        psnr = -10.0 * torch.log10(loss).item()
        train_psnrs.append(psnr)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'psnr': f'{psnr:.2f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'})
        
        if (iteration + 1) % VAL_EVERY == 0 or iteration == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for i in range(len(images_val)):
                    rendered_img = render_novel_view(model, c2ws_val[i], K, H, W, 
                                                     near, far, num_samples, device)
                    mse = np.mean((rendered_img - images_val[i])**2)
                    val_loss += mse
                
                val_loss /= len(images_val)
                val_psnr = -10.0 * np.log10(val_loss)
                val_psnrs.append(val_psnr)
                val_iters.append(iteration)
                
                if val_psnr > best_val_psnr:
                    best_val_psnr = val_psnr
                    best_model_state = model.state_dict().copy()
                
                print(f"\nIter {iteration}: Val PSNR = {val_psnr:.2f} dB (Best: {best_val_psnr:.2f} dB)")
        
        if iteration in SNAPSHOT_ITERS:
            model.eval()
            with torch.no_grad():
                rendered_img = render_novel_view(model, c2ws_val[0], K, H, W, 
                                                near, far, num_samples, device)
                img8 = (np.clip(rendered_img, 0, 1) * 255).astype(np.uint8)
                frame_path = os.path.join(prog_dir, f"iter_{iteration:04d}.png")
                Image.fromarray(img8).save(frame_path)
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with validation PSNR: {best_val_psnr:.2f} dB")
    
    return model, train_losses, train_psnrs, val_psnrs, val_iters


def render_novel_view(model, c2w, K, H, W, near=0.05, far=0.5, num_samples=64, device=None):
    """Render a novel view."""
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    u = u + 0.5
    v = v + 0.5
    uv = np.stack([u.flatten(), v.flatten()], axis=-1)
    
    rays_o, rays_d = pixel_to_ray(K, c2w, uv)
    rays_o = torch.from_numpy(rays_o).float().to(device)
    rays_d = torch.from_numpy(rays_d).float().to(device)
    
    chunk_size = 4096
    rendered_pixels = []
    
    for i in range(0, len(rays_o), chunk_size):
        rays_o_chunk = rays_o[i:i+chunk_size]
        rays_d_chunk = rays_d[i:i+chunk_size]
        
        with torch.no_grad():
            rendered_chunk = render_rays(model, rays_o_chunk, rays_d_chunk, near, far, num_samples)
        
        rendered_pixels.append(rendered_chunk.cpu().numpy())
    
    rendered_pixels = np.concatenate(rendered_pixels, axis=0)
    rendered_image = rendered_pixels.reshape(H, W, 3)
    
    return rendered_image


def generate_circular_cameras(center, radius, num_views=60, height=0.0):
    """Generate camera poses in a circle around the object."""
    c2ws = []
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        
        cam_pos = center + np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            height
        ])
        
        forward = center - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        up = np.array([0, 0, 1])
        
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)
        
        up = np.cross(forward, right)
        
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        c2w[:3, 3] = cam_pos
        
        c2ws.append(c2w)
    
    return np.array(c2ws)


def render_spherical_video(model, c2ws_test, K, H, W, out_dir, filename="spherical_video.mp4",
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

            img = render_novel_view(model, c2w, K, H, W, near=near, far=far, num_samples=num_samples, device=device)
            img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

            frame_path = os.path.join(out_dir, f"frame_{i:04d}.png")
            Image.fromarray(img8).save(frame_path)

            frames.append(img8)

    video_path = os.path.join(out_dir, filename)
    imageio.mimsave(video_path, frames, fps=fps)
    print(f"✓ Saved spherical video to {video_path}")


def load_object_dataset(images_folder, calib_path, target_size=None):
    """Load object scan dataset with camera poses."""
    calib_data = np.load(calib_path)
    camera_matrix = calib_data['camera_matrix']
    dist_coeffs = calib_data['dist_coeffs']
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    marker_size_meters = 0.02
    marker_3d_points = np.array([
        [0, 0, 0],
        [marker_size_meters, 0, 0],
        [marker_size_meters, marker_size_meters, 0],
        [0, marker_size_meters, 0]
    ], dtype=np.float32)
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.heic', '*.HEIC']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_folder, ext)))
    
    image_files.sort()
    
    images = []
    c2ws = []
    
    for image_path in tqdm(image_files, desc="Loading images and poses"):
        image = load_image(image_path)
        if image is None:
            continue
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            corner_2d = corners[0].reshape(4, 2).astype(np.float32)
            
            success, rvec, tvec = cv2.solvePnP(
                marker_3d_points, corner_2d, camera_matrix, dist_coeffs
            )
            
            if success:
                R, _ = cv2.Rodrigues(rvec)
                
                w2c = np.eye(4)
                w2c[:3, :3] = R
                w2c[:3, 3] = tvec.squeeze()
                
                c2w = np.linalg.inv(w2c)
                
                if target_size is not None:
                    image = cv2.resize(image, target_size)
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
                
                images.append(image_rgb)
                c2ws.append(c2w)
    
    images = np.array(images)
    c2ws = np.array(c2ws)
    
    if target_size is not None:
        orig_h, orig_w = gray.shape
        scale_x = target_size[0] / orig_w
        scale_y = target_size[1] / orig_h
        
        camera_matrix = camera_matrix.copy()
        camera_matrix[0, 0] *= scale_x
        camera_matrix[1, 1] *= scale_y
        camera_matrix[0, 2] *= scale_x
        camera_matrix[1, 2] *= scale_y
    
    return images, c2ws, camera_matrix


if __name__ == "__main__":
    print("Loading object dataset...")
    
    images_folder = "./tags1" 
    calib_path = "camera_calibration.npz"
    
    target_size = (200, 200)
    
    images, c2ws, camera_matrix = load_object_dataset(
        images_folder, calib_path, target_size
    )
    
    print(f"Loaded {len(images)} images with poses")
    print(f"Image shape: {images.shape}")
    print(f"Camera matrix:\n{camera_matrix}")
    
    num_train = int(0.9 * len(images))
    indices = np.arange(len(images))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    images_train = images[train_indices]
    c2ws_train = c2ws[train_indices]
    images_val = images[val_indices]
    c2ws_val = c2ws[val_indices]
    
    print(f"Training images: {len(images_train)}")
    print(f"Validation images: {len(images_val)}")
    
    cam_positions = c2ws[:, :3, 3]
    center = np.mean(cam_positions, axis=0)
    distances = np.linalg.norm(cam_positions - center, axis=1)
    avg_distance = np.mean(distances)
    
    near = 0.13 
    far = 1.1  
    
    print(f"Scene center: {center}")
    print(f"Average camera distance: {avg_distance:.3f}")
    print(f"Using MANUAL near/far: near={near:.3f}, far={far:.3f}")
    
    # Train NeRF
    print("\nTraining NeRF...")
    model, train_losses, train_psnrs, val_psnrs, val_iters = train_nerf(
        images_train, images_val, c2ws_train, c2ws_val, camera_matrix,
        num_iterations=5000, batch_size=4096, lr=5e-4,
        near=near, far=far, num_samples=192
    )
    
    # Plot training curves
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
    
    print("\nCreating progression figure...")
    prog_dir = os.path.join(output_dir, "train_progress")
    
    gt0 = images_val[0]
    prog_images = []
    for it in SNAPSHOT_ITERS:
        frame_path = os.path.join(prog_dir, f"iter_{it:04d}.png")
        if os.path.exists(frame_path):
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
    print(f"✓ Saved progression figure to {prog_fig_path}")
    
    print("\nGenerating circular trajectory...")
    radius = avg_distance * 0.9
    c2ws_test = generate_circular_cameras(center, radius, num_views=60, height=0.0)
    
    print("\nRendering circular GIF...")
    H, W = images_train.shape[1], images_train.shape[2]
    device = next(model.parameters()).device
    
    frames = []
    for i in tqdm(range(len(c2ws_test)), desc="Rendering frames"):
        rendered_img = render_novel_view(model, c2ws_test[i], camera_matrix, H, W, 
                                        near, far, 192, device)
        img8 = (np.clip(rendered_img, 0, 1) * 255).astype(np.uint8)
        frames.append(img8)
    
    gif_path = os.path.join(output_dir, "object_rotation.gif")
    imageio.mimsave(gif_path, frames, fps=30, loop=0)
    print(f"✓ Saved rotation GIF to {gif_path}")
    
    print("\nRendering spherical video...")
    spherical_dir = os.path.join(output_dir, "spherical")
    render_spherical_video(
        model, c2ws_test, camera_matrix, H, W, 
        out_dir=spherical_dir, 
        filename="object_spherical.mp4",
        near=near, far=far, num_samples=192, fps=30
    )
    
    model_path = os.path.join(output_dir, "nerf_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✓ Saved trained model to {model_path}")
    
    print("\n=== Training Complete ===")
    if val_psnrs:
        print(f"Final validation PSNR: {val_psnrs[-1]:.2f} dB")