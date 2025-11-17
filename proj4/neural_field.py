import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding"""
    def __init__(self, input_dim=2, max_freq_log2=10):
        super().__init__()
        self.input_dim = input_dim
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = max_freq_log2 + 1
        
        # Calculate output dimension: original input + sin/cos for each frequency
        self.output_dim = input_dim + 2 * input_dim * self.num_freqs
        
    def forward(self, x):
        """
        Apply sinusoidal positional encoding
        Args:
            x: input coordinates of shape (batch_size, input_dim)
        Returns:
            encoded coordinates of shape (batch_size, output_dim)
        """
        # Start with the original input
        encoded = [x]
        
        # Apply sinusoidal functions at different frequencies
        for i in range(self.num_freqs):
            freq = 2.0 ** i
            encoded.append(torch.sin(freq * np.pi * x))
            encoded.append(torch.cos(freq * np.pi * x))
        
        return torch.cat(encoded, dim=-1)


class NeuralField2D(nn.Module):
    """MLP-based Neural Field for 2D image representation"""
    def __init__(self, max_freq_log2=10, hidden_dim=256, num_hidden_layers=4):
        super().__init__()
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(input_dim=2, max_freq_log2=max_freq_log2)
        
        # Build MLP
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.pos_encoding.output_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 3))  # Output RGB
        layers.append(nn.Sigmoid())  # Constrain output to [0, 1]
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, coords):
        """
        Forward pass
        Args:
            coords: pixel coordinates of shape (batch_size, 2)
        Returns:
            rgb colors of shape (batch_size, 3)
        """
        encoded = self.pos_encoding(coords)
        rgb = self.mlp(encoded)
        return rgb


class ImageDataloader:
    """Dataloader for randomly sampling pixels from an image"""
    def __init__(self, image_path, batch_size=10000):
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        self.height, self.width = img_array.shape[:2]
        self.batch_size = batch_size
        
        # Create coordinate grid
        y_coords, x_coords = np.meshgrid(
            np.arange(self.height),
            np.arange(self.width),
            indexing='ij'
        )
        
        # Flatten and normalize coordinates to [0, 1]
        self.coords = np.stack([
            x_coords.flatten() / self.width,
            y_coords.flatten() / self.height
        ], axis=-1).astype(np.float32)
        
        # Flatten and normalize colors to [0, 1]
        self.colors = img_array.reshape(-1, 3).astype(np.float32) / 255.0
        
        self.num_pixels = len(self.coords)
        
    def get_batch(self):
        """Randomly sample a batch of pixels"""
        indices = np.random.randint(0, self.num_pixels, size=self.batch_size)
        coords_batch = self.coords[indices]
        colors_batch = self.colors[indices]
        
        return torch.from_numpy(coords_batch), torch.from_numpy(colors_batch)
    
    def get_all_data(self):
        """Get all pixels (for evaluation)"""
        return torch.from_numpy(self.coords), torch.from_numpy(self.colors)


def compute_psnr(mse):
    """Compute PSNR from MSE"""
    return -10.0 * torch.log10(mse)


def train_neural_field(image_path, max_freq_log2=10, hidden_dim=256, 
                       num_hidden_layers=4, lr=1e-2, num_iterations=2000,
                       batch_size=10000, eval_interval=100, device='cuda',
                       snapshot_iters=None, snapshot_dir=None):
    """
    Train a neural field to fit a 2D image
    """
    # Setup
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create snapshot directory if needed
    if snapshot_iters is not None and snapshot_dir is not None:
        os.makedirs(snapshot_dir, exist_ok=True)
    
    # Create dataloader
    dataloader = ImageDataloader(image_path, batch_size=batch_size)
    
    # Create model
    model = NeuralField2D(
        max_freq_log2=max_freq_log2,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers
    ).to(device)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training history
    history = {
        'iterations': [],
        'train_psnr': [],
        'eval_psnr': [],
        'train_loss': [],
        'snapshots': {}
    }
    
    # Get all data for evaluation
    all_coords, all_colors = dataloader.get_all_data()
    all_coords = all_coords.to(device)
    all_colors = all_colors.to(device)
    
    # Save snapshot at iteration 0 if requested
    if snapshot_iters is not None and 0 in snapshot_iters:
        snapshot_img = render_image(model, dataloader, device=device)
        history['snapshots'][0] = snapshot_img
        if snapshot_dir is not None:
            snapshot_img.save(os.path.join(snapshot_dir, f"iter_0000.png"))
    
    # Training loop
    print(f"\nTraining for {num_iterations} iterations...")
    progress_bar = tqdm(range(num_iterations))
    
    for iteration in progress_bar:
        # Get training batch
        coords_batch, colors_batch = dataloader.get_batch()
        coords_batch = coords_batch.to(device)
        colors_batch = colors_batch.to(device)
        
        # Forward pass
        predicted_colors = model(coords_batch)
        loss = criterion(predicted_colors, colors_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute PSNR
        train_psnr = compute_psnr(loss)
        
        # Update progress bar
        progress_bar.set_description(
            f"Loss: {loss.item():.6f}, PSNR: {train_psnr.item():.2f} dB"
        )
        
        # Evaluate on full image periodically
        if (iteration + 1) % eval_interval == 0 or iteration == 0:
            model.eval()
            with torch.no_grad():
                # Evaluate in chunks to avoid memory issues
                chunk_size = 50000
                all_predictions = []
                for i in range(0, len(all_coords), chunk_size):
                    chunk_coords = all_coords[i:i+chunk_size]
                    chunk_pred = model(chunk_coords)
                    all_predictions.append(chunk_pred)
                
                all_predictions = torch.cat(all_predictions, dim=0)
                eval_loss = criterion(all_predictions, all_colors)
                eval_psnr = compute_psnr(eval_loss)
            
            model.train()
            
            # Record history
            history['iterations'].append(iteration + 1)
            history['train_psnr'].append(train_psnr.item())
            history['eval_psnr'].append(eval_psnr.item())
            history['train_loss'].append(loss.item())
        
        # Save snapshot if requested
        if snapshot_iters is not None and (iteration + 1) in snapshot_iters:
            snapshot_img = render_image(model, dataloader, device=device)
            history['snapshots'][iteration + 1] = snapshot_img
            if snapshot_dir is not None:
                snapshot_img.save(os.path.join(snapshot_dir, f"iter_{iteration+1:04d}.png"))
    
    print(f"\nTraining completed! Final PSNR: {eval_psnr.item():.2f} dB")
    
    return model, dataloader, history


def render_image(model, dataloader, device='cuda'):
    """Render the full image using the trained model"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_coords, _ = dataloader.get_all_data()
    all_coords = all_coords.to(device)
    
    # Render in chunks
    chunk_size = 50000
    all_predictions = []
    
    with torch.no_grad():
        for i in range(0, len(all_coords), chunk_size):
            chunk_coords = all_coords[i:i+chunk_size]
            chunk_pred = model(chunk_coords)
            all_predictions.append(chunk_pred.cpu())
    
    all_predictions = torch.cat(all_predictions, dim=0)
    
    # Reshape to image
    img_array = all_predictions.numpy().reshape(dataloader.height, dataloader.width, 3)
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array)


def visualize_training_progression(image_path, output_dir, max_freq_log2=10, 
                                   hidden_dim=256, num_iterations=2000):
    """Train and visualize the progression"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Train model
    model, dataloader, history = train_neural_field(
        image_path=image_path,
        max_freq_log2=max_freq_log2,
        hidden_dim=hidden_dim,
        num_iterations=num_iterations,
        eval_interval=50
    )
    
    # Load original image for comparison
    original_img = Image.open(image_path).convert('RGB')
    
    # Create progression visualization at specific iterations
    eval_iterations = history['iterations']
    num_snapshots = min(6, len(eval_iterations))
    snapshot_indices = np.linspace(0, len(eval_iterations)-1, num_snapshots, dtype=int)
    
    # We'll need to re-train to get intermediate snapshots
    # For now, just show final result
    final_img = render_image(model, dataloader)
    
    return model, dataloader, history, original_img, final_img


def hyperparameter_comparison(image_path, output_dir):
    """Compare different hyperparameter settings"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Test different configurations
    configs = [
        {'max_freq_log2': 2, 'hidden_dim': 64, 'name': 'Low Freq (2), Narrow (64)'},
        {'max_freq_log2': 2, 'hidden_dim': 256, 'name': 'Low Freq (2), Wide (256)'},
        {'max_freq_log2': 10, 'hidden_dim': 64, 'name': 'High Freq (10), Narrow (64)'},
        {'max_freq_log2': 10, 'hidden_dim': 256, 'name': 'High Freq (10), Wide (256)'},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Training with: {config['name']}")
        print(f"{'='*60}")
        
        model, dataloader, history = train_neural_field(
            image_path=image_path,
            max_freq_log2=config['max_freq_log2'],
            hidden_dim=config['hidden_dim'],
            num_iterations=2000,
            eval_interval=100
        )
        
        final_img = render_image(model, dataloader)
        final_psnr = history['eval_psnr'][-1]
        
        results.append({
            'config': config,
            'image': final_img,
            'psnr': final_psnr,
            'history': history
        })
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, result in enumerate(results):
        axes[idx].imshow(result['image'])
        axes[idx].set_title(f"{result['config']['name']}\nPSNR: {result['psnr']:.2f} dB", 
                           fontsize=10)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hyperparameter_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return results


def plot_psnr_curve(history, output_path):
    """Plot PSNR over training iterations"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['iterations'], history['eval_psnr'], linewidth=2, marker='o', markersize=4)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('Training Progress: PSNR over Iterations', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Create output directory
    output_dir = f"outputs_part1_{tag}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("2D Neural Field Implementation")
    print("="*60)
    
    # You'll need to provide an image path
    print("\nPlease upload an image to use for training.")
    print("The script is ready to run once you provide an image path.")