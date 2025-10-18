import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.ndimage import gaussian_filter
import os

from harris import get_harris_corners, adaptive_non_maximal_suppression

def extract_feature_descriptor(image, x, y, window_size=40, descriptor_size=8, sample_spacing=5):
    """
    Extract an 8x8 feature descriptor from a 40x40 window around a keypoint.
    """
    h, w = image.shape
    half_window = window_size // 2
    if (x - half_window < 0 or x + half_window >= w or 
        y - half_window < 0 or y + half_window >= h):
        return None, False
    x_int, y_int = int(round(x)), int(round(y))
    window = image[y_int - half_window:y_int + half_window,
                   x_int - half_window:x_int + half_window]
    blurred_window = gaussian_filter(window.astype(float), sigma=sample_spacing/2.0)
    descriptor = np.zeros((descriptor_size, descriptor_size))
    for i in range(descriptor_size):
        for j in range(descriptor_size):
            sample_y = i * sample_spacing + (window_size - (descriptor_size-1)*sample_spacing) // 2
            sample_x = j * sample_spacing + (window_size - (descriptor_size-1)*sample_spacing) // 2
            if 0 <= sample_y < window_size and 0 <= sample_x < window_size:
                descriptor[i, j] = blurred_window[sample_y, sample_x]
    descriptor = descriptor.flatten()
    mean = np.mean(descriptor)
    std = np.std(descriptor)
    if std > 1e-10:
        descriptor = (descriptor - mean) / std
    else:
        return None, False
    return descriptor, True

def extract_descriptors_for_corners(image, corners, window_size=40, 
                                   descriptor_size=8, sample_spacing=5):
    """
    Extract feature descriptors for all corner points.
    """
    n_corners = corners.shape[1]
    descriptors = []
    valid_indices = []
    
    for i in range(n_corners):
        y, x = corners[0, i], corners[1, i]
        descriptor, valid = extract_feature_descriptor(
            image, x, y, window_size, descriptor_size, sample_spacing
        )
        
        if valid:
            descriptors.append(descriptor)
            valid_indices.append(i)
    
    if len(descriptors) == 0:
        return np.array([]), corners[:, :0], np.array([])
    
    descriptors = np.array(descriptors)
    valid_indices = np.array(valid_indices)
    valid_corners = corners[:, valid_indices]
    
    return descriptors, valid_corners, valid_indices

def visualize_feature_descriptors(image, corners, descriptors, num_features=9,  save_path_locations=None, save_path_grid=None):
    """
    Visualize several extracted feature descriptors with their locations.
    """
    n_features = min(num_features, corners.shape[1])
    indices = np.linspace(0, corners.shape[1] - 1, n_features, dtype=int)
    fig = plt.figure(figsize=(16, 10))
    ax_img = plt.subplot(2, 1, 1)

    if len(image.shape) == 2:
        ax_img.imshow(image, cmap="gray")
    else:
        ax_img.imshow(image)
         
    window_size = 40
    half_window = window_size // 2
    colors = plt.cm.tab10(np.arange(n_features))
    
    for idx, i in enumerate(indices):
        y, x = corners[0, i], corners[1, i]
        rect = plt.Rectangle(
            (x - half_window, y - half_window),
            window_size,
            window_size,
            fill=False,
            edgecolor=colors[idx],
            linewidth=2.5,
        )
        ax_img.add_patch(rect)
        ax_img.plot(x, y, "o", color=colors[idx], markersize=10,
                    markeredgewidth=2, markerfacecolor="yellow")
        ax_img.text(x + half_window + 5, y - half_window, f"{idx+1}",
                    color="white", fontsize=14, fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor=colors[idx], alpha=0.8))
    ax_img.set_title("Feature Locations with 40×40 Extraction Windows",
                     fontsize=14, fontweight="bold")

    ax_img.axis("off")
    if save_path_locations:
        fig.savefig(save_path_locations, dpi=150, bbox_inches="tight")
        print(f"Saved image visualization to {save_path_locations}")

    grid_size = int(np.ceil(np.sqrt(n_features)))
    nrows = grid_size
    ncols = grid_size
    fig2, axes = plt.subplots(nrows, ncols, figsize=(12, 12))
    axes = np.array(axes).reshape(-1)

    for plot_idx, i in enumerate(indices):
        y, x = corners[0, i], corners[1, i]
        descriptor = descriptors[i].reshape(8, 8)
        ax = axes[plot_idx]
        im = ax.imshow(descriptor, cmap="RdBu_r", interpolation="nearest")

        for spine in ax.spines.values():
            spine.set_edgecolor(colors[plot_idx])
            spine.set_linewidth(2)
        ax.set_title(f"{plot_idx+1}: ({int(x)}, {int(y)})",
                     fontsize=10, fontweight="bold", color=colors[plot_idx])

        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[n_features:]:
        ax.axis("off")

    fig2.suptitle("B.2: Extracted 8×8 Feature Descriptors (Bias/Gain Normalized)",
                  fontsize=16, fontweight="bold", y=0.98)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path_grid:
        fig2.savefig(save_path_grid, dpi=150, bbox_inches="tight")
        print(f"Saved descriptor grid to {save_path_grid}")
    plt.show()

if __name__ == "__main__":
    image_list = ["doe1.png", "mochi1.png"]
    
    os.makedirs("results", exist_ok=True)

    for image_path in image_list:
        prefix = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\n Processing {prefix}\n")

        image = io.imread(image_path)
        if image.shape[-1] == 4:
            image = image[..., :3]
        gray = color.rgb2gray(image)
        
        # Step 1: Get Harris corners
        h, corners_all = get_harris_corners(gray, edge_discard=20)
        
        # Step 2: Apply ANMS
        corners_anms = adaptive_non_maximal_suppression(corners_all, h, num_corners=500)
        
        # Step 3: Extract feature descriptors
        descriptors, valid_corners, _ = extract_descriptors_for_corners(
            gray, corners_anms, window_size=40, descriptor_size=8, sample_spacing=5
        )

        print("Step 4: Visualizing feature descriptors")
        visualize_feature_descriptors(
            image, valid_corners, descriptors, 
            num_features=9,
            save_path_locations=f"results/{prefix}_feature_locations.png",
            save_path_grid=f"results/{prefix}_descriptor_grid.png"
        )
    
    print("\nAll images processed successfully!")