import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import corner_harris, peak_local_max
from skimage import io, color
from scipy.spatial.distance import cdist
import os

def get_harris_corners(im, edge_discard=20):
    """
    Detect Harris corners in a grayscale image.
    
    """
    assert edge_discard >= 20, "edge_discard must be at least 20"
    
    h = corner_harris(im, method='eps', sigma=1)
    
    coords = peak_local_max(h, min_distance=1)
    
    edge = edge_discard
    mask = (coords[:, 0] > edge) & \
           (coords[:, 0] < im.shape[0] - edge) & \
           (coords[:, 1] > edge) & \
           (coords[:, 1] < im.shape[1] - edge)
    coords = coords[mask].T
    
    return h, coords


def adaptive_non_maximal_suppression(corners, harris_response, num_corners=500):
    """
    Perform Adaptive Non-Maximal Suppression (ANMS) on detected corners.
    """
    n = corners.shape[1]
    
    if n <= num_corners:
        return corners
    
    strengths = harris_response[corners[0, :], corners[1, :]]
    
    radii = np.full(n, np.inf)
    
    for i in range(n):
        stronger_mask = strengths > strengths[i]
        
        if np.any(stronger_mask):
            stronger_coords = corners[:, stronger_mask].T
            current_coord = corners[:, i].reshape(1, -1)
            
            distances = cdist(current_coord, stronger_coords, metric='euclidean')
            radii[i] = np.min(distances)
    
    sorted_indices = np.argsort(-radii) 
    selected_indices = sorted_indices[:num_corners]
    
    return corners[:, selected_indices]


def visualize_corners(image, corners, title="Detected Corners", save_path=None):
    """
    Visualize detected corners overlaid on the image.
    """
    plt.figure(figsize=(12, 8))
    
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    
    plt.plot(corners[1, :], corners[0, :], 'r+', markersize=8, markeredgewidth=2)
    plt.plot(corners[1, :], corners[0, :], 'yo', markersize=6, 
             markerfacecolor='none', markeredgewidth=1.5)
    
    plt.title(f'{title} ({corners.shape[1]} corners)', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def compare_harris_anms(image, num_anms_corners=500, edge_discard=20, save_prefix=None):
    """
    Compare Harris corner detection with and without ANMS.
    
    """
    
    if image.shape[-1] == 4:
        image = image[..., :3]
    gray = color.rgb2gray(image)
    
    h, corners_all = get_harris_corners(gray, edge_discard=edge_discard)
    
    corners_anms = adaptive_non_maximal_suppression(
        corners_all, h, num_corners=num_anms_corners
    )

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # All Harris corners
    axes[0].imshow(image if len(image.shape) == 3 else image, 
                   cmap='gray' if len(image.shape) == 2 else None)
    axes[0].plot(corners_all[1, :], corners_all[0, :], 'r+', 
                 markersize=6, markeredgewidth=1.5)
    axes[0].plot(corners_all[1, :], corners_all[0, :], 'yo', 
                 markersize=4, markerfacecolor='none', markeredgewidth=1)
    axes[0].set_title(f'All Harris Corners ({corners_all.shape[1]} detected)', 
                      fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # ANMS corners
    axes[1].imshow(image if len(image.shape) == 3 else image,
                   cmap='gray' if len(image.shape) == 2 else None)
    axes[1].plot(corners_anms[1, :], corners_anms[0, :], 'r+',
                 markersize=8, markeredgewidth=2)
    axes[1].plot(corners_anms[1, :], corners_anms[0, :], 'yo',
                 markersize=6, markerfacecolor='none', markeredgewidth=1.5)
    axes[1].set_title(f'After ANMS ({corners_anms.shape[1]} selected)',
                      fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_prefix:
        comparison_path = f'{save_prefix}_harris_anms_comparison.png'
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved comparison to {comparison_path}")
    
    plt.show()
    
    return corners_all, corners_anms

if __name__ == "__main__":
    image_path = 'door.png' 
    image = io.imread(image_path)

    os.makedirs("results", exist_ok=True)
    prefix = os.path.join("results", "door")

    corners_all, corners_anms = compare_harris_anms(
        image,
        num_anms_corners=500,
        edge_discard=20,
        save_prefix=prefix
    )

    visualize_corners(
        image, corners_all,
        title="All Harris Corners",
        save_path=os.path.join("results", "door_harris_all.png")
    )

    visualize_corners(
        image, corners_anms,
        title="ANMS Selected Corners",
        save_path=os.path.join("results", "door_harris_anms.png")
    )

    plt.close('all')