import numpy as np
import matplotlib.pyplot as plt
import cv2
from imagewarp import computeH, warpImageBilinear, PointSelector

def compute_mosaic_bounds(images, homographies, reference_idx=0):
    """Compute the bounding box for the entire mosaic."""
    corners_all = []
    
    for i, im in enumerate(images):
        h, w = im.shape[:2]
        corners = np.array([
            [0, 0, 1],
            [w-1, 0, 1],
            [0, h-1, 1],
            [w-1, h-1, 1]
        ]).T
        
        if i == reference_idx:
            warped_corners = corners
        else:
            H = homographies[i]
            warped_corners = H @ corners
            warped_corners = warped_corners[:2, :] / warped_corners[2, :]
            warped_corners = np.vstack([warped_corners, np.ones((1, 4))])
        
        corners_all.append(warped_corners[:2, :])
    
    all_corners = np.hstack(corners_all)
    
    x_min = int(np.floor(np.min(all_corners[0, :])))
    x_max = int(np.ceil(np.max(all_corners[0, :])))
    y_min = int(np.floor(np.min(all_corners[1, :])))
    y_max = int(np.ceil(np.max(all_corners[1, :])))
    
    mosaic_shape = (y_max - y_min, x_max - x_min)
    
    return x_min, x_max, y_min, y_max, mosaic_shape


def create_distance_weight_map(shape):
    """Create weight map using distance transform."""
    h, w = shape
    mask = np.ones((h, w), dtype=np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    if dist.max() > 0:
        weight_map = dist / dist.max()
    else:
        weight_map = np.ones((h, w))
    
    return weight_map


def warp_with_offset(im, H, output_shape, offset):
    """Warp image with translation offset."""
    x_offset, y_offset = offset
    
    T = np.array([
        [1, 0, -x_offset],
        [0, 1, -y_offset],
        [0, 0, 1]
    ])
    
    H_combined = T @ H
    warped = warpImageBilinear(im, H_combined, output_shape)
    
    return warped


def build_gaussian_pyramid(image, levels):
    """Build Gaussian pyramid by repeatedly downsampling."""
    pyramid = [image.astype(np.float32)]
    
    for i in range(levels - 1):
        blurred = cv2.GaussianBlur(pyramid[-1], (5, 5), 0)
        downsampled = cv2.pyrDown(blurred)
        pyramid.append(downsampled)
    
    return pyramid


def build_laplacian_pyramid(image, levels):
    """Build Laplacian pyramid from Gaussian pyramid."""
    gaussian_pyramid = build_gaussian_pyramid(image, levels)
    laplacian_pyramid = []
    
    for i in range(levels - 1):
        current = gaussian_pyramid[i]
        next_level = gaussian_pyramid[i + 1]
        upsampled = cv2.pyrUp(next_level, dstsize=(current.shape[1], current.shape[0]))
        laplacian = current - upsampled
        laplacian_pyramid.append(laplacian)
    
    laplacian_pyramid.append(gaussian_pyramid[-1])
    
    return laplacian_pyramid


def reconstruct_from_laplacian_pyramid(laplacian_pyramid):
    """Reconstruct image from Laplacian pyramid."""
    reconstructed = laplacian_pyramid[-1]
    
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        h, w = laplacian_pyramid[i].shape[:2]
        upsampled = cv2.pyrUp(reconstructed, dstsize=(w, h))
        reconstructed = upsampled + laplacian_pyramid[i]
    
    return reconstructed


def blend_two_images_laplacian(im1, im2, H12, reference_idx=1, pyramid_levels=6):
    """Blend two images using Laplacian pyramid blending."""
    print("\n" + "="*70)
    print("LAPLACIAN PYRAMID BLENDING")
    print("="*70)
    print(f"Pyramid levels: {pyramid_levels}")
    
    images = [im1, im2]
    homographies = [H12, np.eye(3)]
    
    x_min, x_max, y_min, y_max, mosaic_shape = compute_mosaic_bounds(
        images, homographies, reference_idx
    )
    print(f"\nMosaic size: {mosaic_shape[1]} x {mosaic_shape[0]} pixels")
    
    warped_images = []
    weight_maps = []
    
    for i, im in enumerate(images):
        H = np.eye(3) if i == reference_idx else homographies[i]
        
        warped = warp_with_offset(im, H, mosaic_shape, (x_min, y_min))
        warped_images.append(warped)
        
        weight = create_distance_weight_map(im.shape[:2])
        weight_3ch = np.stack([weight] * 3, axis=2)
        weight_3ch = (weight_3ch * 255).astype(np.uint8)
        warped_weight = warp_with_offset(weight_3ch, H, mosaic_shape, (x_min, y_min))
        warped_weight = warped_weight[:, :, 0].astype(np.float32) / 255.0
        weight_maps.append(warped_weight)
    
    lap_pyr1 = build_laplacian_pyramid(warped_images[0], pyramid_levels)
    lap_pyr2 = build_laplacian_pyramid(warped_images[1], pyramid_levels)
    
    weight_sum = weight_maps[0] + weight_maps[1]
    weight_sum[weight_sum == 0] = 1
    
    norm_weight1 = weight_maps[0] / weight_sum
    norm_weight2 = weight_maps[1] / weight_sum
    
    mask_pyr1 = build_gaussian_pyramid(norm_weight1[:, :, np.newaxis], pyramid_levels)
    mask_pyr2 = build_gaussian_pyramid(norm_weight2[:, :, np.newaxis], pyramid_levels)
    
    blended_pyr = []
    
    for level in range(pyramid_levels):
        lap1 = lap_pyr1[level]
        lap2 = lap_pyr2[level]
        mask1 = mask_pyr1[level]
        mask2 = mask_pyr2[level]
        
        if len(lap1.shape) == 3:
            if len(mask1.shape) == 2:
                mask1 = np.stack([mask1] * lap1.shape[2], axis=2)
            if len(mask2.shape) == 2:
                mask2 = np.stack([mask2] * lap2.shape[2], axis=2)
        
        blended_level = lap1 * mask1 + lap2 * mask2
        blended_pyr.append(blended_level)
    
    mosaic = reconstruct_from_laplacian_pyramid(blended_pyr)
    mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)
    
    print(f"✓ Blending complete!")
    
    return mosaic


def blend_two_images_simple(im1, im2, H12, reference_idx=1):
    """Simple weighted averaging blend (for comparison)."""
    print("\n" + "="*70)
    print("SIMPLE WEIGHTED AVERAGING BLEND")
    print("="*70)
    
    images = [im1, im2]
    homographies = [H12, np.eye(3)]
    
    x_min, x_max, y_min, y_max, mosaic_shape = compute_mosaic_bounds(
        images, homographies, reference_idx
    )
    
    warped_images = []
    weight_maps = []
    
    for i, im in enumerate(images):
        H = np.eye(3) if i == reference_idx else homographies[i]
        
        warped = warp_with_offset(im, H, mosaic_shape, (x_min, y_min))
        warped_images.append(warped)
        
        weight = create_distance_weight_map(im.shape[:2])
        weight_3ch = np.stack([weight] * 3, axis=2)
        weight_3ch = (weight_3ch * 255).astype(np.uint8)
        warped_weight = warp_with_offset(weight_3ch, H, mosaic_shape, (x_min, y_min))
        warped_weight = warped_weight[:, :, 0].astype(np.float32) / 255.0
        weight_maps.append(warped_weight)
    
    mosaic_sum = np.zeros((*mosaic_shape, 3), dtype=np.float32)
    weight_sum = np.zeros(mosaic_shape, dtype=np.float32)
    
    for warped, weight in zip(warped_images, weight_maps):
        valid = np.any(warped > 0, axis=2)
        
        for c in range(3):
            mosaic_sum[:, :, c] += warped[:, :, c].astype(np.float32) * weight
        
        weight_sum += weight * valid.astype(np.float32)
    
    mosaic = np.zeros_like(mosaic_sum)
    valid = weight_sum > 0
    
    for c in range(3):
        mosaic[valid, c] = mosaic_sum[valid, c] / weight_sum[valid]
    
    mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)
    
    print(f"Simple blending complete!")
    
    return mosaic


def visualize_mosaic(im1, im2, mosaic, save_name="mosaic", title1="Image 1", title2="Image 2"):
    """Visualize source images and final mosaic."""
    
    fig = plt.figure(figsize=(20, 10))
    
    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
    ax1.set_title(f'Source {title1}', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'Source {title2} (Reference)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = plt.subplot(2, 1, 2)
    ax3.imshow(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
    ax3.set_title('Final Blended Mosaic', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_name}_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {save_name}_visualization.png")
    plt.show()


def create_comparison_visualization(im1, im2, mosaic_simple, mosaic_laplacian, save_name):
    """Compare simple vs Laplacian blending side by side."""
    
    fig = plt.figure(figsize=(20, 12))
    
    ax1 = plt.subplot(3, 2, 1)
    ax1.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
    ax1.set_title('Source Image 1', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 2, 2)
    ax2.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
    ax2.set_title('Source Image 2 (Reference)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = plt.subplot(3, 2, 3)
    ax3.imshow(cv2.cvtColor(mosaic_simple, cv2.COLOR_BGR2RGB))
    ax3.set_title('Simple Weighted Averaging', fontsize=14, fontweight='bold', color='#d97706')
    ax3.axis('off')
    
    ax4 = plt.subplot(3, 2, 4)
    ax4.imshow(cv2.cvtColor(mosaic_laplacian, cv2.COLOR_BGR2RGB))
    ax4.set_title('Laplacian Pyramid Blending', fontsize=14, fontweight='bold', color='#059669')
    ax4.axis('off')
    
    h, w = mosaic_simple.shape[:2]
    zoom_y, zoom_x = h // 2, w // 2
    zoom_size = min(200, h // 4, w // 4)
    
    crop_simple = mosaic_simple[zoom_y-zoom_size:zoom_y+zoom_size, 
                                zoom_x-zoom_size:zoom_x+zoom_size]
    crop_laplacian = mosaic_laplacian[zoom_y-zoom_size:zoom_y+zoom_size,
                                      zoom_x-zoom_size:zoom_x+zoom_size]
    
    plt.tight_layout()
    plt.savefig(f'{save_name}_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_name}_comparison.png")
    plt.show()


def create_mosaic_pair(img1_path, img2_path, homography_file, output_name, 
                       img1_name="Image 1", img2_name="Image 2", use_laplacian=True):
    """Create mosaic from image pair with both blending methods."""
    
    print("\n" + "="*70)
    print(f"PANORAMA: {img1_name} + {img2_name}")
    print("="*70)
    
    im1 = cv2.imread(img1_path)
    im2 = cv2.imread(img2_path)
    
    if im1 is None or im2 is None:
        print(f"Error: Could not load images")
        return None
    
    print(f"\nLoaded images:")
    print(f"  {img1_name}: {im1.shape}")
    print(f"  {img2_name}: {im2.shape} (reference)")
    
    try:
        H12 = np.loadtxt(homography_file)
        print(f"\n✓ Loaded existing homography from {homography_file}")
    except:
        print(f"\n--- Computing homography {img1_name} → {img2_name} ---")
        selector = PointSelector(im1, im2)
        im1_pts, im2_pts = selector.select_points()
        
        if len(im1_pts) < 4:
            print("Error: Need at least 4 point correspondences")
            return None
        
        H12 = computeH(im1_pts, im2_pts)
        np.savetxt(homography_file, H12, fmt='%.6f')
        print(f"\n✓ Saved homography to {homography_file}")
    
    print(f"\nHomography matrix:")
    print(H12)
    
    if use_laplacian:
        print("\n" + "="*70)
        print("CREATING SIMPLE BLEND (for comparison)")
        print("="*70)
        mosaic_simple = blend_two_images_simple(im1, im2, H12, reference_idx=1)
        cv2.imwrite(f'{output_name}_mosaic_simple.png', mosaic_simple)
        
        print("\n" + "="*70)
        print("CREATING LAPLACIAN PYRAMID BLEND")
        print("="*70)
        mosaic_laplacian = blend_two_images_laplacian(im1, im2, H12, reference_idx=1, pyramid_levels=6)
        cv2.imwrite(f'{output_name}_mosaic_laplacian.png', mosaic_laplacian)
        
        create_comparison_visualization(im1, im2, mosaic_simple, mosaic_laplacian, output_name)
        
        mosaic = mosaic_simple
        cv2.imwrite(f'{output_name}_mosaic.png', mosaic)
    else:
        mosaic = blend_two_images_simple(im1, im2, H12, reference_idx=1)
        cv2.imwrite(f'{output_name}_mosaic.png', mosaic)
    
    visualize_mosaic(im1, im2, mosaic, output_name, img1_name, img2_name)
    
    return mosaic


def main():
    """Main function"""
    print("\n" + "="*70)
    print("A.4: IMAGE MOSAIC BLENDING")
    print("="*70)
    print("SELECT MOSAIC TO CREATE")
    print("="*70)
    print("1. Library mosaic (doe1 + doe2)")
    print("2. Mochi mosaic (mochi1 + mochi2)")
    print("3. Clothes mosaic (clothes1 + clothes2)")
    print("4. All three mosaics")
    print("="*70)
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1' or choice == '4':
        print("\n\n" + "="*70)
        print("CREATING LIBRARY MOSAIC")
        print("="*70)
        create_mosaic_pair('doe1.png', 'doe2.png', 'homography_im1_to_im2.txt',
                          'library', 'doe1.png', 'doe2.png', use_laplacian=True)
    
    if choice == '2' or choice == '4':
        print("\n\n" + "="*70)
        print("CREATING MOCHI MOSAIC")
        print("="*70)
        create_mosaic_pair('mochi1.png', 'mochi2.png', 'homography_mochi1_to_mochi2.txt',
                          'mochi', 'mochi1.png', 'mochi2.png', use_laplacian=True)
    
    if choice == '3' or choice == '4':
        print("\n\n" + "="*70)
        print("CREATING CLOTHES MOSAIC")
        print("="*70)
        create_mosaic_pair('clothes1.png', 'clothes2.png', 'homography_clothes1_to_clothes2.txt',
                          'clothes', 'clothes1.png', 'clothes2.png', use_laplacian=True)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()