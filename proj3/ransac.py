import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import warp
import cv2
import os

from harris import get_harris_corners, adaptive_non_maximal_suppression
from descriptor import extract_descriptors_for_corners
from feature_matching import match_features_lowe_ratio


def compute_homography(src_pts, dst_pts):
    """
    Compute homography matrix from point correspondences.
    """
    assert src_pts.shape[0] == dst_pts.shape[0] >= 4
    
    n = src_pts.shape[0]
    
    A = []
    for i in range(n):
        x, y = src_pts[i]
        x_prime, y_prime = dst_pts[i]
        
        A.append([-x, -y, -1, 0, 0, 0, x*x_prime, y*x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x*y_prime, y*y_prime, y_prime])
    
    A = np.array(A)
    
    U, S, Vt = np.linalg.svd(A)
    
    h = Vt[-1, :]
    
    H = h.reshape(3, 3)
    H = H / H[2, 2]
    
    return H


def apply_homography(H, points):
    """
    Apply homography to points.
    """
    n = points.shape[0]
    points_homogeneous = np.hstack([points, np.ones((n, 1))])
    
    transformed = (H @ points_homogeneous.T).T
    
    transformed_points = transformed[:, :2] / transformed[:, 2:3]
    
    return transformed_points


def compute_inliers(H, src_pts, dst_pts, threshold=5.0):
    """
    Compute inliers for a given homography.
    
    """
    transformed_pts = apply_homography(H, src_pts)
    
    distances = np.linalg.norm(transformed_pts - dst_pts, axis=1)
    
    inliers = distances < threshold
    num_inliers = np.sum(inliers)
    
    return inliers, num_inliers


def ransac_homography(src_pts, dst_pts, num_iterations=5000, threshold=5.0):
    """
    Compute homography using RANSAC.
    """
    assert src_pts.shape[0] == dst_pts.shape[0] >= 4
    
    n = src_pts.shape[0]
    best_H = None
    best_inliers = None
    max_inliers = 0
    
    for iteration in range(num_iterations):
        indices = np.random.choice(n, 4, replace=False)
        src_sample = src_pts[indices]
        dst_sample = dst_pts[indices]
        
        try:
            H = compute_homography(src_sample, dst_sample)
        except:
            continue
        
        inliers, num_inliers = compute_inliers(H, src_pts, dst_pts, threshold)
        
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_H = H
            best_inliers = inliers

    if best_inliers is not None and np.sum(best_inliers) >= 4:
        best_H = compute_homography(src_pts[best_inliers], dst_pts[best_inliers])
        best_inliers, _ = compute_inliers(best_H, src_pts, dst_pts, threshold)
    
    return best_H, best_inliers


def warp_image(image, H, output_shape):
    h_out, w_out = output_shape
    
    if len(image.shape) == 3:
        warped = np.zeros((h_out, w_out, image.shape[2]), dtype=image.dtype)
    else:
        warped = np.zeros((h_out, w_out), dtype=image.dtype)
    
    H_inv = np.linalg.inv(H)
    
    for y in range(h_out):
        for x in range(w_out):
            src_point = H_inv @ np.array([x, y, 1])
            src_x = src_point[0] / src_point[2]
            src_y = src_point[1] / src_point[2]
            
            if 0 <= src_x < image.shape[1]-1 and 0 <= src_y < image.shape[0]-1:
                x0, y0 = int(src_x), int(src_y)
                x1, y1 = x0 + 1, y0 + 1
                
                wx = src_x - x0
                wy = src_y - y0
                
                if len(image.shape) == 3:
                    warped[y, x] = (
                        (1-wx) * (1-wy) * image[y0, x0] +
                        wx * (1-wy) * image[y0, x1] +
                        (1-wx) * wy * image[y1, x0] +
                        wx * wy * image[y1, x1]
                    )
                else:
                    warped[y, x] = (
                        (1-wx) * (1-wy) * image[y0, x0] +
                        wx * (1-wy) * image[y0, x1] +
                        (1-wx) * wy * image[y1, x0] +
                        wx * wy * image[y1, x1]
                    )
    
    return warped


def create_mosaic(image1, image2, H):

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    corners1 = np.array([
        [0, 0],
        [w1, 0],
        [w1, h1],
        [0, h1]
    ])
    
    warped_corners = apply_homography(H, corners1)
    
    all_corners = np.vstack([
        warped_corners,
        [[0, 0], [w2, 0], [w2, h2], [0, h2]]
    ])
    
    x_min = int(np.floor(np.min(all_corners[:, 0])))
    x_max = int(np.ceil(np.max(all_corners[:, 0])))
    y_min = int(np.floor(np.min(all_corners[:, 1])))
    y_max = int(np.ceil(np.max(all_corners[:, 1])))
    
    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])
    
    output_shape = (y_max - y_min, x_max - x_min)
    
    print(f"Mosaic size: {output_shape[1]} x {output_shape[0]}")
    
    H_translated = translation @ H
    warped1 = warp_image(image1, H_translated, output_shape)
    
    warped2 = np.zeros_like(warped1)
    y_start = -y_min
    x_start = -x_min
    warped2[y_start:y_start+h2, x_start:x_start+w2] = image2
    

    weight1 = np.zeros(output_shape, dtype=np.float32)
    weight2 = np.zeros(output_shape, dtype=np.float32)
    
    if len(warped1.shape) == 3:
        weight1[np.any(warped1 > 0, axis=2)] = 1.0
    else:
        weight1[warped1 > 0] = 1.0
    
    weight2[y_start:y_start+h2, x_start:x_start+w2] = 1.0
    
    weight1 = cv2.distanceTransform((weight1 > 0).astype(np.uint8), 
                                    cv2.DIST_L2, 5)
    weight2 = cv2.distanceTransform((weight2 > 0).astype(np.uint8), 
                                    cv2.DIST_L2, 5)
    
    weight_sum = weight1 + weight2
    weight_sum[weight_sum == 0] = 1  
    
    weight1 = weight1 / weight_sum
    weight2 = weight2 / weight_sum
    
    if len(warped1.shape) == 3:
        mosaic = (warped1 * weight1[:, :, np.newaxis] + 
                  warped2 * weight2[:, :, np.newaxis])
    else:
        mosaic = warped1 * weight1 + warped2 * weight2
    
    return mosaic.astype(image1.dtype)


def visualize_ransac_inliers(image1, image2, corners1, corners2, matches, 
                              inliers, save_path=None):
    """
    Visualize RANSAC inliers vs outliers.
    
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    h_max = max(h1, h2)
    
    canvas = np.zeros((h_max, w1 + w2, 3), dtype=np.uint8)
    
    if len(image1.shape) == 2:
        canvas[:h1, :w1] = np.stack([image1]*3, axis=2)
    else:
        canvas[:h1, :w1] = image1
        
    if len(image2.shape) == 2:
        canvas[:h2, w1:w1+w2] = np.stack([image2]*3, axis=2)
    else:
        canvas[:h2, w1:w1+w2] = image2
    
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(canvas)
    
    for idx, (i1, i2) in enumerate(matches):
        if not inliers[idx]:
            y1, x1 = corners1[0, i1], corners1[1, i1]
            y2, x2 = corners2[0, i2], corners2[1, i2]
            x2_adjusted = x2 + w1
            
            ax.plot([x1, x2_adjusted], [y1, y2], 'r-', 
                   linewidth=1, alpha=0.3)
            ax.plot(x1, y1, 'ro', markersize=4, alpha=0.5)
            ax.plot(x2_adjusted, y2, 'ro', markersize=4, alpha=0.5)
    
    for idx, (i1, i2) in enumerate(matches):
        if inliers[idx]:
            y1, x1 = corners1[0, i1], corners1[1, i1]
            y2, x2 = corners2[0, i2], corners2[1, i2]
            x2_adjusted = x2 + w1
            
            ax.plot([x1, x2_adjusted], [y1, y2], 'g-', 
                   linewidth=1.5, alpha=0.7)
            ax.plot(x1, y1, 'go', markersize=5, 
                   markeredgewidth=1, markerfacecolor='yellow')
            ax.plot(x2_adjusted, y2, 'go', markersize=5,
                   markeredgewidth=1, markerfacecolor='yellow')
    
    num_inliers = np.sum(inliers)
    num_outliers = len(inliers) - num_inliers
    
    ax.set_title(f'RANSAC Results: {num_inliers} Inliers (green) | '
                f'{num_outliers} Outliers (red)', 
                fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved RANSAC visualization to {save_path}")
    
    plt.show()


def compare_manual_automatic_stitching(image1, image2, manual_mosaic, 
                                       auto_mosaic, save_path=None):
    """
    Compare manual and automatic stitching side by side.
    
    """
    fig = plt.figure(figsize=(20, 12))

    ax1 = plt.subplot(3, 2, 1)
    ax1.imshow(image1)
    ax1.set_title('Source Image 1', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 2, 2)
    ax2.imshow(image2)
    ax2.set_title('Source Image 2', fontsize=12, fontweight='bold')
    ax2.axis('off')

    ax3 = plt.subplot(3, 1, 2)
    ax3.imshow(manual_mosaic)
    ax3.set_title('Manual Stitching (Part A)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    ax4 = plt.subplot(3, 1, 3)
    ax4.imshow(auto_mosaic)
    ax4.set_title('Automatic Stitching (Part B - RANSAC)', 
                 fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    plt.suptitle('B.4: Manual vs Automatic Image Stitching Comparison', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("B.4: RANSAC FOR ROBUST HOMOGRAPHY ESTIMATION")
    print("=" * 70)
    
    os.makedirs("results", exist_ok=True)
    
    image_pairs = [
        {
            'name': 'Library (doe)',
            'image1': 'doe1.png',
            'image2': 'doe2.png',
            'manual_mosaic': 'library_mosaic.png',
            'prefix': 'doe'
        },
        {
            'name': 'Mochi',
            'image1': 'mochi1.png',
            'image2': 'mochi2.png',
            'manual_mosaic': 'mochi_mosaic.png',
            'prefix': 'mochi'
        },
        {
            'name': 'Clothes',
            'image1': 'clothes1.png',
            'image2': 'clothes2.png',
            'manual_mosaic': 'clothes_mosaic.png',
            'prefix': 'clothes'
        }
    ]
    
    for pair_idx, pair_info in enumerate(image_pairs):
        print("\n" + "=" * 70)
        print(f"PROCESSING PAIR {pair_idx + 1}/3: {pair_info['name']}")
        print("=" * 70)
        
        image1_path = pair_info['image1']
        image2_path = pair_info['image2']
        
        try:
            image1 = io.imread(image1_path)
            image2 = io.imread(image2_path)
        except FileNotFoundError:
            print(f"ERROR: Could not find {image1_path} or {image2_path}")
            print("Skipping this pair...")
            continue
        
        if image1.shape[-1] == 4:
            image1 = image1[..., :3]
        if image2.shape[-1] == 4:
            image2 = image2[..., :3]
            
        gray1 = color.rgb2gray(image1)
        gray2 = color.rgb2gray(image2)
        
        # Step 1: Feature detection and extraction
        h1, corners1_all = get_harris_corners(gray1, edge_discard=20)
        corners1_anms = adaptive_non_maximal_suppression(corners1_all, h1, num_corners=500)
        descriptors1, corners1, _ = extract_descriptors_for_corners(gray1, corners1_anms)
        print(f"Image 1: {len(descriptors1)} descriptors")
        
        h2, corners2_all = get_harris_corners(gray2, edge_discard=20)
        corners2_anms = adaptive_non_maximal_suppression(corners2_all, h2, num_corners=500)
        descriptors2, corners2, _ = extract_descriptors_for_corners(gray2, corners2_anms)
        print(f"Image 2: {len(descriptors2)} descriptors")
        
         # Step 2: Feature matching
        matches, match_confidences = match_features_lowe_ratio(descriptors1, descriptors2, 
                                                               ratio_threshold=0.8)
        print(f"Found {len(matches)} matches")
        
        if len(matches) < 4:
            print(f"ERROR: Not enough matches ({len(matches)} < 4)")
            print("Skipping this pair...")
            continue
        
        src_pts = np.array([corners1[:, m[0]][::-1] for m in matches])  
        dst_pts = np.array([corners2[:, m[1]][::-1] for m in matches])  
        
        # Step 3: RANSAC
        H_ransac, inliers = ransac_homography(src_pts, dst_pts, 
                                             num_iterations=5000, 
                                             threshold=5.0)
        
        num_inliers = np.sum(inliers)
        print(f"\nFinal: {num_inliers}/{len(matches)} inliers ({num_inliers/len(matches)*100:.1f}%)")
        print(f"\nEstimated Homography Matrix:")
        print(H_ransac)
        
        # Visualize RANSAC results
        visualize_ransac_inliers(image1, image2, corners1, corners2, matches, inliers,
                                save_path=f'results/{pair_info["prefix"]}_ransac_inliers.png')
        
        # Step 4: Create automatic mosaic
        auto_mosaic = create_mosaic(image1, image2, H_ransac)
        
        # Save automatic mosaic
        plt.figure(figsize=(16, 10))
        plt.imshow(auto_mosaic)
        plt.title(f'{pair_info["name"]} - Automatic Mosaic (RANSAC Homography)', 
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'results/{pair_info["prefix"]}_automatic_mosaic.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        # Step 5: Load manual mosaic and compare
        try:
            manual_mosaic = io.imread(pair_info['manual_mosaic'])
            compare_manual_automatic_stitching(image1, image2, manual_mosaic, auto_mosaic,
                                              save_path=f'results/{pair_info["prefix"]}_manual_vs_automatic.png')
            print(f"Comparison saved!")
        except FileNotFoundError:
            print(f"Warning: Manual mosaic {pair_info['manual_mosaic']} not found.")
            print("Skipping comparison for this pair.")
        except Exception as e:
            print(f"Warning: Could not create comparison: {e}")
        
    
    plt.close('all')