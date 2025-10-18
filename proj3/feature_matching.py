import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import os

from harris import get_harris_corners, adaptive_non_maximal_suppression
from descriptor import extract_descriptors_for_corners


def compute_feature_distances(descriptors1, descriptors2):
    """
    Compute pairwise distances between two sets of descriptors.
    
    """

    n1 = descriptors1.shape[0]
    n2 = descriptors2.shape[0]
    
    desc1_expanded = descriptors1[:, np.newaxis, :]  
    desc2_expanded = descriptors2[np.newaxis, :, :] 
    
    squared_diffs = (desc1_expanded - desc2_expanded) ** 2
    distances = np.sqrt(np.sum(squared_diffs, axis=2))
    
    return distances


def match_features_lowe_ratio(descriptors1, descriptors2, ratio_threshold=0.8):
    """
    Match features using Lowe's ratio test.
    """
    distances = compute_feature_distances(descriptors1, descriptors2)
    
    matches = []
    match_confidences = []
    
    for i in range(descriptors1.shape[0]):
        dists = distances[i, :]
        
        sorted_indices = np.argsort(dists)
        if len(sorted_indices) < 2:
            continue
            
        nearest_idx = sorted_indices[0]
        second_nearest_idx = sorted_indices[1]
        
        dist_1nn = dists[nearest_idx]
        dist_2nn = dists[second_nearest_idx]
        
        if dist_2nn > 0:  
            ratio = dist_1nn / dist_2nn
            
            if ratio < ratio_threshold:
                matches.append([i, nearest_idx])
                match_confidences.append(1 - ratio)  
    
    if len(matches) == 0:
        return np.array([]).reshape(0, 2), np.array([])
    
    matches = np.array(matches)
    match_confidences = np.array(match_confidences)
    
    return matches, match_confidences


def visualize_matches(image1, image2, corners1, corners2, matches, 
                     max_matches=50, save_path=None):
    """
    Visualize feature matches between two images.

    """
    n_matches = min(len(matches), max_matches)
    selected_matches = matches[:n_matches]
    
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
    
    colors = plt.cm.rainbow(np.linspace(0, 1, n_matches))
    
    for idx, (i1, i2) in enumerate(selected_matches):
        y1, x1 = corners1[0, i1], corners1[1, i1]
        y2, x2 = corners2[0, i2], corners2[1, i2]
        
        x2_adjusted = x2 + w1
        ax.plot([x1, x2_adjusted], [y1, y2], '-', 
               color=colors[idx], linewidth=1.5, alpha=0.7)
        
        ax.plot(x1, y1, 'o', color=colors[idx], markersize=6, 
               markeredgewidth=2, markerfacecolor='yellow')
        ax.plot(x2_adjusted, y2, 'o', color=colors[idx], markersize=6,
               markeredgewidth=2, markerfacecolor='yellow')
    
    ax.set_title(f'B.3: Feature Matches ({n_matches} matches shown)', 
                fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    image_pairs = [
        ('doe1.png', 'doe2.png'),
        ('mochi1.png', 'mochi2.png'),
        ('clothes1.png', 'clothes2.png')
    ]
    os.makedirs("results", exist_ok=True)
    ratio_threshold = 0.8   # Based on Figure 6b in the paper

    for image1_path, image2_path in image_pairs:
        print("B.3: FEATURE MATCHING for {image1_path} ↔ {image2_path}")

        image1 = io.imread(image1_path)
        image2 = io.imread(image2_path)
        if image1.shape[-1] == 4:
            image1 = image1[..., :3]
        if image2.shape[-1] == 4:
            image2 = image2[..., :3]
        
        gray1 = color.rgb2gray(image1)
        gray2 = color.rgb2gray(image2)
    
        print("\nProcessing Image 1")
        h1, corners1_all = get_harris_corners(gray1, edge_discard=20)
        corners1_anms = adaptive_non_maximal_suppression(corners1_all, h1, num_corners=500)
        descriptors1, corners1, _ = extract_descriptors_for_corners(gray1, corners1_anms)
    
        print("\nProcessing Image 2")
        h2, corners2_all = get_harris_corners(gray2, edge_discard=20)
        corners2_anms = adaptive_non_maximal_suppression(corners2_all, h2, num_corners=500)
        descriptors2, corners2, _ = extract_descriptors_for_corners(gray2, corners2_anms)
    
        print("\nMatching Features")
        matches, match_confidences = match_features_lowe_ratio(
        descriptors1, descriptors2, ratio_threshold=ratio_threshold
        )
        print(f"Match rate: {len(matches)}")
        print(f"Match rate: {len(matches)/len(descriptors1)*100:.1f}% of features in image 1")
    
        os.makedirs("results", exist_ok=True)
    
        save_name = f"results/{os.path.splitext(image1_path)[0]}_matches.png"
        print("\nVisualizing Matches.")
        visualize_matches(image1, image2, corners1, corners2, matches,
                          max_matches=50, save_path=save_name)
        
        plt.close('all')
    