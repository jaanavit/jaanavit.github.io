import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter
import os
from matplotlib.image import imsave

os.makedirs("results", exist_ok=True)
def build_gaussian_stack(image, N=5, sigma0=1.0):
    """
    Build a Gaussian stack (NOT pyramid - no downsampling).
    
    Args:
        image: Input image (grayscale or color)
        N: Number of stack levels
        sigma0: Base sigma for Gaussian filtering
        
    Returns:
        List of images representing the Gaussian stack
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float64) / 255.0
    elif image.max() > 1.0:
        image = image / 255.0
    
    gaussian_stack = [image.copy()]
    
    for i in range(1, N):
        sigma = sigma0 * (2 ** i)
        blurred = gaussian_filter(image, sigma=sigma)
        gaussian_stack.append(blurred)
    
    return gaussian_stack

def build_laplacian_stack(gaussian_stack):
    """
    Build a Laplacian stack from a Gaussian stack.
    
    Args:
        gaussian_stack: List of images from build_gaussian_stack
        
    Returns:
        List of images representing the Laplacian stack
    """
    laplacian_stack = []
    N = len(gaussian_stack)
    
    for i in range(N-1):
        laplacian = gaussian_stack[i] - gaussian_stack[i+1]
        laplacian_stack.append(laplacian)
    
    laplacian_stack.append(gaussian_stack[-1])
    
    return laplacian_stack

def create_mask_stack(mask, N=5):
    """
    Create a stack of masks for blending. Each level has increasing blur.
    
    Args:
        mask: Binary mask (0s and 1s)
        N: Number of stack levels
        
    Returns:
        List of progressively blurred masks
    """
    mask_stack = []
    sigma0 = 1.0
    
    for i in range(N):
        if i == 0:
            mask_stack.append(mask.astype(np.float64))
        else:
            sigma = sigma0 * (2 ** (i-1))
            blurred_mask = gaussian_filter(mask.astype(np.float64), sigma=sigma)
            mask_stack.append(blurred_mask)
    
    return mask_stack

def multiresolution_blend(image1, image2, mask, N=5):
    if image1.dtype == np.uint8:
        image1 = image1.astype(np.float64) / 255.0
    if image2.dtype == np.uint8:
        image2 = image2.astype(np.float64) / 255.0
    
    gauss_stack1 = build_gaussian_stack(image1, N)
    gauss_stack2 = build_gaussian_stack(image2, N)
    
    lap_stack1 = build_laplacian_stack(gauss_stack1)
    lap_stack2 = build_laplacian_stack(gauss_stack2)
    
    mask_stack = create_mask_stack(mask, N)
    
    blended_stack = []
    weighted_stack1 = []
    weighted_stack2 = []
    
    for i in range(N):
        if len(lap_stack1[i].shape) == 3:
            mask_3d = np.stack([mask_stack[i]] * 3, axis=2)
            w1 = lap_stack1[i] * mask_3d
            w2 = lap_stack2[i] * (1 - mask_3d)
        else:
            w1 = lap_stack1[i] * mask_stack[i]
            w2 = lap_stack2[i] * (1 - mask_stack[i])
        
        weighted_stack1.append(w1)
        weighted_stack2.append(w2)
        blended_stack.append(w1 + w2)
    
    result = blended_stack[-1].copy()
    for i in range(N-2, -1, -1):
        result = result + blended_stack[i]
    
    return np.clip(result, 0, 1), lap_stack1, lap_stack2, weighted_stack1, weighted_stack2, blended_stack

def normalize_laplacian_for_display(laplacian_image):
    """
    Normalize Laplacian image for display (add 0.5 to center around gray).
    """
    normalized = laplacian_image + 0.5
    return np.clip(normalized, 0, 1)

def load_images_for_blending(apple_path='apple.jpeg', orange_path='orange.jpeg', target_size=(300, 300)):
    """
    Load apple and orange images for blending.
    
    Args:
        apple_path: Path to apple image
        orange_path: Path to orange image
        target_size: Target size for resizing (width, height)
        
    Returns:
        apple_img, orange_img, mask
    """
    apple_img = None
    for path in [apple_path, 'apple.jpeg', 'apple.png']:
        if os.path.exists(path):
            apple_img = cv2.imread(path)
            if apple_img is not None:
                apple_img = cv2.cvtColor(apple_img, cv2.COLOR_BGR2RGB)
                print(f"Loaded apple image: {path}")
                break
    
    orange_img = None
    for path in [orange_path, 'orange.jpeg', 'orange.png']:
        if os.path.exists(path):
            orange_img = cv2.imread(path)
            if orange_img is not None:
                orange_img = cv2.cvtColor(orange_img, cv2.COLOR_BGR2RGB)
                print(f"Loaded orange image: {path}")
                break
    
    if apple_img is not None and len(apple_img.shape) == 3:
        apple_img = cv2.resize(apple_img, target_size)
    if orange_img is not None and len(orange_img.shape) == 3:
        orange_img = cv2.resize(orange_img, target_size)
    
    mask = np.zeros((target_size[1], target_size[0]))
    mask[:, target_size[0]//2:] = 1  
    
    return apple_img, orange_img, mask

def create_blending_mask(height, width, blend_region_ratio=0.4):
    """
    Create a blending mask with a smooth transition region.
    
    Args:
        height, width: Dimensions of the mask
        blend_region_ratio: Fraction of width for blending region (0.4 = 40%)
    
    Returns:
        Binary mask for blending
    """
    mask = np.zeros((height, width))
    
    blend_width = int(width * blend_region_ratio)
    left_edge = (width - blend_width) // 2
    right_edge = left_edge + blend_width
    
    mask[:, :left_edge] = 0
    
    for i in range(blend_width):
        mask[:, left_edge + i] = i / (blend_width - 1)
      
    mask[:, right_edge:] = 1
    
    return mask

def visualize_complete_blending_process(apple_img, orange_img):
    """
    Complete visualization of the blending process as described in Part 2.3
    """
    print("=" * 80)
    print("PART 2.3: COMPLETE MULTIRESOLUTION BLENDING PROCESS")
    print("=" * 80)
    
    mask = create_blending_mask(apple_img.shape[0], apple_img.shape[1], blend_region_ratio=0.4)
    
    # Step 2: Build and show Gaussian stacks
    apple_gauss = build_gaussian_stack(apple_img, N=5)
    orange_gauss = build_gaussian_stack(orange_img, N=5) 
    mask_gauss = create_mask_stack(mask, N=5)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i in range(5):
        axes[0, i].imshow(apple_gauss[i])
        axes[0, i].set_title(f'Apple Gaussian Level {i}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(orange_gauss[i])
        axes[1, i].set_title(f'Orange Gaussian Level {i}')
        axes[1, i].axis('off')
    
    plt.suptitle('Step 2: Gaussian Stacks (Different Degrees of Low-pass Filtering)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/step2_gaussian_stacks.png")
    plt.show()
    
    # Step 3: Build and show Laplacian stacks
    apple_lap = build_laplacian_stack(apple_gauss)
    orange_lap = build_laplacian_stack(orange_gauss)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i in range(5):
        apple_lap_display = normalize_laplacian_for_display(apple_lap[i])
        orange_lap_display = normalize_laplacian_for_display(orange_lap[i])
        
        axes[0, i].imshow(apple_lap_display)
        axes[0, i].set_title(f'Apple Laplacian Level {i}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(orange_lap_display)
        axes[1, i].set_title(f'Orange Laplacian Level {i}')
        axes[1, i].axis('off')
    
    plt.suptitle('Step 3: Laplacian Stacks (Decomposition into Frequency Ranges)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/step3_laplacian_stacks.png")
    plt.show()
    
    # Step 4: Show mask Gaussian stack
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i in range(5):
        axes[i].imshow(mask_gauss[i], cmap='gray')
        axes[i].set_title(f'Mask Gaussian Level {i}')
        axes[i].axis('off')
    
    plt.suptitle('Step 4: Gaussian Stack of the Blending Mask', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/step4_mask_stack.png")
    plt.show()
    
    # Step 5: Create weighted Laplacian stacks
    apple_weighted = []
    orange_weighted = []
    
    for i in range(5):
        if len(apple_lap[i].shape) == 3:  # Color
            mask_3d = np.stack([mask_gauss[i]] * 3, axis=2)
            apple_w = apple_lap[i] * mask_3d
            orange_w = orange_lap[i] * (1 - mask_3d)
        else:  # Grayscale
            apple_w = apple_lap[i] * mask_gauss[i]
            orange_w = orange_lap[i] * (1 - mask_gauss[i])
        
        apple_weighted.append(apple_w)
        orange_weighted.append(orange_w)
    
    # Show weighted Laplacian stacks
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i in range(5):
        apple_w_display = normalize_laplacian_for_display(apple_weighted[i])
        orange_w_display = normalize_laplacian_for_display(orange_weighted[i])
        
        axes[0, i].imshow(apple_w_display)
        axes[0, i].set_title(f'Apple Weighted Level {i}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(orange_w_display)
        axes[1, i].set_title(f'Orange Weighted Level {i}')
        axes[1, i].axis('off')
    
    plt.suptitle('Step 5: Weighted Laplacian Stacks (Apple×Mask, Orange×(1-Mask))', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Step 6: Collapse weighted stacks individually
    apple_collapsed = apple_weighted[-1].copy()
    for i in range(3, -1, -1):
        apple_collapsed = apple_collapsed + apple_weighted[i]
    
    orange_collapsed = orange_weighted[-1].copy()
    for i in range(3, -1, -1):
        orange_collapsed = orange_collapsed + orange_weighted[i]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(np.clip(apple_collapsed, 0, 1))
    axes[0].set_title('Collapsed Apple Weighted Stack')
    axes[0].axis('off')
    
    axes[1].imshow(np.clip(orange_collapsed, 0, 1))
    axes[1].set_title('Collapsed Orange Weighted Stack')
    axes[1].axis('off')
    
    plt.suptitle('Step 6: Individual Collapsed Weighted Stacks', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/step6_collapsed_weighted.png")
    plt.show()
    
    # Step 7: Create blended Laplacian stack
    blended_lap = []
    for i in range(5):
        blended_lap.append(apple_weighted[i] + orange_weighted[i])
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i in range(5):
        blended_display = normalize_laplacian_for_display(blended_lap[i])
        axes[i].imshow(blended_display)
        axes[i].set_title(f'Blended Laplacian Level {i}')
        axes[i].axis('off')
    
    plt.suptitle('Step 7: Blended Laplacian Stack (Sum of Weighted Stacks)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/step7_blendedlaplacian.png")
    plt.show()
    
    # Step 8: Final blend
    final_blend = blended_lap[-1].copy()
    for i in range(3, -1, -1):
        final_blend = final_blend + blended_lap[i]
    
    final_blend = np.clip(final_blend, 0, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(apple_img)
    axes[0].set_title('Original Apple')
    axes[0].axis('off')
    
    axes[1].imshow(orange_img)
    axes[1].set_title('Original Orange')
    axes[1].axis('off')
    
    axes[2].imshow(final_blend)
    axes[2].set_title('Final Blended "Oraple"')
    axes[2].axis('off')
    
    plt.suptitle('Step 8: Final Blended Image (Collapsed Blended Laplacian Stack)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/step8_final_blend.png")
    plt.show()
    
    imsave("results/apple_original.png", (apple_img * 255).astype(np.uint8))
    imsave("results/orange_original.png", (orange_img * 255).astype(np.uint8))
    imsave("results/oraple_final.png", (final_blend * 255).astype(np.uint8))

    plt.savefig("results/apple_gaussian_stack.png")
    imsave('results/part_2_3_oraple.jpg', (final_blend * 255).astype(np.uint8))
    print("Saved apple_original.png, orange_original.png, and oraple_final.png")

    return final_blend, apple_gauss, orange_gauss, apple_lap, orange_lap, mask_gauss, apple_weighted, orange_weighted, blended_lap

def part_2_3_stacks_demo():
    """
    Main demonstration for Part 2.3: Gaussian and Laplacian Stacks.
    This recreates Figure 3.42 from Szeliski showing the Burt & Adelson 1983 method.
    """
    print("=" * 60)
    print("PART 2.3: GAUSSIAN AND LAPLACIAN STACKS")
    print("=" * 60)
    
    apple_img, orange_img, _ = load_images_for_blending("apple.jpeg", "orange.jpeg")
    
    result = visualize_complete_blending_process(apple_img, orange_img)
    
    print("\n" + "=" * 60)
    print("PART 2.3 COMPLETE!")
    print("=" * 60)
    print("Complete blending process visualized step-by-step")
    print("Gaussian stacks show different low-pass filtering levels")
    print("Laplacian stacks decompose images into frequency ranges")
    print("Weighted stacks show masked contributions")
    print("Final blend achieved through stack collapse")
    print("40% blending region creates smooth transition")
    
    return result

# ============================================================================
# PART 2.4: MULTIRESOLUTION BLENDING WITH CREATIVE APPLICATIONS
# ============================================================================

def create_vertical_seam_mask(height, width, split_ratio=0.5):
    """Create a vertical seam mask for splitting image vertically."""
    mask = np.zeros((height, width))
    split_col = int(width * split_ratio)
    mask[:, split_col:] = 1
    return mask

def create_horizontal_seam_mask(height, width, split_ratio=0.5):
    """Create a horizontal seam mask for splitting image horizontally."""
    mask = np.zeros((height, width))
    split_row = int(height * split_ratio)
    mask[split_row:, :] = 1
    return mask

def create_circular_mask(height, width, center=None, radius=None):
    """Create a circular mask."""
    if center is None:
        center = (height//2, width//2)
    if radius is None:
        radius = min(height, width) // 4
    
    mask = np.zeros((height, width))
    y, x = np.ogrid[:height, :width]
    dist = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    mask[dist <= radius] = 1
    return mask

def irregular_mask_demo():
    """
    Demonstration of irregular mask blending using person outline mask on maninhills and mountain images.
    Includes cropping capability and optimized person silhouette mask.
    """
    print("=" * 60)
    print("IRREGULAR MASK BLENDING DEMO")
    print("=" * 60)
    
    img1, img2 = load_and_prepare_images("maninhills.jpg", "mountain.jpg", target_size=(400, 400))
    
    # cropping of top part of first image
    crop_top = 50  
    if crop_top > 0:
        cropped_img1 = img1[crop_top:, :, :]
        new_img1 = np.zeros_like(img1)
        available_height = 400 - crop_top
        
        if cropped_img1.shape[0] > available_height:
            cropped_img1 = cv2.resize(cropped_img1, (400, available_height))
        
        new_img1[:cropped_img1.shape[0], :, :] = cropped_img1
        img1 = new_img1
    
    mask = np.zeros((400, 400))
    center_x = 400 // 2
    person_top = 200
    person_bottom = 400
    for y in range(person_top, person_bottom):
        for x in range(400):
            y_in_person = (y - person_top) / (person_bottom - person_top)
            dist_from_center = abs(x - center_x)
            if y_in_person < 0.2:
                max_width = 400 * 0.12
            elif y_in_person < 0.5:
                max_width = 400 * 0.25
            else:
                max_width = 400 * 0.22
            if dist_from_center < max_width:
                intensity = 1.0 - (dist_from_center / max_width) ** 1.5
                mask[y, x] = max(0, intensity)
    mask = gaussian_filter(mask, sigma=6)

    result, _, _, _, _, blended_stack = multiresolution_blend(img1, img2, mask, N=6)


    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img1)
    axes[0].set_title('Man in Hills (Cropped)')
    axes[0].axis('off')
    axes[1].imshow(img2)
    axes[1].set_title('Mountain Landscape')
    axes[1].axis('off')
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('Person Outline Mask')
    axes[2].axis('off')
    axes[3].imshow(result)
    axes[3].set_title('Final Blend')
    axes[3].axis('off')

    plt.suptitle('Optimized Person + Landscape Blend', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Save the 4-panel composite visualization
    fig.savefig("results/person_blend_quadview.png", dpi=300)
    print("4-panel composite saved as results/person_blend_quadview.png")

    visualize_blending_process(img1, img2, mask, result, blended_stack, "Person Outline Blending Process")

    return result

def load_and_prepare_images(img1_path, img2_path, target_size=(400, 400)):
    """Load and prepare two images for blending."""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    img1 = cv2.resize(img1, target_size)
    img2 = cv2.resize(img2, target_size)
    
    return img1, img2

def create_diagonal_gradient_mask(height, width, angle=45):
    """Create a diagonal gradient mask."""
    mask = np.zeros((height, width))
    
    for y in range(height):
        for x in range(width):
            if angle == 45:
                progress = (x + y) / (width + height)
            else:
                progress = x / width
            
            mask[y, x] = 1 / (1 + np.exp(-10 * (progress - 0.5)))
    
    return mask

def visualize_blending_process(img1, img2, mask, result, blended_stack, title="Blending Process"):
    """Visualize the complete blending process similar to Figure 10 in the paper.""" 

    def normalize_for_display(lap_img):
        return np.clip(lap_img + 0.5, 0, 1)
    
    fig = plt.figure(figsize=(16, 12))
    
    plt.subplot(4, 5, 1)
    plt.imshow(img1)
    plt.title('Image 1', fontsize=10)
    plt.axis('off')
    
    plt.subplot(4, 5, 2)
    plt.imshow(img2)
    plt.title('Image 2', fontsize=10)
    plt.axis('off')
    
    plt.subplot(4, 5, 3)
    plt.imshow(mask, cmap='gray')
    plt.title('Blend Mask', fontsize=10)
    plt.axis('off')
    
    plt.subplot(4, 5, 4)
    plt.imshow(result)
    plt.title('Final Result', fontsize=10)
    plt.axis('off')
    
    gauss1 = build_gaussian_stack(img1, N=5)
    gauss2 = build_gaussian_stack(img2, N=5)
    lap1 = build_laplacian_stack(gauss1)
    lap2 = build_laplacian_stack(gauss2)
    mask_stack = create_mask_stack(mask, N=5)
    
    levels_to_show = [0, 1, 2, 3, 4]
    
    for i, level in enumerate(levels_to_show):
        plt.subplot(4, 5, 6 + i)
        if len(lap1[level].shape) == 3:
            lap_display = normalize_for_display(lap1[level])
        else:
            lap_display = normalize_for_display(lap1[level])
        plt.imshow(lap_display, cmap='gray' if len(lap1[level].shape) == 2 else None)
        plt.title(f'Img1 L{level}', fontsize=9)
        plt.axis('off')
        
        plt.subplot(4, 5, 11 + i)
        if len(lap2[level].shape) == 3:
            lap_display = normalize_for_display(lap2[level])
        else:
            lap_display = normalize_for_display(lap2[level])
        plt.imshow(lap_display, cmap='gray' if len(lap2[level].shape) == 2 else None)
        plt.title(f'Img2 L{level}', fontsize=9)
        plt.axis('off')
        
        plt.subplot(4, 5, 16 + i)
        blended_display = normalize_for_display(blended_stack[level])
        plt.imshow(blended_display, cmap='gray' if len(blended_stack[level].shape) == 2 else None)
        plt.title(f'Blend L{level}', fontsize=9)
        plt.axis('off')
    
    plt.suptitle(f'{title}\nLaplacian Stack Decomposition and Blending', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def beach_bike_blend_demo():
    print("=" * 60)
    print("BEACH + MOUNTAIN BIKE BLEND DEMO")
    print("=" * 60)
    
    img1, img2 = load_and_prepare_images("beach.jpg", "bike.jpg", target_size=(400, 400))
    
    mask = np.zeros((400, 400))
    
    for y in range(400):
        for x in range(400):
            progress = y / 400
            transition_center = 0.6
            transition_width = 0.3
            if progress < transition_center - transition_width/2:
                mask[y, x] = 0
            elif progress > transition_center + transition_width/2:
                mask[y, x] = 1
            else:
                local_progress = (progress - (transition_center - transition_width/2)) / transition_width
                mask[y, x] = 1 / (1 + np.exp(-10 * (local_progress - 0.5)))
    
    mask = gaussian_filter(mask, sigma=5)
    
    result, _, _, _, _, blended_stack = multiresolution_blend(img1, img2, mask, N=6)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img1)
    axes[0].set_title('Beach Scene')
    axes[0].axis('off')
    axes[1].imshow(img2)
    axes[1].set_title('Mountain Bike Scene')
    axes[1].axis('off')
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('Horizontal Transition Mask')
    axes[2].axis('off')
    axes[3].imshow(result)
    axes[3].set_title('Beach-Mountain Blend')
    axes[3].axis('off')
    plt.suptitle('Beach to Mountain Bike Transition Blend', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    visualize_blending_process(img1, img2, mask, result, blended_stack, "Beach-Mountain Blending Process")
    
    img1 = img1.astype(np.float64) / 255.0 if img1.dtype == np.uint8 else img1
    img2 = img2.astype(np.float64) / 255.0 if img2.dtype == np.uint8 else img2

    imsave('results/beach_scene.jpg', (img1 * 255).astype(np.uint8))
    imsave('results/mountain_bike_scene.jpg', (img2 * 255).astype(np.uint8))
    imsave('results/beach_mountain_blend.jpg', (result * 255).astype(np.uint8))
    mask_gray = np.stack([mask, mask, mask], axis=2) 
    imsave('results/horizontal_mask.jpg', np.clip(mask_gray * 255, 0, 255).astype(np.uint8))
    print("Beach-mountain blend ")
    
    return result


if __name__ == "__main__":

    beach_bike_blend_demo()
    irregular_mask_demo()
    part_2_3_stacks_demo()
    
