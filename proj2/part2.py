import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d
import os
import matplotlib
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from scipy import ndimage
import skimage.transform as sktr
import math
from scipy.ndimage import gaussian_filter, sobel
from matplotlib.image import imsave

def create_gaussian_kernel(size, sigma):
    """
    Create a 2D Gaussian kernel
    """
    kernel_1d = cv2.getGaussianKernel(size, sigma)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d

def unsharp_mask_sequential(image, gaussian_kernel, alpha=1.5):
    """
    Apply unsharp masking using sequential operations:
    1. Blur the image with Gaussian filter
    2. Subtract blurred from original to get high frequencies
    3. Add scaled high frequencies back to original
    
    Formula: sharpened = original + alpha * (original - blurred)
    """
    if len(image.shape) == 3:
        result = np.zeros_like(image, dtype=np.float64)
        for channel in range(image.shape[2]):
            blurred = convolve2d(image[:,:,channel], gaussian_kernel, mode='same', boundary='fill', fillvalue=0)
            high_freq = image[:,:,channel] - blurred
            result[:,:,channel] = image[:,:,channel] + alpha * high_freq
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        blurred = convolve2d(image, gaussian_kernel, mode='same', boundary='fill', fillvalue=0)
        high_freq = image - blurred
        result = image + alpha * high_freq
        return np.clip(result, 0, 255).astype(np.uint8)

def create_unsharp_mask_kernel(gaussian_kernel, alpha=1.5):
    """
    Create a single unsharp mask kernel:
    unsharp_kernel = (1 + alpha) * delta - alpha * gaussian
    where delta is the impulse function (identity kernel)
    """
    identity = np.zeros_like(gaussian_kernel)
    center = gaussian_kernel.shape[0] // 2
    identity[center, center] = 1.0
    
    unsharp_kernel = (1 + alpha) * identity - alpha * gaussian_kernel
    return unsharp_kernel

def unsharp_mask_single_convolution(image, unsharp_kernel):
    """
    Apply unsharp masking using a single convolution operation
    """
    if len(image.shape) == 3:
        result = np.zeros_like(image, dtype=np.float64)
        for channel in range(image.shape[2]):
            result[:,:,channel] = convolve2d(image[:,:,channel], unsharp_kernel, mode='same', boundary='fill', fillvalue=0)
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        result = convolve2d(image, unsharp_kernel, mode='same', boundary='fill', fillvalue=0)
        return np.clip(result, 0, 255).astype(np.uint8)

def load_and_process_images():
    """
    Load and process images for sharpening demonstration
    """
    images_to_process = []
    
    taj_path = 'taj.jpg'  #
    if os.path.exists(taj_path):
        taj_image = cv2.imread(taj_path)
        if taj_image is not None:
            taj_image = cv2.cvtColor(taj_image, cv2.COLOR_BGR2RGB)
            images_to_process.append(('Taj Mahal', taj_image))
            print(f"Loaded Taj Mahal image: {taj_image.shape}")
    
    other_images = ['cameraman.jpg', 'IMG_8011.jpeg']  
    
    for img_path in other_images:
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images_to_process.append((os.path.splitext(img_path)[0], img))
                print(f"Loaded {img_path}: {img.shape}")
    
    return images_to_process

def blur_sharp_comparison_experiment():
    """
    Take a sharp image, blur it, then try to sharpen it again for comparison
    """
    print("\n" + "="*60)
    print("BLUR → SHARPEN COMPARISON EXPERIMENT")
    print("="*60)
    
    sharp_image = None
    for img_path in ['IMG_8011.jpeg', 'taj.jpg']:
        if os.path.exists(img_path):
            sharp_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if sharp_image is not None:
                print(f"Using {img_path} for blur-sharpen experiment")
                break
    
    if sharp_image is None:
        print("No image found for experiment, creating synthetic sharp image...")
    
    blur_kernel = create_gaussian_kernel(15, 3.0) 
    sharpen_kernel_params = create_gaussian_kernel(9, 1.5) 
    
    blurred_image = convolve2d(sharp_image, blur_kernel, mode='same', boundary='fill', fillvalue=0)
    blurred_image = np.clip(blurred_image, 0, 255).astype(np.uint8)
    
    unsharp_kernel = create_unsharp_mask_kernel(sharpen_kernel_params, alpha=1.5)
    sharpened_image = unsharp_mask_single_convolution(blurred_image, unsharp_kernel)
    
    dx_filter = np.array([[-1, 0, 1]]) / 2
    dy_filter = np.array([[-1], [0], [1]]) / 2
    
    def calculate_sharpness_metric(img):
        """Calculate a simple sharpness metric using gradient magnitude"""
        grad_x = convolve2d(img, dx_filter, mode='same', boundary='fill', fillvalue=0)
        grad_y = convolve2d(img, dy_filter, mode='same', boundary='fill', fillvalue=0)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        return np.mean(gradient_mag)
    
    original_sharpness = calculate_sharpness_metric(sharp_image)
    blurred_sharpness = calculate_sharpness_metric(blurred_image)
    recovered_sharpness = calculate_sharpness_metric(sharpened_image)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(sharp_image, cmap='gray')
    axes[0].set_title(f'Original Sharp\nSharpness: {original_sharpness:.2f}')
    axes[0].axis('off')
    
    axes[1].imshow(blurred_image, cmap='gray')
    axes[1].set_title(f'Artificially Blurred\nSharpness: {blurred_sharpness:.2f}')
    axes[1].axis('off')
    
    axes[2].imshow(sharpened_image, cmap='gray')
    axes[2].set_title(f'Unsharp Mask Recovery\nSharpness: {recovered_sharpness:.2f}')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Sharp → Blur → Sharpen Experiment', y=1.02, fontsize=14)
    plt.show()
    
    print(f"Original image sharpness metric: {original_sharpness:.3f}")
    print(f"Blurred image sharpness metric: {blurred_sharpness:.3f}")
    print(f"Recovered image sharpness metric: {recovered_sharpness:.3f}")
    print(f"Sharpness recovery: {(recovered_sharpness/original_sharpness)*100:.1f}% of original")
    
    recovery_vs_blur = (recovered_sharpness - blurred_sharpness) / (original_sharpness - blurred_sharpness)
    print(f"Recovery effectiveness: {recovery_vs_blur*100:.1f}% of lost sharpness recovered")
    
    return sharp_image, blurred_image, sharpened_image

def part_2_1_image_sharpening():
    """
    Main function for Part 2.1: Image Sharpening
    """
    print("="*60)
    print("PART 2.1: IMAGE SHARPENING WITH UNSHARP MASKING")
    print("="*60)
    
    kernel_size = 9
    sigma = 1.5
    alpha_values = [0.5, 1.0, 1.5, 2.0] 
    
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)
    print(f"Created Gaussian kernel: {kernel_size}x{kernel_size}, σ={sigma}")
    
    images_to_process = load_and_process_images()
    
    for img_name, image in images_to_process:
        print(f"\nProcessing: {img_name}")
        print(f"Image shape: {image.shape}")
        
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image.copy()
        
        fig, axes = plt.subplots(2, len(alpha_values) + 1, figsize=(20, 8))
        
        axes[0, 0].imshow(gray_image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(gray_image, cmap='gray')
        axes[1, 0].set_title('Original Image')
        axes[1, 0].axis('off')
        
        for i, alpha in enumerate(alpha_values):
            sharpened_seq = unsharp_mask_sequential(gray_image, gaussian_kernel, alpha)
            
            unsharp_kernel = create_unsharp_mask_kernel(gaussian_kernel, alpha)
            sharpened_single = unsharp_mask_single_convolution(gray_image, unsharp_kernel)
            
            max_diff = np.max(np.abs(sharpened_seq.astype(np.float64) - sharpened_single.astype(np.float64)))
            
            axes[0, i+1].imshow(sharpened_seq, cmap='gray')
            axes[0, i+1].set_title(f'Sequential α={alpha}')
            axes[0, i+1].axis('off')
            
            axes[1, i+1].imshow(sharpened_single, cmap='gray')
            axes[1, i+1].set_title(f'Single Conv α={alpha}\nDiff: {max_diff:.2f}')
            axes[1, i+1].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f'Unsharp Masking Results: {img_name}', y=1.02, fontsize=16)
        plt.show()
        
        default_alpha = 1.5
        unsharp_kernel = create_unsharp_mask_kernel(gaussian_kernel, default_alpha)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        identity = np.zeros_like(gaussian_kernel)
        center = gaussian_kernel.shape[0] // 2
        identity[center, center] = 1.0
        
        axes[0].imshow(identity * (1 + default_alpha), cmap='RdBu_r')
        axes[0].set_title(f'Identity × (1 + α)\nα = {default_alpha}')
        axes[0].axis('off')
        
        axes[1].imshow(-default_alpha * gaussian_kernel, cmap='RdBu_r')
        axes[1].set_title(f'-α × Gaussian\nα = {default_alpha}')
        axes[1].axis('off')
        
        axes[2].imshow(unsharp_kernel, cmap='RdBu_r')
        axes[2].set_title(f'Unsharp Mask Kernel\nα = {default_alpha}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f'Unsharp Mask Kernel Decomposition', y=1.02, fontsize=14)
        plt.show()
    
    blur_sharp_comparison_experiment()
    
    print("\n" + "="*60)
    print("KEY OBSERVATIONS:")
    print("="*60)
    print("1. Unsharp masking enhances edges and fine details")
    print("2. Higher α values create stronger sharpening effects")
    print("3. Sequential and single convolution methods are equivalent")
    print("4. Sharpening cannot fully recover lost information from blurring")
    print("5. Over-sharpening (high α) can introduce artifacts")

def save_sharpening_results_for_webpage():
    """
    Generate and save sharpening result images for the webpage
    """
    os.makedirs('results', exist_ok=True)
    print("Generating sharpening images for webpage...")
    
    test_image = None
    for img_path in ['taj.jpg', 'IMG_8011.jpeg', 'cameraman.jpg']:
        if os.path.exists(img_path):
            test_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if test_image is not None:
                break
    
    gaussian_kernel = create_gaussian_kernel(9, 1.5)
    alpha_values = [0.5, 1.0, 1.5, 2.0]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(test_image, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/sharpening_original.png', dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    for alpha in alpha_values:
        unsharp_kernel = create_unsharp_mask_kernel(gaussian_kernel, alpha)
        sharpened = unsharp_mask_single_convolution(test_image, unsharp_kernel)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(sharpened, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'results/sharpening_alpha_{alpha}.png', dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    unsharp_kernel = create_unsharp_mask_kernel(gaussian_kernel, 1.5)
    plt.figure(figsize=(8, 8))
    plt.imshow(unsharp_kernel, cmap='RdBu_r')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/unsharp_mask_kernel.png', dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    if test_image is not None:
        blur_kernel = create_gaussian_kernel(15, 3.0)
        blurred = convolve2d(test_image, blur_kernel, mode='same', boundary='fill', fillvalue=0)
        blurred = np.clip(blurred, 0, 255).astype(np.uint8)
        
        sharpen_kernel = create_unsharp_mask_kernel(create_gaussian_kernel(9, 1.5), 1.5)
        recovered = unsharp_mask_single_convolution(blurred, sharpen_kernel)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(blurred, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('results/artificially_blurred.png', dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(recovered, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('results/sharpening_recovery.png', dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    print("Sharpening webpage images saved!")

def find_centers(p1, p2):
    """Find center point between two points."""
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy

def align_image_centers(im1, im2, pts):
    """Align the centers of corresponding points in two images."""
    p1, p2, p3, p4 = pts

    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1, im2, pts):
    """Rescale images to match the distance between corresponding points."""
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2 / len1
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale, channel_axis=2 if len(im1.shape) == 3 else None)
    else:
        im2 = sktr.rescale(im2, 1. / dscale, channel_axis=2 if len(im2.shape) == 3 else None)
    return im1, im2

def rotate_im1(im1, im2, pts):
    """Rotate first image to match orientation of second image."""
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta * 180 / np.pi)
    return im1, dtheta

def hybrid_image(im1, im2, sigma1, sigma2):
    """Create a hybrid image by combining high frequencies from im1 and low frequencies from im2."""
    low_frequencies = gaussian_filter(im2, sigma2)
    low_pass_im1 = gaussian_filter(im1, sigma1)
    high_frequencies = im1 - low_pass_im1
    hybrid = high_frequencies + low_frequencies
    hybrid = np.clip(hybrid, 0, 1)
    return hybrid

def rgb_to_gray(image):
    """Convert RGB image to grayscale."""
    if len(image.shape) == 3:
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return image

def recenter(im, r, c):
    """Recenter image around specified point."""
    R, C = im.shape[:2]
    if len(im.shape) == 3:
        rpad = int(np.abs(2*r+1 - R))
        cpad = int(np.abs(2*c+1 - C))
        return np.pad(
            im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
                 (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad),
                 (0, 0)], 'constant')
    else:
        rpad = int(np.abs(2*r+1 - R))
        cpad = int(np.abs(2*c+1 - C))
        return np.pad(
            im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
                 (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad)], 'constant')
                 
def save_hybrid_results(im1, im2, im1_aligned, im2_aligned, hybrid, sigma1, sigma2):
    """Save aligned images and final hybrid to disk."""
    imsave("results/high_freq_source_derek.jpg", im1)
    imsave("results/low_freq_source_nutmeg.jpg", im2)
    imsave("results/derek_aligned.jpg", im1_aligned)
    imsave("results/nutmeg_aligned.jpg", im2_aligned)
    imsave("results/final_hybrid.jpg", hybrid)

    with open("results/parameters.txt", "w") as f:
        f.write(f"High-pass sigma (Derek): {sigma1}\n")
        f.write(f"Low-pass sigma (Nutmeg): {sigma2}\n")

def save_frequency_analysis(im1, im2, hybrid, sigma1, sigma2):
    """Save Fourier transform images for report."""
    gray1 = rgb_to_gray(im1)
    gray2 = rgb_to_gray(im2)
    gray_hybrid = rgb_to_gray(hybrid)

    f1 = np.fft.fftshift(np.fft.fft2(gray1))
    f2 = np.fft.fftshift(np.fft.fft2(gray2))
    f_hybrid = np.fft.fftshift(np.fft.fft2(gray_hybrid))

    log_f1 = np.log(np.abs(f1) + 1)
    log_f2 = np.log(np.abs(f2) + 1)
    log_fh = np.log(np.abs(f_hybrid) + 1)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(log_f1, cmap='gray')
    axs[0].set_title(f'FFT of Derek\nσ={sigma1}')
    axs[0].axis('off')

    axs[1].imshow(log_f2, cmap='gray')
    axs[1].set_title(f'FFT of Nutmeg\nσ={sigma2}')
    axs[1].axis('off')

    axs[2].imshow(log_fh, cmap='gray')
    axs[2].set_title('FFT of Hybrid')
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig("results/frequency_analysis.png")
    plt.close()

def build_gaussian_pyramid(image, N=5):
    """
    Build a Gaussian pyramid with N levels.
    
    Args:
        image: Input grayscale image
        N: Number of pyramid levels
        
    Returns:
        List of images representing the Gaussian pyramid
    """
    if len(image.shape) == 3:
        gray_image = rgb_to_gray(image)
    else:
        gray_image = image.copy()
    
    if gray_image.dtype != np.float64:
        gray_image = gray_image.astype(np.float64)
    
    gaussian_pyramid = [gray_image]
    current = gray_image
    
    for i in range(N-1):
        blurred = gaussian_filter(current, sigma=1.0)
        
        downsampled = blurred[::2, ::2]
        gaussian_pyramid.append(downsampled)
        current = downsampled
    
    return gaussian_pyramid

def build_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    N = len(gaussian_pyramid)
    
    for i in range(N-1):
        current_level = gaussian_pyramid[i]
        next_level = gaussian_pyramid[i+1]
        
        upsampled = np.repeat(np.repeat(next_level, 2, axis=0), 2, axis=1)
        
        h_current, w_current = current_level.shape
        h_up, w_up = upsampled.shape
        
        if h_up > h_current:
            upsampled = upsampled[:h_current, :]
        if w_up > w_current:
            upsampled = upsampled[:, :w_current]
            
        if h_up < h_current:
            pad_h = h_current - h_up
            upsampled = np.pad(upsampled, ((0, pad_h), (0, 0)), 'edge')
        if w_up < w_current:
            pad_w = w_current - w_up
            upsampled = np.pad(upsampled, ((0, 0), (0, pad_w)), 'edge')
        
        upsampled_blurred = gaussian_filter(upsampled, sigma=0.5)
        
        laplacian = current_level - upsampled_blurred
        laplacian_pyramid.append(laplacian)
    
    laplacian_pyramid.append(gaussian_pyramid[-1])
    
    return laplacian_pyramid


def pyramids(image, N=5):
    """
    Compute and display Gaussian and Laplacian pyramids for an image.
    
    Args:
        image: Input image (can be color or grayscale)
        N: Number of pyramid levels (default: 5)
    """
    print(f"Building {N}-level pyramids...")
    
    gaussian_pyramid = build_gaussian_pyramid(image, N)
    laplacian_pyramid = build_laplacian_pyramid(gaussian_pyramid)
    
    fig, axes = plt.subplots(2, N, figsize=(3*N, 6))
    
    if N == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(N):
        if i < len(gaussian_pyramid):
            gp = gaussian_pyramid[i]
            if gp.max() > 1.0:
                gp = gp / 255.0
            axes[0, i].imshow(gaussian_pyramid[i], cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'Gaussian Level {i}\n{gaussian_pyramid[i].shape}')
            axes[0, i].axis('off')
        else:
            axes[0, i].axis('off')
        
        if i < len(laplacian_pyramid):
            lap_display = laplacian_pyramid[i]
            
            if lap_display.max() > lap_display.min():
                lap_normalized = (lap_display - lap_display.min()) / (lap_display.max() - lap_display.min())
            else:
                lap_normalized = np.zeros_like(lap_display)
            
            axes[1, i].imshow(lap_normalized, cmap='gray')
            axes[1, i].set_title(f'Laplacian Level {i}\n{lap_display.shape}')
            axes[1, i].axis('off')
        else:
            axes[1, i].axis('off')
    
    plt.suptitle('Gaussian and Laplacian Pyramids', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print("\nPyramid Statistics:")
    print("-" * 40)
    for i, (gauss, lap) in enumerate(zip(gaussian_pyramid, laplacian_pyramid)):
        print(f"Level {i}:")
        print(f"  Gaussian: {gauss.shape}, range: [{gauss.min():.3f}, {gauss.max():.3f}]")
        print(f"  Laplacian: {lap.shape}, range: [{lap.min():.3f}, {lap.max():.3f}]")
    
    return gaussian_pyramid, laplacian_pyramid

def save_pyramid_analysis(image, N=5, output_dir='results'):
    """
    Save pyramid analysis results to files.
    
    Args:
        image: Input image
        N: Number of pyramid levels
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    gaussian_pyramid = build_gaussian_pyramid(image, N)
    laplacian_pyramid = build_laplacian_pyramid(gaussian_pyramid)
    
    fig, axes = plt.subplots(2, N, figsize=(3*N, 6))
    
    if N == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(N):
        if i < len(gaussian_pyramid):
            axes[0, i].imshow(gaussian_pyramid[i], cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'Gaussian Level {i}')
            axes[0, i].axis('off')
        
        if i < len(laplacian_pyramid):
            lap_display = laplacian_pyramid[i]
            if lap_display.max() > lap_display.min():
                lap_normalized = (lap_display - lap_display.min()) / (lap_display.max() - lap_display.min())
            else:
                lap_normalized = np.zeros_like(lap_display)
            
            axes[1, i].imshow(lap_normalized, cmap='gray')
            axes[1, i].set_title(f'Laplacian Level {i}')
            axes[1, i].axis('off')
    
    plt.suptitle('Gaussian and Laplacian Pyramids of Hybrid Image')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hybrid_pyramids.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    for i, (gauss, lap) in enumerate(zip(gaussian_pyramid, laplacian_pyramid)):
        plt.figure(figsize=(8, 8))
        plt.imshow(gauss, cmap='gray', vmin=0, vmax=1)
        plt.title(f'Gaussian Pyramid Level {i}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/gaussian_level_{i}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(8, 8))
        if lap.max() > lap.min():
            lap_normalized = (lap - lap.min()) / (lap.max() - lap.min())
        else:
            lap_normalized = np.zeros_like(lap)
        
        plt.imshow(lap_normalized, cmap='gray')
        plt.title(f'Laplacian Pyramid Level {i}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/laplacian_level_{i}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Pyramid analysis results saved to {output_dir}/ directory")
    return gaussian_pyramid, laplacian_pyramid

def show_fourier_analysis(im1, im2, hybrid, sigma1, sigma2):
    """Show frequency domain representations for hybrid analysis."""
    gray1 = rgb_to_gray(im1)
    gray2 = rgb_to_gray(im2)
    gray_hybrid = rgb_to_gray(hybrid)

    f1 = np.fft.fftshift(np.fft.fft2(gray1))
    f2 = np.fft.fftshift(np.fft.fft2(gray2))
    f_hybrid = np.fft.fftshift(np.fft.fft2(gray_hybrid))

    log_f1 = np.log(np.abs(f1) + 1)
    log_f2 = np.log(np.abs(f2) + 1)
    log_fh = np.log(np.abs(f_hybrid) + 1)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(log_f1, cmap='gray')
    axs[0].set_title(f'FFT of Derek (High-pass)\nσ={sigma1}')
    axs[0].axis('off')

    axs[1].imshow(log_f2, cmap='gray')
    axs[1].set_title(f'FFT of Nutmeg (Low-pass)\nσ={sigma2}')
    axs[1].axis('off')

    axs[2].imshow(log_fh, cmap='gray')
    axs[2].set_title('FFT of Hybrid Image')
    axs[2].axis('off')

    plt.tight_layout()
    plt.suptitle('Fourier Transform Analysis (Log Scale)', y=1.05)
    plt.show()

def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    print('Suggested points: both eyes, or eye centers and nose tip')

    plt.figure(figsize=(12, 8))
    plt.imshow(im1)
    plt.title("Image 1 (Derek): Click 2 alignment points\n(e.g., left eye center, right eye center)")
    pts1 = plt.ginput(2, timeout=30)
    plt.close()
    if len(pts1) < 2:
        raise ValueError("Not enough points selected on the first image")

    plt.figure(figsize=(12, 8))
    plt.imshow(im2)
    plt.title("Image 2 (Nutmeg): Click corresponding points\n(same features as Image 1)")
    pts2 = plt.ginput(2, timeout=30)
    plt.close()
    if len(pts2) < 2:
        raise ValueError("Not enough points selected on the second image")

    return (*pts1, *pts2)


def match_img_size(im1, im2):
    print(f"Input shapes: im1={im1.shape}, im2={im2.shape}")

    if len(im1.shape) == 2:
        im1 = np.stack([im1] * 3, axis=2)
    if len(im2.shape) == 2:
        im2 = np.stack([im2] * 3, axis=2)

    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape

    if c1 != c2:
        target_channels = max(c1, c2)
        if c1 < target_channels:
            im1_new = np.zeros((h1, w1, target_channels))
            for i in range(target_channels):
                im1_new[:, :, i] = im1[:, :, min(i, c1 - 1)]
            im1 = im1_new
        if c2 < target_channels:
            im2_new = np.zeros((h2, w2, target_channels))
            for i in range(target_channels):
                im2_new[:, :, i] = im2[:, :, min(i, c2 - 1)]
            im2 = im2_new

    min_h = min(h1, h2)
    min_w = min(w1, w2)
    h1_start = (h1 - min_h) // 2
    w1_start = (w1 - min_w) // 2
    h2_start = (h2 - min_h) // 2
    w2_start = (w2 - min_w) // 2

    im1 = im1[h1_start:h1_start + min_h, w1_start:w1_start + min_w]
    im2 = im2[h2_start:h2_start + min_h, w2_start:w2_start + min_w]

    if im1.max() > 1.0:
        im1 = im1.astype(np.float64) / 255.0
    if im2.max() > 1.0:
        im2 = im2.astype(np.float64) / 255.0

    im1 = np.clip(im1, 0, 1)
    im2 = np.clip(im2, 0, 1)

    print(f"Final aligned shapes: im1={im1.shape}, im2={im2.shape}")
    return im1, im2


def align_images(im1, im2):
    pts = get_points(im1, im2)
    print(f"Selected points: {pts}")

    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2


def improved_hybrid_processing():
    print("\n" + "=" * 60)
    print("IMPROVED PART 2.2: HYBRID IMAGES")
    print("=" * 60)

    os.makedirs('results', exist_ok=True)
    im1_path = 'DerekPicture.jpg'
    im2_path = 'nutmeg.jpg'
    if not (os.path.exists(im1_path) and os.path.exists(im2_path)):
        raise FileNotFoundError("Make sure both 'DerekPicture.jpg' and 'nutmeg.jpg' are in the current directory.")

    im1 = cv2.cvtColor(cv2.imread(im1_path), cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(cv2.imread(im2_path), cv2.COLOR_BGR2RGB)
    print(f"Original Derek shape: {im1.shape}")
    print(f"Original Nutmeg shape: {im2.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(im1)
    axes[0].set_title('Derek (Image 1) - High Freq Source')
    axes[0].axis('off')
    axes[1].imshow(im2)
    axes[1].set_title('Nutmeg (Image 2) - Low Freq Source')
    axes[1].axis('off')
    plt.tight_layout()
    plt.suptitle('Original Images - Study these before alignment', y=1.02, fontsize=14)
    plt.show()

    print("\nStep 1: Careful alignment (this is crucial for good results)")
    try:
        im1_aligned, im2_aligned = align_images(im1, im2)
        print("Alignment successful!")
    except Exception as e:
        print(f"Alignment failed: {e}")
        print("Using basic size matching...")
        im1_aligned, im2_aligned = match_img_size(im1, im2)

    print("\nStep 2: Create final hybrid with optimal parameters...")
    sigma1, sigma2 = 2, 5
    final_hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

    print("\nStep 3: Save and visualize results...")
    save_hybrid_results(im1, im2, im1_aligned, im2_aligned, final_hybrid, sigma1, sigma2)
    save_frequency_analysis(im1_aligned, im2_aligned, final_hybrid, sigma1, sigma2)
    save_pyramid_analysis(final_hybrid, N=5)

    pyramids(final_hybrid, N=5)
    show_fourier_analysis(im1_aligned, im2_aligned, final_hybrid, sigma1, sigma2)

    print("\n🎯 Final hybrid image created with σ1=2, σ2=5")
    return final_hybrid, im1_aligned, im2_aligned

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d
import os
import matplotlib
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from scipy import ndimage
import skimage.transform as sktr
import math
from scipy.ndimage import gaussian_filter, sobel
from matplotlib.image import imsave

def create_gaussian_kernel(size, sigma):
    """
    Create a 2D Gaussian kernel
    """
    kernel_1d = cv2.getGaussianKernel(size, sigma)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d

def unsharp_mask_sequential(image, gaussian_kernel, alpha=1.5):
    """
    Apply unsharp masking using sequential operations:
    1. Blur the image with Gaussian filter
    2. Subtract blurred from original to get high frequencies
    3. Add scaled high frequencies back to original
    
    Formula: sharpened = original + alpha * (original - blurred)
    """
    if len(image.shape) == 3:
        result = np.zeros_like(image, dtype=np.float64)
        for channel in range(image.shape[2]):
            blurred = convolve2d(image[:,:,channel], gaussian_kernel, mode='same', boundary='fill', fillvalue=0)
            high_freq = image[:,:,channel] - blurred
            result[:,:,channel] = image[:,:,channel] + alpha * high_freq
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        blurred = convolve2d(image, gaussian_kernel, mode='same', boundary='fill', fillvalue=0)
        high_freq = image - blurred
        result = image + alpha * high_freq
        return np.clip(result, 0, 255).astype(np.uint8)

def create_unsharp_mask_kernel(gaussian_kernel, alpha=1.5):
    """
    Create a single unsharp mask kernel:
    unsharp_kernel = (1 + alpha) * delta - alpha * gaussian
    where delta is the impulse function (identity kernel)
    """
    identity = np.zeros_like(gaussian_kernel)
    center = gaussian_kernel.shape[0] // 2
    identity[center, center] = 1.0
    
    unsharp_kernel = (1 + alpha) * identity - alpha * gaussian_kernel
    return unsharp_kernel

def unsharp_mask_single_convolution(image, unsharp_kernel):
    """
    Apply unsharp masking using a single convolution operation
    """
    if len(image.shape) == 3:
        result = np.zeros_like(image, dtype=np.float64)
        for channel in range(image.shape[2]):
            result[:,:,channel] = convolve2d(image[:,:,channel], unsharp_kernel, mode='same', boundary='fill', fillvalue=0)
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        result = convolve2d(image, unsharp_kernel, mode='same', boundary='fill', fillvalue=0)
        return np.clip(result, 0, 255).astype(np.uint8)

def load_and_process_images():
    """
    Load and process images for sharpening demonstration
    """
    images_to_process = []
    
    taj_path = 'taj.jpg'  #
    if os.path.exists(taj_path):
        taj_image = cv2.imread(taj_path)
        if taj_image is not None:
            taj_image = cv2.cvtColor(taj_image, cv2.COLOR_BGR2RGB)
            images_to_process.append(('Taj Mahal', taj_image))
            print(f"Loaded Taj Mahal image: {taj_image.shape}")
    
    other_images = ['cameraman.jpg', 'IMG_8011.jpeg']  
    
    for img_path in other_images:
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images_to_process.append((os.path.splitext(img_path)[0], img))
                print(f"Loaded {img_path}: {img.shape}")
    
    return images_to_process

def blur_sharp_comparison_experiment():
    """
    Take a sharp image, blur it, then try to sharpen it again for comparison
    """
    print("\n" + "="*60)
    print("BLUR → SHARPEN COMPARISON EXPERIMENT")
    print("="*60)
    
    sharp_image = None
    for img_path in ['IMG_8011.jpeg', 'taj.jpg']:
        if os.path.exists(img_path):
            sharp_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if sharp_image is not None:
                print(f"Using {img_path} for blur-sharpen experiment")
                break
    
    if sharp_image is None:
        print("No image found for experiment, creating synthetic sharp image...")
    
    blur_kernel = create_gaussian_kernel(15, 3.0) 
    sharpen_kernel_params = create_gaussian_kernel(9, 1.5) 
    
    blurred_image = convolve2d(sharp_image, blur_kernel, mode='same', boundary='fill', fillvalue=0)
    blurred_image = np.clip(blurred_image, 0, 255).astype(np.uint8)
    
    unsharp_kernel = create_unsharp_mask_kernel(sharpen_kernel_params, alpha=1.5)
    sharpened_image = unsharp_mask_single_convolution(blurred_image, unsharp_kernel)
    
    dx_filter = np.array([[-1, 0, 1]]) / 2
    dy_filter = np.array([[-1], [0], [1]]) / 2
    
    def calculate_sharpness_metric(img):
        """Calculate a simple sharpness metric using gradient magnitude"""
        grad_x = convolve2d(img, dx_filter, mode='same', boundary='fill', fillvalue=0)
        grad_y = convolve2d(img, dy_filter, mode='same', boundary='fill', fillvalue=0)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        return np.mean(gradient_mag)
    
    original_sharpness = calculate_sharpness_metric(sharp_image)
    blurred_sharpness = calculate_sharpness_metric(blurred_image)
    recovered_sharpness = calculate_sharpness_metric(sharpened_image)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(sharp_image, cmap='gray')
    axes[0].set_title(f'Original Sharp\nSharpness: {original_sharpness:.2f}')
    axes[0].axis('off')
    
    axes[1].imshow(blurred_image, cmap='gray')
    axes[1].set_title(f'Artificially Blurred\nSharpness: {blurred_sharpness:.2f}')
    axes[1].axis('off')
    
    axes[2].imshow(sharpened_image, cmap='gray')
    axes[2].set_title(f'Unsharp Mask Recovery\nSharpness: {recovered_sharpness:.2f}')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Sharp → Blur → Sharpen Experiment', y=1.02, fontsize=14)
    plt.show()
    
    print(f"Original image sharpness metric: {original_sharpness:.3f}")
    print(f"Blurred image sharpness metric: {blurred_sharpness:.3f}")
    print(f"Recovered image sharpness metric: {recovered_sharpness:.3f}")
    print(f"Sharpness recovery: {(recovered_sharpness/original_sharpness)*100:.1f}% of original")
    
    recovery_vs_blur = (recovered_sharpness - blurred_sharpness) / (original_sharpness - blurred_sharpness)
    print(f"Recovery effectiveness: {recovery_vs_blur*100:.1f}% of lost sharpness recovered")
    
    return sharp_image, blurred_image, sharpened_image

def part_2_1_image_sharpening():
    """
    Main function for Part 2.1: Image Sharpening
    """
    print("="*60)
    print("PART 2.1: IMAGE SHARPENING WITH UNSHARP MASKING")
    print("="*60)
    
    kernel_size = 9
    sigma = 1.5
    alpha_values = [0.5, 1.0, 1.5, 2.0] 
    
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)
    print(f"Created Gaussian kernel: {kernel_size}x{kernel_size}, σ={sigma}")
    
    images_to_process = load_and_process_images()
    
    for img_name, image in images_to_process:
        print(f"\nProcessing: {img_name}")
        print(f"Image shape: {image.shape}")
        
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image.copy()
        
        fig, axes = plt.subplots(2, len(alpha_values) + 1, figsize=(20, 8))
        
        axes[0, 0].imshow(gray_image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(gray_image, cmap='gray')
        axes[1, 0].set_title('Original Image')
        axes[1, 0].axis('off')
        
        for i, alpha in enumerate(alpha_values):
            sharpened_seq = unsharp_mask_sequential(gray_image, gaussian_kernel, alpha)
            
            unsharp_kernel = create_unsharp_mask_kernel(gaussian_kernel, alpha)
            sharpened_single = unsharp_mask_single_convolution(gray_image, unsharp_kernel)
            
            max_diff = np.max(np.abs(sharpened_seq.astype(np.float64) - sharpened_single.astype(np.float64)))
            
            axes[0, i+1].imshow(sharpened_seq, cmap='gray')
            axes[0, i+1].set_title(f'Sequential α={alpha}')
            axes[0, i+1].axis('off')
            
            axes[1, i+1].imshow(sharpened_single, cmap='gray')
            axes[1, i+1].set_title(f'Single Conv α={alpha}\nDiff: {max_diff:.2f}')
            axes[1, i+1].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f'Unsharp Masking Results: {img_name}', y=1.02, fontsize=16)
        plt.show()
        
        default_alpha = 1.5
        unsharp_kernel = create_unsharp_mask_kernel(gaussian_kernel, default_alpha)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        identity = np.zeros_like(gaussian_kernel)
        center = gaussian_kernel.shape[0] // 2
        identity[center, center] = 1.0
        
        axes[0].imshow(identity * (1 + default_alpha), cmap='RdBu_r')
        axes[0].set_title(f'Identity × (1 + α)\nα = {default_alpha}')
        axes[0].axis('off')
        
        axes[1].imshow(-default_alpha * gaussian_kernel, cmap='RdBu_r')
        axes[1].set_title(f'-α × Gaussian\nα = {default_alpha}')
        axes[1].axis('off')
        
        axes[2].imshow(unsharp_kernel, cmap='RdBu_r')
        axes[2].set_title(f'Unsharp Mask Kernel\nα = {default_alpha}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f'Unsharp Mask Kernel Decomposition', y=1.02, fontsize=14)
        plt.show()
    
    blur_sharp_comparison_experiment()
    
    print("\n" + "="*60)
    print("KEY OBSERVATIONS:")
    print("="*60)
    print("1. Unsharp masking enhances edges and fine details")
    print("2. Higher α values create stronger sharpening effects")
    print("3. Sequential and single convolution methods are equivalent")
    print("4. Sharpening cannot fully recover lost information from blurring")
    print("5. Over-sharpening (high α) can introduce artifacts")

def save_sharpening_results_for_webpage():
    """
    Generate and save sharpening result images for the webpage
    """
    os.makedirs('results', exist_ok=True)
    print("Generating sharpening images for webpage...")
    
    test_image = None
    for img_path in ['taj.jpg', 'IMG_8011.jpeg', 'cameraman.jpg']:
        if os.path.exists(img_path):
            test_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if test_image is not None:
                break
    
    gaussian_kernel = create_gaussian_kernel(9, 1.5)
    alpha_values = [0.5, 1.0, 1.5, 2.0]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(test_image, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/sharpening_original.png', dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    for alpha in alpha_values:
        unsharp_kernel = create_unsharp_mask_kernel(gaussian_kernel, alpha)
        sharpened = unsharp_mask_single_convolution(test_image, unsharp_kernel)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(sharpened, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'results/sharpening_alpha_{alpha}.png', dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    unsharp_kernel = create_unsharp_mask_kernel(gaussian_kernel, 1.5)
    plt.figure(figsize=(8, 8))
    plt.imshow(unsharp_kernel, cmap='RdBu_r')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/unsharp_mask_kernel.png', dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    if test_image is not None:
        blur_kernel = create_gaussian_kernel(15, 3.0)
        blurred = convolve2d(test_image, blur_kernel, mode='same', boundary='fill', fillvalue=0)
        blurred = np.clip(blurred, 0, 255).astype(np.uint8)
        
        sharpen_kernel = create_unsharp_mask_kernel(create_gaussian_kernel(9, 1.5), 1.5)
        recovered = unsharp_mask_single_convolution(blurred, sharpen_kernel)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(blurred, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('results/artificially_blurred.png', dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(recovered, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('results/sharpening_recovery.png', dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    print("Sharpening webpage images saved!")

def find_centers(p1, p2):
    """Find center point between two points."""
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy

def align_image_centers(im1, im2, pts):
    """Align the centers of corresponding points in two images."""
    p1, p2, p3, p4 = pts

    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1, im2, pts):
    """Rescale images to match the distance between corresponding points."""
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2 / len1
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale, channel_axis=2 if len(im1.shape) == 3 else None)
    else:
        im2 = sktr.rescale(im2, 1. / dscale, channel_axis=2 if len(im2.shape) == 3 else None)
    return im1, im2

def rotate_im1(im1, im2, pts):
    """Rotate first image to match orientation of second image."""
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta * 180 / np.pi)
    return im1, dtheta

def hybrid_image(im1, im2, sigma1, sigma2):
    """Create a hybrid image by combining high frequencies from im1 and low frequencies from im2."""
    low_frequencies = gaussian_filter(im2, sigma2)
    low_pass_im1 = gaussian_filter(im1, sigma1)
    high_frequencies = im1 - low_pass_im1
    hybrid = high_frequencies + low_frequencies
    hybrid = np.clip(hybrid, 0, 1)
    return hybrid

def rgb_to_gray(image):
    """Convert RGB image to grayscale."""
    if len(image.shape) == 3:
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return image

def recenter(im, r, c):
    """Recenter image around specified point."""
    R, C = im.shape[:2]
    if len(im.shape) == 3:
        rpad = int(np.abs(2*r+1 - R))
        cpad = int(np.abs(2*c+1 - C))
        return np.pad(
            im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
                 (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad),
                 (0, 0)], 'constant')
    else:
        rpad = int(np.abs(2*r+1 - R))
        cpad = int(np.abs(2*c+1 - C))
        return np.pad(
            im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
                 (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad)], 'constant')
                 
def save_hybrid_results(im1, im2, im1_aligned, im2_aligned, hybrid, sigma1, sigma2):
    """Save aligned images and final hybrid to disk."""
    imsave("results/high_freq_source_derek.jpg", im1)
    imsave("results/low_freq_source_nutmeg.jpg", im2)
    imsave("results/derek_aligned.jpg", im1_aligned)
    imsave("results/nutmeg_aligned.jpg", im2_aligned)
    imsave("results/final_hybrid.jpg", hybrid)

    with open("results/parameters.txt", "w") as f:
        f.write(f"High-pass sigma (Derek): {sigma1}\n")
        f.write(f"Low-pass sigma (Nutmeg): {sigma2}\n")

def save_frequency_analysis(im1, im2, hybrid, sigma1, sigma2):
    """Save Fourier transform images for report."""
    gray1 = rgb_to_gray(im1)
    gray2 = rgb_to_gray(im2)
    gray_hybrid = rgb_to_gray(hybrid)

    f1 = np.fft.fftshift(np.fft.fft2(gray1))
    f2 = np.fft.fftshift(np.fft.fft2(gray2))
    f_hybrid = np.fft.fftshift(np.fft.fft2(gray_hybrid))

    log_f1 = np.log(np.abs(f1) + 1)
    log_f2 = np.log(np.abs(f2) + 1)
    log_fh = np.log(np.abs(f_hybrid) + 1)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(log_f1, cmap='gray')
    axs[0].set_title(f'FFT of Derek\nσ={sigma1}')
    axs[0].axis('off')

    axs[1].imshow(log_f2, cmap='gray')
    axs[1].set_title(f'FFT of Nutmeg\nσ={sigma2}')
    axs[1].axis('off')

    axs[2].imshow(log_fh, cmap='gray')
    axs[2].set_title('FFT of Hybrid')
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig("results/frequency_analysis.png")
    plt.close()

def build_gaussian_pyramid(image, N=5):
    """
    Build a Gaussian pyramid with N levels.
    
    Args:
        image: Input grayscale image
        N: Number of pyramid levels
        
    Returns:
        List of images representing the Gaussian pyramid
    """
    if len(image.shape) == 3:
        gray_image = rgb_to_gray(image)
    else:
        gray_image = image.copy()
    
    if gray_image.dtype != np.float64:
        gray_image = gray_image.astype(np.float64)
    
    gaussian_pyramid = [gray_image]
    current = gray_image
    
    for i in range(N-1):
        blurred = gaussian_filter(current, sigma=1.0)
        
        downsampled = blurred[::2, ::2]
        gaussian_pyramid.append(downsampled)
        current = downsampled
    
    return gaussian_pyramid

def build_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    N = len(gaussian_pyramid)
    
    for i in range(N-1):
        current_level = gaussian_pyramid[i]
        next_level = gaussian_pyramid[i+1]
        
        upsampled = np.repeat(np.repeat(next_level, 2, axis=0), 2, axis=1)
        
        h_current, w_current = current_level.shape
        h_up, w_up = upsampled.shape
        
        if h_up > h_current:
            upsampled = upsampled[:h_current, :]
        if w_up > w_current:
            upsampled = upsampled[:, :w_current]
            
        if h_up < h_current:
            pad_h = h_current - h_up
            upsampled = np.pad(upsampled, ((0, pad_h), (0, 0)), 'edge')
        if w_up < w_current:
            pad_w = w_current - w_up
            upsampled = np.pad(upsampled, ((0, 0), (0, pad_w)), 'edge')
        
        upsampled_blurred = gaussian_filter(upsampled, sigma=0.5)
        
        laplacian = current_level - upsampled_blurred
        laplacian_pyramid.append(laplacian)
    
    laplacian_pyramid.append(gaussian_pyramid[-1])
    
    return laplacian_pyramid


def pyramids(image, N=5):
    """
    Compute and display Gaussian and Laplacian pyramids for an image.
    
    Args:
        image: Input image (can be color or grayscale)
        N: Number of pyramid levels (default: 5)
    """
    print(f"Building {N}-level pyramids...")
    
    gaussian_pyramid = build_gaussian_pyramid(image, N)
    laplacian_pyramid = build_laplacian_pyramid(gaussian_pyramid)
    
    fig, axes = plt.subplots(2, N, figsize=(3*N, 6))
    
    if N == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(N):
        if i < len(gaussian_pyramid):
            gp = gaussian_pyramid[i]
            if gp.max() > 1.0:
                gp = gp / 255.0
            axes[0, i].imshow(gaussian_pyramid[i], cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'Gaussian Level {i}\n{gaussian_pyramid[i].shape}')
            axes[0, i].axis('off')
        else:
            axes[0, i].axis('off')
        
        if i < len(laplacian_pyramid):
            lap_display = laplacian_pyramid[i]
            
            if lap_display.max() > lap_display.min():
                lap_normalized = (lap_display - lap_display.min()) / (lap_display.max() - lap_display.min())
            else:
                lap_normalized = np.zeros_like(lap_display)
            
            axes[1, i].imshow(lap_normalized, cmap='gray')
            axes[1, i].set_title(f'Laplacian Level {i}\n{lap_display.shape}')
            axes[1, i].axis('off')
        else:
            axes[1, i].axis('off')
    
    plt.suptitle('Gaussian and Laplacian Pyramids', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print("\nPyramid Statistics:")
    print("-" * 40)
    for i, (gauss, lap) in enumerate(zip(gaussian_pyramid, laplacian_pyramid)):
        print(f"Level {i}:")
        print(f"  Gaussian: {gauss.shape}, range: [{gauss.min():.3f}, {gauss.max():.3f}]")
        print(f"  Laplacian: {lap.shape}, range: [{lap.min():.3f}, {lap.max():.3f}]")
    
    return gaussian_pyramid, laplacian_pyramid

def save_pyramid_analysis(image, N=5, output_dir='results'):
    """
    Save pyramid analysis results to files.
    
    Args:
        image: Input image
        N: Number of pyramid levels
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    gaussian_pyramid = build_gaussian_pyramid(image, N)
    laplacian_pyramid = build_laplacian_pyramid(gaussian_pyramid)
    
    fig, axes = plt.subplots(2, N, figsize=(3*N, 6))
    
    if N == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(N):
        if i < len(gaussian_pyramid):
            axes[0, i].imshow(gaussian_pyramid[i], cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'Gaussian Level {i}')
            axes[0, i].axis('off')
        
        if i < len(laplacian_pyramid):
            lap_display = laplacian_pyramid[i]
            if lap_display.max() > lap_display.min():
                lap_normalized = (lap_display - lap_display.min()) / (lap_display.max() - lap_display.min())
            else:
                lap_normalized = np.zeros_like(lap_display)
            
            axes[1, i].imshow(lap_normalized, cmap='gray')
            axes[1, i].set_title(f'Laplacian Level {i}')
            axes[1, i].axis('off')
    
    plt.suptitle('Gaussian and Laplacian Pyramids of Hybrid Image')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hybrid_pyramids.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    for i, (gauss, lap) in enumerate(zip(gaussian_pyramid, laplacian_pyramid)):
        plt.figure(figsize=(8, 8))
        plt.imshow(gauss, cmap='gray', vmin=0, vmax=1)
        plt.title(f'Gaussian Pyramid Level {i}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/gaussian_level_{i}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(8, 8))
        if lap.max() > lap.min():
            lap_normalized = (lap - lap.min()) / (lap.max() - lap.min())
        else:
            lap_normalized = np.zeros_like(lap)
        
        plt.imshow(lap_normalized, cmap='gray')
        plt.title(f'Laplacian Pyramid Level {i}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/laplacian_level_{i}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Pyramid analysis results saved to {output_dir}/ directory")
    return gaussian_pyramid, laplacian_pyramid

def show_fourier_analysis(im1, im2, hybrid, sigma1, sigma2):
    """Show frequency domain representations for hybrid analysis."""
    gray1 = rgb_to_gray(im1)
    gray2 = rgb_to_gray(im2)
    gray_hybrid = rgb_to_gray(hybrid)

    f1 = np.fft.fftshift(np.fft.fft2(gray1))
    f2 = np.fft.fftshift(np.fft.fft2(gray2))
    f_hybrid = np.fft.fftshift(np.fft.fft2(gray_hybrid))

    log_f1 = np.log(np.abs(f1) + 1)
    log_f2 = np.log(np.abs(f2) + 1)
    log_fh = np.log(np.abs(f_hybrid) + 1)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(log_f1, cmap='gray')
    axs[0].set_title(f'FFT of Derek (High-pass)\nσ={sigma1}')
    axs[0].axis('off')

    axs[1].imshow(log_f2, cmap='gray')
    axs[1].set_title(f'FFT of Nutmeg (Low-pass)\nσ={sigma2}')
    axs[1].axis('off')

    axs[2].imshow(log_fh, cmap='gray')
    axs[2].set_title('FFT of Hybrid Image')
    axs[2].axis('off')

    plt.tight_layout()
    plt.suptitle('Fourier Transform Analysis (Log Scale)', y=1.05)
    plt.show()

def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    print('Suggested points: both eyes, or eye centers and nose tip')

    plt.figure(figsize=(12, 8))
    plt.imshow(im1)
    plt.title("Image 1 (Derek): Click 2 alignment points\n(e.g., left eye center, right eye center)")
    pts1 = plt.ginput(2, timeout=30)
    plt.close()
    if len(pts1) < 2:
        raise ValueError("Not enough points selected on the first image")

    plt.figure(figsize=(12, 8))
    plt.imshow(im2)
    plt.title("Image 2 (Nutmeg): Click corresponding points\n(same features as Image 1)")
    pts2 = plt.ginput(2, timeout=30)
    plt.close()
    if len(pts2) < 2:
        raise ValueError("Not enough points selected on the second image")

    return (*pts1, *pts2)


def match_img_size(im1, im2):
    print(f"Input shapes: im1={im1.shape}, im2={im2.shape}")

    if len(im1.shape) == 2:
        im1 = np.stack([im1] * 3, axis=2)
    if len(im2.shape) == 2:
        im2 = np.stack([im2] * 3, axis=2)

    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape

    if c1 != c2:
        target_channels = max(c1, c2)
        if c1 < target_channels:
            im1_new = np.zeros((h1, w1, target_channels))
            for i in range(target_channels):
                im1_new[:, :, i] = im1[:, :, min(i, c1 - 1)]
            im1 = im1_new
        if c2 < target_channels:
            im2_new = np.zeros((h2, w2, target_channels))
            for i in range(target_channels):
                im2_new[:, :, i] = im2[:, :, min(i, c2 - 1)]
            im2 = im2_new

    min_h = min(h1, h2)
    min_w = min(w1, w2)
    h1_start = (h1 - min_h) // 2
    w1_start = (w1 - min_w) // 2
    h2_start = (h2 - min_h) // 2
    w2_start = (w2 - min_w) // 2

    im1 = im1[h1_start:h1_start + min_h, w1_start:w1_start + min_w]
    im2 = im2[h2_start:h2_start + min_h, w2_start:w2_start + min_w]

    if im1.max() > 1.0:
        im1 = im1.astype(np.float64) / 255.0
    if im2.max() > 1.0:
        im2 = im2.astype(np.float64) / 255.0

    im1 = np.clip(im1, 0, 1)
    im2 = np.clip(im2, 0, 1)

    print(f"Final aligned shapes: im1={im1.shape}, im2={im2.shape}")
    return im1, im2


def align_images(im1, im2):
    pts = get_points(im1, im2)
    print(f"Selected points: {pts}")

    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2


def improved_hybrid_processing():
    print("\n" + "=" * 60)
    print("IMPROVED PART 2.2: HYBRID IMAGES")
    print("=" * 60)

    os.makedirs('results', exist_ok=True)
    im1_path = 'DerekPicture.jpg'
    im2_path = 'nutmeg.jpg'
    if not (os.path.exists(im1_path) and os.path.exists(im2_path)):
        raise FileNotFoundError("Make sure both 'DerekPicture.jpg' and 'nutmeg.jpg' are in the current directory.")

    im1 = cv2.cvtColor(cv2.imread(im1_path), cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(cv2.imread(im2_path), cv2.COLOR_BGR2RGB)
    print(f"Original Derek shape: {im1.shape}")
    print(f"Original Nutmeg shape: {im2.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(im1)
    axes[0].set_title('Derek (Image 1) - High Freq Source')
    axes[0].axis('off')
    axes[1].imshow(im2)
    axes[1].set_title('Nutmeg (Image 2) - Low Freq Source')
    axes[1].axis('off')
    plt.tight_layout()
    plt.suptitle('Original Images - Study these before alignment', y=1.02, fontsize=14)
    plt.show()

    print("\nStep 1: Careful alignment (this is crucial for good results)")
    try:
        im1_aligned, im2_aligned = align_images(im1, im2)
        print("Alignment successful!")
    except Exception as e:
        print(f"Alignment failed: {e}")
        print("Using basic size matching...")
        im1_aligned, im2_aligned = match_img_size(im1, im2)

    print("\nStep 2: Create final hybrid with optimal parameters...")
    sigma1, sigma2 = 2, 5
    final_hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

    print("\nStep 3: Save and visualize results...")
    save_hybrid_results(im1, im2, im1_aligned, im2_aligned, final_hybrid, sigma1, sigma2)
    save_frequency_analysis(im1_aligned, im2_aligned, final_hybrid, sigma1, sigma2)
    save_pyramid_analysis(final_hybrid, N=5)

    pyramids(final_hybrid, N=5)
    show_fourier_analysis(im1_aligned, im2_aligned, final_hybrid, sigma1, sigma2)

    print("\n🎯 Final hybrid image created with σ1=2, σ2=5")
    return final_hybrid, im1_aligned, im2_aligned

def dog_monkey_hybrid_demo():
    print("\n" + "=" * 60)
    print("DOG + MONKEY HYBRID IMAGES")
    print("=" * 60)

    os.makedirs('results', exist_ok=True)
    im1_path = 'dog.jpg'
    im2_path = 'monkey.jpg'
    if not (os.path.exists(im1_path) and os.path.exists(im2_path)):
        raise FileNotFoundError("Make sure both 'dog.jpg' and 'monkey.jpg' are in the current directory.")

    im1 = cv2.cvtColor(cv2.imread(im1_path), cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(cv2.imread(im2_path), cv2.COLOR_BGR2RGB)
    print(f"Original Dog shape: {im1.shape}")
    print(f"Original Monkey shape: {im2.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(im1)
    axes[0].set_title('Dog (Image 1) - High Freq Source')
    axes[0].axis('off')
    axes[1].imshow(im2)
    axes[1].set_title('Monkey (Image 2) - Low Freq Source')
    axes[1].axis('off')
    plt.tight_layout()
    plt.suptitle('Original Images - Study these before alignment', y=1.02, fontsize=14)
    plt.show()

    print("\nStep 1: Careful alignment (this is crucial for good results)")
    try:
        im1_aligned, im2_aligned = align_images(im1, im2)
        print("Alignment successful!")
    except Exception as e:
        print(f"Alignment failed: {e}")
        print("Using basic size matching...")
        im1_aligned, im2_aligned = match_img_size(im1, im2)

    print("\nStep 2: Create final hybrid with optimal parameters...")
    sigma1, sigma2 = 3, 6
    final_hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

    print("\nStep 3: Save and visualize results...")
    imsave("results/dog_original.jpg", im1)
    imsave("results/monkey_original.jpg", im2)
    imsave("results/dog_aligned.jpg", im1_aligned)
    imsave("results/monkey_aligned.jpg", im2_aligned)
    imsave("results/dog_monkey_hybrid.jpg", final_hybrid)

    # Display final result
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(im1_aligned)
    axes[0].set_title('Dog (Aligned)')
    axes[0].axis('off')
    axes[1].imshow(im2_aligned)
    axes[1].set_title('Monkey (Aligned)')
    axes[1].axis('off')
    axes[2].imshow(final_hybrid)
    axes[2].set_title('Dog-Monkey Hybrid')
    axes[2].axis('off')
    plt.tight_layout()
    plt.suptitle('Dog + Monkey Hybrid Result', y=1.02, fontsize=16)
    plt.show()

    print(f"\nFinal hybrid image created with σ1={sigma1}, σ2={sigma2}")
    return final_hybrid, im1_aligned, im2_aligned

if __name__ == "__main__":
    part_2_1_image_sharpening()
    print("\n" + "=" * 60)
    print("GENERATING SHARPENING WEBPAGE IMAGES")
    print("=" * 60)
    save_sharpening_results_for_webpage()
    dog_monkey_hybrid_demo()

    final_hybrid, derek_aligned, nutmeg_aligned = improved_hybrid_processing()
    print("\n🚀 All processing complete! Check the 'results/' directory for output files.")
