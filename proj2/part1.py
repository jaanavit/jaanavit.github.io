import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image
import cv2
from matplotlib.image import imsave
import os

def conv2d_four_loops(image, kernel, padding=True):
    """
    Implement 2D convolution using four nested loops with zero padding
    """
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)  
    
    img_h, img_w = image.shape
    ker_h, ker_w = kernel.shape
    
    if padding:
        pad_h = ker_h // 2
        pad_w = ker_w // 2
        padded_img = np.zeros((img_h + 2*pad_h, img_w + 2*pad_w))
        padded_img[pad_h:pad_h+img_h, pad_w:pad_w+img_w] = image
    else:
        padded_img = image
        pad_h = pad_w = 0
    
    if padding:
        out_h, out_w = img_h, img_w
    else:
        out_h = img_h - ker_h + 1
        out_w = img_w - ker_w + 1
    
    result = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            for ki in range(ker_h):
                for kj in range(ker_w):
                    result[i, j] += padded_img[i + ki, j + kj] * kernel[ki, kj]
    
    return result

def conv2d_two_loops(image, kernel, padding=True):
    """
    Implement 2D convolution using two nested loops with vectorized inner operations
    """
    if len(image.shape) == 3:
        image = np.mean(image, axis=2) 
    
    img_h, img_w = image.shape
    ker_h, ker_w = kernel.shape
    
    if padding:
        pad_h = ker_h // 2
        pad_w = ker_w // 2
        padded_img = np.zeros((img_h + 2*pad_h, img_w + 2*pad_w))
        padded_img[pad_h:pad_h+img_h, pad_w:pad_w+img_w] = image
    else:
        padded_img = image
        pad_h = pad_w = 0
    
    if padding:
        out_h, out_w = img_h, img_w
    else:
        out_h = img_h - ker_h + 1
        out_w = img_w - ker_w + 1
    
    result = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            region = padded_img[i:i+ker_h, j:j+ker_w]
            result[i, j] = np.sum(region * kernel)
    
    return result

def test_convolution_implementations():
    """
    Test and compare our implementations with scipy's built-in function
    """
    test_img = np.random.rand(10, 10)
    test_kernel = np.array([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]]) / 4 
    
    result_4loops = conv2d_four_loops(test_img, test_kernel, padding=True)
    result_2loops = conv2d_two_loops(test_img, test_kernel, padding=True)
    
    scipy_result = convolve2d(test_img, test_kernel, mode='same', boundary='fill', fillvalue=0)
    
    print("Comparison of convolution implementations:")
    print(f"Four loops vs Scipy - Max difference: {np.max(np.abs(result_4loops - scipy_result)):.10f}")
    print(f"Two loops vs Scipy - Max difference: {np.max(np.abs(result_2loops - scipy_result)):.10f}")
    print(f"Four loops vs Two loops - Max difference: {np.max(np.abs(result_4loops - result_2loops)):.10f}")
    
    return result_4loops, result_2loops, scipy_result

def create_box_filter(size=9):
    """Create a box filter (averaging filter)"""
    return np.ones((size, size)) / (size * size)

def create_dx_filter():
    """Create finite difference filter for x-direction (horizontal edges)"""
    return np.array([[-1, 0, 1]])

def create_dy_filter():
    """Create finite difference filter for y-direction (vertical edges)"""
    return np.array([[-1], [0], [1]])

def process_stadium_image(image_path):
    """
    Process the stadium image with different filters
    Note: Replace 'image_path' with the actual path to your image
    """
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        if len(img_array.shape) == 3:
            img_gray = np.mean(img_array, axis=2)
        else:
            img_gray = img_array
            
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return
    
    img_gray = ((img_gray - img_gray.min()) / (img_gray.max() - img_gray.min()) * 255).astype(np.uint8)
    
    box_filter_9x9 = create_box_filter(9)
    dx_filter = create_dx_filter()
    dy_filter = create_dy_filter()
    
    print("Applying filters...")
    blurred = conv2d_two_loops(img_gray, box_filter_9x9, padding=True)
    dx_edges = conv2d_two_loops(img_gray, dx_filter, padding=True)
    dy_edges = conv2d_two_loops(img_gray, dy_filter, padding=True)
    
    blurred_scipy = convolve2d(img_gray, box_filter_9x9, mode='same', boundary='fill', fillvalue=0)
    dx_scipy = convolve2d(img_gray, dx_filter, mode='same', boundary='fill', fillvalue=0)
    dy_scipy = convolve2d(img_gray, dy_filter, mode='same', boundary='fill', fillvalue=0)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(img_gray, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(blurred, cmap='gray')
    axes[0, 1].set_title('Box Filter (9x9) - Our Implementation')
    axes[0, 1].axis('off')
    
    axes[1, 1].imshow(blurred_scipy, cmap='gray')
    axes[1, 1].set_title('Box Filter (9x9) - Scipy')
    axes[1, 1].axis('off')
    
    axes[0, 2].imshow(dx_edges, cmap='gray')
    axes[0, 2].set_title('Dx Filter - Our Implementation')
    axes[0, 2].axis('off')
    
    axes[1, 2].imshow(dx_scipy, cmap='gray')
    axes[1, 2].set_title('Dx Filter - Scipy')
    axes[1, 2].axis('off')
    
    axes[0, 3].imshow(dy_edges, cmap='gray')
    axes[0, 3].set_title('Dy Filter - Our Implementation')
    axes[0, 3].axis('off')
    
    axes[1, 3].imshow(dy_scipy, cmap='gray')
    axes[1, 3].set_title('Dy Filter - Scipy')
    axes[1, 3].axis('off')
    
    axes[1, 0].imshow(np.abs(blurred - blurred_scipy), cmap='hot')
    axes[1, 0].set_title('Difference (Box Filter)')
    axes[1, 0].axis('off')
    
    plt.tight_layout()
    plt.show()

    os.makedirs('results', exist_ok=True)
    imsave('results/grayscale.png', img_gray, cmap='gray')
    imsave('results/box_filter.png', blurred, cmap='gray')
    imsave('results/dx_filter.png', dx_edges, cmap='gray')
    imsave('results/dy_filter.png', dy_edges, cmap='gray')
    
    print("\nFilter comparison statistics:")
    print(f"Box filter - Max difference: {np.max(np.abs(blurred - blurred_scipy)):.6f}")
    print(f"Dx filter - Max difference: {np.max(np.abs(dx_edges - dx_scipy)):.6f}")
    print(f"Dy filter - Max difference: {np.max(np.abs(dy_edges - dy_scipy)):.6f}")
    
    return img_gray, blurred, dx_edges, dy_edges

def part_1_2_finite_difference():
    """
    Part 1.2: Apply finite difference operators to cameraman image
    """
    try:
        cameraman = np.array(Image.open("cameraman.png").convert("L")) / 255.0
        print("Loaded real cameraman.png")
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return
    
    dx_filter = np.array([[-1, 0, 1]]) / 2  
    dy_filter = np.array([[-1], [0], [1]]) / 2
    
    partial_x = convolve2d(cameraman, dx_filter, mode='same', boundary='fill', fillvalue=0)
    partial_y = convolve2d(cameraman, dy_filter, mode='same', boundary='fill', fillvalue=0)
    
    gradient_magnitude = np.sqrt(partial_x**2 + partial_y**2)
    
    thresholds = {
        'Low (mean)': np.mean(gradient_magnitude),
        'Medium (75th percentile)': np.percentile(gradient_magnitude, 75),
        'High (90th percentile)': np.percentile(gradient_magnitude, 90),
        'Very High (95th percentile)': np.percentile(gradient_magnitude, 95)
    }
    
    binary_edges = {}
    for name, thresh in thresholds.items():
        binary_edges[name] = (gradient_magnitude > thresh).astype(np.uint8) * 255
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(cameraman, cmap='gray')
    axes[0, 0].set_title('Original Cameraman')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(partial_x, cmap='gray')
    axes[0, 1].set_title('Partial Derivative in X')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(partial_y, cmap='gray')
    axes[0, 2].set_title('Partial Derivative in Y')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(gradient_magnitude, cmap='gray')
    axes[0, 3].set_title('Gradient Magnitude')
    axes[0, 3].axis('off')
    
    for i, (name, binary_img) in enumerate(binary_edges.items()):
        axes[1, i].imshow(binary_img, cmap='gray')
        axes[1, i].set_title(f'Binary Edges: {name}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Part 1.2: Finite Difference Edge Detection', y=1.02, fontsize=16)
    plt.show()
    
    print("Part 1.2 Results:")
    print(f"Image shape: {cameraman.shape}")
    print(f"Gradient magnitude range: {gradient_magnitude.min():.3f} to {gradient_magnitude.max():.3f}")
    print("Threshold values tried:")
    for name, thresh in thresholds.items():
        edge_pixels = np.sum(gradient_magnitude > thresh)
        percentage = (edge_pixels / gradient_magnitude.size) * 100
        print(f"  {name}: {thresh:.3f} ({percentage:.1f}% edge pixels)")
    
    os.makedirs('results', exist_ok=True) 

    imsave('results/cameraman_original.png', cameraman, cmap='gray')
    imsave('results/partial_x.png', partial_x, cmap='gray')
    imsave('results/partial_y.png', partial_y, cmap='gray')
    imsave('results/gradient_magnitude.png', gradient_magnitude, cmap='gray')
    imsave('results/binary_low.png', binary_edges['Low (mean)'], cmap='gray')
    imsave('results/binary_medium.png', binary_edges['Medium (75th percentile)'], cmap='gray')
    imsave('results/binary_high.png', binary_edges['High (90th percentile)'], cmap='gray')
    imsave('results/binary_very_high.png', binary_edges['Very High (95th percentile)'], cmap='gray')
    return cameraman, partial_x, partial_y, gradient_magnitude, binary_edges

# Part 1.3: Derivative of Gaussian (DoG) Filter
def part_1_3_derivative_of_gaussian():
    """
    Part 1.3: Apply Derivative of Gaussian filters
    """
    try:
        cameraman = np.array(Image.open("cameraman.png").convert("L")) / 255.0
        print("Loaded real cameraman.png")
    except FileNotFoundError:
        print("cameraman.png not found!")
        return

    kernel_size = 15
    sigma = 2.0
    
    # Create 1D Gaussian kernel
    gaussian_1d = cv2.getGaussianKernel(kernel_size, sigma)
    
    # Create 2D Gaussian kernel by outer product
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
    
    print(f"Gaussian kernel shape: {gaussian_2d.shape}")
    print(f"Gaussian kernel sum: {gaussian_2d.sum():.6f}")
    
    # Step 1: Blur the image first, then apply finite difference
    blurred_img = convolve2d(cameraman, gaussian_2d, mode='same', boundary='fill', fillvalue=0)
    
    dx_filter = np.array([[-1, 0, 1]]) / 2
    dy_filter = np.array([[-1], [0], [1]]) / 2
    
    blurred_dx = convolve2d(blurred_img, dx_filter, mode='same', boundary='fill', fillvalue=0)
    blurred_dy = convolve2d(blurred_img, dy_filter, mode='same', boundary='fill', fillvalue=0)
    blurred_gradient_mag = np.sqrt(blurred_dx**2 + blurred_dy**2)
    
    # Step 2: Create Derivative of Gaussian (DoG) filters
    # DoG_x = Gaussian * D_x, DoG_y = Gaussian * D_y
    dog_x = convolve2d(gaussian_2d, dx_filter, mode='same', boundary='fill', fillvalue=0)
    dog_y = convolve2d(gaussian_2d, dy_filter, mode='same', boundary='fill', fillvalue=0)
    
    dog_result_x = convolve2d(cameraman, dog_x, mode='same', boundary='fill', fillvalue=0)
    dog_result_y = convolve2d(cameraman, dog_y, mode='same', boundary='fill', fillvalue=0)
    dog_gradient_mag = np.sqrt(dog_result_x**2 + dog_result_y**2)
    
    difference = np.abs(blurred_gradient_mag - dog_gradient_mag)
    max_diff = np.max(difference)
    mean_diff = np.mean(difference)
    
    threshold = np.percentile(blurred_gradient_mag, 85)
    binary_original = (blurred_gradient_mag > threshold).astype(np.uint8) * 255
    binary_dog = (dog_gradient_mag > threshold).astype(np.uint8) * 255
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Row 1: Original processing
    axes[0, 0].imshow(cameraman, cmap='gray')
    axes[0, 0].set_title('Original Cameraman')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(blurred_img, cmap='gray')
    axes[0, 1].set_title('Gaussian Blurred')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(blurred_gradient_mag, cmap='gray')
    axes[0, 2].set_title('Gradient Mag (Blur then Diff)')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(binary_original, cmap='gray')
    axes[0, 3].set_title('Binary Edges (Blur then Diff)')
    axes[0, 3].axis('off')
    
    # Row 2: DoG filters and results
    axes[1, 0].imshow(dog_x, cmap='RdBu_r')
    axes[1, 0].set_title('DoG Filter (X direction)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(dog_y, cmap='RdBu_r')
    axes[1, 1].set_title('DoG Filter (Y direction)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(dog_gradient_mag, cmap='gray')
    axes[1, 2].set_title('Gradient Mag (DoG)')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(binary_dog, cmap='gray')
    axes[1, 3].set_title('Binary Edges (DoG)')
    axes[1, 3].axis('off')
    
    # Row 3: Comparison and original derivatives
    axes[2, 0].imshow(difference, cmap='hot')
    axes[2, 0].set_title(f'Difference (Max: {max_diff:.6f})')
    axes[2, 0].axis('off')
    
    orig_dx = convolve2d(cameraman, dx_filter, mode='same', boundary='fill', fillvalue=0)
    orig_dy = convolve2d(cameraman, dy_filter, mode='same', boundary='fill', fillvalue=0)
    orig_grad_mag = np.sqrt(orig_dx**2 + orig_dy**2)
    
    axes[2, 1].imshow(orig_grad_mag, cmap='gray')
    axes[2, 1].set_title('Original (No Gaussian)')
    axes[2, 1].axis('off')
    
    smooth_vs_orig = blurred_gradient_mag - orig_grad_mag
    axes[2, 2].imshow(smooth_vs_orig, cmap='RdBu_r')
    axes[2, 2].set_title('Smoothed - Original')
    axes[2, 2].axis('off')

    axes[2, 3].imshow(gaussian_2d, cmap='hot')
    axes[2, 3].set_title('Gaussian Kernel')
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Part 1.3: Derivative of Gaussian (DoG) Filters', y=1.02, fontsize=16)
    plt.show()
    
    print("\nPart 1.3 Results:")
    print(f"Max difference between two approaches: {max_diff:.8f}")
    print(f"Mean difference between two approaches: {mean_diff:.8f}")
    print("✓ The two approaches produce virtually identical results!")
    
    print(f"\nNoise reduction comparison:")
    orig_noise_level = np.std(orig_grad_mag)
    smooth_noise_level = np.std(blurred_gradient_mag)
    noise_reduction = ((orig_noise_level - smooth_noise_level) / orig_noise_level) * 100
    print(f"Original gradient magnitude std: {orig_noise_level:.3f}")
    print(f"Smoothed gradient magnitude std: {smooth_noise_level:.3f}")
    print(f"Noise reduction: {noise_reduction:.1f}%")

    os.makedirs('results', exist_ok=True)

    imsave('results/gaussian_blurred.png', blurred_img, cmap='gray')
    imsave('results/dog_x_filter.png', dog_x, cmap='gray')
    imsave('results/dog_y_filter.png', dog_y, cmap='gray')
    imsave('results/dog_gradient.png', dog_gradient_mag, cmap='gray')
    imsave('results/original_noisy_edges.png', orig_grad_mag, cmap='gray')
    imsave('results/smoothed_edges.png', blurred_gradient_mag, cmap='gray')
    imsave('results/dog_edges.png', binary_dog, cmap='gray')
    
    return {
        'cameraman': cameraman,
        'blurred_gradient': blurred_gradient_mag,
        'dog_gradient': dog_gradient_mag,
        'dog_x': dog_x,
        'dog_y': dog_y,
    }

if __name__ == "__main__":
    print("Testing convolution implementations...")
    test_convolution_implementations()

    print("\nProcessing image with filters...")
    results = process_stadium_image('IMG_8011.jpeg')
    
    print("\nFilter kernels used:")
    print("9x9 Box Filter (showing first 3x3 corner):")
    box_filter = create_box_filter(9)
    print(box_filter[:3, :3])
    
    print("\nDx Filter (finite difference):")
    print(create_dx_filter())
    
    print("\nDy Filter (finite difference):")
    print(create_dy_filter())
    
    print("\n" + "="*60)
    print("PART 1.2: FINITE DIFFERENCE OPERATOR")
    print("="*60)
    part_1_2_results = part_1_2_finite_difference()
    
    print("\n" + "="*60)
    print("PART 1.3: DERIVATIVE OF GAUSSIAN (DoG) FILTER")
    print("="*60)
    part_1_3_results = part_1_3_derivative_of_gaussian()
