# CS180 (CS280A): Project 1 - Glass Plate Image Alignment

import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.color
from skimage.transform import rescale
from skimage.filters import sobel
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import glob
import os
import json

def crop_borders(img, crop_percent=0.1):
    h, w = img.shape[:2]
    crop_h = int(h * crop_percent)
    crop_w = int(w * crop_percent)
    
    return img[crop_h:h-crop_h, crop_w:w-crop_w]

def create_image_pyramid(img, max_levels):
    pyramid = [img]
    current = img
    for _ in range(max_levels - 1):
        current = rescale(current, 0.5, anti_aliasing=True, channel_axis=None)
        pyramid.append(current)
    return pyramid

def robust_alignment_score(img1, img2):
    edge1 = sobel(img1)
    edge2 = sobel(img2)
    
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    ncc_orig = np.corrcoef(img1_flat, img2_flat)[0, 1]
    
    edge1_flat = edge1.flatten()
    edge2_flat = edge2.flatten()
    ncc_edge = np.corrcoef(edge1_flat, edge2_flat)[0, 1]
    
    return 0.3 * ncc_orig + 0.7 * ncc_edge

def enhanced_align_ncc(img1, img2, search_window=15):
    best_score = -np.inf
    best_displacement = (0, 0)

    for dy in range(-search_window, search_window + 1):
        for dx in range(-search_window, search_window + 1):
            shifted_img = np.roll(img2, dy, axis=0)
            shifted_img = np.roll(shifted_img, dx, axis=1)

            score = robust_alignment_score(img1, shifted_img)

            if not np.isnan(score) and score > best_score:
                best_score = score
                best_displacement = (dx, dy)

    aligned_img = np.roll(img2, best_displacement[1], axis=0)
    aligned_img = np.roll(aligned_img, best_displacement[0], axis=1)

    return best_displacement, aligned_img

def enhanced_pyramid_alignment(img1, img2, max_levels=4, search_window=15):
    pyramid1 = create_image_pyramid(img1, max_levels)
    pyramid2 = create_image_pyramid(img2, max_levels)
    
    total_displacement = np.array([0, 0])
    
    for level in range(len(pyramid1) - 1, -1, -1):
        if level < len(pyramid1) - 1:
            total_displacement *= 2

        test_img = np.roll(pyramid2[level], int(total_displacement[1]), axis=0)
        test_img = np.roll(test_img, int(total_displacement[0]), axis=1)

        displacement, _ = enhanced_align_ncc(pyramid1[level],test_img,search_window if level == len(pyramid1) - 1 else 3)
        total_displacement += np.array(displacement)
    
    aligned_img = np.roll(img2, int(total_displacement[1]), axis=0)
    aligned_img = np.roll(aligned_img, int(total_displacement[0]), axis=1)
    
    return tuple(total_displacement.astype(int)), aligned_img

def process_glass_plate_image(imname, output_path=None, use_pyramid=None, search_window=15):
    print(f"Processing image: {imname}")
    
    im = skio.imread(imname)
    if len(im.shape) == 3 and im.shape[2] > 1:
        im = sk.color.rgb2gray(im)
    
    im = sk.img_as_float(im)
    height = int(np.floor(im.shape[0] / 3.0))

    b = im[:height]
    g = im[height:2*height]
    r = im[2*height:3*height]
    
    print(f"Image dimensions: {im.shape}")
    print(f"Each channel: {b.shape}")
    
    is_high_res = height > 2000 or imname.lower().endswith('.tif')
    use_pyramid_final = use_pyramid if use_pyramid is not None else is_high_res
    b_crop = crop_borders(b, crop_percent=0.15)  
    g_crop = crop_borders(g, crop_percent=0.15)
    r_crop = crop_borders(r, crop_percent=0.15)

    if use_pyramid_final:
        print("Using enhanced pyramid alignment...")
        disp_g, _ = enhanced_pyramid_alignment(b_crop, g_crop, max_levels=5)  
        disp_r, _ = enhanced_pyramid_alignment(b_crop, r_crop, max_levels=5)
    else:
        print("Using enhanced single-scale alignment...")
        disp_g, _ = enhanced_align_ncc(b_crop, g_crop, search_window)
        disp_r, _ = enhanced_align_ncc(b_crop, r_crop, search_window)

    ag = np.roll(g, disp_g[1], axis=0)
    ag = np.roll(ag, disp_g[0], axis=1)
    
    ar = np.roll(r, disp_r[1], axis=0)  
    ar = np.roll(ar, disp_r[0], axis=1)
    
    print(f"Green channel displacement (x,y): {disp_g}")
    print(f"Red channel displacement (x,y): {disp_r}")
    
    b_norm = sk.exposure.rescale_intensity(b)
    ag_norm = sk.exposure.rescale_intensity(ag) 
    ar_norm = sk.exposure.rescale_intensity(ar)
    
    im_out = np.dstack([ar_norm, ag_norm, b_norm])
    im_out = np.clip(im_out, 0, 1)

    if output_path is None:
        base_name = os.path.splitext(imname)[0]
        output_path = f'{base_name}_aligned.jpg'
    
    skio.imsave(output_path, img_as_ubyte(im_out))
    print(f"Saved aligned image to: {output_path}")
    
    plt.figure(figsize=(6, 6))
    plt.imshow(im_out)
    plt.title('Final Aligned Color Image')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return im_out, disp_g, disp_r

def save_displacement_data(displacement_data, filename='displacements.json'):
    json_data = {}
    for image_name, displacements in displacement_data.items():
        json_data[image_name] = {
            'green': [int(displacements['green'][0]), int(displacements['green'][1])],
            'red': [int(displacements['red'][0]), int(displacements['red'][1])]
        }
    
    with open(filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Displacement data saved to {filename}")


if __name__ == "__main__":
    test_images = sorted([
        f for f in glob.glob("*.jpg") + glob.glob("*.tif")
        if "_aligned" not in f and not f.startswith("._") 
    ])

    displacement_data = {}
    
    for imname in test_images:
        try:
            print(f"\n{'='*50}")
            print(f"Processing {imname}")
            print('='*50)
            
            im_out, disp_g, disp_r = process_glass_plate_image(imname)
            base_name = os.path.splitext(imname)[0]
            displacement_data[base_name] = {
                'green': disp_g,
                'red': disp_r
            }
            
        except FileNotFoundError:
            print(f"File {imname} not found.")
        except Exception as e:
            print(f"Error processing {imname}: {e}")
    
    save_displacement_data(displacement_data)

