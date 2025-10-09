import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

def computeH(im1_pts, im2_pts):
    n = im1_pts.shape[0]
    
    if n < 4:
        raise ValueError("Need at least 4 point correspondences")
    
    display_system_equations(im1_pts, im2_pts)
    A = []
    
    for i in range(n):
        x, y = im1_pts[i]
        xp, yp = im2_pts[i]
        A.append([x, y, 1, 0, 0, 0, -x*xp, -y*xp, -xp])
        A.append([0, 0, 0, x, y, 1, -x*yp, -y*yp, -yp])
    
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = h.reshape(3, 3)
    H = H / H[2, 2]
    
    return H

class PointSelector:
    """Interactive point selection tool for selecting correspondences."""
    
    def __init__(self, im1, im2):
        self.im1 = im1
        self.im2 = im2
        self.im1_pts = []
        self.im2_pts = []
        self.current_image = 1
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        
    def onclick(self, event):
        if event.inaxes is None:
            return
            
        x, y = event.xdata, event.ydata
        
        if event.inaxes == self.ax1 and self.current_image == 1:
            self.im1_pts.append([x, y])
            self.ax1.plot(x, y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
            self.ax1.text(x, y-20, str(len(self.im1_pts)), color='white', 
                         fontsize=12, fontweight='bold', ha='center',
                         bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
            print(f"Image 1 - Point {len(self.im1_pts)}: ({x:.1f}, {y:.1f})")
            self.current_image = 2
            
        elif event.inaxes == self.ax2 and self.current_image == 2:
            self.im2_pts.append([x, y])
            self.ax2.plot(x, y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
            self.ax2.text(x, y-20, str(len(self.im2_pts)), color='white',
                         fontsize=12, fontweight='bold', ha='center',
                         bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
            print(f"Image 2 - Point {len(self.im2_pts)}: ({x:.1f}, {y:.1f})")
            self.current_image = 1
            
        self.fig.canvas.draw()
    
    def select_points(self):
        """Launch interactive point selection."""
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        self.ax1.imshow(cv2.cvtColor(self.im1, cv2.COLOR_BGR2RGB))
        self.ax1.set_title('Image 1 - Click to select points', fontsize=14, fontweight='bold')
        self.ax1.axis('off')
        
        self.ax2.imshow(cv2.cvtColor(self.im2, cv2.COLOR_BGR2RGB))
        self.ax2.set_title('Image 2 - Click corresponding points', fontsize=14, fontweight='bold')
        self.ax2.axis('off')
        
        self.fig.suptitle('Select at least 4 corresponding points\nAlternate between images', 
                         fontsize=14, fontweight='bold')
        
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        
        print("\n" + "="*60)
        print("POINT SELECTION MODE")
        print("="*60)
        print("Instructions:")
        print("1. Click a point in Image 1")
        print("2. Click the CORRESPONDING point in Image 2")
        print("3. Repeat for at least 4 point pairs")
        print("4. Close the window when done")
        print("="*60 + "\n")
        
        plt.tight_layout()
        plt.show()
        
        return np.array(self.im1_pts), np.array(self.im2_pts)


def visualize_correspondences(im1, im2, im1_pts, im2_pts, H=None):
    """
    Visualize point correspondences between two images.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
    ax1.set_title('Image 1', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
    ax2.set_title('Image 2', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(im1_pts)))
    
    for i, (pt1, pt2, color) in enumerate(zip(im1_pts, im2_pts, colors)):
        ax1.plot(pt1[0], pt1[1], 'o', color=color, markersize=10, 
                markeredgecolor='white', markeredgewidth=2)
        ax1.text(pt1[0], pt1[1]-20, str(i+1), color='white', 
                fontsize=12, fontweight='bold',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        ax2.plot(pt2[0], pt2[1], 'o', color=color, markersize=10,
                markeredgecolor='white', markeredgewidth=2)
        ax2.text(pt2[0], pt2[1]-20, str(i+1), color='white',
                fontsize=12, fontweight='bold',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('correspondences_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to 'correspondences_visualization.png'")
    plt.show()

def display_system_equations(im1_pts, im2_pts, num_points_to_show=3):
    print("\nSystem of Equations (Ah = b form):")
    print("="*80)
    
    n_show = min(num_points_to_show, len(im1_pts))
    
    for i in range(n_show):
        x, y = im1_pts[i]
        xp, yp = im2_pts[i]
        
        print(f"Point {i+1}: ({x:.1f}, {y:.1f}) -> ({xp:.1f}, {yp:.1f})")
        print(f"  Equation 1: {x:.1f}*h11 + {y:.1f}*h12 + h13 - {x*xp:.1f}*h31 - {y*xp:.1f}*h32 = {xp:.1f}")
        print(f"  Equation 2: {x:.1f}*h21 + {y:.1f}*h22 + h23 - {x*yp:.1f}*h31 - {y*yp:.1f}*h32 = {yp:.1f}")
        print()
    
    if len(im1_pts) > num_points_to_show:
        print(f"... and {len(im1_pts) - num_points_to_show} more point(s)")
    
    print(f"Total: {len(im1_pts)*2} equations")
    print("="*80)

def warpImageNearestNeighbor(im, H, output_shape):
    H_inv = np.linalg.inv(H)
    h_out, w_out = output_shape
    h_in, w_in = im.shape[:2]
    
    warped = np.zeros((h_out, w_out, 3), dtype=np.uint8)
    
    y_coords, x_coords = np.mgrid[0:h_out, 0:w_out]
    coords = np.stack([x_coords.ravel(), y_coords.ravel(), np.ones(h_out * w_out)])
    
    source_coords = H_inv @ coords
    source_coords = source_coords[:2] / source_coords[2]
    source_x = source_coords[0].reshape(h_out, w_out)
    source_y = source_coords[1].reshape(h_out, w_out)
    
    source_x_int = np.round(source_x).astype(int)
    source_y_int = np.round(source_y).astype(int)
    
    valid_mask = (source_x_int >= 0) & (source_x_int < w_in) & \
                 (source_y_int >= 0) & (source_y_int < h_in)
    
    valid_y_out, valid_x_out = np.where(valid_mask)
    valid_y_in = source_y_int[valid_mask]
    valid_x_in = source_x_int[valid_mask]
    
    warped[valid_y_out, valid_x_out] = im[valid_y_in, valid_x_in]
    
    return warped

def warpImageBilinear(im, H, output_shape):
    H_inv = np.linalg.inv(H)
    h_out, w_out = output_shape
    h_in, w_in = im.shape[:2]
    
    warped = np.zeros((h_out, w_out, 3), dtype=np.float32)
    
    y_coords, x_coords = np.mgrid[0:h_out, 0:w_out]
    coords = np.stack([x_coords.ravel(), y_coords.ravel(), np.ones(h_out * w_out)])
    
    source_coords = H_inv @ coords
    source_coords = source_coords[:2] / source_coords[2]
    source_x = source_coords[0].reshape(h_out, w_out)
    source_y = source_coords[1].reshape(h_out, w_out)
    
    x0 = np.floor(source_x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(source_y).astype(int)
    y1 = y0 + 1
    
    wx = source_x - x0
    wy = source_y - y0
    
    valid_mask = (x0 >= 0) & (x1 < w_in) & (y0 >= 0) & (y1 < h_in)
    
    if np.any(valid_mask):
        valid_indices = np.where(valid_mask)
        y_out_valid = valid_indices[0]
        x_out_valid = valid_indices[1]
        
        x0_valid = x0[valid_mask]
        x1_valid = x1[valid_mask]
        y0_valid = y0[valid_mask]
        y1_valid = y1[valid_mask]
        wx_valid = wx[valid_mask]
        wy_valid = wy[valid_mask]
        
        im_float = im.astype(np.float32)
        
        for c in range(3):
            I00 = im_float[y0_valid, x0_valid, c]
            I10 = im_float[y0_valid, x1_valid, c]
            I01 = im_float[y1_valid, x0_valid, c]
            I11 = im_float[y1_valid, x1_valid, c]
            
            interpolated = (I00 * (1 - wx_valid) * (1 - wy_valid) +
                           I10 * wx_valid * (1 - wy_valid) +
                           I01 * (1 - wx_valid) * wy_valid +
                           I11 * wx_valid * wy_valid)
            
            warped[y_out_valid, x_out_valid, c] = interpolated
    
    warped = np.clip(warped, 0, 255).astype(np.uint8)
    return warped


if __name__ == "__main__":
    print("imagewarp.py loaded successfully")
    im1 = cv2.imread("doe1.png")
    im2 = cv2.imread("doe2.png")

    selector = PointSelector(im1, im2)
    im1_pts, im2_pts = selector.select_points()
    
    H = computeH(im1_pts, im2_pts)
    verify_homography(H, im1_pts, im2_pts)
    visualize_correspondences(im1, im2, im1_pts, im2_pts, H)