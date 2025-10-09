import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from imagewarp import computeH, warpImageNearestNeighbor, warpImageBilinear


def sharpen_image(img, amount=1.0):
    gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
    sharpened = cv2.addWeighted(img, 1.0 + amount, gaussian, -amount, 0)
    return sharpened


class RectificationSelector:
    """Interactive tool for selecting quadrilateral to rectify."""
    
    def __init__(self, im):
        self.im = im
        self.points = []
        self.fig = None
        self.ax = None
        
    def onclick(self, event):
        if event.inaxes is None or len(self.points) >= 4:
            return
            
        x, y = event.xdata, event.ydata
        self.points.append([x, y])
        
        self.ax.plot(x, y, 'ro', markersize=12, markeredgecolor='yellow', markeredgewidth=3)
        self.ax.text(x, y-30, str(len(self.points)), color='yellow', 
                    fontsize=14, fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
        
        if len(self.points) > 1:
            p1 = self.points[-2]
            p2 = self.points[-1]
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'y-', linewidth=2)
        
        if len(self.points) == 4:
            p1 = self.points[-1]
            p2 = self.points[0]
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'y-', linewidth=2)
            self.ax.set_title('4 points selected! Close window to continue.', 
                            fontsize=14, fontweight='bold', color='green')
        
        print(f"Point {len(self.points)}: ({x:.1f}, {y:.1f})")
        self.fig.canvas.draw()
    
    def select_quadrilateral(self):
        """Launch interactive quadrilateral selection."""
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
        
        if len(self.im.shape) == 3:
            self.ax.imshow(cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB))
        else:
            self.ax.imshow(self.im, cmap='gray')
        
        self.ax.set_title('Click 4 corners of the rectangle (in order: TL, TR, BL, BR)', 
                         fontsize=14, fontweight='bold')
        self.ax.axis('off')
        
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        
        print("\n" + "="*60)
        print("RECTIFICATION MODE")
        print("="*60)
        print("Instructions:")
        print("1. Click the 4 corners of a rectangular object")
        print("2. Order: Top-Left → Top-Right → Bottom-Left → Bottom-Right")
        print("3. Close window when done (after 4 points)")
        print("="*60 + "\n")
        
        plt.tight_layout()
        plt.show()
        
        if len(self.points) != 4:
            raise ValueError("Must select exactly 4 points")
        
        return np.array(self.points, dtype=np.float32)


def rectify_with_custom_size(im, src_pts, width, height):
    dst_pts = np.array([
        [0, 0],
        [width-1, 0],
        [0, height-1],
        [width-1, height-1]
    ], dtype=np.float32)
    
    H = computeH(src_pts, dst_pts)
    output_shape = (height, width)

    warped_nn = warpImageNearestNeighbor(im, H, output_shape)
    warped_bil = warpImageBilinear(im, H, output_shape)
    
    return H, warped_nn, warped_bil


def visualize_rectification(im, src_pts, warped_nn, warped_bil, title="Rectification"):
    """
    Visualize the rectification process with side-by-side comparison.
    """
    fig = plt.figure(figsize=(20, 6))
    
    ax1 = plt.subplot(131)
    if len(im.shape) == 3:
        ax1.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    else:
        ax1.imshow(im, cmap='gray')
    
    pts = np.array(src_pts, dtype=np.int32)
    for i in range(4):
        j = (i + 1) % 4
        ax1.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], 
                'y-', linewidth=3)
        ax1.plot(pts[i, 0], pts[i, 1], 'ro', markersize=12, 
                markeredgecolor='yellow', markeredgewidth=3)
        ax1.text(pts[i, 0], pts[i, 1]-20, str(i+1), color='yellow',
                fontsize=12, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
    
    ax1.set_title('Original with Selection', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Rectified - Nearest Neighbor
    ax2 = plt.subplot(132)
    if len(warped_nn.shape) == 3:
        ax2.imshow(cv2.cvtColor(warped_nn, cv2.COLOR_BGR2RGB))
    else:
        ax2.imshow(warped_nn, cmap='gray')
    ax2.set_title('Rectified (Nearest Neighbor)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Rectified - Bilinear
    ax3 = plt.subplot(133)
    if len(warped_bil.shape) == 3:
        ax3.imshow(cv2.cvtColor(warped_bil, cv2.COLOR_BGR2RGB))
    else:
        ax3.imshow(warped_bil, cmap='gray')
    ax3.set_title('Rectified (Bilinear)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    plt.tight_layout()
    filename = f'{title.lower().replace(" ", "_")}_result.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved rectification result to '{filename}'")
    plt.show()


def rectify_door():
    
    im = cv2.imread('door.png')
    if im is None:
        print("Error: Could not load door.png")
        return
    
    print(f"\nLoaded image: {im.shape}")
    
    print("\nSelect the 4 corners of the door:")
    print("  1. Top-left corner")
    print("  2. Top-right corner")
    print("  3. Bottom-left corner")
    print("  4. Bottom-right corner")
    
    selector = RectificationSelector(im)
    src_pts = selector.select_quadrilateral()
    
    print("\n✓ Selected 4 corners:")
    labels = ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']
    for i, (pt, label) in enumerate(zip(src_pts, labels)):
        print(f"  {label}: ({pt[0]:.1f}, {pt[1]:.1f})")
    
    width1 = np.linalg.norm(src_pts[1] - src_pts[0])
    width2 = np.linalg.norm(src_pts[3] - src_pts[2])
    height1 = np.linalg.norm(src_pts[2] - src_pts[0])
    height2 = np.linalg.norm(src_pts[3] - src_pts[1])
    
    avg_width = (width1 + width2) / 2
    avg_height = (height1 + height2) / 2
    
    print(f"\nEstimated door dimensions: {avg_width:.1f} x {avg_height:.1f} pixels")
    output_height = 700
    output_width = 300
    
    print(f"Output size: {output_width} x {output_height} pixels (standard door ratio)")
    
    H, warped_nn, warped_bil = rectify_with_custom_size(
        im, src_pts, output_width, output_height
    )
    
    warped_bil_sharp = sharpen_image(warped_bil, amount=1.0)
    
    cv2.imwrite('door_rectified_nn.png', warped_nn)
    cv2.imwrite('door_rectified_bilinear.png', warped_bil_sharp)
    print("\n✓ Saved rectified door images")

    visualize_rectification(im, src_pts, warped_nn, warped_bil_sharp, "Door Rectification")

def rectify_sign():
    im = cv2.imread('sign.png')
    if im is None:
        print("Error: Could not load sign.png")
        return
    
    print(f"\nLoaded image: {im.shape}")
    
    print("\nSelect the 4 corners of the sign:")
    print("  1. Top-left corner")
    print("  2. Top-right corner")
    print("  3. Bottom-left corner")
    print("  4. Bottom-right corner")
    
    selector = RectificationSelector(im)
    src_pts = selector.select_quadrilateral()
    
    print("\n✓ Selected 4 corners:")
    labels = ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']
    for i, (pt, label) in enumerate(zip(src_pts, labels)):
        print(f"  {label}: ({pt[0]:.1f}, {pt[1]:.1f})")
    
    width1 = np.linalg.norm(src_pts[1] - src_pts[0])
    width2 = np.linalg.norm(src_pts[3] - src_pts[2])
    height1 = np.linalg.norm(src_pts[2] - src_pts[0])
    height2 = np.linalg.norm(src_pts[3] - src_pts[1])
    
    avg_width = (width1 + width2) / 2
    avg_height = (height1 + height2) / 2
    aspect_ratio = avg_width / avg_height
    
    print(f"\nEstimated dimensions: {avg_width:.1f} x {avg_height:.1f} pixels")
    print(f"Aspect ratio: {aspect_ratio:.2f}")
    
    output_width = int(avg_width * 1.5)
    output_height = int(avg_height * 1.5)
    
    print(f"Output size: {output_width} x {output_height} pixels")
    
    H, warped_nn, warped_bil = rectify_with_custom_size(
        im, src_pts, output_width, output_height
    )

    warped_bil_sharp = sharpen_image(warped_bil, amount=1.2)
    
    cv2.imwrite('sign_rectified_nn.png', warped_nn)
    cv2.imwrite('sign_rectified_bilinear.png', warped_bil_sharp)
    print("\n✓ Saved rectified sign images")

    visualize_rectification(im, src_pts, warped_nn, warped_bil_sharp, "Sign Rectification")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("IMAGE RECTIFICATION TOOL - A.3 DELIVERABLE")
    print("="*70)
    print("\nSelect a mode:")
    print("1. Rectify door.png ")
    print("2. Rectify sign.png ")
    print("3. Run all rectifications (sequential)")
    print("="*70)
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        rectify_door()
    elif choice == '2':
        rectify_sign()
    elif choice == '3':
        rectify_door()
        rectify_sign()
    else:
        print("Invalid choice. Running all rectifications...")
        rectify_door()
        rectify_sign()