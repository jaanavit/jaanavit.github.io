import cv2
import numpy as np
import os
import glob
import viser
import time

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    from PIL import Image
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

def load_image(image_path):
    try:
        if HEIC_SUPPORT and image_path.lower().endswith((".heic", ".heif")):
            pil_image = Image.open(image_path)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return image
        else:
            return cv2.imread(image_path)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

calib_data = np.load("camera_calibration.npz")
camera_matrix = calib_data['camera_matrix']
dist_coeffs = calib_data['dist_coeffs']

print("Loaded camera calibration:")
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

images_folder = "./tags1"
marker_size_meters = 0.02

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

marker_3d_points = np.array([
    [0, 0, 0],
    [marker_size_meters, 0, 0],
    [marker_size_meters, marker_size_meters, 0],
    [0, marker_size_meters, 0]
], dtype=np.float32)

image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.heic', '*.HEIC']
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(images_folder, ext)))

print(f"\nFound {len(image_files)} images in '{images_folder}'")

server = viser.ViserServer(share=True)

successful_poses = 0

for i, image_path in enumerate(image_files):
    image = load_image(image_path)
    if image is None:
        continue

    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None and len(ids) > 0:
        corner_2d = corners[0].reshape(4, 2).astype(np.float32)
        
        success, rvec, tvec = cv2.solvePnP(
            marker_3d_points,
            corner_2d,
            camera_matrix,
            dist_coeffs
        )
        
        if success:
            R, _ = cv2.Rodrigues(rvec)
            
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = tvec.squeeze()
            
            c2w = np.linalg.inv(w2c)
            
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            server.scene.add_camera_frustum(
                f"/cameras/{i}",
                fov=2 * np.arctan2(H / 2, camera_matrix[0, 0]),
                aspect=W / H,
                scale=0.02,
                wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
                position=c2w[:3, 3],
                image=img_rgb
            )
            
            successful_poses += 1
            print(f"[OK] {os.path.basename(image_path)} — pose estimated")
        else:
            print(f"[--] {os.path.basename(image_path)} — PnP failed")
    else:
        print(f"[--] {os.path.basename(image_path)} — no marker detected")

print(f"\nSuccessfully estimated poses for {successful_poses}/{len(image_files)} images")
print("Visualization running. Press Ctrl+C to exit.")

while True:
    time.sleep(0.1)
