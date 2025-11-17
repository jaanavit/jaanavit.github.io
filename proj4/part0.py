import cv2
import numpy as np
import os
import glob

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    from PIL import Image
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

def load_image(image_path):
    try:
        if HEIC_SUPPORT:
            pil_image = Image.open(image_path)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return image
        else:
            return cv2.imread(image_path)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

images_folder = "tags"
marker_size_meters = 0.02

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

marker_3d_points = np.array([
    [0, 0, 0],
    [marker_size_meters, 0, 0],
    [marker_size_meters, marker_size_meters, 0],
    [0, marker_size_meters, 0]
], dtype=np.float32)

all_corners = []
all_object_points = []

image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.heic', '*.HEIC']
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(images_folder, ext)))

print(f"Found {len(image_files)} images")

image_size = None

for image_path in image_files:
    image = load_image(image_path)
    if image is None:
        continue

    if image_size is None:
        image_size = (image.shape[1], image.shape[0])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco_detector.detectMarkers(gray)

    if ids is not None:
        for corner in corners:
            corner_2d = corner.reshape(4, 2).astype(np.float32)
            all_corners.append(corner_2d)
            all_object_points.append(marker_3d_points)
        print(f"Detected {len(ids)} markers in {os.path.basename(image_path)}")
    else:
        print(f"No markers detected in {os.path.basename(image_path)}")

print(f"\nCalibrating camera with {len(all_corners)} detected markers...")

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    all_object_points,
    all_corners,
    image_size,
    None,
    None
)

print(f"\nCalibration complete!")
print(f"Re-projection Error: {ret:.4f} pixels")
print(f"\nCamera Matrix:\n{camera_matrix}")
print(f"\nDistortion Coefficients:\n{dist_coeffs}")

np.savez('camera_calibration.npz',
         camera_matrix=camera_matrix,
         dist_coeffs=dist_coeffs,
         rvecs=rvecs,
         tvecs=tvecs,
         image_size=image_size,
         reprojection_error=ret)

print(f"\nCalibration saved to camera_calibration.npz")