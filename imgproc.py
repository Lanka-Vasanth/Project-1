import cv2
import numpy as np
import glob
import open3d as o3d
import os
from pathlib import Path
import random

data_root=Path(".")

output_dir=data_root/"point_cloud_files"

output_dir.mkdir(parents=True,exist_ok=True)

file_name="simple_point_cloud.ply"

file_path=output_dir/file_name

# CheckerBoard Dimensions (Number of Inner corners)
checkerboard_dims = (8, 6)  # 8 inner corners along width, 6 along height

# Preparing object points for Standard Checkerboard Pattern
objp = np.zeros((checkerboard_dims[0] * checkerboard_dims[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)

# Storing object points and image points for both cameras
objpoints = []  # 3D points in World Space
imgpoints_left = []  # 2D points in image plane for Left Cam
imgpoints_right = []  # 2D points in image plane for Right Cam

# Loading Images 
images_left = glob.glob('lcalibpicsnew/*.jpg')  # left camera Calibration images path
images_right = glob.glob('rcalibpicsnew/*.jpg')  # Right camera Calibration images path

print(f"Step 2: Loaded {len(images_left)} image pairs for calibration.")

# Processing pair of images
for i, (fname_left, fname_right) in enumerate(zip(images_left, images_right)):
    # Load images
    img_left = cv2.imread(fname_left)
    img_right = cv2.imread(fname_right)
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Finding chessboard corners
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, checkerboard_dims, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, checkerboard_dims, None)

    # Verifying if corners are found in both Images 
    if ret_left and ret_right:
        objpoints.append(objp)  # 3D points
        imgpoints_left.append(corners_left)  # 2D points for left cam
        imgpoints_right.append(corners_right)  # 2D points for right cam

        # Display corners
        cv2.drawChessboardCorners(img_left, checkerboard_dims, corners_left, ret_left)
        cv2.drawChessboardCorners(img_right, checkerboard_dims, corners_right, ret_right)

        print(f"Stage 2: Pair {i + 1} - Corners found successfully.")
        cv2.imshow('Left Camera - Corners', img_left)
        cv2.imshow('Right Camera - Corners', img_right)
        cv2.waitKey(500)


    else:
        print(f"Stage 2: Pair {i + 1} - Corners not found in one or both images.")

cv2.destroyAllWindows()
print("Stage 2: Finished processing all image pairs.")

# Step 3: Calibrate each camera and display results
print("Step 3: Calibrating the left camera...")
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
    objpoints, imgpoints_left, gray_left.shape[::-1], None, None
)
print("Step 3: Left camera calibration complete.")
print("Left Camera Matrix:\n", mtx_left)
print("Left Camera Distortion Coefficients:\n", dist_left)

print("Stage 3: Calibrating the right camera...")
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
    objpoints, imgpoints_right, gray_right.shape[::-1], None, None
)
print("Stage 3: Right camera calibration complete.")
print("Right Camera Matrix:\n", mtx_right)
print("Right Camera Distortion Coefficients:\n", dist_right)

print("Stage 3: Calibration results displayed.")

# Step 4: Perform stereo calibration
print("Step 4: Performing stereo calibration...")

ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right, 
    mtx_left, dist_left, mtx_right, dist_right, 
    gray_left.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC
)

# Step 4: Display stereo calibration results
print("Stage 4: Stereo calibration complete.")
print("Stereo Calibration Results:")
print("Rotation Matrix (R):\n", R)
print("Translation Vector (T):\n", T)
print("Essential Matrix (E):\n", E)
print("Fundamental Matrix (F):\n", F)

# Using the stereo calibration results to rectify images, and compute the disparity map

# Computing the rectification transformations
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx_left, dist_left, mtx_right, dist_right, gray_left.shape[::-1], R, T)

# Creating undistortion, rectification maps for both images
map1x, map1y = cv2.initUndistortRectifyMap(mtx_left, dist_left, R1, P1, gray_left.shape[::-1], cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(mtx_right, dist_right, R2, P2, gray_right.shape[::-1], cv2.CV_32FC1)

# Using the calibration results from previous steps (Camera Matrices and Distortion Coefficients)
left_camera_matrix = mtx_left 
right_camera_matrix = mtx_right
left_dist_coeffs = dist_left
right_dist_coeffs = dist_right

img_size = (gray_left.shape[1], gray_left.shape[0])

# Stereo rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    left_camera_matrix, left_dist_coeffs,
    right_camera_matrix, right_dist_coeffs,
    img_size, R, T,
    alpha=0 
)

# Computing rectification maps
map1x, map1y = cv2.initUndistortRectifyMap(
    left_camera_matrix, left_dist_coeffs, R1, P1, img_size, cv2.CV_32FC1
)
map2x, map2y = cv2.initUndistortRectifyMap(
    right_camera_matrix, right_dist_coeffs, R2, P2, img_size, cv2.CV_32FC1
)

img_left = cv2.imread("leftcam (18).jpg")
img_right = cv2.imread("rightcam (18).jpg")


# Applying rectification to the images
rectified_left = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
rectified_right = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR)

# Displaying rectified images
cv2.imshow('Rectified Left Image', rectified_left)
cv2.imshow('Rectified Right Image', rectified_right)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("rectified_left.jpg", rectified_left)
cv2.imwrite("rectified_right.jpg", rectified_right)

gray_left = cv2.imread("rectified_left.jpg", cv2.IMREAD_GRAYSCALE)
gray_right = cv2.imread("rectified_right.jpg", cv2.IMREAD_GRAYSCALE)

numDisparities = 16 * 12   # Increase or Decrease depending on scene depth range
blockSize =  9 # Increase for Precision

# Creating the StereoSGBM object
stereo_sgbm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=numDisparities,
    blockSize=blockSize,
    P1=8 * (blockSize ** 2),
    P2=32  * (blockSize ** 2), 
    disp12MaxDiff=1,
    preFilterCap=31,
    uniquenessRatio=5,
    speckleWindowSize=200,
    speckleRange=8
)

# Computing the disparity map using StereoSGBM
disparity = stereo_sgbm.compute(gray_left, gray_right).astype(np.float32) / 16.0

# Normalizing and Displaying Disparity Map
disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disparity_normalized = np.uint8(disparity_normalized)

# Displaying results
cv2.imshow('Disparity Map', disparity_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

disparity_filtered = cv2.bilateralFilter(disparity, 5, 50, 150)

# Normalizing and Displaying Filtered Disparity Map
disparity_filtered_normalized = cv2.normalize(disparity_filtered, None, 0, 255, cv2.NORM_MINMAX)
disparity_filtered_normalized = np.uint8(disparity_filtered_normalized)

cv2.imshow('Filtered Disparity Map', disparity_filtered_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

disparity_map_path = output_dir / "disparity_map.png"
cv2.imwrite(str(disparity_map_path), disparity_filtered_normalized)
print(f"Disparity map saved successfully at {disparity_map_path}")

disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disparity_colored = cv2.applyColorMap(np.uint8(disparity_normalized), cv2.COLORMAP_JET)
cv2.imshow('Disparity Map', disparity_colored)
cv2.waitKey(0)
cv2.destroyAllWindows()

focal_length = mtx_left[0, 0]

# Baseline in meters, from the translation vector (T) (Cam to Cam Distance)
B = 0.065 #  Baseline value from the translation vector (T[0])

height, width = disparity.shape

# Initializing Depth Map
depth_map = np.zeros_like(disparity, dtype=np.float32)

# Avoiding Division by 0 by ensuring disparity values are greater than threshold
for v in range(height):
    for u in range(width):
        if disparity[v, u] > 1.0:  # Minimum threshold for disparity
            Z = (focal_length * B) / disparity[v, u]
            depth_map[v, u] = Z

# Normalizing depth map for visualization
depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_map_normalized = depth_map_normalized.astype(np.uint8)

# Displaying Normalized Depth Map
cv2.imshow('Normalized Depth Map', depth_map_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Displaying Disparity Map Range
print("Disparity Map - Min Value:", np.min(disparity))
print("Disparity Map - Max Value:", np.max(disparity))

# Adjusting Disparity Scaling for better Depth Estimation
scale_factor = 5000.0  # Scaling Value
disparity_scaled = np.clip(disparity * scale_factor, 1.0, None)

# Computing Depth Map using Scaled Disparity
depth_map = (focal_length * B) / disparity_scaled

# Normalizing Depth Map to display
depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_map_normalized = depth_map_normalized.astype(np.uint8)

# Displaying the Normalized Depth Map
cv2.imshow('Normalized Depth Map', depth_map_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Applying Color Map for better Depth Visualization
colored_depth_map = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
cv2.imshow('Colored Depth Map', colored_depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()

focal_length = mtx_left[0, 0]
B = 0.065  # Baseline in meters, converted from the translation vector (T[0])

height, width = disparity.shape
depth_map = np.zeros_like(disparity, dtype=np.float32)

for v in range(height):
    for u in range(width):
        if disparity[v, u] > 1.0:  # Using a minimum threshold for disparity
            Z = (focal_length * B) / disparity[v, u]
            depth_map[v, u] = Z

# Normalizing the Depth Map for visualization
depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_map_normalized = np.uint8(depth_map_normalized)

# Saving  depth map
depth_map_path = output_dir / "depth_map.png"
cv2.imwrite(str(depth_map_path), depth_map_normalized)
print(f"Depth map saved successfully at {depth_map_path}")

# Applying Colormap to the entire depth map
depth_colormap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

points_3d = []
colors_3d = []

for v in range(height):
    for u in range(width):
        if depth_map[v, u] > 0:  # Avoid zero-depth points
            # Converting (u, v, depth) to 3D coordinates   
            x = (u - mtx_left[0, 2]) * depth_map[v, u] / focal_length
            y = (v - mtx_left[1, 2]) * depth_map[v, u] / focal_length
            z = depth_map[v, u]
            
            # Geting colour corresponding to pixel (u, v) in rectified left image
            color = rectified_left[v, u] / 255.0  # Normalizing to [0, 1]
            
            points_3d.append([x, y, z])
            colors_3d.append(color)

# Converting list to numpy array
points_3d = np.array(points_3d)
colors_3d = np.array(colors_3d)

# Creating Open3D point cloud object
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points_3d)
point_cloud.colors = o3d.utility.Vector3dVector(colors_3d)

# Saving the point cloud
point_cloud_path = output_dir / "colored_point_cloud.ply"
if o3d.io.write_point_cloud(str(point_cloud_path), point_cloud):
    print(f"Point cloud saved successfully at {point_cloud_path}")
else:
    print(f"Failed to save point cloud at {point_cloud_path}")

# Verifying if the point cloud file exists
if point_cloud_path.exists():
    print("Point cloud file exists and was created successfully:", point_cloud_path)
else:
    print("Point cloud file was not created.")