import cv2
import numpy as np
import glob

# Set the chessboard size
chessboard_size = (8, 6)  # Number of inside corners in the chessboard pattern
square_size = 1.0         # Size of a square in your defined unit (e.g., cm)

# Prepare object points (3D points in the real world space)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # Scale the points according to the square size

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Load all the calibration images
images = glob.glob('data/Task3/*.jpg')  # Replace with the path to your images

for fname in images:
    print("Processing ", fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                           criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(refined_corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, chessboard_size, refined_corners, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Perform camera calibration
ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print calibration results
print("Camera matrix:")
print(camera_matrix)
print("\nDistortion coefficients:")
print(distortion_coeffs)

# # Save the calibration results to a file
# np.savez('calibration_data.npz', camera_matrix=camera_matrix, distortion_coeffs=distortion_coeffs, 
#          rvecs=rvecs, tvecs=tvecs)
