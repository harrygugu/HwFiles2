import cv2
import numpy as np
import random

# Load the images
left_img = cv2.imread('data/Task3/left.jpg', cv2.IMREAD_COLOR)
right_img = cv2.imread('data/Task3/right.jpg', cv2.IMREAD_COLOR)

# Define the Aruco dictionary
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Detect markers in both images
corners_left, ids_left, _ = detector.detectMarkers(left_img)
corners_right, ids_right, _ = detector.detectMarkers(right_img)

def draw_epilines(img1, img2, points1, points2, F):
    # Compute epipolar lines for points in the first image
    lines1 = cv2.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)

    # Compute epipolar lines for points in the second image
    lines2 = cv2.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)

    # Draw the lines and points
    img1_with_lines = img1.copy()
    img2_with_lines = img2.copy()

    def draw_lines_and_points(img, lines, points):
        for r, point in zip(lines, points):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            x0, y0 = map(int, [0, -r[2] / r[1]])  # Line crossing the left edge
            x1, y1 = map(int, [img.shape[1], -(r[2] + r[0] * img.shape[1]) / r[1]])  # Line crossing the right edge
            cv2.line(img, (x0, y0), (x1, y1), color, 1)
            cv2.circle(img, tuple(map(int, point)), 5, color, -1)

    draw_lines_and_points(img1_with_lines, lines1, points1)
    draw_lines_and_points(img2_with_lines, lines2, points2)

    return img1_with_lines, img2_with_lines

# Ensure markers are detected in both images
if ids_left is not None and ids_right is not None:
    # Find common markers
    ids_left_set = set(ids_left.flatten())
    ids_right_set = set(ids_right.flatten())
    common_ids = list(ids_left_set.intersection(ids_right_set))

    # Collect corresponding points
    points_left = []
    points_right = []
    for marker_id in common_ids:
        idx_left = np.where(ids_left == marker_id)[0][0]
        idx_right = np.where(ids_right == marker_id)[0][0]
        
        # Use the center of the marker as a point (could also use corners)
        center_left = np.mean(corners_left[idx_left][0], axis=0)
        center_right = np.mean(corners_right[idx_right][0], axis=0)

        points_left.append(center_left)
        points_right.append(center_right)

    points_left = np.array(points_left)
    points_right = np.array(points_right)

    # Estimate the fundamental matrix
    F, mask = cv2.findFundamentalMat(points_left, points_right, cv2.FM_RANSAC)
    print("Fundamental Matrix:\n", F)

    # Draw epipolar lines
    img1_with_lines, img2_with_lines = draw_epilines(left_img, right_img, points_left, points_right, F)

    # Show the images with epipolar lines
    cv2.imshow("Image 1 with Epipolar Lines", img1_with_lines)
    cv2.imshow("Image 2 with Epipolar Lines", img2_with_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally, save the images
    cv2.imwrite("result/left_with_epilines.jpg", img1_with_lines)
    cv2.imwrite("result/right_with_epilines.jpg", img2_with_lines)

    # Camera intrinsics (assume these are known or calibrated)
    K = np.array([[1.37700255e+03, 0, 9.89504836e+02],
                [0, 1.38156638e+03, 5.89340505e+02],
                [0,  0,  1]])  # Replace with your camera's intrinsic parameters

    # Compute the Essential Matrix from the Fundamental Matrix
    E = K.T @ F @ K

    # Use cv2.recoverPose to decompose the Essential Matrix into R and t
    points_left_hom = cv2.convertPointsToHomogeneous(points_left).reshape(-1, 1, 2)
    points_right_hom = cv2.convertPointsToHomogeneous(points_right).reshape(-1, 1, 2)

    # Recover the pose (R and t)
    _, R, t, _ = cv2.recoverPose(E, points_left_hom, points_right_hom, K)

    # Output the results
    print("Rotation matrix (R):\n", R)
    print("Translation vector (t):\n", t)
    
else:
    print("Markers not detected in one or both images.")