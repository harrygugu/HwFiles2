import numpy as np
import open3d as o3d
import copy

def icp(source, target, max_iterations=200, tolerance=1e-6, distance_threshold=0.05):
    source_np = np.asarray(source.points)
    target_np = np.asarray(target.points)

    # Initialize transformation
    transformation = np.eye(4)
    prev_error = float('inf')

    # Build a KDTree for the target point cloud
    target_kdtree = o3d.geometry.KDTreeFlann(target)

    for i in range(max_iterations):
        # Transform the source points
        source_transformed = (transformation[:3, :3] @ source_np.T).T + transformation[:3, 3]

        # Find nearest neighbors in the target
        target_tree = o3d.geometry.KDTreeFlann(target)
        correspondences = []
        for pt in source_transformed:
            _, idx, dists = target_tree.search_knn_vector_3d(pt, 1)
            if dists[0] < distance_threshold:  # Filter correspondences by distance
                correspondences.append((pt, target_np[idx[0]]))

        if len(correspondences) < 3:  # Insufficient correspondences to continue
            print(f"Insufficient correspondences after {i+1} iterations.")
            break

        source_corr, target_corr = zip(*correspondences)
        source_corr = np.array(source_corr)
        target_corr = np.array(target_corr)

        # Compute centroids of inlier source and correspondences
        source_centroid = np.mean(source_corr, axis=0)
        target_centroid = np.mean(target_corr, axis=0)

        # Center the points
        source_centered = source_corr - source_centroid
        target_centered = target_corr - target_centroid

        # Compute covariance matrix
        H = source_centered.T @ target_centered

        # Compute Singular Value Decomposition (SVD)
        U, _, VT = np.linalg.svd(H)
        R = VT.T @ U.T
        if np.linalg.det(R) < 0:
            R[:, -1] *= -1  # Fix reflection issue

        t = target_centroid - R @ source_centroid

        # Update transformation matrix
        delta_transformation = np.eye(4)
        delta_transformation[:3, :3] = R
        delta_transformation[:3, 3] = t
        transformation = delta_transformation @ transformation

        # Check for convergence
        error = np.mean(np.linalg.norm(source_corr - target_corr, axis=1))
        print(f"\rIteration {i}/{max_iterations}, error={error}", end="")
        if abs(prev_error - error) < tolerance:
            print(f"\rConverged after {i+1} iterations with error: {error}")
            break
        prev_error = error

    return transformation

# Load demo point clouds
demo_icp_pcds = o3d.data.DemoICPPointClouds()
# source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
# target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
source = o3d.io.read_point_cloud("data\Task2\kitti_frame1.pcd")
target = o3d.io.read_point_cloud("data\Task2\kitti_frame2.pcd")

# Perform ICP
transformation = icp(source, target)
print("Final transformation matrix:")
print(transformation)

# Visualize the results
def draw_registration_result(source, target, transformation, save_path="result/pcd_icp.png"):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    
    # Set the view parameters
    vis.get_view_control().set_zoom(0.4459)
    vis.get_view_control().set_front([0.0, 0.0, 1.0])
    vis.get_view_control().set_lookat([1.6784, 2.0612, 1.4451])
    vis.get_view_control().set_up([0.0, -1.0, 0.0])

    # Save the image
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path)
    vis.destroy_window()

draw_registration_result(source, target, transformation, "result/pcd_icp_kitti.png")
