import numpy as np
import open3d as o3d

def fit_plane_ransac(points, max_iterations=1000, threshold=0.01):
    best_inliers = []
    best_plane = None

    num_points = points.shape[0]

    for i in range(max_iterations):
        # Randomly select 3 points to define a plane
        sample_indices = np.random.choice(num_points, 3, replace=False)
        p1, p2, p3 = points[sample_indices]

        # Compute the plane normal vector (cross product of two edges)
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)

        # Ensure the normal is not degenerate
        if np.linalg.norm(normal) == 0:
            continue

        # Normalize the plane normal
        normal = normal / np.linalg.norm(normal)

        # Plane equation: ax + by + cz + d = 0
        a, b, c = normal
        d = -np.dot(normal, p1)

        # Calculate distances of all points to the plane
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)

        # Determine inliers
        inliers = np.where(distances < threshold)[0]

        # Update the best plane if the current one has more inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (a, b, c, d)

    return best_plane, best_inliers

# Load and visualize the demo point cloud
pcd_point_cloud = o3d.data.PCDPointCloud()
pcd = o3d.io.read_point_cloud(pcd_point_cloud.path)

# Convert Open3D point cloud to NumPy array
points = np.asarray(pcd.points)

# Run RANSAC to fit a plane
plane, inliers = fit_plane_ransac(points, max_iterations=1000, threshold=0.02)
print(f"Plane parameters: {plane}")
print(f"Number of inliers: {len(inliers)}")

# Separate inliers and outliers for visualization
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)

# Color inliers and outliers for visualization
inlier_cloud.paint_uniform_color([1, 0, 0])  # Red for inliers

# Save visualization as PNG
def save_visualization(file_name):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)  # Create a window
    vis.add_geometry(inlier_cloud)
    vis.add_geometry(outlier_cloud)
    
    vis.poll_events()
    vis.update_renderer()

    # Adjust view parameters
    ctr = vis.get_view_control()
    ctr.set_zoom(1)
    ctr.set_front([0.4257, -0.2125, -0.8795])
    ctr.set_lookat([2.6172, 2.0475, 1.532])
    ctr.set_up([-0.0694, -0.9768, 0.2024])

    # Allow the visualizer to fully render with the updated camera settings
    for _ in range(10):
        vis.poll_events()
        vis.update_renderer()
    
    # Capture and save screenshot
    vis.capture_screen_image(file_name)
    vis.destroy_window()

# Save the visualization as a PNG
save_visualization("result/pcd_visualization.png")