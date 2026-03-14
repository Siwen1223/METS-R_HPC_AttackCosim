import os
import time

import numpy as np
import open3d as o3d
from matplotlib.colors import LinearSegmentedColormap


def _lidar_frame_key(path):
    """Extract the numeric frame id from a LiDAR file path for sorting."""
    name = os.path.basename(path)
    return int(name.replace("lidar_", "").replace(".npz", ""))


def collect_lidar_files(folder):
    """Collect existing LiDAR frame files from one folder and sort them by frame number."""
    lidar_files = []
    for name in os.listdir(folder):
        if name.startswith("lidar_") and name.endswith(".npz"):
            lidar_files.append(os.path.join(folder, name))
    return sorted(lidar_files, key=_lidar_frame_key)


def build_distance_colormap():
    """Build the red-to-green colormap used for LiDAR visualization."""
    colors_list = ["red", "orange", "yellow", "green"]
    return LinearSegmentedColormap.from_list("red_to_green", colors_list)


def load_lidar_points(lidar_path):
    """Load one LiDAR frame and return its XYZ point coordinates."""
    data = np.load(lidar_path)["lidar"]
    return data[:, :3].astype(np.float64)


def compute_point_cloud_center(lidar_files):
    """Compute a shared point-cloud center from all replay frames."""
    all_points = []
    for lidar_path in lidar_files:
        all_points.append(load_lidar_points(lidar_path))
    return np.vstack(all_points).mean(axis=0)


def colorize_points(points, center, cmap):
    """Assign colors to LiDAR points based on their distance to a reference center."""
    dist = np.linalg.norm(points - center, axis=1)
    dist_range = dist.max() - dist.min()
    if dist_range <= 1e-9:
        dist_norm = np.zeros_like(dist)
    else:
        dist_norm = np.clip((dist - dist.min()) / dist_range, 0, 1)
    return cmap(dist_norm)[:, :3]


def create_visualizer(title, center):
    """Create and configure an Open3D visualizer window."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=720)

    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])
    opt.point_size = 1.0

    ctr = vis.get_view_control()
    ctr.set_lookat(center)
    ctr.set_front([-1.5, 0.0, 0.5])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(0.2)
    return vis


def replay_lidar_files(lidar_files, pause_time=0.5):
    """Replay a sequence of LiDAR files in an Open3D visualizer."""
    if not lidar_files:
        print("No LiDAR files found.")
        return

    cmap = build_distance_colormap()
    center = compute_point_cloud_center(lidar_files)

    first_points = load_lidar_points(lidar_files[0])
    first_colors = colorize_points(first_points, center, cmap)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(first_points)
    pcd.colors = o3d.utility.Vector3dVector(first_colors)

    vis = create_visualizer("LiDAR point cloud replay", center)
    vis.add_geometry(pcd)

    try:
        for lidar_path in lidar_files:
            points = load_lidar_points(lidar_path)
            colors = colorize_points(points, center, cmap)
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(pause_time)
    except KeyboardInterrupt:
        print("\nPlayback stopped by user.")
    finally:
        vis.destroy_window()


def replay_lidar_folder(folder, pause_time=0.5):
    """Collect and replay all LiDAR frames from one LiDAR-data folder."""
    lidar_files = collect_lidar_files(folder)
    replay_lidar_files(lidar_files, pause_time=pause_time)


if __name__ == "__main__":
    folder = "V2X-Attack-Dataset/runs/run_000001/sensors/1/lidar"
    pause_time = 0.5

    replay_lidar_folder(
        folder=folder,
        pause_time=pause_time,
    )
