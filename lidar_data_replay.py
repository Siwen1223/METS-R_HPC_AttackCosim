import os
import time
import numpy as np
import open3d as o3d
from matplotlib.colors import LinearSegmentedColormap

vid = 2
save_folder = "out_1221"
folder = save_folder + f"/{vid}/lidar"
start_frame = 1335
end_frame = 1850
pause_time = 0.5

lidar_files = [
    os.path.join(folder, f"lidar_{frame:08d}.npz")
    for frame in range(start_frame, end_frame + 1)
    if os.path.exists(os.path.join(folder, f"lidar_{frame:08d}.npz")) and frame%5==0
]

if not lidar_files:
    print("No LiDAR files found for vehicle", vid)
    exit()

# colormap: red -> orange -> yellow -> green
colors_list = ["red", "orange", "yellow", "green"]
cmap = LinearSegmentedColormap.from_list("red_to_green", colors_list)

# vehicle center for distance reference
all_points = []
for lidar_path in lidar_files:
    data = np.load(lidar_path)['lidar']
    all_points.append(data[:, :3])
all_points = np.vstack(all_points)
center = all_points.mean(axis=0)

# initialize first frame
data = np.load(lidar_files[0])['lidar']
points = data[:, :3].astype(np.float64)
dist = np.linalg.norm(points - center, axis=1)
dist_norm = np.clip((dist - dist.min()) / (dist.max() - dist.min()), 0, 1)
colors = cmap(dist_norm)[:, :3]  # near red, far green

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

vis = o3d.visualization.Visualizer()
vis.create_window(window_name=f"LiDAR point cloud vehicle {vid}", width=1280, height=720)
vis.add_geometry(pcd)

opt = vis.get_render_option()
opt.background_color = np.array([0, 0, 0])
opt.point_size = 1.0

ctr = vis.get_view_control()
ctr.set_lookat(center)
ctr.set_front([-1.5, 0.0, 0.5])
ctr.set_up([0.0, 0.0, 1.0])
ctr.set_zoom(0.2)

# dynamic update
try:
    for lidar_path in lidar_files:
        data = np.load(lidar_path)['lidar']
        points = data[:, :3].astype(np.float64)
        dist = np.linalg.norm(points - center, axis=1)
        dist_norm = np.clip((dist - dist.min()) / (dist.max() - dist.min()), 0, 1)
        colors = cmap(dist_norm)[:, :3]  # near red, far green

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(pause_time)

except KeyboardInterrupt:
    print("\nPlayback stopped by user.")

vis.destroy_window()
