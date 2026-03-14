import os

import cv2
import matplotlib.pyplot as plt


def _camera_frame_key(path):
    """Extract the numeric frame id from a camera image path for sorting."""
    name = os.path.basename(path)
    return int(name.replace("im", "").replace(".png", ""))


def collect_camera_frames(folder):
    """Collect existing camera image paths from one folder and sort them by frame number."""
    frame_paths = []
    for name in os.listdir(folder):
        if name.startswith("im") and name.endswith(".png"):
            frame_paths.append(os.path.join(folder, name))
    return sorted(frame_paths, key=_camera_frame_key)


def replay_camera_frames(frame_paths, pause_time=0.5, figure_size=(10, 8)):
    """Replay a sequence of RGB camera frames with matplotlib."""
    if not frame_paths:
        print("No camera images found.")
        return

    plt.ion()
    fig, ax = plt.subplots(figsize=figure_size)

    try:
        for image_path in frame_paths:
            img = cv2.imread(image_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.axis("off")
            frame_number = _camera_frame_key(image_path)
            ax.set_title(f"RGB Camera data - Frame {frame_number}")
            plt.pause(pause_time)
            ax.cla()
    except KeyboardInterrupt:
        print("\nPlayback stopped by user.")
    finally:
        plt.ioff()
        plt.show()


def replay_camera_folder(folder, pause_time=0.5):
    """Collect and replay all camera frames from one camera-data folder."""
    frame_paths = collect_camera_frames(folder)
    replay_camera_frames(frame_paths, pause_time=pause_time)


if __name__ == "__main__":
    folder = "V2X-Attack-Dataset/runs/run_000001/sensors/1/camera"
    pause_time = 0.5

    replay_camera_folder(
        folder=folder,
        pause_time=pause_time,
    )
