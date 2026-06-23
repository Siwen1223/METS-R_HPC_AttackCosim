"""CARLA vehicle sensor deployment and data capture helpers."""

import os
from queue import Empty, Queue

import numpy as np
from PIL import Image
import carla


CAMERA_LAYOUTS = {
    "front": (("front", carla.Transform(carla.Location(x=1.6, z=1.6))),),
    "front_rear": (
        ("front", carla.Transform(carla.Location(x=1.6, z=1.6))),
        ("rear", carla.Transform(carla.Location(x=-1.4, z=1.6), carla.Rotation(yaw=180))),
    ),
    "front_rear_left_right": (
        ("front", carla.Transform(carla.Location(x=1.6, z=1.6))),
        ("rear", carla.Transform(carla.Location(x=-1.4, z=1.6), carla.Rotation(yaw=180))),
        ("left", carla.Transform(carla.Location(y=-0.45, z=1.6), carla.Rotation(yaw=-90))),
        ("right", carla.Transform(carla.Location(y=0.45, z=1.6), carla.Rotation(yaw=90))),
    ),
}


class SensorManager:
    """Manage RGB cameras and LiDAR attached to CARLA vehicles."""

    def __init__(self, world, vehicle_lookup, output_path="_out", camera_layout="front"):
        self.world = world
        self.vehicle_lookup = vehicle_lookup
        self.output_path = output_path
        self.camera_layout = camera_layout
        self.enabled_vids = set()
        self.sensors = {}
        self.queues = {}

    def enable_vehicle(self, vid, deploy_now=True):
        self.enabled_vids.add(vid)
        if deploy_now:
            self.deploy_vehicle_sensors(vid)

    def disable_vehicle(self, vid, destroy=True):
        self.enabled_vids.discard(vid)
        if destroy:
            self.destroy_vehicle_sensors(vid)

    def deploy_vehicle_sensors(self, vid, camera_layout=None):
        if vid in self.sensors:
            return
        vehicle = self.vehicle_lookup(vid)
        if vehicle is None:
            return

        layout_name = camera_layout or self.camera_layout
        camera_layout = CAMERA_LAYOUTS.get(layout_name)
        if camera_layout is None:
            raise ValueError(
                "camera_layout must be one of: " + ", ".join(sorted(CAMERA_LAYOUTS))
            )

        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.filter("sensor.camera.rgb")[0]
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_bp.set_attribute("fov", "110")

        lidar_bp = bp_lib.filter("sensor.lidar.ray_cast")[0]
        lidar_bp.set_attribute("dropoff_general_rate", "0.0")
        lidar_bp.set_attribute("dropoff_intensity_limit", "1.0")
        lidar_bp.set_attribute("dropoff_zero_intensity", "0.0")
        lidar_bp.set_attribute("upper_fov", "15")
        lidar_bp.set_attribute("lower_fov", "-25")
        lidar_bp.set_attribute("channels", "64")
        lidar_bp.set_attribute("range", "100")
        lidar_bp.set_attribute("points_per_second", "100000")
        lidar_bp.set_attribute("noise_stddev", "0.02")

        self.sensors[vid] = {}
        self.queues[vid] = {}

        for name, transform in camera_layout:
            queue = Queue()
            camera = self.world.spawn_actor(camera_bp, transform, attach_to=vehicle)
            camera.listen(lambda data, q=queue: q.put(data))
            sensor_name = f"camera_{name}"
            self.sensors[vid][sensor_name] = camera
            self.queues[vid][sensor_name] = queue

        lidar_queue = Queue()
        lidar = self.world.spawn_actor(
            lidar_bp,
            carla.Transform(carla.Location(x=1.0, z=1.8)),
            attach_to=vehicle,
        )
        lidar.listen(lambda data, q=lidar_queue: q.put(data))
        self.sensors[vid]["lidar"] = lidar
        self.queues[vid]["lidar"] = lidar_queue

    def destroy_vehicle_sensors(self, vid):
        for sensor in self.sensors.get(vid, {}).values():
            try:
                sensor.destroy()
            except Exception:
                pass
        for queue in self.queues.get(vid, {}).values():
            self._drain(queue)
        self.sensors.pop(vid, None)
        self.queues.pop(vid, None)

    def collect_sensor_data(self, output_path=None):
        for vid in list(self.enabled_vids):
            self.save_sensor_data(vid, output_path=output_path)

    def save_sensor_data(self, vid, output_path=None):
        if vid not in self.sensors:
            return
        output_path = output_path or self.output_path
        for name, queue in self.queues.get(vid, {}).items():
            data = self._latest(queue)
            if data is None:
                print(f"[Warning] Some {name} data for vehicle {vid} has been missed")
                continue
            if name.startswith("camera_"):
                self._save_camera(vid, name, data, output_path)
            elif name == "lidar":
                self._save_lidar(vid, data, output_path)

    def _save_camera(self, vid, name, image_data, output_path):
        image_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
        image_array = np.reshape(image_array, (image_data.height, image_data.width, 4))
        image_array = image_array[:, :, :3][:, :, ::-1]
        image = Image.fromarray(image_array)
        folder = os.path.join(output_path, str(vid), name)
        os.makedirs(folder, exist_ok=True)
        image.save(os.path.join(folder, f"im{image_data.frame:08d}.png"))

    def _save_lidar(self, vid, lidar_data, output_path):
        cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype("f4")))
        cloud = np.reshape(cloud, (len(lidar_data), 4))
        folder = os.path.join(output_path, str(vid), "lidar")
        os.makedirs(folder, exist_ok=True)
        np.savez_compressed(
            os.path.join(folder, f"lidar_{lidar_data.frame:08d}.npz"),
            lidar=cloud,
        )

    @staticmethod
    def _latest(queue):
        data = None
        while True:
            try:
                data = queue.get_nowait()
            except Empty:
                return data

    @staticmethod
    def _drain(queue):
        while True:
            try:
                queue.get_nowait()
            except Empty:
                return
