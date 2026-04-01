"""
Helper functions for co-simulation with CARLA
"""
import numpy as np
import carla
from utils.carla_util import snap_to_ground
from clients.METSRClient import METSRClient
import os
from PIL import Image
from queue import Queue
from queue import Empty

"""
Implementation of the CoSim Client

A CoSim client communicates with one METSRClient and one CARLA client to manage the 
data flow between corresponding simulation instances.
"""

# Co-simulation 1: The carla control a submap, and the METS-R SIM control the rest maps
# The visualization of the submap is done in the CARLA simulator
# The visualization of the rest maps is done in the METS-R SIM
class CoSimClient(object):
      def __init__(self, config, carla_client, tm_client):
            self.config = config

            self.carla = carla_client.get_world()
            self.carla_client = carla_client
            self.carla_tm = tm_client
            # self.set_carla_camera(self.carla, config)
            self.set_overlook_camera(self.carla)

            self.metsr = METSRClient(config.metsr_host, int(config.ports[0]), 0, self, verbose = config.verbose)

            self.display_all = config.display_all # display all the vehicles in the CARLA map

            # set the co-sim region - default to empty list if not specified
            metsr_roads = getattr(config, 'metsr_road', [])
            for road in metsr_roads:
                  self.metsr.set_cosim_road(road)

            self.carla_vehs = {} # id of agent and vehicle instance in carla
            self.carla_coordMaps = {} # legacy metadata kept for compatibility with existing scripts
            self.carla_route = {}
            self.carla_destRoad = {}
            self.carla_entered = {} # legacy state bit; ownership now comes from query_coSimVehicle()
            self.carla_private_flags = {}
            self.carla_handoff_locs = {}
            self.carla_handoff_yaws = {}

            self.displayOnly_vehs = {} # id of agent and vehicle controlled by METSR, only used for display all vehicles

            self.carla_waiting_vehs = [] # legacy waiting list kept for compatibility
            self.waypoints = self.carla.get_map().generate_waypoints(2.0) # generate all waypoints at 2-meter intervals

            self.carla_veh_sensors = {} # id of vehicle agents : sensor : sensor instances in CARLA belonging to this vehicle
            self.carla_veh_dataCollect = set() # id of vehicle agents whose sensor data will be collected
            self.carla_veh_sensor_queues = {} # id of vehicle agents : sensor : FIFO queue for storing sensor data

      def set_overlook_camera(self, world): # set the camera to overlook the whole map
            spectator = world.get_spectator()
            transform = carla.Transform()
            transform.location.x = 0
            transform.location.y = 0
            transform.location.z = 300
            transform.rotation.yaw = -90
            transform.rotation.pitch = -90
            spectator.set_transform(transform)

      def set_custom_camera(self, x, y, z):
            spectator = self.carla.get_spectator()
            transform = carla.Transform()
            transform.location.x = x
            transform.location.y = y
            transform.location.z = z
            transform.rotation.yaw = -90
            transform.rotation.pitch = -90
            spectator.set_transform(transform)

      def get_carla_location(self, metsr_x, metsr_y):
            # given x, y, find the corresponding z values and rotation in CARLA
            x, y = metsr_x, -metsr_y
            location = carla.Location(x, y, 0)
            location = snap_to_ground(self.carla, location, 0.05)
            return location
      
      def get_carla_rotation(self, veh_inform):
            # veh_inform['bearing']: compass heading (° clockwise from north)
            bearing = veh_inform['bearing'] % 360
            carla_yaw = (bearing - 90) % 360
            rotation = carla.Rotation(pitch=0.0, yaw=carla_yaw, roll=0.0)
            return rotation, carla_yaw
      
      def get_metsr_rotation(self, carla_yaw):
            """
            Invert carla_yaw = (bearing - 90) % 360
            to recover the original METSR compass bearing.
            """
            # ensure 0 ≤ yaw < 360
            carla_yaw = carla_yaw % 360
            # invert the shift of -90°
            return (carla_yaw + 90) % 360

      def is_in_carla_submap(self, x, y):
            # project x, y to the nearest road in CARLA and check if the road ID is in the co-sim road
            road_id = self.carla.get_map().get_waypoint(carla.Location(x, y), project_to_road=True, lane_type=(carla.LaneType.Driving)).road_id
            #return True
            return road_id in self.config.carla_road
            
      def step(self):
            """
            Advance both simulators and reconcile vehicle ownership.

            Ownership is determined only by METS-R query_coSimVehicle():
            - not managed yet: appears in the ownership set, so spawn a CARLA actor
            - managed by CARLA: stays in the ownership set, so only sync state back
            - left co-sim: disappears from the ownership set, so hand control back to METS-R
            """
            self.carla.tick()
            self.metsr.tick()

            cosim_vehs = self.metsr.query_coSimVehicle()['DATA']
            cosim_ids = [v['ID'] for v in cosim_vehs]
            cosim_private_flags = [v['v_type'] for v in cosim_vehs]
            cosim_info_map = {}
            cosim_meta_map = {}
            if cosim_ids:
                  all_data = self.metsr.query_vehicle(cosim_ids, cosim_private_flags, transform_coords=True)['DATA']
                  for cosim_id, cosim_veh, private_flag, veh_info in zip(cosim_ids, cosim_vehs, cosim_private_flags, all_data):
                        cosim_meta_map[cosim_id] = cosim_veh
                        cosim_info_map[cosim_id] = (private_flag, veh_info)

            current_cosim_ids = set(cosim_ids)
            managed_ids = set(self.carla_vehs.keys())

            # State 1: the vehicle left the co-sim ownership set and should no longer be CARLA-managed.
            for vid in managed_ids - current_cosim_ids:
                  print(f"Vehicle {vid} left the co-sim ownership set and is on longer CARLA-managed.")
                  self.handoff_carla_vehicle(vid)

            # State 2: the vehicle just entered the ownership set, so spawn a CARLA actor for it.
            for cosim_id in cosim_ids:
                  private_flag, veh_info = cosim_info_map[cosim_id]
                  self.carla_private_flags[cosim_id] = private_flag
                  if cosim_id not in self.carla_vehs and veh_info['state'] > 0:
                        if cosim_id in self.displayOnly_vehs:
                              print(f"Vehicle {cosim_id} switched from display-only to CARLA-managed.")
                              self.destroy_carla_vehicle(cosim_id)
                        self.carla_handoff_locs[cosim_id] = self.get_carla_location(veh_info['x'], veh_info['y'])
                        _, handoff_yaw = self.get_carla_rotation(veh_info)
                        self.carla_handoff_yaws[cosim_id] = handoff_yaw
                        self.spawn_carla_vehicle(cosim_id, private_flag, veh_info, display_only=False)
                        self.carla_coordMaps[cosim_id] = cosim_meta_map[cosim_id].get('coord_map', [])
                        self.carla_route[cosim_id] = cosim_meta_map[cosim_id].get('route', [])
                        route = self.carla_route[cosim_id]
                        self.carla_destRoad[cosim_id] = route[-1] if route else None
                        self.carla_entered[cosim_id] = True
                        print(f"Vehicle {cosim_id} entered the co-sim ownership set and is now CARLA-managed.")

            # State 3: the vehicle remains in the ownership set, so only synchronize CARLA state back to METS-R.
            for cosim_id in cosim_ids:
                  if cosim_id in self.carla_vehs:
                        private_flag, veh_info = cosim_info_map[cosim_id]
                        self.sync_carla_vehicle(cosim_id, private_flag, veh_info)
                        #print(f"Vehicle {cosim_id} is CARLA-managed and its state is synchronized back to METS-R.")

            if self.display_all:
                  self.sync_display_only_vehicles(current_cosim_ids)
     
      def run(self):
            try:
                  for t in range(int(self.config.sim_minutes * 60 / self.config.sim_step_size)):
                        print("Tick:", t)
                        if t % 600 == 0:
                              # generate 10 random trips every 1 minute
                              self.generate_random_trips(10, start_vid = int(t // 6))
                              print(f"Generated 10 random trips at time {t * self.config.sim_step_size // 60} minute!")
                        self.step()
            except KeyboardInterrupt:
                  print("simulation interrupted by user")

            finally:
                  self.metsr.terminate()

      def get_distance(self, x1, y1, x2, y2):
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
      
      def spawn_carla_vehicle(self, vid, private_veh, veh_inform, display_only=False):
            tmp_rotation, tmp_yaw = self.get_carla_rotation(veh_inform)
            spawn_loc = self.get_carla_location(veh_inform['x'], veh_inform['y'])
            spawn_point = carla.Transform(spawn_loc, tmp_rotation)

            blueprint = self.carla.get_blueprint_library().find('vehicle.audi.tt' if private_veh else 'vehicle.tesla.model3')
            tmp_speed = max(min(veh_inform['speed'], 10), 5)
            tmp_speed_x = tmp_speed * np.cos(tmp_yaw * np.pi / 180)
            tmp_speed_y = tmp_speed * np.sin(tmp_yaw * np.pi / 180)
            if not display_only:
                  print(
                        f"[handoff] veh={vid} metsr_speed={veh_inform['speed']:.2f} "
                        f"spawn_loc=({spawn_loc.x:.2f},{spawn_loc.y:.2f},{spawn_loc.z:.2f}) "
                        f"spawn_yaw={tmp_yaw:.2f} "
                        f"target_velocity=({tmp_speed_x:.2f},{tmp_speed_y:.2f})"
                  )

            tmp_veh = self.carla.try_spawn_actor(blueprint, spawn_point)

            if tmp_veh:
                  # Start each handoff from a neutral control state so the new physics actor does not
                  # inherit an arbitrary steering command before the CARLA-side controller takes over.
                  tmp_veh.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
                  tmp_veh.set_autopilot(False)
                  #self.carla_tm.ignore_lights_percentage(tmp_veh, 100)
                  tmp_veh.set_target_velocity(carla.Vector3D(x=tmp_speed_x, y=tmp_speed_y, z=0))

                  if display_only:
                        tmp_veh.set_simulate_physics(False) # Fix the bug of roll axis shaking
                        tmp_veh.set_autopilot(False)
                        self.displayOnly_vehs[vid] = tmp_veh
                  else:
                        self.carla_vehs[vid] = tmp_veh

                  if vid in self.carla_veh_dataCollect:
                        self.deploy_vehicle_sensors(vid)

      def destroy_carla_vehicle(self, vid):
            if vid in self.carla_vehs:
                  self.carla_vehs[vid].set_autopilot(False)
                  try:
                        self.carla_vehs[vid].destroy()
                  except:
                        pass
                  self.carla_vehs.pop(vid, None)
                  self.carla_coordMaps.pop(vid, None)
                  self.carla_route.pop(vid, None)
                  self.carla_destRoad.pop(vid, None)
                  self.carla_entered.pop(vid, None)
                  self.carla_private_flags.pop(vid, None)
                  self.carla_handoff_locs.pop(vid, None)
                  self.carla_handoff_yaws.pop(vid, None)
                  if vid in self.carla_waiting_vehs:
                        self.carla_waiting_vehs.remove(vid)
            if vid in self.displayOnly_vehs:
                  try:
                        self.displayOnly_vehs[vid].destroy()
                  except:
                        pass
                  self.displayOnly_vehs.pop(vid, None)
            self.destroy_vehicle_sensors(vid)

      def update_display_only_vehicle(self, vid, veh_inform):
            carla_veh = self.displayOnly_vehs.get(vid)
            if carla_veh:
                  target_loc = self.get_carla_location(veh_inform['x'], veh_inform['y'])
                  tmp_rotation, _ = self.get_carla_rotation(veh_inform)
                  carla_veh.set_transform(carla.Transform(target_loc, tmp_rotation))

      def handoff_carla_vehicle(self, vid):
            """
            Stop CARLA-side management once METS-R no longer reports this vehicle as co-sim owned.

            Trip completion is not decided here. If display_all is enabled, the same vehicle can
            appear again as a display-only METS-R vehicle until METS-R marks it finished.
            """
            if vid in self.carla_vehs:
                  print(f"Vehicle {vid} left the co-sim ownership set.")
                  self.destroy_carla_vehicle(vid)

      def sync_display_only_vehicles(self, current_cosim_ids):
            """
            Mirror METS-R-controlled vehicles in CARLA for visualization only.

            These actors are never part of the co-sim ownership set. They are spawned when METS-R
            reports them alive and destroyed once METS-R reports state <= 0.
            """
            private_agents = self.metsr.query_vehicle()['private_vids']

            batch_size = 10
            for i in range(0, len(private_agents), batch_size):
                  batch_ids = private_agents[i:i+batch_size]
                  batch_infos = self.metsr.query_vehicle(batch_ids, private_veh=True, transform_coords=True)['DATA']

                  for vid, veh_info in zip(batch_ids, batch_infos):
                        if vid in current_cosim_ids or vid in self.carla_vehs:
                              continue
                        if vid not in self.displayOnly_vehs:
                              if veh_info['state'] > 0:
                                    self.spawn_carla_vehicle(vid, True, veh_info, display_only=True)
                        else:
                              if veh_info['state'] > 0:
                                    import time
                                    time.sleep(0.0001)
                                    self.update_display_only_vehicle(vid, veh_info)
                              else:
                                    self.destroy_carla_vehicle(vid)
      
      def sync_carla_vehicle(self, vid, private_veh, veh_inform):
            """
            Synchronize one CARLA-managed vehicle back to METS-R.

            This function no longer performs path pushing, submap checks, or destination handling.
            It only reads the current CARLA pose/speed and teleports that state into METS-R.
            """
            try:
                  carla_veh = self.carla_vehs[vid]
                  loc = carla_veh.get_location()
            except RuntimeError:
                  # re-add the vehicle if it is removed by CARLA
                  print(f"Vehicle {vid} removed by CARLA, re-adding it.")
                  self.destroy_carla_vehicle(vid)
                  self.spawn_carla_vehicle(vid, private_veh, veh_inform, display_only=False)
                  carla_veh = self.carla_vehs[vid]
                  loc = carla_veh.get_location()
            vel = carla_veh.get_velocity()
            speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5
            bearing = self.get_metsr_rotation(carla_veh.get_transform().rotation.yaw)
            self.metsr.teleport_cosim_vehicle(vid, loc.x, -loc.y, bearing, speed=speed, private_veh=private_veh, transform_coords=True)


      def generate_random_trips(self, num_trips, start_vid = 0):
            self.metsr.generate_trip(list(range(start_vid, start_vid+num_trips))) 

      def enable_vehicle_sensor(self, vid):
            self.carla_veh_dataCollect.add(vid)

      def disable_vehicle_sensor(self, vid):
            self.carla_veh_dataCollect.discard(vid)

      def deploy_vehicle_sensors(self, vid):
            if vid in self.carla_veh_sensors:
                  return  # already deployed
            bp_lib = self.carla.get_blueprint_library()
            image_width = 680
            image_height = 420
            camera_bp = bp_lib.filter("sensor.camera.rgb")[0]
            lidar_bp = bp_lib.filter("sensor.lidar.ray_cast")[0]
            # Configure the blueprints
            camera_bp.set_attribute("image_size_x", str(image_width))
            camera_bp.set_attribute("image_size_y", str(image_height))

            lidar_bp.set_attribute('dropoff_general_rate', '0.0')
            lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
            lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
            lidar_bp.set_attribute('upper_fov', str(30))
            lidar_bp.set_attribute('lower_fov', str(-25))
            lidar_bp.set_attribute('channels', str(64.0))
            lidar_bp.set_attribute('range', str(100))
            lidar_bp.set_attribute('points_per_second', str(100000))

            if vid in self.carla_vehs:
                  vehicle = self.carla_vehs[vid]
            elif vid in self.displayOnly_vehs:
                  vehicle = self.displayOnly_vehs[vid]

            self.carla_veh_sensors[vid] = {}
            self.carla_veh_sensor_queues[vid] = {}
            camera = self.carla.spawn_actor(
                  blueprint=camera_bp,
                  transform=carla.Transform(carla.Location(x=1.6, z=1.6)),
                  attach_to=vehicle)
            lidar = self.carla.spawn_actor(
                  blueprint=lidar_bp,
                  transform=carla.Transform(carla.Location(x=1.0, z=1.8)),
                  attach_to=vehicle)
            self.carla_veh_sensor_queues[vid]['camera'] = Queue()
            self.carla_veh_sensor_queues[vid]['lidar'] = Queue()
            camera.listen(lambda data: self.sensor_callback(data, self.carla_veh_sensor_queues[vid]['camera']))
            lidar.listen(lambda data: self.sensor_callback(data, self.carla_veh_sensor_queues[vid]['lidar']))
            
            self.carla_veh_sensors[vid]['camera'] = camera
            self.carla_veh_sensors[vid]['lidar'] = lidar

      def sensor_callback(self, data, queue):
            queue.put(data)
      
      def destroy_vehicle_sensors(self, vid):
            if vid in self.carla_veh_sensors:
            # Destroy the sensors in the scene.
                  if self.carla_veh_sensors[vid]['camera']:
                        self.carla_veh_sensors[vid]['camera'].destroy()
                        while True:
                              try:
                                    self.carla_veh_sensor_queues[vid]['camera'].get_nowait()
                              except Empty:
                                    break
                  if self.carla_veh_sensors[vid]['lidar']:
                        self.carla_veh_sensors[vid]['lidar'].destroy()
                        while True:
                              try:
                                    self.carla_veh_sensor_queues[vid]['lidar'].get_nowait()
                              except Empty:
                                    break
                  self.carla_veh_sensors.pop(vid, None)

      def collect_sensor_data(self, output_path=None):
            for vid in self.carla_veh_dataCollect:
                  self.save_sensor_data(vid, output_path)
            
      def save_sensor_data(self, vid, output_path=None):
            if vid not in self.carla_vehs and vid not in self.displayOnly_vehs:
                  return
            elif vid not in self.carla_veh_sensors:
                  print(f"[Warning] Vehicle {vid} has no deployed sensor.")
                  return
            #camera = self.carla_veh_sensors[vid]['camera']
            #image_w = int(camera.attributes['image_size_x'])
            #image_h = int(camera.attributes['image_size_y'])
            #fov = float(camera.attributes['fov'])
            #focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

            #image_data = self.carla_veh_sensor_queues[vid]['camera'].get(True, 1.0)
            if output_path is None:
                  output_path = "_out"
            image_data = None
            while True:
                  try:
                        image_data = self.carla_veh_sensor_queues[vid]['camera'].get_nowait()
                  except Empty:
                        break
            if image_data is not None:
                  # Get the raw BGRA buffer and convert it to an array of RGB of
                  # shape (image_data.height, image_data.width, 3).
                  im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
                  im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
                  im_array = im_array[:, :, :3][:, :, ::-1]
                  # Save the image using Pillow module.
                  image = Image.fromarray(im_array)
                  os.makedirs(os.path.join(output_path, str(vid), "camera"), exist_ok=True)
                  image.save(os.path.join(output_path, str(vid), "camera", f"im{image_data.frame:08d}.png"))
            else:
                  print(f"[Warning] Some Camera data for vehicle {vid} has been missed")
            #lidar_data = self.carla_veh_sensor_queues[vid]['lidar'].get(True, 1.0)
            lidar_data = None
            while True:
                  try:
                        lidar_data = self.carla_veh_sensor_queues[vid]['lidar'].get_nowait()
                  except Empty:
                        break
            if lidar_data is not None:
                  p_cloud_size = len(lidar_data)
                  p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
                  p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))
                  os.makedirs(os.path.join(output_path, str(vid), "lidar"), exist_ok=True)
                  lidar_points = np.column_stack((p_cloud[:, 0], p_cloud[:, 1], p_cloud[:, 2], p_cloud[:, 3]))
                  np.savez_compressed(os.path.join(output_path, str(vid), "lidar", f"lidar_{lidar_data.frame:08d}.npz"), lidar=lidar_points)
            else:
                print(f"[Warning] Some Lidar data for vehicle {vid} has been missed")
                

            
