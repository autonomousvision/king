import random
import torch
import copy
import time
import carla

import numpy as np
import torch.nn.functional as F

from collections import deque
from proxy_simulator.utils import state_dict_to_transform, _GradientScaling


# Global Flags
PIXELS_PER_METER = 5


class BaseRenderer():
    """
        Base class. Implements various common things, such as coordinate transforms,
        visualization and a function to render local views of the global map.
    """
    def __init__(self, args, map_offset, map_dims, viz=False):
        """
        """
        self.args = args
        self.map_offset = map_offset
        self.map_dims = map_dims

        if viz:
            self.PIXELS_AHEAD_VEHICLE = torch.tensor(
                [0],  # for visualization we center the ego vehicle
                device=self.args.device,
                dtype=torch.float32
            )
            self.crop_dims = (300, 300)  # we use a bigger crop for visualization
        else:
            self.PIXELS_AHEAD_VEHICLE = torch.tensor(
                [100 + 10],
                device=self.args.device,
                dtype=torch.float32
            )
            self.crop_dims = (192, 192)

        self.gpu_pi = torch.tensor([np.pi], device=self.args.device, dtype=torch.float32)

        self.crop_scale = (
            self.crop_dims[1] / self.map_dims[1],
            self.crop_dims[0] / self.map_dims[0]
        )

        # we precompute several static quantities for efficiency
        self.world_to_rel_map_dims = torch.tensor(
            [self.map_dims[1],self.map_dims[0]],
            device=self.args.device,
            dtype=torch.float32
        )

        self.world_to_pix_crop_shift_tensor = torch.tensor(
            [0., - self.PIXELS_AHEAD_VEHICLE],
            device=self.args.device,
            dtype=torch.float32
        )

        self.world_to_pix_crop_half_crop_dims = torch.tensor(
            [self.crop_dims[1] / 2, self.crop_dims[0] / 2],
            device=self.args.device
        )

        self.get_local_birdview_scale_transform = torch.tensor(
            [[self.crop_scale[1], 0, 0],
            [0, self.crop_scale[0], 0],
            [0, 0, 1]],
            device=self.args.device,
            dtype=torch.float32,
        ).view(1, 3, 3).expand(self.args.batch_size, -1, -1)

        self.get_local_birdview_shift_tensor = torch.tensor(
            [0., - 2 * self.PIXELS_AHEAD_VEHICLE / self.map_dims[0]],
            device=self.args.device,
            dtype=torch.float32,
        )

    def get_local_birdview(self, global_map, position, orientation):
        """
        """
        global_map = global_map.expand(self.args.batch_size, -1, -1, -1)

        # convert position from world to relative image coordinates
        position = self.world_to_rel(position)
        orientation = orientation + self.gpu_pi / 2

        scale_transform = self.get_local_birdview_scale_transform

        zeros = torch.zeros_like(orientation)
        ones = torch.ones_like(orientation)

        rotation_transform = torch.stack(
            [torch.cos(orientation), -torch.sin(orientation), zeros,
             torch.sin(orientation),  torch.cos(orientation), zeros,
             zeros,                   zeros,                  ones],
             dim=-1,
        ).view(self.args.batch_size, 3, 3)

        shift = self.get_local_birdview_shift_tensor

        position = position + (rotation_transform[:, 0:2, 0:2] @ shift).unsqueeze(1)

        translation_transform = torch.stack(
            [ones,  zeros, position[..., 0:1] / self.crop_scale[0],
             zeros, ones,  position[..., 1:2] / self.crop_scale[1],
             zeros, zeros, ones],
            dim=-1,
        ).view(self.args.batch_size, 3, 3)

        # chain tansforms
        local_view_transform = scale_transform @ translation_transform @ rotation_transform

        affine_grid = F.affine_grid(
            local_view_transform[:, 0:2, :],
            (self.args.batch_size, 1, self.crop_dims[0], self.crop_dims[0]),
            align_corners=True,
        )

        # loop saves gpu memory
        local_views = []
        for batch_idx in range(self.args.batch_size):
            local_view_per_elem = F.grid_sample(
                global_map[batch_idx:batch_idx+1],
                affine_grid[batch_idx:batch_idx+1],
                align_corners=True,
            )
            local_views.append(local_view_per_elem)
        local_view = torch.cat(local_views, dim=0)

        return local_view

    def world_to_pix(self, pos):
        pos_px = (pos-self.map_offset) * PIXELS_PER_METER

        return pos_px

    def world_to_rel(self, pos):
        pos_px = self.world_to_pix(pos)
        pos_rel = pos_px / self.world_to_rel_map_dims

        pos_rel = pos_rel * 2 - 1

        return pos_rel

    def world_to_pix_crop(self, query_pos, crop_pos, crop_yaw, offset=(0, 0)):
        crop_yaw = crop_yaw + self.gpu_pi / 2
        batch_size = crop_pos.shape[0]

        # transform to crop pose
        rotation = torch.cat(
            [torch.cos(crop_yaw), -torch.sin(crop_yaw),
            torch.sin(crop_yaw),  torch.cos(crop_yaw)],
            dim=-1,
        ).view(batch_size, -1, 2, 2)

        crop_pos_px = self.world_to_pix(crop_pos)

        # correct for the fact that crop is only in front of ego agent
        shift = self.world_to_pix_crop_shift_tensor

        query_pos_px_map = self.world_to_pix(query_pos)

        query_pos_px = torch.transpose(rotation, -2, -1) @ \
            (query_pos_px_map - crop_pos_px).unsqueeze(-1)
        query_pos_px = query_pos_px.squeeze(-1) - shift

        # shift coordinate frame to top left corner of the crop
        pos_px_crop = query_pos_px + self.world_to_pix_crop_half_crop_dims

        return pos_px_crop

    def reset(self):
        """
        """
        pass


class STNRenderer(BaseRenderer):
    def __init__(self, args, map_offset, map_dims, viz=False):
        """
        """
        super().__init__(args, map_offset, map_dims, viz)

        # precompute some static quantities for efficiency
        self.vehicle_template = torch.ones(
            [1, 1, 22, 9],
            device=self.args.device,
            dtype=torch.float32
        )

        self.render_agent_bv_normalization = torch.tensor(
            [self.crop_dims[0], self.crop_dims[1]],
            device=self.args.device,
            dtype=torch.float32
        )

        self.render_agent_bv_vehicle_scale_h = torch.tensor(
            [self.crop_dims[0] / self.vehicle_template.size(2)],
            device=self.args.device,
            dtype=torch.float32
        )
        self.render_agent_bv_vehicle_scale_w = torch.tensor(
            [self.crop_dims[1] / self.vehicle_template.size(3)],
            device=self.args.device,
            dtype=torch.float32
        )

        self.render_agent_bv_vehicle_scale_transform = torch.tensor(
            [[self.render_agent_bv_vehicle_scale_w, 0, 0],
            [0, self.render_agent_bv_vehicle_scale_h, 0],
            [0, 0, 1]],
            device=self.args.device,
            dtype=torch.float32
        ).view(1, 1, 3, 3).expand(self.args.batch_size, -1, -1, -1)

    def render_agent_bv(
            self,
            grid,
            grid_pos,
            grid_orientation,
            position,
            orientation,
        ):
        """
        """
        vehicle = self.vehicle_template.expand(self.args.batch_size, -1, -1, -1)

        orientation = orientation + self.gpu_pi / 2

        pos_pix_bv = self.world_to_pix_crop(position, grid_pos, grid_orientation)

        # to centered relative coordinates for STN
        pos_rel_bv = pos_pix_bv / self.render_agent_bv_normalization
        pos_rel_bv = pos_rel_bv * 2 - 1
        pos_rel_bv = pos_rel_bv * -1

        scale_transform = self.render_agent_bv_vehicle_scale_transform

        grid_orientation = grid_orientation + self.gpu_pi / 2
        angle_delta = orientation - grid_orientation
        zeros = torch.zeros_like(angle_delta)
        ones = torch.ones_like(angle_delta)
        rotation_transform = torch.stack(
            [torch.cos(angle_delta), torch.sin(angle_delta), zeros,
            -torch.sin(angle_delta), torch.cos(angle_delta), zeros,
            zeros,                   zeros,                  ones],
            dim=-1
        ).view(self.args.batch_size, -1, 3, 3)

        translation_transform = torch.stack(
            [ones,  zeros, pos_rel_bv[..., 0:1],
             zeros, ones,  pos_rel_bv[..., 1:2],
             zeros, zeros, ones],
            dim=-1,
        ).view(self.args.batch_size, -1, 3, 3)

        scale_transform = scale_transform.expand(
            self.args.batch_size, translation_transform.size(1), -1, -1
        )

        affine_transform = scale_transform @ rotation_transform @ translation_transform
        affine_transform = affine_transform.view(-1, 3, 3)

        affine_grid = F.affine_grid(
            affine_transform[:, 0:2, :], # expects Nx2x3
            (self.args.batch_size * translation_transform.size(1), 1, grid.shape[2], grid.shape[3]),
            align_corners=True,
        )

        vehicle = vehicle.repeat(translation_transform.size(1), 1, 1, 1)
        vehicle_rendering = F.grid_sample(
            vehicle,
            affine_grid,
            align_corners=True,
        )

        vehicle_rendering = vehicle_rendering.view(
            self.args.batch_size,
            -1,
            vehicle_rendering.size(2),
            vehicle_rendering.size(3)
        )

        vehicle_rendering = torch.sum(vehicle_rendering, dim=1, keepdim=True)

        # pad the channel dimension and add to BEV grid
        grid = grid + F.pad(vehicle_rendering, (0, 0, 0, 0, 2, 1))

        return grid, vehicle_rendering

    def get_observations(self, hd_map, ego_state, adv_state):
        local_map = self.get_local_birdview(
            hd_map,
            ego_state["pos"],
            ego_state["yaw"],
        )

        pos = torch.cat([ego_state["pos"], adv_state["pos"]], dim=1)
        yaw = torch.cat([ego_state["yaw"], adv_state["yaw"]], dim=1)

        birdview, _ = self.render_agent_bv(
            local_map,
            ego_state["pos"],
            ego_state["yaw"],
            pos,
            yaw,
        )

        if self.args.gradient_clip > 0:
            birdview = _GradientScaling.apply(birdview, self.args.gradient_clip)

        light_hazard = torch.tensor(
            [0.],
            device=self.args.device, dtype=torch.float32
        ).view(1, 1).expand(birdview.shape[0], -1)

        observations = {
            "birdview": birdview,
            "light_hazard": light_hazard,
        }

        return observations, local_map


class CARLARenderer(BaseRenderer):
    """
        Renders sensor data via the CARLA engine.

        It accesses the proxy simulator's CARLA wrapper to set the state of a
        twin CARLA server to the proxy simulator's state and then retrieves the
        sensor data for that state for a given sensor config.
    """
    def __init__(self, args, map_offset, map_dims, viz=False):
        super().__init__(args, map_offset, map_dims, viz=viz)

        self.town = None

        self.actors = []
        self.sensors = []

        self.spectator_cam_buffer = deque([], maxlen=10)
        self.ego_cam_buffer = deque([], maxlen=10)
        self.egoleft_cam_buffer = deque([], maxlen=10)
        self.egoright_cam_buffer = deque([], maxlen=10)
        self.ego_lidar_buffer = deque([], maxlen=10)

    def _parse_image_cb(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = copy.deepcopy(array)
        array = np.reshape(array, (image.height, image.width, 4))
        return array

    def _parse_lidar_cb(self, lidar_data):
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        return points

    def get_observations(self, hd_map, ego_state, adv_state):
        """
        """

        local_map = self.get_local_birdview(
            hd_map,
            ego_state["pos"],
            ego_state["yaw"],
        )

        # guard for carla server having died in the background
        try:
            obs_dict = self._fetch_carla_data(ego_state, adv_state)
        except RuntimeError:
            # re-connect to resurrected carla server
            self.carla_wrapper._initialize_from_carla(town=self.town, port=self.args.port)
            # allow some time to re-establish the connection
            time.sleep(60)
            # re-attach to wrapper (so world objects etc. get updated)
            self.attach_carla_wrapper(self.carla_wrapper)
            # try fetching observations again
            obs_dict = self._fetch_carla_data(ego_state, adv_state)

        return obs_dict, local_map

    def _fetch_carla_data(self, ego_state, adv_state):
        """
        """
        self.set_carla_state(ego_state, adv_state)
        self.carla_wrapper.world.tick()

        counter = 0
        try:
            obs_dict = {
                'spectator': self._parse_image_cb(self.spectator_cam_buffer[-1]),
                'rgb': self._parse_image_cb(self.ego_cam_buffer[-1]),
                'rgb_left': self._parse_image_cb(self.egoleft_cam_buffer[-1]),
                'rgb_right': self._parse_image_cb(self.egoright_cam_buffer[-1]),
                'lidar': self._parse_lidar_cb(self.ego_lidar_buffer[-1]),
            }
        except IndexError:
            print("Encountered empty buffer, retrying.")
            counter += 1
            if counter >= 5:
                pass
            else:
                self.set_carla_state(ego_state, adv_state)
                time.sleep(1)
                self.carla_wrapper.world.tick()
                obs_dict = {
                    'spectator': self._parse_image_cb(self.spectator_cam_buffer[-1]),
                    'rgb': self._parse_image_cb(self.ego_cam_buffer[-1]),
                    'rgb_left': self._parse_image_cb(self.egoleft_cam_buffer[-1]),
                    'rgb_right': self._parse_image_cb(self.egoright_cam_buffer[-1]),
                    'lidar': self._parse_lidar_cb(self.ego_lidar_buffer[-1]),
                }

        return obs_dict

    def attach_carla_wrapper(self, carla_wrapper):
        """
        """
        self.carla_wrapper = carla_wrapper
        self.world = carla_wrapper.world
        self.bp_library = self.world.get_blueprint_library()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1/20.
        self.world.apply_settings(settings)

    def initialize_carla_state(self, ego_state, adv_state, town=None, diverse_actors=False):
        """
        """
        self.town = town
        ego_transform = state_dict_to_transform(ego_state)[0][0]

        actors = self.world.get_actors()
        traffic_lights = actors.filter('*traffic_light*')
        for traffic_light in traffic_lights:
            traffic_light.set_state(carla.TrafficLightState.Green)
            traffic_light.set_green_time(100000.0)

        # spawn ego
        ego_bp = self.bp_library.filter("vehicle.lincoln.mkz2017")[0]
        ego_bp.set_attribute('color', '255,0,0')
        ego_actor = self.world.try_spawn_actor(ego_bp, ego_transform)
        tries = 0
        while ego_actor is None and tries < 20:
            ego_transform.location.z += 0.05
            ego_actor = self.world.try_spawn_actor(ego_bp, ego_transform)
            tries += 1
            if tries == 20 and ego_actor is None:
                if hasattr(self.args, 'raise_failed_carla_spawn'):
                    if self.args.raise_failed_carla_spawn:
                        raise RuntimeError(f"WARNING: Could not spawn ego agent.")
                    else:
                        print(f"WARNING: Could not spawn ego agent.")
                else:
                    print(f"WARNING: Could not spawn ego agent.")

        self.actors.append(ego_actor)

        # set spectator to BEV a few meters ahead of the ego agent
        spectator_transform = ego_transform
        shift = spectator_transform.get_forward_vector() * 40
        spectator_transform.location.z += 50.
        spectator_transform.location.x += shift.x
        spectator_transform.location.y += shift.y
        spectator_transform.rotation.pitch -= 90.
        self.world.get_spectator().set_transform(ego_transform)

        # attach camera to spectator
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x",str(1024))
        camera_bp.set_attribute("image_size_y",str(1024))
        camera_bp.set_attribute("fov",str(105))
        camera_bp.set_attribute('lens_circle_multiplier', str(3.0))
        camera_bp.set_attribute('lens_circle_falloff', str(3.0))
        camera_bp.set_attribute('chromatic_aberration_intensity', str(0.5))
        camera_bp.set_attribute('chromatic_aberration_offset', str(0))
        camera_location = carla.Location(0,0,0)
        camera_rotation = carla.Rotation(0,0,0)
        camera_transform = carla.Transform(camera_location, camera_rotation)
        self.spectator_cam = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.world.get_spectator(),
            attachment_type=carla.AttachmentType.Rigid
        )
        self.spectator_cam.listen(lambda image: self.spectator_cam_buffer.append(image))
        self.sensors.append(self.spectator_cam)

        # rgb cam for transfuser
        camera_bp2 = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp2.set_attribute("image_size_x", str(400))
        camera_bp2.set_attribute("image_size_y", str(300))
        camera_bp2.set_attribute("fov", str(100))
        camera_bp2.set_attribute('lens_circle_multiplier', str(3.0))
        camera_bp2.set_attribute('lens_circle_falloff', str(3.0))
        camera_bp2.set_attribute('chromatic_aberration_intensity', str(0.5))
        camera_bp2.set_attribute('chromatic_aberration_offset', str(0))
        camera_location = carla.Location(1.3, 0, 2.3)
        camera_rotation = carla.Rotation(0, 0, 0)
        camera_transform = carla.Transform(camera_location, camera_rotation)
        self.ego_cam = self.world.spawn_actor(
            camera_bp2,
            camera_transform,
            attach_to=ego_actor,
            attachment_type=carla.AttachmentType.Rigid
        )
        self.ego_cam.listen(lambda image: self.ego_cam_buffer.append(image))
        self.sensors.append(self.ego_cam)

        # rgb left for trnasfuser
        camera_bp3 = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp3.set_attribute("image_size_x", str(400))
        camera_bp3.set_attribute("image_size_y", str(300))
        camera_bp3.set_attribute("fov", str(100))
        camera_bp3.set_attribute('lens_circle_multiplier', str(3.0))
        camera_bp3.set_attribute('lens_circle_falloff', str(3.0))
        camera_bp3.set_attribute('chromatic_aberration_intensity', str(0.5))
        camera_bp3.set_attribute('chromatic_aberration_offset', str(0))
        # camera_location = carla.Location(0,0,0)
        camera_location = carla.Location(1.3, 0, 2.3)
        camera_rotation = carla.Rotation(0, -60, 0)
        camera_transform = carla.Transform(camera_location, camera_rotation)
        self.egoleft_cam = self.world.spawn_actor(
            camera_bp3,
            camera_transform,
            attach_to=ego_actor,
            attachment_type=carla.AttachmentType.Rigid
        )
        self.egoleft_cam.listen(lambda image: self.egoleft_cam_buffer.append(image))
        self.sensors.append(self.egoleft_cam)

        # rgb right for transfuser
        camera_bp4 = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp4.set_attribute("image_size_x", str(400))
        camera_bp4.set_attribute("image_size_y", str(300))
        camera_bp4.set_attribute("fov", str(100))
        camera_bp4.set_attribute('lens_circle_multiplier', str(3.0))
        camera_bp4.set_attribute('lens_circle_falloff', str(3.0))
        camera_bp4.set_attribute('chromatic_aberration_intensity', str(0.5))
        camera_bp4.set_attribute('chromatic_aberration_offset', str(0))
        camera_location = carla.Location(1.3, 0, 2.3)
        camera_rotation = carla.Rotation(0, 60, 0)
        camera_transform = carla.Transform(camera_location, camera_rotation)
        self.egoright_cam = self.world.spawn_actor(
            camera_bp4,
            camera_transform,
            attach_to=ego_actor,
            attachment_type=carla.AttachmentType.Rigid
        )
        self.egoright_cam.listen(lambda image: self.egoright_cam_buffer.append(image))
        self.sensors.append(self.egoright_cam)

        # lidar for transfuser
        lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', str(85))
        lidar_bp.set_attribute('rotation_frequency', str(20))  # default: 10, change to 20 to generate 360 degree LiDAR point cloud
        lidar_bp.set_attribute('channels', str(64))
        lidar_bp.set_attribute('upper_fov', str(10))
        lidar_bp.set_attribute('lower_fov', str(-30))
        lidar_bp.set_attribute('points_per_second', str(2*600000))
        lidar_bp.set_attribute('atmosphere_attenuation_rate', str(0.004))
        lidar_bp.set_attribute('dropoff_general_rate', str(0.45))
        lidar_bp.set_attribute('dropoff_intensity_limit', str(0.8))
        lidar_bp.set_attribute('dropoff_zero_intensity', str(0.4))

        camera_location = carla.Location(1.3, 0, 2.5)
        camera_rotation = carla.Rotation(0, -90, 0)
        camera_transform = carla.Transform(camera_location, camera_rotation)
        self.ego_lidar = self.world.spawn_actor(
            lidar_bp,
            camera_transform,
            attach_to=ego_actor,
            attachment_type=carla.AttachmentType.Rigid
        )
        self.ego_lidar.listen(lambda image: self.ego_lidar_buffer.append(image))
        self.sensors.append(self.ego_lidar)

        # attach sensors
        adv_bp_blacklist = ["vehicle.carlamotors.carlacola"]
        if adv_state is not None:
            adv_transform = state_dict_to_transform(adv_state)[0]

            # spawn adversarials
            for agent_id in range(self.args.num_agents):
                # fetch blueprint
                if diverse_actors:
                    # fetch random car CAD model
                    blueprints = self.bp_library.filter('vehicle.*.*')
                    blueprints = [bp for bp in blueprints if not bp.id in adv_bp_blacklist]
                    adv_bp = random.choice(blueprints)

                    # set random color if possible
                    if adv_bp.has_attribute("color"):
                        color = random.choice(adv_bp.get_attribute('color').recommended_values)
                        adv_bp.set_attribute('color', color)
                else:
                    adv_bp = self.bp_library.filter("model3")[0]

                # try to spawn and move up in 5cm increments if spawning fails
                adv_actor = self.world.try_spawn_actor(adv_bp, adv_transform[agent_id])
                tries = 0
                while adv_actor is None and tries < 20:
                    adv_transform[agent_id].location.z += 0.05
                    adv_actor = self.world.try_spawn_actor(adv_bp, adv_transform[agent_id])
                    tries += 1
                    if tries == 20 and adv_actor is None:
                        print(f"WARNING: Could not spawn adv agent {agent_id}.")

                self.actors.append(adv_actor)

        for actor in self.actors:
            if actor:
                actor.set_simulate_physics(False)

        self.carla_wrapper.world.tick()
        self.carla_wrapper.world.tick()
        self.carla_wrapper.world.tick()

    def set_carla_state(self, ego_state, adv_state):
        """
        """
        ego_transform = state_dict_to_transform(ego_state)[0][0]
        # set ego state
        self.actors[0].set_transform(ego_transform)

        if adv_state is not None:
            adv_transform = state_dict_to_transform(adv_state)[0]

            for agent_id in range(self.args.num_agents):
                # if the agent has not been spawned we dont attempt to set
                # its transform
                if self.actors[agent_id+1] is None:
                    continue

                self.actors[agent_id+1].set_transform(adv_transform[agent_id])

    def reset(self):
        """
            Resets the CARLA state by destroying all actors and performing
            additional cleanup as necessary.
        """
        for sensor in self.sensors:
            if sensor is not None:
                sensor.destroy()

        for actor in self.actors:
            if actor is not None:
                actor.destroy()

        self.actors = []
        self.sensors = []

        self.spectator_cam_buffer = deque([], maxlen=10)
        self.ego_cam_buffer = deque([], maxlen=10)
        self.egoleft_cam_buffer = deque([], maxlen=10)
        self.egoright_cam_buffer = deque([], maxlen=10)
        self.ego_lidar_buffer = deque([], maxlen=10)
