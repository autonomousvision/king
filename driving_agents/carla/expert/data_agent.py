from copy import deepcopy
import cv2
import carla
from PIL import Image
import os
import argparse

import torch
import numpy as np
import pygame
import json

from driving_agents.king.aim_bev.datagen_bev_renderer import DatagenBEVRenderer
from driving_agents.king.common.utils.map_utils import MapImage
from driving_agents.carla.expert.expert_agent import AutoPilot

PIXELS_PER_METER = 5
RENDER_PERSPECTIVE = int(os.environ.get('RENDER_PERSPECTIVE'))
SHUFFLE_WEATHER = int(os.environ.get('SHUFFLE_WEATHER'))

WEATHERS = {
		'Clear': carla.WeatherParameters.ClearNoon,
		'Cloudy': carla.WeatherParameters.CloudySunset,
		'Wet': carla.WeatherParameters.WetSunset,
		'MidRain': carla.WeatherParameters.MidRainSunset,
		'WetCloudy': carla.WeatherParameters.WetCloudySunset,
		'HardRain': carla.WeatherParameters.HardRainNoon,
		'SoftRain': carla.WeatherParameters.SoftRainSunset,
}

azimuths = [45.0 * i for i in range(8)]

daytimes = {
	'Night': -80.0,
	'Twilight': 0.0,
	'Dawn': 5.0,
	'Sunset': 15.0,
	'Morning': 35.0,
	'Noon': 75.0,
}

WEATHERS_IDS = list(WEATHERS)


def get_entry_point():
    return 'DataAgent'


class DataAgent(AutoPilot):
    def setup(self, path_to_conf_file, route_index=None, data_generation=True):
        super().setup(path_to_conf_file, route_index)
        self.data_generation = data_generation
        self.cam_config = {
            'width': 640,
            'height': 256,
            'fov': 60
        }

        if self.save_path is not None:
            (self.save_path / 'topdown').mkdir()
            (self.save_path / 'lidar').mkdir()
            (self.save_path / 'rgb').mkdir()
            (self.save_path / 'label_raw').mkdir()

            if 'args' not in dir(self):
                self.args = {'device': 'cuda'}

        self.channels = 4 # without traffic lights
        self.channel_offset = -3 # if we don't use traffic lights in the BEV we need to move all channels behind that

    def _init(self, hd_map):
        super()._init(hd_map)
        self._sensors = self.sensor_interface._sensors_objects

        self.vehicle_template = torch.ones(1, 1, 22, 9, device=self.args['device'])
        self.walker_template = torch.ones(1, 1, 10, 7, device=self.args['device'])
        self.traffic_light_template = torch.ones(1, 1, 4, 4, device=self.args['device'])

        # create map for renderer
        map_image = MapImage(self._world, self.world_map, PIXELS_PER_METER)
        make_image = lambda x: np.swapaxes(pygame.surfarray.array3d(x), 0, 1).mean(axis=-1)
        road = make_image(map_image.map_surface)
        lane = make_image(map_image.lane_surface)

        self.global_map = np.zeros((1, self.channels,) + road.shape)
        self.global_map[:, 0, ...] = road / 255.
        self.global_map[:, 1, ...] = lane / 255.

        self.global_map = torch.tensor(self.global_map, device=self.args['device'], dtype=torch.float32)
        world_offset = torch.tensor(map_image._world_offset, device=self.args['device'], dtype=torch.float32)
        self.map_dims = self.global_map.shape[2:4]

        self.renderer = DatagenBEVRenderer(argparse.Namespace(**self.args), world_offset, self.map_dims, data_generation=self.data_generation)

    def sensors(self):
        result = super().sensors()
        if self.save_path is not None:
            result += [
                    {
                        'type': 'sensor.camera.rgb',
                        'x': 1.3, 'y': 0.0, 'z':2.3,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                        'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                        'id': 'rgb_front'
                    },
                    {
                        'type': 'sensor.camera.rgb',
                        'x': 1.3, 'y': 0.0, 'z':2.3,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                        'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                        'id': 'rgb_left'
                    },
                    {
                        'type': 'sensor.camera.rgb',
                        'x': 1.3, 'y': 0.0, 'z':2.3,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                        'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                        'id': 'rgb_right'
                    },
                    {
                        'type': 'sensor.lidar.ray_cast',
                        'x': 1.3, 'y': 0.0, 'z': 2.5,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                        'rotation_frequency': 20,
                        'points_per_second': 1200000,
                        'id': 'lidar'
                    }
                    ]
        return result

    def tick(self, input_data):
        result = super().tick(input_data)
        result['topdown'] = self.render_BEV()

        if self.save_path is not None:
            rgb = []
            for pos in ['left', 'front', 'right']:
                rgb_cam = 'rgb_' + pos
                rgb.append(cv2.cvtColor(input_data[rgb_cam][1][:, :, :3], cv2.COLOR_BGR2RGB))
            rgb = np.concatenate(rgb, axis=1)
            lidar = input_data['lidar']
            cars = self.get_bev_cars(lidar=lidar)

            result.update({'lidar': lidar,
                            'rgb': rgb,
                            'cars': cars})

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        control = super().run_step(input_data, timestamp)

        if self.step % self.save_freq == 0:
            if self.save_path is not None:
                tick_data = self.tick(input_data)
                self.save_sensors(tick_data)
            if SHUFFLE_WEATHER:
                self.shuffle_weather()

        return control

    def save_sensors(self, tick_data):
        frame = self.step // self.save_freq

        rgb = Image.fromarray(tick_data['rgb'])
        rgb.save(self.save_path / 'rgb' / ('%04d.png' % frame))

        np.save(self.save_path / 'lidar' / ('%04d.npy' % frame), tick_data['lidar'], allow_pickle=True)
        np.save(self.save_path / 'topdown' / ('%04d.npy' % frame), tick_data['topdown'].cpu().numpy(), allow_pickle=True)
        self.save_labels(self.save_path / 'label_raw' / ('%04d.json' % frame), tick_data['cars'])

    def save_labels(self, filename, result):
        with open(filename, 'w') as f:
            json.dump(result, f, indent=4)
        return

    def save_points(self, filename, points):
        points_to_save = deepcopy(points[1])
        points_to_save[:, 1] = -points_to_save[:, 1]
        np.save(filename, points_to_save)
        return

    def destroy(self):
        del self.global_map
        del self.vehicle_template
        del self.walker_template
        del self.traffic_light_template
        del self.map_dims
        torch.cuda.empty_cache()

    def get_bev_cars(self, lidar=None):
        results = []
        ego_location = self._vehicle.get_transform().location
        ego_rotation = self._vehicle.get_transform().rotation
        ego_matrix = np.array(self._vehicle.get_transform().get_matrix())

        ego_extent = self._vehicle.bounding_box.extent
        ego_pos =  np.array([ego_location.x, ego_location.y, ego_location.z], dtype=np.float32)
        ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z]) * 2.
        ego_yaw =  ego_rotation.yaw/180*np.pi

        # also add ego box for visulization
        relative_yaw = 0
        relative_pos = self.get_relative_transform(ego_matrix, ego_matrix)

        # add vehicle velocity and brake flag
        ego_transform = self._vehicle.get_transform()
        ego_control   = self._vehicle.get_control()
        ego_velocity  = self._vehicle.get_velocity()
        ego_speed = self._get_forward_speed(transform=ego_transform, velocity=ego_velocity) # In m/s
        ego_brake = ego_control.brake

        result = {"class": "Car",
                  "extent": [ego_dx[2], ego_dx[0], ego_dx[1] ],
                  "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                  "yaw": relative_yaw,
                  "num_points": -1,
                  "distance": -1,
                  "speed": ego_speed,
                  "brake": ego_brake,
                  "id": int(self._vehicle.id),
                  'ego_matrix': self._vehicle.get_transform().get_matrix()
                }
        results.append(result)

        self._actors = self._world.get_actors()
        vehicles = self._actors.filter('*vehicle*')
        for vehicle in vehicles:
            if (vehicle.get_location().distance(self._vehicle.get_location()) < 50):
                if (vehicle.id != self._vehicle.id):
                    vehicle_location = vehicle.get_transform().location
                    vehicle_rotation = vehicle.get_transform().rotation
                    vehicle_matrix = np.array(vehicle.get_transform().get_matrix())
                    vehicle_id = vehicle.id

                    vehicle_extent = vehicle.bounding_box.extent
                    pos =  np.array([vehicle_location.x, vehicle_location.y, vehicle_location.z], dtype=np.float32)
                    dx = np.array([vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]) * 2.
                    yaw =  vehicle_rotation.yaw/180*np.pi

                    relative_yaw = yaw - ego_yaw
                    relative_pos = self.get_relative_transform(ego_matrix, vehicle_matrix)

                    vehicle_transform = vehicle.get_transform()
                    vehicle_control   = vehicle.get_control()
                    vehicle_velocity  = vehicle.get_velocity()
                    vehicle_speed = self._get_forward_speed(transform=vehicle_transform, velocity=vehicle_velocity) # In m/s
                    vehicle_brake = vehicle_control.brake

                    # filter bbox that didn't contains points of contains less points
                    if not lidar is None:
                        num_in_bbox_points = self.get_points_in_bbox(ego_matrix, vehicle_matrix, dx, lidar)
                    else:
                        num_in_bbox_points = -1

                    distance = np.linalg.norm(relative_pos)

                    result = {
                        "class": "Car",
                        "extent": [dx[2], dx[0], dx[1]],
                        "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                        "yaw": relative_yaw,
                        "num_points": int(num_in_bbox_points),
                        "distance": distance,
                        "speed": vehicle_speed,
                        "brake": vehicle_brake,
                        "id": int(vehicle_id),
                        "ego_matrix": vehicle.get_transform().get_matrix()
                    }
                    results.append(result)

        return results

    def get_points_in_bbox(self, ego_matrix, vehicle_matrix, dx, lidar):
        # inverse transform lidar to
        Tr_lidar_2_ego = self.get_lidar_to_vehicle_transform()

        # construct transform from lidar to vehicle
        Tr_lidar_2_vehicle = np.linalg.inv(vehicle_matrix) @ ego_matrix @ Tr_lidar_2_ego

        # transform lidar to vehicle coordinate
        lidar_vehicle = Tr_lidar_2_vehicle[:3, :3] @ lidar[1][:, :3].T + Tr_lidar_2_vehicle[:3, 3:]

        # check points in bbox
        x, y, z = dx / 2.
        x, y = y, x
        num_points = ((lidar_vehicle[0] < x) & (lidar_vehicle[0] > -x) &
                      (lidar_vehicle[1] < y) & (lidar_vehicle[1] > -y) &
                      (lidar_vehicle[2] < z) & (lidar_vehicle[2] > -z)).sum()
        return num_points

    def get_relative_transform(self, ego_matrix, vehicle_matrix):
        """
        return the relative transform from ego_pose to vehicle pose
        """
        relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3]
        rot = ego_matrix[:3, :3].T
        relative_pos = rot @ relative_pos

        # transform to right handed system
        relative_pos[1] = - relative_pos[1]

        # transform relative pos to virtual lidar system
        rot = np.eye(3)
        trans = - np.array([1.3, 0.0, 2.5])
        relative_pos = rot @ relative_pos + trans

        return relative_pos

    def get_lidar_to_vehicle_transform(self):
        # yaw = -90
        rot = np.array([[0, 1, 0],
                        [-1, 0, 0],
                        [0, 0, 1]], dtype=np.float32)
        T = np.eye(4)

        T[0, 3] = 1.3
        T[1, 3] = 0.0
        T[2, 3] = 2.5
        T[:3, :3] = rot
        return T


    def get_vehicle_to_lidar_transform(self):
        return np.linalg.inv(self.get_lidar_to_vehicle_transform())

    def get_image_to_vehicle_transform(self):
        # yaw = 0.0 as rot is Identity
        T = np.eye(4)
        T[0, 3] = 1.3
        T[1, 3] = 0.0
        T[2, 3] = 2.3

        # rot is from vehicle to image
        rot = np.array([[0, -1, 0],
                        [0, 0, -1],
                        [1, 0, 0]], dtype=np.float32)

        # so we need a transpose here
        T[:3, :3] = rot.T
        return T

    def get_vehicle_to_image_transform(self):
        return np.linalg.inv(self.get_image_to_vehicle_transform())

    def get_lidar_to_image_transform(self):
        Tr_lidar_to_vehicle = self.get_lidar_to_vehicle_transform()
        Tr_image_to_vehicle = self.get_image_to_vehicle_transform()
        T_lidar_to_image = np.linalg.inv(Tr_image_to_vehicle) @ Tr_lidar_to_vehicle
        return T_lidar_to_image

    def render_BEV(self):
        semantic_grid = self.global_map

        vehicle_position = self._vehicle.get_location()
        ego_pos_list =  [self._vehicle.get_transform().location.x, self._vehicle.get_transform().location.y]
        ego_yaw_list =  [self._vehicle.get_transform().rotation.yaw/180*np.pi]

        # fetch local birdview per agent
        ego_pos =  torch.tensor([self._vehicle.get_transform().location.x, self._vehicle.get_transform().location.y], device=self.args['device'], dtype=torch.float32)
        ego_yaw =  torch.tensor([self._vehicle.get_transform().rotation.yaw/180*np.pi], device=self.args['device'], dtype=torch.float32)
        birdview = self.renderer.get_local_birdview(
            semantic_grid,
            ego_pos,
            ego_yaw
        )

        self._actors = self._world.get_actors()
        vehicles = self._actors.filter('*vehicle*')
        for vehicle in vehicles:
            if (vehicle.get_location().distance(self._vehicle.get_location()) < self.detection_radius):
                if (vehicle.id != self._vehicle.id):
                    pos =  torch.tensor([vehicle.get_transform().location.x, vehicle.get_transform().location.y], device=self.args['device'], dtype=torch.float32)
                    yaw =  torch.tensor([vehicle.get_transform().rotation.yaw/180*np.pi], device=self.args['device'], dtype=torch.float32)
                    veh_x_extent = int(max(vehicle.bounding_box.extent.x*2, 1) * PIXELS_PER_METER)
                    veh_y_extent = int(max(vehicle.bounding_box.extent.y*2, 1) * PIXELS_PER_METER)

                    self.vehicle_template = torch.ones(1, 1, veh_x_extent, veh_y_extent, device=self.args['device'])
                    self.renderer.render_agent_bv(
                        birdview,
                        ego_pos,
                        ego_yaw,
                        self.vehicle_template,
                        pos,
                        yaw,
                        channel=5+self.channel_offset
                    )

        ego_pos_batched = []
        ego_yaw_batched = []
        pos_batched = []
        yaw_batched = []
        template_batched = []
        channel_batched = []

        # -----------------------------------------------------------
        # Pedestrian rendering
        # -----------------------------------------------------------

        walkers = self._actors.filter('*walker*')
        for walker in walkers:
            ego_pos_batched.append(ego_pos_list)
            ego_yaw_batched.append(ego_yaw_list)
            pos_batched.append([walker.get_transform().location.x, walker.get_transform().location.y])
            yaw_batched.append([walker.get_transform().rotation.yaw/180*np.pi])
            channel_batched.append(6+self.channel_offset)
            template_batched.append(np.ones([20, 7]))

        if len(ego_pos_batched)>0:
            ego_pos_batched_torch = torch.tensor(ego_pos_batched, device=self.args['device'], dtype=torch.float32).unsqueeze(1)
            ego_yaw_batched_torch = torch.tensor(ego_yaw_batched, device=self.args['device'], dtype=torch.float32).unsqueeze(1)
            pos_batched_torch = torch.tensor(pos_batched, device=self.args['device'], dtype=torch.float32).unsqueeze(1)
            yaw_batched_torch = torch.tensor(yaw_batched, device=self.args['device'], dtype=torch.float32).unsqueeze(1)
            template_batched_torch = torch.tensor(template_batched, device=self.args['device'], dtype=torch.float32).unsqueeze(1)
            channel_batched_torch = torch.tensor(channel_batched, device=self.args['device'], dtype=torch.float32)

            self.renderer.render_agent_bv_batched(
                birdview,
                ego_pos_batched_torch,
                ego_yaw_batched_torch,
                template_batched_torch,
                pos_batched_torch,
                yaw_batched_torch,
                channel=channel_batched_torch,
            )

        return birdview