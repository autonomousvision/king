import os
import json
import time
import datetime
import pathlib
import math
import copy
import cv2
import carla
from PIL import Image
from matplotlib import cm

import torch
import numpy as np

from driving_agents.king.common import autonomous_agent
from driving_agents.king.transfuser.model import TransFuser
from driving_agents.king.transfuser.config import GlobalConfig
from driving_agents.king.transfuser.utils import scale_and_crop_image, lidar_to_histogram_features, transform_2d_points
from driving_agents.king.common.utils.planner import RoutePlanner, interpolate_trajectory


SAVE_PATH = os.environ.get('SAVE_PATH', None)


def get_entry_point():
    return 'TransFuserAgent'


class TransFuserAgent(autonomous_agent.AutonomousAgent):
    def setup(self, args, path_to_conf_file=None, device=None):
        self.device = device
        self.args = args
        self.lidar_processed = list()
        self.track = autonomous_agent.Track.SENSORS
        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        self.input_buffer = {'rgb': [], 'rgb_left': [], 'rgb_right': [], 'lidar': [], 'gps': [], 'thetas': []}

        self.config = GlobalConfig()
        self.net = TransFuser(self.config, device)
        self.net.load_state_dict(torch.load(os.path.join(path_to_conf_file, 'best_model.pth')))
        self.net.to(device)
        self.net.eval()

        self.save_path = None
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

            print (string)

            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)

            (self.save_path / 'rgb').mkdir(parents=True, exist_ok=False)
            (self.save_path / 'lidar_0').mkdir(parents=True, exist_ok=False)
            (self.save_path / 'lidar_1').mkdir(parents=True, exist_ok=False)
            (self.save_path / 'meta').mkdir(parents=True, exist_ok=False)

    def _init(self, world):
        self.world_map = world.carla_wrapper.map # carla: carla.Map("RouteMap", hd_map[1]['opendrive'])

        for _global_plan, _global_plan_world_coord in zip(self._global_plan_list, self._global_plan_world_coord_list):
            trajectory = [item[0].location for item in _global_plan_world_coord]
            self._dense_route, _ = interpolate_trajectory(self.world_map, trajectory)

            self._route_planner = RoutePlanner(4.0, 50.0)
            self._route_planner.set_route(_global_plan, True)

        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def _get_position_batched(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - torch.from_numpy(self._route_planner.mean).to(device=self.device)) * torch.from_numpy(self._route_planner.scale).to(device=self.device)
        return gps

    def sensors(self):
        pass

    def _carla_img_to_bgra_array(self, image):
        """Convert a CARLA raw image to a BGRA numpy array."""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        return array

    def _carla_img_to_rgb_array(self, image):
        """Convert a CARLA raw image to a RGB numpy array."""
        array = self._carla_img_to_bgra_array(image)
        # Convert BGRA to RGB.
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array

    def parse_lidar(self, data):
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        return points

    def tick(self, input_data):
        self.step += 1

        rgb = cv2.cvtColor(input_data['rgb'][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data['rgb_left'][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data['rgb_right'][:, :, :3], cv2.COLOR_BGR2RGB)

        gps = input_data['gps']
        speed = input_data['speed']
        compass = input_data['imu']
        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0

        lidar = input_data['lidar'][:, :3]

        result = {
                'rgb': rgb,
                'rgb_left': rgb_left,
                'rgb_right': rgb_right,
                'lidar': lidar,
                'gps': gps,
                'speed': speed,
                'compass': compass,
                }

        pos = self._get_position_batched(result)
        result['gps'] = pos

        target_point_list = []
        for ix in range(pos.shape[0]):
            pos_local = pos[ix].squeeze().detach().cpu().numpy()
            compass_local = compass[ix].squeeze().detach().cpu().numpy()

            next_wp, next_cmd = self._route_planner.run_step(pos_local)
            result['next_command'] = next_cmd.value

            # coordinates world to egocar
            theta = compass_local + np.pi/2
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
                ])

            local_command_point = np.array([next_wp[0]-pos_local[0], next_wp[1]-pos_local[1]])
            local_command_point = R.T.dot(local_command_point)
            target_point_list.append(torch.from_numpy(local_command_point))

        result['target_point'] = torch.stack(target_point_list, dim=0).to(self.device, dtype=torch.float32)
        return result


    @torch.no_grad()
    def run_step(self, input_data, world):
        if not self.initialized:
            self._init(world)

        tick_data = self.tick(input_data)

        gt_velocity = tick_data['speed'].squeeze(1).squeeze(1)
        target_point = tick_data['target_point'].squeeze(1)

        rgb = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb']), crop=self.config.input_resolution)).unsqueeze(0)
        self.input_buffer['rgb'] = [rgb.to(self.net.device, dtype=torch.float32)]

        if not self.config.ignore_sides:
            rgb_left = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_left']), crop=self.config.input_resolution)).unsqueeze(0)
            self.input_buffer['rgb_left'] = [rgb_left.to(self.net.device, dtype=torch.float32)]

            rgb_right = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_right']), crop=self.config.input_resolution)).unsqueeze(0)
            self.input_buffer['rgb_right'] = [rgb_right.to(self.net.device, dtype=torch.float32)]

        if not self.config.ignore_rear:
            rgb_rear = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_rear']), crop=self.config.input_resolution)).unsqueeze(0)
            self.input_buffer['rgb_rear'] = [rgb_rear.to(self.net.device, dtype=torch.float32)]

        self.input_buffer['lidar'] = [tick_data['lidar']]
        self.input_buffer['gps'] = [tick_data['gps']]
        self.input_buffer['thetas'] = [tick_data['compass']]

        # transform the lidar point clouds to local coordinate frame
        ego_theta = self.input_buffer['thetas'][-1][0][0].cpu()
        ego_x, ego_y = self.input_buffer['gps'][-1][0][0].cpu()

        #Only predict every second step because we only get a LiDAR every second frame.
        if(self.step  % 2 == 0 or self.step <= 4):
            for i, lidar_point_cloud in enumerate(self.input_buffer['lidar']):

                curr_theta = self.input_buffer['thetas'][i][0][0].cpu().item()
                curr_x, curr_y = self.input_buffer['gps'][i][0][0].cpu().numpy()

                lidar_point_cloud[:,1] *= -1 # inverts x, y

                lidar_transformed = transform_2d_points(lidar_point_cloud,
                        np.pi/2-curr_theta, -curr_x, -curr_y, np.pi/2-ego_theta, -ego_x, -ego_y)
                lidar_transformed = torch.from_numpy(lidar_to_histogram_features(lidar_transformed, crop=self.config.input_resolution)).unsqueeze(0)
                self.lidar_processed = list()
                self.lidar_processed.append(lidar_transformed.to(self.net.device, dtype=torch.float32))

            self.pred_wp = self.net(self.input_buffer['rgb'] + self.input_buffer['rgb_left'] + \
                               self.input_buffer['rgb_right'], \
                               self.lidar_processed, target_point, gt_velocity)

        steer, throttle, brake, metadata = self.net.control_pid(self.pred_wp, gt_velocity)
        self.pid_metadata = metadata

        # apply hard thresholding
        if brake < 0.05: brake = 0.0
        if throttle > brake: brake = 0.0

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        actions = {
            "steer":  torch.tensor(steer).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(self.net.device, dtype=torch.float32),
            "throttle": torch.tensor(throttle).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(self.net.device, dtype=torch.float32),
            "brake": torch.tensor(brake).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(self.net.device, dtype=torch.float32),
        }

        if SAVE_PATH is not None and self.step % 10 == 0:
            self.save(tick_data)

        return actions

    def save(self, tick_data):
        frame = self.step // 10

        Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))

        Image.fromarray(cm.gist_earth(self.lidar_processed[0].cpu().numpy()[0, 0], bytes=True)).save(self.save_path / 'lidar_0' / ('%04d.png' % frame))
        Image.fromarray(cm.gist_earth(self.lidar_processed[0].cpu().numpy()[0, 1], bytes=True)).save(self.save_path / 'lidar_1' / ('%04d.png' % frame))

        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

    def destroy(self):
        del self.net

    def reset(self):
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

