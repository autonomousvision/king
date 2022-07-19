import os
import json
import numpy as np
from collections import deque

import torch

from driving_agents.king.aim_bev import datagen_bev_renderer
from driving_agents.king.expert.expert_agent import AutoPilot
from driving_agents.king.aim_bev.model import AimBev


def get_entry_point():
    return 'AimBEVAgent'

class AimBEVAgent(AutoPilot):
    def setup(self, args, path_to_conf_file=None, device=None):
        super().setup(args, path_to_conf_file, device)
        self.args = args

        args_file = open(os.path.join(path_to_conf_file, 'args.txt'), 'r')
        self.args_map_agent = json.load(args_file)
        args_file.close()

        self.net = AimBev(args, 'cuda', self.args_map_agent['pred_len'], batch_size=self.args.batch_size)

        self.net.load_state_dict(torch.load(os.path.join(path_to_conf_file, 'model.pth')))
        self.net.cuda()
        self.net.eval()


    def _init(self, world):
        super()._init(world)

        self.vehicle_template = torch.ones(1, 1, 22, 9)

        self.global_map = world.map
        world_offset = world.map_offset 
        self.map_dims = self.global_map.shape[2:4]

        self.renderer = datagen_bev_renderer.DatagenBEVRenderer(self.args, world_offset, self.map_dims, data_generation=True)

    def reset(self):
        self.net.speed_controller.reset()
        self.net.turn_controller.reset()
        super().reset()

    def sensors(self):
        pass

    def tick(self, input_data):

        gps = input_data['gps']
        speed = input_data['speed'] 
        compass = input_data['imu']

        pos = self._get_position_batched(gps)
        target_point_list = []
        for ix in range(pos.shape[0]):
            pos_local = pos[ix].squeeze().detach().cpu().numpy()
            compass_local = compass[ix].squeeze().detach().cpu().numpy()

            next_wp, _ = self._command_planners[ix].run_step(pos_local)

            # coordinates world to egocar
            theta = compass_local + np.pi/2
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
                ])

            local_command_point = np.array([next_wp[0]-pos_local[0], next_wp[1]-pos_local[1]])
            local_command_point = R.T.dot(local_command_point)
            target_point_list.append(torch.from_numpy(local_command_point))

        input_data['target_point'] = torch.stack(target_point_list, dim=0).to(self.device, dtype=torch.float32)

        return input_data

    def run_step(self, input_data, world):
        self.step += 1
        with torch.no_grad():
            if not self.initialized:
                self._init(world)
                for ix in range(len(input_data['gps'])):
                    self.steer_buffer.append(deque(maxlen=self.steer_buffer_size))
            tick_data = self.tick(input_data)

        return self._get_control(tick_data)

    def _get_control(self, input_data, steer=None, throttle=None,
                        vehicle_hazard=None, light_hazard=None, walker_hazard=None, stop_sign_hazard=None):
        """
        """
        self._vehicle_state = self._world.get_ego_state()

        seg_bev = input_data['birdview']

        gt_velocity = input_data['speed'].squeeze(1)
        light_hazard = input_data['light_hazard']
        target_point = input_data['target_point']

        encoding = self.net.image_encoder([seg_bev.to(self.args.device)])
        pred_wp = self.net([encoding], target_point, light_hazard=light_hazard)
        steer, throttle, brake = self.net.control_pid(pred_wp, gt_velocity)

        actions = {
            "steer":  steer.unsqueeze(dim=1),
            "throttle": throttle.unsqueeze(dim=1),
            "brake": brake.unsqueeze(dim=1),
        }

        return actions

    def _get_position_batched(self, gps):
        gps = (gps - torch.from_numpy(self._command_planner.mean).to(device=self.args_map_agent['device'])) * torch.from_numpy(self._command_planner.scale).to(device=self.args_map_agent['device'])
        return gps

    def destroy(self):
        pass
