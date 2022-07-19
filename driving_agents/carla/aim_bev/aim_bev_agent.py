import os
import json
import carla

from collections import deque
import torch
import numpy as np

from driving_agents.carla.expert.data_agent import DataAgent
from driving_agents.king.aim_bev.model import AimBev


def get_entry_point():
    return 'AimBEVAgent'


class AimBEVAgent(DataAgent):
    def setup(self, path_to_conf_file, route_index=None):
        self.route_index = route_index
        # first args than super setup is important!
        args_file = open(os.path.join(path_to_conf_file, 'args.txt'), 'r')
        self.args = json.load(args_file)
        args_file.close()
        super().setup(path_to_conf_file, route_index, data_generation=False)

        self.input_buffer = {'seg_bev': deque(maxlen=self.args['seq_len'])}

        lat_pid_params = {"K_P": 0.9, "K_I": 0.75, "K_D": 0.3, "window_size": 20}
        lon_pid_params = {"K_P": 5.0, "K_I": 0.5, "K_D": 1.0, "window_size": 20}
        self.net = AimBev(
            self.args, 'cuda', self.args['pred_len'],
            lat_pid_params=lat_pid_params, lon_pid_params=lon_pid_params
        )

        self.net.load_state_dict(torch.load(os.path.join(path_to_conf_file, 'model.pth')))
        self.net.cuda()
        self.net.eval()

    def _init(self, hd_map):
        super()._init(hd_map)
        self.control = carla.VehicleControl()
        self.control.steer = 0.0
        self.control.throttle = 0.0
        self.control.brake = 1.0

    def tick(self, input_data):
        result = super().tick(input_data)

        pos = self._get_position(result['gps'])
        command_route = self._command_planner.run_step(pos)
        next_wp, _  = command_route[1] if len(command_route) > 1 else command_route[0]

        theta = result['compass'] + np.pi/2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
            ])

        local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result['target_point'] = tuple(local_command_point)

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        self.step += 1
        if not self.initialized:
            if ('hd_map' in input_data.keys()):
                self._init(input_data['hd_map'])
            else:
                self.control = carla.VehicleControl()
                self.control.steer = 0.0
                self.control.throttle = 0.0
                self.control.brake = 1.0
                return self.control

        _ = super()._get_brake(stop_sign_hazard=0, walker_hazard=0, vehicle_hazard=0)
        tick_data = self.tick(input_data)
        self.control = self._get_control(tick_data)

        if((self.step % self.save_freq == 0) and (self.save_path is not None)):
            self.save_sensors(tick_data)
            self.shuffle_weather()

        return self.control

    def _get_control(self, input_data):
        if len(self.input_buffer['seg_bev']) < self.args['seq_len'] or self.step < 60:
            seg_bev = input_data['topdown'].squeeze()
            self.input_buffer['seg_bev'].append(seg_bev.to('cuda', dtype=torch.float32))
            should_brake = self._get_safety_box()

            control = carla.VehicleControl()
            control.steer = 0.0
            if should_brake:
                control.throttle = 0.0
                control.brake = 1.0
            else:
                control.throttle = 0.5
                control.brake = 0.0
            return control

        gt_velocity = torch.FloatTensor([input_data['speed']]).to('cuda', dtype=torch.float32).unsqueeze(0)
        light_hazard = torch.FloatTensor([self.traffic_light_hazard]).to('cuda', dtype=torch.float32).unsqueeze(0)

        input_data['target_point'] = [torch.FloatTensor([input_data['target_point'][0]]),
                                            torch.FloatTensor([input_data['target_point'][1]])]
        target_point = torch.stack(input_data['target_point'], dim=1).to('cuda', dtype=torch.float32)

        encoding = []
        seg_bev = input_data['topdown'].squeeze().unsqueeze(0)

        self.input_buffer['seg_bev'].append(seg_bev.to('cuda', dtype=torch.float32))
        encoding.append(self.net.image_encoder(list(self.input_buffer['seg_bev'])))

        pred_wp = self.net(encoding, target_point, light_hazard=light_hazard)

        steer, throttle, brake = self.net.control_pid(pred_wp, gt_velocity)

        if brake < 0.05: brake = 0.0
        if throttle > brake: brake = 0.0

        if brake:
            steer *= self.steer_damping

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        return control

    def destroy(self):
        super().destroy()
        del self.net
        del self.input_buffer
        torch.cuda.empty_cache()