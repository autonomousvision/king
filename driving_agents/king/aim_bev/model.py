import torch 
from torch import nn
from torchvision import models

import numpy as np


GPU_PI = torch.tensor(np.pi, device="cuda",)
class AimBev(nn.Module):
    """
    """
    def __init__(self, args, device, pred_len, batch_size=1, lat_pid_params=None, lon_pid_params=None, **kwargs):
        super(AimBev, self).__init__()
        self.pred_len = pred_len
        self.device = device
        self.input_size = 4 # we use 4 channel BEV
        self.args = args

        if not lat_pid_params: lat_pid_params = {"K_P": 1.25, "K_I": 0.75, "K_D": 0.3, "window_size": 4}
        if not lon_pid_params: lon_pid_params = {"K_P": 5.0, "K_I": 0.5, "K_D": 1.0, "window_size": 4}
        self.turn_controller = PIDController(**lat_pid_params, batch_size=batch_size)
        self.speed_controller = PIDController(**lon_pid_params, batch_size=batch_size)
        self.stop_slope = 10.
        self.decel_slope = 1.
        self.emergency_thresh = 0.6
        self.delta_slope = 0.5

        self.image_encoder = ImageCNN(512, self.input_size, use_linear=False).to(self.device)
        self.join = nn.Sequential(
                            nn.Linear(512, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 64),
                            nn.ReLU(inplace=True),
                        ).to(self.device)

        self.decoder = nn.GRUCell(input_size=2, hidden_size=65).to(self.device)
        self.relu_wp = nn.ReLU()
        
        self.output = nn.Linear(65, 2).to(self.device)

    def forward(self, feature_emb, target_point, light_hazard=None):
        feature_emb_new = torch.cat((feature_emb), 1)

        z = self.join(feature_emb_new)
        z = torch.cat((z, light_hazard), 1)

        output_wp = list()

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(self.device)

        # autoregressive generation of output waypoints
        for _ in range(self.pred_len):
            x_in = x + target_point
            z = self.decoder(x_in, z)
            dx = self.output(z)
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)

        return pred_wp


    def control_pid(self, waypoints, velocity):
        waypoints[:,:,1] = waypoints[:,:,1] * -1

        speed = velocity 

        desired_speed = torch.linalg.norm(
                    waypoints[:,0:1] - waypoints[:,1:2], dim=2) * 2.0

        stop_probability = torch.sigmoid(-self.stop_slope * (desired_speed - self.emergency_thresh))
        speed_violation = torch.relu(speed - desired_speed)
        decel_probability = torch.tanh(self.decel_slope * (speed_violation))

        if desired_speed < self.emergency_thresh:
            brake_probability = stop_probability
        else:
            brake_probability = decel_probability

        brake_flag = torch.gt(brake_probability, torch.ones_like(brake_probability) * 0.1)

        soft_delta = torch.tanh(self.delta_slope * torch.relu(desired_speed - speed)) / 4
        throttle = self.speed_controller.step(soft_delta)
        throttle = torch.tanh(torch.relu(throttle)) * 0.75
        throttle = throttle * ~brake_flag 

        aim = (waypoints[:,1] + waypoints[:,0]) / 2.0
        angle = torch.rad2deg(GPU_PI / 2 - torch.atan2(aim[:,1], aim[:,0])) / 90

        angle = angle.unsqueeze(-1) * ~brake_flag

        steer = self.turn_controller.step(angle)
        steer = torch.tanh(steer)

        waypoints[:,:,1] = waypoints[:,:,1] * -1

        return steer, throttle, brake_probability


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, window_size=20, batch_size=1):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D
        self.batch_size = batch_size
        self.window_size = window_size

        self._window = torch.zeros((batch_size, window_size), device='cuda')
        self._max = torch.zeros((batch_size, 1), device='cuda')
        self._min = torch.zeros((batch_size, 1), device='cuda')

    def step(self, error):
        self._window = torch.cat((self._window[:, 1:], error), dim=1)
        self._max = torch.max(self._max, abs(error))
        self._min = -abs(self._max)

        if self._window.shape[1] >= 2:
            integral = torch.mean(self._window, dim=1).unsqueeze(-1)
            derivative = (self._window[:,-1] - self._window[:,-2]).unsqueeze(-1)
        else:
            integral = torch.zeros((self.batch_size, 1), device='cuda')
            derivative = torch.zeros((self.batch_size, 1), device='cuda')

        return self._K_P * error + self._K_I * integral + self._K_D * derivative
    
    def reset(self):
        self._window = torch.zeros((self.batch_size, self.window_size), device='cuda')
        self._max = torch.zeros((self.batch_size, 1), device='cuda')
        self._min = torch.zeros((self.batch_size, 1), device='cuda')


class ImageCNN(nn.Module):
    """ Encoder network for image input list.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    """

    def __init__(self, c_dim, input_dim=3, use_linear=False, **kwargs):
        super().__init__()
        self.use_linear = use_linear
        
        self.features = models.mobilenet_v3_small(pretrained=True)
        self.features.classifier._modules['3'] = nn.Linear(1024, 512)
        self.features.features._modules['0']._modules['0'] = nn.Conv2d(input_dim, 16, kernel_size=(3,3), stride=(2, 2), padding=(1,1), bias=False)

        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, inputs):
        c = 0
        for x in inputs:
            net = self.features(x)
            c += self.fc(net)
        return c

    def load_state_dict(self, state_dict):
        errors = super().load_state_dict(state_dict, strict=False)
        print (errors)