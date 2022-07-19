import numpy as np


class EgoModel():
    def __init__(self, dt=1./4):
        self.dt = dt

        # Kinematic bicycle model. Numbers are the tuned parameters from World on Rails
        self.front_wb    = np.array(-0.090769015)
        self.rear_wb     = np.array(1.4178275)

        self.steer_gain  = np.array(0.36848336)
        self.brake_accel = np.array([-4.952399])
        self.throt_accel = np.array([[0.5633837]])

    def forward(self, locs, yaws, spds, acts): #Old torch version takes between 200 and 400 mu. New numpy version uses between 50 and 100 mu

        steer = acts[...,0:1]
        throt = acts[...,1:2]
        brake = acts[...,2:3].astype(np.uint8)

        if(brake):
            accel = self.brake_accel
        else:
            accel = self.throt_accel @ throt

        wheel = self.steer_gain * steer

        beta = np.arctan(self.rear_wb/(self.front_wb+self.rear_wb) * np.tan(wheel))

        next_locs = locs + spds * np.concatenate([np.cos(yaws+beta), np.sin(yaws+beta)],-1) * self.dt

        next_yaws = yaws + spds / self.rear_wb * np.sin(beta) * self.dt
        next_spds = spds + accel * self.dt
        next_spds = next_spds * (next_spds > 0)  # Fast ReLU

        return next_locs, next_yaws, next_spds
