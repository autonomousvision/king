import carla
import torch

import numpy as np


def state_vel_to_carla_vector3d(vel):
    vec = carla.Vector3D(vel[0].item(), vel[1].item(), 0.0)
    return vec


def state_pos_to_carla_location(pos):
    loc = carla.Location(pos[0].item(), pos[1].item(), 0.0)
    return loc
    

def state_yaw_to_carla_rotation(yaw):
    rot = carla.Rotation(pitch=0.0, yaw=yaw.item() / np.pi * 180, roll=0.0)
    return rot


def state_to_carla_transform(pos, yaw):
    loc = state_pos_to_carla_location(pos)
    rot = state_yaw_to_carla_rotation(yaw)
    transform = carla.Transform(loc, rot)

    return transform

def actions_list_to_tensor(actions, device):
    actions_tn = {}
    for key, value in enumerate(actions):
        actions_tn.update({
            key: torch.tensor(value).float().to(device)
        })
    return actions_tn