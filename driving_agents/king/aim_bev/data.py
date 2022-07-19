import os
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset

from driving_agents.king.aim_bev.training_utils import augment_all, get_crop_offsets, transform_2d_points

# Global Flags
PIXEL_SHIFT = 14
PIXELS_AHEAD_VEHICLE = 96 + PIXEL_SHIFT


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--dataset_load_path', type=str)
    parser.add_argument('--dataset_save_path', type=str)
    parser.add_argument('--do_augmentation', type=int, default=1)
    parser.add_argument('--aug_max_rotation', type=float, default=20, help='Max rotation angle [degree] for augmentation. 0.0 equals to no agmentation.')
    parser.add_argument('--seq_len', type=int, default=1, help='Input sequence length (factor of 10).')
    parser.add_argument('--pred_len', type=int, default=4, help='Number of timesteps to predict.')

    return parser


class CARLA_Data(Dataset):
    """
    Dataset class for topdown maps and vehicle control in CARLA.
    """
    def __init__(self, args, root, aug_max_rotation=0.0, create_webdataset=False):

        self.args = args

        self.seq_len = args.seq_len
        self.pred_len = args.pred_len

        self.create_webdataset = create_webdataset

        self.target_im_size = args.target_im_size # (192, 192)

        self.aug_max_rotation = aug_max_rotation

        self.bev = []
        self.x = []
        self.y = []
        self.brake_seq = []
        self.x_old = []
        self.y_old = []
        self.x_command = []
        self.y_command = []
        self.command = []
        self.theta = []
        self.theta_old = []
        self.speed = []
        self.steer = []
        self.throttle = []
        self.brake = []
        self.light = []

        # get offsets for bigger crop in case we augment the BEVs
        self.offsets = get_crop_offsets(self.aug_max_rotation, self.target_im_size)
        for root_ix, sub_root in enumerate(root):
            sub_root_k = sub_root.replace('kchitta31', 'krenz73')
            Path(sub_root_k).mkdir(parents=True, exist_ok=True)
            preload_file = os.path.join(sub_root_k, 'pl_'+str(self.seq_len)+'_'+str(self.pred_len)+'bev_planner.npy')

            # dump to npy if no preload
            if not os.path.exists(preload_file):
                preload_bev = []
                preload_x = []
                preload_y = []
                preload_x_old = []
                preload_y_old = []
                preload_brake_seq = []
                preload_x_command = []
                preload_y_command = []
                preload_command = []
                preload_theta = []
                preload_theta_old = []
                preload_speed = []
                preload_steer = []
                preload_throttle = []
                preload_brake = []
                preload_light = []

                # list sub-directories in root
                root_files = os.listdir(sub_root)

                routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root,folder))]

                for route in routes:
                    route_dir = os.path.join(sub_root, route)
                    file_list = os.listdir(route_dir+"/topdown/")
                    file_list.sort()
                    measurement_file_list = os.listdir(route_dir+"/measurements/")
                    measurement_file_list.sort()

                    if len(file_list) == 0 or len(measurement_file_list) == 0:
                        print(f'Skip: {route_dir}')
                        continue

                    _factor = -(-len(measurement_file_list) // len(file_list)) #ceiling division

                    _recording_freq_mult = self.args.king_data_fps // 2

                    _usable_measurements = len(measurement_file_list) - int(self.pred_len * _recording_freq_mult)  - 1

                    _usable_images = _usable_measurements // _factor

                    num_seq = _usable_images // self.seq_len
                    for seq in range(1,num_seq-1):
                        bev = []
                        xs = []
                        ys = []
                        thetas = []
                        speeds = []
                        steer = []
                        throttle = []
                        brake = []
                        brake_seq = []
                        light = []

                        for i in range(self.seq_len):
                            # segmentation images
                            filename_topdown = file_list[seq*self.seq_len+i]
                            filename = file_list[seq*self.seq_len+i].split(".")[0].split("_")[1]
                            bev.append(route_dir+"/topdown/"+filename_topdown)

                            # position
                            with open(route_dir + f"/measurements/{filename}.json", "r") as read_file:
                                data = json.load(read_file)
                            xs.append(data['x'])
                            ys.append(data['y'])
                            brake_seq.append(data['brake'])
                            thetas.append(data['theta'])
                            speeds.append(data['speed'])
                            steer.append(data['steer'])
                            throttle.append(data['throttle'])
                            brake.append(data['brake'])
                            light.append(data['light_hazard'])


                        filename_prev = file_list[seq*self.seq_len-_recording_freq_mult].split(".")[0].split("_")[1]
                        with open(route_dir + f"/measurements/{filename_prev}.json", "r") as read_file_prev:
                            data_prev = json.load(read_file_prev)

                        preload_command.append([data_prev['command'], data['command']])

                        # get control value of final frame in sequence
                        preload_x_command.append(data['x_command'])
                        preload_y_command.append(data['y_command'])
                        preload_x.append([wp[0] for wp in data['waypoints']])
                        preload_y.append([wp[1] for wp in data['waypoints']])
                        preload_theta.append([wp[2] for wp in data['waypoints']])

                        # read files sequentially (future frames)
                        curr_file_index = int(filename)
                        for i in range(self.seq_len, self.seq_len + self.pred_len):

                            with open(route_dir + f"/measurements/{str(curr_file_index+i * _recording_freq_mult).zfill(4)}.json", "r") as read_file:
                                data = json.load(read_file)
                            xs.append(data['x'])
                            ys.append(data['y'])
                            brake_seq.append(data['brake'])

                            # fix for theta=nan in some measurements
                            if np.isnan(data['theta']):
                                thetas.append(0)
                            else:
                                thetas.append(data['theta'])

                        preload_bev.append(bev)

                        preload_x_old.append(xs)
                        preload_y_old.append(ys)
                        preload_brake_seq.append(brake_seq)
                        preload_theta_old.append(thetas)
                        preload_speed.append(speeds)
                        preload_steer.append(steer)
                        preload_throttle.append(throttle)
                        preload_brake.append(brake)
                        preload_light.append(light)

                # dump to npy
                preload_dict = {}
                preload_dict['bev'] = preload_bev
                preload_dict['x'] = preload_x_old
                preload_dict['y'] = preload_y_old
                preload_dict['brake_seq'] = preload_brake_seq
                preload_dict['x_old'] = preload_x_old
                preload_dict['y_old'] = preload_y_old
                preload_dict['x_command'] = preload_x_command
                preload_dict['y_command'] = preload_y_command
                preload_dict['command'] = preload_command
                preload_dict['theta'] = preload_theta_old
                preload_dict['theta_old'] = preload_theta_old
                preload_dict['speed'] = preload_speed
                preload_dict['steer'] = preload_steer
                preload_dict['throttle'] = preload_throttle
                preload_dict['brake'] = preload_brake
                preload_dict['light'] = preload_light
                np.save(preload_file, preload_dict)

            # load from npy if available
            preload_dict = np.load(preload_file, allow_pickle=True)
            self.bev += preload_dict.item()['bev']
            self.x += preload_dict.item()['x']
            self.y += preload_dict.item()['y']
            self.brake_seq += preload_dict.item()['brake_seq']
            self.x_old += preload_dict.item()['x_old']
            self.y_old += preload_dict.item()['y_old']
            self.x_command += preload_dict.item()['x_command']
            self.y_command += preload_dict.item()['y_command']
            self.command += preload_dict.item()['command']
            self.theta += preload_dict.item()['theta']
            self.theta_old += preload_dict.item()['theta_old']
            self.speed += preload_dict.item()['speed']
            self.steer += preload_dict.item()['steer']
            self.throttle += preload_dict.item()['throttle']
            self.brake += preload_dict.item()['brake']
            self.light += preload_dict.item()['light']

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.bev)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['bev'] = []

        seq_bev = self.bev[index]
        seq_x = self.x[index]
        seq_y = self.y[index]
        seq_x_old = self.x_old[index]
        seq_y_old = self.y_old[index]
        seq_theta = self.theta[index]
        seq_theta_old = self.theta_old[index]

        if self.args.do_augmentation and self.aug_max_rotation!=0:
            no_augment = np.random.randint(2) # we augment 50% of the samples
        else:
            no_augment = 1

        for i in range(self.seq_len):

            self.angle = torch.tensor([0.0]) # we don't want to augment now -> this would be static. We augment in the Webdatset dataloader dynamically.
            crop = load_crop_bev_png(seq_bev[i], self.args.do_augmentation, self.offsets, self.target_im_size, 'Encoded_cropped')

            #only needed for normal dataloader not for webdataset
            crop=np.moveaxis(crop,0,2)

            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            crop=np.moveaxis(crop,2,0)

            data['bev'].append(crop)

            # fix for theta=nan in some measurements
            if np.isnan(seq_theta[i]):
                seq_theta[i] = 0.
            if np.isnan(seq_theta_old[i]):
                seq_theta_old[i] = 0.

        ego_x = seq_x[i]
        ego_y = seq_y[i]
        ego_theta = seq_theta[i]

        # ##### waypoint processing to local coordinates
        waypoints = []
        # we use the future position of the expert as gt
        for i in range(len(seq_x)):
            # waypoint is the transformed version of the origin in local coordinates
            # we use 90-theta instead of theta
            # LBC code uses 90+theta, but x is to the right and y is downwards here
            local_waypoint = transform_2d_points(np.zeros((1,3)),
                np.pi/2-seq_theta[i], -seq_x[i], -seq_y[i], np.pi/2-ego_theta-self.angle.item(), -ego_x, -ego_y)
            waypoints.append(tuple(local_waypoint[0,:2]))

        data['waypoints'] = waypoints
        # convert x_command, y_command to local coordinates
        # taken from LBC code (uses 90+theta instead of theta),
        # see https://github.com/dotchen/LearningByCheating
        local_command_point = []
        R = np.array([
            [np.cos(np.pi/2+ego_theta+self.angle.item()), -np.sin(np.pi/2+ego_theta+self.angle.item())],
            [np.sin(np.pi/2+ego_theta+self.angle.item()),  np.cos(np.pi/2+ego_theta+self.angle.item())]
            ])
        local_command_point = np.array([self.x_command[index]-ego_x, self.y_command[index]-ego_y])
        local_command_point = R.T.dot(local_command_point)

        data['target_point'] = tuple(local_command_point)
        data['steer'] = self.steer[index]
        data['throttle'] = self.throttle[index]
        data['brake'] = self.brake[index]
        data['brake_seq'] = self.brake_seq[index]
        data['light'] = self.light[index]
        data['speed'] = self.speed[index]
        data['theta'] = self.theta[index]
        data['seq_x'] = seq_x
        data['seq_y'] = seq_y
        data['seq_theta'] = seq_theta
        data['seq_x_old'] = seq_x_old
        data['seq_y_old'] = seq_y_old
        data['seq_theta_old'] = seq_theta_old
        data['x_command'] = self.x_command[index]
        data['y_command'] = self.y_command[index]
        data['command'] = self.command[index]
        data['no_aug'] = no_augment

        data['y_offset'] = self.offsets['y_offset']
        data['x_offset_pos'] = self.offsets['x_offset_pos']
        data['x_offset_neg'] = self.offsets['x_offset_neg']
        data['target_im_size'] = self.target_im_size
        data['aug_max_rotation'] = self.aug_max_rotation
        data['bev'] = data['bev'][0] # we just use a single frame
        data = augment_all(data, channels=4)

        return data


def load_crop_bev_png(filename, do_augment, offsets, target_im_size, bev_type):
    """
    Load and crop an Image.
    Crop depends on augmentation angle.
    """
    if do_augment == 0:
        offsets=None

    # load png image
    image = Image.open(filename)
    bev_array = np.array(image)
    bev_array = np.moveaxis(bev_array, 2, 0)

    if bev_type == 'Encoded_cropped' or bev_type=='Org_cropped':
        cropped_image = crop_bev(bev_array, target_im_size, offsets)
    else:
        cropped_image = bev_array

    return cropped_image

def crop_bev(array, target_im_size, offsets=None):
    if offsets is not None:
        y_offset = offsets['y_offset']
        x_offset_pos = offsets['x_offset_pos']
        x_offset_neg = offsets['x_offset_neg']
        PIXELS_AHEAD_VEHICLE = 96 + PIXEL_SHIFT
    else:
        y_offset = 0
        x_offset_pos = 0
        x_offset_neg = 0
        if array.shape[1] == target_im_size[0]:
            PIXELS_AHEAD_VEHICLE = 0
        else:
            PIXELS_AHEAD_VEHICLE = 96 + PIXEL_SHIFT


    (width, height) = (array.shape[1], array.shape[2])
    crop_x = int(target_im_size[0]) + x_offset_neg + x_offset_pos
    crop_y = int(target_im_size[1]) + y_offset * 2

    start_x = height//2 - target_im_size[0]//2 # this would be central crop
    start_x = start_x - PIXELS_AHEAD_VEHICLE # this would be the 'normal' shifted crop
    start_x = start_x - x_offset_pos # bigger crop for augmentation
    start_y = width//2 - target_im_size[1]//2 - y_offset
    cropped_image = array[:, start_x:start_x+crop_x, start_y:start_y+crop_y]
    return cropped_image