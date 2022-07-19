import os
from math import ceil
from skimage.transform import rotate

import numpy as np
import torch

# Global Flags
PIXEL_SHIFT = 14

def encode_npy_to_pil(bev_array):
    """
        Encode BEV array with 15 channels to 3 channels -> each channel is rpresented by one bit.
    """
    (channels, width, height) = (bev_array.shape[0], bev_array.shape[1], bev_array.shape[2])

    img = np.zeros([3, width, height]).astype('uint8')
    bev = np.ceil(bev_array).astype('uint8')

    for i in range(channels):
        if i<=4 and i >= 0:
            # Road, Lane, Lights
            img[0] = img[0] | (bev[i] << (8-i-1))
        elif i<=9 and i >= 5:
            # Vehicle, Pedestrian
            img[1] = img[1] | (bev[i] << (8-(i-5)-1))
        elif i<=14 and i >= 10:
            # futue vehicles
            img[2] = img[2] | (bev[i]<< (8-(i-10)-1))

    return img

def decode_pil_to_npy(img):
    (channels, width, height) = (15, img.shape[1], img.shape[2])

    bev_array = np.zeros([channels, width, height])

    for ix in range(5):
        bit_pos = 8-ix-1
        bev_array[[ix, ix+5, ix+5+5]] = (img & (1<<bit_pos)) >> bit_pos

    return bev_array

def get_crop_offsets(angle_deg, target_im_size):
    phi = np.deg2rad(angle_deg)
    rot = np.array(
        [[np.cos(phi), -np.sin(phi)],
        [np.sin(phi), np.cos(phi)]]
    )

    p1 = np.array([-target_im_size[0]/2, PIXEL_SHIFT])
    p2 = np.array([target_im_size[0]/2, PIXEL_SHIFT])
    p3 = np.array([-target_im_size[0]/2, target_im_size[1] + PIXEL_SHIFT])
    p4 = np.array([target_im_size[0]/2, target_im_size[1] + PIXEL_SHIFT])

    p1s = rot @ p1
    p2s = rot @ p2
    p3s = rot @ p3
    p4s = rot @ p4

    y_offset = ceil(abs(p3s[0]) - target_im_size[0]/2)
    x_offset_pos = ceil(abs(p4s[1]))- target_im_size[0]
    x_offset_neg = ceil(abs(p2s[1]))
    return {'y_offset': y_offset, 'x_offset_pos': x_offset_pos, 'x_offset_neg':x_offset_neg}


def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:,2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T

    # reset z-coordinate
    out[:,2] = xyz[:,2]

    return out


def augment_names(input):
    our_dict = {}
    for key, value in input.items():
        our_dict[f"{key.replace('.pyd','')}"] = value

    return our_dict


def augment_all(input, channels=15):
    ## augmentation
    if int(os.environ["aug_max_rotation"]) != 0:
        no_augment = np.random.randint(2) # we augment 50% of the samples
    else:
        no_augment = 1

    input['no_aug'] = no_augment

    angle = (torch.rand(1)*2 - 1.0) * int(os.environ["aug_max_rotation"])
    angle_rad = np.deg2rad(angle)

    target_im_size = input['target_im_size']
    seq_len = 1
    pred_len = 4
    bev_array = input['bev']
    bev_array = np.moveaxis(bev_array, 0, -1)

    if angle == 0:
        image_rotated = bev_array
    else:
        image_rotated = rotate(bev_array, angle, center=(bev_array.shape[1]/2, bev_array.shape[0]-input['x_offset_neg']+PIXEL_SHIFT), order=0, preserve_range=True).astype(np.uint8)

    image_rotated = np.moveaxis(image_rotated, -1, 0)
    image_rotated = decode_pil_to_npy(image_rotated)

    # ensure correct shapes for regular and king data
    assert image_rotated.shape[0] == 15 or image_rotated.shape[0] == 4
    if channels == 15:
        image_rotated = np.delete(image_rotated, [2,3,4], 0)#.astype(bool) # remove traffic light channels
        image_rotated = image_rotated[:-8] # remove future channels
    if channels == 4:
        image_rotated = image_rotated[:4]

    (height, width) = (image_rotated.shape[1], image_rotated.shape[2])
    if height != target_im_size[0]:
        crop_x = int(target_im_size[0])
        crop_y = int(target_im_size[1])
        start_x = input['x_offset_pos']
        start_y = width//2 - target_im_size[1]//2
        cropped_image = image_rotated[:, start_x:start_x+crop_x, start_y:start_y+crop_y]
        input['bev'] = cropped_image
    else:
        input['bev'] = image_rotated

    # lidar and waypoint processing to local coordinates
    waypoints = []
    seq_theta_old = input['seq_theta_old']
    seq_x_old = input['seq_x_old']
    seq_y_old = input['seq_y_old']
    x_command = input['x_command']
    y_command = input['y_command']
    ego_x = seq_x_old[0]
    ego_y = seq_y_old[0]
    ego_theta = seq_theta_old[0]

    for i in range(seq_len, seq_len + pred_len):
        # waypoint is the transformed version of the origin in local coordinates
        # we use 90-theta instead of theta
        # LBC code uses 90+theta, but x is to the right and y is downwards here
        local_waypoint = transform_2d_points(np.zeros((1,3)),
            np.pi/2-seq_theta_old[i], -seq_x_old[i], -seq_y_old[i], np.pi/2-ego_theta-angle_rad.item(), -ego_x, -ego_y)
        waypoints.append(tuple(local_waypoint[0,:2]))
    input['waypoints'] = waypoints

    # convert x_command, y_command to local coordinates
    # taken from LBC code (uses 90+theta instead of theta),
    # see https://github.com/dotchen/LearningByCheating
    local_command_point = []
    R = np.array([
        [np.cos(np.pi/2+ego_theta+angle_rad.item()), -np.sin(np.pi/2+ego_theta+angle_rad.item())],
        [np.sin(np.pi/2+ego_theta+angle_rad.item()),  np.cos(np.pi/2+ego_theta+angle_rad.item())]
        ])
    local_command_point = np.array([x_command-ego_x, y_command-ego_y])
    local_command_point = R.T.dot(local_command_point)
    input['target_point'] = tuple(local_command_point)

    return input
