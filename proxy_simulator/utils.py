import math
import PIL
import carla
import datetime
import os
import pickle
import copy
import torch

import numpy as np

from enum import Enum


def make_grid(image_batch, nrows=1):
    """
    Takes a batch of images and arranges them into a tiled gallery in a single
    image for visualization. This is intended for the numpy side of things.
    Similar to torchvision.utils.make_grid().
    """
    # add padding between tiles to get some separation
    pad = 2

    # check the type of image_batch
    if isinstance(image_batch, list):
        if isinstance(image_batch[0], np.ndarray):
            image_batch = np.stack(image_batch, axis=0)
        elif isinstance(image_batch[0], PIL.Image.Image):
            image_batch = np.stack(
                [np.asarray(elem) for elem in image_batch],
                axis=0,
            )
        else:
            raise ValueError("Unsupported data type.")
    elif isinstance(image_batch, np.ndarray):
        assert len(image_batch.shape) == 4
    else:
        raise ValueError("Unsupported data type.")

    # check that we have enough rows to actually fit the entire batch
    batch_size = image_batch.shape[0]
    assert nrows <= batch_size

    # we use as many columns as we need to fit all batch elements within the
    # specified number of rows
    ncols = math.ceil(batch_size / nrows)

    h = image_batch.shape[1] + pad
    w = image_batch.shape[2] + pad

    canvas = np.zeros(
        (h * nrows + pad,
         w * ncols + pad,
         image_batch.shape[-1])
    )

    for row_idx in range(nrows):
        for col_idx in range(ncols):
            # compute index of current batch element
            elem_idx = row_idx * ncols + col_idx

            # if we have more tiles than images we simply leave them black
            if elem_idx + 1 > batch_size:
                break

            # paste current batch element into corresponding tile in canvas
            start_h, end_h = row_idx * h + pad, row_idx * h + h
            start_w, end_w = col_idx * w + pad, col_idx * w + w
            canvas[start_h:end_h, start_w:end_w, :] = image_batch[elem_idx]

    return PIL.Image.fromarray(canvas.astype(np.uint8))


def detach_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def state_dict_to_transform(state_dict):
    """
        Maps a dict of batched states as used in the differentiable simulator to
        a nested list of carla.Transform objects per batch element and actor.
    """
    transforms_batched = []
    for batch_idx in range(state_dict["pos"].size(0)):
        transforms = []
        for agent_id in range(state_dict["pos"].size(1)):
            transform = carla.Transform(
                location=carla.Location(
                    x=state_dict["pos"][batch_idx][agent_id][0].item(),
                    y=state_dict["pos"][batch_idx][agent_id][1].item(),
                    z=0.05,
                ),
                rotation=carla.Rotation(
                    pitch=0.,
                    yaw=state_dict["yaw"][batch_idx][agent_id][0].item() / np.pi * 180,
                    roll=0.,
                )
            )
            transforms.append(transform)
        transforms_batched.append(transforms)
    return transforms_batched


class _GradientScaling(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, threshold):
        # ctx.save_for_backward(threshold)
        ctx.threshold = threshold
        return x.clone()

    @staticmethod
    def backward(ctx, gradients):
        threshold = ctx.threshold
        if gradients.norm() > threshold:
            gradients = gradients / gradients.norm() * threshold
        return gradients, None


def carla_img_to_rgb_array(image):
    """
    Convert a CARLA raw image to a RGB numpy array.
    """
    array = carla_img_to_bgra_array(image)
    # Convert BGRA to RGB.
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


def carla_img_to_bgra_array(image):
    """
    Convert a CARLA raw image to a BGRA numpy array.
    """
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    return array


def parse_lidar(data):
    points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
    points = copy.deepcopy(points)
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    return points


class RoadOption(Enum):
    """
    Taken from CARLA Python API

    https://github.com/carla-simulator/carla/blob/a1b37f7f1cf34b0f6f77973c469926ea368d1507/PythonAPI/carla/agents/navigation/local_planner.py#L17

    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


def roadopt_from_int(opt):
    if opt == -1:
        return RoadOption.VOID
    elif opt == 1:
        return RoadOption.LEFT
    elif opt == 2:
        return RoadOption.RIGHT
    elif opt == 3:
        return RoadOption.STRAIGHT
    elif opt == 4:
        return RoadOption.LANEFOLLOW
    elif opt == 5:
        return RoadOption.CHANGELANELEFT
    elif opt == 6:
        return RoadOption.CHANGELANERIGHT
    else:
        raise ValueError


# code taken from https://github.com/RenzKa/sign-segmentation/blob/master/utils/utils.py
def save_args(args, save_folder, opt_prefix="opt"):
    """
    """
    opts = vars(args)
    os.makedirs(save_folder, exist_ok=True)

    # Save to text
    opt_filename = f"{opt_prefix}.txt"
    opt_path = os.path.join(save_folder, opt_filename)
    with open(opt_path, "a") as opt_file:
        opt_file.write("====== Options ======\n")
        for k, v in sorted(opts.items()):
            opt_file.write(f"{str(k)}: {str(v)}\n")
        opt_file.write("=====================\n")
        opt_file.write(f"launched at {str(datetime.datetime.now())}\n")

    # Save as pickle
    opt_picklename = f"{opt_prefix}.pkl"
    opt_picklepath = os.path.join(save_folder, opt_picklename)
    with open(opt_picklepath, "wb") as opt_file:
        pickle.dump(opts, opt_file)
    print(f"Saved options to {opt_path}")
