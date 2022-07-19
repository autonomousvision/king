import numpy as np

import torch
import torch.nn.functional as F

# Global Flags
PIXELS_PER_METER = 5


class DatagenBEVRenderer():
    def __init__(self, args, map_offset, map_dims, data_generation=False):
        self.args = args
        if data_generation:
            self.PIXELS_AHEAD_VEHICLE = 0 # ego car is central
            self.local_view_dims = (500, 500)
            self.crop_dims = (500, 500)
        else:
            self.PIXELS_AHEAD_VEHICLE = 100 + 10
            self.local_view_dims = (320, 320)
            self.crop_dims = (192, 192)

        self.map_offset = map_offset
        self.map_dims = map_dims
        self.local_view_scale = (
            self.local_view_dims[1] / self.map_dims[1],
            self.local_view_dims[0] / self.map_dims[0]
        )
        self.crop_scale = (
            self.crop_dims[1] / self.map_dims[1],
            self.crop_dims[0] / self.map_dims[0]
        )

    def world_to_pix(self, pos):
        pos_px = (pos-self.map_offset) * PIXELS_PER_METER

        return pos_px

    def world_to_pix_crop(self, query_pos, crop_pos, crop_yaw, offset=(0, 0)):
        crop_yaw = crop_yaw + np.pi / 2

        # transform to crop pose
        rotation = torch.tensor(
            [[torch.cos(crop_yaw), -torch.sin(crop_yaw)],
            [torch.sin(crop_yaw),  torch.cos(crop_yaw)]],
            device=self.args.device,
        )

        crop_pos_px = self.world_to_pix(crop_pos)

        # correct for the fact that crop is only in front of ego agent
        shift = torch.tensor(
            [0., - self.PIXELS_AHEAD_VEHICLE],
            device=self.args.device,
        )

        query_pos_px_map = self.world_to_pix(query_pos)

        query_pos_px = rotation.T @ (query_pos_px_map - crop_pos_px) - shift

        # shift coordinate frame to top left corner of the crop
        pos_px_crop = query_pos_px + torch.tensor([self.crop_dims[1] / 2, self.crop_dims[0] / 2], device=self.args.device)

        return pos_px_crop

    def world_to_rel(self, pos):
        pos_px = self.world_to_pix(pos)
        pos_rel = pos_px / torch.tensor([self.map_dims[1],self.map_dims[0]], device=self.args.device)

        pos_rel = pos_rel * 2 - 1

        return pos_rel

    def render_agent_bv(
            self,
            grid,
            grid_pos,
            grid_orientation,
            vehicle,
            position,
            orientation,
            channel=5,
        ):
        """
        """
        orientation = orientation + np.pi / 2

        pos_pix_bv = self.world_to_pix_crop(position, grid_pos, grid_orientation)

        # to centered relative coordinates
        h, w = (grid.size(-2), grid.size(-1))
        pos_rel_bv = pos_pix_bv / torch.tensor([h, w], device=self.args.device)
        pos_rel_bv = pos_rel_bv * 2 - 1
        pos_rel_bv = pos_rel_bv * -1

        scale_h = torch.tensor([grid.size(2) / vehicle.size(2)], device=self.args.device)
        scale_w = torch.tensor([grid.size(3) / vehicle.size(3)], device=self.args.device)

        scale_transform = torch.tensor(
            [[scale_w, 0, 0],
            [0, scale_h, 0],
            [0, 0, 1]],
            device=self.args.device
        ).view(1, 3, 3)

        grid_orientation = grid_orientation + np.pi / 2
        rotation_transform = torch.tensor(
            [[torch.cos(orientation - grid_orientation), torch.sin(orientation - grid_orientation), 0],
            [- torch.sin(orientation - grid_orientation), torch.cos(orientation - grid_orientation), 0],
            [0, 0, 1]],
            device=self.args.device
        ).view(1, 3, 3)

        translation_transform = torch.tensor(
            [[1, 0, pos_rel_bv[0]],
            [0, 1, pos_rel_bv[1]],
            [0, 0, 1]],
            device=self.args.device
        ).view(1, 3, 3)

        affine_transform = scale_transform @ rotation_transform @ translation_transform

        affine_grid = F.affine_grid(
            affine_transform[:, 0:2, :], # expects Nx2x3
            (1, 1, grid.shape[2], grid.shape[3]),
            align_corners=True,
        )

        vehicle_rendering = F.grid_sample(
            vehicle,
            affine_grid,
            align_corners=True,
        )

        grid[:, channel, ...] += vehicle_rendering.squeeze()

    def get_local_birdview(self, grid, position, orientation):
        """
        """

        position = self.world_to_rel(position) #, self.map_dims)
        orientation = orientation + np.pi/2 #+ np.pi

        scale_transform = torch.tensor(
            [[self.crop_scale[1], 0, 0],
            [0, self.crop_scale[0], 0],
            [0, 0, 1]],
            device=self.args.device
        ).view(1, 3, 3)

        rotation_transform = torch.tensor(
            [[torch.cos(orientation), -torch.sin(orientation), 0],
            [torch.sin(orientation), torch.cos(orientation), 0],
            [0, 0, 1]],
            device=self.args.device
        ).view(1, 3, 3)

        # shift cropping position so ego agent is at bottom boundary
        shift = torch.tensor([0., - 2 * self.PIXELS_AHEAD_VEHICLE / self.map_dims[0]], device=self.args.device)
        position = position + rotation_transform[0, 0:2, 0:2] @ shift

        translation_transform = torch.tensor(
            [[1, 0, position[0] / self.crop_scale[0]],
            [0, 1, position[1] / self.crop_scale[1]],
            [0, 0, 1]],
            device=self.args.device
        ).view(1, 3, 3)

        local_view_transform = scale_transform @ translation_transform @ rotation_transform

        affine_grid = F.affine_grid(
            local_view_transform[:, 0:2, :],
            (1, 1, self.crop_dims[0], self.crop_dims[0]),
            align_corners=True,
        )

        local_view = F.grid_sample(
            grid,
            affine_grid,
            align_corners=True,
        )

        return local_view

    def render_agent_bv_batched(
            self,
            grid,
            grid_pos,
            grid_orientation,
            vehicle,
            position,
            orientation,
            channel=5,
        ):
        """
        """
        orientation = orientation + np.pi / 2
        batch_size = position.shape[0]

        pos_pix_bv = self.world_to_pix_crop_batched(position, grid_pos, grid_orientation)

        # to centered relative coordinates
        h, w = (grid.size(-2), grid.size(-1))
        pos_rel_bv = pos_pix_bv / torch.tensor([h, w], device=self.args.device)
        pos_rel_bv = pos_rel_bv * 2 -1
        pos_rel_bv = pos_rel_bv * -1

        scale_h = torch.tensor([grid.size(2) / vehicle.size(2)], device=self.args.device)
        scale_w = torch.tensor([grid.size(3) / vehicle.size(3)], device=self.args.device)

        scale_transform = torch.tensor(
            [[scale_w, 0, 0],
            [0, scale_h, 0],
            [0, 0, 1]],
            device=self.args.device
        ).view(1, 3, 3).expand(batch_size, -1, -1)

        grid_orientation = grid_orientation + np.pi / 2
        angle_delta = orientation - grid_orientation
        zeros = torch.zeros_like(angle_delta)
        ones = torch.ones_like(angle_delta)
        rotation_transform = torch.stack(
            [ torch.cos(angle_delta), torch.sin(angle_delta), zeros,
            -torch.sin(angle_delta), torch.cos(angle_delta), zeros,
            zeros,                   zeros,                  ones],
            dim=-1
        ).view(batch_size, 3, 3)

        translation_transform = torch.stack(
            [ones,  zeros, pos_rel_bv[..., 0:1],
             zeros, ones,  pos_rel_bv[..., 1:2],
             zeros, zeros, ones],
            dim=-1,
        ).view(batch_size, 3, 3)

        affine_transform = scale_transform @ rotation_transform @ translation_transform

        affine_grid = F.affine_grid(
            affine_transform[:, 0:2, :], # expects Nx2x3
            (batch_size, 1, grid.shape[2], grid.shape[3]),
            align_corners=True,
        )

        vehicle_rendering = F.grid_sample(
            vehicle,
            affine_grid,
            align_corners=True,
        )



        for i in range(batch_size):
            grid[:, int(channel[i].item()), ...] += vehicle_rendering[i].squeeze()

    def world_to_pix_crop_batched(self, query_pos, crop_pos, crop_yaw, offset=(0, 0)):
        crop_yaw = crop_yaw + np.pi / 2
        batch_size = crop_pos.shape[0]

        # transform to crop pose
        rotation = torch.stack(
            [torch.cos(crop_yaw), -torch.sin(crop_yaw),
            torch.sin(crop_yaw),  torch.cos(crop_yaw)],
            dim=-1,
        ).view(batch_size, 2, 2)

        crop_pos_px = self.world_to_pix(crop_pos)

        shift = torch.tensor(
            [0., - self.PIXELS_AHEAD_VEHICLE],
            device=self.args.device,
        )

        query_pos_px_map = self.world_to_pix(query_pos)

        query_pos_px = torch.transpose(rotation, -2, -1).unsqueeze(1) @ \
            (query_pos_px_map - crop_pos_px).unsqueeze(-1)
        query_pos_px = query_pos_px.squeeze(-1) - shift

        pos_px_crop = query_pos_px + torch.tensor([self.crop_dims[1] / 2, self.crop_dims[0] / 2], device=self.args.device)

        return pos_px_crop