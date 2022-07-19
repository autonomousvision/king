import torch

import numpy as np


GPU_PI = torch.tensor([np.pi], device="cuda", dtype=torch.float32)


class RouteDeviationCostRasterized():
    """
    """
    def __init__(self, sim_args):
        """
        This cost measures the overlap between an agent and
        the non-drivable area using a Gaussian Kernel.
        """
        self.batch_size = sim_args.batch_size
        self.num_agents = sim_args.num_agents
        self.sigma_x = 5*128
        self.sigma_y = 2*128
        self.variance_x = self.sigma_x**2.
        self.variance_y = self.sigma_y**2.

        # default vehicle corners given an agent at the origin
        self.original_corners = torch.tensor(
            [[1.0, 2.5], [1.0, -2.5], [-1.0, 2.5], [-1.0, -2.5]]
        ).cuda()

    def get_corners(self, pos, yaw):
        """
        Obtain agent corners given the position and yaw.
        """
        yaw = GPU_PI/2 - yaw

        rot_mat = torch.cat(
            [
                torch.cos(yaw), -torch.sin(yaw),
                torch.sin(yaw), torch.cos(yaw),
            ],
            dim=-1,
        ).view(yaw.size(1), 1, 2, 2).expand(yaw.size(1), 4, 2, 2)

        rotated_corners = rot_mat @ self.original_corners.unsqueeze(-1)

        rotated_corners = rotated_corners.view(yaw.size(1), 4, 2) + pos[0].unsqueeze(1)

        return rotated_corners.view(1, -1, 2)

    def crop_map(self, j, i, y_extent, x_extent, road_rasterized):
        i_min, i_max = int(max(0, i - 32)), int(min(i + 32, x_extent))
        j_min, j_max = int(max(0, j - 32)), int(min(j + 32, y_extent))
        road_rasterized = road_rasterized[i_min:i_max:2, j_min:j_max:2]
        return road_rasterized

    def get_pixel_grid(self, i, j, x_extent, y_extent):

        i_min, i_max = int(max(0, i - 32)), int(min(i + 32, x_extent))
        j_min, j_max = int(max(0, j - 32)), int(min(j + 32, y_extent))

        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(i_min, i_max - 1, (i_max - i_min)),
                torch.linspace(j_min, j_max - 1, (j_max - j_min))
            ),
            -1
        )  # (H, W, 2)

        coords = coords[::2, ::2]
        return coords.float()

    def apply_gauss_kernels(self, coords, pos):
        sigma = 5.
        pos = pos[0, :, :]
        coords = torch.cat(coords, dim=0)

        gk = torch.mean(((coords - pos[: ,None, None, :])/sigma)**2, dim=-1)
        gk = (1./(2*GPU_PI*sigma*sigma))*torch.exp(-gk)

        return gk

    def __call__(self, road_rasterized, pos, yaw, crop_center, pos_w2m):
        """
        Computes the cost.
        """
        pos = self.get_corners(pos, yaw)
        crop_center = pos_w2m(crop_center[None])[0]
        pos = pos_w2m(pos.view(-1, 2))
        pos = pos.view(self.batch_size, self.num_agents * 4, 2)

        x_extent, y_extent = road_rasterized.size(0), road_rasterized.size(1)

        roads_rasterized = []
        for i in range(pos.size(1)):
            crop_center = pos[0, i, :]
            crop_road_rasterized = self.crop_map(
                crop_center[0].item(),
                crop_center[1].item(),
                x_extent,
                y_extent,
                road_rasterized
            )
            crop_road_rasterized = 1. - crop_road_rasterized
            roads_rasterized.append(crop_road_rasterized)

        coords = []
        for i in range(pos.size(1)):
            crop_center = pos[0,i,:]
            coords.append(
                self.get_pixel_grid(
                    crop_center[0].item(),
                    crop_center[1].item(),
                    x_extent,
                    y_extent
                ).cuda().unsqueeze(0)
            )

        gks = self.apply_gauss_kernels(coords, pos)
        roads_rasterized = torch.cat([road_rasterized[None] for road_rasterized in roads_rasterized],dim=0)
        costs = torch.sum(roads_rasterized*torch.transpose(gks[:, :, :], 1, 2))[None, None]
        return costs


class BatchedPolygonCollisionCost():
    """
    """
    def __init__(self, sim_args):
        """
        """
        self.sim_args = sim_args
        self.batch_size = sim_args.batch_size
        self.num_agents = sim_args.num_agents

        self.unit_square = torch.tensor(
            [
                [ 1.,  1.],  # back right corner
                [-1.,  1.],  # back left corner
                [-1., -1.],  # front left corner
                [ 1., -1.],  # front right corner
            ],
            device=sim_args.device,
        ).view(1, 1, 4, 2).expand(self.batch_size, self.num_agents+1, 4, 2)

        self.segment_start_transform = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
            device='cuda',
        ).reshape(1, 4, 4)

        self.segment_end_transform = torch.tensor(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
            ],
            dtype=torch.float32,
            device='cuda',
        ).reshape(1, 4, 4)

    def vertices_to_edges_vectorized(self, vertices):
        """
        """
        segment_start = self.segment_start_transform @ vertices
        segment_end = self.segment_end_transform @ vertices
        return segment_start, segment_end

    def __call__(self, ego_state, ego_extent, adv_state, adv_extent):
        """
        """
        ego_pos = ego_state["pos"]
        ego_yaw = ego_state["yaw"]
        ego_extent = torch.diag_embed(ego_extent)

        adv_pos = adv_state["pos"]
        adv_yaw = adv_state["yaw"]
        adv_extent = torch.diag_embed(adv_extent)

        pos = torch.cat([ego_pos, adv_pos], dim=1)
        yaw = torch.cat([ego_yaw, adv_yaw], dim=1)
        extent = torch.cat([ego_extent, adv_extent], dim=1)

        rot_mat = torch.cat(
            [
                torch.cos(yaw), -torch.sin(yaw),
                torch.sin(yaw), torch.cos(yaw),
            ],
            dim=-1,
        ).view(self.batch_size, self.num_agents+1, 2, 2)

        corners = self.unit_square @ extent

        corners = corners @ rot_mat.permute(0, 1, 3, 2)

        corners = corners + pos.unsqueeze(-2)

        segment_starts, segment_ends = self.vertices_to_edges_vectorized(corners)
        segments = segment_ends - segment_starts

        corners = corners.repeat_interleave(self.num_agents+1, dim=1)
        segment_starts = segment_starts.repeat(1, self.num_agents+1, 1, 1)
        segment_ends = segment_ends.repeat(1, self.num_agents+1, 1, 1)
        segments = segments.repeat(1, self.num_agents+1, 1, 1)

        corners = corners.repeat_interleave(4, dim=2)
        segment_starts = segment_starts.repeat(1, 1, 4, 1)
        segment_ends = segment_ends.repeat(1, 1, 4, 1)
        segments = segments.repeat(1, 1, 4, 1)

        projections = torch.matmul(
            (corners - segment_starts).unsqueeze(-2),
            segments.unsqueeze(-1)
        ).squeeze(-1)

        projections = projections / torch.sum(segments**2,dim=-1, keepdim=True)

        projections = torch.clamp(projections, 0., 1.)

        closest_points = segment_starts + segments * projections

        distances = torch.norm(corners - closest_points, dim=-1, keepdim=True)
        closest_points_list = closest_points.view(-1,2).clone()

        distances, distances_idxs = torch.min(distances, dim=-2)

        distances_idxs = distances_idxs.unsqueeze(-1).repeat(1, 1, 1, 2)

        distances = distances.view(self.batch_size, self.num_agents + 1, self.num_agents + 1, 1)

        n = self.num_agents + 1
        distances = distances[0, :, :, 0].flatten()[1:].view(n-1, n+1)[:, :-1].reshape(n, n-1)

        ego_cost = torch.min(distances[0])[None, None]

        if distances.size(0) > 2:
            distances_adv = distances[1:, 1:]
            adv_cost = torch.min(distances_adv, dim=-1)[0][None]
        else:
            adv_cost = torch.zeros(1, 0).cuda()

        return ego_cost, adv_cost, closest_points_list
