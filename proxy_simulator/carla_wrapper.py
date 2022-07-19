import torch
import carla
import pygame

import numpy as np

from external_code.lbc.bird_view.utils.map_utils import MapImage # type: ignore

# Global Flags
PIXELS_PER_METER = 5


class CarlaWrapper():
    def __init__(self, args):
        self._vehicle = None
        self.args = args
        self.town = None

    def _initialize_from_carla(self, town='Town01', port=2000):
        self.town = town
        self.client = carla.Client('localhost', port)

        self.client.set_timeout(360.0)

        self.world = self.client.load_world(town)
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()

        map_image = MapImage(self.world, self.map, PIXELS_PER_METER)
        make_image = lambda x: np.swapaxes(pygame.surfarray.array3d(x), 0, 1).mean(axis=-1)
        road = make_image(map_image.map_surface)
        lane = make_image(map_image.lane_surface)

        global_map = np.zeros((1, 4,) + road.shape)
        global_map[:, 0, ...] = road / 255.
        global_map[:, 1, ...] = lane / 255.

        global_map = torch.tensor(global_map, device=self.args.device, dtype=torch.float32)
        world_offset = torch.tensor(map_image._world_offset, device=self.args.device, dtype=torch.float32)

        return global_map, world_offset
