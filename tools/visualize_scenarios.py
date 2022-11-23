import os
import sys
import cv2
import json
import glob
import math
import torch
import argparse
sys.path.append(os.getcwd())

import numpy as np

from tqdm import tqdm
from PIL import ImageDraw, Image

from proxy_simulator.renderer import BaseRenderer
from proxy_simulator.carla_wrapper import CarlaWrapper


GPU_PI = torch.tensor([np.pi], device="cuda", dtype=torch.float32)
BB_EXTENT = torch.tensor([[0.90,2.20],[0.90,-2.20],[-0.90, -2.20],[-0.90, 2.20]]).cuda()


class BEVVisualizer:
    def __init__(self, args):
        """
        """
        self.args = args
        self.carla_wrapper = CarlaWrapper(args)

        self.log_file_paths, self.results_file_paths = self.parse_scenario_log_dir()

        self.town = None
        self.max_iter = self.fetch_max_iter()

    def visualize(self):
        """
        Visualizes logs in a simple abstract BEV representation that is centered
        on the ego agent. Can dump .gifs or .mp4s.
        """
        # loop over all logs
        for log_path, results_path in tqdm(
            zip(self.log_file_paths, self.results_file_paths), total=len(self.log_file_paths)
            ):
            log = self.parse_json_file(log_path)
            results = self.parse_json_file(results_path)

            # extract meta data
            name = log["meta_data"]["name"]
            town = log["meta_data"]["town"]
            density = log_path.split("/")[1].split("_")[-1]
            critical_iter = results["first_metrics"]["iteration"]

            if self.args.opt_iter == -1:
                if critical_iter <= self.max_iter[density][town]:
                    opt_iter = critical_iter
                else:
                    opt_iter = self.max_iter[density][town]
            else:
                opt_iter = self.args.opt_iter
    
            # set town in relevant components if necessary
            if town != self.town:
                global_map, map_offset = self.carla_wrapper._initialize_from_carla(town)
                renderer = BaseRenderer(
                    self.args, map_offset, global_map.shape[2:4], viz=True
                )

            bev_overview_vis_per_t = []
            for t, state in enumerate(log["states"][opt_iter]):
                # map dict of lists to dict of tensors
                for substate in state:
                    state[substate] = torch.tensor(
                            state[substate],
                            device=self.args.device,
                    )

                # fetch local crop of map
                local_map = renderer.get_local_birdview(
                    global_map,
                    state["pos"].unsqueeze(0)[:, 0:1], # ego pos as origin
                    state["yaw"].unsqueeze(0)[:, 0:1], # ego yaw as reference
                )

                vehicle_corners = self.get_corners_vectorized(
                    BB_EXTENT,
                    state["pos"].unsqueeze(0),
                    state["yaw"].unsqueeze(0),
                )

                vehicle_corners = renderer.world_to_pix_crop(
                        vehicle_corners, 
                        state["pos"].unsqueeze(0)[:, 0:1], # ego pos as origin
                        state["yaw"].unsqueeze(0)[:, 0:1], # ego yaw as reference
                )

                vehicle_corners = vehicle_corners.detach().cpu().numpy()
                vehicle_corners = vehicle_corners[0].reshape(vehicle_corners.shape[1]//4,4,2)

                bev_vis = self.tensor_to_pil(local_map)
                bev_overview_vis_draw = ImageDraw.Draw(bev_vis)

                for i in range(vehicle_corners.shape[0]):
                    if i == 0:
                        bev_overview_vis_draw.polygon(vehicle_corners[i].flatten(),fill=(222, 112, 97),outline=(0, 0, 0))
                        bev_overview_vis_draw.polygon(np.concatenate([vehicle_corners[i][2], vehicle_corners[i][1], np.mean(vehicle_corners[i], axis=0)]),outline=(0, 0, 0))
                    else:
                        bev_overview_vis_draw.polygon(vehicle_corners[i].flatten(),fill=(105, 156, 219),outline=(0, 0, 0))
                        bev_overview_vis_draw.polygon(np.concatenate([vehicle_corners[i][2], vehicle_corners[i][1], np.mean(vehicle_corners[i], axis=0)]),outline=(0, 0, 0))

                bev_overview_vis_per_t.append(bev_vis)
    
            save_path = os.path.join(os.path.dirname(log_path), name + f"_iter_{opt_iter}")

            # save frames as gif
            bev_overview_vis_per_t[0].save(
                save_path + '.gif', 
                save_all=True, 
                append_images=bev_overview_vis_per_t[1:], 
                optimize=True,
                loop=0,
            )

            # save frames as .mp4
            # codec = cv2.VideoWriter_fourcc(*'mp4v')    
            # video_writer = cv2.VideoWriter(save_path + ".mp4",codec, 4, bev_overview_vis_per_t[0].size)       
            # for timestep in range(len(log["states"])):
            #         video_writer.write(cv2.cvtColor(
            #             np.array(bev_overview_vis_per_t[timestep]), cv2.COLOR_RGB2BGR))
            # video_writer.release()

    def tensor_to_pil(self, grid):
        """
        """
        colors = [
            (120, 120, 120), # road
            (253, 253, 17), # lane
            (0, 0, 142), # vehicle
        ]
        
        grid = grid.detach().cpu()

        grid_img = np.zeros((grid.shape[2:4] + (3,)), dtype=np.uint8)
        grid_img[...] = [225, 225, 225]
        
        for i in range(len(colors)):
            grid_img[grid[0, i, ...] > 0] = colors[i]

        pil_img = Image.fromarray(grid_img)

        return pil_img

    def get_corners_vectorized(self, extent, pos, yaw):
        yaw = GPU_PI/2 -yaw
        extent = extent.unsqueeze(-1)

        rot_mat = torch.cat(
            [
                torch.cos(yaw), torch.sin(yaw),
                -torch.sin(yaw), torch.cos(yaw),
            ],
            dim=-1,
        ).view(yaw.size(1), 1, 2, 2).expand(yaw.size(1), 4, 2, 2)

        rotated_corners = rot_mat @ extent

        rotated_corners = rotated_corners.view(yaw.size(1), 4, 2) + pos[0].unsqueeze(1)
        
        return rotated_corners.view(1, -1, 2)

    def fetch_max_iter(self):
        # read timings dict from json file
        with open('tools/timings.json') as f:
            timings = json.load(f)

        max_iter = {}
        max_GPU_seconds = 180
        for density in ["1", "2", "4"]:
            timings_per_town_per_density = timings[self.args.optim_method][density]
            max_iter[density] = {} 
            for town, timing in timings_per_town_per_density.items():
                max_iter[density][str(town)] = math.floor(max_GPU_seconds / timing)
        return max_iter
    
    def parse_scenario_log_dir(self):
        """
        Parse generation results directory and gather 
        the JSON file paths from the per-route directories.
        """
        route_scenario_dirs = sorted(
            glob.glob(
                self.args.scenario_log_dir + "/**/RouteScenario*/", recursive=True
            ),
            key=lambda path: (path.split("_")[-6]),
        )

        # gather all records and results JSON files
        results_files = []
        records_files = []
        for dir in route_scenario_dirs:
            results_files.extend(
                sorted(
                    glob.glob(dir + "results.json")
                )
            )
            records_files.extend(
                sorted(
                    glob.glob(dir + "scenario_records.json")
                )
            )

        return records_files, results_files

    def parse_json_file(self, records_file):
        """
        """
        return json.loads(open(records_file).read())


def main(args):
    """
    """
    vizualizer =  BEVVisualizer(args)
    vizualizer.visualize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--scenario_log_dir",
        type=str,
        default="generation_results",
        help="The directory containing the per-route directories with the "
             "corresponding scenario log .json files.",
    )
    parser.add_argument(
        "--opt_iter",
        type=int,
        default=-1,
        help="Specifies at which iteration in the optimization process the "
             "scenarios should be visualized. Set to -1 to automatically "
             "select the critical perturbation for each scenario.",
    )
    parser.add_argument(
        "--optim_method",
        default="Adam",
        choices=["Adam", "Both_Paths"]
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="Carla port."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The number of parallel simulations."
    )

    args = parser.parse_args()

    main(args)