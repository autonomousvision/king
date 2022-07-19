import argparse
import random
import glob
import json
import os
import torch
import math
import sys
sys.path.append(os.getcwd())

import numpy as np

from srunner.tools.route_manipulation import interpolate_trajectory
from leaderboard.utils.route_indexer import RouteIndexer
from tqdm import tqdm

from driving_agents.king.expert.expert_agent import AutoPilot
from proxy_simulator.motion_model import BicycleModel
from proxy_simulator.simulator import ProxySimulator


class SolvabilityEngine:
    def __init__(self, args):
        """
        """
        self.args = args
        self.town = None

        ego_expert = AutoPilot(self.args, device=args.device)

        self.simulator = ProxySimulator(
            self.args,
            ego_expert=ego_expert,
            motion_model=BicycleModel(1/args.sim_tickrate).to(self.args.device),
        )

        self.records_files, self.results_files = self.parse_summary_dir()
        self.route_indexer = RouteIndexer(self.args.routes_file, None, 1)
        self.max_iter = self.fetch_max_iter()

    def fetch_max_iter(self):
        # read timings dict from json file
        with open('tools/timings.json') as f:
            timings = json.load(f)

        timings_per_town = timings[self.args.optim_method][str(self.args.num_agents)]
        max_iter = {}
        max_GPU_seconds = self.args.max_GPU_hours * 60 * 60
        for town, timing in timings_per_town.items():
            max_iter[str(town)] = math.floor(max_GPU_seconds / timings_per_town[town])
        return max_iter

    @torch.no_grad()
    def run_solvability_check(self):
        """
        """
        solvability = {}
        for records_idx, records in enumerate(tqdm(self.records_files)):
            # parse scenario files
            scenario_def = self.parse_json_file(records)
            scenario_metrics = self.parse_json_file(
                self.results_files[records_idx]
            )

            critical_iteration = scenario_metrics["first_metrics"]

            town = scenario_def["meta_data"]["town"]
            name = scenario_def["meta_data"]["name"]

            max_iter = self.max_iter[town]
            iteration = critical_iteration["iteration"]
            best_loss_iter = np.argmin(
                [v["Loss"] for k, v in scenario_metrics["all_iterations"].items()
                    if int(k) <= max_iter]
            )
            if iteration > max_iter:
                iteration = best_loss_iter

            # if this is a new town we re-initialize
            if town != self.town:
                self.simulator.set_new_town(self.args, town)
                self.town = town

            self.simulator.renderer.reset()
            self.simulator.ego_expert.reset()

            # fetch initial state and map to tensors
            initial_state = scenario_def["states"][iteration][0]
            for substate in initial_state:
                self.simulator.state[substate] = torch.tensor(
                        initial_state[substate],
                        device=self.args.device,
                ).unsqueeze(0)

            # set route so agent can navigate
            gps_route, route, route_config = self.parse_route(scenario_def["meta_data"]["name"]+'.0')
            self.simulator.set_route(gps_route, route, route_config)

            semantic_grid = self.simulator.map
            for t in range(self.args.sim_horizon):
                recorded_state = scenario_def["states"][iteration][t]
                if t == 0:
                    for substate in recorded_state:
                        self.simulator.state[substate] = torch.tensor(
                                recorded_state[substate],
                                device=self.args.device,
                        ).unsqueeze(0)

                input_data = self.simulator.get_ego_sensor()

                # fetch observations for ego agent
                observations, _ = self.simulator.renderer.get_observations(
                    semantic_grid,
                    self.simulator.get_ego_state(),
                    self.simulator.get_adv_state(),
                )
                input_data.update(observations)

                input_data.update({
                    "timestep": t,
                    "route_name": name + "_to_" + name.split("_")[-1]
                })

                # fetch actions from ego for current observations
                to_tn = lambda x: torch.tensor(x).float().cuda()
                adv_actions = scenario_def['adv_actions'][iteration][t]
                adv_actions = {k: to_tn(v) for k, v in adv_actions.items()}

                expert_actions = self.simulator.ego_expert.run_step(
                    input_data, self.simulator, adv_actions=adv_actions
                )

                self.simulator.run_termination_checks()

                self.simulator.step(expert_actions, adv_actions)

            collisions = self.simulator.ego_collision[self.simulator.ego_collision == 1.]
            col_metric = len(collisions) / self.args.batch_size

            solvability[scenario_def["meta_data"]["name"]] = col_metric

        # save dict as json
        with open(os.path.join(self.args.results_dir, "solvability.json"), 'w') as f:
            json.dump(solvability, f, indent=4)

        return solvability

    def parse_summary_dir(self):
        """
            Parse the records directory and gather the
            results and records JSON file paths from
            the "RouteScenario_*_to_*" directories.
        """
        route_scenario_dirs = sorted(
            glob.glob(
                self.args.results_dir + "/**/RouteScenario_*", recursive=True
            ),
            key=lambda path: int(path.split("_")[-1]),
        )

        # gather all route results and scenario records JSON files
        results_files = []
        records_files = []
        for dir in route_scenario_dirs:
            results_files.extend(
                sorted(
                    glob.glob(dir + "/results.json")
                )
            )
            records_files.extend(
                sorted(
                    glob.glob(dir + "/scenario_records.json")
                )
            )
        return records_files, results_files

    def parse_json_file(self, records_file):
        """
        """
        return json.loads(open(records_file).read())

    def parse_route(self, route_id):
        """
        """
        route_config = self.route_indexer._configs_dict[route_id]

        gps_route, route = interpolate_trajectory(
            self.simulator.carla_wrapper.world, route_config.trajectory
        )

        return [gps_route], [route], [route_config]


def main(args):
    engine = SolvabilityEngine(args)
    solvability_dict = engine.run_solvability_check()
    collision_rate = sum([route for route in solvability_dict.values()])/len(solvability_dict)
    solvability = (1 - collision_rate) * 100
    print(f"Solvability for the provided scenarios: {solvability:.2f}.")
    save_path = os.path.join(args.results_dir, "solvability.json")
    print(f"Per-route results have been saved to \"{save_path}\".")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Compute target.'
    )
    parser.add_argument(
        "--max_GPU_hours",
        default=0.05,
    )
    parser.add_argument(
        "--optim_method",
        default="Adam",
        choices=["Adam", "Both_Paths"]
    )

    # KING arguments, as expected by proxy simulator
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The number of parallel simulations."
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=4,
        help="The number of other agents in the simulation."
    )
    parser.add_argument(
        "--sim_tickrate",
        type=int,
        default=4,
        help="Inverse of the delta_t between subsequent timesteps of the simulation."
    )
    parser.add_argument(
        "--sim_horizon",
        type=int,
        default=80,
        help="The number of timesteps to run the simulation for."
    )
    parser.add_argument(
        "--renderer_class",
        type=str,
        default='STN',
        choices=['STN', "CARLA"],
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="Carla port."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./outputs/agents_4",
        help="The directory containing the scenario records and results files for each of the \
             optimized scenarios/routes.",
    )
    parser.add_argument(
        "--routes_file",
        type=str,
        default="leaderboard/data/routes/train_all.xml",
        help="Path to the .xml file describing the routes to be driven."
    )
    parser.add_argument(
        "--routes_file_adv",
        type=str,
        default="leaderboard/data/routes/adv_all.xml",
        help="Path to the .xml file describing the routes to be driven."
    )
    parser.add_argument(
        "--init_root",
        type=str,
        default="driving_agents/king/aim_bev/king_initializations/initializations_subset",
        help="Path to the scenario initalization files for this agent."
    )
    parser.add_argument(
        "--ego_agent",
        type=str,
        default='aim-bev',
        choices=['aim-bev', 'transfuser'],
        help="The agent under test."
    )
    parser.add_argument(
        "--ego_agent_ckpt",
        type=str,
        default="driving_agents/king/aim_bev/model_checkpoints/regular",
        help="Path to the model checkpoint for the agent under test."
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=0.0
    )

    args = parser.parse_args()

    # reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    main(args)
