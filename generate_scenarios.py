import os
import argparse
import torch
import cv2
import os
import json
import copy
from tqdm.auto import trange
import random
import carla
from pathlib import Path

from PIL import ImageShow
import numpy as np
import torch.nn.functional as F

from driving_agents.king.expert.expert_agent import AutoPilot
from driving_agents.king.transfuser.transfuser_agent import TransFuserAgent
from driving_agents.king.aim_bev.aim_bev_agent import AimBEVAgent
from leaderboard.utils.route_indexer import RouteIndexer
from srunner.tools.route_manipulation import interpolate_trajectory

from proxy_simulator.simulator import ProxySimulator
from proxy_simulator.motion_model import BicycleModel
from proxy_simulator.utils import save_args
from proxy_simulator.bm_policy import BMActionSequence
from proxy_simulator.driving_costs import RouteDeviationCostRasterized, BatchedPolygonCollisionCost


# Global Flags
PIXELS_PER_METER = 5
PIXELS_AHEAD_VEHICLE = 110


class GenerationEngine:
    """Engine that controls the differentiable simulator.

    Args
        clargs (Namespace): The arguments parsed from the command line.
    """
    def __init__(self, args):
        # MISC #
        self.args = args

        # DRIVING AGENTS #
        adv_policy = BMActionSequence(
            self.args,
            self.args.batch_size,
            self.args.num_agents,
            self.args.sim_horizon,
        )

        if args.ego_agent == 'aim-bev':
            ego_policy = AimBEVAgent(
                self.args,
                device=args.device,
                path_to_conf_file=args.ego_agent_ckpt
            )
        elif args.ego_agent == 'transfuser':
            ego_policy = TransFuserAgent(
                self.args,
                device=args.device,
                path_to_conf_file=args.ego_agent_ckpt
            )

        # SIMULATOR #
        self.simulator = ProxySimulator(
            self.args,
            ego_policy = ego_policy,
            ego_expert = AutoPilot(self.args, device=args.device),
            adv_policy = adv_policy.to(self.args.device),
            motion_model=BicycleModel(1/self.args.sim_tickrate).to(self.args.device),
        )

        # COSTS
        self.rd_cost_fn_rasterized = RouteDeviationCostRasterized(self.args)
        self.col_cost_fn = BatchedPolygonCollisionCost(self.args)

        # ROUTE SETTING #
        self.route_indexer = RouteIndexer(self.args.routes_file, None, 1)
        if self.args.max_num_routes == -1:
            self.args.max_num_routes = self.route_indexer.total

    def run(self):
        """
        """

        scenario_params = [
            self.simulator.adv_policy.steer,
            self.simulator.adv_policy.throttle,
        ]
        scenario_optim = torch.optim.Adam(scenario_params, lr=self.args.learning_rate, betas=(self.args.beta1, self.args.beta2),)

        route_loop_bar = trange(
            self.route_indexer.total // self.args.batch_size
        )

        all_metrics = []
        first_metric_per_route = []
        for ix, self.route_iter in enumerate(route_loop_bar):
            state_buffers = []
            ego_actions_buffers = []
            adv_actions_buffers = []

            # ROUTE SETUP #
            with torch.no_grad():
                gps_route, route, route_config = self.get_next_route()

            first_metric_per_route.append([])
            self.curr_route_name = [conf.name for conf in route_config]

            # re-initialize ADAM's state for each route
            scenario_optim = torch.optim.Adam(scenario_params, lr=self.args.learning_rate, betas=(self.args.beta1, self.args.beta2),)

            # OPTIMIZATION LOOP FOR CURRENT ROUTE
            opt_loop_bar = trange(self.args.opt_iters, leave=False)
            for i in opt_loop_bar:
                if len(all_metrics) <= i:
                    all_metrics.append([])

                scenario_optim.zero_grad(set_to_none=True)

                with torch.no_grad():
                    self.simulator.set_route(gps_route, route, route_config)
                    self.simulator.renderer.reset()
                    self.simulator.ego_policy.reset()
                    self.simulator.ego_expert.reset()

                cost_dict, num_oob_per_t = self.unroll_simulation()

                # aggregate costs and build total objective
                cost_dict["ego_col"] = torch.min(
                    torch.mean(
                        torch.stack(cost_dict["ego_col"], dim=1),
                        dim=1,
                    ),
                    dim=1,
                )[0]
                cost_dict["adv_col"] = torch.min(
                    torch.min(
                        torch.stack(cost_dict["adv_col"], dim=1),
                        dim=1,
                    )[0],
                    dim=1,
                )[0]
                cost_dict["adv_rd"] = torch.mean(
                    torch.stack(cost_dict["adv_rd"], dim=1),
                    dim=1,
                )
                total_objective = sum([
                    self.args.w_ego_col * cost_dict["ego_col"].mean(),
                    self.args.w_adv_rd * cost_dict["adv_rd"].mean(),
                    -1*self.args.w_adv_col * cost_dict["adv_col"].mean()
                ])

                collisions = self.simulator.ego_collision[self.simulator.ego_collision == 1.]
                col_metric = len(collisions) / self.args.batch_size

                if col_metric != 1.0:
                    total_objective.backward()
                    scenario_optim.step()

                #### BUFFERS ###
                state_buffers.append(self.simulator.state_buffer)
                ego_actions_buffers.append(self.simulator.ego_action_buffer)
                adv_actions_buffers.append(self.simulator.adv_action_buffer)

                cumulative_oob = torch.sum(num_oob_per_t, dim=1) / self.args.sim_horizon
                mean_cumulative_oob = torch.mean(cumulative_oob)
                oob_fraction = torch.sum(torch.gt(num_oob_per_t, 0), dim=1) / self.args.sim_horizon
                mean_oob_fraction = torch.mean(oob_fraction)

                opt_loop_bar.set_postfix({
                    "Total Cost": total_objective.detach().cpu().item() * -1,
                    "Col.": col_metric,
                    "OOB": mean_cumulative_oob.detach().cpu().item(),
                })

                log = {
                    "Loss": total_objective.item(),
                }

                log.update({key: torch.mean(value).cpu().item() for (key, value) in cost_dict.items()})
                log.update({'Collision Metric': col_metric})
                log.update({'Time of Termination': self.simulator.tot.float().mean(dim=0).cpu().item()})
                log.update({'Cumulative OOB': mean_cumulative_oob.item()})
                log.update({'Time spent OOB': mean_oob_fraction.item()})
                log.update({'adv_collision': self.simulator.adv_collision.tolist()[0]})
                log.update({'iteration': i})

                if col_metric == 1:
                    first_metric_per_route[-1].append(log)

                # in case we have no collision
                if i + 1 == self.args.opt_iters and len(first_metric_per_route[-1]) == 0:
                    first_metric_per_route[-1].append(log)

                all_metrics[i].append(log)

                if col_metric == 1:
                    break

            # prepare and save results of route
            for batch_idx in range(self.args.batch_size):
                # make buffers json dumpable
                # nested lists of opt_iter and timestep
                state_records = []
                for opt_iter, curr_buffer in enumerate(state_buffers):
                    states_per_opt_iter = []
                    for t in curr_buffer:
                        state_per_t = {"pos": None, "yaw": None, "vel": None}
                        for key in t.keys():
                            state_per_t[key] = t[key][batch_idx].cpu().tolist()
                        states_per_opt_iter.append(state_per_t)
                    state_records.append(states_per_opt_iter)

                ego_actions_records = []
                for opt_iter, curr_buffer in enumerate(ego_actions_buffers):
                    actions_per_opt_iter = []
                    for t in curr_buffer:
                        actions_per_t = {"steer": None, "throttle": None, "brake": None}
                        for key in t.keys():
                            actions_per_t[key] = t[key][batch_idx].cpu().tolist()
                        actions_per_opt_iter.append(actions_per_t)
                    ego_actions_records.append(actions_per_opt_iter)

                adv_actions_records = []
                for opt_iter, curr_buffer in enumerate(adv_actions_buffers):
                    actions_per_opt_iter = []
                    for t in curr_buffer:
                        actions_per_t = {"steer": None, "throttle": None, "brake": None}
                        for key in t.keys():
                            actions_per_t[key] = t[key][batch_idx].cpu().tolist()
                        actions_per_opt_iter.append(actions_per_t)
                    adv_actions_records.append(actions_per_opt_iter)

                # assemble results dict and dump to json
                meta_data = {
                    "name": route_config[batch_idx].name,
                    "index": route_config[batch_idx].index,
                    "town": route_config[batch_idx].town,
                    "Num_agents": args.num_agents
                }

                scenario_records = {
                    "meta_data": meta_data,
                    "states": state_records,
                    "ego_actions": ego_actions_records,
                    "adv_actions": adv_actions_records,
                }

                route_results = {
                    "meta_data": meta_data,
                    "is_terminated": self.simulator.is_terminated.tolist()[batch_idx],
                    "tot": self.simulator.tot.tolist()[batch_idx],
                    "adv_collision": self.simulator.adv_collision.tolist()[batch_idx],
                    "adv_rel_pos_at_collision": self.simulator.adv_rel_pos_at_collision.tolist()[batch_idx],
                    "adv_rel_yaw_at_collision": self.simulator.adv_rel_yaw_at_collision.tolist()[batch_idx],
                }
                route_results.update(log)

                route_results.update({
                    "first_metrics": first_metric_per_route[-1][-1],
                    "all_iterations": {str(iter_index): all_metrics[iter_index][-1] for iter_index in range(len(all_metrics))}
                })

                # dump route results
                delim = "_"
                route_results_path = \
                    f"{args.save_path}/{self.curr_route_name[0]}_to_{self.curr_route_name[-1].split(delim)[-1]}/results.json"

                if not os.path.exists(os.path.dirname(route_results_path)):
                    os.makedirs(os.path.dirname(route_results_path))

                with open(route_results_path, "w") as f:
                    json.dump(route_results, f, indent=4)

                # dump route scenario records
                scenario_records_path = \
                    f"{args.save_path}/{self.curr_route_name[0]}_to_{self.curr_route_name[-1].split(delim)[-1]}/scenario_records.json"

                if not os.path.exists(os.path.dirname(scenario_records_path)):
                    os.makedirs(os.path.dirname(scenario_records_path))

                with open(scenario_records_path, "w") as f:
                    json.dump(scenario_records, f)

            # check if were done and break if yes
            if self.route_indexer._index >= self.args.max_num_routes:
                break

        for iter_ix, iter_dicts in enumerate(all_metrics): # opt iters
            new_dict = {}
            for key, value in iter_dicts[0].items():
                new_dict.update({key: []})
                for iter_dict in iter_dicts: # routes
                    new_dict[key].append(iter_dict[key])

            results = {f'{key}_all': np.asarray(value).mean() for key, value in new_dict.items()}
            results.update({'step': iter_ix})

        tmp_already_collided = {}
        for iter_ix, iter_dicts in enumerate(all_metrics):
            new_dict = {}
            for key, value in iter_dicts[0].items():
                if 'Collision' not in key:
                    continue
                new_dict.update({key: []})
                for route_ix, iter_dict in enumerate(iter_dicts):
                    if iter_dict['Collision Metric']==1 and f'{route_ix}' not in tmp_already_collided:
                        tmp_already_collided[f'{route_ix}'] = iter_dict #[key]

                    if f'{route_ix}' in tmp_already_collided:
                        new_dict[key].append(tmp_already_collided[f'{route_ix}'][key])
                    else:
                        new_dict[key].append(iter_dict[key])

            results = {f'{key}_cum': np.asarray(value).mean() for key, value in new_dict.items()}
            results.update({'step': iter_ix})

        new_dict = {}
        for iter_ix, route_res in enumerate(first_metric_per_route):
            for key, value in route_res[-1].items():
                if key not in new_dict:
                    new_dict.update({key: []})
                new_dict[key].append(route_res[-1][key])

        results = {f'{key}_first': np.asarray(value).mean() for key, value in new_dict.items()}

    def unroll_simulation(self):
        """
        Simulates one episode of length `args.sim_horizon` in the differentiable
        simulator.
        """
        # initializations
        semantic_grid = self.simulator.map
        cost_dict = {"ego_col": [], "adv_rd": [], "adv_col": []}

        num_oob_agents_per_t = []
        if self.args.renderer_class == 'CARLA':
            self.simulator.renderer.initialize_carla_state(
                self.simulator.get_ego_state(),
                self.simulator.get_adv_state(),
                town=self.town,
            )

            rgb_per_t = []
            observations_per_t = []
            lidar_per_t1 = []
            lidar_per_t2 = []

        for t in range(self.args.sim_horizon):
            input_data = self.simulator.get_ego_sensor()
            input_data.update({"timestep": self.simulator.timestep})

            observations, _ = self.simulator.renderer.get_observations(
                semantic_grid,
                self.simulator.get_ego_state(),
                self.simulator.get_adv_state(),
            )
            input_data.update(observations)

            ego_actions = self.simulator.ego_policy.run_step(
                input_data, self.simulator
            )

            if self.args.detach_ego_path:
                ego_actions["steer"] = ego_actions["steer"].detach()
                ego_actions["throttle"] = ego_actions["throttle"].detach()
                ego_actions["brake"] = ego_actions["brake"].detach()

            adv_actions = self.simulator.adv_policy.run_step(
                input_data
            )

            num_oob_agents = self.simulator.run_termination_checks()
            num_oob_agents_per_t.append(num_oob_agents)

            ego_col_cost, adv_col_cost, adv_rd_cost = self.compute_cost()

            cost_dict["adv_rd"].append(adv_rd_cost)
            cost_dict["adv_col"].append(adv_col_cost)
            cost_dict["ego_col"].append(ego_col_cost)

            # compute next state given current state and actions
            self.simulator.step(ego_actions, adv_actions)

        # stack timesteps for oob metric
        num_oob_agents_per_t = torch.stack(num_oob_agents_per_t, dim=1)

        torch.cuda.empty_cache()
        return cost_dict, num_oob_agents_per_t

    def compute_cost(self):
        """
        """
        ego_state = self.simulator.get_ego_state()
        adv_state = self.simulator.get_adv_state()

        ego_col_cost, adv_col_cost, _ = self.col_cost_fn(
            ego_state,
            self.simulator.ego_extent,
            adv_state,
            self.simulator.adv_extent,
        )

        if adv_col_cost.size(-1) == 0:
            adv_col_cost = torch.zeros(1,1).cuda()
            assert adv_col_cost.size(0) == 1, 'This works only for batchsize 1!'

        adv_col_cost = torch.minimum(
            adv_col_cost, torch.tensor([self.args.adv_col_thresh]).float().cuda()
        )

        adv_rd_cost = self.rd_cost_fn_rasterized(
            self.simulator.map[0, 0, :, :], adv_state['pos'], adv_state['yaw'],
            ego_state['pos'], self.simulator.renderer.world_to_pix
        )

        return ego_col_cost, adv_col_cost, adv_rd_cost

    def get_next_route(self):
        """
        Fetches the next batch of routes from the route iterator and returns
        them as a list.
        """
        route_configs = []
        gps_routes = []
        routes = []
        route_config = None
        for idx in range(self.args.batch_size):
            route_config = self.route_indexer.next()

            route_configs.append(route_config)
            self.town = route_config.town

            if not hasattr(self, "current_town"):
                self.simulator.set_new_town(self.args, self.town)
                self.current_town = self.town
            elif self.town != self.current_town:
                self.simulator.set_new_town(self.args, self.town)
                self.current_town = self.town

            # this try/except guards for the carla server having died in the background
            # for this to work carla needs to be run in a cronjob that relaunches it if
            # it terminates unexpectedly.
            try:
                gps_route, route = interpolate_trajectory(
                    self.simulator.carla_wrapper.world, route_config.trajectory
                )
            except RuntimeError:
                self.simulator.carla_wrapper._initialize_from_carla(town=self.town, port=self.args.port)
                gps_route, route = interpolate_trajectory(
                    self.simulator.carla_wrapper.world, route_config.trajectory
                )

            gps_routes.append(gps_route)
            routes.append(route)

        return gps_routes, routes, route_configs


def main(args):
    engine = GenerationEngine(args)
    engine.run()


if __name__ == '__main__':
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    main_parser.add_argument(
        "--save_path",
        type=str,
        default="./outputs",
        help="Directory to write generation results to."
    )
    main_parser.add_argument(
        "--seed",
        type=int,
        default=10,
    )
    main_parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The number of parallel simulations."
    )
    main_parser.add_argument(
        "--max_num_routes",
        type=int,
        default=-1,
        help="The maximum number of routes from the routes_file \
            to optimize over. Set to -1 to optimize over all of them.",
    )
    main_parser.add_argument(
        "--opt_iters",
        type=int,
        default=151,
        help="The number of optimization steps to perform.",
    )
    main_parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.005,
    )
    main_parser.add_argument(
        "--num_agents",
        type=int,
        default=4,
        help="The number of other agents in the scene."
    )
    main_parser.add_argument(
        "--sim_tickrate",
        type=int,
        default=4,
        help="Inverse of the delta_t between subsequent timesteps of the \
              simulation."
    )
    main_parser.add_argument(
        "--sim_horizon",
        type=int,
        default=80,
        help="The number of timesteps to run the simulation for."
    )
    main_parser.add_argument(
        "--renderer_class",
        type=str,
        default='STN',
        choices=['STN', 'CARLA'],
    )
    main_parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="Carla port."
    )
    main_parser.add_argument(
        "--routes_file",
        type=str,
        default="leaderboard/data/routes/subset_20perTown.xml",
        help="Path to the .xml file describing the routes to be driven."
    )
    main_parser.add_argument(
        "--routes_file_adv",
        type=str,
        default="leaderboard/data/routes/adv_all.xml",
        help="Path to the .xml file describing the routes to be driven."
    )
    main_parser.add_argument(
        "--ego_agent",
        type=str,
        default='aim-bev',
        choices=['aim-bev', 'transfuser'],
        help="The agent under test."
    )
    main_parser.add_argument(
        "--ego_agent_ckpt",
        type=str,
        default="driving_agents/king/aim_bev/model_checkpoints/regular",
        help="Path to the model checkpoint for the agent under test."
    )
    main_parser.add_argument(
        "--gradient_clip",
        type=float,
        default=0.
    )
    main_parser.add_argument(
        "--detach_ego_path",
        type=int,
        default=1
    )
    main_parser.add_argument(
        "--w_ego_col",
        type=float,
        default=1
    )
    main_parser.add_argument(
        "--w_adv_col",
        type=float,
        default=0
    )
    main_parser.add_argument(
        "--adv_col_thresh",
        type=float,
        default=1.25
    )
    main_parser.add_argument(
        "--w_adv_rd",
        type=float,
        default=1
    )
    main_parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
    )
    main_parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
    )
    main_parser.add_argument(
        '--king_data_fps',
        type=int,
        default=2,
        help='Unique experiment identifier.'
    )
    main_parser.add_argument(
        "--init_root",
        type=str,
        default="driving_agents/king/aim_bev/king_initializations/initializations_subset/",
        help="Path to the scenario initalization files for the current agent and routes",
    )

    args = main_parser.parse_args()

    # reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args.save_path = f'{args.save_path}'

    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    save_args(args, args.save_path)

    main(args)
