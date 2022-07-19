import glob
import json
import os
import torch
from tqdm import tqdm

from srunner.tools.route_manipulation import interpolate_trajectory
from leaderboard.utils.route_indexer import RouteIndexer

from proxy_simulator.motion_model import BicycleModel
from proxy_simulator.simulator import ProxySimulator


class RobustTrainingEngine:
    def __init__(self, args, ego_agent=None, ego_expert=None):
        """
        """
        self.args = args
        self.town = None

        self.simulator = ProxySimulator(
            self.args,
            ego_policy=ego_agent,
            ego_expert=ego_expert,
            motion_model=BicycleModel(1/self.args.sim_tickrate).to(self.args.device),
        )

        self.records_files, self.results_files = self.parse_summary_dir()
        self.route_indexer = RouteIndexer(self.args.routes_file, None, 1)

    def collect_data(self):
        """
        """
        for records_idx, records in enumerate(tqdm(self.records_files)):
            # parse scenario definition from records file
            records = self.records_files[records_idx]
            scenario_def = self.parse_json_file(records)

            scenario_metrics = self.parse_json_file(
                self.results_files[records_idx]
            )

            critical_iteration = scenario_metrics["first_metrics"]["iteration"]

            town = scenario_def["meta_data"]["town"]
            name = scenario_def["meta_data"]["name"]

            # if this is a new town we re-initialize both simulators
            if town != self.town:
                self.simulator.set_new_town(self.args, town)
                self.town = town

            self.simulator.renderer.reset()
            self.simulator.ego_expert.reset()

            # fetch initial state and map to tensors
            initial_state = scenario_def["states"][critical_iteration][0]
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
                if torch.any(self.simulator.is_terminated):
                    break

                if self.args.renderer_class == 'CARLA':
                    try:
                        self.simulator.renderer.initialize_carla_state(
                            self.simulator.get_ego_state(),
                            self.simulator.get_adv_state(),
                            self.args.diverse_actors,
                        )
                    except RuntimeError as e:
                        print(e, f"Happened in route {name}. Skipping...")
                        continue

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
                to_tn = lambda x: torch.tensor(x).float().to(self.args.device)
                adv_actions = scenario_def['adv_actions'][critical_iteration][t]
                adv_actions = {k: to_tn(v) for k, v in adv_actions.items()}

                expert_actions = self.simulator.ego_expert.run_step(
                    input_data, self.simulator, adv_actions=adv_actions
                )

                self.simulator.run_termination_checks()

                self.simulator.step(expert_actions, adv_actions)

            self.simulator.renderer.reset()

    def run_king_eval(self, epoch=-1):
        """
        """
        avg_collision_metric = 0
        total_routes = 0
        for records_idx, records in enumerate(tqdm(self.records_files)):
            # parse scenario files
            scenario_def = self.parse_json_file(records)
            scenario_metrics = self.parse_json_file(
                self.results_files[records_idx]
            )

            critical_iteration = scenario_metrics["first_metrics"]["iteration"]
            time_of_termination = scenario_metrics["first_metrics"]["Time of Termination"]

            town = scenario_def["meta_data"]["town"]
            name = scenario_def["meta_data"]["name"]

            # if this is a new town we re-initialize
            if town != self.town:
                self.simulator.set_new_town(self.args, town)
                self.town = town

            self.simulator.renderer.reset()
            self.simulator.ego_policy.reset()

            # fetch initial state and map to tensors
            initial_state = scenario_def["states"][critical_iteration][0]
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
                # we test three seconds beyond the original time of collision
                if t > time_of_termination + self.args.sim_tickrate * 3:
                    break

                # get imu and compass sensor readings for ego
                input_data = self.simulator.get_ego_sensor()

                # fetch observations for ego agent
                observations, _ = self.simulator.renderer.get_observations(
                    semantic_grid,
                    self.simulator.get_ego_state(),
                    self.simulator.get_adv_state(),
                )
                input_data.update(observations)

                # update input data dict
                input_data.update({
                    "timestep": t
                })

                # fetch actions from ego for current observations
                ego_actions = self.simulator.ego_policy.run_step(
                    input_data, self.simulator
                )

                to_tn = lambda x: torch.tensor(x).float().to(self.args.device)
                adv_actions = scenario_def['adv_actions'][critical_iteration][t]
                adv_actions = {k: to_tn(v) for k, v in adv_actions.items()}

                self.simulator.run_termination_checks()

                self.simulator.step(ego_actions, adv_actions)

            collisions = self.simulator.ego_collision[self.simulator.ego_collision == 1.]
            col_metric = len(collisions) / self.args.batch_size

            avg_collision_metric += col_metric
            total_routes += 1

            if not os.path.exists(os.path.join(self.args.logdir, "king_eval", f"epoch_{epoch}")):
                os.makedirs(os.path.join(self.args.logdir, "king_eval", f"epoch_{epoch}"))

            state_records = []
            for t in self.simulator.state_buffer:
                state_per_t = {"pos": None, "yaw": None, "vel": None}
                for key in t.keys():
                    state_per_t[key] = t[key][0].cpu().tolist()
                state_records.append(state_per_t)

            ego_actions_records = []
            for t in self.simulator.ego_action_buffer:
                actions_per_t = {"steer": None, "throttle": None, "brake": None}
                for key in t.keys():
                    actions_per_t[key] = t[key][0].cpu().tolist()
                ego_actions_records.append(actions_per_t)

            adv_actions_records = []
            for t in self.simulator.adv_action_buffer:
                actions_per_t = {"steer": None, "throttle": None, "brake": None}
                for key in t.keys():
                    actions_per_t[key] = t[key][0].cpu().tolist()
                adv_actions_records.append(actions_per_t)

            # dump records json for replay
            scenario_records_val = {
                "meta_data": scenario_def["meta_data"],
                "states": [state_records],
                "ego_actions": [ego_actions_records],
                "adv_actions": [adv_actions_records],
            }

            with open(os.path.join(self.args.logdir, "king_eval", f"epoch_{epoch}", f"{name}_scenario_records.json"), "w") as f:
                json.dump(scenario_records_val, f)

        return float(avg_collision_metric)/total_routes

    def parse_summary_dir(self):
        """
            Parse the records directory and gather the
            results and records JSON file paths from
            the "RouteScenario_*_to_*" directories.
        """
        route_scenario_dirs = sorted(
            glob.glob(
                self.args.scenario_summary_dir + "/**/RouteScenario_*", recursive=True
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

        return [gps_route], [route], [route_config]
