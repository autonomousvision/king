import torch
import carla
import json

import numpy as np
import xml.etree.ElementTree as ET

from srunner.scenarioconfigs.route_scenario_configuration import RouteScenarioConfiguration
from proxy_simulator.carla_wrapper import CarlaWrapper
from proxy_simulator.collision import check_collision
from proxy_simulator.renderer import STNRenderer, CARLARenderer
import proxy_simulator.utils


# Global Flags
PIXELS_PER_METER = torch.tensor([5], device='cuda', dtype=torch.float32)


class ProxySimulator:
    """
    """
    def __init__(self, args, ego_policy=None, ego_expert=None, adv_policy=None, motion_model=None):
        """
        """
        # MISC #
        self.timestep = 0
        self.args = args

        # pre-initialized for faster later construction of corners
        # of agent bounding boxes
        unit_square = torch.tensor(
            [
                [ 1.,  1.],  # back right corner
                [-1.,  1.],  # back left corner
                [-1., -1.],  # front left corner
                [ 1., -1.],  # front right corner
            ],
            device=args.device,
        ).view(1, 1, 4, 2)
        self.unit_square = unit_square.expand(
            self.args.batch_size, self.args.num_agents+1, 4, 2
        )

        #  CARLA INTERFACE #
        self.carla_wrapper = CarlaWrapper(args)

        # STATIC WORLD #
        self.map, self.map_offset = self.carla_wrapper._initialize_from_carla(port=args.port)
        self.map_dims = self.map.shape[2:4]

        # conversion constants for gps coordinates
        self.gps_centroid = torch.tensor(
            [0.0, 0.0], device=self.args.device, dtype=torch.float32
        )
        self.gps_scale = torch.tensor(
            [111324.60662786, 111319.490945], device=self.args.device, dtype=torch.float32
        )
        self.gps_axis_flip = torch.tensor(
            [[[0, 1], [-1, 0]]], device=self.args.device, dtype=torch.float32
        ).expand(self.args.batch_size, -1, -1)

        # NAVIGATION/ROUTE HANDLING #
        self.routes_per_town = parse_routes(self.carla_wrapper.world, self.args.routes_file_adv)
        # current ego route
        self.gps_route = []
        self.route = []
        self.route_config = []
        # current adversarial routes
        self.adv_spawn_points = []  # list of lists (batch and agent id)
        self.adv_routes = []  # list of lists (batch and agent id)
        self.adv_routes_gps = []  # list of lists (batch and agent id)

        # DYMAMIC WORLD #
        # The dyn. state of the simulation is represented as a dict of tensors.
        # Each tensor is of shape BxNxS, where B is the batch dimension,
        # N is the agent dimension and S is the state dimension. This is of
        # size 2 for pos and vel, and of size 1 for yaw.
        self.state = {"pos": None, "vel": None, "yaw": None}
        self.corners = None  # corners of all agents' bounding boxes
        self.vehicle_bounding_box_extent = carla.Vector3D(2.20, .90, .755)

        self.state_buffer = []
        self.ego_action_buffer = []
        self.adv_action_buffer = []

        self.ego_policy = ego_policy
        self.ego_expert = ego_expert
        self.adv_policy = adv_policy

        # DYNAMICS MODEL #
        self.motion_model = motion_model

        # RENDERING FUNCTION #
        if self.args.renderer_class == "STN":
            self.renderer_constructor = STNRenderer
        elif self.args.renderer_class == "CARLA":
            self.renderer_constructor = CARLARenderer
        else:
            raise ValueError

        self.renderer = self.renderer_constructor(self.args, self.map_offset, self.map_dims, viz=False)

    def set_new_town(self, args, town):
        """
        Re-initializes static world for new town. Also re-initializes CARLA renderer, if in use.
        """
        self.carla_wrapper = CarlaWrapper(args)
        self.map, self.map_offset = self.carla_wrapper._initialize_from_carla(town=town, port=args.port)
        self.map_dims = self.map.shape[2:4]
        self.gps_route = []
        self.route = []
        self.renderer = self.renderer_constructor(self.args, self.map_offset, self.map_dims, viz=False)
        if self.args.renderer_class == "CARLA":
            self.renderer.attach_carla_wrapper(self.carla_wrapper)

    def set_route(self, gps_route, route, route_config):
        """
        Sets the route to be driven in the current simulation. Also initializes
        the dynamic state of the simulator.
        """
        # some sanity checks
        assert isinstance(gps_route, list) and isinstance(route, list)
        assert len(gps_route) == len(route) == self.args.batch_size

        # reset timestep
        self.timestep = 0

        # reset state buffer
        self.state_buffer = []
        self.ego_action_buffer = []
        self.adv_action_buffer = []

        # we check if the route id has changed
        # assuming the first batch element is representative of all
        # if this is the firt route we set the previous route id to -1
        try:
            prev_route_id = self.route_config[0].name
        except IndexError:
            prev_route_id = -1
        self.gps_route, self.route, self.route_config = gps_route, route, route_config
        longest_route = max([len(r) for r in self.route])

        if prev_route_id == route_config[0].name:
            self._initialize_dynamic_world()
            # re-initialize route in agents
            self.ego_expert.set_global_plan(self.gps_route, self.route)
            self.ego_policy.set_global_plan(self.gps_route, self.route)
        else:
            # FETCH AND SET ROUTES AND SPAWNPOINTS
            self.adv_routes = []
            self.adv_routes_gps = []
            self.adv_spawn_points = []
            for ix in range(self.args.batch_size):
                with open(f"{self.args.init_root}/{self.args.num_agents}_agents/{route_config[ix].name}.json") as f:
                    init_data = json.load(f)
                    f.close()
                adv_routes_serialized = init_data["adv_routes"]
                adv_routes_gps_serialized = init_data["adv_routes_gps"]
                adv_routes = []
                adv_routes_gps = []
                assert len(adv_routes_serialized) == len(adv_routes_gps_serialized)
                for i, route in enumerate(adv_routes_serialized):
                    adv_routes.append([])
                    adv_routes_gps.append([])
                    for t in range(len(adv_routes_serialized[i])):
                        wp_transform = carla.Transform(
                            carla.Location(
                                x=adv_routes_serialized[i][t]["x"],
                                y=adv_routes_serialized[i][t]["y"],
                                z=adv_routes_serialized[i][t]["z"],
                            ),
                            carla.Rotation(
                                yaw=adv_routes_serialized[i][t]["yaw"]
                            )
                        )
                        road_option = proxy_simulator.utils.roadopt_from_int(adv_routes_serialized[i][t]["road_option"])

                        adv_routes[-1].append((wp_transform, road_option))
                        adv_routes_gps[-1].append((adv_routes_gps_serialized[i][t], road_option))
                    if len(route) > longest_route:
                        longest_route = len(route)

                self.adv_routes.append(adv_routes)
                self.adv_routes_gps.append(adv_routes_gps)
                adv_spawn_points = []
                for sp in init_data["adv_spawn_points"]:
                    sp_transform = carla.Transform(
                        carla.Location(
                            x=sp["x"],
                            y=sp["y"],
                            z=sp["z"],
                        ),
                        carla.Rotation(
                            yaw=sp["yaw"]
                        )
                    )
                    adv_spawn_points.append(sp_transform)
                self.adv_spawn_points.append(adv_spawn_points)
            self._initialize_dynamic_world()
            # FETCH AND SET ACTION INIT
            if self.adv_policy:
                action_seqs = []
                for ix in range(self.args.batch_size):
                    with open(f"{self.args.init_root}/{self.args.num_agents}_agents/{route_config[ix].name}.json") as f:
                        init_data = json.load(f)
                        f.close()
                    for action_seq in init_data["action_seq"]:
                        for timestep in action_seq:
                            timestep['throttle'] = torch.tensor(
                                timestep['throttle'], device=self.args.device,
                            )
                            timestep['steer'] = torch.tensor(
                                timestep['steer'], device=self.args.device,
                            )
                            timestep['brake'] = torch.tensor(
                                timestep['brake'], device=self.args.device,
                            )
                    action_seqs.append(init_data["action_seq"])

                self.adv_policy.initialize_non_critical_actions(action_seqs)
                del action_seqs

            if self.ego_expert:
                self.ego_expert.set_global_plan(self.gps_route, self.route)
            if self.ego_policy:
                self.ego_policy.set_global_plan(self.gps_route, self.route)

    def get_ego_sensor(self, adv=None):
        """
        Get measurements as expected by CARLA agents.
        """
        ego_sensor = {}
        if adv is not None:
            ego_state = adv

        else:
            ego_state = self.get_ego_state()

        pos = ego_state['pos'] @ self.gps_axis_flip

        gps = pos / self.gps_scale - self.gps_centroid
        compass = ego_state['yaw'] + np.pi/2

        # velocity to speed
        velocity = ego_state['vel']
        velocity = torch.stack((velocity[:,:,0], velocity[:,:,1], velocity[:,:,1]*0.0), axis=-1).squeeze()
        speed = torch.norm(velocity.unsqueeze(0), dim=-1).unsqueeze(0).unsqueeze(0)

        if adv is not None:
            ego_sensor = {
                'gps': gps[:1],
                'imu': compass[:1],
                'speed': speed[:1],
            }
        else:
            ego_sensor = {
                'gps': gps,
                'imu': compass,
                'speed': speed,
            }

        return ego_sensor

    def get_ego_state(self):
        """
        Fetches the ego state.

        Returns:
            A dictionary holding tensors of shape B x 1 x S for each substate,
            where B is the batch size and S is the dimensionality of the substate.
        """
        ego_state = {}
        for substate in self.state.keys():
            ego_state.update({substate: self.state[substate][:, 0:1, ...]})
        return ego_state

    def set_ego_state(self, new_state):
        """
        Sets a new ego state.

        Arguments:
            new_state (dict of tensors): Dictionary holding tensors of shape
                B x 1 x S for each substate of the ego state to be updated,
                where B is the batch size and S is the dimensionality of the
                substate.

        Returns:
            None
        """
        for substate in self.state.keys():
            self.state[substate] = torch.cat(
                [new_state[substate], self.state[substate][:, 1:, ...]],
                dim=1
            )

    def get_adv_state(self, id=None):
        """
        Fetches the adversarial agent state. If id is None, the state for all
        adversarial agents is returned. If the id is not None, only the state of
        the corresponding agent is returned.

        Arguments:
            id (int): Index of a specific adversarial agent to be returned.

        Returns:
            A dictionary holding tensors of shape B x N x S for each substate,
            where B is the batch size, N is the number of adversarial agents
            and S is the dimensionality of the substate. If id is not None, N=1.
        """
        adv_state = {}
        for substate in self.state.keys():
            if id == None:
                adv_state.update({substate: self.state[substate][:, 1:, ...]})
            else:
                # we index for id+1 since id 0 in the tensor is the ego agent
                adv_state.update(
                    {substate: self.state[substate][:, id+1:id+2, ...]}
                )
        return adv_state

    def set_adv_state(self, new_state, id=None):
        """
        Set a new adversarial agent state. If id is None, the state for all
        adversarial agents is updated. If the id is not None, only the state
        of the corresponding agent is updated.

        Arguments:
            new_state (dict of tensors): Dictionary holding tensors of shape
                B x N x S for each substate of the adversarial agent states
                to be updated, where B is the batch size, N is the number of
                adversarial agents and S is the dimensionality of the substate.
                If id is not None, N=1.
            id (int): Index of a specific adversarial agent to be returned.

        Returns:
            None
        """

        for substate in self.state.keys():
            if not id:
                self.state[substate] = torch.cat(
                    [self.state[substate][:, 0:1, ...], new_state[substate]],
                    dim=1
                )
            else:
                # we index for id+1 since id 0 in the tensor is the ego agent
                self.state[substate][:, id+1:id+2, ...] = self.cat(
                    [
                        self.state[substate][:, 0:id+1, ...],
                        self.state[substate][:, id+1:id+2, ...],
                        self.state[substate][:, id+2:, ...],
                    ],
                    dim=1,
                )

    def _initialize_dynamic_world(self):
        """
        Initialize all dynamic agents and their states
        """
        # A boolean tensor of shape Bx1 indicating wether a simulation has
        # terminated and should no longer be updated.
        self.is_terminated = torch.zeros(
            [self.args.batch_size],
            dtype=torch.bool,
            device=self.args.device,
        )
        # A tensor of shape Bx1 indicating the timestep at which the respective
        # simulation has terminated. A value of -1 indicates the simulation has
        # not yet terminated.
        self.tot = torch.zeros(
            [self.args.batch_size],
            dtype=torch.int16,
            device=self.args.device,
        ) - 1
        self.adv_rel_pos_at_collision = torch.zeros(
            [self.args.batch_size, 2],
            dtype=torch.float32,
            device=self.args.device,
        ) - 1
        self.adv_rel_yaw_at_collision = torch.zeros(
            [self.args.batch_size],
            dtype=torch.float32,
            device=self.args.device,
        ) - 1
        self.ego_collision = torch.zeros(
            [self.args.batch_size],
            dtype=torch.bool,
            device=self.args.device
        )
        self.adv_collision = torch.zeros(
            [self.args.batch_size],
            dtype=torch.bool,
            device=self.args.device
        )

        starting_speed = 4.  # m/s

        # EGO AGENT #
        ego_pos = torch.empty(size=(self.args.batch_size, 1, 2), device=self.args.device, dtype=torch.float32)
        ego_yaw = torch.empty(size=(self.args.batch_size, 1, 1), device=self.args.device, dtype=torch.float32)
        ego_vel = torch.empty(size=(self.args.batch_size, 1, 2), device=self.args.device, dtype=torch.float32)
        for ix, route in enumerate(self.route):
            ego_pos[ix] = torch.tensor(
                [[[route[0][0].location.x, route[0][0].location.y]]],
                device=self.args.device, dtype=torch.float32
            )
            ego_yaw[ix] = torch.tensor(
                [[[route[0][0].rotation.yaw /180 * np.pi]]],
                device=self.args.device, dtype=torch.float32
            )
            ego_vel[ix] = torch.tensor(
                [[[
                    starting_speed * torch.cos(ego_yaw[ix]),
                    starting_speed * torch.sin(ego_yaw[ix]),
                ]]],
                device=self.args.device, dtype=torch.float32
            )

        ego_state = {
            "pos": ego_pos,
            "vel": ego_vel,
            "yaw": ego_yaw,
        }

        # add ego state to global sim state
        for sub_state in self.state.keys():
            self.state[sub_state] = ego_state[sub_state]

        # ADVERSARIAL AGENTS #
        adv_pos = torch.empty(size=(self.args.batch_size, self.args.num_agents, 2), device=self.args.device)
        adv_yaw = torch.empty(size=(self.args.batch_size, self.args.num_agents, 1), device=self.args.device)
        adv_vel = torch.empty(size=(self.args.batch_size, self.args.num_agents, 2), device=self.args.device)

        for ix in range(self.args.batch_size):
            for id in range(self.args.num_agents):
                adv_pos[ix][id] = torch.tensor(
                    [[[self.adv_spawn_points[ix][id].location.x, self.adv_spawn_points[ix][id].location.y]]],
                    device=self.args.device, dtype=torch.float32
                )
                adv_yaw[ix][id] = torch.tensor(
                    [[[self.adv_spawn_points[ix][id].rotation.yaw /180 * np.pi]]],
                    device=self.args.device, dtype=torch.float32
                )
                adv_vel[ix][id] = torch.tensor(
                    [[[
                        starting_speed * torch.cos(adv_yaw[ix][id]),
                        starting_speed * torch.sin(adv_yaw[ix][id]),
                    ]]],
                    device=self.args.device, dtype=torch.float32
                )

        adv_state = {
            "pos": adv_pos,
            "vel": adv_vel,
            "yaw": adv_yaw,
        }

        # add ego state to global sim state
        for sub_state in self.state.keys():
            self.state[sub_state] = torch.cat(
                [self.state[sub_state], adv_state[sub_state]], dim=1
            )

        # add initial state to state buffer
        state_detached = {"pos": None, "yaw": None, "vel": None}
        for substate in self.state.keys():
            state_detached[substate] = self.state[substate].clone().detach()
        self.state_buffer.append(state_detached)

        # set dimensions/extent for all agents
        self.ego_extent = torch.tensor(
            [self.vehicle_bounding_box_extent.x,
             self.vehicle_bounding_box_extent.y,],
            device=self.args.device,
        ).view(1, 1, 2).expand(self.args.batch_size, 1, 2)
        self.adv_extent = torch.tensor(
            [self.vehicle_bounding_box_extent.x,
             self.vehicle_bounding_box_extent.y,],
            device=self.args.device,
        ).view(1, 1, 2).expand(self.args.batch_size, self.args.num_agents, 2)

    @torch.no_grad()
    def run_termination_checks(self):
        """
        """
        self.run_route_completion_check()
        self.run_collision_check()
        num_oob_agents = self.run_oob_check()

        return num_oob_agents

    @torch.no_grad()
    def run_route_completion_check(self):
        """
        """
        for idx, route in enumerate(self.route):
            goal_location = torch.tensor(
                [route[-1][0].location.x, route[-1][0].location.y],
                device=self.args.device,
                dtype=torch.float32
            ).view(1, 2)
            ego_location = self.get_ego_state()["pos"][idx]
            dist_to_goal = torch.linalg.norm(
                (goal_location - ego_location).squeeze()
            )
            if dist_to_goal < 1:
                self.is_terminated[idx] = True
                if self.tot[idx] == -1:
                    self.tot[idx] = self.timestep

    @torch.no_grad()
    def run_collision_check(self):
        """
        """
        ego_pos = self.get_ego_state()["pos"]
        ego_yaw = self.get_ego_state()["yaw"]
        ego_extent = torch.diag_embed(self.ego_extent)

        adv_pos = self.get_adv_state()["pos"]
        adv_yaw = self.get_adv_state()["yaw"]
        adv_extent = torch.diag_embed(self.adv_extent)

        pos = torch.cat([ego_pos, adv_pos], dim=1)
        yaw = torch.cat([ego_yaw, adv_yaw], dim=1)
        extent = torch.cat([ego_extent, adv_extent], dim=1)

        rot_mat = torch.cat(
            [
                torch.cos(yaw), -torch.sin(yaw),
                torch.sin(yaw), torch.cos(yaw),
            ],
            dim=-1,
        ).view(-1, self.args.num_agents+1, 2, 2)

        corners = self.unit_square @ extent
        corners = corners @ rot_mat.permute(0, 1, 3, 2)
        corners = corners + pos.unsqueeze(-2)

        # we add the corners the simulator attributes so we can re-use them
        self.corners = corners

        corners_a = corners.repeat_interleave(self.args.num_agents + 1, 1)
        corners_b = corners.repeat(1, self.args.num_agents + 1, 1, 1)

        collision = check_collision(
            corners_a.view(-1, 4, 2),
            corners_b.view(-1, 4, 2)
        )

        collision_vec = collision.view(
            self.args.batch_size, -1
        )

        # determine indeces of adversarial-adversarial-pairs
        adv_indices = []
        for i in range(collision_vec.size(1)):
            # discard identity entries in adjacency matrix
            is_identity = False
            if (i % (self.args.num_agents + 1)) == (i // (self.args.num_agents + 1)):
                is_identity = True
            # discard entries involving the ego agent in adjacency matrix
            is_ego = i < (self.args.num_agents + 1) or i % (self.args.num_agents + 1) == 0
            if not is_identity and not is_ego:
                adv_indices.append(i)

        # determine if there is a collision between adversarial agents
        adv_collision = torch.any(collision_vec[:, adv_indices], dim=-1)
        self.adv_collision = torch.logical_or(
            self.adv_collision,
            adv_collision,
        )

        ego_collision = collision_vec[:, 1:self.args.num_agents+1]
        ego_collision = torch.any(ego_collision, dim=-1)
        self.ego_collision = torch.logical_or(
            self.ego_collision,
            ego_collision
        )

        # update termination flags
        new_collision = ~self.is_terminated & ego_collision
        self.is_terminated = self.is_terminated | ego_collision
        self.tot[new_collision] = self.timestep

        # get index of colliding adversarial agent
        reverse_index_array = torch.arange(self.args.num_agents, 0, -1).view(1, -1)
        reverse_index_array = reverse_index_array.expand(
            self.args.batch_size, self.args.num_agents
        ).to("cuda")
        colliding_adv_index = torch.argmax(
            ego_collision * reverse_index_array,
            dim=-1
        )

        # additional collision meta data
        if torch.any(new_collision):
            # compute relative position
            rel_pos = adv_pos[new_collision, colliding_adv_index] - ego_pos.squeeze(1)
            rel_pos = rel_pos.unsqueeze(-2) @ rot_mat[new_collision, 0, ...]
            rel_pos = rel_pos.squeeze(1) / rel_pos.norm(dim=-1)
            self.adv_rel_pos_at_collision[new_collision] = rel_pos[new_collision]

            # compute relative yaw
            constrained_adv_yaw = torch.atan2(
                torch.sin(adv_yaw[new_collision, colliding_adv_index]),
                torch.cos(adv_yaw[new_collision, colliding_adv_index])
            )
            constrained_ego_yaw = torch.atan2(
                torch.sin(ego_yaw[new_collision, 0]),
                torch.cos(ego_yaw[new_collision, 0])
            )
            rel_yaw = constrained_adv_yaw - constrained_ego_yaw
            self.adv_rel_yaw_at_collision[new_collision] = rel_yaw

        # we also want to terminate in case of adv collision
        self.is_terminated = self.is_terminated | self.adv_collision

    @torch.no_grad()
    def run_oob_check(self):
        """
        """
        # exract road channel from map
        road = self.map[:, 0, ...]

        # transform corners from world to global HD map coordinates (in pixels)
        # we exclude the ego vehicle
        corners = self.corners[:, 1:, ...].clone().detach()
        corners_pix = torch.round(self.renderer.world_to_pix(corners)).long()

        # flatten query points for easy indexing
        corners_flattened = corners_pix.view(-1, 2)

        # look up value in map for each of the query points/agent corners
        oob = road[:, corners_flattened[:, 1], corners_flattened[:, 0]]

        # back to original shape so we can tell entities appart
        oob = oob.view(
            corners_pix.size(0),
            corners_pix.size(1),
            corners_pix.size(2),
        )

        # check if any agent corner is out of bounds
        oob = torch.any(~torch.gt(oob, 0), dim=-1)

        # sum up agents that are out of bounds
        oob = torch.sum(oob, dim=-1)

        # set termination flag
        is_oob = torch.gt(oob, 0)
        self.is_terminated = self.is_terminated | is_oob

        return oob

    def step(self, ego_actions, adv_actions):
        """
        """
        # 1. update ego agent
        self.set_ego_state(
            self.motion_model(self.get_ego_state(), ego_actions, self.is_terminated)
        )

        # 2. update adversarial agents
        if self.args.num_agents > 0 and adv_actions is not None:
            self.set_adv_state(
                self.motion_model(self.get_adv_state(), adv_actions, self.is_terminated)
            )

        # keep running buffers of states and actions
        state_detached = {"pos": None, "yaw": None, "vel": None}
        for substate in self.state.keys():
            state_detached[substate] = self.state[substate].clone().detach()
        self.state_buffer.append(state_detached)

        ego_actions_detached = {"steer": None, "throttle": None, "brake": None}
        for action_type in ego_actions.keys():
            ego_actions_detached[action_type] = \
                ego_actions[action_type].clone().detach()
        self.ego_action_buffer.append(ego_actions_detached)

        adv_actions_detached = {"steer": None, "throttle": None, "brake": None}
        if adv_actions:
            for action_type in adv_actions.keys():
                adv_actions_detached[action_type] = \
                    adv_actions[action_type].clone().detach()
            self.adv_action_buffer.append(adv_actions_detached)

        # increment timestep
        self.timestep +=1


def parse_routes(world, routes_file):
    """
    """
    # initialize per town dictionary
    routes_per_town = {}

    tree = ET.parse(routes_file)
    for route in tree.iter("route"):
        route_config = RouteScenarioConfiguration()
        route_config.town = route.attrib['town']

        waypoint_list = []  # the list of waypoints that can be found on this route
        for waypoint in route.iter('waypoint'):
            location = carla.Location(
                x=float(waypoint.attrib['x']),
                y=float(waypoint.attrib['y']),
                z=float(waypoint.attrib['z'])
            )
            rotation =  carla.Rotation(
                yaw=float(waypoint.attrib['yaw'])
            )
            waypoint_list.append(carla.Transform(location, rotation))

        route_config.trajectory = waypoint_list

        try:
            routes_per_town[route_config.town].append(route_config.trajectory)
        except KeyError:
            routes_per_town[route_config.town] = [route_config.trajectory]

    return routes_per_town

