import os
import json
import time
import statistics
import math
import cv2
import numpy as np
from copy import deepcopy
from collections import deque, defaultdict

import carla
import pygame
import torch

from driving_agents.king.common.utils.conversions import state_vel_to_carla_vector3d, state_pos_to_carla_location, state_to_carla_transform
from driving_agents.king.common import autonomous_agent
from driving_agents.king.common.utils.pid_controller import PIDController
from driving_agents.king.common.utils.planner import RoutePlanner, interpolate_trajectory
from driving_agents.king.aim_bev.datagen_bev_renderer import DatagenBEVRenderer
from driving_agents.king.aim_bev.training_utils import encode_npy_to_pil
from external_code.lbc.bird_view.utils.map_utils import MapImage
from external_code.wor.ego_model import EgoModel
from proxy_simulator.motion_model import BicycleModel

PIXELS_PER_METER = 5


def get_entry_point():
    return 'AutoPilot'


class AutoPilot(autonomous_agent.AutonomousAgent):
    def setup(self, args, path_to_conf_file=None, device=None):
        self.AGENT_TYPE = 'Expert'
        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.save_path = None
        self.device=device
        self.args = args
        self._dense_route = None

        # Dynamics models
        self.frame_rate = args.sim_tickrate
        self.ego_model     = EgoModel(dt=(1.0 / self.frame_rate))
        self.vehicle_model = BicycleModel(delta_t=(1.0 / self.frame_rate)).to(self.device)

        self.EXPERT_WAYPOINT_DISTANCE = self.frame_rate // 2 # in timesteps
        self.EXPERT_WAYPOINTS_NUMBER = 4 # number of future waypoints

        if hasattr(self.args, "king_data_fps"):
            self.save_freq = self.frame_rate // self.args.king_data_fps  # save every half second, assuming a framerate of 4 fps
        else:
            self.save_freq = 2


        # Controllers
        self.steer_buffer_size = 5     # Number of elements to average steering over
        if args.ego_agent == "transfuser":
            self.target_speed_slow = 3.0	# Speed at junctions, m/s
        if args.ego_agent == "aim-bev":
            self.target_speed_slow = 5.0
        self.target_speed_fast = 4.0	# Speed outside junctions, m/s
        self.clip_delta = 0.25			# Max angular error for turn controller
        self.clip_throttle = 0.75		# Max throttle (0-1)
        self.steer_damping = 0.5		# Steer multiplicative reduction while braking
        self.slope_pitch = 10.0			# Pitch above which throttle is increased
        self.slope_throttle = 0.4		# Excess throttle applied on slopes

        self.steer_buffer = [] # because of batches we need a list of deque objects; carla: deque(maxlen=self.steer_buffer_size)


        # Red light detection
        # Coordinates of the center of the red light detector bounding box. In local coordinates of the vehicle, units are meters
        self.center_bb_light_x = -2.0
        self.center_bb_light_y = 0.0
        self.center_bb_light_z = 0.0

        # Extent of the red light detector bounding box. In local coordinates of the vehicle, units are meters. Size are half of the bounding box
        self.extent_bb_light_x = 4.5
        self.extent_bb_light_y = 1.5
        self.extent_bb_light_z = 2.0

        # Obstacle detection
        self.extrapolation_seconds_no_junction = 1.0    # Amount of seconds we look into the future to predict collisions (>= 1 frame)
        self.extrapolation_seconds = 4.0                # Amount of seconds we look into the future to predict collisions at junctions
        self.detection_radius = 30.0                    # Distance of obstacles (in meters) in which we will check for collisions
        self.light_radius = 15.0                        # Distance of traffic lights considered relevant (in meters)

        # Speed buffer for detecting "stuck" vehicles
        self.vehicle_speed_buffer = defaultdict( lambda: {"velocity": [], "throttle": [], "brake": []})
        self.stuck_buffer_size = 30
        self.stuck_vel_threshold = 0.1
        self.stuck_throttle_threshold = 0.1
        self.stuck_brake_threshold = 0.1

        self.steer = np.zeros(shape=(self.args.batch_size))
        self.brake = np.zeros(shape=(self.args.batch_size))
        self.throttle = np.zeros(shape=(self.args.batch_size))
        self.target_speed = 4.0

        self.angle                = 0.0   # Angle to the next waypoint. Normalized in [-1, 1] corresponding to [-90, 90]
        self.stop_sign_hazard     = False
        self.traffic_light_hazard = False
        self.walker_hazard        = [False for i in range(int(self.extrapolation_seconds * self.frame_rate))]
        self.vehicle_hazard       = [False for i in range(int(self.extrapolation_seconds * self.frame_rate))]
        self.junction             = False
        self.cleared_stop_signs = []  # A list of all stop signs that we have cleared

        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam


    def _init(self, world):
        # Near node
        self.world_map = world.carla_wrapper.map # carla: carla.Map("RouteMap", hd_map[1]['opendrive'])
        self._waypoint_planners = []
        self._command_planners = []
        self._turn_controllers = []
        self._speed_controllers = []
        self._speed_controller_extrapolations = []
        for _global_plan, _global_plan_world_coord in zip(self._global_plan_list, self._global_plan_world_coord_list):
            trajectory = [item[0].location for item in _global_plan_world_coord]
            # if not self._dense_route:
            self._dense_route, _ = interpolate_trajectory(self.world_map, trajectory)

            self._waypoint_planner = RoutePlanner(4.0, 50, prepop_skip=1)
            self._waypoint_planner.set_route(self._dense_route, True)
            self._waypoint_planners.append(self._waypoint_planner)

            # Far node
            self._command_planner = RoutePlanner(7.5, 25.0, 1600, prepop_skip=0)
            self._command_planner.set_route(_global_plan, True)
            self._command_planners.append(self._command_planner)

            self._turn_controllers.append(PIDController(K_P=1.75, K_I=0.75, K_D=3.5, n=8))
            self._speed_controllers.append(PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=8))
            self._speed_controller_extrapolations.append(PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=8))

        self._world = world

        self.initialized = True

    def get_state(self):
        if self.initialized:
            state = {
                "_waypoint_planners": deepcopy(self._waypoint_planners),
                "_waypoint_planner": deepcopy(self._waypoint_planner),
                "_command_planners": deepcopy(self._command_planners),
                "_command_planner": deepcopy(self._command_planner),
                "_turn_controllers": deepcopy(self._turn_controllers),
                "_speed_controllers": deepcopy(self._speed_controllers),
                "_speed_controller_extrapolations": deepcopy(self._speed_controller_extrapolations),
                "_dense_route": deepcopy(self._dense_route),
                "steer_buffer": deepcopy(self.steer_buffer),
                "vehicle_speed": deepcopy(self.vehicle_speed_buffer),
                "steer": deepcopy(self.steer),
                "brake": deepcopy(self.brake),
                "throttle": deepcopy(self.throttle),
                "initialized": True
            }
        else:
            state = {}
        return state

    def set_state(self, world, state):
        self._world = world
        self.world_map = world.carla_wrapper.map
        for key, value in state.items():
            setattr(self, key, value)

    def reset(self):
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        self.ego_model     = EgoModel(dt=(1.0 / self.frame_rate))
        self.vehicle_model = BicycleModel(delta_t=(1.0 / self.frame_rate)).to(self.device)

        self.steer_buffer = []
        self.vehicle_speed_buffer = defaultdict( lambda: {"velocity": [], "throttle": [], "brake": []})

        self.steer = np.zeros(shape=(self.args.batch_size))
        self.brake = np.zeros(shape=(self.args.batch_size))
        self.throttle = np.zeros(shape=(self.args.batch_size))

    def sensors(self):
        pass

    def tick(self, input_data):
        return input_data

    def run_step(self, input_data, world, save_data=True, adv_actions=None):
        self.step += 1
        if not self.initialized:
            self._init(world)
            for ix in range(len(input_data['gps'])):
                self.steer_buffer.append(deque(maxlen=self.steer_buffer_size))

        return self._get_control(input_data, save_data=save_data, adv_actions=adv_actions)


    def _get_control(self, input_data, adv_actions=None, steer_init=None, throttle_init=None,
                        vehicle_hazard_init=None, light_hazard_init=None, walker_hazard_init=None, stop_sign_hazard_init=None, save_data=True):
        """
        Get control for the whole batch. Since every of our other agents inherits
        from the expert we can call this function also if we only want parts of the control.
        In this case we pass the actions we already know (from e.g. MapAgent) - this method will only
        compute the missing actions.
        For the hazards we pass None if we want to calculate it for the whole batch. We pass a list of
        hazards for each batch element in case we use the pre computed ones.

        Parameters:
        """
        self._vehicle_state = self._world.get_ego_state()
        steer_act = torch.empty(size=(input_data['gps'].shape[0], 1, 1), device=self.device)
        throttle_act = torch.empty(size=(input_data['gps'].shape[0], 1, 1), device=self.device)
        brake_act = torch.empty(size=(input_data['gps'].shape[0], 1, 1), device=self.device)
        birdviews = None

        for ix in range(len(input_data['gps'])):
            control = carla.VehicleControl()
            # insert missing controls
            speed = input_data['speed'][ix][0].item()
            _ = self._world.get_ego_sensor()
            if vehicle_hazard_init is None or light_hazard_init is None or walker_hazard_init is None or stop_sign_hazard_init is None:
                brake = self._get_brake(ix, vehicle_hazard_init, light_hazard_init, walker_hazard_init, stop_sign_hazard_init, adv_actions=adv_actions) # privileged
            else:
                brake = vehicle_hazard_init or light_hazard_init or walker_hazard_init or stop_sign_hazard_init
            target_speed = self.target_speed_slow if self.junction else self.target_speed_fast

            if throttle_init is None:
                throttle = self._get_throttle(brake, target_speed, speed, ix)

            pos = self._get_position(input_data['gps'][ix][0].detach().cpu().numpy())
            if steer_init is None:
                theta = input_data['imu'][ix][0].item()
                near_node, near_command = self._waypoint_planners[ix].run_step(pos)  # needs HD map
                if self.save_path is not None:
                    far_node, far_command = self._command_planners[ix].run_step(pos)
                steer = self._get_steer(brake, near_node, pos, theta, speed, ix)

                self.steer_buffer[ix].append(steer)

                control.steer = np.mean(self.steer_buffer[ix])
                self.steer[ix] = control.steer
                steer_act[ix] = torch.tensor(
                    [control.steer],
                ).view(1, 1, 1)
            else:
                control.steer = steer_init[ix].item()
                self.steer[ix] = control.steer

            control.throttle = throttle
            control.brake = float(brake)

            self.throttle[ix] = control.throttle
            self.brake[ix] = control.brake
            self.target_speed = target_speed

            throttle_act[ix] = torch.tensor(
                    [control.throttle],
                ).view(1, 1, 1)

            brake_act[ix] = torch.tensor(
                    [control.brake],
                ).view(1, 1, 1)

        if((self.step % self.save_freq == 0) and (self.save_path is not None) and save_data):
            # we don't do batches here so everything is the 0-th element (the loop above only does one iteration)
            assert np.isscalar(throttle) # make sure this isnt run in batched mode by accident
            self.save_measurements(far_node, near_command, steer, throttle, brake, target_speed, input_data)

        if steer_init is not None:
            steer_act = steer_init
        actions = {
            "steer":  steer_act,
            "throttle": throttle_act,
            "brake": brake_act,
        }

        return actions

    def _get_steer(self, brake, target, pos, theta, speed, batch_id):
        angle_unnorm = self._get_angle_to(pos, theta, target)
        angle = angle_unnorm / 90

        if((speed < 0.01) and (brake == True)):
            angle = 0.0
        self.angle = angle

        steer = self._turn_controllers[batch_id].step(angle)
        steer = np.clip(steer, -0.99, 0.99)
        steer = round(steer, 3)

        if brake:
            steer *= self.steer_damping

        return steer

    def _get_throttle(self, brake, target_speed, speed, batch_id):
        target_speed = target_speed if not brake else 0.0

        delta = np.clip(target_speed - speed, 0.0, self.clip_delta)
        throttle = self._speed_controllers[batch_id].step(delta)
        throttle = np.clip(throttle, 0.0, self.clip_throttle)

        if brake:
            throttle = 0.0

        return throttle


    def _get_brake(self, batch_id, vehicle_hazard=None, light_hazard=None, walker_hazard=None, stop_sign_hazard=None, adv_actions=None):

        vehicle_location = state_pos_to_carla_location(self._vehicle_state["pos"][batch_id].squeeze())
        vehicle_transform = state_to_carla_transform(self._vehicle_state["pos"][batch_id].squeeze(), self._vehicle_state["yaw"][batch_id].squeeze())
        self._vehicle_transform = vehicle_transform

        speed = self._get_forward_speed(batch_id=batch_id)

        map = self._world.carla_wrapper.map
        ego_vehicle_waypoint = map.get_waypoint(vehicle_location)
        self.junction = ego_vehicle_waypoint.is_junction

        # -----------------------------------------------------------
        # Obstacle detection
        # -----------------------------------------------------------
        if vehicle_hazard is None or walker_hazard is None:
            vehicle_hazard = False
            self.vehicle_hazard = [False for i in range(int(self.extrapolation_seconds * self.frame_rate))]
            extrapolation_seconds   = self.extrapolation_seconds  # amount of seconds we look into the future to predict collisions
            detection_radius        = self.detection_radius       # distance in which we check for collisions
            number_of_future_frames = int(extrapolation_seconds * self.frame_rate)
            number_of_future_frames_no_junction = int(self.extrapolation_seconds_no_junction * self.frame_rate)

            # -----------------------------------------------------------
            # Vehicle detection
            # -----------------------------------------------------------

            adv_vehicles_state = self._world.get_adv_state()

            nearby_vehicles = {}
            tmp_near_vehicle_id = []
            tmp_stucked_vehicle_id = []
            for ix in range(adv_vehicles_state["pos"][batch_id].shape[0]):

                adv_loc = state_pos_to_carla_location(adv_vehicles_state["pos"][batch_id][ix].squeeze())

                if (adv_loc.distance(vehicle_location) < detection_radius):
                    tmp_near_vehicle_id.append(ix)
                    veh_future_bbs    = []
                    traffic_transform = state_to_carla_transform(
                                                                adv_vehicles_state["pos"][batch_id][ix].squeeze(),
                                                                adv_vehicles_state["yaw"][batch_id][ix].squeeze()
                                                                )
                    traffic_control   = carla.VehicleControl()
                    traffic_control.steer = adv_actions["steer"][ix].clone().cpu().numpy().item()
                    traffic_control.throttle = adv_actions["throttle"][ix].clone().cpu().numpy().item()
                    traffic_control.brake = adv_actions["brake"][ix].clone().cpu().numpy().item()

                    traffic_velocity  = state_vel_to_carla_vector3d(adv_vehicles_state["vel"][batch_id][ix].squeeze())
                    traffic_speed     = self._get_forward_speed(transform=traffic_transform, velocity=traffic_velocity) # In m/s

                    self.vehicle_speed_buffer[ix]["velocity"].append(traffic_speed)
                    self.vehicle_speed_buffer[ix]["throttle"].append(traffic_control.throttle)
                    self.vehicle_speed_buffer[ix]["brake"].append(traffic_control.brake)
                    if len(self.vehicle_speed_buffer[ix]["velocity"]) > self.stuck_buffer_size:
                        self.vehicle_speed_buffer[ix]["velocity"] = self.vehicle_speed_buffer[ix]["velocity"][-self.stuck_buffer_size:]
                        self.vehicle_speed_buffer[ix]["throttle"] = self.vehicle_speed_buffer[ix]["throttle"][-self.stuck_buffer_size:]
                        self.vehicle_speed_buffer[ix]["brake"] = self.vehicle_speed_buffer[ix]["brake"][-self.stuck_buffer_size:]

                    next_loc   = np.array([traffic_transform.location.x, traffic_transform.location.y])
                    next_yaw   = np.array([traffic_transform.rotation.yaw / 180.0 * np.pi])
                    next_speed = np.array([traffic_speed])
                    next_state = {
                        "pos": adv_vehicles_state["pos"][:, ix:ix+1],
                        "yaw": adv_vehicles_state["yaw"][:, ix:ix+1],
                        "vel": adv_vehicles_state["vel"][:, ix:ix+1]
                    }
                    action = {
                        "steer":    adv_actions["steer"][None, ix:ix+1, :],
                        "throttle": adv_actions["throttle"][None, ix:ix+1, :],
                        "brake":    adv_actions["brake"][None, ix:ix+1, :],
                    }

                    for i in range(number_of_future_frames):
                        next_state = self.vehicle_model.forward(next_state, action, self._world.is_terminated)

                        delta_yaws = next_state["yaw"].item() * 180.0 / math.pi

                        transform             = carla.Transform(carla.Location(x=next_state["pos"][0, 0, 0].item(), y=next_state["pos"][0, 0, 1].item(), z=traffic_transform.location.z))
                        bounding_box          = carla.BoundingBox(transform.location, carla.Vector3D(2.45, 1.06, .755))
                        bounding_box.rotation = carla.Rotation(pitch=float(traffic_transform.rotation.pitch),
                                                                yaw=float(delta_yaws),
                                                                roll=float(traffic_transform.rotation.roll))

                        veh_future_bbs.append(bounding_box)

                    if (statistics.mean(self.vehicle_speed_buffer[ix]["velocity"]) < self.stuck_vel_threshold
                            and statistics.mean(self.vehicle_speed_buffer[ix]["throttle"]) > self.stuck_throttle_threshold
                            and statistics.mean(self.vehicle_speed_buffer[ix]["brake"]) < self.stuck_brake_threshold):
                        tmp_stucked_vehicle_id.append(ix)

                    nearby_vehicles[ix] = veh_future_bbs

            # delete old vehicles
            to_delete = set(self.vehicle_speed_buffer.keys()).difference(tmp_near_vehicle_id)
            for d in to_delete:
                del self.vehicle_speed_buffer[d]

            # -----------------------------------------------------------
            # Intersection checks with ego vehicle
            # -----------------------------------------------------------
            next_loc   = np.array([vehicle_transform.location.x, vehicle_transform.location.y])

            #NOTE intentionally set ego vehicle to move at the target speed (we want to know if there is an intersection if we would not brake)
            delta_extrapolation = np.clip(self.target_speed - speed, 0.0, 0.25)
            throttle_extrapolation = self._speed_controller_extrapolations[batch_id].step(delta_extrapolation)
            throttle_extrapolation = np.clip(throttle_extrapolation, 0.0, 0.75)
            action     = np.array(np.stack([self.steer[batch_id], throttle_extrapolation, 0.0], axis=-1))

            next_yaw   = np.array([vehicle_transform.rotation.yaw / 180.0 * np.pi])
            next_speed = np.array([speed])

            back_only_vehicle_id = []

            for i in range(number_of_future_frames):
                # Calculate ego vehicle bounding box for the next timestep. We don't consider timestep 0 because it is from the past and has already happened.
                next_loc, next_yaw, next_speed = self.ego_model.forward(next_loc, next_yaw, next_speed, action)
                delta_yaws = next_yaw.item() * 180.0 / math.pi

                cosine = np.cos(next_yaw.item())
                sine = np.sin(next_yaw.item())

                extent           = carla.Vector3D(2.45, 1.06, .755) #HACK
                extent.x         = extent.x / 2.

                # front half
                transform             = carla.Transform(carla.Location(x=next_loc[0].item()+extent.x*cosine, y=next_loc[1].item()+extent.y*sine, z=vehicle_transform.location.z))
                bounding_box          = carla.BoundingBox(transform.location, extent)
                bounding_box.rotation = carla.Rotation(pitch=float(vehicle_transform.rotation.pitch), yaw=float(delta_yaws), roll=float(vehicle_transform.rotation.roll))

                # back half
                transform_back             = carla.Transform(carla.Location(x=next_loc[0].item()-extent.x*cosine, y=next_loc[1].item()-extent.y*sine, z=vehicle_transform.location.z))
                bounding_box_back          = carla.BoundingBox(transform_back.location, extent)
                bounding_box_back.rotation = carla.Rotation(pitch=float(vehicle_transform.rotation.pitch), yaw=float(delta_yaws), roll=float(vehicle_transform.rotation.roll))

                for id, traffic_participant in nearby_vehicles.items():
                    if self.junction==False and i > number_of_future_frames_no_junction:
                        break
                    if id in tmp_stucked_vehicle_id:
                        continue
                    back_intersect = (self.check_obb_intersection(bounding_box_back, traffic_participant[i]) == True)
                    front_intersect = (self.check_obb_intersection(bounding_box, traffic_participant[i]) == True)
                    if id in back_only_vehicle_id:
                        back_only_vehicle_id.remove(id)
                        if back_intersect:
                            back_only_vehicle_id.append(id)
                        continue
                    if back_intersect and not front_intersect:
                        back_only_vehicle_id.append(id)
                    if front_intersect:
                        vehicle_hazard = True
                        self.vehicle_hazard[i] = True

                        # check if visible in transfuser sensors
                        current_nearby_vehicle_pos = adv_vehicles_state["pos"][batch_id][id]
                        current_nearby_vehicle_yaw = adv_vehicles_state["yaw"][batch_id][id]
                        current_nearby_vehicle_yaw = current_nearby_vehicle_yaw.item() * 180.0 / math.pi

                        current_nearby_vehicle_transform = carla.Transform(carla.Location(x=current_nearby_vehicle_pos[0].item(), y=current_nearby_vehicle_pos[1].item(), z=0.755))
                        current_box          = carla.BoundingBox(current_nearby_vehicle_transform.location, carla.Vector3D(2.45, 1.06, .755))
                        current_box.rotation = carla.Rotation(pitch=float(0),
                                                                yaw=float(current_nearby_vehicle_yaw),
                                                                roll=float(0))

            #Add safety bounding box in front. If there is anything in there we won't start driving
            bremsweg = ((speed * 3.6) / 10.0)**2 / 2.0 # Bremsweg formula for emergency break
            safety_x = np.clip(bremsweg + 1.0, a_min=2.0, a_max=4.0)# Plus One meter is the car.

            center_safety_box     = vehicle_transform.transform(carla.Location(x=safety_x, y=0.0, z=0.0))
            bounding_box          = carla.BoundingBox(center_safety_box, carla.Vector3D(2.45, 1.06, .755))
            bounding_box.rotation = vehicle_transform.rotation

            for _, traffic_participant in nearby_vehicles.items():
                if (self.check_obb_intersection(bounding_box, traffic_participant[0]) == True): #Check the first BB of the traffic participant. We don't extrapolate into the future here.
                    vehicle_hazard = True
                    self.vehicle_hazard[0] = True

        else:
            self.vehicle_hazard = vehicle_hazard
            self.walker_hazard = walker_hazard

        self.stop_sign_hazard     = stop_sign_hazard
        self.traffic_light_hazard = light_hazard

        return (vehicle_hazard)

    def _get_forward_speed(self, transform=None, velocity=None, batch_id=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = state_vel_to_carla_vector3d(self._vehicle_state["vel"][batch_id].squeeze())
        if not transform:
            transform = self._vehicle_transform

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    def dot_product(self, vector1, vector2):
        return (vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z)

    def cross_product(self, vector1, vector2):
        return carla.Vector3D(x=vector1.y * vector2.z - vector1.z * vector2.y, y=vector1.z * vector2.x - vector1.x * vector2.z, z=vector1.x * vector2.y - vector1.y * vector2.x)

    def get_separating_plane(self, rPos, plane, obb1, obb2):
        ''' Checks if there is a seperating plane
        rPos Vec3
        plane Vec3
        obb1  Bounding Box
        obb2 Bounding Box
        '''
        return (abs(self.dot_product(rPos, plane)) > (abs(self.dot_product((obb1.rotation.get_forward_vector() * obb1.extent.x), plane)) +
                                                      abs(self.dot_product((obb1.rotation.get_right_vector()   * obb1.extent.y), plane)) +
                                                      abs(self.dot_product((obb1.rotation.get_up_vector()      * obb1.extent.z), plane)) +
                                                      abs(self.dot_product((obb2.rotation.get_forward_vector() * obb2.extent.x), plane)) +
                                                      abs(self.dot_product((obb2.rotation.get_right_vector()   * obb2.extent.y), plane)) +
                                                      abs(self.dot_product((obb2.rotation.get_up_vector()      * obb2.extent.z), plane)))
                )

    def check_obb_intersection(self, obb1, obb2):
        RPos = obb2.location - obb1.location

        return not(self.get_separating_plane(RPos, obb1.rotation.get_forward_vector(), obb1, obb2) or
                   self.get_separating_plane(RPos, obb1.rotation.get_right_vector(),   obb1, obb2) or
                   self.get_separating_plane(RPos, obb1.rotation.get_up_vector(),      obb1, obb2) or
                   self.get_separating_plane(RPos, obb2.rotation.get_forward_vector(), obb1, obb2) or
                   self.get_separating_plane(RPos, obb2.rotation.get_right_vector(),   obb1, obb2) or
                   self.get_separating_plane(RPos, obb2.rotation.get_up_vector(),      obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_forward_vector()), obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_right_vector()),   obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_up_vector()),      obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_right_vector()  , obb2.rotation.get_forward_vector()), obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_right_vector()  , obb2.rotation.get_right_vector()),   obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_right_vector()  , obb2.rotation.get_up_vector()),      obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_up_vector()     , obb2.rotation.get_forward_vector()), obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_up_vector()     , obb2.rotation.get_right_vector()),   obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_up_vector()     , obb2.rotation.get_up_vector()),      obb1, obb2))

    def _get_angle_to(self, pos, theta, target):
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        aim = R.T.dot(target - pos).squeeze()
        angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
        angle = 0.0 if np.isnan(angle) else angle

        return angle

    def _get_position(self, gps):
        gps = (gps - self._waypoint_planner.mean) * self._waypoint_planner.scale
        return gps



class BEVDataAgent(AutoPilot):
    def __init__(self, args, device='cpu', path_to_conf_file=None, save_path=None):
        super().__init__(args, device=device, path_to_conf_file=path_to_conf_file)
        self.save_path = save_path

        self.vehicle_template = torch.ones(
            [1, 1, 22, 9],
            device=self.args.device,
            dtype=torch.float32
        )

    def setup(self, args, path_to_conf_file=None, device=None, route_index=None, data_generation=True):
        super().setup(args, path_to_conf_file=path_to_conf_file, device=device)
        self.data_generation = data_generation

        self.channels = 4 # without traffic lights


    def _init(self, world):
        super()._init(world)

        self.vehicle_template = torch.ones(1, 1, 22, 9, device=self.args.device)

        # create map for renderer
        map_image = MapImage(self._world, self.world_map, PIXELS_PER_METER)
        make_image = lambda x: np.swapaxes(pygame.surfarray.array3d(x), 0, 1).mean(axis=-1)
        road = make_image(map_image.map_surface)
        lane = make_image(map_image.lane_surface)

        self.global_map = np.zeros((1, self.channels,) + road.shape)
        self.global_map[:, 0, ...] = road / 255.
        self.global_map[:, 1, ...] = lane / 255.

        self.global_map = torch.tensor(self.global_map, device=self.args.device, dtype=torch.float32)
        world_offset = torch.tensor(map_image._world_offset, device=self.args.device, dtype=torch.float32)
        self.map_dims = self.global_map.shape[2:4]

        self.renderer = DatagenBEVRenderer(self.args, world_offset, self.map_dims, data_generation=self.data_generation)

    def tick(self, input_data):
        result = super().tick(input_data)
        result['topdown'] = self.render_BEV()

        return result

    @torch.no_grad()
    def run_step(self, input_data, world, adv_actions=None):
        control = super().run_step(input_data, world, adv_actions=adv_actions)

        if self.step % self.save_freq == 0:
            if self.save_path is not None:
                tick_data = self.tick(input_data)
                self.save_sensors(tick_data)

        return control

    def save_sensors(self, tick_data):
        path = os.path.join(self.save_path, tick_data["route_name"])
        if "init_timestep" in tick_data:
            init_timestep = tick_data["init_timestep"]
            path = path + f"_initstep_{init_timestep}"

        if "variation" in tick_data:
            variation = tick_data["variation"]
            path = path + f"_variation_{variation}"

        frame_id = f"{(self.step // self.save_freq):04d}"

        if not os.path.exists(os.path.join(path, 'topdown')):
            os.makedirs(os.path.join(path, 'topdown'))

        topdown_img = encode_npy_to_pil(np.asarray(tick_data['topdown'].squeeze().cpu()))
        topdown_img_save=np.moveaxis(topdown_img,0,2)
        cv2.imwrite(os.path.join(path, 'topdown', f'encoded_{frame_id}.png'), topdown_img_save)

    def save_measurements(self, far_node, near_command, steer, throttle, brake, target_speed, tick_data):
        frame = self.step // self.save_freq
        route_name = tick_data["route_name"]

        path = os.path.join(self.save_path, route_name)
        if "init_timestep" in tick_data:
            init_timestep = tick_data["init_timestep"]
            path = path + f"_initstep_{init_timestep}"

        if "variation" in tick_data:
            variation = tick_data["variation"]
            path = path + f"_variation_{variation}"

        if not os.path.exists(os.path.join(path, 'measurements')):
            os.makedirs(os.path.join(path, 'measurements'))

        pos = self._get_position(tick_data['gps'].cpu().numpy())
        theta = tick_data['imu'].cpu().numpy()
        speed = tick_data['speed'].cpu().numpy()

        data = {
            'x': pos[0, 0, 0].tolist(),
            'y': pos[0, 0, 1].tolist(),
            'theta': theta[0, 0].item(),
            'speed': speed[0, 0].item(),
            'target_speed': target_speed,
            'x_command': far_node[0],
            'y_command': far_node[1],
            'command': near_command.value,
            'waypoints': [],
            'steer': steer,
            'throttle': throttle,
            'brake': brake,
            'junction':         self.junction,
            'vehicle_hazard':   self.vehicle_hazard,
            'light_hazard':     False, # self.traffic_light_hazard,
            'walker_hazard':    self.walker_hazard,
            'stop_sign_hazard': False, # self.stop_sign_hazard,
            'angle':            self.angle,
            # 'ego_matrix': self._vehicle.get_transform().get_matrix()
        }

        measurements_file = os.path.join(path, "measurements", ('%04d.json' % frame))
        with open(measurements_file, 'w') as f:
            json.dump(data, f, indent=4)

    def render_BEV(self):
        semantic_grid = self.global_map

        ego_pos = self._world.get_ego_state()["pos"][0, 0]
        ego_yaw = self._world.get_ego_state()["yaw"][0, 0]

        adv_pos = self._world.get_adv_state()["pos"]
        adv_yaw = self._world.get_adv_state()["yaw"]

        birdview = self.renderer.get_local_birdview(
            semantic_grid,
            ego_pos,
            ego_yaw
        )

        # render vehicles
        for agent_id in range(adv_pos.shape[1]):
            self.renderer.render_agent_bv(
                birdview,
                ego_pos,
                ego_yaw,
                self.vehicle_template,
                adv_pos[0, agent_id],
                adv_yaw[0, agent_id],
                channel=2 # vehicles are rendered in second channel, in king we don't have pedestrians
            )

        return birdview
