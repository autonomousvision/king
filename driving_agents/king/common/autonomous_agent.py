#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the base class for all autonomous agents
"""

from __future__ import print_function

from enum import Enum

import carla

from leaderboard.utils.route_manipulation import downsample_route


class Track(Enum):

    """
    This enum represents the different tracks of the CARLA AD leaderboard.
    """
    SENSORS = 'SENSORS'
    MAP = 'MAP'

class AutonomousAgent(object):

    """
    Autonomous agent base class. All user agents have to be derived from this class
    """

    def __init__(self, args, device='cpu', path_to_conf_file=None):
        self.track = Track.SENSORS
        #  current global plans to reach a destination
        self._global_plan = None
        self._global_plan_world_coord = None

        # this data structure will contain all sensor data
        # self.sensor_interface = SensorInterface()

        # agent's initialization
        self.setup(args, device=device, path_to_conf_file=path_to_conf_file)

        self.wallclock_t0 = None

    def setup(self, path_to_conf_file):
        """
        Initialize everything needed by your agent and set the track attribute to the right type:
            Track.SENSORS : CAMERAS, LIDAR, RADAR, GPS and IMU sensors are allowed
            Track.MAP : OpenDRIVE map is also allowed
        """
        pass

    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]

        """
        sensors = []

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        :return: control
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False

        return control

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        pass

    def set_global_plan(self, global_plan_gps_list, global_plan_world_coord_list):
        """
        Set the plan (route) for the agent;  batched

        Parameters:
        - global_plan_gps_list: List of global_plan_gps
        - global_plan_world_coord_list: List of global_plan_world_coord
        """
        self._global_plan_world_coord_list = []
        self._global_plan_list = []

        for global_plan_gps, global_plan_world_coord in zip(global_plan_gps_list, global_plan_world_coord_list):
            ds_ids = downsample_route(global_plan_world_coord, 50)
            self._global_plan_world_coord_list.append([(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids])
            self._global_plan_list.append([global_plan_gps[x] for x in ds_ids])
