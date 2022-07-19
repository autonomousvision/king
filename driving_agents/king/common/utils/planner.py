import math
from collections import deque
import xml.etree.ElementTree as ET
import numpy as np

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

DEBUG = False


class Plotter(object):
    def __init__(self, size):
        self.size = size
        self.clear()
        self.title = str(self.size)

    def clear(self):
        from PIL import Image, ImageDraw

        self.img = Image.fromarray(np.zeros((self.size, self.size, 3), dtype=np.uint8))
        self.draw = ImageDraw.Draw(self.img)

    def dot(self, pos, node, color=(255, 255, 255), r=2):
        x, y = 5.5 * (pos - node)
        x += self.size / 2
        y += self.size / 2

        self.draw.ellipse((x-r, y-r, x+r, y+r), color)

    def show(self):
        if not DEBUG:
            return

        import cv2

        cv2.imshow(self.title, cv2.cvtColor(np.array(self.img), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)


class RoutePlanner(object):
    def __init__(self, min_distance, max_distance, debug_size=512, prepop_skip=0):
        self.route = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.prepop_skip = prepop_skip

        # self.mean = np.array([49.0, 8.0]) # for carla 9.9
        # self.scale = np.array([111324.60662786, 73032.1570362]) # for carla 9.9
        self.mean = np.array([0.0, 0.0]) # for carla 9.10
        self.scale = np.array([111324.60662786, 111319.490945]) # for carla 9.10

    def set_route(self, global_plan, gps=False):
        self.route.clear()

        for pos, cmd in global_plan:
            if gps:
                pos = np.array([pos['lat'], pos['lon']])
                pos -= self.mean
                pos *= self.scale
            else:
                pos = np.array([pos.location.x, pos.location.y])
                pos -= self.mean

            self.route.append((pos, cmd))

    def run_step(self, gps):
        if len(self.route) == 1:
            return self.route[0]

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
            distance = np.linalg.norm(self.route[i][0] - gps)

            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i
            r = 255 * int(distance > self.min_distance)
            g = 255 * int(self.route[i][1].value == 4)
            b = 255

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        return self.route[1]


# the functions below are taken and partially adapted from the CARLA leaderboard
# see: leaderboard/leaderboard/utils/route_manipulation.py
def interpolate_trajectory(world_map, waypoints_trajectory, hop_resolution=1.0, max_len=100):
    """
    Given some raw keypoints interpolate a full dense trajectory to be used by the user.
    returns the full interpolated route both in GPS coordinates and also in its original form.

    Args:
        - world: an reference to the CARLA world so we can use the planner
        - waypoints_trajectory: the current coarse trajectory
        - hop_resolution: is the resolution, how dense is the provided trajectory going to be made
    """

    dao = GlobalRoutePlannerDAO(world_map, hop_resolution)
    grp = GlobalRoutePlanner(dao)
    grp.setup()
    # Obtain route plan
    route = []
    for i in range(len(waypoints_trajectory) - 1):   # Goes until the one before the last.
        waypoint = waypoints_trajectory[i]
        waypoint_next = waypoints_trajectory[i + 1]
        if waypoint.x != waypoint_next.x or waypoint.y != waypoint_next.y:
            interpolated_trace = grp.trace_route(waypoint, waypoint_next)
            if len(interpolated_trace) > max_len:
                waypoints_trajectory[i + 1] = waypoints_trajectory[i]
            else:
                for wp_tuple in interpolated_trace:
                    route.append((wp_tuple[0].transform, wp_tuple[1]))

    lat_ref, lon_ref = _get_latlon_ref(world_map)

    return location_route_to_gps(route, lat_ref, lon_ref), route


def location_route_to_gps(route, lat_ref, lon_ref):
    """
        Locate each waypoint of the route into gps, (lat long ) representations.
    :param route:
    :param lat_ref:
    :param lon_ref:
    :return:
    """
    gps_route = []

    for transform, connection in route:
        gps_point = _location_to_gps(lat_ref, lon_ref, transform.location)
        gps_route.append((gps_point, connection))

    return gps_route


def _get_latlon_ref(world_map):
    """
    Convert from waypoints world coordinates to CARLA GPS coordinates
    :return: tuple with lat and lon coordinates
    """
    xodr = world_map.to_opendrive()
    tree = ET.ElementTree(ET.fromstring(xodr))

    # default reference
    lat_ref = 42.0
    lon_ref = 2.0

    for opendrive in tree.iter("OpenDRIVE"):
        for header in opendrive.iter("header"):
            for georef in header.iter("geoReference"):
                if georef.text:
                    str_list = georef.text.split(' ')
                    for item in str_list:
                        if '+lat_0' in item:
                            lat_ref = float(item.split('=')[1])
                        if '+lon_0' in item:
                            lon_ref = float(item.split('=')[1])
    return lat_ref, lon_ref


def _location_to_gps(lat_ref, lon_ref, location):
    """
    Convert from world coordinates to GPS coordinates
    :param lat_ref: latitude reference for the current map
    :param lon_ref: longitude reference for the current map
    :param location: location to translate
    :return: dictionary with lat, lon and height
    """

    EARTH_RADIUS_EQUA = 6378137.0   # pylint: disable=invalid-name
    scale = math.cos(lat_ref * math.pi / 180.0)
    mx = scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    my = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
    mx += location.x
    my -= location.y

    lon = mx * 180.0 / (math.pi * EARTH_RADIUS_EQUA * scale)
    lat = 360.0 * math.atan(math.exp(my / (EARTH_RADIUS_EQUA * scale))) / math.pi - 90.0
    z = location.z

    return {'lat': lat, 'lon': lon, 'z': z}