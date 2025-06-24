import numpy as np
from scipy.interpolate import splprep, splev
import math

from .pathfinder import Pathfinder

class Navigator:
    """
    Handles path planning, path smoothing, and path-following navigation logic.
    """

    def __init__(self, config: dict):
        """
        Initializes the Navigator.
        :param config: The main robot configuration dictionary.
        """
        self.config = config
        self.pathfinder = Pathfinder(config['CELL_SIZE_M'])
        self.path = None
        self.smoothed_path = None
        self.current_target_index = 0

    def find_path(self, start_world: tuple, end_world: tuple, grid: np.ndarray):
        """
        Plans a path from a start to an end coordinate, avoiding obstacles in the grid.
        :param start_world: The starting (x, y) coordinates in meters.
        :param end_world: The ending (x, y) coordinates in meters.
        :param grid: The occupancy grid.
        """
        self.pathfinder.set_grid(grid)
        start_cell = self.pathfinder.world_to_grid(start_world)
        end_cell = self.pathfinder.world_to_grid(end_world)
        
        path_nodes = self.pathfinder.find_path(start_cell, end_cell)
        
        if path_nodes:
            self.path = path_nodes
            if len(self.path) > 1:
                # Convert path to world coordinates for smoothing
                path_world = np.array([self.pathfinder.grid_to_world(p) for p in self.path])
                self.smoothed_path = self._generate_smoothed_path(path_world)
            else:
                self.smoothed_path = np.array([self.pathfinder.grid_to_world(self.path[0])])
            self.current_target_index = 0
            print(f"Path planned: {len(self.path)} waypoints. Smoothing complete.")
            return True
        else:
            print(f"Failed to plan path from {start_world} to {end_world}")
            self.path = None
            self.smoothed_path = None
            return False

    def _generate_smoothed_path(self, path_world: np.ndarray):
        """Generates a smooth B-spline path from world coordinate waypoints."""
        if path_world.shape[0] < 2:
            return path_world

        # Ensure there are no duplicate points, which can cause issues with splprep
        unique_path, indices = np.unique(path_world, axis=0, return_index=True)
        unique_path = unique_path[np.argsort(indices)]

        if unique_path.shape[0] < 2:
            return unique_path
            
        x, y = unique_path[:, 0], unique_path[:, 1]
        
        # k is the degree of the spline. Must be <= number of points - 1.
        k = min(len(x) - 1, 3)
        if k < 1: 
            return unique_path
        
        tck, u = splprep([x, y], s=0.1, k=k) # s is a smoothing factor
        u_new = np.linspace(u.min(), u.max(), 100)
        x_new, y_new = splev(u_new, tck)
        
        return np.column_stack((x_new, y_new))

    def pure_pursuit_controller(self, robot_pose: tuple) -> tuple:
        """
        Calculates the required robot velocity (vx, vy, v_theta) to follow the smoothed path.
        :param robot_pose: The current pose of the robot (x, y, heading) in world coordinates.
        :return: A tuple of (vx, vy, v_theta) for the robot velocity.
        """
        if self.smoothed_path is None or self.is_mission_complete(robot_pose):
            return 0.0, 0.0, 0.0 # Stop

        base_lookahead = 0.3 # meters
        
        # Find the best lookahead point on the path
        lookahead_point, self.current_target_index = self._find_lookahead_point(robot_pose, base_lookahead)

        if lookahead_point is None:
            return 0.0, 0.0, 0.0 # Stop if no lookahead point found

        # Calculate the steering angle to the lookahead point
        world_x, world_y, robot_heading = robot_pose
        
        angle_to_target = np.arctan2(lookahead_point[1] - world_y, lookahead_point[0] - world_x)
        steering_angle = self._normalize_angle(angle_to_target - robot_heading)

        target_speed = self._get_adaptive_speed(steering_angle)

        v_theta = (2 * target_speed * np.sin(steering_angle)) / base_lookahead
        
        vx = target_speed
        vy = 0.0
        
        return vx, vy, v_theta

    def _find_lookahead_point(self, robot_pose: tuple, lookahead_distance: float):
        """Finds the lookahead point on the smoothed path for pure pursuit."""
        robot_pos = np.array([robot_pose[0], robot_pose[1]])
        
        path_segment = self.smoothed_path[self.current_target_index:]
        if len(path_segment) == 0:
            return None, self.current_target_index

        distances = np.linalg.norm(path_segment - robot_pos, axis=1)
        
        possible_points_indices = np.where(distances >= lookahead_distance)[0]
        
        if len(possible_points_indices) > 0:
            best_i = self.current_target_index + possible_points_indices[0]
            return self.smoothed_path[best_i], best_i
        else:
            # Return the last point if we are close to the end and no points are far enough
            return self.smoothed_path[-1], len(self.smoothed_path) - 1

    def _get_adaptive_speed(self, steering_angle: float) -> float:
        """Calculates an appropriate speed based on the required turn angle."""
        if not self.config['FEATURES'].get('ADAPTIVE_SPEED_ENABLED', False):
            return self.config.get('BASE_SPEED', 50) / 100.0 * 0.7 # Scale to m/s

        turn_factor = 1.0 - (abs(steering_angle) / (np.pi / 2))**0.8
        
        base_speed_ms = self.config.get('BASE_SPEED', 50) / 100.0 * 0.7
        corner_speed_ms = self.config.get('CORNER_SPEED', 30) / 100.0 * 0.7

        adaptive_speed = corner_speed_ms + (base_speed_ms - corner_speed_ms) * turn_factor
        return max(corner_speed_ms, adaptive_speed)

    def _normalize_angle(self, angle: float) -> float:
        """Normalize an angle to the range [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def is_mission_complete(self, robot_pose: tuple, threshold: float = 0.1) -> bool:
        """
        Check if the robot has reached the end of the path.
        """
        if self.smoothed_path is None:
            return True
        
        end_point = self.smoothed_path[-1]
        robot_pos = np.array([robot_pose[0], robot_pose[1]])
        distance_to_end = np.linalg.norm(robot_pos - end_point)
        
        return distance_to_end < threshold 