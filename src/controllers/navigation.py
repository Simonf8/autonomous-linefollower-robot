import numpy as np
from scipy.interpolate import splprep, splev
import math
from typing import List, Tuple

from pathfinder import Pathfinder

class Navigator:
    """Handles path planning, path following, and navigation logic."""

    def __init__(self, cell_size_m, start_cell, end_cell):
        self.cell_size_m = cell_size_m
        self.start_cell = start_cell
        self.end_cell = end_cell

        self.path = []
        self.smoothed_path = None
        self.current_target_index = 0
        self.lookahead_distance = 0.2 # Lookahead distance in meters for pure pursuit
        
        maze_grid = Pathfinder.create_maze_grid()
        self.pathfinder = Pathfinder(grid=maze_grid, cell_size_m=self.cell_size_m)

    def plan_path(self, current_cell) -> bool:
        """Plans a path from the current cell to the end cell."""
        print(f"Planning path from {current_cell} to {self.end_cell}...")
        path_nodes = self.pathfinder.find_path(current_cell, self.end_cell)
        
        if path_nodes:
            self.path = path_nodes
            if len(self.path) > 1:
                self.smoothed_path = self._generate_smoothed_path(self.path)
            else:
                self.smoothed_path = None
            self.current_target_index = 0
            print(f"Path planned: {len(self.path)} waypoints. Smoothing enabled.")
            return True
        else:
            print(f"Failed to plan path from {current_cell} to {self.end_cell}")
            return False

    def _generate_smoothed_path(self, path_nodes: List[Tuple[int, int]]):
        """Generates a smooth B-spline path from waypoints."""
        if len(path_nodes) < 2:
            return None

        path_m = np.array([(p[0] * self.cell_size_m, p[1] * self.cell_size_m) for p in path_nodes])
        x, y = path_m[:, 0], path_m[:, 1]
        
        k = min(len(x) - 1, 3)
        if k < 1: return None
        
        tck, u = splprep([x, y], s=0, k=k)
        u_new = np.linspace(u.min(), u.max(), 100)
        x_new, y_new = splev(u_new, tck)
        
        return np.column_stack((x_new, y_new))

    def get_lookahead_point(self, pose):
        """Finds the lookahead point on the smoothed path for pure pursuit."""
        if self.smoothed_path is None:
            return None

        robot_pos = np.array([pose[0], pose[1]])
        
        # Find the closest point on the path to the robot
        distances = np.linalg.norm(self.smoothed_path - robot_pos, axis=1)
        closest_index = np.argmin(distances)
        
        # Search forward from the closest point to find the lookahead point
        lookahead_index = closest_index
        while lookahead_index < len(self.smoothed_path) - 1:
            dist_from_robot = np.linalg.norm(self.smoothed_path[lookahead_index] - robot_pos)
            if dist_from_robot > self.lookahead_distance:
                return self.smoothed_path[lookahead_index]
            lookahead_index += 1
            
        # If no point is far enough, return the last point
        return self.smoothed_path[-1]

    def pure_pursuit_controller(self, pose):
        """Calculates wheel speeds to follow the path using Pure Pursuit."""
        robot_x, robot_y, robot_heading = pose
        
        lookahead_point = self.get_lookahead_point(pose)
        if lookahead_point is None:
            return 0, 0, 0, 0 # Stop if no path

        # Transform lookahead point to robot's coordinate frame
        dx = lookahead_point[0] - robot_x
        dy = lookahead_point[1] - robot_y
        
        # Angle to the lookahead point
        angle_to_point = math.atan2(dy, dx)
        alpha = angle_to_point - robot_heading
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi # Normalize

        # High-level control outputs: forward speed and rotational speed
        # The curvature of the path to the lookahead point determines the turn rate
        curvature = (2 * math.sin(alpha)) / self.lookahead_distance
        
        # Convert curvature to a turn command (this is simplified)
        # A more advanced model would use robot kinematics
        turn_command = curvature * 100 # Scaling factor to be tuned
        
        return turn_command, alpha # Return turn command and heading error

    def is_mission_complete(self):
        """Check if the end of the path has been reached."""
        return not self.path or self.current_target_index >= len(self.path)

    def advance_waypoint(self, current_cell):
        """Advance to the next waypoint and return the new heading."""
        target_cell = self.path[self.current_target_index]
        print(f"Waypoint detected! Advancing from {current_cell} to {target_cell}")
        self.current_target_index += 1
        
        if self.current_target_index < len(self.path):
            next_target = self.path[self.current_target_index]
            dx = next_target[0] - target_cell[0]
            dy = next_target[1] - target_cell[1]
            return math.atan2(dy, dx)
        
        return None # No new heading if it's the last waypoint
    
    def update_obstacle(self, obstacle_cell):
        self.pathfinder.update_obstacle(obstacle_cell[0], obstacle_cell[1], is_obstacle=True) 