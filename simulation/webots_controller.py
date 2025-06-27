"""
Simon(CJ), Heba
Hardware-in-the-Loop Webots Controller with ESP32 Integration
Real-time path planning with sensor-based visualization and obstacle detection
"""
from controller import Supervisor, DistanceSensor, Motor, Camera
import time
import math
import heapq
import matplotlib.pyplot as plt
import json
import numpy as np
import sys

# Network Configuration
# ESP32 communication is disabled for local simulation
# ESP32_IP_ADDRESS = "192.168.53.245"
# ESP32_PORT = 8080

# Robot Physical Parameters
# These are no longer needed as we are using vision-based odometry simulation
# WHEEL_RADIUS = 0.0205
# AXLE_LENGTH = 0.05810

# Grid Configuration
GRID_ROWS = 15
GRID_COLS = 21
GRID_CELL_SIZE = 0.05099
GRID_ORIGIN_X = 0.050002
GRID_ORIGIN_Z = -0.639e-05

# Navigation Parameters
GOAL_ROW = 14
GOAL_COL = 0
FORWARD_SPEED = 1.5
LINE_THRESHOLD = 600

# Obstacle Detection Configuration
DISTANCE_SENSOR_THRESHOLD = 110
OBSTACLE_DETECTION_ENABLED = True

# Movement Control Parameters
TURN_SPEED_FACTOR = 1.5
MIN_INITIAL_SPIN_DURATION = 0.6
MAX_SEARCH_SPIN_DURATION = 18.9
MAX_ADJUST_DURATION = 2.20
TURN_ADJUST_BASE_SPEED = FORWARD_SPEED * 0.8
TURN_UNTIL_LINE_FOUND = True

# Line Following Parameters
AGGRESSIVE_CORRECTION_DIFFERENTIAL = FORWARD_SPEED * 1.3
MODERATE_CORRECTION_DIFFERENTIAL = FORWARD_SPEED * 1.2

# Starting Position Configuration
INITIAL_GRID_ROW = 2
INITIAL_GRID_COL = 20

# World grid definition (0 = Black Line, 1 = White Space)
world_grid = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
]


class VisionOdometry:
    """
    Simulates a visual odometry system.
    It uses the simulator's ground truth pose and adds noise to mimic the
    behavior of a real camera-based positioning system, including drift.
    """
    def __init__(self, robot, noise_level_pos=0.001, noise_level_angle=0.002, drift_blend=0.98):
        self.robot = robot
        # Get the robot node to access pose information
        self.robot_node = self.robot.getSelf()
        if self.robot_node is None:
            print("\n\nFATAL ERROR: Could not get Supervisor node for the robot.", file=sys.stderr)
            print("This controller MUST be run as a Supervisor.", file=sys.stderr)
            print("Please go to the Webots world, select your robot in the Scene Tree,", file=sys.stderr)
            print("and set its 'supervisor' field to TRUE.\n\n", file=sys.stderr)
            sys.exit(1)

        self.noise_level_pos = noise_level_pos
        self.noise_level_angle = noise_level_angle
        self.drift_blend = drift_blend # How much of the old (drifted) pose to keep

        # Initialize pose from ground truth
        initial_pos = self.robot_node.getPosition()
        if math.isnan(initial_pos[0]):
            print("\n\nFATAL ERROR: getPosition() returned NaN. The robot is not a Supervisor.", file=sys.stderr)
            print("This controller MUST be run as a Supervisor.", file=sys.stderr)
            print("Please go to the Webots world, select your robot in the Scene Tree,", file=sys.stderr)
            print("and set its 'supervisor' field to TRUE.\n\n", file=sys.stderr)
            sys.exit(1)
            
        initial_rot = self.robot_node.getOrientation()
        self.pose = {
            'x': initial_pos[0],
            'z': initial_pos[2],
            'theta': self._get_yaw_from_orientation(initial_rot)
        }

    def _get_yaw_from_orientation(self, orientation_matrix):
        """Calculates yaw angle from a 3x3 rotation matrix (Webots format)."""
        # For a rotation about the Y-axis (yaw), the matrix is:
        # [[cos(t), 0, sin(t)], [0, 1, 0], [-sin(t), 0, cos(t)]]
        # We can get yaw = atan2(sin(t), cos(t))
        m = orientation_matrix
        return math.atan2(m[2], m[0])

    def update_and_get_pose(self):
        """
        Updates the pose with simulated drift and noise, and returns it.
        This simulates a real VO system where each new pose is an estimate
        based on the last one, leading to drift.
        """
        # Get ground truth from the simulator to base our simulation on
        true_pos = self.robot_node.getPosition()
        true_rot_matrix = self.robot_node.getOrientation()
        true_yaw = self._get_yaw_from_orientation(true_rot_matrix)

        # Simulate drift by blending the old pose with the new ground truth
        self.pose['x'] = self.drift_blend * self.pose['x'] + (1 - self.drift_blend) * true_pos[0]
        self.pose['z'] = self.drift_blend * self.pose['z'] + (1 - self.drift_blend) * true_pos[2]
        
        # Blend angles correctly using their complex representation
        noisy_complex = complex(math.cos(self.pose['theta']), math.sin(self.pose['theta']))
        true_complex = complex(math.cos(true_yaw), math.sin(true_yaw))
        blended_complex = self.drift_blend * noisy_complex + (1 - self.drift_blend) * true_complex
        self.pose['theta'] = math.atan2(blended_complex.imag, blended_complex.real)

        # Add small random noise to the final estimate
        self.pose['x'] += np.random.normal(0, self.noise_level_pos)
        self.pose['z'] += np.random.normal(0, self.noise_level_pos)
        self.pose['theta'] += np.random.normal(0, self.noise_level_angle)

        return self.pose


class Pathfinder:
    """A* pathfinding algorithm."""

    def __init__(self, grid):
        self.grid_def = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.obstacles = set()

    def update_obstacles(self, new_obstacles):
        """Add new obstacles to the pathfinder."""
        self.obstacles.update(new_obstacles)

    def is_valid(self, row, col):
        """Check if a cell is valid (within bounds and not an obstacle)."""
        return 0 <= row < self.rows and 0 <= col < self.cols and \
               self.grid_def[row][col] == 0 and (row, col) not in self.obstacles

    @staticmethod
    def heuristic(a, b):
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start, end):
        """Find the shortest path from start to end using A*."""
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {(r, c): float('inf') for r in range(self.rows) for c in range(self.cols)}
        g_score[start] = 0
        f_score = {(r, c): float('inf') for r in range(self.rows) for c in range(self.cols)}
        f_score[start] = self.heuristic(start, end)

        open_set_hash = {start}

        while open_set:
            current = heapq.heappop(open_set)[1]
            open_set_hash.remove(current)

            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # 4 directions
                neighbor = (current[0] + dr, current[1] + dc)
                if self.is_valid(neighbor[0], neighbor[1]):
                    tentative_g_score = g_score[current] + 1
                    if tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, end)
                        if neighbor not in open_set_hash:
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))
                            open_set_hash.add(neighbor)
        return None  # No path found

class CoordinateConverter:
    """Handle conversion between world and grid coordinates."""
    
    @staticmethod
    def world_to_grid(world_x, world_z):
        """Convert world coordinates to grid coordinates."""
        col = round((world_x - GRID_ORIGIN_X) / GRID_CELL_SIZE)
        row = round((world_z - GRID_ORIGIN_Z) / GRID_CELL_SIZE)
        col = max(0, min(col, GRID_COLS - 1))
        row = max(0, min(row, GRID_ROWS - 1))
        return row, col
    
    @staticmethod
    def grid_to_world_center(row, col):
        """Convert grid coordinates to world coordinates (center of cell)."""
        world_x = GRID_ORIGIN_X + col * GRID_CELL_SIZE
        world_z = GRID_ORIGIN_Z + row * GRID_CELL_SIZE
        return world_x, world_z


class ObstacleDetector:
    """Handle obstacle detection using distance sensors."""
    
    def __init__(self):
        self.detected_obstacles = set()
        self.recent_obstacles = []
    
    def process_sensor_readings(self, robot_world_pose, robot_orientation, distance_values):
        """Process distance sensor readings and detect obstacles."""
        if not OBSTACLE_DETECTION_ENABLED:
            return []
        
        new_obstacles = []
        current_row, current_col = CoordinateConverter.world_to_grid(
            robot_world_pose['x'], robot_world_pose['z'])
        
        # Process three sensors: front, front-left, front-right
        for sensor_index, distance_value in enumerate(distance_values):
            if distance_value > DISTANCE_SENSOR_THRESHOLD:
                obstacle_position = self._calculate_obstacle_position(
                    current_row, current_col, robot_orientation, sensor_index)
                
                if self._is_valid_position(obstacle_position):
                    if obstacle_position not in self.detected_obstacles:
                        new_obstacles.append(obstacle_position)
                        self.detected_obstacles.add(obstacle_position)
        
        return new_obstacles
    
    def _calculate_obstacle_position(self, robot_row, robot_col, orientation, sensor_index):
        """Calculate obstacle position based on robot orientation and sensor."""
        theta_degrees = math.degrees(orientation) % 360
        
        # Determine robot facing direction
        if -45 <= theta_degrees <= 45 or 315 <= theta_degrees <= 360:
            # Robot facing RIGHT
            offsets = [(0, 1), (-1, 1), (1, 1)]  # front, front-left, front-right
        elif 45 < theta_degrees <= 135:
            # Robot facing DOWN
            offsets = [(1, 0), (1, 1), (1, -1)]
        elif 135 < theta_degrees <= 225:
            # Robot facing LEFT
            offsets = [(0, -1), (1, -1), (-1, -1)]
        else:
            # Robot facing UP
            offsets = [(-1, 0), (-1, -1), (-1, 1)]
        
        delta_row, delta_col = offsets[sensor_index]
        return (robot_row + delta_row, robot_col + delta_col)
    
    def _is_valid_position(self, position):
        """Check if position is within grid bounds."""
        row, col = position
        return 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS
    
    def get_recent_obstacles(self):
        """Get and clear recent obstacles for transmission."""
        obstacles = self.recent_obstacles.copy()
        self.recent_obstacles.clear()
        return obstacles
    
    def add_recent_obstacles(self, obstacles):
        """Add obstacles to recent list for transmission."""
        self.recent_obstacles.extend(obstacles)


class VisualizationManager:
    """Manage real-time visualization of navigation system."""
    
    def __init__(self):
        self.figure = None
        self.axis = None
        self.robot_trail = []
        self.planned_path = []
        self.obstacle_detector = ObstacleDetector()
    
    def initialize_display(self):
        """Initialize matplotlib visualization."""
        plt.ion()
        self.figure, self.axis = plt.subplots(figsize=(12, 9))
        self.axis.set_aspect('equal')
        self.axis.set_title('Hardware-in-the-Loop Navigation System', fontsize=14, fontweight='bold')
        self.axis.set_xlabel('World X (m)')
        self.axis.set_ylabel('World Z (m)')
        
        self._draw_grid_lines()
        self._setup_display_limits()
        self._create_legend()
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)
    
    def _draw_grid_lines(self):
        """Draw grid overlay on visualization."""
        # Horizontal lines
        for row in range(GRID_ROWS + 1):
            z_coord = GRID_ORIGIN_Z + row * GRID_CELL_SIZE
            self.axis.plot([GRID_ORIGIN_X, GRID_ORIGIN_X + GRID_COLS * GRID_CELL_SIZE], 
                          [z_coord, z_coord], 'k-', alpha=0.2, lw=0.5)
        
        # Vertical lines
        for col in range(GRID_COLS + 1):
            x_coord = GRID_ORIGIN_X + col * GRID_CELL_SIZE
            self.axis.plot([x_coord, x_coord], 
                          [GRID_ORIGIN_Z, GRID_ORIGIN_Z + GRID_ROWS * GRID_CELL_SIZE], 
                          'k-', alpha=0.2, lw=0.5)
    
    def _setup_display_limits(self):
        """Set appropriate display limits with margin."""
        margin = GRID_CELL_SIZE * 2
        self.axis.set_xlim(GRID_ORIGIN_X - margin, GRID_ORIGIN_X + GRID_COLS * GRID_CELL_SIZE + margin)
        self.axis.set_ylim(GRID_ORIGIN_Z - margin, GRID_ORIGIN_Z + GRID_ROWS * GRID_CELL_SIZE + margin)
    
    def _create_legend(self):
        """Create legend for visualization elements."""
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(fc='black', alpha=0.7, label='Black Line Path'),
            Patch(fc='lightgrey', alpha=0.3, label='White Space'),
            Patch(fc='red', alpha=0.7, label='Detected Obstacle'),
            plt.Line2D([0], [0], color='cyan', lw=2, label='Robot Trail'),
            plt.Line2D([0], [0], color='magenta', marker='o', ms=5, ls='--', lw=2, label='Planned Path'),
            plt.Line2D([0], [0], color='red', marker='o', ms=8, ls='', label='Robot Position'),
            plt.Line2D([0], [0], color='green', marker='*', ms=12, ls='', label='Goal Position')
        ]
        self.axis.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    def update_display(self, robot_world_pose, robot_grid_position, line_sensors, planned_path):
        """Update visualization with current system state."""
        if self.figure is None:
            self.initialize_display()
        
        self._clear_dynamic_elements()
        self._draw_grid_cells()
        self._update_robot_trail(robot_world_pose)
        self._draw_planned_path(planned_path)
        self._draw_robot_state(robot_world_pose, robot_grid_position, line_sensors)
        self._draw_goal_position()
        self._display_system_info(robot_grid_position, line_sensors)
        
        plt.draw()
        plt.pause(0.001)
    
    def _clear_dynamic_elements(self):
        """Clear dynamic visualization elements for redraw."""
        num_static_lines = (GRID_ROWS + 1) + (GRID_COLS + 1)
        
        # Clear patches and texts
        for patch in self.axis.patches[:]:
            patch.remove()
        
        texts_to_remove = [t for t in self.axis.texts 
                          if t.get_position()[0] > GRID_ORIGIN_X - GRID_CELL_SIZE]
        for text in texts_to_remove:
            text.remove()
        
        # Clear dynamic lines
        lines_to_remove = self.axis.lines[num_static_lines:]
        for line in lines_to_remove:
            line.remove()
    
    def _draw_grid_cells(self):
        """Draw grid cells with appropriate coloring."""
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                center_x, center_z = CoordinateConverter.grid_to_world_center(row, col)
                
                # Determine cell color based on obstacle status
                if (row, col) in self.obstacle_detector.detected_obstacles:
                    color, alpha = 'red', 0.7
                else:
                    color = 'black' if world_grid[row][col] == 0 else 'lightgrey'
                    alpha = 0.6 if color == 'black' else 0.3
                
                rect = plt.Rectangle(
                    (center_x - GRID_CELL_SIZE/2, center_z - GRID_CELL_SIZE/2),
                    GRID_CELL_SIZE, GRID_CELL_SIZE,
                    facecolor=color, alpha=alpha, edgecolor='gray', linewidth=0.5
                )
                self.axis.add_patch(rect)
    
    def _update_robot_trail(self, robot_world_pose):
        """Update and display robot movement trail."""
        self.robot_trail.append((robot_world_pose['x'], robot_world_pose['z']))
        if len(self.robot_trail) > 200:
            self.robot_trail.pop(0)
        
        if len(self.robot_trail) > 1:
            trail_x, trail_z = zip(*self.robot_trail)
            self.axis.plot(trail_x, trail_z, 'cyan', lw=2, alpha=0.7)
    
    def _draw_planned_path(self, planned_path):
        """Display planned path if available."""
        if planned_path and len(planned_path) > 1:
            self.planned_path = planned_path
            path_world = [CoordinateConverter.grid_to_world_center(r, c) for r, c in planned_path]
            if path_world:
                path_x, path_z = zip(*path_world)
                self.axis.plot(path_x, path_z, 'mo--', lw=2, ms=5, alpha=0.8)
    
    def _draw_robot_state(self, robot_world_pose, robot_grid_position, line_sensors):
        """Draw robot position and orientation."""
        # Robot position
        self.axis.plot(robot_world_pose['x'], robot_world_pose['z'], 'ro', ms=10, 
                      mec='darkred', mew=1)
        
        # Orientation arrow
        arrow_length = GRID_CELL_SIZE * 0.7
        dx = arrow_length * math.cos(robot_world_pose['theta'])
        dz = arrow_length * math.sin(robot_world_pose['theta'])
        arrow = plt.matplotlib.patches.FancyArrowPatch(
            (robot_world_pose['x'], robot_world_pose['z']), 
            (robot_world_pose['x'] + dx, robot_world_pose['z'] + dz),
            arrowstyle='->', mutation_scale=15, color='darkred', lw=2
        )
        self.axis.add_patch(arrow)
        
        # Highlight current grid cell
        if robot_grid_position:
            self._highlight_current_cell(robot_grid_position, line_sensors)
    
    def _highlight_current_cell(self, grid_position, line_sensors):
        """Highlight robot's current grid cell."""
        center_x, center_z = CoordinateConverter.grid_to_world_center(grid_position[0], grid_position[1])
        sensors_active = any(line_sensors)
        
        highlight_color = 'green' if sensors_active else 'yellow'
        highlight_alpha = 0.5 if sensors_active else 0.3
        
        highlight_rect = plt.Rectangle(
            (center_x - GRID_CELL_SIZE/2, center_z - GRID_CELL_SIZE/2),
            GRID_CELL_SIZE, GRID_CELL_SIZE,
            edgecolor=highlight_color, facecolor=highlight_color, 
            alpha=highlight_alpha, linewidth=3
        )
        self.axis.add_patch(highlight_rect)
        
        # Status text
        sensor_status = "ON LINE" if sensors_active else "NO LINE"
        status_color = 'green' if sensors_active else 'orange'
        
        self.axis.text(center_x, center_z + GRID_CELL_SIZE * 0.6, sensor_status, 
                      ha='center', va='bottom', fontsize=7, color=status_color, weight='bold',
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    def _draw_goal_position(self):
        """Draw goal position marker."""
        goal_x, goal_z = CoordinateConverter.grid_to_world_center(GOAL_ROW, GOAL_COL)
        self.axis.plot(goal_x, goal_z, 'g*', ms=15, mec='darkgreen', mew=1.5)
    
    def _display_system_info(self, grid_position, line_sensors):
        """Display system information panel."""
        line_status = "ON BLACK LINE" if any(line_sensors) else "NO LINE DETECTED"
        info_background = 'lightgreen' if any(line_sensors) else 'lightcoral'
        
        info_text = (f"Grid Position: {grid_position} -> Goal: ({GOAL_ROW},{GOAL_COL})\n"
                    f"Sensor Status: {line_status}\n"
                    f"Obstacles Detected: {len(self.obstacle_detector.detected_obstacles)}")
        
        self.axis.text(0.02, 0.98, info_text, transform=self.axis.transAxes, va='top', fontsize=8,
                      bbox=dict(boxstyle='round,pad=0.4', facecolor=info_background, alpha=0.8))


class TurnController:
    """Handle robot turning operations with line detection."""
    
    def __init__(self):
        self.current_phase = 'NONE'
        self.active_command = None
        self.phase_start_time = 0.0
    
    def initiate_turn(self, turn_direction, current_time):
        """Start a new turning operation."""
        if self.active_command != turn_direction or self.current_phase == 'NONE':
            self.active_command = turn_direction
            self.current_phase = 'INITIATE_SPIN'
            self.phase_start_time = current_time
    
    def execute_turn(self, turn_direction, line_sensors, current_time):
        """Execute turn operation and return motor speeds."""
        if self.current_phase == 'INITIATE_SPIN':
            return self._execute_initial_spin(current_time)
        elif self.current_phase == 'SEARCHING_LINE':
            return self._execute_line_search(line_sensors, current_time)
        elif self.current_phase == 'ADJUSTING_ON_LINE':
            return self._execute_line_adjustment(line_sensors, current_time)
        
        return 0.0, 0.0
    
    def _execute_initial_spin(self, current_time):
        """Execute initial spin phase to get off current line."""
        spin_speeds = self._calculate_turn_speeds(0.8, 1.1)
        
        if current_time - self.phase_start_time > MIN_INITIAL_SPIN_DURATION:
            self.current_phase = 'SEARCHING_LINE'
            self.phase_start_time = current_time
        
        return spin_speeds
    
    def _execute_line_search(self, line_sensors, current_time):
        """Search for line during turn operation."""
        search_speeds = self._calculate_turn_speeds(0.5, 0.9)
        
        if any(line_sensors):
            self.current_phase = 'ADJUSTING_ON_LINE'
            self.phase_start_time = current_time
        elif (not TURN_UNTIL_LINE_FOUND and 
              current_time - self.phase_start_time > MAX_SEARCH_SPIN_DURATION):
            self.current_phase = 'NONE'
            return 0.0, 0.0
        
        return search_speeds
    
    def _execute_line_adjustment(self, line_sensors, current_time):
        """Fine-tune position on detected line."""
        left_sensor, center_sensor, right_sensor = line_sensors
        base_speed = TURN_ADJUST_BASE_SPEED
        moderate_diff = MODERATE_CORRECTION_DIFFERENTIAL * (base_speed / FORWARD_SPEED)
        aggressive_diff = AGGRESSIVE_CORRECTION_DIFFERENTIAL * (base_speed / FORWARD_SPEED)
        
        if not left_sensor and center_sensor and not right_sensor:
            # Perfect center - turn complete
            self.current_phase = 'NONE'
            self.active_command = None
            return base_speed * 0.3, base_speed * 0.3
        elif left_sensor and center_sensor and not right_sensor:
            return base_speed - moderate_diff, base_speed
        elif not left_sensor and center_sensor and right_sensor:
            return base_speed, base_speed - moderate_diff
        elif left_sensor and not center_sensor and not right_sensor:
            return base_speed - aggressive_diff, base_speed
        elif not left_sensor and not center_sensor and right_sensor:
            return base_speed, base_speed - aggressive_diff
        elif not any(line_sensors):
            # Line lost - return to search
            self.current_phase = 'SEARCHING_LINE'
            self.phase_start_time = current_time
            return self._calculate_turn_speeds(0.5, 0.9)
        else:
            return base_speed * 0.7, base_speed * 0.7
        
        # Timeout check
        if current_time - self.phase_start_time > MAX_ADJUST_DURATION:
            self.current_phase = 'NONE'
            self.active_command = None
            return 0.0, 0.0
    
    def _calculate_turn_speeds(self, inner_factor, outer_factor):
        """Calculate motor speeds for turning."""
        inner_speed = -FORWARD_SPEED * TURN_SPEED_FACTOR * inner_factor
        outer_speed = FORWARD_SPEED * TURN_SPEED_FACTOR * outer_factor
        
        if self.active_command == 'turn_left':
            return inner_speed, outer_speed
        else:  # turn_right
            return outer_speed, inner_speed
    
    def is_turning(self):
        """Check if currently executing a turn."""
        return self.current_phase != 'NONE'
    
    def reset(self):
        """Reset turn controller state."""
        self.current_phase = 'NONE'
        self.active_command = None


class AutonomousController:
    """Handles path planning and command generation for the robot."""
    def __init__(self, grid, start_pos, goal_pos):
        self.pathfinder = Pathfinder(grid)
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.planned_path = []
        self.current_path_index = 0
        self.robot_command = 'stop'

    def plan_initial_path(self):
        self.planned_path = self.pathfinder.find_path(self.start_pos, self.goal_pos)
        if self.planned_path:
            self.current_path_index = 0
            print(f"Path found: {self.planned_path}")
        else:
            print("No path found!")
        return self.planned_path

    def replan_path(self, current_pos, obstacles):
        self.pathfinder.update_obstacles(obstacles)
        self.planned_path = self.pathfinder.find_path(current_pos, self.goal_pos)
        if self.planned_path:
            self.current_path_index = 0
            print(f"Obstacle detected, replanned path: {self.planned_path}")
        else:
            print("Failed to find new path after obstacle detection.")
            self.robot_command = 'stop'
        return self.planned_path

    def update_command(self, current_grid_pos, last_grid_pos, robot_orientation_rad):
        if not self.planned_path or self.current_path_index >= len(self.planned_path):
            self.robot_command = 'stop'
            return self.robot_command

        # Check if we have arrived at the next waypoint
        if current_grid_pos != last_grid_pos and current_grid_pos == self.planned_path[self.current_path_index]:
            if self.current_path_index >= len(self.planned_path) - 1:
                print("Goal Reached!")
                self.robot_command = 'stop'
                self.planned_path = []
                return self.robot_command

            self.current_path_index += 1
            
            # Determine turn required to face the next waypoint
            p_curr = self.planned_path[self.current_path_index - 1]
            p_next = self.planned_path[self.current_path_index]

            # Simplified turning logic based on orientation
            angle_to_next = math.atan2(p_next[0] - p_curr[0], p_next[1] - p_curr[1])
            robot_angle = robot_orientation_rad
            
            # Adjust angles for grid where +z is down (pi/2) and +x is right (0)
            # This logic is complex, so we use a simpler approach:
            # At an intersection, the TurnController will turn until a new line is found.
            # We just need to give it a direction.
            
            # Determine direction of travel from previous to current
            p_prev = self.planned_path[self.current_path_index - 2] if self.current_path_index > 1 else self.start_pos
            d_in = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
            d_out = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

            # Cross product to find turn direction (z = x1*y2 - x2*y1)
            turn = d_in[1] * d_out[0] - d_in[0] * d_out[1]

            if turn > 0:
                self.robot_command = 'turn_left'
            elif turn < 0:
                self.robot_command = 'turn_right'
            else:
                self.robot_command = 'forward' # Straight
        
        elif not self.is_turning():
            self.robot_command = 'forward'

        return self.robot_command

    def is_turning(self):
        return self.robot_command in ['turn_left', 'turn_right']


def initialize_robot_systems():
    """Initialize all robot hardware systems."""
    # The robot needs to be a Supervisor to get its own pose.
    # Make sure to set the 'supervisor' field of the robot to TRUE in the Webots world.
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    
    # Initialize motors
    left_motor = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')
    for motor in [left_motor, right_motor]:
        motor.setPosition(float('inf'))
        motor.setVelocity(0.0)
    
    # Initialize ground sensors
    ground_sensors = []
    for name in ['gs0', 'gs1', 'gs2']:
        sensor = robot.getDevice(name)
        sensor.enable(timestep)
        ground_sensors.append(sensor)
    
    # Initialize distance sensors
    distance_sensors = []
    sensor_names = ['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7']
    for name in sensor_names:
        sensor = robot.getDevice(name)
        sensor.enable(timestep)
        distance_sensors.append(sensor)
    
    # Select specific sensors for obstacle detection
    obstacle_sensors = [distance_sensors[0], distance_sensors[7], distance_sensors[5]]

    # ---- Initialize Camera ----
    # The camera is enabled and can be used for visualization or real CV.
    camera = robot.getDevice('camera')
    camera.enable(timestep)
    
    return {
        'robot': robot,
        'timestep': timestep,
        'left_motor': left_motor,
        'right_motor': right_motor,
        'ground_sensors': ground_sensors,
        'obstacle_sensors': obstacle_sensors,
        'camera': camera
    }


def calculate_line_following_speeds(line_sensors):
    """Calculate motor speeds for line following behavior."""
    left_sensor, center_sensor, right_sensor = line_sensors
    base_speed = FORWARD_SPEED
    
    if not left_sensor and center_sensor and not right_sensor:
        return base_speed, base_speed
    elif left_sensor and center_sensor and not right_sensor:
        return base_speed - MODERATE_CORRECTION_DIFFERENTIAL, base_speed
    elif not left_sensor and center_sensor and right_sensor:
        return base_speed, base_speed - MODERATE_CORRECTION_DIFFERENTIAL
    elif left_sensor and not center_sensor and not right_sensor:
        return base_speed - AGGRESSIVE_CORRECTION_DIFFERENTIAL, base_speed
    elif not left_sensor and not center_sensor and right_sensor:
        return base_speed, base_speed - AGGRESSIVE_CORRECTION_DIFFERENTIAL
    elif left_sensor and center_sensor and right_sensor:
        return base_speed * 0.7, base_speed * 0.7
    elif not any(line_sensors):
        return base_speed * 0.2, base_speed * 0.2
    else:
        return base_speed * 0.3, base_speed * 0.3


def main():
    """Main program execution loop."""
    # Initialize systems
    hardware = initialize_robot_systems()
    visualization = VisualizationManager()
    turn_controller = TurnController()
    
    # --- Set Robot's Initial Pose ---
    # Get the supervisor and the robot's node to set its position.
    robot_supervisor = hardware['robot']
    robot_node = robot_supervisor.getSelf()
    start_pos_grid = (INITIAL_GRID_ROW, INITIAL_GRID_COL)
    if robot_node:
        # Convert grid coordinates to world coordinates
        start_pos_world_x, start_pos_world_z = CoordinateConverter.grid_to_world_center(*start_pos_grid)
        
        # Get the translation field and set the robot's position
        # We keep the current Y value to avoid putting the robot underground.
        translation_field = robot_node.getField('translation')
        current_translation = translation_field.getSFVec3f()
        translation_field.setSFVec3f([start_pos_world_x, current_translation[1], start_pos_world_z])

        # Get the rotation field and set the robot's orientation to face "down" the grid.
        rotation_field = robot_node.getField('rotation')
        rotation_field.setSFRotation([0, 1, 0, math.pi / 2.0]) # 90 degrees around Y axis
        
        # Allow the simulation to update to reflect the new pose
        robot_supervisor.step(hardware['timestep'])

    # Initialize the simulated Vision Odometry system *after* setting the pose
    vision_odometry = VisionOdometry(hardware['robot'])

    # Setup autonomous controller
    goal_pos = (GOAL_ROW, GOAL_COL)
    controller = AutonomousController(world_grid, start_pos_grid, goal_pos)
    
    # Initialize robot state from the (now correct) pose
    robot_world_pose = vision_odometry.update_and_get_pose()
    
    current_grid_position = CoordinateConverter.world_to_grid(robot_world_pose['x'], robot_world_pose['z'])
    last_grid_position = current_grid_position
    
    planned_path = controller.plan_initial_path()
    
    # Control loop variables
    iteration_count = 0
    last_obstacle_check = 0
    
    print("Hardware-in-the-Loop Navigation System Started")
    
    # Main control loop
    while hardware['robot'].step(hardware['timestep']) != -1:
        current_time = hardware['robot'].getTime()
        iteration_count += 1
        
        # Read sensor data
        line_sensor_values = [s.getValue() for s in hardware['ground_sensors']]
        line_detected = [1 if v < LINE_THRESHOLD else 0 for v in line_sensor_values]
        
        obstacle_sensor_values = [s.getValue() for s in hardware['obstacle_sensors']]
        
        # Update robot position from simulated Vision Odometry
        robot_world_pose = vision_odometry.update_and_get_pose()
        
        last_grid_position = current_grid_position
        current_grid_position = CoordinateConverter.world_to_grid(robot_world_pose['x'], robot_world_pose['z'])
        
        # Process obstacle detection
        if current_time - last_obstacle_check > 0.2:
            new_obstacles = visualization.obstacle_detector.process_sensor_readings(
                robot_world_pose, robot_world_pose['theta'], obstacle_sensor_values)
            if new_obstacles:
                planned_path = controller.replan_path(current_grid_position, 
                                                      visualization.obstacle_detector.detected_obstacles)
            last_obstacle_check = current_time
        
        
        # Get command from autonomous controller
        robot_command = controller.update_command(current_grid_position, last_grid_position, robot_world_pose['theta'])
        
        # If a turn command is issued, let the turn controller handle it.
        if robot_command in ['turn_left', 'turn_right']:
            if not turn_controller.is_turning():
                turn_controller.initiate_turn(robot_command, current_time)
        elif turn_controller.is_turning():
             # If command is no longer turn, but controller is, let it finish.
             pass
        else: # Not turning, command is not turn
            turn_controller.reset()

        # Determine effective command based on sensor state
        effective_command = robot_command
        if turn_controller.is_turning():
            effective_command = turn_controller.active_command
        elif not any(line_detected) and robot_command == 'forward':
            # Lost line while trying to go forward, search for it
            effective_command = 'turn_left' # default search
        
        # Execute movement commands
        left_speed, right_speed = 0.0, 0.0
        
        if effective_command == 'stop':
            turn_controller.reset()
        elif effective_command == 'forward':
            turn_controller.reset()
            left_speed, right_speed = calculate_line_following_speeds(line_detected)
        elif effective_command in ['turn_left', 'turn_right']:
            # The initiate_turn call is now handled above
            left_speed, right_speed = turn_controller.execute_turn(effective_command, line_detected, current_time)
        
        # Apply motor velocities
        hardware['left_motor'].setVelocity(left_speed)
        hardware['right_motor'].setVelocity(right_speed)
        
        # Update visualization
        if iteration_count % 3 == 0:
            visualization.update_display(robot_world_pose, current_grid_position, line_detected, planned_path)
        
        # Status reporting
        if iteration_count % 100 == 0:
            sensor_status = "Line Detected" if any(line_detected) else "No Line"
            print(f"Status: {robot_command} | {sensor_status} | Grid: {current_grid_position} | Next Waypoint: {controller.planned_path[controller.current_path_index] if controller.planned_path else 'N/A'}")
    
    # Cleanup
    if visualization.figure:
        plt.ioff()
        plt.show(block=True)


if __name__ == "__main__":
    main() 