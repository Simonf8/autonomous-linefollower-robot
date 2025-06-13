#!/usr/bin/env python3

import socket
import time
import logging
import cv2
import numpy as np
from ultralytics import YOLO
import math
from collections import deque
import threading
import json
import base64
from flask import Flask, render_template, jsonify, Response, request
import heapq
from typing import List, Tuple, Set, Dict, Optional

# ============================================================================
# Robot Configuration
# ============================================================================
# -- Grid Configuration --
# Define start and goal cells for grid-based placement.
# Coordinates are (column, row) from the top-left of the UNMAPPED maze grid.
CELL_WIDTH_M = 0.12     # Width of a single grid cell in meters (12cm)
_START_CELL_RAW = (0, 14)   # (column, row)
_GOAL_CELL_RAW = (20, 0)     # (column, row)
_START_DIRECTION_RAW = 'UP'  # Initial robot orientation ('UP', 'DOWN', 'LEFT', 'RIGHT')

# -- Map Mirroring Correction --
# The physical maze is a horizontal mirror of the one defined in the code.
# We correct this by flipping the grid and all related coordinates.
MAZE_WIDTH_CELLS = 21
START_CELL = (MAZE_WIDTH_CELLS - 1 - _START_CELL_RAW[0], _START_CELL_RAW[1])
GOAL_CELL = (MAZE_WIDTH_CELLS - 1 - _GOAL_CELL_RAW[0], _GOAL_CELL_RAW[1])

_DIRECTION_FLIP_MAP = {'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
START_DIRECTION = _DIRECTION_FLIP_MAP.get(_START_DIRECTION_RAW.upper(), _START_DIRECTION_RAW)

# -- Odometry Calibration --
# These values MUST be calibrated for your specific robot for accurate tracking.
WHEEL_RADIUS_M = 0.0325         # Wheel radius in meters (3.25 cm)
AXLE_LENGTH_M = 0.15            # Distance between wheels in meters (15 cm)
TICKS_PER_REVOLUTION = 40       # Encoder ticks for one full wheel revolution

# -- PID Controller Configuration --
# These gains MUST be tuned for your specific robot for smooth line following.
KP = 0.015               # Proportional gain: How strongly to react to current error.
KI = 0.008               # Integral gain: Corrects for steady-state error over time.
KD = 0.05                # Derivative gain: Dampens oscillations by anticipating future error.

# -- Computer Vision Path Detection --
# Source points for perspective warp (trapezoid in the original image).
# These points need to be tuned to your camera's mounting angle and position.
# Format: [top-left, top-right, bottom-right, bottom-left]
IMG_PATH_SRC_PTS = np.float32([[200, 300], [440, 300], [580, 480], [60, 480]])
# Destination points for the top-down view.
IMG_PATH_DST_PTS = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])

# -- World Coordinate Configuration (calculated from grid) --
# World coordinates are calculated from the grid cells, using the center of the cell.
# The world origin (0,0) is the top-left corner of the maze.
START_POSITION = ((START_CELL[0] + 0.5) * CELL_WIDTH_M, (START_CELL[1] + 0.5) * CELL_WIDTH_M)

# Calculate heading in radians from the direction string
HEADING_MAP = {
    'UP': -math.pi / 2,    # Negative Y
    'DOWN': math.pi / 2,   # Positive Y
    'LEFT': math.pi,       # Negative X
    'RIGHT': 0             # Positive X
}
START_HEADING = HEADING_MAP.get(START_DIRECTION.upper(), -math.pi / 2) # Default to UP

GOAL_POSITION = ((GOAL_CELL[0] + 0.5) * CELL_WIDTH_M, (GOAL_CELL[1] + 0.5) * CELL_WIDTH_M)
GOAL_THRESHOLD = 0.15          # Stop within 15cm of the goal

WebVisualization = None  # Will be defined below

class ESP32Interface:
    """Simple ESP32 communication for sensor data and motor control"""
    
    def __init__(self, ip_address, port=1234):
        self.ip_address = ip_address
        self.port = port
        self.socket = None
        self.connected = False
        
        # Current sensor state
        self.sensors = [0, 0, 0, 0, 0]  # [L2, L1, C, R1, R2]
        self.line_detected = False
        
        # Encoder data
        self.left_encoder_ticks = 0
        self.right_encoder_ticks = 0
    
    def connect(self):
        """Connect to ESP32"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)  # 5 second timeout instead of 2.0
            self.socket.connect((self.ip_address, self.port))
            self.socket.settimeout(0.1)
            self.connected = True
            return True
        except Exception as e:
            self.connected = False
            return False
    
    def send_motor_speeds(self, left_speed, right_speed):
        """Send motor speeds directly to ESP32"""
        if not self.connected:
            return False
        
        try:
            command = f"{left_speed},{right_speed}"
            self.socket.send(command.encode('utf-8'))
            self.receive_sensor_data()
            return True
        except Exception as e:
            logging.error(f"Failed to send motor speeds: {e}")
            self.connected = False
            return False
        
    def receive_sensor_data(self):
        """Receive and parse sensor data from ESP32"""
        try:
            data = self.socket.recv(128).decode('utf-8').strip()
            if ',' in data:
                parts = data.split(',')
                if len(parts) >= 7: # 5 sensors + 2 encoders
                    self.sensors = [int(float(part)) for part in parts[:5]]
                    self.line_detected = sum(self.sensors) > 0
                    self.left_encoder_ticks = int(parts[5])
                    self.right_encoder_ticks = int(parts[6])
        except socket.timeout:
            pass
        except Exception as e:
            logging.debug(f"Sensor data error: {e}")
    
    def close(self):
        """Close connection"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False

class WheelOdometry:
    """Calculates robot position and heading using wheel encoder data."""

    def __init__(self, initial_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        # Robot parameters are now defined globally for easier calibration
        self.WHEEL_RADIUS = WHEEL_RADIUS_M
        self.AXLE_LENGTH = AXLE_LENGTH_M
        self.TICKS_PER_REVOLUTION = TICKS_PER_REVOLUTION

        # State variables, initialized to the provided start pose
        self.x, self.y, self.heading = initial_pose

        # Previous tick counts
        self.prev_left_ticks = 0
        self.prev_right_ticks = 0
        
        # Calculate distance per tick
        self.DISTANCE_PER_TICK = (2 * math.pi * self.WHEEL_RADIUS) / self.TICKS_PER_REVOLUTION

    def update(self, left_ticks: int, right_ticks: int) -> Tuple[float, float, float]:
        """Update robot pose based on new encoder tick counts."""
        # Calculate tick differences
        delta_left = left_ticks - self.prev_left_ticks
        delta_right = right_ticks - self.prev_right_ticks

        # Update previous tick counts
        self.prev_left_ticks = left_ticks
        self.prev_right_ticks = right_ticks

        # Calculate distance traveled by each wheel
        left_dist = delta_left * self.DISTANCE_PER_TICK
        right_dist = delta_right * self.DISTANCE_PER_TICK

        # Calculate change in distance and heading
        delta_dist = (left_dist + right_dist) / 2.0
        delta_heading = (right_dist - left_dist) / self.AXLE_LENGTH

        # Update pose
        self.x += delta_dist * math.cos(self.heading)
        self.y += delta_dist * math.sin(self.heading)
        self.heading += delta_heading

        # Normalize heading to be within [-pi, pi]
        while self.heading > math.pi: self.heading -= 2 * math.pi
        while self.heading < -math.pi: self.heading += 2 * math.pi
        
        return self.x, self.y, self.heading

class Mapper:
    """Handles the grid map visualization for the web dashboard."""
    
    def __init__(self):
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_heading = 0.0
        self.path_history = []
        self.maze_grid = self.create_maze_grid()
        self.cell_width_m = CELL_WIDTH_M  # Use global cell width

    def create_maze_grid(self):
        """Creates the grid representation of the maze from the simulation."""
        maze = [
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
        # To fix the horizontal mirroring, we flip each row of the maze.
        flipped_maze = [row[::-1] for row in maze]
        return flipped_maze

    def update_robot_position(self, x, y, heading):
        """Update the robot's state and path history."""
        self.robot_x = x
        self.robot_y = y
        self.robot_heading = heading
        self.path_history.append((self.robot_x, self.robot_y))
        if len(self.path_history) > 100: # Keep path history to a reasonable length
            self.path_history.pop(0)
    
    def get_grid_visualization(self, planned_path: Optional[List[Tuple[int, int]]] = None):
        """Create an image of the maze with the robot's position and trail."""
        cell_size = 25
        maze_h, maze_w = len(self.maze_grid), len(self.maze_grid[0])
        img_h, img_w = maze_h * cell_size, maze_w * cell_size
        
        # Create a dark canvas
        vis_img = np.full((img_h, img_w, 3), (10, 10, 10), dtype=np.uint8)

        # Draw the maze paths
        path_color = (80, 80, 80)
        for r in range(maze_h):
            for c in range(maze_w):
                if self.maze_grid[r][c] == 0: # Path
                    cv2.rectangle(vis_img, (c*cell_size, r*cell_size), ((c+1)*cell_size, (r+1)*cell_size), path_color, -1)
        
        # --- Coordinate Conversion Helper ---
        def world_to_pixel(world_x, world_y):
            """Converts world coordinates (meters) to image pixel coordinates."""
            # Assumes world origin (0,0) is the top-left corner of the maze.
            world_width_m = maze_w * self.cell_width_m
            world_height_m = maze_h * self.cell_width_m
            
            px = int((world_x / world_width_m) * img_w)
            py = int((world_y / world_height_m) * img_h)
            return px, py

        # Draw Start and Goal markers
        start_px, start_py = world_to_pixel(START_POSITION[0], START_POSITION[1])
        cv2.circle(vis_img, (start_px, start_py), int(cell_size * 0.5), (0, 255, 0), -1)
        cv2.putText(vis_img, "S", (start_px - 7, start_py + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        
        goal_px, goal_py = world_to_pixel(GOAL_POSITION[0], GOAL_POSITION[1])
        cv2.circle(vis_img, (goal_px, goal_py), int(cell_size * 0.5), (0, 0, 255), -1)
        cv2.putText(vis_img, "G", (goal_px - 7, goal_py + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        # Draw the D* Lite planned path
        if planned_path and len(planned_path) > 1:
            # Convert grid cells to pixel coordinates (centers of cells)
            planned_path_pixels = np.array(
                [world_to_pixel((cell[0] + 0.5) * self.cell_width_m, (cell[1] + 0.5) * self.cell_width_m) for cell in planned_path],
                dtype=np.int32
            )
            cv2.polylines(vis_img, [planned_path_pixels], isClosed=False, color=(255, 255, 0), thickness=1) # Yellow planned path

        # Draw the path history
        if len(self.path_history) > 1:
            path_points = np.array([world_to_pixel(x, y) for x, y in self.path_history], dtype=np.int32)
            cv2.polylines(vis_img, [path_points], isClosed=False, color=(255, 255, 0), thickness=2)

        # Draw the robot
        robot_px, robot_py = world_to_pixel(self.robot_x, self.robot_y)
        cv2.circle(vis_img, (robot_px, robot_py), int(cell_size/2), (255, 0, 255), -1) # Magenta robot

        # Draw robot heading
        arrow_len = cell_size * 0.75
        arrow_end_x = int(robot_px + arrow_len * math.cos(self.robot_heading))
        arrow_end_y = int(robot_py + arrow_len * math.sin(self.robot_heading))
        cv2.arrowedLine(vis_img, (robot_px, robot_py), (arrow_end_x, arrow_end_y), (0, 255, 0), 2)

        return vis_img

class Pathfinder:
    """
    Finds the shortest path using D* Lite and handles dynamic replanning.
    """
    def __init__(self, grid: List[List[int]], start_cell: Tuple[int, int], goal_cell: Tuple[int, int]):
        self.grid = [row[:] for row in grid] # Make a copy
        self.start_cell = start_cell
        self.goal_cell = goal_cell
        
        self.width = len(grid[0])
        self.height = len(grid)
        
        # D* Lite state
        self.g_scores = {}
        self.rhs_scores = {}
        self.U = []  # Priority queue (min-heap)
        
        self.km = 0
        self.last_pos = self.start_cell
        
        # Initialize D* Lite
        self._initialize()

    def _h(self, s1, s2):
        """Heuristic (Manhattan distance)"""
        return abs(s1[0] - s2[0]) + abs(s1[1] - s2[1])

    def _get_g(self, s):
        return self.g_scores.get(s, float('inf'))

    def _get_rhs(self, s):
        return self.rhs_scores.get(s, float('inf'))

    def _calculate_key(self, s):
        h = self._h(self.start_cell, s)
        val = min(self._get_g(s), self._get_rhs(s))
        return (val + h + self.km, val)

    def _initialize(self):
        """Initialize the D* Lite algorithm"""
        self.U = []
        self.km = 0
        self.g_scores = {(c, r): float('inf') for r in range(self.height) for c in range(self.width)}
        self.rhs_scores = {(c, r): float('inf') for r in range(self.height) for c in range(self.width)}
        
        self.rhs_scores[self.goal_cell] = 0
        heapq.heappush(self.U, (self._calculate_key(self.goal_cell), self.goal_cell))

    def _update_vertex(self, u):
        """Update a vertex's position in the priority queue."""
        if u != self.goal_cell:
            min_rhs = float('inf')
            for v in self._get_successors(u):
                min_rhs = min(min_rhs, self._get_g(v) + self._cost(u, v))
            self.rhs_scores[u] = min_rhs

        # Remove from queue if it's there
        self.U = [(key, s) for key, s in self.U if s != u]
        heapq.heapify(self.U)
        
        if self._get_g(u) != self._get_rhs(u):
            heapq.heappush(self.U, (self._calculate_key(u), u))

    def _compute_shortest_path(self):
        """Compute the path until the start node is consistent."""
        while self.U and (heapq.nsmallest(1, self.U)[0][0] < self._calculate_key(self.start_cell) or \
               self._get_rhs(self.start_cell) != self._get_g(self.start_cell)):
            
            key, u = heapq.heappop(self.U)
            
            if self._get_g(u) > self._get_rhs(u):
                self.g_scores[u] = self._get_rhs(u)
                for s_pred in self._get_predecessors(u):
                    self._update_vertex(s_pred)
            else:
                self.g_scores[u] = float('inf')
                self._update_vertex(u)
                for s_pred in self._get_predecessors(u):
                    self._update_vertex(s_pred)

    def _get_neighbors(self, cell):
        """Get valid neighbors (up, down, left, right) of a cell."""
        neighbors = []
        x, y = cell
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny][nx] == 0:
                neighbors.append((nx, ny))
        return neighbors

    def _get_successors(self, cell):
        return self._get_neighbors(cell)
    
    def _get_predecessors(self, cell):
        return self._get_neighbors(cell)

    def _cost(self, s1, s2):
        """Cost between adjacent cells."""
        if self.grid[s2[1]][s2[0]] == 1: # Obstacle
            return float('inf')
        return 1 # Normal movement cost

    def get_path(self) -> Optional[List[Tuple[int, int]]]:
        """Generate the path from start to goal."""
        self._compute_shortest_path()
        
        if self._get_g(self.start_cell) == float('inf'):
            return None # No path exists

        path = [self.start_cell]
        curr = self.start_cell
        
        while curr != self.goal_cell:
            successors = self._get_successors(curr)
            if not successors:
                return None # Dead end
            
            next_cell = min(successors, key=lambda s: self._cost(curr, s) + self._get_g(s))
            path.append(next_cell)
            curr = next_cell
            
            if len(path) > self.width * self.height: # Failsafe
                return None 
                
        return path
    
    def update_obstacle(self, cell_x: int, cell_y: int, is_obstacle: bool):
        """Update the map with a new obstacle and trigger replanning."""
        if 0 <= cell_x < self.width and 0 <= cell_y < self.height:
            # Update grid cost
            self.grid[cell_y][cell_x] = 1 if is_obstacle else 0
            
            # Update the vertex for the changed cell and its neighbors
            self.km += self._h(self.last_pos, self.start_cell)
            self.last_pos = self.start_cell

            self._update_vertex((cell_x, cell_y))
            for neighbor in self._get_neighbors((cell_x, cell_y)):
                self._update_vertex(neighbor)


class RobotController:
    """Controls the robot using D* Lite for pathfinding and YOLO for obstacle detection."""
    
    def __init__(self, esp32_ip):
        self.esp32 = ESP32Interface(esp32_ip)
        
        # -- Navigation State --
        self.path: Optional[List[Tuple[int, int]]] = None
        self.current_waypoint_idx = 0
        self.state = "PLANNING" # PLANNING, NAVIGATING, AT_INTERSECTION, AVOIDING, GOAL_REACHED

        # -- Speed & Control Settings --
        self.base_speed = 32 # Base motor speed for line following
        self.turn_speed_factor = 0.8
        self.sharp_turn_speed = 45
        self.waypoint_threshold = 0.12 # 12cm tolerance for reaching a waypoint
        
        # -- PID Controller State --
        self.integral = 0.0
        self.last_error = 0.0
        self.pid_last_time = time.time()
        
        # -- Object Detection --
        self.camera = None
        self.yolo_model = None
        self.setup_camera_and_yolo()
        self.obstacle_detected = False
        self.visual_turn_cue = "STRAIGHT" # STRAIGHT, CORNER_LEFT, CORNER_RIGHT
        self.perspective_transform_matrix = None
        
        # -- Position & Mapping --
        initial_pose = (START_POSITION[0], START_POSITION[1], START_HEADING)
        self.odometry = WheelOdometry(initial_pose=initial_pose)
        self.mapper = Mapper()
        self.current_position = self.odometry.x, self.odometry.y, self.odometry.heading # x, y, heading
        self.goal_reached = False

        # -- D* Lite Path Planner --
        self.planner = Pathfinder(self.mapper.create_maze_grid(), START_CELL, GOAL_CELL)

    def setup_camera_and_yolo(self):
        """Initialize camera and YOLO model"""
        try:
            # Initialize camera
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 10)  # Lower FPS for better performance
            
            # Load YOLO model
            self.yolo_model = YOLO('yolo11n.pt')
            
            # Create perspective transform for path detection
            self.perspective_transform_matrix = cv2.getPerspectiveTransform(IMG_PATH_SRC_PTS, IMG_PATH_DST_PTS)
            
            logging.info("Camera, YOLO model, and perspective transform initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize camera/YOLO: {e}")
            self.camera = None
            self.yolo_model = None

    def run(self):
        """Main control loop for D* Lite navigation and obstacle avoidance."""
        logging.info("Starting D* Lite navigation controller.")
        
        # Connect to ESP32
        if self.esp32.connect():
            print("ESP32 connected successfully!")
        else:
            print("ESP32 connection failed - continuing without motor control")
        
        # Get initial path
        self.plan_initial_path()

        try:
            while not self.goal_reached:
                # 1. Update robot's current position from odometry
                self.update_position_tracking()

                # 2. Check for obstacles
                self.check_for_obstacles()
                
                # 3. Main navigation logic
                if self.state == "NAVIGATING":
                    self.navigate_path()
                elif self.state == "AT_INTERSECTION":
                    # This state is handled within navigate_path, but motors are stopped briefly
                    pass
                elif self.state == "AVOIDING":
                    # Stop, replan, and then resume navigation
                    self.handle_obstacle_and_replan()
                elif self.state == "PLANNING":
                    # Waiting for a path
                    self.stop_motors()
                
                # 4. Check if goal is reached
                self.check_if_goal_reached()
                
                # 5. Vision processing (run less frequently)
                if not hasattr(self, '_vision_counter'): self._vision_counter = 0
                self._vision_counter += 1
                if self._vision_counter >= 5: # Check every 5 cycles
                    self._vision_counter = 0
                    if self.camera:
                        ret, frame = self.camera.read()
                        if ret:
                            # Run obstacle detection
                            if self.detect_objects_from_frame(frame) and self.state != "AVOIDING":
                                self.state = "AVOIDING"
                            
                            # Run path shape detection
                            self.visual_turn_cue = self.detect_path_shape_from_frame(frame)

                time.sleep(0.05) # 20Hz control loop
                
        except KeyboardInterrupt:
            logging.info("Stopping...")
        finally:
            self.stop()
    
    def plan_initial_path(self):
        """Computes the first path to the goal."""
        print("Planning initial path...")
        self.state = "PLANNING"
        self.path = self.planner.get_path()
        if self.path:
            self.current_waypoint_idx = 0
            self.state = "NAVIGATING"
            print(f"Path found! Length: {len(self.path)} waypoints.")
                            else:
            self.state = "NO_PATH"
            print("No path to goal could be found.")

    def navigate_path(self):
        """Follows the line and uses the D* path to make decisions at intersections."""
        if not self.path or self.current_waypoint_idx >= len(self.path):
            self.state = "GOAL_REACHED"
            self.stop_motors()
            return
            
        # Get current waypoint world coordinates
        waypoint_cell = self.path[self.current_waypoint_idx]
        waypoint_world_x = (waypoint_cell[0] + 0.5) * CELL_WIDTH_M
        waypoint_world_y = (waypoint_cell[1] + 0.5) * CELL_WIDTH_M
        
        # Check distance to the waypoint
        dist_to_waypoint = math.sqrt(
            (self.current_position[0] - waypoint_world_x)**2 +
            (self.current_position[1] - waypoint_world_y)**2
        )
        
        # Detect if we are at an intersection using sensors
        sensors = self.esp32.sensors
        is_intersection = sum(sensors) >= 4 or (sensors[0] and sensors[4])

        # If we are close to a waypoint AND the sensors see an intersection, it's time to decide the next turn
        if dist_to_waypoint < self.waypoint_threshold and is_intersection:
            self.state = "AT_INTERSECTION"
            self.stop_motors()
            time.sleep(0.25) # Pause briefly at the intersection to stabilize

            # Get the direction for the next turn
            turn_direction = self.get_turn_direction_for_next_waypoint()
            print(f"Intersection at {waypoint_cell}. Turning: {turn_direction}")
            
            # Execute the turn, then resume line following
            self.execute_intersection_turn(turn_direction)

            # Move to the next waypoint in the path
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.path):
                self.state = "GOAL_REACHED"
                return
            
            # After turning, go back to navigating/line-following
            self.state = "NAVIGATING"
            return

        # Default behavior: follow the line
        self.line_follower_control_loop()

    def get_turn_direction_for_next_waypoint(self) -> str:
        """Calculates if the next waypoint requires a left, right, or straight movement."""
        # Ensure we have a "next" waypoint to look at
        if self.current_waypoint_idx + 1 >= len(self.path):
            return "STRAIGHT" # Final approach to goal

        # Current robot pose
        robot_x, robot_y, robot_heading = self.current_position
        
        # Next waypoint in world coordinates
        next_waypoint_cell = self.path[self.current_waypoint_idx + 1]
        next_waypoint_world_x = (next_waypoint_cell[0] + 0.5) * CELL_WIDTH_M
        next_waypoint_world_y = (next_waypoint_cell[1] + 0.5) * CELL_WIDTH_M

        # Calculate the angle from the robot to the next waypoint
        angle_to_target = math.atan2(next_waypoint_world_y - robot_y, next_waypoint_world_x - robot_x)
        
        # Calculate the difference between the robot's heading and the target angle
        heading_error = angle_to_target - robot_heading
        while heading_error > math.pi: heading_error -= 2 * math.pi
        while heading_error < -math.pi: heading_error += 2 * math.pi
        
        # Determine turn direction based on the heading error
        if -math.pi / 4 <= heading_error <= math.pi / 4:
            return "STRAIGHT"
        elif heading_error > math.pi / 4:
            return "LEFT"
            else:
            return "RIGHT"
    
    def execute_intersection_turn(self, direction: str):
        """Executes a timed turn at an intersection."""
        # Move forward slightly to enter the intersection center
        self.esp32.send_motor_speeds(self.base_speed, self.base_speed)
        time.sleep(0.3) # Adjust time as needed

        if direction == "LEFT":
            # Pivot turn left
            self.esp32.send_motor_speeds(-self.sharp_turn_speed, self.sharp_turn_speed)
            time.sleep(0.5) # Adjust turn duration
        elif direction == "RIGHT":
            # Pivot turn right
            self.esp32.send_motor_speeds(self.sharp_turn_speed, -self.sharp_turn_speed)
            time.sleep(0.5) # Adjust turn duration
        
        # After turning (or going straight), stop briefly to allow sensors to re-acquire the line
        self.stop_motors()
        time.sleep(0.2)

        # Reset PID controller after a turn to prevent integral windup from the turn
        self.integral = 0.0
        self.last_error = 0.0

    def line_follower_control_loop(self):
        """Controls the robot's motors using a PID controller to follow the line."""
        sensors = self.esp32.sensors
        
        # 1. Calculate the error from the sensor readings
        # A weighted average is used to get a numeric position value from -2 to 2.
        # Negative means the line is to the left, positive means to the right.
        error = 0.0
        num_active_sensors = sum(sensors)
        if num_active_sensors > 0:
            # Weighted average of sensor positions
            # Sensor indices: 0, 1, 2, 3, 4
            # Corresponding weights: -2, -1, 0, 1, 2
            error = ( (sensors[0] * -2) + (sensors[1] * -1) + (sensors[2] * 0) + (sensors[3] * 1) + (sensors[4] * 2) ) / num_active_sensors
            self.last_error = error
        else:
            # If the line is lost, use the last known error to decide which way to turn.
            # A large error value will cause a sharp turn in that direction.
            if self.last_error > 0:
                error = 2.5 
            elif self.last_error < 0:
                error = -2.5
            # If last_error is 0, it will just continue straight briefly.
        
        # 2. PID Calculation
        current_time = time.time()
        dt = current_time - self.pid_last_time
        if dt == 0: dt = 1e-6 # Avoid division by zero

        # Integral term (with anti-windup)
        self.integral += error * dt
        self.integral = max(min(self.integral, 50), -50) # Clamp integral to prevent windup

        # Derivative term
        derivative = (error - self.last_error) / dt
        
        self.last_error = error
        self.pid_last_time = current_time
        
        # 3. Calculate motor speed correction
        correction = (KP * error) + (KI * self.integral) + (KD * derivative)
        
        # 4. Apply correction to motor speeds
        left_speed = self.base_speed - correction
        right_speed = self.base_speed + correction
        
        # Clamp motor speeds to a safe range (e.g., -60 to 60)
        max_speed = 60
        left_speed = max(min(left_speed, max_speed), -max_speed)
        right_speed = max(min(right_speed, max_speed), -max_speed)

        self.esp32.send_motor_speeds(int(left_speed), int(right_speed))

    def navigate_to_point(self, target_x, target_y):
        """Steers the robot towards a specific world coordinate."""
        robot_x, robot_y, robot_heading = self.current_position
        
        # Calculate heading to target
        angle_to_target = math.atan2(target_y - robot_y, target_x - robot_x)
        
        # Calculate heading error
        heading_error = angle_to_target - robot_heading
        # Normalize error to [-pi, pi]
        while heading_error > math.pi: heading_error -= 2 * math.pi
        while heading_error < -math.pi: heading_error += 2 * math.pi
        
        # Proportional controller for turning
        turn_adjustment = heading_error * self.turn_speed_factor
        
        # Slow down if turning sharply
        if abs(heading_error) > math.pi / 4: # 45 degrees
            forward_speed = self.base_speed * 0.5
        else:
            forward_speed = self.base_speed
        
        # Set motor speeds
        left_speed = int(forward_speed - (forward_speed * turn_adjustment))
        right_speed = int(forward_speed + (forward_speed * turn_adjustment))
        
        self.esp32.send_motor_speeds(left_speed, right_speed)

    def check_for_obstacles(self):
        """Periodically checks for obstacles using YOLO."""
                if not hasattr(self, '_detection_counter'):
                    self._detection_counter = 0
                
                self._detection_counter += 1
        if self._detection_counter >= 5: # Check every 5 cycles
                    self._detection_counter = 0
            if self.camera:
                ret, frame = self.camera.read()
                if ret:
                    if self.detect_objects_from_frame(frame):
                        self.obstacle_detected = True
                        self.state = "AVOIDING"
                    else:
                        self.obstacle_detected = False

    def handle_obstacle_and_replan(self):
        """Stops the robot, updates the map, and triggers a replan."""
        print("Obstacle detected! Stopping and replanning...")
        self.stop_motors()
        
        # Estimate obstacle position (cell in front of the robot)
        robot_x, robot_y, robot_heading = self.current_position
        obstacle_dist_m = 0.2 # Assume obstacle is 20cm in front
        
        obstacle_world_x = robot_x + math.cos(robot_heading) * obstacle_dist_m
        obstacle_world_y = robot_y + math.sin(robot_heading) * obstacle_dist_m
        
        obstacle_cell_x = int(obstacle_world_x / CELL_WIDTH_M)
        obstacle_cell_y = int(obstacle_world_y / CELL_WIDTH_M)

        print(f"Updating map. Obstacle at cell: ({obstacle_cell_x}, {obstacle_cell_y})")
        self.planner.update_obstacle(obstacle_cell_x, obstacle_cell_y, is_obstacle=True)
        
        # Replan path
        self.planner.start_cell = self.planner.last_pos # Start replanning from current pos
        new_path = self.planner.get_path()
        if new_path:
            print("New path found!")
            self.path = new_path
            self.current_waypoint_idx = 0 # Start from the beginning of the new path
            self.state = "NAVIGATING"
        else:
            print("Failed to find a new path around the obstacle.")
            self.state = "NO_PATH" # Stuck
        
        self.obstacle_detected = False # Reset detection flag

    def check_if_goal_reached(self):
        """Checks if the robot is within the goal threshold."""
                dist_to_goal = math.sqrt(
                    (self.current_position[0] - GOAL_POSITION[0])**2 +
                    (self.current_position[1] - GOAL_POSITION[1])**2
                )
                if dist_to_goal < GOAL_THRESHOLD:
                    print(f"Goal reached! Distance: {dist_to_goal:.3f}m")
                    self.goal_reached = True
            self.state = "GOAL_REACHED"
            self.stop_motors()
    
    def detect_objects_from_frame(self, frame):
        """Detect objects from provided frame (for efficiency)"""
        if not self.yolo_model:
            return False

        try:
            # Run YOLO detection
            results = self.yolo_model(frame, verbose=False)
            
            # Objects to ignore (not real obstacles)
            ignore_objects = {
                'tie', 'necktie', 'person', 'chair', 'dining table', 'laptop', 
                'mouse', 'remote', 'keyboard', 'cell phone', 'book', 'clock',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            }
            
            # Objects that ARE obstacles (things robot should avoid)
            obstacle_objects = {
                'bottle', 'cup', 'bowl', 'banana', 'apple', 'orange', 'broccoli',
                'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'potted plant',
                'vase', 'backpack', 'handbag', 'suitcase', 'sports ball',
                'baseball bat', 'skateboard', 'surfboard', 'tennis racket'
            }
            
            # Check if any obstacle objects detected with confidence > 0.6
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        confidence = float(box.conf[0])
                        if confidence > 0.6:  # Higher confidence for obstacles
                            class_id = int(box.cls[0])
                            class_name = self.yolo_model.names[class_id]
                            
                            # Only trigger on actual obstacles, ignore ties and other non-obstacles
                            if class_name.lower() in obstacle_objects:
                                logging.info(f"Obstacle detected: {class_name} (confidence: {confidence:.2f})")
                                self.state = "AVOIDING" # Set state to avoiding
                                return True
                            elif class_name.lower() in ignore_objects:
                                logging.debug(f"Ignoring non-obstacle: {class_name} (confidence: {confidence:.2f})")
                            else:
                                # Unknown object - be cautious and treat as obstacle
                                logging.info(f"Unknown object detected: {class_name} (confidence: {confidence:.2f})")
                                self.state = "AVOIDING" # Set state to avoiding
                                return True
            
            return False

        except Exception as e:
            logging.debug(f"Object detection error: {e}")
            return False

    def update_position_tracking(self):
        """Update robot's position using data from wheel odometry."""
        left_ticks = self.esp32.left_encoder_ticks
        right_ticks = self.esp32.right_encoder_ticks
        
        # Update odometry and get new position
        x, y, heading = self.odometry.update(left_ticks, right_ticks)
        self.current_position = (x, y, heading)
        
        # Update the mapper for visualization
        self.mapper.update_robot_position(x, y, heading)

    def stop_motors(self):
        """Convenience function to stop motors."""
        self.esp32.send_motor_speeds(0, 0)
    
    def stop(self):
        """Stop the robot and cleanup resources"""
        self.stop_motors()
        self.esp32.close()
        
        # Cleanup camera
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        logging.info("Camera resources cleaned up")

    def detect_path_shape_from_frame(self, frame):
        """Analyzes the camera frame to detect upcoming corners or path shape."""
        if self.perspective_transform_matrix is None:
            return "UNKNOWN"

        try:
            # 1. Warp the image to a top-down bird's-eye view
            warped_img = cv2.warpPerspective(frame, self.perspective_transform_matrix, (frame.shape[1], frame.shape[0]))
            
            # 2. Convert to grayscale and find edges
            gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            # 3. Use Hough Line Transform to find lines in the path
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=50, maxLineGap=20)
            
            if lines is None:
                return "STRAIGHT" # Assume straight if no lines detected

            # 4. Analyze the angles of detected lines to infer path shape
            left_lines, right_lines = [], []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0: continue # Skip vertical lines to avoid division by zero
                
                angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
                
                # Filter lines based on their angle to classify them as left or right boundaries
                if angle > -80 and angle < -10:
                    left_lines.append(line)
                elif angle > 10 and angle < 80:
                    right_lines.append(line)

            # 5. Simple heuristic to decide if a corner is ahead.
            # If there's a strong imbalance in detected left vs. right lines, it implies a turn.
            if len(right_lines) > len(left_lines) + 5: # More vertical lines on right -> left turn
                logging.info("Visual Cue: Possible LEFT turn ahead.")
                return "CORNER_LEFT"
            elif len(left_lines) > len(right_lines) + 5: # More vertical lines on left -> right turn
                logging.info("Visual Cue: Possible RIGHT turn ahead.")
                return "CORNER_RIGHT"
                
            return "STRAIGHT"

        except Exception as e:
            logging.debug(f"Path shape detection error: {e}")
            return "UNKNOWN"

class WebVisualization:
    """Flask web app for visualizing robot state with cyberpunk theme"""
    
    def __init__(self, robot):
        self.robot = robot
        self.app = Flask(__name__, template_folder='../../templates', static_folder='../../static')
        self.app.config['SECRET_KEY'] = 'robot_visualization_key'
        
        # Setup routes
        self.setup_routes()
        
        # Data update thread
        self.running = True
        self.update_thread = None
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('navigation.html')
        
        @self.app.route('/api/robot_data')
        def get_robot_data():
            """Get current robot data as JSON"""
            # Get grid visualization
            grid_img = self.robot.mapper.get_grid_visualization(planned_path=self.robot.path)
            _, buffer = cv2.imencode('.png', grid_img)
            grid_base64 = base64.b64encode(buffer).decode('utf-8')

            data = {
                'state': self.robot.state,
                'sensors': self.robot.esp32.sensors,
                'position': {
                    'x': self.robot.current_position[0],
                    'y': self.robot.current_position[1],
                    'heading': self.robot.current_position[2]
                },
                'grid_image': grid_base64
            }
            return jsonify(data)
        
        @self.app.route('/video_feed')
        def video_feed():
            """Video streaming route"""
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def generate_frames(self):
        """Generate camera frames for video streaming"""
        while True:
            try:
                if self.robot.camera is not None:
                    success, frame = self.robot.camera.read()
                    if success:
                        # Add cyberpunk overlay effects
                        frame = self.add_cyberpunk_overlay(frame)
                        
                        # Encode frame
                        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ret:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    else:
                        # Camera error - send black frame
                        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(black_frame, 'CAMERA OFFLINE', (200, 240), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        ret, buffer = cv2.imencode('.jpg', black_frame)
                        if ret:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # No camera - send placeholder
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, 'NO CAMERA DETECTED', (180, 240), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', placeholder)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.1)  # 10 FPS for web streaming
                
            except Exception as e:
                logging.error(f"Video streaming error: {e}")
                time.sleep(1)
    
    def add_cyberpunk_overlay(self, frame):
        """Add cyberpunk-style overlay to camera frame"""
        try:
            height, width = frame.shape[:2]
            
            # Add subtle cyan tint
            overlay = frame.copy()
            overlay[:, :, 1] = np.minimum(overlay[:, :, 1] + 10, 255)  # Slight green boost
            overlay[:, :, 2] = np.minimum(overlay[:, :, 2] + 5, 255)   # Slight red boost
            frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
            
            # Add corner brackets
            bracket_length = 30
            bracket_thickness = 2
            bracket_color = (0, 255, 255)  # Cyan
            
            # Top-left
            cv2.line(frame, (10, 10), (10 + bracket_length, 10), bracket_color, bracket_thickness)
            cv2.line(frame, (10, 10), (10, 10 + bracket_length), bracket_color, bracket_thickness)
            
            # Top-right
            cv2.line(frame, (width-10, 10), (width-10-bracket_length, 10), bracket_color, bracket_thickness)
            cv2.line(frame, (width-10, 10), (width-10, 10 + bracket_length), bracket_color, bracket_thickness)
            
            # Bottom-left
            cv2.line(frame, (10, height-10), (10 + bracket_length, height-10), bracket_color, bracket_thickness)
            cv2.line(frame, (10, height-10), (10, height-10-bracket_length), bracket_color, bracket_thickness)
            
            # Bottom-right
            cv2.line(frame, (width-10, height-10), (width-10-bracket_length, height-10), bracket_color, bracket_thickness)
            cv2.line(frame, (width-10, height-10), (width-10, height-10-bracket_length), bracket_color, bracket_thickness)
            
            # Add sensor overlay
            sensor_text = f"SENSORS: {self.robot.esp32.sensors}"
            cv2.putText(frame, sensor_text, (10, height-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Add state overlay
            state_text = f"STATE: {self.robot.state}"
            cv2.putText(frame, state_text, (10, height-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Add crosshair in center
            center_x, center_y = width // 2, height // 2
            crosshair_size = 20
            cv2.line(frame, (center_x - crosshair_size, center_y), 
                    (center_x + crosshair_size, center_y), (0, 255, 255), 1)
            cv2.line(frame, (center_x, center_y - crosshair_size), 
                    (center_x, center_y + crosshair_size), (0, 255, 255), 1)
            cv2.circle(frame, (center_x, center_y), crosshair_size, (0, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            logging.error(f"Overlay error: {e}")
            return frame
    
    def start_web_server(self, host='0.0.0.0', port=5000):
        """Start the Flask web server"""
        try:
            logging.info(f"Starting cyberpunk web dashboard at http://{host}:{port}")
            self.app.run(host=host, port=port, debug=False, threaded=True)
        except Exception as e:
            logging.error(f"Failed to start web server: {e}")
    
    def stop(self):
        """Stop the web server"""
        self.running = False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Replace with your ESP32 IP
    robot = RobotController("192.168.2.21")
    
    # Create and start web visualization automatically
    try:
        web_viz = WebVisualization(robot)
        print("Cyberpunk web visualization initialized")
        
        # Start web server automatically in background
        web_thread = threading.Thread(target=web_viz.start_web_server, daemon=True)
        web_thread.start()
        print("Cyberpunk dashboard started at http://0.0.0.0:5000")
    except Exception as e:
        print(f"Could not initialize web visualization: {e}")
        web_viz = None
    
    print("\n" + "="*60)
    print("Robot running with odometry tracking.")
    print("="*60)
    
    print("╔════════════════════════════════════════════════════════════╗")
    print("║            CYBERPUNK ROBOT NAVIGATION MATRIX              ║")
    print("╠════════════════════════════════════════════════════════════╣")
    print("║ > D* Lite pathfinding with dynamic replanning             ║")
    print("║ > Hybrid Navigation (PID Line Following + Waypoints)      ║")
    print("║ > Odometry-based position tracking                        ║")
    print("║ > YOLOv8n obstacle detection with evasive maneuvers      ║")
    print("║ > Live camera feed with cyberpunk overlay effects        ║")
    print("║ > Interactive dashboard at http://192.168.2.20:5000      ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print("SYSTEM STATUS: READY FOR DEPLOYMENT")
    print("Press Ctrl+C to terminate")
    
    # Start robot
    try:
        robot.run() 
    except KeyboardInterrupt:
        print("\nSYSTEM SHUTDOWN INITIATED...")
        robot.stop()
        if web_viz:
            web_viz.stop()