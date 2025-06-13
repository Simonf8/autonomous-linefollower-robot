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
# SIMPLE NAVIGATION CONTROL - Set your start and goal positions here
# ============================================================================

def setup_navigation(robot):
    """Simple function to set up D* navigation - edit these coordinates"""
    
    print("\nNAVIGATION SETUP")
    print("Available positions in your maze:")
    print("  Top-left corner: (0.06, 0.30)")
    print("  Top-right corner: (2.46, 0.30)")
    print("  Center: (1.26, 0.90)")
    print("  Bottom-left: (0.06, 1.50)")
    print("  Bottom-right: (2.46, 1.50)")
    
    # EDIT THESE COORDINATES TO SET YOUR START AND GOAL
    start_position = (0.06, 0.30)   # Top-left corner
    goal_position = (2.46, 1.50)   # Bottom-right corner
    
    print(f"Setting START: {start_position}")
    print(f"Setting GOAL: {goal_position}")
    
    # Set the navigation goal
    path = robot.set_navigation_goal(start_position, goal_position)
    if path:
        print(f" D* Navigation ready! Path has {len(path)} waypoints")
        robot.enable_dstar_navigation()
        return True
    else:
        print("Failed to create navigation path")
        return False

# ============================================================================

# Web visualization is now integrated in this file
WebVisualization = None  # Will be defined below

class DStarLite:
    """D* Lite algorithm for dynamic pathfinding in robotics"""
    
    def __init__(self, maze_grid: List[List[int]], cell_size: float = 0.12):
        """
        Initialize D* Lite pathfinder
        
        Args:
            maze_grid: 2D grid where 0 = navigable, 1 = obstacle
            cell_size: Size of each cell in meters (default 12cm)
        """
        self.maze = maze_grid
        self.height = len(maze_grid)
        self.width = len(maze_grid[0])
        self.cell_size = cell_size
        
        # D* Lite data structures
        self.U = []  # Priority queue
        self.rhs = {}  # Right-hand side values
        self.g = {}  # Cost-to-come values
        self.km = 0  # Key modifier
        
        # Robot state
        self.start = None
        self.goal = None
        self.last_start = None
        
        # Path storage
        self.current_path = []
        self.path_changed = False
        
        # Initialize all cells
        self._initialize_grid()
        
        # D* Lite initialized
    
    def _initialize_grid(self):
        """Initialize g and rhs values for all cells"""
        for y in range(self.height):
            for x in range(self.width):
                self.g[(x, y)] = float('inf')
                self.rhs[(x, y)] = float('inf')
    
    def _is_valid_cell(self, x: int, y: int) -> bool:
        """Check if cell coordinates are valid and navigable"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.maze[y][x] == 0  # 0 = navigable, 1 = obstacle
        return False
    
    def _get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighboring cells - only allow movement along line paths"""
        neighbors = []
        
        # Only allow 4-connected movement (no diagonal) to stay on line paths
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down, right, left
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self._is_valid_cell(nx, ny):
                # Additional check: ensure we're moving along a continuous line path
                if self._is_line_path_connection(x, y, nx, ny):
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def _is_line_path_connection(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if two cells are connected by a valid line path"""
        # Both cells must be navigable
        if not (self._is_valid_cell(x1, y1) and self._is_valid_cell(x2, y2)):
            return False
        
        # Check if this is an intersection point (allows transitions between line segments)
        if self._is_intersection(x1, y1) or self._is_intersection(x2, y2):
            return True
        
        # Check if this is a horizontal or vertical line segment connection
        if x1 == x2:  # Vertical movement
            # Check if we're on the same vertical line segment
            return self._are_on_same_vertical_line(x1, y1, y2)
        elif y1 == y2:  # Horizontal movement
            # Check if we're on the same horizontal line segment
            return self._are_on_same_horizontal_line(y1, x1, x2)
        
        return False
    
    def _are_on_same_horizontal_line(self, row: int, x1: int, x2: int) -> bool:
        """Check if two x positions are on the same horizontal line segment"""
        if row < 0 or row >= self.height:
            return False
        
        # Find the start and end of the line segment containing x1
        start_x = x1
        while start_x > 0 and self.maze[row][start_x - 1] == 0:
            start_x -= 1
        
        end_x = x1
        while end_x < self.width - 1 and self.maze[row][end_x + 1] == 0:
            end_x += 1
        
        # Check if x2 is within the same line segment
        return start_x <= x2 <= end_x
    
    def _are_on_same_vertical_line(self, col: int, y1: int, y2: int) -> bool:
        """Check if two y positions are on the same vertical line segment"""
        if col < 0 or col >= self.width:
            return False
        
        # Find the start and end of the line segment containing y1
        start_y = y1
        while start_y > 0 and self.maze[start_y - 1][col] == 0:
            start_y -= 1
        
        end_y = y1
        while end_y < self.height - 1 and self.maze[end_y + 1][col] == 0:
            end_y += 1
        
        # Check if y2 is within the same line segment
        return start_y <= y2 <= end_y
    
    def _cost(self, from_cell: Tuple[int, int], to_cell: Tuple[int, int]) -> float:
        """Calculate movement cost between adjacent cells - prefer straight line movement"""
        x1, y1 = from_cell
        x2, y2 = to_cell
        
        # Only horizontal and vertical movement allowed (no diagonal)
        if abs(x2 - x1) + abs(y2 - y1) != 1:
            return float('inf')  # Invalid movement
        
        return 1.0  # All valid movements have equal cost
    
    def _heuristic(self, from_cell: Tuple[int, int], to_cell: Tuple[int, int]) -> float:
        """Heuristic function (Euclidean distance)"""
        x1, y1 = from_cell
        x2, y2 = to_cell
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _calculate_key(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        """Calculate priority key for a cell"""
        g_val = self.g.get(cell, float('inf'))
        rhs_val = self.rhs.get(cell, float('inf'))
        
        min_val = min(g_val, rhs_val)
        heuristic_val = self._heuristic(cell, self.start) if self.start else 0
        
        return (min_val + heuristic_val + self.km, min_val)
    
    def _update_vertex(self, cell: Tuple[int, int]):
        """Update vertex in the priority queue"""
        # Remove from queue if present
        self.U = [(key, c) for key, c in self.U if c != cell]
        heapq.heapify(self.U)
        
        # If locally inconsistent, add to queue
        if self.g.get(cell, float('inf')) != self.rhs.get(cell, float('inf')):
            key = self._calculate_key(cell)
            heapq.heappush(self.U, (key, cell))
    
    def _compute_shortest_path(self):
        """Main D* Lite computation"""
        while (len(self.U) > 0 and 
               (self.U[0][0] < self._calculate_key(self.start) or 
                self.rhs.get(self.start, float('inf')) != self.g.get(self.start, float('inf')))):
            
            k_old, u = heapq.heappop(self.U)
            k_new = self._calculate_key(u)
            
            if k_old < k_new:
                heapq.heappush(self.U, (k_new, u))
            elif self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs[u]
                for neighbor in self._get_neighbors(u[0], u[1]):
                    if neighbor != self.goal:
                        cost = self._cost(neighbor, u)
                        self.rhs[neighbor] = min(self.rhs.get(neighbor, float('inf')), 
                                               self.g.get(u, float('inf')) + cost)
                    self._update_vertex(neighbor)
            else:
                g_old = self.g[u]
                self.g[u] = float('inf')
                
                cells_to_update = self._get_neighbors(u[0], u[1]) + [u]
                for cell in cells_to_update:
                    if (cell != self.goal and 
                        self.rhs.get(cell, float('inf')) == g_old + self._cost(cell, u)):
                        
                        if cell != self.goal:
                            min_rhs = float('inf')
                            for neighbor in self._get_neighbors(cell[0], cell[1]):
                                cost = self._cost(cell, neighbor)
                                min_rhs = min(min_rhs, self.g.get(neighbor, float('inf')) + cost)
                            self.rhs[cell] = min_rhs
                    
                    self._update_vertex(cell)
    
    def set_start_goal(self, start_world: Tuple[float, float], goal_world: Tuple[float, float]):
        """Set start and goal positions in world coordinates"""
        # Convert world coordinates to grid coordinates
        start_grid = (int(start_world[0] / self.cell_size), int(start_world[1] / self.cell_size))
        goal_grid = (int(goal_world[0] / self.cell_size), int(goal_world[1] / self.cell_size))
        
        # Validate coordinates
        if not self._is_valid_cell(start_grid[0], start_grid[1]):
            print(f"Warning: Start position {start_grid} is not valid, finding nearest valid cell")
            start_grid = self._find_nearest_valid_cell(start_grid)
        
        if not self._is_valid_cell(goal_grid[0], goal_grid[1]):
            print(f"Warning: Goal position {goal_grid} is not valid, finding nearest valid cell")
            goal_grid = self._find_nearest_valid_cell(goal_grid)
        
        self.start = start_grid
        self.goal = goal_grid
        
        # D* Lite positions set
        
        # Initialize goal
        self.rhs[self.goal] = 0
        self._update_vertex(self.goal)
        
        # Compute initial path
        self._compute_shortest_path()
        self._extract_path()
    
    def _find_nearest_valid_cell(self, target: Tuple[int, int]) -> Tuple[int, int]:
        """Find the nearest valid (navigable) cell to the target"""
        x, y = target
        
        # Search in expanding squares
        for radius in range(1, max(self.width, self.height)):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:  # Only check perimeter
                        nx, ny = x + dx, y + dy
                        if self._is_valid_cell(nx, ny):
                            return (nx, ny)
        
        # Fallback: return first valid cell found
        for y in range(self.height):
            for x in range(self.width):
                if self._is_valid_cell(x, y):
                    return (x, y)
        
        raise ValueError("No valid cells found in maze!")
    
    def _extract_path(self):
        """Extract the optimal path from start to goal"""
        if not self.start or not self.goal:
            return
        
        path = []
        current = self.start
        
        # Follow the path by choosing the neighbor with minimum g + cost
        while current != self.goal:
            path.append(current)
            
            best_neighbor = None
            best_cost = float('inf')
            
            for neighbor in self._get_neighbors(current[0], current[1]):
                cost = self.g.get(neighbor, float('inf')) + self._cost(current, neighbor)
                if cost < best_cost:
                    best_cost = cost
                    best_neighbor = neighbor
            
            if best_neighbor is None:
                # No path found
                break
            
            current = best_neighbor
            
            # Prevent infinite loops
            if len(path) > self.width * self.height:
                # Path too long, stopping
                break
        
        path.append(self.goal)
        
        # Convert grid coordinates back to world coordinates
        world_path = []
        for x, y in path:
            world_x = (x + 0.5) * self.cell_size  # Center of cell
            world_y = (y + 0.5) * self.cell_size
            world_path.append((world_x, world_y))
        
        self.current_path = world_path
        self.path_changed = True
        
        # Path found
        return world_path
    
    def update_robot_position(self, new_position: Tuple[float, float]):
        """Update robot position and replan if necessary"""
        # Convert to grid coordinates
        new_start = (int(new_position[0] / self.cell_size), int(new_position[1] / self.cell_size))
        
        if new_start != self.start and self._is_valid_cell(new_start[0], new_start[1]):
            # Update km for replanning
            if self.last_start:
                self.km += self._heuristic(self.last_start, new_start)
            
            self.last_start = self.start
            self.start = new_start
            
            # Recompute path
            self._compute_shortest_path()
            self._extract_path()
    
    def update_obstacles(self, changed_cells: List[Tuple[int, int, int]]):
        """Update obstacle information and replan
        
        Args:
            changed_cells: List of (x, y, new_value) where new_value is 0 (free) or 1 (obstacle)
        """
        path_needs_update = False
        
        for x, y, new_value in changed_cells:
            if 0 <= x < self.width and 0 <= y < self.height:
                old_value = self.maze[y][x]
                self.maze[y][x] = new_value
                
                if old_value != new_value:
                    path_needs_update = True
                    cell = (x, y)
                    
                    # Update rhs values for affected cells
                    affected_cells = self._get_neighbors(x, y) + [cell]
                    for affected in affected_cells:
                        if affected != self.goal:
                            min_rhs = float('inf')
                            for neighbor in self._get_neighbors(affected[0], affected[1]):
                                if self._is_valid_cell(neighbor[0], neighbor[1]):
                                    cost = self._cost(affected, neighbor)
                                    min_rhs = min(min_rhs, self.g.get(neighbor, float('inf')) + cost)
                            self.rhs[affected] = min_rhs
                        
                        self._update_vertex(affected)
        
        if path_needs_update:
            self._compute_shortest_path()
            self._extract_path()
            # Replanned due to obstacles
    
    def get_path(self) -> List[Tuple[float, float]]:
        """Get the current optimal path in world coordinates"""
        return self.current_path.copy()
    
    def get_next_waypoint(self, current_pos: Tuple[float, float], lookahead_distance: float = 0.2) -> Optional[Tuple[float, float]]:
        """Get the next waypoint for the robot to follow
        
        Args:
            current_pos: Current robot position in world coordinates
            lookahead_distance: How far ahead to look for the next waypoint
        """
        if not self.current_path:
            return None
        
        # Find the closest point on the path
        min_distance = float('inf')
        closest_index = 0
        
        for i, waypoint in enumerate(self.current_path):
            distance = math.sqrt((waypoint[0] - current_pos[0])**2 + (waypoint[1] - current_pos[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        # Look ahead from the closest point
        for i in range(closest_index, len(self.current_path)):
            waypoint = self.current_path[i]
            distance = math.sqrt((waypoint[0] - current_pos[0])**2 + (waypoint[1] - current_pos[1])**2)
            
            if distance >= lookahead_distance:
                return waypoint
        
        # If no waypoint is far enough, return the last waypoint
        return self.current_path[-1] if self.current_path else None
    
    def _is_intersection(self, x: int, y: int) -> bool:
        """Check if a cell is an intersection point where multiple line segments meet"""
        if not self._is_valid_cell(x, y):
            return False
        
        # Count the number of valid directions from this cell
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down, right, left
        valid_directions = 0
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self._is_valid_cell(nx, ny):
                valid_directions += 1
        
        # An intersection has 3 or more valid directions
        return valid_directions >= 3

class ESP32Interface:
    """Simple ESP32 communication for sensor data and motor control"""
    
    def __init__(self, ip_address, port=1234):
        self.ip_address = ip_address
        self.port = port
        self.socket = None
        self.connected = False
        
        # Current sensor state - use raw sensor array
        self.sensors = [0, 0, 0, 0, 0]  # [L2, L1, C, R1, R2]
        self.line_detected = False
    
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
                if len(parts) >= 5:
                    self.sensors = [int(float(part)) for part in parts[:5]]
                    self.line_detected = sum(self.sensors) > 0
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

class VisualOdometry:
    """Enhanced visual odometry optimized for structured maze environments"""
    
    def __init__(self):
        # Feature detection parameters - optimized for maze corridors
        self.feature_params = dict(
            maxCorners=50,  # Fewer features for cleaner tracking
            qualityLevel=0.4,  # Higher quality threshold
            minDistance=15,  # More spacing between features
            blockSize=7
        )
        
        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),  # Larger window for better tracking
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
        )
        
        # State variables
        self.prev_gray = None
        self.prev_pts = None
        
        # Position tracking with calibrated scale
        self.x = 0.0  # X position in meters
        self.y = 0.0  # Y position in meters
        self.heading = 0.0  # Heading in radians
        
        # Calibration parameters for your track  
        self.pixel_to_meter_scale = 0.004  # 4mm per pixel (calibrated for line following)
        self.heading_smoothing = 0.08  # Moderate heading smoothing for line following
        
        # Motion filtering
        self.motion_history = deque(maxlen=3)  # Shorter history for faster response
        self.min_motion_threshold = 2.0  # Minimum pixel motion to register
        
        # Drift correction for structured environment
        self.position_drift_correction = 0.95  # Slight drift correction
        self.heading_drift_correction = 0.98
        
    def update(self, frame):
        """Update position based on optical flow - simplified for stability"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            # Initialize with first frame
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            return self.x, self.y, self.heading
        
        if self.prev_pts is not None and len(self.prev_pts) > 0:
            # Calculate optical flow
            next_pts, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_pts, None, **self.lk_params
            )
            
            # Simple good point selection
            if next_pts is not None and status is not None:
                # Get only good tracking points
                good_indices = status.reshape(-1) == 1
                
                if np.sum(good_indices) >= 5:  # Need minimum points for reliable tracking
                    good_new = next_pts[good_indices]
                    good_old = self.prev_pts[good_indices]
                    
                    # Calculate simple average motion
                    motion_vectors = good_new.reshape(-1, 2) - good_old.reshape(-1, 2)
                    
                    # Use median for robustness against outliers
                    dx = np.median(motion_vectors[:, 0])
                    dy = np.median(motion_vectors[:, 1])
                    
                    # Only update if motion is significant enough
                    motion_magnitude = math.sqrt(dx*dx + dy*dy)
                    if motion_magnitude > self.min_motion_threshold:
                        
                        # Convert to world coordinates with calibrated scale
                        world_dx = dx * self.pixel_to_meter_scale
                        world_dy = dy * self.pixel_to_meter_scale
                        
                        # Apply drift correction
                        world_dx *= self.position_drift_correction
                        world_dy *= self.position_drift_correction
                        
                        # Update position in robot's reference frame
                        # Forward is positive X, left is positive Y in robot frame
                        delta_forward = -world_dy  # Camera Y is inverted to robot forward
                        delta_left = -world_dx     # Camera X is inverted to robot left
                        
                        # Transform to world coordinates
                        world_delta_x = delta_forward * math.cos(self.heading) - delta_left * math.sin(self.heading)
                        world_delta_y = delta_forward * math.sin(self.heading) + delta_left * math.cos(self.heading)
                        
                        self.x += world_delta_x
                        self.y += world_delta_y
                        
                        # Simple heading estimation
                        if len(good_new) >= 8:
                            # Calculate rotation from average point cloud shift
                            center_old = np.mean(good_old.reshape(-1, 2), axis=0)
                            center_new = np.mean(good_new.reshape(-1, 2), axis=0)
                            
                            # Simple rotation estimation using a few points
                            rotation_angles = []
                            for i in range(min(5, len(good_new))):
                                old_vec = good_old.reshape(-1, 2)[i] - center_old
                                new_vec = good_new.reshape(-1, 2)[i] - center_new
                                
                                old_norm = np.linalg.norm(old_vec)
                                new_norm = np.linalg.norm(new_vec)
                                
                                if old_norm > 5 and new_norm > 5:  # Avoid noise from small vectors
                                    # Calculate angle between vectors
                                    cos_angle = np.dot(old_vec, new_vec) / (old_norm * new_norm)
                                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                                    
                                    # Determine rotation direction
                                    cross_product = np.cross(old_vec, new_vec)
                                    if abs(cross_product) > 1e-6:  # Avoid division by very small numbers
                                        angle = math.acos(cos_angle) * np.sign(cross_product)
                                        rotation_angles.append(angle)
                            
                            if len(rotation_angles) > 0:
                                avg_rotation = np.median(rotation_angles)  # Use median for robustness
                                # Apply smoothing and drift correction
                                self.heading += avg_rotation * self.heading_smoothing * self.heading_drift_correction
                                
                                # Normalize heading to [-pi, pi]
                                while self.heading > math.pi:
                                    self.heading -= 2 * math.pi
                                while self.heading < -math.pi:
                                    self.heading += 2 * math.pi
                        
                        # Store motion for analysis
                        self.motion_history.append((world_delta_x, world_delta_y))
        
        # Refresh feature points when needed
        if self.prev_pts is None or (status is not None and np.sum(status == 1) < 20):
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        else:
            # Update previous points
            if next_pts is not None and status is not None:
                good_indices = status.reshape(-1) == 1
                self.prev_pts = next_pts[good_indices].reshape(-1, 1, 2)
        
        self.prev_gray = gray
        return self.x, self.y, self.heading

class LineBasedMapper:
    """Enhanced navigation with D* Lite pathfinding for optimal route planning"""
    
    def __init__(self):
        # Robot tracking in continuous coordinates
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_heading = 0.0
        
        # Path history
        self.path_history = []
        
        # Create the maze grid for D* Lite
        self.maze_grid = self.create_maze_grid()
        
        # Initialize D* Lite pathfinder
        self.pathfinder = DStarLite(self.maze_grid, cell_size=0.12)
        
        # Navigation state
        self.start_position = None
        self.goal_position = None
        self.optimal_path = []
        self.current_waypoint_index = 0
        
        # Fallback to line segments if needed
        self.line_segments = self.create_actual_line_paths()
        
        self.current_segment = 0
        self.waypoints = []
        self.current_waypoint = 0
        
        # Navigation mode
        self.use_dstar_navigation = False
    
    def create_maze_grid(self):
        """Create the maze grid for D* Lite pathfinding"""
        # Your actual maze layout (horizontally mirrored)
        original_maze = [
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
        
        # Horizontally mirror the maze
        maze_grid = []
        for row in original_maze:
            mirrored_row = list(reversed(row))
            maze_grid.append(mirrored_row)
        
        return maze_grid
    
    def set_start_goal_positions(self, start_world: Tuple[float, float], goal_world: Tuple[float, float]):
        """Set start and goal positions for D* Lite pathfinding"""
        self.start_position = start_world
        self.goal_position = goal_world
        
        # Initialize D* Lite with start and goal
        self.pathfinder.set_start_goal(start_world, goal_world)
        
        # Get the optimal path
        self.optimal_path = self.pathfinder.get_path()
        self.current_waypoint_index = 0
        
        # Enable D* navigation
        self.use_dstar_navigation = True
        
        # D* Navigation path set
        
        return self.optimal_path
    
    def get_available_positions(self):
        """Get list of available start/goal positions in the maze"""
        positions = []
        cell_size = 0.12
        
        for y in range(len(self.maze_grid)):
            for x in range(len(self.maze_grid[0])):
                if self.maze_grid[y][x] == 0:  # Navigable cell
                    world_x = (x + 0.5) * cell_size
                    world_y = (y + 0.5) * cell_size
                    positions.append((world_x, world_y, f"({x},{y})"))
        
        return positions
    
    def get_corner_positions(self):
        """Get strategic positions throughout the maze for easy start/goal selection"""
        positions = [
            # Top row positions
            (0.06, 0.30, "Top-Left Corner"),      # Row 2, Col 0
            (0.90, 0.30, "Top-Left-Center"),      # Row 2, Col 7
            (1.26, 0.30, "Top-Center"),           # Row 2, Col 10
            (1.62, 0.30, "Top-Right-Center"),     # Row 2, Col 13
            (2.46, 0.30, "Top-Right Corner"),     # Row 2, Col 20
            
            # Middle-upper row positions
            (0.06, 0.66, "Upper-Left"),           # Row 5, Col 0
            (1.26, 0.66, "Upper-Center"),         # Row 5, Col 10
            (2.46, 0.66, "Upper-Right"),          # Row 5, Col 20
            
            # Center row positions
            (0.06, 0.90, "Center-Left"),          # Row 7, Col 0
            (0.54, 0.90, "Center-Left-Mid"),      # Row 7, Col 4
            (1.26, 0.90, "Center"),               # Row 7, Col 10
            (1.98, 0.90, "Center-Right-Mid"),     # Row 7, Col 16
            (2.46, 0.90, "Center-Right"),         # Row 7, Col 20
            
            # Middle-lower row positions
            (0.06, 1.14, "Lower-Left"),           # Row 9, Col 0
            (1.26, 1.14, "Lower-Center"),         # Row 9, Col 10
            (2.46, 1.14, "Lower-Right"),          # Row 9, Col 20
            
            # Bottom row positions
            (0.06, 1.50, "Bottom-Left"),          # Row 12, Col 0
            (0.90, 1.50, "Bottom-Left-Center"),   # Row 12, Col 7
            (1.26, 1.50, "Bottom-Center"),        # Row 12, Col 10
            (1.62, 1.50, "Bottom-Right-Center"),  # Row 12, Col 13
            (2.46, 1.50, "Bottom-Right"),         # Row 12, Col 20
            
            # Special positions
            (0.18, 1.68, "Exit-Left"),            # Row 14, Col 1
            (0.54, 1.68, "Exit-Left-Mid"),        # Row 14, Col 4
            (1.26, 1.68, "Exit-Center"),          # Row 14, Col 10
            (2.46, 1.68, "Exit-Right"),           # Row 14, Col 20
        ]
        return positions
    
    def update_robot_position_dstar(self, robot_pos: Tuple[float, float]):
        """Update robot position for D* Lite pathfinding"""
        if self.use_dstar_navigation:
            # Update pathfinder with new robot position
            self.pathfinder.update_robot_position(robot_pos)
            
            # Check if path changed and update our local copy
            if self.pathfinder.path_changed:
                self.optimal_path = self.pathfinder.get_path()
                self.pathfinder.path_changed = False
    
    def get_current_dstar_waypoint(self):
        """Get current waypoint from D* Lite path"""
        if not self.use_dstar_navigation or not self.optimal_path:
            return None
        
        current_pos = (self.robot_x, self.robot_y)
        
        # Use D* Lite's lookahead waypoint selection
        next_waypoint = self.pathfinder.get_next_waypoint(current_pos, lookahead_distance=0.15)
        
        return next_waypoint
    
    def detect_new_obstacles(self, detected_objects_positions: List[Tuple[float, float]]):
        """Update D* Lite with newly detected obstacles"""
        if not self.use_dstar_navigation:
            return
        
        changed_cells = []
        cell_size = 0.12
        
        for obj_x, obj_y in detected_objects_positions:
            # Convert world coordinates to grid coordinates
            grid_x = int(obj_x / cell_size)
            grid_y = int(obj_y / cell_size)
            
            # Mark as obstacle (value = 1)
            if (0 <= grid_x < len(self.maze_grid[0]) and 
                0 <= grid_y < len(self.maze_grid) and
                self.maze_grid[grid_y][grid_x] == 0):  # Was navigable
                
                changed_cells.append((grid_x, grid_y, 1))  # Mark as obstacle
        
        if changed_cells:
            self.pathfinder.update_obstacles(changed_cells)
            self.optimal_path = self.pathfinder.get_path()
    
    def disable_dstar_navigation(self):
        """Disable D* navigation and fall back to line following"""
        self.use_dstar_navigation = False
        self.generate_current_path_waypoints()  # Generate line-following waypoints
    
    def enable_dstar_navigation(self):
        """Re-enable D* navigation if start/goal are set"""
        if self.start_position and self.goal_position:
            self.use_dstar_navigation = True
    
    def create_actual_line_paths(self):
        """Define the actual line paths based on your exact maze layout"""
        
        # Your actual maze layout (horizontally mirrored as you requested)
        original_maze = [
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
        
        # Horizontally mirror the maze
        actual_maze = []
        for row in original_maze:
            mirrored_row = list(reversed(row))
            actual_maze.append(mirrored_row)
        
        # Now extract actual line segments from your maze
        segments = []
        cell_size = 0.12  # 12cm per cell
        
        # Find horizontal line segments (consecutive 0s in rows)
        for y in range(len(actual_maze)):
            x = 0
            while x < len(actual_maze[y]):
                if actual_maze[y][x] == 0:  # Start of line segment
                    start_x = x
                    # Find end of consecutive 0s
                    while x < len(actual_maze[y]) and actual_maze[y][x] == 0:
                        x += 1
                    end_x = x - 1
                    
                    # Only create segment if it's at least 3 cells long
                    if end_x - start_x >= 2:
                        world_start_x = start_x * cell_size
                        world_end_x = end_x * cell_size
                        world_y = y * cell_size
                        
                        segment_name = f"horizontal_row_{y}"
                        segments.append((world_start_x, world_y, world_end_x, world_y, segment_name))
                else:
                    x += 1
        
        # Find vertical line segments (consecutive 0s in columns)
        for x in range(len(actual_maze[0])):
            y = 0
            while y < len(actual_maze):
                if actual_maze[y][x] == 0:  # Start of line segment
                    start_y = y
                    # Find end of consecutive 0s
                    while y < len(actual_maze) and actual_maze[y][x] == 0:
                        y += 1
                    end_y = y - 1
                    
                    # Only create segment if it's at least 3 cells long
                    if end_y - start_y >= 2:
                        world_start_y = start_y * cell_size
                        world_end_y = end_y * cell_size
                        world_x = x * cell_size
                        
                        segment_name = f"vertical_col_{x}"
                        segments.append((world_x, world_start_y, world_x, world_end_y, segment_name))
                else:
                    y += 1
        
        # Line segments extracted
        
        return segments
    
    def generate_current_path_waypoints(self):
        """Generate waypoints along the current line segment"""
        if self.current_segment >= len(self.line_segments):
            self.current_segment = 0
        
        if len(self.line_segments) == 0:
            return
        
        segment = self.line_segments[self.current_segment]
        start_x, start_y, end_x, end_y, name = segment
        
        # Create waypoints along this segment
        waypoints = []
        
        # Calculate segment length
        length = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        
        # Create waypoints every 10cm along the segment
        if length > 0.05:  # Only if segment is longer than 5cm
            num_points = max(2, int(length / 0.10))  # One waypoint every 10cm
            
            for i in range(num_points + 1):
                t = i / num_points  # 0.0 to 1.0
                x = start_x + t * (end_x - start_x)
                y = start_y + t * (end_y - start_y)
                waypoints.append((x, y))
        else:
            # Very short segment, just start and end
            waypoints = [(start_x, start_y), (end_x, end_y)]
        
        self.waypoints = waypoints
        self.current_waypoint = 0
    
    def update_robot_position_from_sensors(self, sensors):
        """Update waypoint progression based on current robot position"""
        # Check if we're close to current waypoint
        if self.current_waypoint < len(self.waypoints):
            target_x, target_y = self.waypoints[self.current_waypoint]
            
            # Calculate distance to target waypoint
            dx = target_x - self.robot_x
            dy = target_y - self.robot_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Move to next waypoint if close (robot position is updated from visual odometry)
            if distance < 0.12:  # Within 12cm - larger tolerance for camera-based positioning
                self.current_waypoint += 1
                
                # If finished current segment, move to next
                if self.current_waypoint >= len(self.waypoints):
                    self.current_segment += 1
                    if self.current_segment >= len(self.line_segments):
                        self.current_segment = 0  # Loop back
                    self.generate_current_path_waypoints()
                    
                    # Starting new line segment
        
        # Update path history with current robot position
        self.path_history.append((self.robot_x, self.robot_y))
        if len(self.path_history) > 60:
            self.path_history = self.path_history[-30:]
    
    def get_current_waypoint(self):
        """Get current waypoint"""
        if self.current_waypoint < len(self.waypoints):
            return self.waypoints[self.current_waypoint]
        return None
    
    def get_grid_visualization(self):
        """Create enhanced visualization showing robot tracking and navigation"""
        # Your actual maze layout (horizontally mirrored)
        original_maze = [
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
        
        # Horizontally mirror the maze
        actual_maze = []
        for row in original_maze:
            mirrored_row = list(reversed(row))
            actual_maze.append(mirrored_row)
        
        # Create larger canvas for better visibility
        cell_size = 35  # Increased from 25 to 35 for better visibility
        maze_height = len(actual_maze)
        maze_width = len(actual_maze[0])
        canvas_width = maze_width * cell_size
        canvas_height = maze_height * cell_size
        
        # Create dark background for cyberpunk look
        vis_grid = np.full((canvas_height, canvas_width, 3), 20, dtype=np.uint8)  # Very dark background
        
        # Draw grid lines for better structure visibility
        grid_color = (40, 40, 40)  # Dark gray grid
        for x in range(0, canvas_width, cell_size):
            cv2.line(vis_grid, (x, 0), (x, canvas_height), grid_color, 1)
        for y in range(0, canvas_height, cell_size):
            cv2.line(vis_grid, (0, y), (canvas_width, y), grid_color, 1)
        
        # Draw line paths with better visibility
        line_color = (80, 80, 80)  # Gray for inactive paths
        
        # First, draw all horizontal line segments
        for y in range(maze_height):
            x = 0
            while x < maze_width:
                if actual_maze[y][x] == 0:  # Start of line segment
                    start_x = x
                    # Find end of consecutive 0s
                    while x < maze_width and actual_maze[y][x] == 0:
                        x += 1
                    end_x = x - 1
                    
                    # Draw this horizontal line segment with thicker lines
                    if end_x > start_x:
                        pixel_y = int(y * cell_size + cell_size/2)
                        pixel_start_x = int(start_x * cell_size + cell_size/2)
                        pixel_end_x = int(end_x * cell_size + cell_size/2)
                        
                        cv2.line(vis_grid, (pixel_start_x, pixel_y), (pixel_end_x, pixel_y), line_color, 6)
                else:
                    x += 1
        
        # Then, draw all vertical line segments
        for x in range(maze_width):
            y = 0
            while y < maze_height:
                if actual_maze[y][x] == 0:  # Start of line segment
                    start_y = y
                    # Find end of consecutive 0s
                    while y < maze_height and actual_maze[y][x] == 0:
                        y += 1
                    end_y = y - 1
                    
                    # Draw this vertical line segment with thicker lines
                    if end_y > start_y:
                        pixel_x = int(x * cell_size + cell_size/2)
                        pixel_start_y = int(start_y * cell_size + cell_size/2)
                        pixel_end_y = int(end_y * cell_size + cell_size/2)
                        
                        cv2.line(vis_grid, (pixel_x, pixel_start_y), (pixel_x, pixel_end_y), line_color, 6)
                else:
                    y += 1
        
        # Highlight current line segment with bright cyberpunk color
        if self.current_segment < len(self.line_segments):
            segment = self.line_segments[self.current_segment]
            start_x, start_y, end_x, end_y, name = segment
            
            # Convert real coordinates back to pixel coordinates
            px1 = int(start_x / 0.12 * cell_size + cell_size/2)
            py1 = int(start_y / 0.12 * cell_size + cell_size/2)
            px2 = int(end_x / 0.12 * cell_size + cell_size/2)
            py2 = int(end_y / 0.12 * cell_size + cell_size/2)
            
            # Draw current path with bright cyan/orange gradient effect
            cv2.line(vis_grid, (px1, py1), (px2, py2), (0, 255, 255), 8)  # Bright cyan
            cv2.line(vis_grid, (px1, py1), (px2, py2), (0, 165, 255), 4)  # Orange center
        
        # Draw path history with fading trail effect
        if len(self.path_history) > 1:
            for i in range(1, len(self.path_history)):
                x1, y1 = self.path_history[i-1]
                x2, y2 = self.path_history[i]
                
                px1 = int(x1 / 0.12 * cell_size + cell_size/2)
                py1 = int(y1 / 0.12 * cell_size + cell_size/2)
                px2 = int(x2 / 0.12 * cell_size + cell_size/2)
                py2 = int(y2 / 0.12 * cell_size + cell_size/2)
                
                if (0 <= px1 < canvas_width and 0 <= py1 < canvas_height and 
                    0 <= px2 < canvas_width and 0 <= py2 < canvas_height):
                    
                    # Fade effect - newer trail segments are brighter
                    fade_factor = i / len(self.path_history)
                    trail_brightness = int(255 * fade_factor)
                    trail_color = (trail_brightness, trail_brightness, 0)  # Yellow trail
                    cv2.line(vis_grid, (px1, py1), (px2, py2), trail_color, 3)
        
        # Draw available start/goal positions as small markers - REMOVED FOR CLICK-TO-SELECT
        # available_positions = self.get_corner_positions()
        # for pos_x, pos_y, name in available_positions:
        #     pos_px = int(pos_x / 0.12 * cell_size + cell_size/2)
        #     pos_py = int(pos_y / 0.12 * cell_size + cell_size/2)
        #     
        #     if 0 <= pos_px < canvas_width and 0 <= pos_py < canvas_height:
        #         # Small cyan circles for available positions
        #         cv2.circle(vis_grid, (pos_px, pos_py), 4, (255, 255, 0), 1)  # Yellow outline
        #         cv2.circle(vis_grid, (pos_px, pos_py), 2, (0, 255, 255), -1)  # Cyan center
        
        # Add prominent corner dots for easy reference - REMOVED FOR CLEAN INTERFACE
        # corner_positions = [
        #     (0, 0, "Top-Left"),           # Top-left corner
        #     (canvas_width-1, 0, "Top-Right"),        # Top-right corner
        #     (0, canvas_height-1, "Bottom-Left"),     # Bottom-left corner
        #     (canvas_width-1, canvas_height-1, "Bottom-Right")  # Bottom-right corner
        # ]
        # 
        # for corner_x, corner_y, corner_name in corner_positions:
        #     # Large corner markers
        #     cv2.circle(vis_grid, (corner_x, corner_y), 8, (0, 255, 255), 2)  # Cyan outline
        #     cv2.circle(vis_grid, (corner_x, corner_y), 5, (255, 255, 255), -1)  # White center
        #     cv2.circle(vis_grid, (corner_x, corner_y), 2, (0, 255, 255), -1)  # Cyan inner dot
        
        # Draw D* Lite optimal path if available
        if self.use_dstar_navigation and self.optimal_path:
            for i in range(1, len(self.optimal_path)):
                x1, y1 = self.optimal_path[i-1]
                x2, y2 = self.optimal_path[i]
                
                px1 = int(x1 / 0.12 * cell_size + cell_size/2)
                py1 = int(y1 / 0.12 * cell_size + cell_size/2)
                px2 = int(x2 / 0.12 * cell_size + cell_size/2)
                py2 = int(y2 / 0.12 * cell_size + cell_size/2)
                
                if (0 <= px1 < canvas_width and 0 <= py1 < canvas_height and 
                    0 <= px2 < canvas_width and 0 <= py2 < canvas_height):
                    
                    # D* path in bright blue with white center
                    cv2.line(vis_grid, (px1, py1), (px2, py2), (255, 100, 0), 6)  # Bright blue
                    cv2.line(vis_grid, (px1, py1), (px2, py2), (255, 255, 255), 2)  # White center
            
            # Draw start and goal positions
            if self.start_position:
                start_x, start_y = self.start_position
                start_px = int(start_x / 0.12 * cell_size + cell_size/2)
                start_py = int(start_y / 0.12 * cell_size + cell_size/2)
                
                if 0 <= start_px < canvas_width and 0 <= start_py < canvas_height:
                    # Start position as bright green square
                    cv2.rectangle(vis_grid, (start_px-12, start_py-12), (start_px+12, start_py+12), (0, 255, 0), -1)
                    cv2.rectangle(vis_grid, (start_px-8, start_py-8), (start_px+8, start_py+8), (255, 255, 255), -1)
                    cv2.putText(vis_grid, "S", (start_px-6, start_py+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            if self.goal_position:
                goal_x, goal_y = self.goal_position
                goal_px = int(goal_x / 0.12 * cell_size + cell_size/2)
                goal_py = int(goal_y / 0.12 * cell_size + cell_size/2)
                
                if 0 <= goal_px < canvas_width and 0 <= goal_py < canvas_height:
                    # Goal position as bright red square
                    cv2.rectangle(vis_grid, (goal_px-12, goal_py-12), (goal_px+12, goal_py+12), (0, 0, 255), -1)
                    cv2.rectangle(vis_grid, (goal_px-8, goal_py-8), (goal_px+8, goal_py+8), (255, 255, 255), -1)
                    cv2.putText(vis_grid, "G", (goal_px-6, goal_py+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw current waypoint with pulsing target effect
        current_waypoint = None
        if self.use_dstar_navigation:
            current_waypoint = self.get_current_dstar_waypoint()
        elif self.current_waypoint < len(self.waypoints):
            current_waypoint = self.waypoints[self.current_waypoint]
        
        if current_waypoint:
            wx, wy = current_waypoint
            waypoint_px = int(wx / 0.12 * cell_size + cell_size/2)
            waypoint_py = int(wy / 0.12 * cell_size + cell_size/2)
            
            if 0 <= waypoint_px < canvas_width and 0 <= waypoint_py < canvas_height:
                # Target crosshair with multiple rings
                target_color = (0, 255, 0)  # Bright green
                cv2.circle(vis_grid, (waypoint_px, waypoint_py), 15, target_color, 3)
                cv2.circle(vis_grid, (waypoint_px, waypoint_py), 10, target_color, 2)
                cv2.circle(vis_grid, (waypoint_px, waypoint_py), 5, target_color, -1)
                
                # Crosshair lines
                cv2.line(vis_grid, (waypoint_px-20, waypoint_py), (waypoint_px+20, waypoint_py), target_color, 2)
                cv2.line(vis_grid, (waypoint_px, waypoint_py-20), (waypoint_px, waypoint_py+20), target_color, 2)
        
        # Draw robot position with prominent indicator
        robot_px = int(self.robot_x / 0.12 * cell_size + cell_size/2)
        robot_py = int(self.robot_y / 0.12 * cell_size + cell_size/2)
        
        if 0 <= robot_px < canvas_width and 0 <= robot_py < canvas_height:
            # Multi-layer robot indicator for high visibility
            robot_color = (255, 0, 255)  # Bright magenta/pink
            
            # Outer glow ring
            cv2.circle(vis_grid, (robot_px, robot_py), 20, (50, 0, 50), 3)
            # Main robot body
            cv2.circle(vis_grid, (robot_px, robot_py), 12, robot_color, -1)
            # Inner highlight
            cv2.circle(vis_grid, (robot_px, robot_py), 8, (255, 100, 255), -1)
            # Center dot
            cv2.circle(vis_grid, (robot_px, robot_py), 3, (255, 255, 255), -1)
            
            # Direction indicator (heading arrow)
            arrow_length = 25
            arrow_end_x = int(robot_px + arrow_length * math.cos(self.robot_heading))
            arrow_end_y = int(robot_py + arrow_length * math.sin(self.robot_heading))
            cv2.arrowedLine(vis_grid, (robot_px, robot_py), (arrow_end_x, arrow_end_y), 
                           (255, 255, 0), 3, tipLength=0.3)  # Yellow arrow
        
        # Add legend in bottom right - REMOVED FOR CLEANER VIEW
        # legend_x = canvas_width - 200
        # legend_y = canvas_height - 160  # Increased height for corner dots
        # cv2.rectangle(vis_grid, (legend_x, legend_y), (canvas_width-10, canvas_height-10), (0, 0, 0), -1)
        # cv2.rectangle(vis_grid, (legend_x, legend_y), (canvas_width-10, canvas_height-10), (0, 255, 255), 1)
        
        # Font and color for legend only
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # text_color = (0, 255, 255)  # Cyan text
        
        # Legend items
        # cv2.circle(vis_grid, (legend_x + 15, legend_y + 15), 5, (255, 0, 255), -1)
        # cv2.putText(vis_grid, "Robot", (legend_x + 25, legend_y + 20), font, 0.4, text_color, 1)
        
        # cv2.circle(vis_grid, (legend_x + 15, legend_y + 35), 5, (0, 255, 0), -1)
        # cv2.putText(vis_grid, "Target", (legend_x + 25, legend_y + 40), font, 0.4, text_color, 1)
        
        # cv2.line(vis_grid, (legend_x + 10, legend_y + 55), (legend_x + 20, legend_y + 55), (255, 255, 0), 2)
        # cv2.putText(vis_grid, "Trail", (legend_x + 25, legend_y + 60), font, 0.4, text_color, 1)
        
        # cv2.line(vis_grid, (legend_x + 10, legend_y + 75), (legend_x + 20, legend_y + 75), (0, 255, 255), 3)
        # cv2.putText(vis_grid, "Current Path", (legend_x + 25, legend_y + 80), font, 0.4, text_color, 1)
        
        # Available positions marker
        # cv2.circle(vis_grid, (legend_x + 15, legend_y + 95), 4, (255, 255, 0), 1)
        # cv2.circle(vis_grid, (legend_x + 15, legend_y + 95), 2, (0, 255, 255), -1)
        # cv2.putText(vis_grid, "Positions", (legend_x + 25, legend_y + 100), font, 0.4, text_color, 1)
        
        # Corner dots marker
        # cv2.circle(vis_grid, (legend_x + 15, legend_y + 115), 5, (255, 255, 255), -1)
        # cv2.circle(vis_grid, (legend_x + 15, legend_y + 115), 2, (0, 255, 255), -1)
        # cv2.putText(vis_grid, "Corners", (legend_x + 25, legend_y + 120), font, 0.4, text_color, 1)
        
        # D* navigation elements
        # if self.use_dstar_navigation:
        #     cv2.line(vis_grid, (legend_x + 10, legend_y + 135), (legend_x + 20, legend_y + 135), (255, 100, 0), 4)
        #     cv2.putText(vis_grid, "D* Path", (legend_x + 25, legend_y + 140), font, 0.4, text_color, 1)
        #     
        #     cv2.rectangle(vis_grid, (legend_x + 10, legend_y + 145), (legend_x + 20, legend_y + 155), (0, 255, 0), -1)
        #     cv2.putText(vis_grid, "Start", (legend_x + 25, legend_y + 155), font, 0.4, text_color, 1)
        
        # Resize for consistent web display
        target_height = 500  # Increased for better visibility
        target_width = int(target_height * canvas_width / canvas_height)
        vis_grid = cv2.resize(vis_grid, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        return vis_grid

class SimpleLineFollower:
    """Pattern-based line following with object detection and visual navigation using discrete sensor states"""
    
    def __init__(self, esp32_ip):
        self.esp32 = ESP32Interface(esp32_ip)
        
        # Simple states
        self.state = "SEARCHING"
        self.last_turn_direction = "right"  # Remember last turn for search
        
        # Speed settings - immediate but controlled corrections
        self.forward_speed = 32  # Slower forward speed for better control
        self.gentle_turn_factor = 0.68  # Immediate but gentle corrections (68% balanced)
        self.sharp_turn_speed = 50
        self.search_speed = 35
        
        # State tracking
        self.line_lost_time = 0
        self.last_correction = "NONE"  # Track last correction to avoid abrupt changes
        
        # Object detection setup
        self.camera = None
        self.yolo_model = None
        self.setup_camera_and_yolo()
        
        # Object detection state
        self.object_detected = False
        self.turning_180 = False
        self.turn_180_start_time = 0
        
        # Visual odometry for camera-based position tracking
        self.visual_odometry = VisualOdometry()
        
        # Line-based mapping for navigation
        self.mapper = LineBasedMapper()
        
        # Navigation state - always enabled since we need both line following AND navigation
        self.navigation_mode = True  # Always use both together
        self.current_position = (0.0, 0.0, 0.0)  # (x, y, heading)

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
            
            logging.info("Camera and YOLO model initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize camera/YOLO: {e}")
            self.camera = None
            self.yolo_model = None

    def detect_objects(self):
        """Detect objects using YOLO and return True if any obstacle is detected"""
        if not self.camera or not self.yolo_model:
            return False
            
        try:
            ret, frame = self.camera.read()
            if not ret:
                return False
                
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
                                return True
                            elif class_name.lower() in ignore_objects:
                                logging.debug(f"Ignoring non-obstacle: {class_name} (confidence: {confidence:.2f})")
                            else:
                                # Unknown object - be cautious and treat as obstacle
                                logging.info(f"Unknown object detected: {class_name} (confidence: {confidence:.2f})")
                                return True
            
            return False
            
        except Exception as e:
            logging.debug(f"Object detection error: {e}")
            return False

    def run(self):
        """Main control loop with visual odometry and navigation"""
        logging.info("Starting pattern-based line follower with visual navigation")
        
        # Try to connect to ESP32 in a separate thread to avoid blocking
        def connect_esp32():
            esp32_connected = self.esp32.connect()
            if esp32_connected:
                print("✅ ESP32 connected successfully!")
            else:
                print("❌ ESP32 connection failed - continuing without motor control")
        
        # Start ESP32 connection in background
        esp32_thread = threading.Thread(target=connect_esp32, daemon=True)
        esp32_thread.start()
        
        try:
            while True:
                # Get camera frame for both object detection and visual odometry
                frame = None
                if self.camera:
                    ret, frame = self.camera.read()
                    if ret:
                        # Update position tracking
                        x, y, heading = self.update_position_tracking(frame)
                        
                        # Position tracking active
                
                # Check for objects every few cycles (not every cycle for performance)
                if not hasattr(self, '_detection_counter'):
                    self._detection_counter = 0
                
                self._detection_counter += 1
                if self._detection_counter >= 5 and frame is not None:  # Check every 5 cycles
                    self.object_detected = self.detect_objects_from_frame(frame)
                    self._detection_counter = 0
                
                self.control_loop()
                time.sleep(0.05)  # 20Hz - stable control
                
        except KeyboardInterrupt:
            logging.info("Stopping...")
        finally:
            self.stop()
    
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
                                return True
                            elif class_name.lower() in ignore_objects:
                                logging.debug(f"Ignoring non-obstacle: {class_name} (confidence: {confidence:.2f})")
                            else:
                                # Unknown object - be cautious and treat as obstacle
                                logging.info(f"Unknown object detected: {class_name} (confidence: {confidence:.2f})")
                                return True
            
            return False

        except Exception as e:
            logging.debug(f"Object detection error: {e}")
            return False

    def control_loop(self):
        """Single control loop using sensor patterns with object detection and navigation"""
        
        # Priority 1: Handle 180-degree turn if object detected
        if self.object_detected and not self.turning_180:
            logging.info("Object detected! Starting 180-degree turn")
            self.turning_180 = True
            self.turn_180_start_time = time.time()
            self.state = "TURN_180"
            self.execute_state()
            return
        
        # Priority 2: Continue 180-degree turn if in progress
        if self.turning_180:
            elapsed_time = time.time() - self.turn_180_start_time
            if elapsed_time < 3.0:  # Turn for 3 seconds (adjust as needed)
                self.state = "TURN_180"
                self.execute_state()
                return
            else:
                # Finished 180 turn, reset and resume line following
                logging.info("180-degree turn completed, resuming line following")
                self.turning_180 = False
                self.object_detected = False
                self.state = "SEARCH"  # Start searching for line again
        
        # Priority 3: Navigation mode (if enabled)
        if self.navigation_mode:
            nav_command = self.get_navigation_command()
            if nav_command:
                # Override line following with navigation command
                if nav_command == "FORWARD":
                    self.state = "FORWARD"
                elif nav_command == "TURN_LEFT":
                    self.state = "TURN_LEFT_GENTLE"
                elif nav_command == "TURN_RIGHT":
                    self.state = "TURN_RIGHT_GENTLE"
                
                self.execute_state()
            return
        
        # Priority 4: Normal line following
        # Get sensor pattern [L2, L1, C, R1, R2]
        sensors = self.esp32.sensors
        L2, L1, C, R1, R2 = sensors
        
        # Determine state based on 5cm tape + 7cm sensor array
        # IDEAL: Middle 3 sensors (01110) should be on tape when centered
        
        if not L2 and L1 and C and R1 and not R2:
            # Pattern: 01110 - Perfect center (3 middle sensors on 5cm tape)
            self.state = "FORWARD"
            
        elif not L2 and not L1 and C and not R1 and not R2:
            # Pattern: 00100 - Only center sensor - PERFECT, go forward
            self.state = "FORWARD"
            
        elif not L2 and not L1 and C and R1 and not R2:
            # Pattern: 00110 - Drifting right, correct LEFT immediately
            self.state = "TURN_LEFT_GENTLE"
            self.last_turn_direction = "left"
            
        elif not L2 and L1 and C and not R1 and not R2:
            # Pattern: 01100 - Drifting left, correct RIGHT immediately
            self.state = "TURN_RIGHT_GENTLE"
            self.last_turn_direction = "right"
            
        elif not L2 and L1 and C and R1 and R2:
            # Pattern: 01111 - Drifting right (right edge sensor active)
            self.state = "TURN_LEFT_GENTLE"
            self.last_turn_direction = "left"
            
        elif L2 and L1 and C and R1 and not R2:
            # Pattern: 11110 - Drifting left (left edge sensor active)  
            self.state = "TURN_RIGHT_GENTLE"
            self.last_turn_direction = "right"
            
        elif not L2 and not L1 and not C and R1 and R2:
            # Pattern: 00011 - Robot overshot left, line on right side, turn RIGHT to center
            self.state = "TURN_RIGHT_GENTLE"
            self.last_turn_direction = "right"
            
        elif L2 and L1 and not C and not R1 and not R2:
            # Pattern: 11000 - Robot overshot right, line on left side, turn LEFT to center
            self.state = "TURN_LEFT_GENTLE"
            self.last_turn_direction = "left"
            
        elif not L2 and L1 and not C and not R1 and not R2:
            # Pattern: 01000 - Left sensor only, gentle right turn
            self.state = "TURN_RIGHT_GENTLE"
            self.last_turn_direction = "right"
            
        elif L2 and not L1 and not C and not R1 and not R2:
            # Pattern: 10000 - Far left sensor only, gentle right turn
            self.state = "TURN_RIGHT_GENTLE"
            self.last_turn_direction = "right"
            
        elif not L2 and not L1 and not C and R1 and not R2:
            # Pattern: 00010 - Right sensor only, gentle left turn
            self.state = "TURN_LEFT_GENTLE"
            self.last_turn_direction = "left"
            
        elif not L2 and not L1 and not C and not R1 and R2:
            # Pattern: 00001 - Far right sensor only, gentle left turn
            self.state = "TURN_LEFT_GENTLE"
            self.last_turn_direction = "left"
            
        elif L1 and C and R1:
            # Pattern: X111X - Wide line (intersection or corner approach)
            self.state = "FORWARD"  # Go straight through
            
        elif (L2 and L1 and C) or (C and R1 and R2):
            # Patterns like 111XX or XX111 - Corner detected, use gentle turns
            if L2 and L1 and C:
                self.state = "TURN_LEFT_GENTLE"
                self.last_turn_direction = "left"
            else:
                self.state = "TURN_RIGHT_GENTLE"
                self.last_turn_direction = "right"
                
        elif sum(sensors) >= 4:
            # Pattern: 4+ sensors active - very wide line or intersection
            self.state = "FORWARD"
            
        elif sum(sensors) == 0:
            # Pattern: 00000 - No line detected
            if self.state != "SEARCH":
                self.line_lost_time = time.time()
            self.state = "SEARCH"
            
        else:
            # Any other pattern - continue with last known direction or search
            if hasattr(self, 'line_lost_time') and time.time() - self.line_lost_time > 1.0:
                self.state = "SEARCH"
            # Otherwise keep current state
        
        # Execute the determined state
        self.execute_state()
        
        # Control loop active
    
    def execute_state(self):
        """Execute motor commands based on current state"""
        left_speed = 0
        right_speed = 0
        
        if self.state == "FORWARD":
            left_speed = self.forward_speed
            right_speed = self.forward_speed
            
        elif self.state == "TURN_LEFT_GENTLE":
            # Gentle left: slow down left wheel
            left_speed = int(self.forward_speed * self.gentle_turn_factor)
            right_speed = self.forward_speed
            
        elif self.state == "TURN_LEFT_SHARP":
            # Sharp left: stop left wheel, turn right wheel
            left_speed = 0
            right_speed = self.sharp_turn_speed
            
        elif self.state == "TURN_RIGHT_GENTLE":
            # Gentle right: slow down right wheel
            left_speed = self.forward_speed
            right_speed = int(self.forward_speed * self.gentle_turn_factor)
            
        elif self.state == "TURN_RIGHT_SHARP":
            # Sharp right: turn left wheel, stop right wheel
            left_speed = self.sharp_turn_speed
            right_speed = 0
            
        elif self.state == "TURN_180":
            # 180-degree turn: spin in place
            left_speed = -self.sharp_turn_speed  # Turn left (reverse left wheel)
            right_speed = self.sharp_turn_speed   # Turn left (forward right wheel)
            
        elif self.state == "SEARCH":
            # Search based on last known direction
            if self.last_turn_direction == "left":
                left_speed = -self.search_speed  # Spin left
                right_speed = self.search_speed
            else:
                left_speed = self.search_speed   # Spin right
                right_speed = -self.search_speed
        
        # Send commands to ESP32
        self.esp32.send_motor_speeds(left_speed, right_speed)
    
    def stop(self):
        """Stop the robot and cleanup resources"""
        self.esp32.send_motor_speeds(0, 0)
        self.esp32.close()
        
        # Cleanup camera
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        logging.info("Camera resources cleaned up")

    def enable_navigation_mode(self):
        """Enable navigation mode to follow waypoints"""
        self.navigation_mode = True
        self.mapper.current_waypoint = 0
        logging.info("Navigation mode enabled - robot will follow waypoints")
    
    def disable_navigation_mode(self):
        """Disable navigation mode - return to line following"""
        self.navigation_mode = False
        logging.info("Navigation mode disabled - returning to line following")
    
    def update_position_tracking(self, frame):
        """Update robot position using camera visual odometry"""
        # Get position from visual odometry (camera-based tracking)
        vo_x, vo_y, vo_heading = self.visual_odometry.update(frame)
        
        # Update mapper with visual odometry position
        self.mapper.robot_x = vo_x
        self.mapper.robot_y = vo_y
        self.mapper.robot_heading = vo_heading
        
        # Update D* Lite pathfinder with new position
        self.mapper.update_robot_position_dstar((vo_x, vo_y))
        
        # Also update mapper with sensor data for waypoint progression (fallback)
        self.mapper.update_robot_position_from_sensors(self.esp32.sensors)
        
        # Use visual odometry position as the authoritative position
        self.current_position = (vo_x, vo_y, vo_heading)
        return vo_x, vo_y, vo_heading
    
    def get_navigation_command(self):
        """Get navigation command - prioritize D* Lite over line following"""
        if not self.navigation_mode:
            return None
        
        x, y, heading = self.current_position
        
        # Priority 1: Use D* Lite pathfinding if enabled
        if self.mapper.use_dstar_navigation:
            waypoint = self.mapper.get_current_dstar_waypoint()
            if waypoint is None:
                logging.info("D* Navigation: Goal reached! Disabling D* navigation")
                self.mapper.disable_dstar_navigation()
                return None
            
            target_x, target_y = waypoint
            
            # Calculate distance to target
            distance = math.sqrt((target_x - x)**2 + (target_y - y)**2)
            
            # Get direction to navigate
            direction = self.get_navigation_direction(x, y, heading, target_x, target_y)
            
            # D* navigation active
            
            return direction
        
        # Priority 2: Fallback to line following navigation
        waypoint = self.mapper.get_current_waypoint()
        if waypoint is None:
            logging.info("All waypoints reached! Restarting waypoint sequence")
            self.mapper.current_waypoint = 0
            waypoint = self.mapper.get_current_waypoint()
            if waypoint is None:
                return None
        
        target_x, target_y = waypoint
        
        # Calculate distance to target
        distance = math.sqrt((target_x - x)**2 + (target_y - y)**2)
        
        # Get direction to navigate
        direction = self.get_navigation_direction(x, y, heading, target_x, target_y)
        
        # For maze navigation, prioritize navigation over line following at longer distances
        if distance < 0.08:  # Within 8cm, use line following for final precision
            return None  # Fall back to line following
        
        # Line navigation active
        
        return direction
    
    def set_navigation_goal(self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float]):
        """Set start and goal positions for D* Lite navigation"""
        try:
            path = self.mapper.set_start_goal_positions(start_pos, goal_pos)
            logging.info(f"D* Navigation: Path set with {len(path)} waypoints")
            return path
        except Exception as e:
            logging.error(f"Failed to set navigation goal: {e}")
            return None
    
    def get_available_navigation_positions(self):
        """Get available positions for start/goal selection"""
        return self.mapper.get_corner_positions()
    
    def enable_dstar_navigation(self):
        """Enable D* Lite navigation mode"""
        self.mapper.enable_dstar_navigation()
        self.navigation_mode = True
        logging.info("D* Lite navigation enabled")
    
    def disable_dstar_navigation(self):
        """Disable D* Lite navigation and use line following"""
        self.mapper.disable_dstar_navigation()
        logging.info("D* Lite navigation disabled, using line following")

    def get_navigation_direction(self, x, y, heading, target_x, target_y):
        """Get navigation direction based on current position and target"""
        # Calculate relative position
        dx = target_x - x
        dy = target_y - y
        
        # Calculate angle to target
        angle_to_target = math.atan2(dy, dx)
        
        # Calculate angle difference
        angle_diff = angle_to_target - heading
        
        # Normalize angle difference to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        return angle_diff

class WebVisualization:
    """Flask web app for visualizing robot position and map with cyberpunk theme"""
    
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
            return render_template('index.html')
        
        @self.app.route('/navigation')
        def navigation_control():
            """D* navigation control interface"""
            return render_template('navigation.html')
        
        @self.app.route('/test')
        def test_map():
            """Test page for debugging map display"""
            with open('test_map.html', 'r') as f:
                return f.read()
        
        @self.app.route('/api/robot_data')
        def get_robot_data():
            """Get current robot data as JSON"""
            x, y, heading = self.robot.current_position
            
            # Get maze visualization
            grid_image = self.create_grid_visualization()
            
            # Get D* navigation info
            dstar_info = {
                'enabled': self.robot.mapper.use_dstar_navigation,
                'start_position': self.robot.mapper.start_position,
                'goal_position': self.robot.mapper.goal_position,
                'path_length': len(self.robot.mapper.optimal_path) if self.robot.mapper.optimal_path else 0
            }
            
            data = {
                'position': {
                    'x': float(x),
                    'y': float(y),
                    'heading': float(heading)
                },
                'state': self.robot.state,
                'sensors': self.robot.esp32.sensors,
                'current_waypoint': self.robot.mapper.current_waypoint,
                'total_waypoints': len(self.robot.mapper.waypoints),
                'grid_image': grid_image,
                'dstar_navigation': dstar_info
            }
            
            return jsonify(data)
        
        @self.app.route('/api/navigation/positions')
        def get_navigation_positions():
            """Get available start/goal positions"""
            positions = self.robot.get_available_navigation_positions()
            return jsonify({
                'positions': [
                    {'x': pos[0], 'y': pos[1], 'name': pos[2]} 
                    for pos in positions
                ]
            })
        
        @self.app.route('/api/navigation/set_goal', methods=['POST'])
        def set_navigation_goal():
            """Set start and goal positions for D* navigation"""
            try:
                data = request.get_json()
                start_x = float(data['start_x'])
                start_y = float(data['start_y'])
                goal_x = float(data['goal_x'])
                goal_y = float(data['goal_y'])
                
                path = self.robot.set_navigation_goal((start_x, start_y), (goal_x, goal_y))
                
                if path:
                    return jsonify({
                        'success': True,
                        'message': f'D* path set with {len(path)} waypoints',
                        'path_length': len(path),
                        'path': [{'x': p[0], 'y': p[1]} for p in path[:10]]  # First 10 waypoints
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': 'Failed to compute path'
                    })
                    
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': f'Error: {str(e)}'
                })
        
        @self.app.route('/api/navigation/enable', methods=['POST'])
        def enable_navigation():
            """Enable D* navigation"""
            try:
                self.robot.enable_dstar_navigation()
                return jsonify({
                    'success': True,
                    'message': 'D* navigation enabled'
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': f'Error: {str(e)}'
                })
        
        @self.app.route('/api/navigation/disable', methods=['POST'])
        def disable_navigation():
            """Disable D* navigation"""
            try:
                self.robot.disable_dstar_navigation()
                return jsonify({
                    'success': True,
                    'message': 'D* navigation disabled'
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': f'Error: {str(e)}'
                })
        
        @self.app.route('/video_feed')
        def video_feed():
            """Video streaming route"""
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/debug/map')
        def debug_map():
            """Debug route to test map generation"""
            try:
                grid_image = self.create_grid_visualization()
                return jsonify({
                    'status': 'success',
                    'image_length': len(grid_image),
                    'has_image': bool(grid_image and len(grid_image) > 100)
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'error': str(e)
                })
    
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
    
    def create_grid_visualization(self):
        """Create visualization of the predefined maze"""
        try:
            # Get the maze visualization from mapper
            rgb_grid = self.robot.mapper.get_grid_visualization()
            
            # Resize for better visibility (make it bigger)
            rgb_grid = cv2.resize(rgb_grid, (500, 400), interpolation=cv2.INTER_NEAREST)
            
            # Convert to base64
            _, buffer = cv2.imencode('.png', rgb_grid)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return img_base64
            
        except Exception as e:
            # Return a simple test image as fallback
            test_img = np.full((400, 500, 3), 64, dtype=np.uint8)  # Dark background
            cv2.putText(test_img, 'MAP ERROR', (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.rectangle(test_img, (50, 50), (450, 350), (0, 255, 255), 2)
            
            _, buffer = cv2.imencode('.png', test_img)
            fallback_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return fallback_base64
    
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
    robot = SimpleLineFollower("192.168.2.21")
    
    # Create and start web visualization automatically
    try:
        web_viz = WebVisualization(robot)
        print("✅ Cyberpunk web visualization initialized")
        
        # Start web server automatically in background
        web_thread = threading.Thread(target=web_viz.start_web_server, daemon=True)
        web_thread.start()
        print("✅ Cyberpunk dashboard started at http://0.0.0.0:5000")
    except Exception as e:
        print(f"❌ Could not initialize web visualization: {e}")
        web_viz = None
    
    # Setup D* navigation with predefined start/goal positions
    print("\n" + "="*60)
    setup_navigation(robot)
    print("="*60)
    
    print("╔════════════════════════════════════════════════════════════╗")
    print("║            CYBERPUNK ROBOT NAVIGATION MATRIX              ║")
    print("╠════════════════════════════════════════════════════════════╣")
    print("║ > Neural line-following with visual odometry tracking     ║")
    print("║ > Real-time maze navigation through extracted coordinates ║")
    print("║ > YOLO11n obstacle detection with evasive maneuvers      ║")
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