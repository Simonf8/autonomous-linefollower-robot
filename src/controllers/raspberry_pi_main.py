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
from flask import Flask, render_template, jsonify, Response

# Web visualization is now integrated in this file
WebVisualization = None  # Will be defined below

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
            self.socket.settimeout(2.0)
            self.socket.connect((self.ip_address, self.port))
            self.socket.settimeout(0.1)
            self.connected = True
            logging.info(f"Connected to ESP32 at {self.ip_address}")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to ESP32: {e}")
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
    """Pure line-based navigation - no grid, just actual line paths"""
    
    def __init__(self):
        # Robot tracking in continuous coordinates
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_heading = 0.0
        
        # Path history
        self.path_history = []
        
        # Define your actual line paths directly (no grid conversion)
        # Based on your maze layout, these are the real line segments
        self.line_segments = self.create_actual_line_paths()
        
        # Current navigation state
        self.current_segment = 0
        self.position_on_segment = 0.0  # 0.0 to 1.0 along current segment
        
        # Waypoints along current path
        self.waypoints = []
        self.current_waypoint = 0
        self.generate_current_path_waypoints()
    
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
        
        print(f"Extracted {len(segments)} line segments from your actual maze:")
        for i, (sx, sy, ex, ey, name) in enumerate(segments):
            length = math.sqrt((ex-sx)**2 + (ey-sy)**2)
            print(f"  {i+1}. {name}: ({sx:.2f},{sy:.2f}) -> ({ex:.2f},{ey:.2f}) [{length:.2f}m]")
        
        return segments
    
    def generate_current_path_waypoints(self):
        """Generate waypoints along the current line segment"""
        if self.current_segment >= len(self.line_segments):
            self.current_segment = 0
        
        if len(self.line_segments) == 0:
            print("No line segments found!")
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
        print(f"Following line: {name} ({length:.2f}m, {len(waypoints)} waypoints)")
    
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
                print(f"Reached waypoint {self.current_waypoint-1}, distance was {distance:.3f}m")
                
                # If finished current segment, move to next
                if self.current_waypoint >= len(self.waypoints):
                    self.current_segment += 1
                    if self.current_segment >= len(self.line_segments):
                        self.current_segment = 0  # Loop back
                    self.generate_current_path_waypoints()
                    
                    if self.current_segment < len(self.line_segments):
                        segment_name = self.line_segments[self.current_segment][4]
                        print(f"Starting new line: {segment_name}")
        
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
        
        # Draw current waypoint with pulsing target effect
        if self.current_waypoint < len(self.waypoints):
            wx, wy = self.waypoints[self.current_waypoint]
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
        
        # Add legend in bottom right
        legend_x = canvas_width - 200
        legend_y = canvas_height - 100
        cv2.rectangle(vis_grid, (legend_x, legend_y), (canvas_width-10, canvas_height-10), (0, 0, 0), -1)
        cv2.rectangle(vis_grid, (legend_x, legend_y), (canvas_width-10, canvas_height-10), (0, 255, 255), 1)
        
        # Font and color for legend only
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (0, 255, 255)  # Cyan text
        
        # Legend items
        cv2.circle(vis_grid, (legend_x + 15, legend_y + 20), 5, (255, 0, 255), -1)
        cv2.putText(vis_grid, "Robot", (legend_x + 25, legend_y + 25), font, 0.4, text_color, 1)
        
        cv2.circle(vis_grid, (legend_x + 15, legend_y + 40), 5, (0, 255, 0), -1)
        cv2.putText(vis_grid, "Target", (legend_x + 25, legend_y + 45), font, 0.4, text_color, 1)
        
        cv2.line(vis_grid, (legend_x + 10, legend_y + 60), (legend_x + 20, legend_y + 60), (255, 255, 0), 2)
        cv2.putText(vis_grid, "Trail", (legend_x + 25, legend_y + 65), font, 0.4, text_color, 1)
        
        cv2.line(vis_grid, (legend_x + 10, legend_y + 80), (legend_x + 20, legend_y + 80), (0, 255, 255), 3)
        cv2.putText(vis_grid, "Current Path", (legend_x + 25, legend_y + 85), font, 0.4, text_color, 1)
        
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
        
        if not self.esp32.connect():
            logging.error("Failed to connect to ESP32")
            return
                
        try:
            while True:
                # Get camera frame for both object detection and visual odometry
                frame = None
                if self.camera:
                    ret, frame = self.camera.read()
                    if ret:
                        # Update position tracking
                        x, y, heading = self.update_position_tracking(frame)
                        
                        # Debug position info
                        if not hasattr(self, '_last_position_debug'):
                            self._last_position_debug = 0
                        if time.time() - self._last_position_debug > 2.0:
                            print(f"Position: ({x:.2f}, {y:.2f}), Heading: {heading:.2f} rad")
                            self._last_position_debug = time.time()
                
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
        
        # Debug output
        if not hasattr(self, '_last_debug'):
            self._last_debug = 0
        if time.time() - self._last_debug > 1.0:
            pattern = ''.join(map(str, sensors))
            
            # Determine current mode
            if self.navigation_mode:
                nav_command = self.get_navigation_command()
                if nav_command:
                    mode = f"NAV->{nav_command}"
                else:
                    mode = "NAV->LINE"  # Navigation falling back to line following
            else:
                mode = "LINE"
            
            print(f"Mode: {mode}, State: {self.state}, Pattern: {pattern}, Sensors: {sensors}")
            self._last_debug = time.time()
    
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
        
        # Also update mapper with sensor data for waypoint progression
        self.mapper.update_robot_position_from_sensors(self.esp32.sensors)
        
        # Use visual odometry position as the authoritative position
        self.current_position = (vo_x, vo_y, vo_heading)
        return vo_x, vo_y, vo_heading
    
    def get_navigation_command(self):
        """Get navigation command when in navigation mode"""
        if not self.navigation_mode:
            return None
        
        x, y, heading = self.current_position
        
        # Get next waypoint
        waypoint = self.mapper.get_current_waypoint()
        if waypoint is None:
            logging.info("All waypoints reached! Restarting waypoint sequence")
            # Restart waypoint sequence instead of disabling navigation
            self.mapper.current_waypoint = 0
            waypoint = self.mapper.get_current_waypoint()
            if waypoint is None:
                # If still no waypoints, temporarily use line following
                return None
        
        target_x, target_y = waypoint
        
        # Get direction to navigate
        direction = self.get_navigation_direction(x, y, heading, target_x, target_y)
        
        # Calculate distance to target
        distance = math.sqrt((target_x - x)**2 + (target_y - y)**2)
        
        # For maze navigation, prioritize navigation over line following at longer distances
        # Only fall back to line following when very close or if we're lost
        if distance < 0.08:  # Within 8cm, use line following for final precision
            return None  # Fall back to line following
        
        # If distance is getting larger (we might be going wrong way), fall back temporarily
        if hasattr(self, '_last_distance') and distance > self._last_distance * 1.5:
            if not hasattr(self, '_fallback_counter'):
                self._fallback_counter = 0
            self._fallback_counter += 1
            if self._fallback_counter > 10:  # Fall back for a few cycles
                self._fallback_counter = 0
                return None
        else:
            self._fallback_counter = 0
        
        self._last_distance = distance
        
        # Debug output (less frequent)
        if not hasattr(self, '_last_nav_debug'):
            self._last_nav_debug = 0
        if time.time() - self._last_nav_debug > 3.0:  # Every 3 seconds
            logging.info(f"Navigating to waypoint {self.mapper.current_waypoint}: ({target_x:.2f}, {target_y:.2f}), distance: {distance:.2f}m, direction: {direction}")
            self._last_nav_debug = time.time()
        
        return direction

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
                'grid_image': grid_image
            }
            
            return jsonify(data)
        
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
            print("DEBUG: Starting map visualization creation...")
            
            # Get the maze visualization from mapper
            rgb_grid = self.robot.mapper.get_grid_visualization()
            print(f"DEBUG: Grid shape: {rgb_grid.shape}")
            print(f"DEBUG: Grid type: {type(rgb_grid)}")
            
            # Resize for better visibility (make it bigger)
            rgb_grid = cv2.resize(rgb_grid, (500, 400), interpolation=cv2.INTER_NEAREST)
            print(f"DEBUG: Resized grid shape: {rgb_grid.shape}")
            
            # Convert to base64
            _, buffer = cv2.imencode('.png', rgb_grid)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            print(f"DEBUG: Base64 length: {len(img_base64)}")
            
            return img_base64
            
        except Exception as e:
            print(f"DEBUG: Error creating grid visualization: {e}")
            logging.error(f"Error creating grid visualization: {e}")
            
            # Return a simple test image as fallback
            test_img = np.full((400, 500, 3), 64, dtype=np.uint8)  # Dark background
            cv2.putText(test_img, 'MAP ERROR', (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.rectangle(test_img, (50, 50), (450, 350), (0, 255, 255), 2)
            
            _, buffer = cv2.imencode('.png', test_img)
            fallback_base64 = base64.b64encode(buffer).decode('utf-8')
            print(f"DEBUG: Fallback image length: {len(fallback_base64)}")
            
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
        print("Cyberpunk web visualization initialized")
        
        # Start web server automatically in background
        web_thread = threading.Thread(target=web_viz.start_web_server, daemon=True)
        web_thread.start()
        print("Cyberpunk dashboard started at http://0.0.0.0:5000")
    except Exception as e:
        print(f"Could not initialize web visualization: {e}")
        web_viz = None
    
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