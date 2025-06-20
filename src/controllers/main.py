#!/usr/bin/env python3

import time
import logging
import cv2
import numpy as np
import math
import socket
import threading
import base64
from typing import List, Tuple, Optional
from flask import Flask, jsonify, render_template, request, Response

# Import our clean modules
from object_detection import ObjectDetector, PathShapeDetector
from pathfinder import Pathfinder
from box import BoxHandler
from position_tracker import OmniWheelOdometry, PositionTracker
from pid import LineFollowPID
from camera_line_detector import CameraLineDetector

# ================================
# FEATURE CONFIGURATION
# ================================
# Enable/disable features for easy testing and debugging
FEATURES = {
    'OBJECT_DETECTION_ENABLED': False,      # Enable YOLO object detection - DISABLED for performance
    'PATH_SHAPE_DETECTION_ENABLED': True,   # Enable path shape analysis
    'OBSTACLE_AVOIDANCE_ENABLED': False,     # Enable obstacle avoidance behavior
    'VISION_SYSTEM_ENABLED': True,          # Enable camera and vision processing - ENABLED for camera line following
    'USE_ESP32_LINE_SENSOR': False,          # Use ESP32 hardware sensor for line following
    'POSITION_CORRECTION_ENABLED': True,    # Enable waypoint position corrections
    'PERFORMANCE_LOGGING_ENABLED': True,    # Enable detailed performance logging
    'DEBUG_VISUALIZATION_ENABLED': True,    # Enable debug visualization windows - ENABLED to see camera feed
    'SMOOTH_CORNERING_ENABLED': True,       # Enable smooth cornering like normal wheels
    'ADAPTIVE_SPEED_ENABLED': True,         # Enable speed adaptation based on conditions
}

# ================================
# ROBOT CONFIGURATION
# ================================
ESP32_IP = "192.168.128.245"
CELL_SIZE_M = 0.11
BASE_SPEED = 60
TURN_SPEED = 50
CORNER_SPEED = 55  # Slower speed for smooth cornering

# Robot physical constants
PULSES_PER_REV = 960
WHEEL_DIAMETER_M = 0.025
ROBOT_WIDTH_M = 0.225
ROBOT_LENGTH_M = 0.075

# Mission configuration
START_CELL = (0, 12)
END_CELL = (8, 12)
START_POSITION = ((START_CELL[0] + 0.5) * CELL_SIZE_M, (START_CELL[1] + 0.5) * CELL_SIZE_M)
START_HEADING = 0.0  # Facing right for horizontal movement

# Line following configuration
LINE_FOLLOW_SPEED = 50

# Vision configuration
IMG_PATH_SRC_PTS = np.float32([[200, 300], [440, 300], [580, 480], [60, 480]])
IMG_PATH_DST_PTS = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])

class ESP32Bridge:
    """ESP32 communication bridge for motors, encoders, and line sensors."""
    
    def __init__(self, ip: str, port: int = 1234):
        self.ip = ip
        self.port = port
        self.socket = None
        self.connected = False
        self.connection_attempts = 0
        
        # Latest sensor data from ESP32
        self.latest_encoder_data = [0, 0, 0, 0]
        self.latest_line_position = -1  # -1 means no line detected
        self.latest_line_error = 0
        self.latest_sensor_values = [0, 0, 0, 0, 0]
        
        # Command tracking
        self.last_command = None
        self.last_send_time = 0.0
        
    def start(self):
        """Start communication with ESP32."""
        return self.connect()
        
    def connect(self):
        """Establish connection to ESP32."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        try:
            import socket
            self.socket = socket.create_connection((self.ip, self.port), timeout=3)
            self.socket.settimeout(0.5)
            self.connected = True
            self.connection_attempts = 0
            print(f"Connected to ESP32 at {self.ip}:{self.port}")
            return True
        except Exception as e:
            self.connected = False
            self.connection_attempts += 1
            if self.connection_attempts % 10 == 1:  # Log every 10 attempts
                print(f"Failed to connect to ESP32 (attempt {self.connection_attempts}): {e}")
            self.socket = None
            return False
        
    def send_motor_speeds(self, fl: int, fr: int, bl: int, br: int):
        """Send motor speeds to ESP32."""
        if not self.connected and not self.connect():
            return False
            
        try:
            command = f"{fl},{fr},{bl},{br}"
            return self._send_command(command)
        except Exception as e:
            print(f"Error sending motor speeds: {e}")
            self.connected = False
            return False
    
    def send_line_follow_command(self, base_speed: int = 60):
        """Send line following command to ESP32."""
        if not self.connected and not self.connect():
            return False
            
        try:
            command = f"LINE_FOLLOW,{base_speed}"
            return self._send_command(command)
        except Exception as e:
            print(f"Error sending line follow command: {e}")
            self.connected = False
            return False
    
    def send_calibrate_command(self):
        """Send calibration command to ESP32."""
        if not self.connected and not self.connect():
            return False
            
        try:
            return self._send_command("CALIBRATE")
        except Exception as e:
            print(f"Error sending calibrate command: {e}")
            self.connected = False
            return False
    
    def _send_command(self, command: str):
        """Internal method to send command to ESP32."""
        if not self.socket:
            return False
            
        try:
            full_command = f"{command}\n"
            current_time = time.time()
            
            # Send commands to ESP32
            self.socket.sendall(full_command.encode())
            self.last_command = full_command
            self.last_send_time = current_time
            
            if FEATURES['PERFORMANCE_LOGGING_ENABLED']:
                print(f"Sent to ESP32: {command}")
            
            return True
        except Exception as e:
            print(f"Socket error: {e}")
            self.connected = False
            self.socket = None
            return False
            
    def get_encoder_ticks(self) -> List[int]:
        """Get encoder ticks from ESP32."""
        self._receive_data()  # Try to get fresh data
        return self.latest_encoder_data
    
    def get_line_sensor_data(self) -> Tuple[int, int, List[int]]:
        """
        Get line sensor data from ESP32.
        
        Returns:
            Tuple of (line_position, line_error, sensor_values)
            line_position: 0-4000 (0=leftmost, 2000=center, 4000=rightmost, -1=no line)
            line_error: -2000 to +2000 (error from center)
            sensor_values: List of 5 calibrated sensor readings (0-1000 each)
        """
        self._receive_data()  # Try to get fresh data
        return (self.latest_line_position, self.latest_line_error, self.latest_sensor_values)
    
    def is_line_detected(self) -> bool:
        """Check if line is currently detected."""
        self._receive_data()  # Try to get fresh data
        return self.latest_line_position != -1
    
    def _receive_data(self):
        """Try to receive data from ESP32 (non-blocking)."""
        if not self.socket or not self.connected:
            return
            
        try:
            # Non-blocking receive
            data = self.socket.recv(1024)
            if data:
                data_string = data.decode().strip()
                for line in data_string.split('\n'):
                    if line:
                        self.update_sensor_data(line)
        except Exception:
            # No data available or connection error - that's okay for non-blocking
            pass
    
    def update_sensor_data(self, data_string: str):
        """
        Update sensor data from received ESP32 data.
        Expected format: "encoder_fl,encoder_fr,encoder_bl,encoder_br,line_pos,line_error,s0,s1,s2,s3,s4"
        """
        try:
            parts = data_string.strip().split(',')
            if len(parts) >= 11:
                # Update encoder data
                self.latest_encoder_data = [int(parts[i]) for i in range(4)]
                
                # Update line sensor data
                self.latest_line_position = int(parts[4])
                self.latest_line_error = int(parts[5])
                self.latest_sensor_values = [int(parts[i]) for i in range(6, 11)]
                
                # Show sensor readings in your format [Left Center Right]
                if len(self.latest_sensor_values) >= 3:
                    left = 1 if self.latest_sensor_values[0] < 500 else 0
                    center = 1 if self.latest_sensor_values[2] < 500 else 0  
                    right = 1 if self.latest_sensor_values[4] < 500 else 0
                    print(f"[{left} {center} {right}]")
        except (ValueError, IndexError):
            # Invalid data format - keep previous values
            if FEATURES['PERFORMANCE_LOGGING_ENABLED']:
                print(f"Invalid data format: {data_string}")
        
    def stop(self):
        """Stop ESP32 communication."""
        if self.socket:
            try:
                self._send_command("STOP")
                time.sleep(0.1)
                self.socket.close()
            except:
                pass
            self.socket = None
        self.connected = False



class RobotController:
    """Main robot controller using modular components."""
    
    def __init__(self):
        """Initialize robot controller with all modules."""
        # Hardware interfaces
        self.esp32_bridge = ESP32Bridge(ESP32_IP, 1234)
        
        # Position tracking
        initial_pose = (START_POSITION[0], START_POSITION[1], START_HEADING)
        self.odometry = OmniWheelOdometry(
            initial_pose=initial_pose,
            pulses_per_rev=PULSES_PER_REV,
            wheel_diameter=WHEEL_DIAMETER_M,
            robot_width=ROBOT_WIDTH_M,
            robot_length=ROBOT_LENGTH_M
        )
        self.position_tracker = PositionTracker(self.odometry, CELL_SIZE_M)
        
        # Navigation
        self.pathfinder = Pathfinder([], CELL_SIZE_M)  # Initialize with empty grid first
        maze_grid = self.pathfinder.create_maze_grid()  # Get the maze grid
        self.pathfinder = Pathfinder(maze_grid, CELL_SIZE_M)  # Reinitialize with actual grid
        self.current_path = None
        self.current_waypoint_idx = 0
        
        # Vision system
        self.camera = None
        self.object_detector = None
        self.path_detector = None
        self.camera_line_detector = None
        self._setup_vision()
        
        # Control
        self.line_follower = LineFollowPID()
        self.state = "STARTING"
        self.latest_frame = None
        self.last_known_line_position = 0.0
        self.camera_line_position = None
        self.recovery_start_time = None
    
    def set_line_detector_threshold(self, value: int):
        """Sets the threshold for the camera line detector."""
        if self.camera_line_detector:
            self.camera_line_detector.set_threshold(value)

    def _setup_vision(self):
        """Initialize camera and vision systems based on feature flags."""
        if not FEATURES['VISION_SYSTEM_ENABLED']:
            self.camera = None
            self.object_detector = None
            self.path_detector = None
            return
            
        try:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Initialize object detector if enabled
            if FEATURES['OBJECT_DETECTION_ENABLED']:
                self.object_detector = ObjectDetector('yolo11n.pt', confidence_threshold=0.5)
            else:
                self.object_detector = None
            
            # Initialize path shape detector if enabled
            if FEATURES['PATH_SHAPE_DETECTION_ENABLED']:
                self.path_detector = PathShapeDetector(IMG_PATH_SRC_PTS, IMG_PATH_DST_PTS)
            else:
                self.path_detector = None

            # Initialize camera line detector if needed
            if not FEATURES['USE_ESP32_LINE_SENSOR']:
                self.camera_line_detector = CameraLineDetector(
                    width=640, 
                    height=480,
                    src_pts=IMG_PATH_SRC_PTS,
                    dst_pts=IMG_PATH_DST_PTS
                )
                print("Camera-based line following is enabled.")
            
        except Exception as e:
            if FEATURES['PERFORMANCE_LOGGING_ENABLED']:
                print(f"Vision system initialization failed: {e}")
            self.camera = None
            self.object_detector = None
            self.path_detector = None
    
    def run(self):
        """Main control loop."""
        self.esp32_bridge.start()
        
        try:
            while True:
                # Update position tracking
                encoder_ticks = self.esp32_bridge.get_encoder_ticks()
                self.position_tracker.update(encoder_ticks)
                
                # Update ESP32 sensor data (this would normally come from network)
                # For now, we'll simulate receiving data
                # self.esp32_bridge.update_sensor_data(received_data_string)
                
                # Process vision
                self._process_vision()
                
                # State machine
                self._run_state_machine()
                
                time.sleep(0.05)  # 20Hz control loop
                
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def _process_vision(self):
        """Process camera frame for object detection and path analysis."""
        if not FEATURES['VISION_SYSTEM_ENABLED'] or self.camera is None:
            return
            
        ret, frame = self.camera.read()
        if ret:
            self.latest_frame = frame.copy()
            
            # Object detection
            if FEATURES['OBJECT_DETECTION_ENABLED'] and self.object_detector:
                detections = self.object_detector.detect_objects(frame)
                self._handle_detections(detections)
            
            # Path shape detection
            if FEATURES['PATH_SHAPE_DETECTION_ENABLED'] and self.path_detector:
                path_shape = self.path_detector.detect_path_shape(frame)
                self._handle_path_shape(path_shape)
            
            # Camera-based line detection
            if not FEATURES['USE_ESP32_LINE_SENSOR'] and self.camera_line_detector:
                line_pos, confidence, navigation_img = self.camera_line_detector.detect(frame)
                self.camera_line_position = line_pos
                
                if FEATURES['DEBUG_VISUALIZATION_ENABLED']:
                    # Show enhanced navigation view
                    cv2.imshow('Navigation View', navigation_img)
                    
                    # Show original camera view with line position overlay
                    display_frame = frame.copy()
                    if line_pos is not None:
                        # Draw line position on original frame
                        line_px = int((line_pos * 320) + 320)  # Convert normalized to pixel
                        cv2.line(display_frame, (line_px, 0), (line_px, 480), (0, 255, 0), 3)
                        
                        # Add status text
                        status_text = f"Line Pos: {line_pos:.2f} | Conf: {confidence:.2f}"
                        cv2.putText(display_frame, status_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, "NO LINE DETECTED", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    cv2.imshow('Robot Camera', display_frame)

            # Debug visualization
            if FEATURES['DEBUG_VISUALIZATION_ENABLED']:
                cv2.waitKey(1)
    
    def _handle_detections(self, detections: dict):
        """Handle object detection results."""
        if FEATURES['OBSTACLE_AVOIDANCE_ENABLED'] and detections['obstacle_detected']:
            if self.state not in ["AVOIDING_OBSTACLE"]:
                self.state = "AVOIDING_OBSTACLE"
                if FEATURES['PERFORMANCE_LOGGING_ENABLED']:
                    print("Obstacle detected - switching to avoidance mode")
    
    def _handle_path_shape(self, path_shape: str):
        """Handle path shape detection results."""
        # Path shape detection
    
    def _run_state_machine(self):
        """Main state machine logic."""
        if self.state == "STARTING":
            self._start_mission()
        elif self.state == "PLANNING_PATH":
            self._plan_path_to_target()
        elif self.state == "FOLLOWING_PATH":
            self._follow_path()
        elif self.state == "RECOVERING_LINE":
            self._recover_line()
        elif self.state == "AVOIDING_OBSTACLE":
            self._handle_obstacle_avoidance()
        elif self.state == "MISSION_COMPLETE":
            self._handle_mission_complete()
    
    def _start_mission(self):
        """Initialize mission."""
        print("Starting robot mission...")
        
        # Check for valid configuration
        if not FEATURES['USE_ESP32_LINE_SENSOR'] and not FEATURES['VISION_SYSTEM_ENABLED']:
            print("ERROR: Camera line following requires VISION_SYSTEM_ENABLED to be True.")
            print("Please update FEATURE configuration. Exiting.")
            self.state = "MISSION_COMPLETE" # Stop execution
            return
        
        # Calibrate sensors if needed
        if not self.esp32_bridge.connected:
            print("Connecting to ESP32...")
            if not self.esp32_bridge.connect():
                print("Warning: ESP32 not connected - running in simulation mode")
        else:
            print("ESP32 connected - starting real mission")
        
        self.state = "PLANNING_PATH"
    
    def _plan_path_to_target(self):
        """Plan path to the end cell."""
        current_cell = self.position_tracker.get_current_cell()
        
        if self.state == "MISSION_COMPLETE" or current_cell == END_CELL:
            self.state = "MISSION_COMPLETE"
            return
            
        print(f"Planning path from {current_cell} to {END_CELL}...")
        self.current_path = self.pathfinder.find_path(current_cell, END_CELL)
        
        if self.current_path:
            self.current_waypoint_idx = 0
            self.state = "FOLLOWING_PATH"
            print(f"Path found! Length: {len(self.current_path)} waypoints.")
        else:
            print(f"Could not find a path from {current_cell} to {END_CELL}")
            self.state = "MISSION_COMPLETE"
    
    def _follow_path(self):
        """Follow the planned path using line following."""
        if not self.current_path or self.current_waypoint_idx >= len(self.current_path):
            # Reached end of path
            print("Reached the end of the path.")
            self.state = "MISSION_COMPLETE"
            return

        # Check if we're at current waypoint
        current_waypoint = self.current_path[self.current_waypoint_idx]
        if self.position_tracker.is_at_cell(current_waypoint[0], current_waypoint[1]):
            # Correct odometry at waypoint if enabled
            if FEATURES['POSITION_CORRECTION_ENABLED']:
                # We'll disable the correction to prevent it from interfering with the line follower,
                # but we'll keep the waypoint tracking active.
                # self.position_tracker.correct_at_waypoint(current_waypoint)
                print(f"--- Reached waypoint {self.current_waypoint_idx}: {current_waypoint} ---")
            
            # Move to next waypoint
            self.current_waypoint_idx += 1
            
        # Check if we are at an intersection and should handle a turn
        if self._is_at_intersection() and self.position_tracker.is_at_cell(current_waypoint[0], current_waypoint[1]):
            self._handle_intersection()
            return # Skip line following for one cycle to execute the turn

        # Follow line using either hardware sensor or camera
        if FEATURES['USE_ESP32_LINE_SENSOR']:
            self._follow_line_with_sensor()
        else:
            self._follow_line_with_camera()
    
    def _get_line_following_control(self, normalized_position):
        """Calculates the control signals for line following and applies cornering logic."""
        
        # Get base control signals from PID
        vx, vy, omega = self.line_follower.calculate_control(normalized_position, base_speed=LINE_FOLLOW_SPEED)
        
        # Apply smooth cornering logic if enabled
        if FEATURES['SMOOTH_CORNERING_ENABLED']:
            # Eliminate sideways movement for car-like turning
            vy = 0
            
            # Slow down in corners based on rotation speed
            # The faster the rotation, the slower the forward speed
            speed_reduction = abs(omega) / TURN_SPEED 
            vx = max(CORNER_SPEED, LINE_FOLLOW_SPEED * (1 - speed_reduction))

        self._move_omni(vx, vy, omega)

    def _follow_line_with_sensor(self):
        """Follow line using the ESP32 hardware sensors."""
        if self.esp32_bridge.is_line_detected():
            line_position, _, _ = self.esp32_bridge.get_line_sensor_data()
            # Convert ESP32 line position (0-4000) to our expected range (-1.0 to 1.0)
            normalized_position = (line_position - 2000) / 2000.0
            self.last_known_line_position = normalized_position
            
            self._get_line_following_control(normalized_position)
            
            # Performance logging
            if FEATURES['PERFORMANCE_LOGGING_ENABLED']:
                stats = self.position_tracker.get_tracking_statistics()
                if stats['status']['is_strafing']:
                    print(f"Strafing: pos={normalized_position:.2f}, strafe_eff={stats['performance']['strafe_efficiency']:.2f}")
        else:
            # Line lost - initiate recovery
            if self.state == "FOLLOWING_PATH": # Only trigger once
                print("Line lost! Attempting to recover...")
                self.state = "RECOVERING_LINE"
                self.recovery_start_time = time.time()
            self._stop_motors()

    def _follow_line_with_camera(self):
        """Follow line using the camera."""
        if self.camera_line_position is not None:
            self.last_known_line_position = self.camera_line_position
            
            self._get_line_following_control(self.camera_line_position)
        else:
            # Line lost - initiate recovery
            if self.state == "FOLLOWING_PATH":
                print("Line lost (camera)! Attempting to recover...")
                self.state = "RECOVERING_LINE"
                self.recovery_start_time = time.time()
            self._stop_motors()
    
    def _handle_obstacle_avoidance(self):
        """Handle obstacle avoidance."""
        # Executing obstacle avoidance
        
        # Simple avoidance: stop, turn around, replan
        self._stop_motors()
        time.sleep(1.0)
        
        # Add obstacle to map
        current_cell = self.position_tracker.get_current_cell()
        # Estimate obstacle position
        robot_pos = self.position_tracker.odometry.get_position()
        robot_heading = self.position_tracker.odometry.get_heading()
        obstacle_x = robot_pos[0] + 0.2 * math.cos(robot_heading)
        obstacle_y = robot_pos[1] + 0.2 * math.sin(robot_heading)
        obstacle_cell = self.pathfinder.world_to_cell(obstacle_x, obstacle_y)
        
        self.pathfinder.update_obstacle(obstacle_cell[0], obstacle_cell[1], True)
        
        # Replan path
        self.state = "PLANNING_PATH"
    
    def _handle_mission_complete(self):
        """Handle mission completion."""
        self._stop_motors()
        print("Mission complete: Reached target destination.")
    
    def _move_omni(self, vx: float, vy: float, omega: float):
        """Move robot using omni-wheel kinematics."""
        R = ROBOT_WIDTH_M / 2
        
        # Inverse kinematics
        v_fl = (vx - vy - R * omega)
        v_fr = (vx + vy + R * omega)
        v_bl = (vx + vy - R * omega)
        v_br = (vx - vy + R * omega)
        
        # Scale to motor range
        speeds = [v_fl, v_fr, v_bl, v_br]
        max_speed = max(abs(s) for s in speeds)
        if max_speed > 100:
            scale = 100 / max_speed
            speeds = [s * scale for s in speeds]
        
        # Convert to integers for the ESP32
        int_speeds = [int(s) for s in speeds]
        
        self.esp32_bridge.send_motor_speeds(*int_speeds)
    
    def _stop_motors(self):
        """Stop all motors."""
        self.esp32_bridge.send_motor_speeds(0, 0, 0, 0)
    
    def stop(self):
        """Stop robot and cleanup resources."""
        self._stop_motors()
        self.esp32_bridge.stop()
        
        if self.camera:
            self.camera.release()
        
        cv2.destroyAllWindows()

    def _recover_line(self):
        """Try to find the line again after losing it."""
        # If line is found, go back to following
        if self.esp32_bridge.is_line_detected():
            print("Line re-acquired! Resuming path following.")
            self.state = "FOLLOWING_PATH"
            self.recovery_start_time = None
            self.line_follower.reset_controllers() # Reset PID to avoid integral windup jump
            return

        # Check for recovery timeout
        if self.recovery_start_time and (time.time() - self.recovery_start_time > 2.0):
            print("Line recovery failed. Stopping robot.")
            self._stop_motors()
            self.state = "MISSION_COMPLETE" # Or a new FAILED state
            return
            
        # Recovery maneuver: move backward and strafe toward last known line position
        # If last position was > 0 (right), we need to strafe right (vy > 0)
        strafe_speed = 40.0
        vy_recovery = strafe_speed if self.last_known_line_position > 0 else -strafe_speed
        vx_recovery = -20.0 # Slowly move backward

        self._move_omni(vx_recovery, vy_recovery, 0)

    def _is_at_intersection(self) -> bool:
        """Check if the robot is at an intersection based on sensor readings."""
        if not self.esp32_bridge.is_line_detected():
            return False
        
        _, _, sensor_values = self.esp32_bridge.get_line_sensor_data()
        
        # An intersection is detected if 3 or more sensors see the line.
        # A low sensor value (e.g., < 300) indicates a line.
        line_detections = sum(1 for value in sensor_values if value < 300)
        
        return line_detections >= 3

    def _get_turn_direction(self) -> str:
        """Calculate turn direction for the next waypoint based on path geometry."""
        if self.current_waypoint_idx + 1 >= len(self.current_path):
            return "STRAIGHT"  # End of path

        # Determine the vector of the path segment the robot is currently on
        prev_waypoint = self.current_path[self.current_waypoint_idx - 1] if self.current_waypoint_idx > 0 else START_CELL
        current_waypoint = self.current_path[self.current_waypoint_idx]
        dx_in = current_waypoint[0] - prev_waypoint[0]
        dy_in = current_waypoint[1] - prev_waypoint[1]
        angle_in = math.atan2(dy_in, dx_in)

        # Determine the vector of the path segment the robot needs to join
        next_waypoint = self.current_path[self.current_waypoint_idx + 1]
        dx_out = next_waypoint[0] - current_waypoint[0]
        dy_out = next_waypoint[1] - current_waypoint[1]
        angle_out = math.atan2(dy_out, dx_out)

        # Find the difference in angle to determine the turn
        turn_angle = angle_out - angle_in
        
        # Normalize angle to the range [-pi, pi]
        while turn_angle <= -math.pi:
            turn_angle += 2 * math.pi
        while turn_angle > math.pi:
            turn_angle -= 2 * math.pi

        # Classify the turn based on the angle
        if abs(turn_angle) < math.pi / 4:  # Less than 45 degrees is straight
            return "STRAIGHT"
        elif turn_angle > 0:
            return "LEFT"
        else:
            return "RIGHT"

    def _handle_intersection(self):
        """Handle navigation at an intersection."""
        print(f"Intersection detected at waypoint {self.current_waypoint_idx}.")

        # Decide which way to go
        turn_direction = self._get_turn_direction()
        print(f"Intersection decision: {turn_direction}")

        # Move forward a bit to center the robot over the intersection
        self._move_omni(vx=30, vy=0, omega=0)
        time.sleep(0.3)

        if turn_direction != "STRAIGHT":
            # Execute the turn by rotating in place
            turn_angle_rad = math.pi / 2  # 90 degrees
            
            # Determine omega based on turn direction
            omega = TURN_SPEED if turn_direction == "LEFT" else -TURN_SPEED
            
            # Calculate time needed for the turn. omega is in rad/s (scaled).
            # We need to calibrate this. Let's assume TURN_SPEED of 50 is approx 90deg/s.
            turn_duration = (turn_angle_rad / (abs(omega) / 50 * (math.pi / 2))) * 0.9
            
            self._move_omni(vx=0, vy=0, omega=omega)
            time.sleep(turn_duration)

        # Move forward to exit the intersection and find the new line
        self._move_omni(vx=40, vy=0, omega=0)
        time.sleep(0.4)

        # The waypoint has now been passed
        self.current_waypoint_idx += 1
        self.line_follower.reset_controllers()
        print(f"Proceeding to waypoint {self.current_waypoint_idx}.")

def print_feature_status():
    """Print current feature configuration for debugging."""
    print("=" * 50)
    print("ROBOT FEATURE CONFIGURATION")
    print("=" * 50)
    for feature, enabled in FEATURES.items():
        status = "ENABLED" if enabled else "DISABLED"
        print(f"{feature:<30} : {status}")
    print("=" * 50)
    print()

def main():
    """Main entry point."""
    print_feature_status()
    
    # Global instance to be shared between threads
    robot_controller = RobotController()
    
    # Start the robot's main control loop in a background thread
    robot_thread = threading.Thread(target=robot_controller.run, daemon=True)
    robot_thread.start()
    
    # Start the Flask web server in the main thread
    app = Flask(__name__, template_folder='../../templates', static_folder='../../static')
    
    @app.route('/')
    def index():
        """Serve the main dashboard page."""
        return render_template('navigation.html')

    @app.route('/api/robot_data')
    def robot_data():
        """Provide robot status as JSON."""
        if not robot_controller or not robot_controller.position_tracker:
            return jsonify({})

        pose = robot_controller.position_tracker.get_pose()
        robot_cell = robot_controller.position_tracker.get_current_cell()
        _, _, sensor_values = robot_controller.esp32_bridge.get_line_sensor_data()
        
        sensors = [1 if val < 500 else 0 for val in sensor_values]
        
        grid_image = generate_grid_image(
            robot_controller.pathfinder,
            robot_cell,
            robot_controller.current_path,
            START_CELL,
            END_CELL
        )

        data = {
            "position": {"x": pose[0], "y": pose[1], "heading": pose[2]},
            "state": robot_controller.state,
            "has_package": False,
            "package_detected": False,
            "collected_boxes": 0,
            "delivered_boxes": 0,
            "total_tasks": 0,
            "sensors": sensors,
            "grid_image": grid_image,
            "line_threshold": robot_controller.camera_line_detector.manual_threshold if not FEATURES['USE_ESP32_LINE_SENSOR'] and robot_controller.camera_line_detector else 0
        }
        return jsonify(data)

    @app.route('/api/set_threshold', methods=['POST'])
    def set_threshold():
        """Set the line detector threshold."""
        data = request.get_json()
        if not data or 'threshold' not in data:
            return jsonify({"status": "error", "message": "Missing threshold value"}), 400
        
        try:
            threshold_value = int(data['threshold'])
            if not 0 <= threshold_value <= 255:
                raise ValueError("Threshold must be between 0 and 255")
            
            robot_controller.set_line_detector_threshold(threshold_value)
            
            return jsonify({"status": "success", "threshold": threshold_value})
        except (ValueError, TypeError) as e:
            return jsonify({"status": "error", "message": str(e)}), 400

    @app.route('/video_feed')
    def video_feed():
        """Stream camera feed with navigation overlay."""
        response = Response(generate_camera_frames(), 
                           mimetype='multipart/x-mixed-replace; boundary=frame')
        # Add headers to prevent caching and ensure proper streaming
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['Connection'] = 'close'
        return response
    
    def generate_camera_frames():
        """Generate camera frames for streaming."""
        while True:
            try:
                if robot_controller.latest_frame is not None:
                    # Get the latest camera frame
                    frame = robot_controller.latest_frame.copy()
                    
                    # Add navigation overlay if line detection is active
                    if not FEATURES['USE_ESP32_LINE_SENSOR'] and robot_controller.camera_line_detector:
                        line_pos, confidence, navigation_img = robot_controller.camera_line_detector.detect(frame)
                        
                        # Create a side-by-side view: original camera + navigation view
                        # Resize navigation view to match camera frame height
                        nav_resized = cv2.resize(navigation_img, (frame.shape[1], frame.shape[0]))
                        
                        # Combine horizontally
                        combined_frame = np.hstack([frame, nav_resized])
                        
                        # Add labels
                        cv2.putText(combined_frame, "CAMERA VIEW", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(combined_frame, "NAVIGATION VIEW", (frame.shape[1] + 10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Add line position info on camera view
                        if line_pos is not None:
                            line_px = int((line_pos * 320) + 320)
                            cv2.line(combined_frame, (line_px, 0), (line_px, frame.shape[0]), (0, 255, 0), 3)
                            status_text = f"Line: {line_pos:.2f} | Conf: {confidence:.2f}"
                            cv2.putText(combined_frame, status_text, (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            cv2.putText(combined_frame, "NO LINE DETECTED", (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        display_frame = combined_frame
                    else:
                        # Just show camera if line detection is disabled
                        display_frame = frame
                    
                    # Encode frame as JPEG with optimized settings for streaming
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 75,  # Good quality vs size balance
                                   cv2.IMWRITE_JPEG_OPTIMIZE, 1]    # Optimize for smaller file size
                    ret, buffer = cv2.imencode('.jpg', display_frame, encode_params)
                    
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # No frame available, send a black frame to keep the stream alive
                    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(black_frame, "WAITING FOR CAMERA...", (150, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', black_frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
            except Exception as e:
                print(f"Error in video stream: {e}")
                # Send error frame
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, "VIDEO ERROR", (250, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', error_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS

    print("Starting Flask web server...")
    app.run(host='0.0.0.0', port=5000)

def generate_grid_image(pathfinder, robot_cell, path, start_cell, end_cell):
    cell_size = 20
    grid = pathfinder.grid
    height, width = len(grid), len(grid[0])
    
    img = np.zeros((height * cell_size, width * cell_size, 3), dtype=np.uint8) + 26

    for y in range(height):
        for x in range(width):
            if grid[y][x] == 1:
                cv2.rectangle(img, (x * cell_size, y * cell_size), ((x+1) * cell_size, (y+1) * cell_size), (22, 33, 62), -1)

    if path:
        for (x, y) in path:
             cv2.rectangle(img, (x * cell_size, y * cell_size), ((x+1) * cell_size, (y+1) * cell_size), (0, 255, 255), -1)

    if start_cell:
        x, y = start_cell
        cv2.rectangle(img, (x * cell_size, y * cell_size), ((x+1) * cell_size, (y+1) * cell_size), (0, 255, 65), -1)
        
    if end_cell:
        x, y = end_cell
        cv2.rectangle(img, (x * cell_size, y * cell_size), ((x+1) * cell_size, (y+1) * cell_size), (255, 140, 0), -1)

    if robot_cell:
        x, y = robot_cell
        cv2.rectangle(img, (x * cell_size, y * cell_size), ((x+1) * cell_size, (y+1) * cell_size), (255, 0, 128), -1)

    is_success, buffer = cv2.imencode(".png", img)
    if not is_success:
        return None
    
    return base64.b64encode(buffer).decode('utf-8')

if __name__ == '__main__':
    main() 