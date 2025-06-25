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
from pid import PIDController
from intersection_detector import IntersectionDetector

# ================================
# FEATURE CONFIGURATION
# ================================
# Enable/disable features for easy testing and debugging
FEATURES = {
    'OBJECT_DETECTION_ENABLED': False,      # Enable YOLO object detection 
    'PATH_SHAPE_DETECTION_ENABLED': False,   # Enable path shape analysis
    'OBSTACLE_AVOIDANCE_ENABLED': False,     # Enable obstacle avoidance behavior
    'VISION_SYSTEM_ENABLED': True,          # Enable camera and vision processing
    'INTERSECTION_CORRECTION_ENABLED': True,# Enable intersection-based position correction
    'USE_ESP32_LINE_SENSOR': True,          # Use ESP32 hardware sensor for line following
    'POSITION_CORRECTION_ENABLED': True,    # Enable waypoint position corrections
    'PERFORMANCE_LOGGING_ENABLED': True,    # Enable detailed performance logging
    'DEBUG_VISUALIZATION_ENABLED': False,   # Enable debug visualization windows - DISABLED for headless operation
    'SMOOTH_CORNERING_ENABLED': True,       # Enable smooth cornering like normal wheels
    'ADAPTIVE_SPEED_ENABLED': True,         # Enable speed adaptation based on conditions
}

# ================================
# ROBOT CONFIGURATION
# ================================
ESP32_IP = "192.168.83.245"
CELL_SIZE_M = 0.11
BASE_SPEED = 60
TURN_SPEED = 50
CORNER_SPEED = 55  

# Robot physical constants
PULSES_PER_REV = 920
WHEEL_DIAMETER_M = 0.025
ROBOT_WIDTH_M = 0.225
ROBOT_LENGTH_M = 0.075

# Mission configuration
START_CELL = (14, 14)
END_CELL = (2, 0)
START_POSITION = ((START_CELL[0] + 0.5) * CELL_SIZE_M, (START_CELL[1] + 0.5) * CELL_SIZE_M)
START_HEADING = 0.0  # Facing right for horizontal movement

# Line following configuration
LINE_FOLLOW_SPEED = 50

# Corner turning configuration
CORNER_TURN_MODES = {
    'SMOOTH': 'smooth',     
    'SIDEWAYS': 'sideways',       
    'PIVOT': 'pivot',             
    'FRONT_TURN': 'front_turn'    
}

# Corner detection thresholds
CORNER_DETECTION_THRESHOLD = 0.35    # Line offset to detect corner
CORNER_TURN_DURATION = 30            # Frames to execute corner turn
SHARP_CORNER_THRESHOLD = 0.6         # Threshold for sharp vs gentle corners

# Vision configuration (adjusted for webcam processing resolution)
# These points define the perspective transformation for path detection
# Adjusted for 640x480 processing resolution
IMG_PATH_SRC_PTS = np.float32([[160, 240], [480, 240], [640, 480], [0, 480]])
IMG_PATH_DST_PTS = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])

# Camera configuration - USB Webcam
# Webcam specs: 1920x1080 @ 30 FPS with integrated microphone
WEBCAM_INDEX = 0  # Usually 0 for built-in camera, 1 for external USB webcam
CAMERA_WIDTH, CAMERA_HEIGHT = 1920, 1080  # Full HD resolution as per webcam specs
CAMERA_FPS = 30  # 30 FPS as specified in webcam specs
# Note: For processing efficiency, frames are resized to 640x480 for vision algorithms

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
        self.latest_line_position = -1  
        self.latest_line_error = 0
        self.latest_sensor_values = [0, 0, 0] 
        
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
        
        if self.connection_attempts > 0:
            time.sleep(1)

        try:
            self.socket = socket.create_connection((self.ip, self.port), timeout=3)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            if hasattr(socket, "TCP_KEEPIDLE"):
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 1)
            if hasattr(socket, "TCP_KEEPINTVL"):
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 3)
            if hasattr(socket, "TCP_KEEPCNT"):
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)

            self.socket.settimeout(0.5)
            self.connected = True
            self.connection_attempts = 0
            print(f"Connected to ESP32 at {self.ip}:{self.port}")
            return True
        except Exception as e:
            self.connected = False
            self.connection_attempts += 1
            if self.connection_attempts % 10 == 1:
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
        self._receive_data()
        return self.latest_encoder_data
    
    def get_line_sensor_data(self) -> Tuple[int, int, List[int]]:
        """
        Get line sensor data from ESP32.
        
        Returns:
            Tuple of (line_position, line_error, sensor_values)
            line_position: 0-2000 for 3 sensors (0=left, 1000=center, 2000=right, -1=no line)
            line_error: -1000 to +1000 (error from center)
            sensor_values: List of 3 calibrated sensor readings (0-1000 each)
        """
        self._receive_data()
        return (self.latest_line_position, self.latest_line_error, self.latest_sensor_values)
    
    def is_line_detected(self) -> bool:
        """Check if line is currently detected."""
        self._receive_data()
        return self.latest_line_position != -1
    
    def _receive_data(self):
        """Try to receive data from ESP32 (non-blocking)."""
        if not self.socket or not self.connected:
            return
            
        try:
            data = self.socket.recv(1024)
            if data:
                data_string = data.decode().strip()
                for line in data_string.split('\n'):
                    if line:
                        self.update_sensor_data(line)
            elif len(data) == 0:
                print("ESP32 closed the connection.")
                self.connected = False
                self.socket.close()
                self.socket = None
        except (socket.timeout, BlockingIOError):
            pass
        except Exception as e:
            print(f"Socket receive error: {e}")
            self.connected = False
            self.socket = None
            
    def update_sensor_data(self, data_string: str):
        """Update sensor data from ESP32 string."""
        try:
            if FEATURES['PERFORMANCE_LOGGING_ENABLED']:
                print(f"Received from ESP32: {data_string}")

            parts = data_string.split(',')
            
            if parts[0] == "ENCODERS" and len(parts) == 5:
                self.latest_encoder_data = [int(p) for p in parts[1:]]
            elif parts[0] == "LINE" and len(parts) == 6:
                self.latest_line_position = int(parts[1])
                self.latest_line_error = int(parts[2])
                self.latest_sensor_values = [int(p) for p in parts[3:]]

        except (ValueError, IndexError) as e:
            print(f"Error parsing sensor data '{data_string}': {e}")

    def stop(self):
        """Close connection to ESP32."""
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None


class RobotController:
    """Main robot controller."""

    def __init__(self):
        self.running = True
        self.esp32 = ESP32Bridge(ESP32_IP)
        
        self.object_detector = None
        self.path_shape_detector = None
        self.intersection_detector = None
        self.frame = None
        self.processed_frame = None
        self.frame_lock = threading.Lock()
        
        self.detections = {}
        self.path_shape = "straight"
        self.last_intersection_time = 0
        
        # Odometry and position tracking setup
        initial_pose = (START_POSITION[0], START_POSITION[1], START_HEADING)
        odometry = OmniWheelOdometry(
            initial_pose=initial_pose,
            pulses_per_rev=PULSES_PER_REV,
            wheel_diameter=WHEEL_DIAMETER_M,
            robot_width=ROBOT_WIDTH_M,
            robot_length=ROBOT_LENGTH_M
        )
        self.position_tracker = PositionTracker(odometry=odometry, cell_size_m=CELL_SIZE_M)
        
        # Pathfinder setup
        maze_grid = Pathfinder.create_maze_grid()
        self.pathfinder = Pathfinder(grid=maze_grid, cell_size_m=CELL_SIZE_M)
        
        self.path = []
        self.current_target_index = 0
        
        self.line_pid = PIDController(kp=0.5, ki=0.01, kd=0.1, output_limits=(-100, 100))
        self.state = "idle"
        
        if FEATURES['VISION_SYSTEM_ENABLED']:
            self._setup_vision()

    def set_line_detector_threshold(self, value: int):
        """Placeholder for setting line detector threshold."""
        print("Camera line detector is disabled. Threshold not set.")

    def _setup_vision(self):
        """Initialize vision systems if enabled."""
        if not FEATURES['VISION_SYSTEM_ENABLED']:
            print("Vision system is disabled.")
            return

        print("Initializing vision system...")
        if FEATURES['INTERSECTION_CORRECTION_ENABLED']:
            self.intersection_detector = IntersectionDetector(debug=FEATURES['DEBUG_VISUALIZATION_ENABLED'])
        if FEATURES['OBJECT_DETECTION_ENABLED']:
            self.object_detector = ObjectDetector()
        if FEATURES['PATH_SHAPE_DETECTION_ENABLED']:
            self.path_shape_detector = PathShapeDetector(
                source_pts=IMG_PATH_SRC_PTS,
                dest_pts=IMG_PATH_DST_PTS,
                debug=FEATURES['DEBUG_VISUALIZATION_ENABLED']
            )

    def run(self):
        """Main control loop."""
        if not self.esp32.start():
            print("WARNING: ESP32 not connected. Running in simulation mode for pathfinding visualization.")
        
        # Auto-start mission for pathfinding visualization
        self._start_mission()

        while self.running:
            start_time = time.time()
            
            # Update odometry
            encoder_ticks = self.esp32.get_encoder_ticks()
            if encoder_ticks:
                self.position_tracker.update(encoder_ticks)
            
            self._run_state_machine()

            if FEATURES['PERFORMANCE_LOGGING_ENABLED']:
                cycle_time = (time.time() - start_time) * 1000
                if cycle_time > 2: # Log only if loop time is significant
                    print(f"Loop time: {cycle_time:.2f} ms")
            
            time.sleep(0.01)
        
        self.stop()
        
    def _start_mission(self):
        """Start the defined mission."""
        if self.state != "idle":
            print("Cannot start mission, robot is not idle.")
            return

        print("Starting mission...")
        self.position_tracker.odometry.set_pose(START_POSITION[0], START_POSITION[1], START_HEADING)
        self.state = "planning"

    def _process_vision(self):
        """Process the latest camera frame for events."""
        if not FEATURES['VISION_SYSTEM_ENABLED'] or self.frame is None:
            return

        with self.frame_lock:
            frame_copy = self.frame.copy()

        processed_frame = frame_copy
        
        if FEATURES['INTERSECTION_CORRECTION_ENABLED'] and self.intersection_detector:
            # Only check for intersection every so often to avoid multiple triggers
            if time.time() - self.last_intersection_time > 3.0: # 3 second cooldown
                intersection_type = self.intersection_detector.detect(frame_copy)
                if intersection_type:
                    print(f"Intersection detected! Type: {intersection_type}. Recalibrating position.")
                    self.position_tracker.recalibrate_position_to_nearest_cell()
                    self.last_intersection_time = time.time()
                
                if FEATURES['DEBUG_VISUALIZATION_ENABLED']:
                    processed_frame = self.intersection_detector.draw_debug_info(processed_frame, intersection_type)

        # Store the processed frame for the video feed
        with self.frame_lock:
            self.processed_frame = processed_frame

    def _plan_path_to_target(self):
        """Plan the path to the target cell."""
        current_cell = self.position_tracker.get_current_cell()
        print(f"Planning path from {current_cell} to {END_CELL}...")
        
        path_nodes = self.pathfinder.find_path(current_cell, END_CELL)
        
        if path_nodes:
            self.path = path_nodes
            self.current_target_index = 0
            self.state = "path_following"
            print(f"Path planned: {len(self.path)} waypoints")
        else:
            print(f"Failed to plan path from {current_cell} to {END_CELL}")
            self.state = "error"

    def _follow_path(self):
        """Follow the planned path using line following."""
        if not self.path or self.current_target_index >= len(self.path):
            print("Mission complete!")
            self.state = "mission_complete"
            self._stop_motors()
            return
        
        current_cell = self.position_tracker.get_current_cell()
        target_cell = self.path[self.current_target_index]
        
        if current_cell == target_cell:
            print(f"Reached waypoint {self.current_target_index}: {target_cell}")
            self.current_target_index += 1
        
        self._follow_line_with_sensor()

    def _run_state_machine(self):
        """Run the robot's state machine."""
        if self.state == "idle":
            self._stop_motors()
        elif self.state == "planning":
            self._plan_path_to_target()
        elif self.state == "path_following":
            self._follow_path()
        elif self.state == "mission_complete":
            self._stop_motors()
        elif self.state == "error":
            self._stop_motors()
            self.running = False
            
    def _follow_line_with_sensor(self):
        """Follow the line using ESP32 line sensor data."""
        line_pos, line_err, sensor_vals = self.esp32.get_line_sensor_data()
        
        if line_pos == -1:
            print("Line lost!")
            self._recover_line()
            return

        # Detect corners based on line error magnitude
        abs_error = abs(line_err)
        
        if abs_error > CORNER_DETECTION_THRESHOLD * 1000:  # ESP32 sends error in range -1000 to +1000
            # Corner detected - determine direction
            corner_direction = "left" if line_err < 0 else "right"
            
            # Check if it's a sharp corner
            is_sharp_corner = abs_error > SHARP_CORNER_THRESHOLD * 1000
            
            print(f"Corner detected: {corner_direction} ({'sharp' if is_sharp_corner else 'gentle'})")
            
            # Execute corner turn using the selected mode
            self._execute_corner_turn(corner_direction, line_err)
        else:
            # Normal line following
            turn_correction = self.line_pid.update(line_err)
            
            left_speed = int(LINE_FOLLOW_SPEED - turn_correction)
            right_speed = int(LINE_FOLLOW_SPEED + turn_correction)

            self.esp32.send_motor_speeds(left_speed, right_speed, left_speed, right_speed)

    def _execute_corner_turn(self, corner_direction: str, line_error: int):
        """Execute different types of corner turns based on the robot's location."""
        
        # Determine if the robot is in a special zone (pickup/dropoff area)
        # We assume these are at the beginning and end of the planned path.
        is_special_zone = False
        if self.path:
            # Check if near the start of the path (e.g., first waypoint)
            if self.current_target_index <= 0:
                is_special_zone = True
            
            # Check if near the end of the path (e.g., last two waypoints)
            if self.current_target_index >= len(self.path) - 2:
                is_special_zone = True

        if is_special_zone:
            # Use a precise turn (e.g., pivot) in special zones for maneuvering
            print("Executing PIVOT turn for precision maneuver.")
            return self._pivot_corner_turn(corner_direction, line_error)
        else:
            # Use a smooth turn for general navigation
            return self._smooth_corner_turn(corner_direction, line_error)

    def _smooth_corner_turn(self, corner_direction: str, line_error: int):
        """Normal wheel-like smooth cornering (current behavior)."""
        turn_correction = self.line_pid.update(line_error)
        
        # Gradual turn like normal wheels
        left_speed = int(CORNER_SPEED - turn_correction)
        right_speed = int(CORNER_SPEED + turn_correction)
        
        self.esp32.send_motor_speeds(left_speed, right_speed, left_speed, right_speed)
        return True

    def _sideways_corner_turn(self, corner_direction: str, line_error: int):
        """Strafe sideways through corners using omni-wheel capabilities."""
        # Determine strafe direction based on corner direction
        if corner_direction == "left":
            # Strafe left while maintaining forward motion
            fl_speed = CORNER_SPEED // 2      # Front left: reduced forward
            fr_speed = CORNER_SPEED           # Front right: full forward  
            bl_speed = CORNER_SPEED           # Back left: full forward
            br_speed = CORNER_SPEED // 2      # Back right: reduced forward
        else:  # right turn
            # Strafe right while maintaining forward motion
            fl_speed = CORNER_SPEED           # Front left: full forward
            fr_speed = CORNER_SPEED // 2      # Front right: reduced forward
            bl_speed = CORNER_SPEED // 2      # Back left: reduced forward
            br_speed = CORNER_SPEED           # Back right: full forward
        
        self.esp32.send_motor_speeds(fl_speed, fr_speed, bl_speed, br_speed)
        return True

    def _pivot_corner_turn(self, corner_direction: str, line_error: int):
        """Turn in place like a tank - pure rotation."""
        if corner_direction == "left":
            # Rotate counter-clockwise (left turn)
            fl_speed = -TURN_SPEED    # Front left: reverse
            fr_speed = TURN_SPEED     # Front right: forward
            bl_speed = -TURN_SPEED    # Back left: reverse  
            br_speed = TURN_SPEED     # Back right: forward
        else:  # right turn
            # Rotate clockwise (right turn)
            fl_speed = TURN_SPEED     # Front left: forward
            fr_speed = -TURN_SPEED    # Front right: reverse
            bl_speed = TURN_SPEED     # Back left: forward
            br_speed = -TURN_SPEED    # Back right: reverse
        
        self.esp32.send_motor_speeds(fl_speed, fr_speed, bl_speed, br_speed)
        return True

    def _front_turn_corner(self, corner_direction: str, line_error: int):
        """Turn primarily using front wheels like a front-wheel-drive car."""
        base_speed = CORNER_SPEED
        
        if corner_direction == "left":
            # Front wheels turn left, back wheels follow
            fl_speed = base_speed // 2        # Front left: slower
            fr_speed = base_speed             # Front right: normal
            bl_speed = base_speed * 3 // 4    # Back left: moderate
            br_speed = base_speed             # Back right: normal
        else:  # right turn
            # Front wheels turn right, back wheels follow  
            fl_speed = base_speed             # Front left: normal
            fr_speed = base_speed // 2        # Front right: slower
            bl_speed = base_speed             # Back left: normal
            br_speed = base_speed * 3 // 4    # Back right: moderate
        
        self.esp32.send_motor_speeds(fl_speed, fr_speed, bl_speed, br_speed)
        return True

    def _recover_line(self):
        """Basic line recovery: stop for now."""
        self._stop_motors()

    def _stop_motors(self):
        """Stop all motors."""
        self.esp32.send_motor_speeds(0, 0, 0, 0)
    
    def stop(self):
        """Stop the robot and clean up resources."""
        print("Stopping robot...")
        self.running = False
        self._stop_motors()
        self.esp32.stop()


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
    """Main entry point for the robot controller."""
    print_feature_status()
    
    robot = RobotController()
    
    app = Flask(__name__, template_folder='../../templates', static_folder='../../static')
    
    @app.route('/')
    def index():
        return render_template('navigation.html')

    @app.route('/api/robot_data')
    def robot_data():
        """Provide robot data to the web UI."""
        line_pos, line_err, sensor_vals = robot.esp32.get_line_sensor_data()
        x, y, heading_rad = robot.position_tracker.get_pose()
        heading_deg = math.degrees(heading_rad)
        motor_speeds = robot.esp32.latest_encoder_data
        
        data = {
            'state': robot.state,
            'x': x,
            'y': y,
            'heading': heading_deg,
            'line_position': line_pos,
            'line_error': line_err,
            'line_sensors': sensor_vals,
            'motors': {
                'fl': motor_speeds[0], 'fr': motor_speeds[1],
                'bl': motor_speeds[2], 'br': motor_speeds[3],
            },
            'path': robot.path,
            'current_target_index': robot.current_target_index,
            'camera_image': None # Camera disabled
        }
        return jsonify(data)

    @app.route('/video_feed')
    def video_feed():
        """Video streaming route. Returns a placeholder since camera is disabled."""
        if not FEATURES['VISION_SYSTEM_ENABLED']:
            return Response(status=204) # No content

        def generate_frames():
            while robot.running:
                with robot.frame_lock:
                    if robot.processed_frame is None:
                        time.sleep(0.1)
                        continue
                    frame = robot.processed_frame.copy()
                
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.05) # Limit frame rate

        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @app.route('/start_mission')
    def start_mission():
        """Manually start the mission and pathfinding."""
        robot._start_mission()
        robot._run_state_machine()  # Run one cycle to trigger planning
        return jsonify({
            'status': 'Mission started',
            'robot_state': robot.state,
            'path_length': len(robot.path)
        })

    @app.route('/grid_feed')
    def grid_feed():
        """Streams the grid map visualization."""
        def generate():
            while robot.running:
                robot_cell = robot.position_tracker.get_current_cell()
                path = robot.path  # Use the main path attribute
                
                grid_img = generate_grid_image(
                    pathfinder=robot.pathfinder,
                    robot_cell=robot_cell,
                    path=path,
                    start_cell=START_CELL,
                    end_cell=END_CELL
                )
                
                _, buffer = cv2.imencode('.jpg', grid_img)
                frame = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
                time.sleep(0.1)
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def generate_grid_image(pathfinder, robot_cell, path, start_cell, end_cell):
        """Generates the grid image for the web UI."""
        # Debug: Print path info
        if path and len(path) > 0:
            print(f"DEBUG: Rendering path with {len(path)} waypoints")
        else:
            print("DEBUG: No path to render")
            
        grid = np.array(pathfinder.get_grid())
        cell_size = 20
        height, width = grid.shape
        # Image is created with (height, width) but cv2 functions use (x, y) coordinates
        grid_img = np.zeros((height * cell_size, width * cell_size, 3), dtype=np.uint8)

        for r in range(height):
            for c in range(width):
                # grid is indexed by (row, col) which is (y, x)
                # Obstacles (1) are white, paths (0) are black
                color = (255, 255, 255) if grid[r, c] == 1 else (0, 0, 0)
                # cv2.rectangle uses (x, y) for its points
                cv2.rectangle(grid_img, (c * cell_size, r * cell_size), 
                              ((c + 1) * cell_size, (r + 1) * cell_size), color, -1)
        
        if path:
            for i in range(len(path) - 1):
                # Path cells are (x, y)
                # cv2.line expects points as (x, y)
                p1_x = path[i][0] * cell_size + cell_size // 2
                p1_y = path[i][1] * cell_size + cell_size // 2
                p2_x = path[i+1][0] * cell_size + cell_size // 2
                p2_y = path[i+1][1] * cell_size + cell_size // 2
                cv2.line(grid_img, (p1_x, p1_y), (p2_x, p2_y), (128, 0, 128), 2)

        # Start cell is (x, y), draw it in green
        start_color = (0, 255, 0)
        start_x, start_y = start_cell[0], start_cell[1]
        cv2.rectangle(grid_img, (start_x * cell_size, start_y * cell_size),
                      ((start_x + 1) * cell_size, (start_y + 1) * cell_size), start_color, -1)
        
        # End cell is (x, y), draw it in red (but path is also red, let's use blue)
        end_color = (0, 0, 255)
        end_x, end_y = end_cell[0], end_cell[1]
        cv2.rectangle(grid_img, (end_x * cell_size, end_y * cell_size),
                      ((end_x + 1) * cell_size, (end_y + 1) * cell_size), end_color, -1)

        if robot_cell:
            # Robot cell is (x, y), draw it as an orange circle
            robot_x, robot_y = robot_cell[0], robot_cell[1]
            cv2.circle(grid_img, 
                       (robot_x * cell_size + cell_size // 2, robot_y * cell_size + cell_size // 2), 
                       cell_size // 3, (255, 165, 0), -1)
        
        return grid_img
    
    # Start the camera capture thread if vision is enabled
    if FEATURES['VISION_SYSTEM_ENABLED']:
        def camera_capture_thread(robot_controller):
            # Try different camera indices in case the webcam is not at index 0
            cap = None
            for cam_index in [WEBCAM_INDEX, 0, 1, 2]:
                print(f"Trying to connect to camera at index {cam_index}...")
                cap = cv2.VideoCapture(cam_index)
                if cap.isOpened():
                    print(f"Successfully connected to camera at index {cam_index}")
                    break
                cap.release()
                cap = None
            
            if cap is None:
                print("ERROR: Could not connect to any camera")
                return

            # Configure camera settings for optimal performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
            
            # Verify actual camera settings
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Camera configured: {actual_width}x{actual_height} @ {actual_fps} FPS")

            print("Camera connected successfully.")
            while robot_controller.running:
                ret, frame = cap.read()
                if ret:
                    # For processing, we might want to use a smaller resolution to improve performance
                    # You can adjust this based on your processing needs
                    processing_width, processing_height = 640, 480
                    resized_frame = cv2.resize(frame, (processing_width, processing_height))
                    with robot_controller.frame_lock:
                        robot_controller.frame = resized_frame
                else:
                    print("Warning: Failed to read frame from camera. Retrying...")
                    time.sleep(1)
            cap.release()

        camera_thread = threading.Thread(target=camera_capture_thread, args=(robot,), daemon=True)
        camera_thread.start()

    # Start the vision processing thread
    if FEATURES['VISION_SYSTEM_ENABLED']:
        def vision_processing_thread(robot_controller):
            while robot_controller.running:
                robot_controller._process_vision()
                time.sleep(0.1) # Process at 10Hz

        vision_thread = threading.Thread(target=vision_processing_thread, args=(robot,), daemon=True)
        vision_thread.start()

    # Start the robot controller in a separate thread
    robot_thread = threading.Thread(target=robot.run, daemon=True)
    robot_thread.start()
    
    # Run Flask app
    print("Starting Flask web server...")
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting program.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logging.exception("Error details:") 
