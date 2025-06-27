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
from camera_line_follower import CameraLineFollower, CameraLineFollowingMixin
from visual_localizer import PreciseMazeLocalizer

# ================================
# FEATURE CONFIGURATION
# ================================
# Enable/disable features for easy testing and debugging
FEATURES = {
    'OBJECT_DETECTION_ENABLED': False,
    'PATH_SHAPE_DETECTION_ENABLED': False,
    'OBSTACLE_AVOIDANCE_ENABLED': False,
    'VISION_SYSTEM_ENABLED': True,
    'CAMERA_LINE_FOLLOWING_ENABLED': False, # Disabled in favor of ESP32 line sensor
    'INTERSECTION_CORRECTION_ENABLED': True,
    'USE_ESP32_LINE_SENSOR': True,       # Use ESP32 hardware sensor for line following
    'POSITION_CORRECTION_ENABLED': True,
    'PERFORMANCE_LOGGING_ENABLED': False,    # Disabled to reduce log spam
    'DEBUG_VISUALIZATION_ENABLED': True,
    'SMOOTH_CORNERING_ENABLED': True,
    'ADAPTIVE_SPEED_ENABLED': True,
}

# ================================
# ROBOT CONFIGURATION
# ================================
ESP32_IP = "192.168.2.115"  # IMPORTANT: Set this to your ESP32's IP address
CELL_SIZE_M = 0.11
BASE_SPEED = 25
TURN_SPEED = 20
CORNER_SPEED = 22

# Robot physical constants (for odometry, if used)
PULSES_PER_REV = 920
WHEEL_DIAMETER_M = 0.025
ROBOT_WIDTH_M = 0.225
ROBOT_LENGTH_M = 0.075

# Maze and Mission Configuration
MAZE_GRID = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0], # Row 0
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0], # Row 1
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Row 2
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 3
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 4
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0], # Row 5
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 6
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Row 7
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 8
    [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0], # Row 9
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 10
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 11
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Row 12
    [0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1], # Row 13
    [0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1]  # Row 14
]
START_CELL = (0, 14) # Start position (col, row)
END_CELL = (0, 2)   # End position (col, row)
START_POSITION = ((START_CELL[0] + 0.5) * CELL_SIZE_M, (START_CELL[1] + 0.5) * CELL_SIZE_M)
START_HEADING = 0.0  # Facing right for horizontal movement
START_DIRECTION = 'E' # For visual localizer

# Line following configuration
LINE_FOLLOW_SPEED = 25

# Corner turning configuration
CORNER_TURN_MODES = {
    'SMOOTH': 'smooth',
    'SIDEWAYS': 'sideways',
    'PIVOT': 'pivot',
    'FRONT_TURN': 'front_turn'
}
CORNER_DETECTION_THRESHOLD = 0.35
CORNER_TURN_DURATION = 30
SHARP_CORNER_THRESHOLD = 0.6

IMG_PATH_SRC_PTS = np.float32([[160, 240], [480, 240], [640, 480], [0, 480]])
IMG_PATH_DST_PTS = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])

# Camera configuration
WEBCAM_INDEX = 1
CAMERA_WIDTH, CAMERA_HEIGHT = 1920, 1080
CAMERA_FPS = 30

class ESP32Controller:
    """Controller for communicating with the ESP32 over WiFi."""
    def __init__(self, ip, port=1234):
        self.ip = ip
        self.port = port
        self.sock = None
        self.is_connected = False
        self.lock = threading.Lock()
        
        # Sensor data storage
        self.latest_encoder_data = [0, 0, 0, 0]
        self.latest_line_position = 1000  # Center
        self.latest_line_error = 0
        self.latest_sensor_values = [1000, 1000, 1000] # Assuming no line initially
        
        self.receiver_thread = threading.Thread(target=self._receive_data, daemon=True)

    def connect(self):
        """Establish connection with the ESP32 server."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.ip, self.port))
            self.sock.settimeout(2.0)
            self.is_connected = True
            self.receiver_thread.start()
            print(f"Successfully connected to ESP32 at {self.ip}:{self.port}")
            return True
        except (socket.error, socket.timeout) as e:
            print(f"ERROR: Could not connect to ESP32 at {self.ip}:{self.port}. Reason: {e}")
            self.sock = None
            self.is_connected = False
            return False

    def _receive_data(self):
        """Continuously receive and parse sensor data from the ESP32."""
        buffer = ""
        while self.is_connected and self.sock:
            try:
                data = self.sock.recv(128)
                if not data:
                    print("ESP32 connection lost.")
                    self.is_connected = False
                    break
                
                buffer += data.decode('utf-8')
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if not line: continue
                    
                    parts = line.split(',')
                    with self.lock:
                        if parts[0] == "ENCODERS" and len(parts) == 5:
                            self.latest_encoder_data = [int(p) for p in parts[1:]]
                        elif parts[0] == "LINE" and len(parts) >= 3:
                            self.latest_line_position = int(parts[1])
                            self.latest_line_error = int(parts[2])
                            if len(parts) > 3:
                                self.latest_sensor_values = [int(p) for p in parts[3:]]

            except (socket.timeout, ConnectionResetError):
                continue # Ignore timeouts, just means no new data
            except Exception as e:
                print(f"Error receiving data from ESP32: {e}")
                self.is_connected = False
                break
        print("ESP32 receiver thread stopped.")

    def send_motor_speeds(self, fl: int, fr: int, bl: int, br: int):
        """Send motor speeds to the ESP32."""
        if not self.is_connected or not self.sock:
            # print("Not connected to ESP32. Cannot send motor speeds.")
            return False
        try:
            command = f"{fl},{fr},{bl},{br}\n"
            self.sock.sendall(command.encode('utf-8'))
            return True
        except socket.error as e:
            print(f"Failed to send motor speeds to ESP32: {e}")
            self.is_connected = False
            return False

    def get_encoder_ticks(self) -> List[int]:
        """Get the latest encoder ticks from the ESP32."""
        with self.lock:
            return self.latest_encoder_data

    def get_line_sensor_data(self) -> Tuple[int, int, List[int]]:
        """Get the latest line sensor data from the ESP32."""
        with self.lock:
            return (self.latest_line_position, self.latest_line_error, self.latest_sensor_values)

    def is_line_detected(self) -> bool:
        """Check if a line is currently detected."""
        with self.lock:
            return self.latest_line_position != -1

    def stop(self):
        """Stop all motors and close the connection."""
        self.send_motor_speeds(0, 0, 0, 0)
        self.is_connected = False
        if self.sock:
            self.sock.close()
            self.sock = None
        print("ESP32 controller stopped.")

class RobotController(CameraLineFollowingMixin):
    """Main robot controller integrating visual localization and ESP32 control."""

    def __init__(self):
        self.running = True
        
        # Initialize motor controller (ESP32 or fallback)
        if FEATURES['USE_ESP32_LINE_SENSOR']:
            self.motor_controller = ESP32Controller(ip=ESP32_IP)
        else:
            # This part is now a fallback if ESP32 is disabled
            from main import DirectMotorController # Lazy import
            self.motor_controller = DirectMotorController()

        self.object_detector = None
        self.path_shape_detector = None
        self.intersection_detector = None
        self.frame = None
        self.processed_frame = None
        self.frame_lock = threading.Lock()

        self.detections = {}
        self.path_shape = "straight"
        self.last_intersection_time = 0

        # Position tracking using Visual Localizer
        self.position_tracker = PreciseMazeLocalizer(
            maze=MAZE_GRID,
            start_pos=START_CELL,
            start_direction=START_DIRECTION
        )

        # Pathfinder setup
        self.pathfinder = Pathfinder(grid=MAZE_GRID, cell_size_m=CELL_SIZE_M)

        self.path = []
        self.current_target_index = 0

        self.line_pid = PIDController(kp=0.5, ki=0.01, kd=0.1, output_limits=(-100, 100))
        self.state = "idle"

        if FEATURES['VISION_SYSTEM_ENABLED']:
            self._setup_vision()

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
        if not self.motor_controller.connect():
            print("WARNING: Motor controller failed to connect. Running in simulation mode.")
        
        if isinstance(self.position_tracker, PreciseMazeLocalizer):
            # Try to initialize camera with different indices
            for i in range(4): # Try first 4 indices
                if self.position_tracker.initialize_camera(i):
                    break
            self.position_tracker.start_localization()

        self._start_mission()

        while self.running:
            self._run_state_machine()
            time.sleep(0.01)
        
        self.stop()
        
    def _start_mission(self):
        """Start the defined mission."""
        if self.state != "idle":
            print("Cannot start mission, robot is not idle.")
            return

        print("Starting mission...")
        # Pose is now managed by the visual localizer
        self.state = "planning"

    def _process_vision(self):
        """Process the latest camera frame for events."""
        if not FEATURES['VISION_SYSTEM_ENABLED'] or self.frame is None:
            return

        with self.frame_lock:
            frame_copy = self.frame.copy()

        processed_frame = frame_copy
        
        # Visual localization is handled by its own thread, no need to call it here.
        # The intersection logic below can be used to augment or validate.
        
        if FEATURES['INTERSECTION_CORRECTION_ENABLED'] and self.intersection_detector:
            # This logic can be adapted to work with the visual localizer's output
            # For now, we rely on the primary localizer.
            pass # Temporarily disabled to avoid conflicts with PreciseMazeLocalizer

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
        """Follow the line using hardware sensors via ESP32."""
        if not FEATURES['USE_ESP32_LINE_SENSOR']:
            print("ESP32 line sensor is disabled in features.")
            self._stop_motors()
            return
            
        line_pos, line_err, _ = self.motor_controller.get_line_sensor_data()
        
        if line_pos == -1: # ESP32 signals -1 when line is lost
            # print("Line lost!") # This can be spammy, maybe handle recovery differently
            self._recover_line()
            return

        # Simple PID control for line following
        turn_correction = self.line_pid.update(line_err)
        
        left_speed = int(LINE_FOLLOW_SPEED - turn_correction)
        right_speed = int(LINE_FOLLOW_SPEED + turn_correction)

        # Assuming a 2-wheel drive differential steering setup for simplicity
        # For omni-wheels, this would be a simple forward movement with steering
        self.motor_controller.send_motor_speeds(left_speed, right_speed, left_speed, right_speed)

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
        
        self.motor_controller.send_motor_speeds(left_speed, right_speed, left_speed, right_speed)
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
        
        self.motor_controller.send_motor_speeds(fl_speed, fr_speed, bl_speed, br_speed)
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
        
        self.motor_controller.send_motor_speeds(fl_speed, fr_speed, bl_speed, br_speed)
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
            #
            fl_speed = base_speed             # Front left: normal
            fr_speed = base_speed // 2        # Front right: slower
            bl_speed = base_speed             # Back left: normal
            br_speed = base_speed * 3 // 4    # Back right: moderate
        
        self.motor_controller.send_motor_speeds(fl_speed, fr_speed, bl_speed, br_speed)
        return True

    def _recover_line(self):
        """Basic line recovery: stop for now."""
        self._stop_motors()

    def _stop_motors(self):
        """Stop all motors."""
        self.motor_controller.send_motor_speeds(0, 0, 0, 0)
    
    def stop(self):
        """Stop the robot and clean up resources."""
        print("Stopping robot...")
        self.running = False
        self._stop_motors()
        if isinstance(self.position_tracker, PreciseMazeLocalizer):
            self.position_tracker.stop_localization()
        self.motor_controller.stop()


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
        line_pos, line_err, sensor_vals = robot.motor_controller.get_line_sensor_data()
        x, y, heading_rad = robot.position_tracker.get_pose()
        heading_deg = math.degrees(heading_rad)
        
        # Get motor speeds if they are available from the controller
        motor_speeds = [0,0,0,0]
        if hasattr(robot.motor_controller, 'latest_encoder_data'):
            motor_speeds = robot.motor_controller.latest_encoder_data
        
        # Get camera line following status if available
        camera_line_status = {}
        if hasattr(robot, 'get_camera_line_status'):
            camera_line_status = robot.get_camera_line_status()
        
        data = {
            'state': robot.state,
            'x': x,
            'y': y,
            'heading': heading_deg,
            'line_position': line_pos,
            'line_error': line_err,
            'line_sensors': sensor_vals,
            'camera_line_following': camera_line_status,
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
                frame = None
                if isinstance(robot.position_tracker, PreciseMazeLocalizer):
                    frame = robot.position_tracker.get_camera_frame()

                if frame is None:
                    # If no frame, send a placeholder or just wait
                    time.sleep(0.1)
                    continue
                
                # Add camera line following debug overlay if available
                if (hasattr(robot, 'camera_line_result') and 
                    robot.camera_line_result.get('processed_frame') is not None):
                    frame = robot.camera_line_result['processed_frame']
                
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
                path = robot.path
                
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
        grid = np.array(pathfinder.get_grid())
        cell_size = 20
        height, width = grid.shape
        grid_img = np.zeros((height * cell_size, width * cell_size, 3), dtype=np.uint8)

        for r in range(height):
            for c in range(width):
                color = (255, 255, 255) if grid[r, c] == 1 else (0, 0, 0)
                cv2.rectangle(grid_img, (c * cell_size, r * cell_size), 
                              ((c + 1) * cell_size, (r + 1) * cell_size), color, -1)
        
        if path:
            for i in range(len(path) - 1):
                p1_x = path[i][0] * cell_size + cell_size // 2
                p1_y = path[i][1] * cell_size + cell_size // 2
                p2_x = path[i+1][0] * cell_size + cell_size // 2
                p2_y = path[i+1][1] * cell_size + cell_size // 2
                # Use blue for path to distinguish from red end cell
                cv2.line(grid_img, (p1_x, p1_y), (p2_x, p2_y), (255, 0, 0), 2)

        start_color = (0, 255, 0)
        start_x, start_y = start_cell[0], start_cell[1]
        cv2.rectangle(grid_img, (start_x * cell_size, start_y * cell_size),
                      ((start_x + 1) * cell_size, (start_y + 1) * cell_size), start_color, -1)
        
        end_color = (0, 0, 255)
        end_x, end_y = end_cell[0], end_cell[1]
        cv2.rectangle(grid_img, (end_x * cell_size, end_y * cell_size),
                      ((end_x + 1) * cell_size, (end_y + 1) * cell_size), end_color, -1)

        if robot_cell:
            robot_x, robot_y = robot_cell[0], robot_cell[1]
            cv2.circle(grid_img, 
                       (robot_x * cell_size + cell_size // 2, robot_y * cell_size + cell_size // 2), 
                       cell_size // 3, (255, 165, 0), -1)
        
        return grid_img
    
    # Start the camera capture thread if vision is enabled
    if FEATURES['VISION_SYSTEM_ENABLED']:
        print("Vision system enabled. Camera is managed by PreciseMazeLocalizer.")

    # Start the Flask web server in a separate thread
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False), daemon=True)
    flask_thread.start()

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
