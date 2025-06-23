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
import io

# Import our clean modules
from object_detection import ObjectDetector, PathShapeDetector, LineObstacleDetector
from pathfinder import Pathfinder
from box import BoxHandler
from pid import PIDController
from intersection_detector import IntersectionDetector
from camera_line_follower import CameraLineFollower

# ================================
# FEATURE CONFIGURATION
# ================================
# Enable/disable features for easy testing and debugging
FEATURES = {
    'OBJECT_DETECTION_ENABLED': False,      # Enable YOLO object detection - DISABLED for performance
    'PATH_SHAPE_DETECTION_ENABLED': False,   # Enable path shape analysis
    'OBSTACLE_AVOIDANCE_ENABLED': True,     # Enable obstacle avoidance behavior
    'VISION_SYSTEM_ENABLED': True,          # Enable camera and vision processing
    'INTERSECTION_CORRECTION_ENABLED': True,# Enable intersection-based position correction
    'USE_ESP32_LINE_SENSOR': True,          # Use ESP32 hardware sensor for line following
    'POSITION_CORRECTION_ENABLED': True,    # Enable waypoint position corrections
    'PERFORMANCE_LOGGING_ENABLED': True,    # Enable detailed performance logging
    'DEBUG_VISUALIZATION_ENABLED': True,    # Enable debug visualization for web interface
    'SMOOTH_CORNERING_ENABLED': True,       # Enable smooth cornering like normal wheels
    'ADAPTIVE_SPEED_ENABLED': True,         # Enable speed adaptation based on conditions
}

# ================================
# ROBOT CONFIGURATION
# ================================
ESP32_IP = "192.168.2.38"
CELL_SIZE_M = 0.11
BASE_SPEED = 60
TURN_SPEED = 50
CORNER_SPEED = 55  

# Robot physical constants (camera-based navigation only)
# REMOVED: Encoder constants since we're not using encoders
# PULSES_PER_REV = 960
# WHEEL_DIAMETER_M = 0.025
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
    'SMOOTH': 'smooth',           # Normal wheel-like smooth cornering (current)
    'SIDEWAYS': 'sideways',       # Strafe sideways through corners
    'PIVOT': 'pivot',             # Turn in place like a tank
    'FRONT_TURN': 'front_turn'    # Turn using front wheels primarily
}

# Corner detection thresholds
CORNER_DETECTION_THRESHOLD = 0.35    # Line offset to detect corner
CORNER_TURN_DURATION = 30            # Frames to execute corner turn
SHARP_CORNER_THRESHOLD = 0.6         # Threshold for sharp vs gentle corners

# Vision configuration (placeholders, not used for line following)
IMG_PATH_SRC_PTS = np.float32([[200, 300], [440, 300], [580, 480], [60, 480]])
IMG_PATH_DST_PTS = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])

# Camera configuration
PHONE_IP = "192.168.2.6" # CHANGE THIS to your phone's camera stream IP
WEBCAM_INDEX = 0  # USB webcam device index (usually 0 for first webcam)
CAMERA_WIDTH, CAMERA_HEIGHT = 416, 320

class ESP32Bridge:
    """ESP32 communication bridge for motors only (camera-based navigation)."""
    
    def __init__(self, ip: str, port: int = 1234):
        self.ip = ip
        self.port = port
        self.socket = None
        self.connected = False
        self.connection_attempts = 0
        
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
    
    def send_stop_command(self):
        """Send stop command to ESP32."""
        if not self.connected and not self.connect():
            return False
            
        try:
            return self._send_command("STOP")
        except Exception as e:
            print(f"Error sending stop command: {e}")
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
            

    
    def _receive_data(self):
        """Try to receive acknowledgment from ESP32 (non-blocking)."""
        if not self.socket or not self.connected:
            return
            
        try:
            data = self.socket.recv(1024)
            if data:
                response = data.decode().strip()
                if FEATURES['PERFORMANCE_LOGGING_ENABLED']:
                    print(f"ESP32 response: {response}")
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
        self.line_obstacle_detector = None
        self.frame = None
        self.processed_frame = None
        self.frame_lock = threading.Lock()
        
        # Camera-based line following
        self.camera_line_follower = CameraLineFollower(debug=FEATURES['DEBUG_VISUALIZATION_ENABLED'])
        self.latest_line_result = None
        self.debug_frame = None  # Store debug frames from camera line follower
        
        self.detections = {}
        self.path_shape = "straight"
        self.last_intersection_time = 0
        
        # SIMPLIFIED: Camera-only position tracking
        # No encoder-based odometry, using basic cell-based position tracking
        self.current_cell = START_CELL
        self.estimated_heading = START_HEADING
        self.position_tracker = None  # Will be camera-based only
        
        # Pathfinder setup (still useful for path planning)
        maze_grid = Pathfinder.create_maze_grid()
        self.pathfinder = Pathfinder(grid=maze_grid, cell_size_m=CELL_SIZE_M)
        
        self.path = []
        self.current_target_index = 0
        
        self.line_pid = PIDController(kp=0.5, ki=0.01, kd=0.1, output_limits=(-100, 100))
        self.state = "idle"
        self.recovery_state = "idle"  # "idle", "start", "swing_left", "swing_right", "reverse"
        self.recovery_start_time = 0
        
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
        if FEATURES['OBSTACLE_AVOIDANCE_ENABLED']:
            self.line_obstacle_detector = LineObstacleDetector(debug=FEATURES['DEBUG_VISUALIZATION_ENABLED'])
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
            
            # REMOVED: Encoder-based position tracking
            # Now relying solely on camera for navigation and position tracking
            # encoder_ticks = self.esp32.get_encoder_ticks()
            # if encoder_ticks:
            #     self.position_tracker.update(encoder_ticks)
            
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
        # SIMPLIFIED: Set current cell instead of precise odometry pose
        self.current_cell = START_CELL
        self.estimated_heading = START_HEADING
        self.state = "planning"

    def _process_vision(self):
        """Process the latest camera frame for events."""
        if not FEATURES['VISION_SYSTEM_ENABLED'] or self.frame is None:
            return

        with self.frame_lock:
            frame_copy = self.frame.copy()

        processed_frame = frame_copy
        
        # --- Obstacle Detection ---
        if FEATURES['OBSTACLE_AVOIDANCE_ENABLED'] and self.line_obstacle_detector and self.state == "path_following":
            obstacle = self.line_obstacle_detector.detect(frame_copy)
            if obstacle:
                print("Line is blocked! Initiating avoidance maneuver.")
                # SIMPLIFIED: Estimate obstacle position based on current cell and heading
                obstacle_cell_x = self.current_cell[0] + int(math.cos(self.estimated_heading))
                obstacle_cell_y = self.current_cell[1] + int(math.sin(self.estimated_heading))
                obstacle_cell = (obstacle_cell_x, obstacle_cell_y)
                
                # Update the grid and trigger replanning
                self.pathfinder.update_obstacle(obstacle_cell[0], obstacle_cell[1], is_obstacle=True)
                self.state = "replanning"
                return # Stop further vision processing this frame

        # --- Intersection Detection ---
        if FEATURES['INTERSECTION_CORRECTION_ENABLED'] and self.intersection_detector:
            # Only check for intersection every so often to avoid multiple triggers
            if time.time() - self.last_intersection_time > 3.0: # 3 second cooldown
                intersection_type = self.intersection_detector.detect(frame_copy)
                if intersection_type:
                    print(f"Intersection detected! Type: {intersection_type}. Using camera-based position correction.")
                    # SIMPLIFIED: Camera-based position correction logic would go here
                    # For now, we just log the detection
                    self.last_intersection_time = time.time()
                
                if FEATURES['DEBUG_VISUALIZATION_ENABLED']:
                    processed_frame = self.intersection_detector.draw_debug_info(processed_frame, intersection_type)

        # Store the processed frame for the video feed
        with self.frame_lock:
            self.processed_frame = processed_frame

    def _plan_path_to_target(self):
        """Plan the path to the target cell."""
        current_cell = self.current_cell  # SIMPLIFIED: Use camera-based current cell
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
        
        # Camera-based waypoint detection using line sensor patterns
        target_cell = self.path[self.current_target_index]
        
        # Check if we've reached an intersection (simplified logic)
        if self._detect_intersection_or_waypoint():
            print(f"Waypoint detected! Advancing from {self.current_cell} to {target_cell}")
            self.current_cell = target_cell
            self.current_target_index += 1
            
            # Update estimated heading based on path direction
            if self.current_target_index < len(self.path):
                next_target = self.path[self.current_target_index]
                dx = next_target[0] - self.current_cell[0]
                dy = next_target[1] - self.current_cell[1]
                self.estimated_heading = math.atan2(dy, dx)
        
        self._follow_line_with_sensor()
    
    def _detect_intersection_or_waypoint(self) -> bool:
        """
        Detect if robot has reached an intersection or waypoint.
        Uses camera-based line detection to identify intersections.
        Returns True if intersection/waypoint detected.
        """
        if not FEATURES['VISION_SYSTEM_ENABLED'] or self.frame is None:
            return False
        
        with self.frame_lock:
            frame_copy = self.frame.copy()
        
        # Use camera line follower to detect intersections
        line_result = self.camera_line_follower.detect_line(frame_copy)
        
        # Check if intersection is detected by camera
        if line_result.get('intersection_detected', False):
            # Add a small delay to avoid multiple triggers
            if not hasattr(self, '_last_camera_intersection_time'):
                self._last_camera_intersection_time = 0
            
            current_time = time.time()
            if current_time - self._last_camera_intersection_time > 2.0:  # 2 second cooldown
                self._last_camera_intersection_time = current_time
                return True
        
        return False

    def _run_state_machine(self):
        """Run the robot's state machine."""
        if self.state == "idle":
            self._stop_motors()
        elif self.state == "planning":
            self._plan_path_to_target()
        elif self.state == "path_following":
            self._follow_path()
        elif self.state == "replanning":
            print("State: Replanning due to obstacle.")
            self._stop_motors()
            self._plan_path_to_target() # Re-run planning with the updated grid
        elif self.state == "recovering_line":
            self._execute_line_recovery()
        elif self.state == "mission_complete":
            self._stop_motors()
        elif self.state == "error":
            self._stop_motors()
            self.running = False
            
    def _follow_line_with_sensor(self):
        """Follow the line using camera-based vision instead of ESP32 sensors."""
        if not FEATURES['VISION_SYSTEM_ENABLED'] or self.frame is None:
            # Fallback to stopping if no camera available
            print("No camera frame available for line following")
            self._stop_motors()
            return
        
        with self.frame_lock:
            frame_copy = self.frame.copy()
        
        # Use camera line follower to detect line
        line_result = self.camera_line_follower.detect_line(frame_copy)
        self.latest_line_result = line_result
        
        # Store debug frame for web interface
        with self.frame_lock:
            self.debug_frame = line_result.get('processed_frame', frame_copy)
        
        # Check if line is detected
        if not line_result['line_detected']:
            print("Camera: Line lost! Initiating recovery sequence.")
            self.state = "recovering_line"
            self.recovery_state = "start"
            self._stop_motors()
            return
        
        # Check for intersection (waypoint detection)
        if line_result.get('intersection_detected', False):
            # Intersection detected - this could be a waypoint
            if not hasattr(self, '_last_camera_intersection_time'):
                self._last_camera_intersection_time = 0
            
            current_time = time.time()
            if current_time - self._last_camera_intersection_time > 2.0:  # 2 second cooldown
                self._last_camera_intersection_time = current_time
                print("Camera detected intersection - potential waypoint!")
                # Let the waypoint detection logic handle this
        
        # Get motor speeds from camera line follower
        fl, fr, bl, br = self.camera_line_follower.get_motor_speeds(line_result, LINE_FOLLOW_SPEED)
        
        # Send motor commands
        self.esp32.send_motor_speeds(fl, fr, bl, br)
        
        # Debug output
        if FEATURES['PERFORMANCE_LOGGING_ENABLED']:
            confidence = line_result.get('confidence', 0.0)
            offset = line_result.get('line_offset', 0.0)
            print(f"Camera line following: offset={offset:.3f}, confidence={confidence:.3f}, motors=({fl},{fr},{bl},{br})")

    def _execute_line_recovery(self):
        """Executes a sequence of maneuvers to find a lost line using camera."""
        # Check if line is found using camera
        if FEATURES['VISION_SYSTEM_ENABLED'] and self.frame is not None:
            with self.frame_lock:
                frame_copy = self.frame.copy()
            
            line_result = self.camera_line_follower.detect_line(frame_copy)
            
            if line_result['line_detected'] and line_result['confidence'] > 0.3:
                print("Camera: Line re-acquired! Resuming path following.")
                self.state = "path_following"
                self.recovery_state = "idle"
                return

        # Initialize the recovery timer on the first attempt.
        if self.recovery_state == "start":
            self.recovery_start_time = time.time()
            self.recovery_state = "swing_left"
            print("Camera recovery: Swinging left...")

        # If recovery takes too long, give up and enter an error state.
        if time.time() - self.recovery_start_time > 8.0:  # 8-second timeout (longer for camera)
            print("Camera line recovery failed. Could not find the line.")
            self.state = "error"
            self.recovery_state = "idle"
            self._stop_motors()
            return

        duration = time.time() - self.recovery_start_time

        # --- Recovery Maneuver Sequence ---
        # 1. Swing left for 1 second.
        if self.recovery_state == "swing_left":
            if duration < 1.0:
                self.esp32.send_motor_speeds(-TURN_SPEED, TURN_SPEED, -TURN_SPEED, TURN_SPEED)
            else:
                self.recovery_state = "swing_right"
                print("Camera recovery: Swinging right...")

        # 2. Swing right for 2 seconds (to pass the center).
        elif self.recovery_state == "swing_right":
            if duration < 3.0:  # 1s left + 2s right
                self.esp32.send_motor_speeds(TURN_SPEED, -TURN_SPEED, TURN_SPEED, -TURN_SPEED)
            else:
                self.recovery_state = "reverse"
                print("Camera recovery: Reversing...")
        
        # 3. Return to center and reverse for 1 second.
        elif self.recovery_state == "reverse":
            if duration < 4.0: # bring it back to center
                self.esp32.send_motor_speeds(-TURN_SPEED, TURN_SPEED, -TURN_SPEED, TURN_SPEED)
            elif duration < 5.5:
                speed = -LINE_FOLLOW_SPEED // 2
                self.esp32.send_motor_speeds(speed, speed, speed, speed)
            else:
                # If the line is still not found, repeat the search cycle.
                self.recovery_state = "swing_left"

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
        # SIMPLIFIED: Camera-based line data
        line_pos = -1
        line_err = 0
        sensor_vals = [0, 0, 0]
        
        # Get camera-based line data if available
        camera_status = "No Data"
        if robot.latest_line_result:
            if robot.latest_line_result['line_detected']:
                # Convert camera offset to position-like value
                offset = robot.latest_line_result['line_offset']
                line_pos = int(1000 + offset * 1000)  # Convert -1,1 to 0,2000 range
                line_err = int(offset * 1000)         # Convert to -1000,1000 range
                confidence = robot.latest_line_result['confidence']
                # Simulate sensor values based on line position and confidence
                base_val = int(confidence * 1000)
                left_val = max(0, base_val - abs(int(offset * 500)))
                center_val = int(confidence * 1000)
                right_val = max(0, base_val - abs(int(offset * 500)))
                sensor_vals = [left_val, center_val, right_val]
                
                zone = robot.latest_line_result.get('zone_used', 'unknown')
                camera_status = f"Line Detected ({zone})"
                
                # Determine movement mode for display
                abs_offset = abs(offset)
                is_intersection = robot.latest_line_result.get('intersection_detected', False)
                
                if is_intersection or abs_offset > 0.4:
                    movement_mode = "TURNING"
                elif abs_offset > 0.15:
                    movement_mode = "STRAFE+FWD" 
                elif abs_offset > 0.03:
                    movement_mode = "SIDEWAYS"
                else:
                    movement_mode = "STRAIGHT"
            else:
                camera_status = "No Line Detected"
                movement_mode = "STOPPED"
        
        # Convert current cell to world coordinates for display
        x = robot.current_cell[0] * CELL_SIZE_M
        y = robot.current_cell[1] * CELL_SIZE_M
        heading_deg = math.degrees(robot.estimated_heading)
        
        # Note: motor speeds now represent last commanded speeds, not encoder feedback
        motor_speeds = [0, 0, 0, 0]  # No encoder feedback available
        
        data = {
            'esp32_connected': robot.esp32.connected,
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
            'camera_image': None, # Camera disabled
            'position_source': 'camera_based',  # Indicate camera-based positioning
            'line_source': 'camera_vision',     # Indicate camera-based line detection
            'camera_status': camera_status,
            'camera_confidence': robot.latest_line_result.get('confidence', 0.0) if robot.latest_line_result else 0.0,
            'camera_offset': robot.latest_line_result.get('line_offset', 0.0) if robot.latest_line_result else 0.0,
            'intersection_detected': robot.latest_line_result.get('intersection_detected', False) if robot.latest_line_result else False,
            'movement_mode': movement_mode if 'movement_mode' in locals() else 'UNKNOWN'
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

    @app.route('/camera_debug_feed')
    def camera_debug_feed():
        """Camera debug feed showing line detection visualization - optimized for speed."""
        if not FEATURES['VISION_SYSTEM_ENABLED']:
            return Response(status=204) # No content

        def generate_debug_frames():
            last_frame_time = 0
            frame_skip_count = 0
            target_fps = 15  # Reduce from ~20fps to 15fps for debug feed
            frame_interval = 1.0 / target_fps
            
            while robot.running:
                current_time = time.time()
                
                # Skip frame if not enough time has passed
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.01)
                    continue
                
                with robot.frame_lock:
                    if robot.debug_frame is None:
                        time.sleep(0.1)
                        continue
                    frame = robot.debug_frame.copy()
                
                # Use faster JPEG encoding with lower quality for speed
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]  # Reduce quality for speed
                _, buffer = cv2.imencode('.jpg', frame, encode_params)
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
                last_frame_time = current_time

        return Response(generate_debug_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
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
        """Generate real-time grid visualization."""
        def generate():
            while True:
                robot_cell = robot.current_cell  # SIMPLIFIED: Use camera-based position
                grid_array = generate_grid_image(robot.pathfinder, robot_cell, robot.path, START_CELL, END_CELL)
                
                # Convert numpy array to PIL Image
                from PIL import Image
                grid_image = Image.fromarray(cv2.cvtColor(grid_array, cv2.COLOR_BGR2RGB))
                
                # Convert PIL image to bytes
                img_io = io.BytesIO()
                grid_image.save(img_io, 'PNG')
                img_io.seek(0)
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/png\r\n\r\n' + img_io.read() + b'\r\n')
                time.sleep(0.1)
        
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def generate_grid_image(pathfinder, robot_cell, path, start_cell, end_cell):
        """Generates the grid image for the web UI."""
        # Removed debug prints for performance
            
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
            cap = None
            camera_source = None
            
            # Try phone camera first
            print(f"Attempting to connect to phone camera at {PHONE_IP}:8080...")
            try:
                phone_url = f"http://{PHONE_IP}:8080/video"
                cap = cv2.VideoCapture(phone_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize delay
                
                # Test if phone camera is working
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    camera_source = "phone"
                    print(f"✓ Phone camera connected successfully at {PHONE_IP}")
                else:
                    print(f"✗ Phone camera at {PHONE_IP} not responding")
                    cap.release()
                    cap = None
            except Exception as e:
                print(f"✗ Failed to connect to phone camera: {e}")
                if cap:
                    cap.release()
                cap = None
            
            # Fall back to USB webcam if phone camera failed
            if cap is None:
                print(f"Attempting to connect to USB webcam (index {WEBCAM_INDEX})...")
                try:
                    cap = cv2.VideoCapture(WEBCAM_INDEX)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                    
                    # Test if webcam is working
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        camera_source = "webcam"
                        print(f"✓ USB webcam connected successfully")
                    else:
                        print(f"✗ USB webcam not responding")
                        cap.release()
                        cap = None
                except Exception as e:
                    print(f"✗ Failed to connect to USB webcam: {e}")
                    if cap:
                        cap.release()
                    cap = None
            
            # Exit if no camera available
            if cap is None:
                print("ERROR: No camera source available. Vision system disabled.")
                return
            
            print(f"Camera capture thread started using {camera_source}")
            frame_count = 0
            
            while robot_controller.running:
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Resize frame to standard size
                    resized_frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
                    
                    # Store frame for processing
                    with robot_controller.frame_lock:
                        robot_controller.frame = resized_frame
                    
                    frame_count += 1
                    if frame_count % 100 == 0:  # Log every 100 frames
                        print(f"Camera frames captured: {frame_count} (source: {camera_source})")
                        
                else:
                    print(f"Warning: Failed to read frame from {camera_source}. Retrying...")
                    time.sleep(0.5)
                    
                    # Try to reconnect if connection lost
                    if camera_source == "phone":
                        cap.release()
                        cap = cv2.VideoCapture(f"http://{PHONE_IP}:8080/video")
                    
            cap.release()
            print(f"Camera capture thread stopped")

        camera_thread = threading.Thread(target=camera_capture_thread, args=(robot,), daemon=True)
        camera_thread.start()

    # Start the vision processing thread
    if FEATURES['VISION_SYSTEM_ENABLED']:
        def vision_processing_thread(robot_controller):
            while robot_controller.running:
                robot_controller._process_vision()
                # Adaptive processing rate - slower when ESP32 disconnected
                if robot_controller.esp32.connected:
                    time.sleep(0.1)  # 10Hz when connected
                else:
                    time.sleep(0.2)  # 5Hz when disconnected to save CPU

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