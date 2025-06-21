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

# ================================
# FEATURE CONFIGURATION
# ================================
# Enable/disable features for easy testing and debugging
FEATURES = {
    'OBJECT_DETECTION_ENABLED': False,      # Enable YOLO object detection - DISABLED for performance
    'PATH_SHAPE_DETECTION_ENABLED': False,   # Enable path shape analysis
    'OBSTACLE_AVOIDANCE_ENABLED': False,     # Enable obstacle avoidance behavior
    'VISION_SYSTEM_ENABLED': False,          # Enable camera and vision processing - DISABLED for ESP32 line following
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
ESP32_IP = "192.168.2.36"
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

# Vision configuration (placeholders, not used for line following)
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
        self.latest_sensor_values = [0, 0, 0] # Using 3 middle sensors
        
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
            parts = data_string.split(',')
            
            if parts[0] == "ENCODERS" and len(parts) == 5:
                self.latest_encoder_data = [int(p) for p in parts[1:]]
            elif parts[0] == "LINE" and len(parts) == 5:
                self.latest_line_position = int(parts[1])
                self.latest_line_error = int(parts[2])
                self.latest_sensor_values = [int(p) for p in parts[3:]]
            elif FEATURES['PERFORMANCE_LOGGING_ENABLED']:
                print(f"Received from ESP32: {data_string}")
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
        self.frame = None
        self.detections = {}
        self.path_shape = "straight"
        
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
        
        self.pathfinder = Pathfinder()
        self.path = []
        self.current_target_index = 0
        
        self.line_pid = LineFollowPID(kp=0.5, ki=0.01, kd=0.1, setpoint=0)
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
            print("CRITICAL: ESP32 not connected. Robot cannot start.")
            return

        self.state = "line_following"

        while self.running:
            start_time = time.time()
            
            # Update odometry
            encoder_ticks = self.esp32.get_encoder_ticks()
            self.position_tracker.update(encoder_ticks)
            
            self._run_state_machine()

            if FEATURES['PERFORMANCE_LOGGING_ENABLED']:
                cycle_time = (time.time() - start_time) * 1000
                if cycle_time > 2: # Log only if loop time is significant
                    print(f"Loop time: {cycle_time:.2f} ms")
            
            time.sleep(0.01)
        
        self.stop()
        
    def _run_state_machine(self):
        """Run the robot's state machine."""
        if self.state == "idle":
            self._stop_motors()
        elif self.state == "line_following":
            if FEATURES['USE_ESP32_LINE_SENSOR']:
                self._follow_line_with_sensor()
            else:
                print("Error: Line following with camera is disabled.")
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

        turn_correction = self.line_pid.update(line_err)
        
        left_speed = int(LINE_FOLLOW_SPEED - turn_correction)
        right_speed = int(LINE_FOLLOW_SPEED + turn_correction)

        self.esp32.send_motor_speeds(left_speed, right_speed, left_speed, right_speed)

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
    
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return render_template('navigation.html')

    @app.route('/api/robot_data')
    def robot_data():
        """Provide robot data to the web UI."""
        line_pos, line_err, sensor_vals = robot.esp32.get_line_sensor_data()
        x, y, heading_deg = robot.position_tracker.get_state_for_web()
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
        return Response(status=204) # No content

    @app.route('/grid_feed')
    def grid_feed():
        """Streams the grid map visualization."""
        def generate():
            while robot.running:
                robot_cell = robot.position_tracker.get_current_cell()
                path = robot.pathfinder.last_path_nodes
                
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
            
                time.sleep(0.5)
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def generate_grid_image(pathfinder, robot_cell, path, start_cell, end_cell):
        """Generates the grid image for the web UI."""
        grid = pathfinder.get_grid()
        cell_size = 20
        height, width = grid.shape
        grid_img = np.zeros((height * cell_size, width * cell_size, 3), dtype=np.uint8)

        for r in range(height):
            for c in range(width):
                color = (255, 255, 255) if grid[r, c] == 0 else (100, 100, 100)
                cv2.rectangle(grid_img, (c * cell_size, r * cell_size), 
                              ((c + 1) * cell_size, (r + 1) * cell_size), color, -1)
        
        if path:
            for i in range(len(path) - 1):
                p1 = (path[i][1] * cell_size + cell_size // 2, path[i][0] * cell_size + cell_size // 2)
                p2 = (path[i+1][1] * cell_size + cell_size // 2, path[i+1][0] * cell_size + cell_size // 2)
                cv2.line(grid_img, p1, p2, (255, 0, 0), 2)

        start_color = (0, 255, 0)
        cv2.rectangle(grid_img, (start_cell[1] * cell_size, start_cell[0] * cell_size),
                      ((start_cell[1] + 1) * cell_size, (start_cell[0] + 1) * cell_size), start_color, -1)
        
        end_color = (0, 0, 255)
        cv2.rectangle(grid_img, (end_cell[1] * cell_size, end_cell[0] * cell_size),
                      ((end_cell[1] + 1) * cell_size, (end_cell[0] + 1) * cell_size), end_color, -1)

        if robot_cell:
            cv2.circle(grid_img, 
                       (robot_cell[1] * cell_size + cell_size // 2, robot_cell[0] * cell_size + cell_size // 2), 
                       cell_size // 3, (255, 165, 0), -1)
        
        return grid_img
    
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