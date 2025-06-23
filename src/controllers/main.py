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
from scipy.interpolate import splprep, splev
from PIL import Image

# Import our clean modules
from object_detection import ObjectDetector, PathShapeDetector, LineObstacleDetector
from pathfinder import Pathfinder
from box import BoxHandler
from pid import PIDController
from intersection_detector import IntersectionDetector
from camera_line_follower import CameraLineFollower

# Import the new Robot class and configuration
from robot import Robot

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
    'PERFORMANCE_LOGGING_ENABLED': False,    # Disabled for cleaner output
    'DEBUG_VISUALIZATION_ENABLED': True,    # Enable debug visualization for web interface
    'SMOOTH_CORNERING_ENABLED': True,       # Enable smooth cornering like normal wheels
    'ADAPTIVE_SPEED_ENABLED': True,         # Enable speed adaptation based on conditions
}

# ================================
# ROBOT CONFIGURATION
# ================================
ESP32_IP = "192.168.2.38"
CELL_SIZE_M = 0.11
MAX_SPEED = 80
BASE_SPEED = 75
TURN_SPEED = 50
CORNER_SPEED = 45

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
    
    robot = Robot(CONFIG)
    app = Flask(__name__, template_folder='../../templates', static_folder='../../static')
    
    @app.route('/')
    def index():
        return render_template('navigation.html')

    @app.route('/api/robot_data')
    def robot_data():
        """Provide robot data to the web UI from the robot object."""
        # This route now pulls data directly from the robot's components
        line_result = robot.perception.latest_line_result
        movement_mode = 'UNKNOWN'
        
        if line_result:
            if line_result['line_detected']:
                offset = line_result['line_offset']
                abs_offset = abs(offset)
                is_intersection = line_result.get('intersection_detected', False)
                if is_intersection or abs_offset > 0.4: movement_mode = "TURNING"
                elif abs_offset > 0.15: movement_mode = "STRAFE+FWD" 
                elif abs_offset > 0.03: movement_mode = "SIDEWAYS"
                else: movement_mode = "STRAIGHT"
            else:
                movement_mode = "STOPPED"

        data = {
            'esp32_connected': robot.esp32.connected,
            'state': robot.state,
            'x': robot.current_cell[0] * CONFIG['CELL_SIZE_M'],
            'y': robot.current_cell[1] * CONFIG['CELL_SIZE_M'],
            'heading': math.degrees(robot.estimated_heading),
            'path': robot.navigator.path,
            'smoothed_path': robot.navigator.smoothed_path.tolist() if robot.navigator.smoothed_path is not None else [],
            'current_target_index': robot.navigator.current_target_index,
            'movement_mode': movement_mode
        }
        return jsonify(data)

    @app.route('/camera_debug_feed')
    def camera_debug_feed():
        if not CONFIG['FEATURES']['VISION_SYSTEM_ENABLED']:
            return Response(status=204)

        def generate_debug_frames():
            while robot.running:
                time.sleep(1.0 / 15) # 15 FPS
                with robot.frame_lock:
                    if robot.perception.debug_frame is None:
                        continue
                    frame = robot.perception.debug_frame.copy()
                
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        return Response(generate_debug_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @app.route('/start_mission')
    def start_mission():
        robot.start_mission()
        return jsonify({'status': 'Mission started', 'robot_state': robot.state})

    @app.route('/grid_feed')
    def grid_feed():
        def generate():
            while robot.running:
                grid_array = generate_grid_image(
                    robot.navigator.pathfinder, 
                    robot.current_cell, 
                    robot.navigator.path, 
                    CONFIG['START_CELL'], 
                    CONFIG['END_CELL'],
                    robot.navigator.smoothed_path
                )
                img_io = io.BytesIO()
                Image.fromarray(cv2.cvtColor(grid_array, cv2.COLOR_BGR2RGB)).save(img_io, 'PNG')
                img_io.seek(0)
                yield (b'--frame\r\n'
                       b'Content-Type: image/png\r\n\r\n' + img_io.read() + b'\r\n')
                time.sleep(0.1)
        
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def generate_grid_image(pathfinder, robot_cell, path, start_cell, end_cell, smoothed_path):
        grid = np.array(pathfinder.get_grid())
        cell_size = 20
        h, w = grid.shape
        img = np.zeros((h * cell_size, w * cell_size, 3), dtype=np.uint8)

        # Draw grid lines and obstacles
        for r in range(h):
            for c in range(w):
                color = (255, 255, 255) if grid[r, c] == 1 else (0, 0, 0)
                cv2.rectangle(img, (c*cell_size, r*cell_size), ((c+1)*cell_size, (r+1)*cell_size), color, -1)
        
        if path:
            for i in range(len(path) - 1):
                p1 = (path[i][0]*cell_size + cell_size//2, path[i][1]*cell_size + cell_size//2)
                p2 = (path[i+1][0]*cell_size + cell_size//2, path[i+1][1]*cell_size + cell_size//2)
                cv2.line(img, p1, p2, (128, 0, 128), 2)
        
        if smoothed_path is not None:
            path_pixels = (smoothed_path / CONFIG['CELL_SIZE_M'] * cell_size).astype(int)
            cv2.polylines(img, [path_pixels], isClosed=False, color=(255, 100, 0), thickness=1)

        # Draw start, end, and robot
        cv2.rectangle(img, (start_cell[0]*cell_size, start_cell[1]*cell_size), ((start_cell[0]+1)*cell_size, (start_cell[1]+1)*cell_size), (0,255,0), -1)
        cv2.rectangle(img, (end_cell[0]*cell_size, end_cell[1]*cell_size), ((end_cell[0]+1)*cell_size, (end_cell[1]+1)*cell_size), (0,0,255), -1)
        if robot_cell:
            cv2.circle(img, (robot_cell[0]*cell_size + cell_size//2, robot_cell[1]*cell_size + cell_size//2), cell_size//3, (255,165,0), -1)
        
        return img
    
    def camera_capture_thread(robot_instance):
        cap = None
        # Simplified camera connection logic from before
        sources = [
            f"http://{CONFIG['PHONE_IP']}:8080/video",
            CONFIG['WEBCAM_INDEX']
        ]
        for source in sources:
            try:
                cap = cv2.VideoCapture(source)
                if cap.isOpened():
                    print(f"✓ Camera connected successfully at {source}")
                    break
            except Exception:
                continue
        
        if cap is None or not cap.isOpened():
            print("ERROR: No camera source available. Vision system disabled.")
            robot_instance.config['FEATURES']['VISION_SYSTEM_ENABLED'] = False
            return

        while robot_instance.running:
            ret, frame = cap.read()
            if ret and frame is not None:
                resized_frame = cv2.resize(frame, (CONFIG['CAMERA_WIDTH'], CONFIG['CAMERA_HEIGHT']))
                with robot_instance.frame_lock:
                    robot_instance.frame = resized_frame
            else:
                time.sleep(0.5)
        cap.release()

    # Start threads
    robot_thread = threading.Thread(target=robot.run_main_loop, daemon=True)
    robot_thread.start()
    
    if CONFIG['FEATURES']['VISION_SYSTEM_ENABLED']:
        camera_thread = threading.Thread(target=camera_capture_thread, args=(robot,), daemon=True)
        camera_thread.start()
    
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