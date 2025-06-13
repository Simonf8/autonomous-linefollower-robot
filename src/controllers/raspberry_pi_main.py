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
# Code cleaned up: VisualOdometry, DStarLite, and LineBasedMapper removed.
# The robot will now operate as a pure line follower with obstacle detection,
# without camera-based position tracking and navigation.
# ============================================================================

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

class SimpleLineFollower:
    """Pattern-based line following with object detection"""
    
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
        """Main control loop for line following and obstacle detection."""
        logging.info("Starting pattern-based line follower")
        
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
                # Get camera frame for object detection
                frame = None
                if self.camera:
                    ret, frame = self.camera.read()
                
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
        """Single control loop using sensor patterns with object detection."""
        
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
        
        # Priority 3: Normal line following
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
            return render_template('index.html')
        
        @self.app.route('/api/robot_data')
        def get_robot_data():
            """Get current robot data as JSON"""
            data = {
                'state': self.robot.state,
                'sensors': self.robot.esp32.sensors
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
    
    print("\n" + "="*60)
    print("Robot now running in simple line-following mode.")
    print("="*60)
    
    print("╔════════════════════════════════════════════════════════════╗")
    print("║            CYBERPUNK ROBOT NAVIGATION MATRIX              ║")
    print("╠════════════════════════════════════════════════════════════╣")
    print("║ > Simple line-following with IR sensors                   ║")
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