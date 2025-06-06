#!/usr/bin/env python3

import cv2
import numpy as np
import socket
import time
import logging
import threading
import json
from collections import deque
from flask import Flask, Response, render_template_string, jsonify
import torch
from ultralytics import YOLO
from PIL import Image

# -----------------------------------------------------------------------------
# --- CONFIGURATION FOR ENHANCED LINE FOLLOWING WITH YOLO ---
# -----------------------------------------------------------------------------
ESP32_IP = '192.168.53.117'  # Change this to your ESP32's IP address
ESP32_PORT = 1234
CAMERA_WIDTH, CAMERA_HEIGHT = 320, 240
CAMERA_FPS = 10  # Reduced for better processing

# Line detection parameters
BLACK_THRESHOLD = 60
BLUR_SIZE = 5
MIN_CONTOUR_AREA = 100

# YOLO Configuration
YOLO_MODEL_PATH = "yolov5n.pt"  # Nano version for speed
YOLO_CONFIDENCE = 0.5
YOLO_IOU_THRESHOLD = 0.45
OBSTACLE_CLASSES = [0, 1, 2, 3, 5, 7, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # Relevant obstacle classes
# 0:person, 1:bicycle, 2:car, 3:motorcycle, 5:bus, 7:truck, 15:cat, 16:dog, 17:horse, etc.

# Zone configuration
ZONE_BOTTOM_HEIGHT = 0.30   # Bottom 30% for line following
ZONE_TOP_HEIGHT = 0.70      # Top 70% for obstacle detection

# PID controller values
KP = 0.7
KI = 0.02
KD = 0.15
MAX_INTEGRAL = 5.0

# Commands for ESP32
COMMANDS = {'FORWARD': 'FORWARD', 'LEFT': 'LEFT', 'RIGHT': 'RIGHT', 'STOP': 'STOP', 
           'AVOID_LEFT': 'AVOID_LEFT', 'AVOID_RIGHT': 'AVOID_RIGHT'}

# Avoidance parameters
AVOIDANCE_DURATION = 8  # Frames to avoid
SAFE_DISTANCE_RATIO = 0.3  # Object must be closer than this to trigger avoidance
OBSTACLE_WIDTH_THRESHOLD = 0.2  # Minimum width ratio to consider as blocking

# Global variables
output_frame = None
frame_lock = threading.Lock()
line_offset = 0.0
steering_value = 0.0
turn_command = COMMANDS['STOP']
robot_status = "Initializing"
line_detected = False
current_fps = 0.0
confidence = 0.0
esp_connection = None
esp_connected = False

# YOLO-specific variables
yolo_model = None
obstacle_detected = False
obstacle_bbox = None
avoidance_counter = 0
avoidance_direction = None

# Statistics
robot_stats = {
    'uptime': 0,
    'total_frames': 0,
    'obstacles_detected': 0,
    'avoidance_maneuvers': 0,
    'start_time': time.time()
}

# Logging setup
logging.basicConfig(level=logging.INFO, 
                   format='[%(asctime)s] %(levelname)s: %(message)s', 
                   datefmt='%H:%M:%S')
logger = logging.getLogger("YOLOLineFollower")

# -----------------------------------------------------------------------------
# --- YOLO OBSTACLE DETECTION ---
# -----------------------------------------------------------------------------
class YOLOObstacleDetector:
    def __init__(self, model_path=YOLO_MODEL_PATH):
        try:
            logger.info("🔄 Loading YOLO model...")
            self.model = YOLO(model_path)
            
            # Warm up the model with a dummy image
            dummy_img = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            _ = self.model(dummy_img, verbose=False)
            
            logger.info("✅ YOLO model loaded successfully")
            self.available = True
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            logger.info("🔄 Falling back to simple contour detection")
            self.available = False
            self.model = None
    
    def detect_obstacles(self, frame):
        """Detect obstacles using YOLO"""
        if not self.available:
            return self._fallback_detection(frame)
        
        try:
            # Get detections from YOLO
            results = self.model(frame, conf=YOLO_CONFIDENCE, iou=YOLO_IOU_THRESHOLD, verbose=False)
            
            obstacles = []
            if results and len(results) > 0:
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes
                        for i in range(len(boxes)):
                            # Get class, confidence, and coordinates
                            cls = int(boxes.cls[i])
                            conf = float(boxes.conf[i])
                            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                            
                            # Check if it's a relevant obstacle class
                            if cls in OBSTACLE_CLASSES and conf > YOLO_CONFIDENCE:
                                # Calculate obstacle properties
                                width = x2 - x1
                                height = y2 - y1
                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                
                                # Check if obstacle is in the path
                                frame_center_x = frame.shape[1] / 2
                                distance_from_center = abs(center_x - frame_center_x) / frame_center_x
                                width_ratio = width / frame.shape[1]
                                height_ratio = height / frame.shape[0]
                                
                                # Only consider obstacles that are:
                                # 1. Large enough (significant width)
                                # 2. Close to the center of the frame
                                # 3. In the lower portion of the frame (closer to robot)
                                if (width_ratio > OBSTACLE_WIDTH_THRESHOLD and 
                                    distance_from_center < 0.4 and 
                                    center_y > frame.shape[0] * 0.3):
                                    
                                    obstacles.append({
                                        'class': cls,
                                        'confidence': conf,
                                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                        'center': (center_x, center_y),
                                        'width_ratio': width_ratio,
                                        'distance_from_center': distance_from_center
                                    })
            
            return obstacles
            
        except Exception as e:
            logger.warning(f"YOLO detection error: {e}")
            return self._fallback_detection(frame)
    
    def _fallback_detection(self, frame):
        """Simple contour-based fallback detection"""
        try:
            # Use top portion for obstacle detection
            height = frame.shape[0]
            detect_height = int(height * ZONE_TOP_HEIGHT)
            roi = frame[0:detect_height, :]
            
            # Convert to grayscale and threshold
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, BLACK_THRESHOLD + 40, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            obstacles = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area for obstacle
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w/2
                    width_ratio = w / frame.shape[1]
                    distance_from_center = abs(center_x - frame.shape[1]/2) / (frame.shape[1]/2)
                    
                    if width_ratio > OBSTACLE_WIDTH_THRESHOLD and distance_from_center < 0.4:
                        obstacles.append({
                            'class': -1,  # Unknown class for fallback
                            'confidence': 0.8,
                            'bbox': (x, y, x+w, y+h),
                            'center': (center_x, y + h/2),
                            'width_ratio': width_ratio,
                            'distance_from_center': distance_from_center
                        })
            
            return obstacles
            
        except Exception as e:
            logger.error(f"Fallback detection error: {e}")
            return []

# -----------------------------------------------------------------------------
# --- PID CONTROLLER ---
# -----------------------------------------------------------------------------
class PIDController:
    def __init__(self, kp, ki, kd, max_integral=5.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        
    def calculate(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        if dt <= 0:
            dt = 0.001
        
        # Integral with windup protection
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        
        # Derivative
        derivative = (error - self.previous_error) / dt
        
        # PID output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        
        return np.clip(output, -1.0, 1.0)
    
    def reset(self):
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

# -----------------------------------------------------------------------------
# --- ESP32 COMMUNICATION ---
# -----------------------------------------------------------------------------
class ESP32Connection:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.socket = None
        self.last_command = None
        self.connect()
    
    def connect(self):
        global esp_connected
        try:
            if self.socket:
                self.socket.close()
            
            self.socket = socket.create_connection((self.ip, self.port), timeout=2)
            self.socket.settimeout(0.2)
            esp_connected = True
            logger.info(f"✅ Connected to ESP32 at {self.ip}:{self.port}")
            return True
        except Exception as e:
            esp_connected = False
            logger.error(f"❌ ESP32 connection failed: {e}")
            self.socket = None
            return False
    
    def send_command(self, command):
        if not self.socket and not self.connect():
            return False
        
        try:
            if command != self.last_command:
                self.socket.sendall(f"{command}\n".encode())
                self.last_command = command
                logger.debug(f"📡 Sent: {command}")
            return True
        except Exception as e:
            logger.error(f"Send error: {e}")
            self.socket = None
            return False
    
    def close(self):
        if self.socket:
            try:
                self.send_command("STOP")
                self.socket.close()
            except:
                pass

# -----------------------------------------------------------------------------
# --- LINE DETECTION ---
# -----------------------------------------------------------------------------
def detect_line(frame):
    """Detect line in the bottom zone of the frame"""
    height, width = frame.shape[:2]
    
    # Define bottom zone for line detection
    bottom_start = int(height * (1 - ZONE_BOTTOM_HEIGHT))
    line_roi = frame[bottom_start:height, :]
    
    # Convert to grayscale and process
    gray = cv2.cvtColor(line_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)
    _, binary = cv2.threshold(blurred, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, 0.0
    
    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    if area < MIN_CONTOUR_AREA:
        return None, 0.0
    
    # Get center of contour
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None, 0.0
    
    cx = int(M["m10"] / M["m00"])
    
    # Calculate confidence
    confidence = min(area / (width * ZONE_BOTTOM_HEIGHT * height * 0.1), 1.0)
    
    return cx, confidence

# -----------------------------------------------------------------------------
# --- MOVEMENT CONTROL ---
# -----------------------------------------------------------------------------
def get_movement_command(steering, obstacles):
    """Determine movement command based on steering and obstacles"""
    global avoidance_counter, avoidance_direction, robot_stats
    
    # Check for obstacles that require avoidance
    if obstacles and avoidance_counter <= 0:
        # Find the most threatening obstacle (closest to center and largest)
        threat_obstacle = min(obstacles, key=lambda x: x['distance_from_center'])
        
        if threat_obstacle['distance_from_center'] < SAFE_DISTANCE_RATIO:
            avoidance_counter = AVOIDANCE_DURATION
            robot_stats['avoidance_maneuvers'] += 1
            
            # Determine avoidance direction based on obstacle position
            frame_center = CAMERA_WIDTH / 2
            obstacle_center = threat_obstacle['center'][0]
            
            if obstacle_center < frame_center:
                avoidance_direction = 'RIGHT'
                return COMMANDS['AVOID_RIGHT']
            else:
                avoidance_direction = 'LEFT'
                return COMMANDS['AVOID_LEFT']
    
    # Continue avoidance maneuver
    if avoidance_counter > 0:
        avoidance_counter -= 1
        if avoidance_direction == 'LEFT':
            return COMMANDS['AVOID_LEFT']
        elif avoidance_direction == 'RIGHT':
            return COMMANDS['AVOID_RIGHT']
    
    # Normal line following
    if abs(steering) < 0.1:
        return COMMANDS['FORWARD']
    elif steering < 0:
        return COMMANDS['RIGHT']
    else:
        return COMMANDS['LEFT']

# -----------------------------------------------------------------------------
# --- VISUALIZATION ---
# -----------------------------------------------------------------------------
def draw_debug_info(frame, line_x=None, obstacles=None, line_confidence=0.0):
    """Draw debug information on the frame"""
    height, width = frame.shape[:2]
    
    # Draw zone boundaries
    bottom_start = int(height * (1 - ZONE_BOTTOM_HEIGHT))
    cv2.rectangle(frame, (0, bottom_start), (width, height), (0, 255, 255), 2)
    cv2.putText(frame, "LINE DETECTION", (5, bottom_start + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    top_end = int(height * ZONE_TOP_HEIGHT)
    cv2.rectangle(frame, (0, 0), (width, top_end), (255, 100, 100), 2)
    cv2.putText(frame, "OBSTACLE DETECTION", (5, 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
    
    # Draw obstacles
    if obstacles:
        for obstacle in obstacles:
            x1, y1, x2, y2 = obstacle['bbox']
            
            # Color based on threat level
            if obstacle['distance_from_center'] < SAFE_DISTANCE_RATIO:
                color = (0, 0, 255)  # Red for immediate threat
                thickness = 3
            else:
                color = (0, 165, 255)  # Orange for detected but not threatening
                thickness = 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Label
            label = f"Obstacle: {obstacle['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw center line
    center_x = width // 2
    cv2.line(frame, (center_x, 0), (center_x, height), (200, 200, 200), 1)
    
    # Draw detected line
    if line_x is not None:
        line_y = bottom_start + int(ZONE_BOTTOM_HEIGHT * height / 2)
        cv2.circle(frame, (line_x, line_y), 8, (0, 255, 0), -1)
        cv2.line(frame, (center_x, line_y), (line_x, line_y), (255, 0, 255), 2)
    
    # Status information
    status_color = (0, 255, 0) if line_detected else (0, 0, 255)
    if obstacles:
        status_color = (0, 100, 255)  # Orange for obstacles
    
    info_lines = [
        f"Status: {robot_status}",
        f"Line Offset: {line_offset:.2f}",
        f"Line Conf: {line_confidence:.2f}",
        f"Command: {turn_command}",
        f"FPS: {current_fps:.1f}",
        f"Obstacles: {len(obstacles) if obstacles else 0}"
    ]
    
    for i, line in enumerate(info_lines):
        y_pos = 20 + (i * 20)
        cv2.putText(frame, line, (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    status_color if i == 0 else (255, 255, 255), 1)
    
    # Avoidance status
    if avoidance_counter > 0:
        cv2.putText(frame, f"AVOIDING {avoidance_direction}", 
                   (width//2 - 60, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

# -----------------------------------------------------------------------------
# --- FLASK WEB INTERFACE ---
# -----------------------------------------------------------------------------
app = Flask(__name__)

@app.route('/')
def index():
    return """<!DOCTYPE html>
<html>
<head>
    <title>YOLO Enhanced Line Follower</title>
    <style>
        body { font-family: Arial; background: #1a1a1a; color: white; padding: 20px; }
        .dashboard { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
        .video-section { background: #333; padding: 20px; border-radius: 10px; }
        .stats-section { background: #333; padding: 20px; border-radius: 10px; }
        .video-feed { width: 100%; border-radius: 8px; }
        .stat-item { margin: 10px 0; padding: 10px; background: #444; border-radius: 5px; }
        .status-online { color: #4CAF50; }
        .status-offline { color: #f44336; }
        h1 { text-align: center; color: #4CAF50; }
        h3 { color: #81C784; }
    </style>
    <script>
        function updateStats() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('robot-status').textContent = data.status;
                    document.getElementById('line-offset').textContent = data.offset.toFixed(2);
                    document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(1) + '%';
                    document.getElementById('command').textContent = data.command;
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('obstacles').textContent = data.obstacles_detected;
                    document.getElementById('esp-status').className = data.esp_connected ? 'status-online' : 'status-offline';
                    document.getElementById('esp-status').textContent = data.esp_connected ? 'Connected' : 'Disconnected';
                });
        }
        setInterval(updateStats, 500);
        updateStats();
    </script>
</head>
<body>
    <h1>🤖 YOLO Enhanced Line Follower Robot</h1>
    <div class="dashboard">
        <div class="video-section">
            <h3>📷 Live Camera Feed with YOLO Detection</h3>
            <img src="/video_feed" class="video-feed" alt="Robot Camera Feed">
        </div>
        <div class="stats-section">
            <h3>📊 Robot Statistics</h3>
            <div class="stat-item">Status: <span id="robot-status">Loading...</span></div>
            <div class="stat-item">Line Offset: <span id="line-offset">0.00</span></div>
            <div class="stat-item">Line Confidence: <span id="confidence">0%</span></div>
            <div class="stat-item">Current Command: <span id="command">STOP</span></div>
            <div class="stat-item">FPS: <span id="fps">0.0</span></div>
            <div class="stat-item">Obstacles Detected: <span id="obstacles">0</span></div>
            <div class="stat-item">ESP32 Status: <span id="esp-status" class="status-offline">Disconnected</span></div>
        </div>
    </div>
</body>
</html>"""

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    global robot_stats
    uptime = time.time() - robot_stats['start_time']
    
    return jsonify({
        'status': robot_status,
        'command': turn_command,
        'offset': line_offset,
        'fps': current_fps,
        'confidence': confidence,
        'line_detected': line_detected,
        'esp_connected': esp_connected,
        'uptime': uptime,
        'obstacles_detected': robot_stats.get('obstacles_detected', 0),
        'avoidance_maneuvers': robot_stats.get('avoidance_maneuvers', 0),
        'total_frames': robot_stats.get('total_frames', 0)
    })

def generate_frames():
    global output_frame, frame_lock
    while True:
        with frame_lock:
            if output_frame is not None:
                frame_to_send = output_frame.copy()
            else:
                frame_to_send = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame_to_send, "No Camera Feed", 
                           (CAMERA_WIDTH//2-80, CAMERA_HEIGHT//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        ret, jpeg = cv2.imencode('.jpg', frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(1/30)

def run_flask_server():
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        logger.error(f"Flask server error: {e}")

# -----------------------------------------------------------------------------
# --- MAIN APPLICATION ---
# -----------------------------------------------------------------------------
def main():
    global output_frame, line_offset, steering_value, turn_command, robot_status
    global line_detected, current_fps, confidence, esp_connection, yolo_model
    global obstacle_detected, obstacle_bbox, robot_stats
    
    logger.info("🚀 Starting YOLO Enhanced Line Follower Robot")
    robot_status = "Initializing"
    
    # Initialize YOLO detector
    obstacle_detector = YOLOObstacleDetector()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("❌ Failed to open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    logger.info(f"📷 Camera initialized: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    
    # Initialize PID controller
    pid = PIDController(KP, KI, KD, MAX_INTEGRAL)
    
    # Connect to ESP32
    esp_connection = ESP32Connection(ESP32_IP, ESP32_PORT)
    
    # Start web interface
    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()
    logger.info("🌐 Web dashboard available at http://localhost:5000")
    
    # FPS calculation
    fps_history = deque(maxlen=10)
    offset_history = deque(maxlen=3)
    
    logger.info("✅ Robot ready! Starting enhanced line detection with YOLO...")
    robot_status = "Ready - Enhanced detection active"
    
    try:
        frame_count = 0
        while True:
            start_time = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("⚠️ Failed to capture frame")
                robot_status = "Camera error"
                time.sleep(0.1)
                continue
            
            frame_count += 1
            robot_stats['total_frames'] = frame_count
            
            display_frame = frame.copy()
            
            # Detect obstacles using YOLO
            obstacles = obstacle_detector.detect_obstacles(frame)
            if obstacles:
                robot_stats['obstacles_detected'] += 1
                obstacle_detected = True
            else:
                obstacle_detected = False
            
            # Detect line
            line_x, line_confidence = detect_line(frame)
            confidence = line_confidence
            
            if line_x is not None and confidence > 0.2:
                line_detected = True
                
                # Calculate offset
                center_x = CAMERA_WIDTH / 2
                raw_offset = (line_x - center_x) / center_x
                offset_history.append(raw_offset)
                
                # Smooth offset
                line_offset = sum(offset_history) / len(offset_history) if offset_history else 0.0
                
                # Calculate steering
                steering_error = -line_offset  # Invert for proper steering
                steering_value = pid.calculate(steering_error)
                
                # Determine command with obstacle avoidance
                turn_command = get_movement_command(steering_value, obstacles)
                
                if obstacles and any(obs['distance_from_center'] < SAFE_DISTANCE_RATIO for obs in obstacles):
                    robot_status = f"⚠️ Obstacle detected - avoiding (C:{confidence:.2f})"
                else:
                    robot_status = f"✅ Following line (C:{confidence:.2f})"
                
            else:
                line_detected = False
                if obstacles:
                    robot_status = "⚠️ Line lost - obstacle present"
                    turn_command = COMMANDS['STOP']
                else:
                    robot_status = "🔍 Searching for line"
                    # Continue last command briefly or stop
                    if frame_count % 30 > 15:
                        turn_command = COMMANDS['STOP']
            
            # Send command to ESP32
            if esp_connection:
                esp_connection.send_command(turn_command)
            
            # Draw debug information
            draw_debug_info(display_frame, line_x, obstacles, confidence)
            
            # Update output frame
            with frame_lock:
                output_frame = display_frame.copy()
            
            # Calculate FPS
            processing_time = time.time() - start_time
            if processing_time > 0:
                fps_history.append(1.0 / processing_time)
                current_fps = sum(fps_history) / len(fps_history)
            
            # Log status periodically
            if frame_count % 60 == 0:
                logger.info(f"📊 Status: {robot_status} | FPS: {current_fps:.1f} | "
                           f"Command: {turn_command} | ESP32: {'✅' if esp_connected else '❌'} | "
                           f"Obstacles: {len(obstacles) if obstacles else 0}")
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        logger.info("🛑 Stopping robot (Ctrl+C pressed)")
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}", exc_info=True)
    finally:
        # Cleanup
        if esp_connection:
            esp_connection.send_command(COMMANDS['STOP'])
            esp_connection.close()
        
        if cap.isOpened():
            cap.release()
        
        logger.info("✅ Robot stopped cleanly")

if __name__ == "__main__":
    main() 