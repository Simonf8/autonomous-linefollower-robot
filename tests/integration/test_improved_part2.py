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

# -----------------------------------------------------------------------------
# --- CONFIGURATION FOR BLACK LINE FOLLOWING ---
# -----------------------------------------------------------------------------
# ESP32 Configuration - UPDATE THIS TO MATCH YOUR ESP32's IP
ESP32_IP = '192.168.53.117'  # Change this to your ESP32's IP address
ESP32_PORT = 1234
CAMERA_WIDTH, CAMERA_HEIGHT = 320, 240
CAMERA_FPS = 15

# Image processing parameters
BLACK_THRESHOLD = 60  # Higher values detect darker lines
BLUR_SIZE = 5
MIN_CONTOUR_AREA = 100  # Minimum area to be considered a line

# Corner detection parameters
CORNER_DETECTION_ENABLED = True
CORNER_CONFIDENCE_BOOST = 1.2
CORNER_CIRCULARITY_THRESHOLD = 0.4  # Lower values indicate corners

# Simple PID controller values
KP = 0.6  # Proportional gain
KI = 0.02  # Integral gain 
KD = 0.1   # Derivative gain
MAX_INTEGRAL = 5.0  # Prevent integral windup

# Commands for ESP32
COMMANDS = {'FORWARD': 'FORWARD', 'LEFT': 'LEFT', 'RIGHT': 'RIGHT', 'STOP': 'STOP'}
SPEED = 'SLOW'  # Default speed

# Steering parameters
STEERING_DEADZONE = 0.1  # Ignore small errors
MAX_STEERING = 1.0  # Maximum steering value

# -----------------------------------------------------------------------------
# --- Global Variables ---
# -----------------------------------------------------------------------------
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
robot_stats = {
    'uptime': 0,
    'total_frames': 0,
    'lost_frames': 0,
    'corner_count': 0
}

# -----------------------------------------------------------------------------
# --- Logging Setup ---
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, 
                   format='[%(asctime)s] %(levelname)s: %(message)s', 
                   datefmt='%H:%M:%S')
logger = logging.getLogger("SimpleLineFollower")

# -----------------------------------------------------------------------------
# --- PID Controller ---
# -----------------------------------------------------------------------------
class SimplePID:
    def __init__(self, kp, ki, kd, max_integral=5.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        
    def calculate(self, error):
        # Calculate time difference
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Ensure dt is not too small
        dt = max(dt, 0.001)
        
        # Calculate integral term with windup protection
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        
        # Calculate derivative term
        derivative = (error - self.previous_error) / dt
        
        # Calculate PID output
        p_term = self.kp * error
        i_term = self.ki * self.integral
        d_term = self.kd * derivative
        
        output = p_term + i_term + d_term
        
        # Store error for next iteration
        self.previous_error = error
        
        # Limit output to range [-1.0, 1.0]
        return np.clip(output, -1.0, 1.0)
    
    def reset(self):
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        logger.info("🔄 PID Controller Reset")

# -----------------------------------------------------------------------------
# --- ESP32 Communication ---
# -----------------------------------------------------------------------------
class ESP32Connection:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.socket = None
        self.last_command = None
        self.connection_attempts = 0
        self.connect()
    
    def connect(self):
        global esp_connected
        if self.socket:
            try: 
                self.socket.close()
            except: 
                pass
            self.socket = None
        
        try:
            self.socket = socket.create_connection((self.ip, self.port), timeout=2)
            self.socket.settimeout(0.2)
            esp_connected = True
            self.connection_attempts = 0
            logger.info(f"✅ Connected to ESP32 at {self.ip}:{self.port}")
            return True
        except Exception as e:
            esp_connected = False
            self.connection_attempts += 1
            if self.connection_attempts % 10 == 1:  # Log every 10 attempts
                logger.error(f"❌ Failed to connect to ESP32 (attempt {self.connection_attempts}): {e}")
            self.socket = None
            return False
    
    def send_command(self, command):
        global esp_connected
        if not self.socket and not self.connect():
            return False
        
        try:
            # Add newline to command
            full_command = f"{command}\n"
            
            # Only send if command changed
            if full_command != self.last_command:
                self.socket.sendall(full_command.encode())
                self.last_command = full_command
                logger.debug(f"📡 Sent to ESP32: {command}")
            
            return True
        except Exception as e:
            esp_connected = False
            logger.error(f"Error sending command to ESP32: {e}")
            self.socket = None
            return False
    
    def close(self):
        global esp_connected
        if self.socket:
            try:
                self.send_command("STOP")
                time.sleep(0.1)
                self.socket.close()
            except:
                pass
            self.socket = None
            esp_connected = False
            logger.info("Disconnected from ESP32")

# -----------------------------------------------------------------------------
# --- Image Processing ---
# -----------------------------------------------------------------------------
def detect_line_in_roi(roi):
    """Helper function to detect line in a specific ROI"""
    if roi.size == 0:
        return None, 0.0
        
    # Find contours in the ROI
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, 0.0
    
    # Find the largest contour (most likely the line)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Check if contour is large enough to be a line
    area = cv2.contourArea(largest_contour)
    if area < MIN_CONTOUR_AREA:  # Minimum area threshold
        return None, 0.0
    
    # Get the center of the contour
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None, 0.0
    
    # Calculate x-position of line center
    cx = int(M["m10"] / M["m00"])
    
    # Calculate confidence based on contour area
    height, width = roi.shape
    confidence = min(area / (width * height * 0.1), 1.0)
    
    # Check if contour might be a corner by looking at its shape
    # Calculate circularity - corners tend to be less circular
    perimeter = cv2.arcLength(largest_contour, True)
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Low circularity might indicate a corner
        if circularity < CORNER_CIRCULARITY_THRESHOLD:
            # Increase confidence for potential corners
            confidence *= CORNER_CONFIDENCE_BOOST
    
    return cx, confidence

def process_image(frame):
    """Detect black line in image and return line position, handling corners better"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)
    
    # Threshold the image to identify black regions
    _, binary = cv2.threshold(blurred, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    
    # Create multiple ROIs to better detect corners
    height, width = binary.shape
    
    # Bottom ROI (closest to robot) - Primary detection area
    bottom_height = int(height * 0.3)
    bottom_roi = binary[height - bottom_height:height, :]
    
    # Middle ROI (for corner detection) - Look ahead to see upcoming turns
    middle_height = int(height * 0.2)
    middle_roi = binary[height - bottom_height - middle_height:height - bottom_height, :]
    
    # Process bottom ROI first (where the line should be)
    bottom_x, bottom_confidence = detect_line_in_roi(bottom_roi)
    
    # If bottom confidence is high, use it directly
    if bottom_confidence > 0.5:
        return bottom_x, bottom_roi, bottom_confidence
        
    # Process middle ROI to detect upcoming corners
    middle_x, middle_confidence = detect_line_in_roi(middle_roi)
    
    # Corner detection logic
    if bottom_confidence > 0.2 and middle_confidence > 0.2:
        # Both ROIs have some line - could be a corner
        # Calculate how much the line shifts - indicates a corner
        if bottom_x is not None and middle_x is not None:
            # Calculate shift which could indicate a corner
            shift = abs(middle_x - bottom_x)
            
            if shift > width * 0.1:  # Significant shift indicates corner
                logger.debug(f"Corner detected! Shift: {shift}")
                robot_stats['corner_count'] += 1

                # Use a weighted average favoring the bottom ROI
                weighted_x = int((bottom_x * 0.7) + (middle_x * 0.3))
                weighted_confidence = max(bottom_confidence, middle_confidence)
                return weighted_x, bottom_roi, weighted_confidence
    
    # If no clear corner, prioritize bottom ROI
    if bottom_confidence > 0.2:
        return bottom_x, bottom_roi, bottom_confidence
    elif middle_confidence > 0.3:  # Higher threshold for middle section
        return middle_x, middle_roi, middle_confidence * 0.8  # Slightly reduce confidence
    else:
        return None, None, 0.0

# -----------------------------------------------------------------------------
# --- Movement Control ---
# -----------------------------------------------------------------------------
def get_turn_command(steering):
    """Convert steering value to turn command with improved corner handling"""
    # Apply deadzone to avoid oscillation
    if abs(steering) < STEERING_DEADZONE:
        return COMMANDS['FORWARD']
    
    # For sharper turns (might be corners), use more aggressive turning
    if abs(steering) > 0.7:
        if steering < 0:  # Negative steering turns right
            return COMMANDS['RIGHT']
        else:  # Positive steering turns left
            return COMMANDS['LEFT']
    
    # Normal steering behavior
    if steering < 0:  # Negative steering turns right
        return COMMANDS['RIGHT']
    else:  # Positive steering turns left
        return COMMANDS['LEFT']

# -----------------------------------------------------------------------------
# --- Visualization ---
# -----------------------------------------------------------------------------
def draw_debug_info(frame, line_x=None, roi=None, confidence=0.0):
    """Draw debug information on frame"""
    height, width = frame.shape[:2]
    
    # Draw ROI rectangles for both bottom and middle sections
    bottom_height = int(height * 0.3)
    middle_height = int(height * 0.2)
    
    # Bottom ROI (main detection area)
    bottom_top = height - bottom_height
    cv2.rectangle(frame, (0, bottom_top), (width, height), (0, 255, 255), 2)
    cv2.putText(frame, "Bottom ROI", (5, bottom_top + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Middle ROI (corner detection)
    middle_top = bottom_top - middle_height
    cv2.rectangle(frame, (0, middle_top), (width, bottom_top), (0, 200, 255), 2)
    cv2.putText(frame, "Corner ROI", (5, middle_top + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    
    # Draw center line
    center_x = width // 2
    cv2.line(frame, (center_x, 0), (center_x, height), (200, 200, 200), 1)
    
    # Draw detected line position
    if line_x is not None:
        # Draw circle at detected line position
        line_y = bottom_top + bottom_height // 2
        cv2.circle(frame, (line_x, line_y), 10, (0, 0, 255), -1)
        cv2.line(frame, (center_x, line_y), (line_x, line_y), (255, 0, 255), 2)
        
        # Draw offset line
        offset_text = f"Offset: {line_offset:.2f}"
        cv2.putText(frame, offset_text, (line_x - 40, line_y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    
    # Draw status information
    status_color = (0, 255, 0) if line_detected else (0, 0, 255)
    cv2.putText(frame, f"Status: {robot_status}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    cv2.putText(frame, f"Offset: {line_offset:.2f}", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"Command: {turn_command}", (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw corner detection indicator
    if "corner" in robot_status.lower():
        cv2.putText(frame, "CORNER DETECTED", (width//2 - 80, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Draw direction arrow
    arrow_y = height - 30
    arrow_color = (0, 255, 0)  # Green for forward
    arrow_text = "FORWARD"
    
    if turn_command == COMMANDS['LEFT']:
        arrow_color = (0, 255, 255)  # Yellow for left
        arrow_text = "LEFT"
    elif turn_command == COMMANDS['RIGHT']:
        arrow_color = (255, 255, 0)  # Cyan for right
        arrow_text = "RIGHT"
    elif turn_command == COMMANDS['STOP']:
        arrow_color = (0, 0, 255)  # Red for stop
        arrow_text = "STOP"
    
    cv2.putText(frame, arrow_text, (width // 2 - 30, arrow_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, arrow_color, 2)

# -----------------------------------------------------------------------------
# --- Flask Web Interface ---
# -----------------------------------------------------------------------------
app = Flask(__name__)

@app.route('/')
def index():
    """Serve the modern dashboard directly as HTML"""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Line Follower Robot Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            color: #fff;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-left: 10px;
            animation: pulse 2s infinite;
        }
        
        .status-online { background-color: #4CAF50; }
        .status-offline { background-color: #f44336; }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        @media (max-width: 768px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
        }
        
        .video-section {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .video-feed {
            width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .stats-section {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-header {
            font-size: 0.9em;
            opacity: 0.8;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 0.8em;
            opacity: 0.7;
        }
        
        .command-display {
            text-align: center;
            padding: 15px;
            border-radius: 10px;
            font-weight: bold;
            font-size: 1.2em;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-top: 10px;
        }
        
        .cmd-forward { background: linear-gradient(45deg, #4CAF50, #8BC34A); }
        .cmd-left { background: linear-gradient(45deg, #FF9800, #FFC107); }
        .cmd-right { background: linear-gradient(45deg, #2196F3, #03DAC6); }
        .cmd-stop { background: linear-gradient(45deg, #f44336, #E91E63); }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        
        .connection-status {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
        }
        
        .connection-item {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Line Follower Robot</h1>
            <div>Control Dashboard <span id="status-dot" class="status-indicator status-offline"></span></div>
        </div>
        
        <div class="dashboard">
            <div class="video-section">
                <h3 style="margin-bottom: 15px;">📷 Live Camera Feed</h3>
                <img src="/video_feed" class="video-feed" alt="Robot Camera Feed">
            </div>
            
            <div class="stats-section">
                <div class="stat-card">
                    <div class="stat-header">Robot Status</div>
                    <div class="stat-value" id="robot-status">Loading...</div>
                    <div class="stat-label">Current Operation</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-header">Current Command</div>
                    <div class="command-display cmd-stop" id="command-display">STOP</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-header">Line Detection</div>
                    <div class="stat-value" id="confidence">0%</div>
                    <div class="stat-label">Confidence Level</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="confidence-bar" style="width: 0%"></div>
                    </div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-header">Performance</div>
                    <div class="stat-value" id="fps">0.0</div>
                    <div class="stat-label">Frames Per Second</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-header">Steering</div>
                    <div class="stat-value" id="offset">0.00</div>
                    <div class="stat-label">Line Offset</div>
                </div>
            </div>
        </div>
        
        <div class="connection-status">
            <div class="connection-item">
                <span>ESP32 Connection:</span>
                <span id="esp-status">
                    <div class="loading"></div> Connecting...
                </span>
            </div>
            <div class="connection-item">
                <span>Camera Feed:</span>
                <span id="camera-status">
                    <div class="loading"></div> Loading...
                </span>
            </div>
            <div class="connection-item">
                <span>Uptime:</span>
                <span id="uptime">00:00:00</span>
            </div>
        </div>
    </div>
    
    <script>
        let startTime = Date.now();
        let lastUpdateTime = Date.now();
        
        // Update status periodically
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update robot status
                    document.getElementById('robot-status').textContent = data.status || 'Unknown';
                    
                    // Update command display
                    const commandDisplay = document.getElementById('command-display');
                    const command = data.command || 'STOP';
                    commandDisplay.textContent = command;
                    commandDisplay.className = 'command-display cmd-' + command.toLowerCase();
                    
                    // Update confidence
                    const confidence = Math.round((data.confidence || 0) * 100);
                    document.getElementById('confidence').textContent = confidence + '%';
                    document.getElementById('confidence-bar').style.width = confidence + '%';
                    
                    // Update FPS
                    document.getElementById('fps').textContent = (data.fps || 0).toFixed(1);
                    
                    // Update offset
                    document.getElementById('offset').textContent = (data.offset || 0).toFixed(2);
                    
                    // Update connection statuses
                    const statusDot = document.getElementById('status-dot');
                    const espStatus = document.getElementById('esp-status');
                    const cameraStatus = document.getElementById('camera-status');
                    
                    if (data.esp_connected) {
                        statusDot.className = 'status-indicator status-online';
                        espStatus.innerHTML = '✅ Connected';
                    } else {
                        statusDot.className = 'status-indicator status-offline';
                        espStatus.innerHTML = '❌ Disconnected';
                    }
                    
                    // Camera status based on recent updates
                    const timeSinceUpdate = Date.now() - lastUpdateTime;
                    if (timeSinceUpdate < 2000) {
                        cameraStatus.innerHTML = '✅ Active';
                    } else {
                        cameraStatus.innerHTML = '⚠️ No Signal';
                    }
                    
                    lastUpdateTime = Date.now();
                })
                .catch(error => {
                    console.error('Error fetching status:', error);
                    document.getElementById('robot-status').textContent = 'Connection Error';
                    document.getElementById('status-dot').className = 'status-indicator status-offline';
                });
        }
        
        // Update uptime
        function updateUptime() {
            const elapsed = Date.now() - startTime;
            const hours = Math.floor(elapsed / 3600000);
            const minutes = Math.floor((elapsed % 3600000) / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            
            const uptime = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            document.getElementById('uptime').textContent = uptime;
        }
        
        // Start periodic updates
        setInterval(updateStatus, 500);  // Update every 500ms
        setInterval(updateUptime, 1000); // Update uptime every second
        
        // Initial load
        updateStatus();
        updateUptime();
        
        // Handle image load events
        const videoFeed = document.querySelector('.video-feed');
        videoFeed.addEventListener('load', () => {
            document.getElementById('camera-status').innerHTML = '✅ Active';
        });
        
        videoFeed.addEventListener('error', () => {
            document.getElementById('camera-status').innerHTML = '❌ Error';
        });
    </script>
</body>
</html>"""

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    global robot_stats
    
    # Calculate uptime
    current_time = time.time()
    if 'start_time' not in robot_stats:
        robot_stats['start_time'] = current_time
    
    uptime = current_time - robot_stats['start_time']
    
    return jsonify({
        'status': robot_status,
        'command': turn_command,
        'offset': line_offset,
        'fps': current_fps,
        'confidence': confidence,
        'line_detected': line_detected,
        'esp_connected': esp_connected,
        'uptime': uptime,
        'total_frames': robot_stats.get('total_frames', 0),
        'corner_count': robot_stats.get('corner_count', 0)
    })

# Legacy endpoint for backward compatibility
@app.route('/status')
def status():
    return api_status().get_json()

def generate_frames():
    global output_frame, frame_lock, robot_stats
    while True:
        with frame_lock:
            if output_frame is not None:
                frame_to_send = output_frame.copy()
                robot_stats['total_frames'] = robot_stats.get('total_frames', 0) + 1
            else:
                frame_to_send = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame_to_send, "No Camera Feed", 
                           (CAMERA_WIDTH//2-80, CAMERA_HEIGHT//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                robot_stats['lost_frames'] = robot_stats.get('lost_frames', 0) + 1
        
        ret, jpeg = cv2.imencode('.jpg', frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(1/30)  # Limit to 30 FPS for web streaming

def run_flask_server():
    logger.info("🌐 Starting web server on http://0.0.0.0:5000")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        logger.error(f"Failed to start web server: {e}")

# -----------------------------------------------------------------------------
# --- Main Application ---
# -----------------------------------------------------------------------------
def main():
    global output_frame, line_offset, steering_value, turn_command, robot_status
    global line_detected, current_fps, confidence, esp_connection, robot_stats
    
    logger.info("🚀 Starting Simple Line Follower Robot")
    robot_status = "Starting camera"
    robot_stats['start_time'] = time.time()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Try camera 0 first
    if not cap.isOpened():
        logger.warning("Camera 0 failed, trying camera 1")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            logger.error("❌ Failed to open any camera")
            robot_status = "Camera Error"
            # Start web server anyway for status monitoring
            run_flask_server()
            return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
    
    # Get actual camera resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"📷 Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
    
    # Initialize PID controller
    pid = SimplePID(KP, KI, KD, MAX_INTEGRAL)
    
    # Connect to ESP32
    robot_status = "Connecting to ESP32"
    esp_connection = ESP32Connection(ESP32_IP, ESP32_PORT)
    
    # Start web interface in a separate thread
    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()
    logger.info("🌐 Web dashboard available at http://localhost:5000")
    
    # FPS calculation
    fps_history = deque(maxlen=30)
    search_counter = 0
    
    # History for smoothing
    offset_history = deque(maxlen=3)
    steering_history = deque(maxlen=2)
    last_known_good_offset = 0.0
    corner_detected_count = 0
    
    logger.info("✅ Robot ready! Starting line detection...")
    robot_status = "Ready - Searching for line"
    
    try:
        frame_count = 0
        while True:
            # Measure processing time for FPS calculation
            start_time = time.time()
            
            # Capture frame from camera
            ret, frame = cap.read()
            if not ret:
                logger.warning("⚠️ Failed to capture frame")
                robot_status = "Camera read error"
                robot_stats['lost_frames'] = robot_stats.get('lost_frames', 0) + 1
                time.sleep(0.1)
                continue
            
            frame_count += 1
            robot_stats['total_frames'] = frame_count
            
            # Process image to find line
            line_x, roi, detection_confidence = process_image(frame)
            confidence = detection_confidence
            
            # Create a copy for visualization
            display_frame = frame.copy()
            
            # Update line following logic
            if line_x is not None and confidence > 0.2:
                # Line detected
                line_detected = True
                search_counter = 0
                
                # Calculate line position relative to center (-1.0 to 1.0)
                center_x = frame.shape[1] / 2
                raw_offset = (line_x - center_x) / center_x
                
                # Add to history for smoothing
                offset_history.append(raw_offset)
                
                # Use average of recent offsets for stability
                if len(offset_history) > 0:
                    line_offset = sum(offset_history) / len(offset_history)
                else:
                    line_offset = raw_offset
                
                # Remember this offset as last known good position
                last_known_good_offset = line_offset
                
                # Check for corner
                if abs(line_offset) > 0.4 and confidence > 0.3:
                    corner_detected_count += 1
                    
                    # Adjust status based on corner detection duration
                    if corner_detected_count > 5:
                        robot_status = f"🔄 Taking corner ({corner_detected_count})"
                        
                        # For sharp corners, exaggerate the steering to make tighter turns
                        if abs(line_offset) > 0.5:
                            line_offset *= 1.5  # Increase turning response
                    else:
                        robot_status = f"⚠️ Corner detected ({corner_detected_count})"
                else:
                    if corner_detected_count > 0:
                        corner_detected_count -= 1
                    else:
                        corner_detected_count = 0
                        robot_status = f"✅ Following line (C:{confidence:.2f})"
                
                # Calculate steering using PID controller
                # Negative offset means line is to the left, so invert for steering
                steering_error = -line_offset
                steering_value = pid.calculate(steering_error)
                
                # Add to steering history
                steering_history.append(steering_value)
                
                # Use average of recent steering values for smoother response
                if len(steering_history) > 0:
                    avg_steering = sum(steering_history) / len(steering_history)
                else:
                    avg_steering = steering_value
                
                # Convert steering to command
                turn_command = get_turn_command(avg_steering)
                
                logger.debug(f"Line at x={line_x}, offset={line_offset:.2f}, steering={steering_value:.2f}")
                
            else:
                # Line not detected - search mode
                line_detected = False
                search_counter += 1
                
                if search_counter < 5:
                    # Keep last command briefly (helps with short line gaps)
                    robot_status = "🔍 Searching for line (brief gap)"
                elif search_counter < 15:
                    # Try turning based on last known position
                    if last_known_good_offset < 0:
                        turn_command = COMMANDS['RIGHT']
                        robot_status = "🔍 Searching right (last seen left)"
                    else:
                        turn_command = COMMANDS['LEFT']
                        robot_status = "🔍 Searching left (last seen right)"
                elif search_counter < 30:
                    # Switch direction
                    if turn_command == COMMANDS['LEFT']:
                        turn_command = COMMANDS['RIGHT']
                    else:
                        turn_command = COMMANDS['LEFT']
                    robot_status = f"🔄 Searching opposite direction ({search_counter})"
                elif search_counter < 45:
                    # Try moving forward a bit
                    turn_command = COMMANDS['FORWARD']
                    robot_status = "⬆️ Moving forward to find line"
                else:
                    # Stop and reset if line completely lost
                    turn_command = COMMANDS['STOP']
                    robot_status = "🛑 Line lost - stopped"
                    
                    # Reset search after a pause
                    if search_counter > 60:
                        search_counter = 0
                        last_known_good_offset = 0.0
                        pid.reset()
                        offset_history.clear()
                        steering_history.clear()
                        logger.info("🔄 Search pattern reset")
            
            # Send command to ESP32
            if esp_connection:
                success = esp_connection.send_command(turn_command)
                if not success and frame_count % 30 == 0:  # Log occasionally
                    logger.warning("📡 ESP32 communication failed")
            
            # Draw debug information
            draw_debug_info(display_frame, line_x, roi, confidence)
            
            # Update output frame for web interface
            with frame_lock:
                output_frame = display_frame.copy()
            
            # Calculate FPS
            processing_time = time.time() - start_time
            if processing_time > 0:
                fps_history.append(1.0 / processing_time)
                current_fps = sum(fps_history) / len(fps_history)
            
            # Log status periodically
            if frame_count % 60 == 0:  # Every 2 seconds at 30fps
                logger.info(f"📊 Status: {robot_status} | FPS: {current_fps:.1f} | "
                           f"Command: {turn_command} | ESP32: {'✅' if esp_connected else '❌'}")
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        logger.info("🛑 Stopping robot (Ctrl+C pressed)")
        robot_status = "Shutting down"
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}", exc_info=True)
        robot_status = f"Error: {str(e)}"
    finally:
        # Clean up
        robot_status = "Cleaning up"
        
        if esp_connection:
            logger.info("📡 Stopping robot and disconnecting ESP32")
            esp_connection.send_command(COMMANDS['STOP'])
            time.sleep(0.2)  # Give time for stop command
            esp_connection.close()
        
        if 'cap' in locals() and cap.isOpened():
            cap.release()
            logger.info("📷 Camera released")
        
        logger.info("✅ Robot stopped cleanly")

if __name__ == "__main__":
    main()