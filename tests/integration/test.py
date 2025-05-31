#!/usr/bin/env python3

import cv2
import numpy as np
import socket
import time
import logging
import sys
import math
import threading
from collections import deque
from flask import Flask, Response, render_template_string

# -----------------------------------------------------------------------------
# --- CONFIGURATION FOR BLACK LINE FOLLOWING ---
# -----------------------------------------------------------------------------
ESP32_IP = '192.168.53.117'
ESP32_PORT = 1234
REQUESTED_CAM_W, REQUESTED_CAM_H = 320, 240
REQUESTED_CAM_FPS = 20

# BLACK line detection parameters - OPTIMIZED
LINE_THICKNESS_MIN = 8
LINE_THICKNESS_MAX = 150
CANNY_LOW = 60
CANNY_HIGH = 180
HOUGH_THRESHOLD = 30
HOUGH_MIN_LINE_LENGTH = 30
HOUGH_MAX_LINE_GAP = 10

# BLACK line thresholds
BLACK_THRESHOLD = 80  # Adjusted for better black detection
ADAPTIVE_BLOCK_SIZE = 11  # Smaller for more sensitive detection
ADAPTIVE_C = 5  # Lower for more detection

# ROI (Region of Interest) - focus on middle and bottom where line should be
ROI_ZONES = [
    {"height_ratio": 0.25, "top_offset": 0.35, "weight": 2.0},  # Middle zone (most important for following)
    {"height_ratio": 0.3, "top_offset": 0.65, "weight": 1.5},  # Bottom zone (for direction)
]

# PID Controller - TUNED FOR RESPONSIVE BLACK LINE FOLLOWING
PID_KP = 0.6   # Increased for more responsive steering
PID_KI = 0.02  # Slightly increased to correct drift
PID_KD = 0.12  # Reduced to prevent oscillation
PID_INT_MAX = 0.3

# Speed commands
SPEEDS = {
    'FAST': 's',
    'NORMAL': 'n', 
    'SLOW': 'S',
    'TURN': 'T',
    'STOP': 'H'
}

# Control thresholds - OPTIMIZED FOR RESPONSIVE FOLLOWING
STEERING_DEADZONE = 0.03    # Much smaller deadzone for better response
OFFSET_THRESHOLDS = {
    'PERFECT': 0.05,    # Tighter for more precise following
    'GOOD': 0.12,       # Reasonable tolerance
    'MODERATE': 0.25,   # Moderate corrections
    'LARGE': 0.4        # Major corrections
}

# Movement control
MIN_COMMAND_INTERVAL = 0.03
STABILITY_FRAMES = 1

# -----------------------------------------------------------------------------
# --- Global Variables ---
# -----------------------------------------------------------------------------
CAM_W, CAM_H = REQUESTED_CAM_W, REQUESTED_CAM_H
output_frame_flask = None
frame_lock = threading.Lock()

# Status variables
current_line_angle = 0.0
current_line_offset = 0.0
current_steering = 0.0
current_speed_cmd = SPEEDS['STOP']
current_turn_cmd = "F"
lines_detected = 0
robot_status = "Initializing"
fps_current = 0.0
confidence_score = 0.0
detection_method = "Black Line Detection"

# Movement control
last_command_time = 0
stable_direction_counter = 0
last_turn_direction = "F"
search_memory = deque(maxlen=30)

# -----------------------------------------------------------------------------
# --- Logging Setup ---
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='🤖 [%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("BlackLineFollower")

# -----------------------------------------------------------------------------
# --- PID Controller ---
# -----------------------------------------------------------------------------
class BlackLinePIDController:
    def __init__(self, kp, ki, kd, integral_max):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_max = integral_max
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        self.error_history = deque(maxlen=5)
    
    def calculate(self, error, dt=None):
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_time
        dt = max(dt, 1e-3)
        
        # Smooth error
        self.error_history.append(error)
        if len(self.error_history) >= 3:
            smoothed_error = np.median(list(self.error_history))
        else:
            smoothed_error = error
        
        # Integral with windup protection
        self.integral += smoothed_error * dt
        self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)
        
        # Derivative
        derivative = (smoothed_error - self.prev_error) / dt
        
        # PID output
        output = (self.kp * smoothed_error) + (self.ki * self.integral) + (self.kd * derivative)
        
        self.prev_error = smoothed_error
        self.last_time = current_time
        
        return np.clip(output, -1.0, 1.0)
    
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.error_history.clear()

# -----------------------------------------------------------------------------
# --- ESP32 Communication ---
# -----------------------------------------------------------------------------
class ESP32Communicator:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = None
        self.last_speed_cmd = None
        self.last_turn_cmd = None
        self.connect()
    
    def connect(self):
        try:
            if self.sock:
                self.sock.close()
            self.sock = socket.create_connection((self.ip, self.port), timeout=2)
            logger.info(f"✅ ESP32 connected: {self.ip}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"❌ ESP32 connection failed: {e}")
            self.sock = None
            return False
    
    def send_command(self, speed_cmd, turn_cmd):
        if not self.sock:
            self.connect()
            if not self.sock:
                return False
        
        try:
            if self.last_speed_cmd != speed_cmd:
                self.sock.sendall(speed_cmd.encode())
                self.last_speed_cmd = speed_cmd
            
            if self.last_turn_cmd != turn_cmd:
                self.sock.sendall(turn_cmd.encode())
                self.last_turn_cmd = turn_cmd
            
            return True
        except Exception as e:
            logger.error(f"💥 Command send failed: {e}")
            self.sock = None
            self.last_speed_cmd = None
            self.last_turn_cmd = None
            return False
    
    def close(self):
        if self.sock:
            try:
                self.sock.sendall((SPEEDS['STOP'] + "F").encode())
                self.sock.close()
            except:
                pass
            self.sock = None
            logger.info("🔌 ESP32 disconnected")

# -----------------------------------------------------------------------------
# --- BLACK LINE DETECTION FUNCTIONS ---
# -----------------------------------------------------------------------------
def preprocess_for_black_lines(frame):
    """Optimized preprocessing for BLACK line detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Slight blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Remove bright reflections that interfere with black detection
    bright_mask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
    if cv2.countNonZero(bright_mask) > 0:
        blurred = cv2.inpaint(blurred, bright_mask, 3, cv2.INPAINT_TELEA)
    
    # PRIMARY: Simple threshold for black pixels
    _, binary_simple = cv2.threshold(blurred, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    
    # SECONDARY: Adaptive threshold for varying lighting
    binary_adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
    )
    
    # Combine both methods
    combined = cv2.bitwise_or(binary_simple, binary_adaptive)
    
    # Morphological operations to clean up
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)
    
    return cleaned

def extract_roi_zones(frame):
    """Extract multiple ROI zones for detection"""
    height, width = frame.shape[:2]
    rois = []
    
    for zone in ROI_ZONES:
        roi_top = int(height * zone["top_offset"])
        roi_height = int(height * zone["height_ratio"])
        roi = frame[roi_top:roi_top + roi_height, :]
        rois.append({
            'roi': roi,
            'top': roi_top,
            'height': roi_height,
            'weight': zone["weight"]
        })
    
    return rois

def detect_black_lines(roi_data):
    """Detect black lines using center-of-mass method - more stable than Hough"""
    best_offset = None
    best_confidence = 0
    all_lines = []
    
    for roi_info in roi_data:
        roi = roi_info['roi']
        weight = roi_info['weight']
        
        if roi.size == 0:
            continue
        
        roi_height, roi_width = roi.shape
        
        # Count black pixels in each column
        column_sums = np.sum(roi, axis=0)
        total_pixels = roi_height * 255  # Maximum possible sum per column
        
        # Convert to "blackness" - higher values = more black pixels
        blackness = total_pixels - column_sums
        
        # Smooth the signal to reduce noise
        if roi_width > 10:
            kernel_size = min(11, roi_width // 10 * 2 + 1)  # Odd number
            blackness = cv2.GaussianBlur(blackness.astype(np.float32), (kernel_size, 1), 0).flatten()
        
        # Find the weighted center of the black pixels
        total_blackness = np.sum(blackness)
        
        if total_blackness > roi_height * 50:  # Minimum threshold for line detection
            # Calculate center of mass
            x_coords = np.arange(roi_width)
            center_x = np.sum(x_coords * blackness) / total_blackness
            
            # Calculate offset from center
            center_offset = center_x - (roi_width / 2)
            normalized_offset = center_offset / (roi_width / 2)
            
            # Calculate confidence based on:
            # 1. Amount of black pixels
            # 2. How concentrated they are (peak detection)
            # 3. Position consistency
            
            max_blackness = np.max(blackness)
            concentration = max_blackness / (total_blackness / roi_width + 1e-6)  # Avoid division by zero
            pixel_ratio = total_blackness / (roi_height * roi_width * 255)
            
            confidence = min(
                pixel_ratio * 5,     # Boost for more black pixels
                concentration * 0.3,  # Boost for concentrated line
                1.0                  # Cap at 100%
            ) * weight
            
            if confidence > best_confidence:
                best_offset = normalized_offset
                best_confidence = confidence
                
                # Create a simple "line" representation for visualization
                line_x = int(center_x)
                all_lines = [{
                    'coords': (line_x, 0, line_x, roi_height-1),
                    'center_x': center_x,
                    'score': confidence * 100
                }]
    
    return best_offset, 90.0, all_lines, best_confidence  # Assume vertical line

# -----------------------------------------------------------------------------
# --- MOVEMENT CONTROL ---
# -----------------------------------------------------------------------------
def get_speed_command(offset, confidence, steering_output):
    """Get speed command based on line conditions - prioritize forward movement when confident"""
    if confidence < 0.8:  # Lower threshold for better detection
        return SPEEDS['SLOW']
    
    abs_offset = abs(offset) if offset is not None else 1.0
    abs_steering = abs(steering_output)
    
    # HIGH CONFIDENCE (88%+) - Drive forward aggressively!
    if confidence > 0.88:
        if abs_offset < OFFSET_THRESHOLDS['MODERATE']:  # Even with moderate offset, go fast when very confident
            return SPEEDS['FAST']
        else:
            return SPEEDS['NORMAL']  # Still move forward even with larger offset
    
    # MEDIUM-HIGH CONFIDENCE - Standard behavior
    elif confidence > 0.4:
        if abs_offset < OFFSET_THRESHOLDS['PERFECT']:
            return SPEEDS['FAST']
        elif abs_offset < OFFSET_THRESHOLDS['GOOD']:
            return SPEEDS['NORMAL']
        else:
            return SPEEDS['SLOW']
    
    # LOW-MEDIUM CONFIDENCE - Be more cautious
    elif confidence > 0.3:
        if abs_offset < OFFSET_THRESHOLDS['GOOD']:
            return SPEEDS['NORMAL']
        else:
            return SPEEDS['SLOW']
    
    # LOW CONFIDENCE - Move slowly
    else:
        return SPEEDS['SLOW']

def get_turn_command(steering_output):
    """Get turn command based on steering output"""
    if abs(steering_output) < STEERING_DEADZONE:
        return 'F'
    elif steering_output < -0.05:  # More responsive thresholds
        return 'L'
    elif steering_output > 0.05:   # More responsive thresholds
        return 'R'
    else:
        return 'F'

def search_behavior(search_counter, last_known_offset):
    """Search pattern when line is lost"""
    if search_counter < 8:
        # Turn towards last known position
        return ('L' if last_known_offset < 0 else 'R'), SPEEDS['SLOW']
    elif search_counter < 16:
        # Try opposite direction
        return ('R' if last_known_offset < 0 else 'L'), SPEEDS['SLOW']
    elif search_counter < 30:
        # Sweep left and right
        return ('L' if (search_counter // 3) % 2 == 0 else 'R'), SPEEDS['SLOW']
    else:
        # Move forward and reset
        return 'F', SPEEDS['SLOW']

# -----------------------------------------------------------------------------
# --- VISUAL EFFECTS ---
# -----------------------------------------------------------------------------
def draw_car_arrow(display_frame):
    """Draw car-style navigation arrow"""
    height, width = display_frame.shape[:2]
    
    # Arrow position
    arrow_base_y = int(height * 0.8)
    arrow_center_x = width // 2
    
    # Calculate arrow direction and color based on current command
    if current_turn_cmd == 'L':
        arrow_angle = -25 + (current_steering * 15)
        arrow_color = (0, 255, 255)  # Cyan for left
        arrow_text = "TURNING LEFT"
    elif current_turn_cmd == 'R':
        arrow_angle = 25 + (current_steering * 15)
        arrow_color = (255, 255, 0)  # Yellow for right
        arrow_text = "TURNING RIGHT"
    else:
        arrow_angle = current_steering * 10
        arrow_color = (0, 255, 0)    # Green for forward
        arrow_text = "GOING FORWARD"
    
    # Draw arrow
    arrow_length = 50
    arrow_end_x = int(arrow_center_x + arrow_length * math.sin(math.radians(arrow_angle)))
    arrow_end_y = int(arrow_base_y - arrow_length * math.cos(math.radians(arrow_angle)))
    
    # Multi-layer arrow with glow
    for thickness in [8, 6, 4, 2]:
        alpha = 0.4 if thickness == 8 else 1.0
        overlay = display_frame.copy()
        cv2.arrowedLine(overlay, (arrow_center_x, arrow_base_y), 
                       (arrow_end_x, arrow_end_y), arrow_color, thickness, tipLength=0.4)
        cv2.addWeighted(overlay, alpha, display_frame, 1-alpha, 0, display_frame)
    
    # Arrow text
    text_size = cv2.getTextSize(arrow_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = arrow_center_x - text_size[0] // 2
    cv2.putText(display_frame, arrow_text, (text_x, arrow_base_y + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, arrow_color, 2)

def draw_line_overlays(display_frame, roi_data, detected_lines, offset, confidence):
    """Draw line detection overlays"""
    height, width = display_frame.shape[:2]
    
    # Draw ROI zones
    for i, roi_info in enumerate(roi_data):
        color = (255, 255, 0) if i == 0 else (255, 165, 0)
        roi_top = roi_info['top']
        roi_height = roi_info['height']
        
        cv2.rectangle(display_frame, (5, roi_top), (width-5, roi_top + roi_height), color, 2)
        cv2.putText(display_frame, f"ROI {i+1}", (10, roi_top + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw detected lines
    if detected_lines:
        for line_info in detected_lines:
            x1, y1, x2, y2 = line_info['coords']
            y1 += roi_data[0]['top']
            y2 += roi_data[0]['top']
            
            # Multi-layer line for visibility
            cv2.line(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 6)
            cv2.line(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
    
    # Draw center line
    center_x = width // 2
    cv2.line(display_frame, (center_x, 0), (center_x, height), (255, 0, 255), 2)
    
    # Draw line target
    if offset is not None:
        target_x = int(center_x + (offset * width // 4))
        target_y = roi_data[0]['top'] + roi_data[0]['height'] // 2
        
        cv2.circle(display_frame, (target_x, target_y), 15, (0, 0, 255), 3)
        cv2.circle(display_frame, (target_x, target_y), 5, (0, 0, 255), -1)
        
        # Line from center to target
        cv2.line(display_frame, (center_x, target_y), (target_x, target_y), (255, 0, 255), 2)

def draw_status_panel(display_frame):
    """Draw status information panel"""
    height, width = display_frame.shape[:2]
    
    # Background
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (5, 5), (300, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, display_frame, 0.2, 0, display_frame)
    
    # Border
    cv2.rectangle(display_frame, (5, 5), (300, 160), (0, 255, 255), 2)
    
    # Status text
    y = 25
    status_color = (0, 255, 0) if "Following" in robot_status else (255, 255, 0)
    
    cv2.putText(display_frame, f"Status: {robot_status}", (10, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    y += 20
    cv2.putText(display_frame, f"Lines: {lines_detected}", (10, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    y += 18
    cv2.putText(display_frame, f"Offset: {current_line_offset:.3f}", (10, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    y += 18
    cv2.putText(display_frame, f"Steering: {current_steering:.3f}", (10, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    y += 18
    cv2.putText(display_frame, f"Confidence: {confidence_score:.1%}", (10, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    y += 18
    cv2.putText(display_frame, f"Black Threshold: {BLACK_THRESHOLD}", (10, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
    y += 18
    
    speed_text = {'s': 'FAST', 'n': 'NORMAL', 'S': 'SLOW', 'T': 'TURN', 'H': 'STOP'}.get(current_speed_cmd, current_speed_cmd)
    turn_text = {'F': 'FORWARD', 'L': 'LEFT', 'R': 'RIGHT'}.get(current_turn_cmd, current_turn_cmd)
    
    cv2.putText(display_frame, f"Cmd: {speed_text}-{turn_text}", (10, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

# -----------------------------------------------------------------------------
# --- FLASK WEB INTERFACE ---
# -----------------------------------------------------------------------------
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🤖 Black Line Following Robot</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {
            --primary: #00ffff;
            --secondary: #ff6b35;
            --success: #00ff88;
            --danger: #ff3366;
            --dark: #0a0e1a;
            --card: #1a1f35;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, var(--dark) 0%, #0f1419 100%);
            color: white;
            min-height: 100vh;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            grid-column: 1 / -1;
            text-align: center;
            margin-bottom: 20px;
        }
        
        .header h1 {
            font-size: 2.5rem;
            background: linear-gradient(45deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .card {
            background: var(--card);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid var(--primary);
            box-shadow: 0 8px 32px rgba(0, 255, 255, 0.2);
        }
        
        .video-container img {
            width: 100%;
            height: auto;
            border-radius: 10px;
        }
        
        .status-grid {
            display: grid;
            gap: 15px;
        }
        
        .status-card {
            background: var(--card);
            border-radius: 12px;
            padding: 15px;
            border: 1px solid var(--primary);
        }
        
        .status-card h3 {
            color: var(--primary);
            margin-bottom: 10px;
        }
        
        .status-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: var(--success);
        }
        
        .connection-status {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        
        .connected {
            background: rgba(0, 255, 136, 0.2);
            border: 1px solid var(--success);
        }
        
        .disconnected {
            background: rgba(255, 51, 102, 0.2);
            border: 1px solid var(--danger);
        }
        
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        .connected .status-dot { background: var(--success); }
        .disconnected .status-dot { background: var(--danger); }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--success));
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        
        @media (max-width: 768px) {
            .dashboard { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>🤖 BLACK LINE FOLLOWING ROBOT</h1>
            <div id="connection-status" class="connection-status disconnected">
                <div class="status-dot"></div>
                <span>ESP32 Status: Checking...</span>
            </div>
        </div>
        
        <div class="card video-container">
            <img src="{{ url_for('video_feed') }}" id="video-feed" alt="Robot Camera Feed">
        </div>
        
        <div class="status-grid">
            <div class="status-card">
                <h3>👁️ Line Detection</h3>
                <div class="status-value" id="lines-count">0</div>
                <div>Lines Detected</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="detection-confidence" style="width: 0%"></div>
                </div>
            </div>
            
            <div class="status-card">
                <h3>📍 Position</h3>
                <div class="status-value" id="line-offset">0.000</div>
                <div>Offset from Center</div>
            </div>
            
            <div class="status-card">
                <h3>🎯 Steering</h3>
                <div class="status-value" id="steering-value">0.000</div>
                <div>PID Output</div>
            </div>
            
            <div class="status-card">
                <h3>🤖 Status</h3>
                <div class="status-value" id="robot-status">Initializing</div>
                <div>Current State</div>
            </div>
            
            <div class="status-card">
                <h3>⚡ Performance</h3>
                <div class="status-value" id="fps-value">0.0 FPS</div>
                <div>Processing Speed</div>
            </div>
            
            <div class="status-card">
                <h3>🔮 Confidence</h3>
                <div class="status-value" id="confidence-value">0.0%</div>
                <div>Detection Confidence</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="confidence-bar" style="width: 0%"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('lines-count').textContent = data.lines_detected || 0;
                    document.getElementById('line-offset').textContent = (data.line_offset || 0).toFixed(3);
                    document.getElementById('steering-value').textContent = (data.steering || 0).toFixed(3);
                    document.getElementById('robot-status').textContent = data.robot_status || 'Unknown';
                    document.getElementById('fps-value').textContent = (data.fps?.toFixed(1) || '0.0') + ' FPS';
                    
                    const conf = (data.confidence || 0) * 100;
                    document.getElementById('confidence-value').textContent = conf.toFixed(1) + '%';
                    document.getElementById('confidence-bar').style.width = conf + '%';
                    document.getElementById('detection-confidence').style.width = Math.min(100, (data.lines_detected || 0) * 33) + '%';
                    
                    const connectionDiv = document.getElementById('connection-status');
                    if (data.robot_status && !data.robot_status.includes('Error')) {
                        connectionDiv.className = 'connection-status connected';
                        connectionDiv.innerHTML = '<div class="status-dot"></div><span>ESP32 Status: Connected</span>';
                    } else {
                        connectionDiv.className = 'connection-status disconnected';
                        connectionDiv.innerHTML = '<div class="status-dot"></div><span>ESP32 Status: Disconnected</span>';
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                });
        }
        
        setInterval(updateStatus, 200);
        updateStatus();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return {
        'line_angle': float(current_line_angle),
        'line_offset': float(current_line_offset),
        'steering': float(current_steering),
        'speed_cmd': current_speed_cmd,
        'turn_cmd': current_turn_cmd,
        'lines_detected': int(lines_detected),
        'robot_status': robot_status,
        'fps': float(fps_current),
        'confidence': float(confidence_score)
    }

def generate_frames():
    global output_frame_flask, frame_lock
    
    while True:
        with frame_lock:
            if output_frame_flask is not None:
                frame = output_frame_flask.copy()
            else:
                frame = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera Initializing...", (10, CAM_H//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(1/REQUESTED_CAM_FPS)

def run_flask():
    logger.info("🌐 Starting web server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)

# -----------------------------------------------------------------------------
# --- MAIN APPLICATION ---
# -----------------------------------------------------------------------------
def main():
    global CAM_W, CAM_H, output_frame_flask, frame_lock
    global current_line_angle, current_line_offset, current_steering
    global current_speed_cmd, current_turn_cmd, lines_detected, robot_status, fps_current
    global confidence_score, detection_method, search_memory
    
    logger.info("🚀 Starting BLACK Line Following Robot...")
    robot_status = "Initializing Camera"
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("❌ Failed to open camera")
        robot_status = "Camera Error"
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_CAM_H)
    cap.set(cv2.CAP_PROP_FPS, REQUESTED_CAM_FPS)
    
    CAM_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    CAM_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"📷 Camera initialized: {CAM_W}x{CAM_H}")
    
    # Initialize controllers
    robot_status = "Connecting to ESP32"
    pid_controller = BlackLinePIDController(PID_KP, PID_KI, PID_KD, PID_INT_MAX)
    esp_comm = ESP32Communicator(ESP32_IP, ESP32_PORT)
    
    # Start web interface
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Performance monitoring
    fps_deque = deque(maxlen=30)
    frame_count = 0
    search_counter = 0
    last_known_offset = 0
    
    # Stability control
    stable_direction_counter = 0
    last_turn_direction = "F"
    last_speed_command = SPEEDS['STOP']
    command_stability_count = 0
    
    logger.info("🤖 BLACK Line Following Robot READY!")
    logger.info("🌐 Web interface: http://0.0.0.0:5000")
    
    try:
        while True:
            loop_start = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("⚠️ Failed to capture frame")
                robot_status = "Camera Read Error"
                continue
            
            if frame.shape[1] != CAM_W or frame.shape[0] != CAM_H:
                frame = cv2.resize(frame, (CAM_W, CAM_H))
            
            display_frame = frame.copy()
            
            # Process frame for black line detection
            processed_frame = preprocess_for_black_lines(frame)
            roi_data = extract_roi_zones(processed_frame)
            
            # Detect black lines
            final_offset, final_angle, detected_lines, confidence = detect_black_lines(roi_data)
            
            # Update global variables
            current_line_offset = final_offset if final_offset is not None else 0.0
            current_line_angle = final_angle if final_angle is not None else 0.0
            lines_detected = len(detected_lines) if detected_lines else 0
            confidence_score = confidence
            
            # Control logic
            if final_offset is not None and confidence > 0.15:  # Even lower confidence threshold
                # Line detected - follow it
                robot_status = f"Following Black Line (Conf: {confidence:.1%})"
                search_counter = 0
                
                # Add to search memory
                search_memory.append(final_offset)
                last_known_offset = final_offset
                
                # Calculate steering
                steering_error = -final_offset
                steering_output = pid_controller.calculate(steering_error)
                current_steering = steering_output
                
                # Determine commands
                current_speed_cmd = get_speed_command(final_offset, confidence, steering_output)
                new_turn_cmd = get_turn_command(steering_output)
                
                # HIGH CONFIDENCE MODE - Be more aggressive!
                if confidence > 0.88:
                    robot_status = f"HIGH CONFIDENCE LINE FOLLOWING! (Conf: {confidence:.1%})"
                    # For very high confidence, reduce stability requirement for faster response
                    stability_requirement = 0  # Immediate response when very confident
                else:
                    stability_requirement = STABILITY_FRAMES
                
                # Stability control - require multiple frames before changing direction
                if new_turn_cmd == last_turn_direction:
                    stable_direction_counter += 1
                else:
                    stable_direction_counter = 0
                
                # Only change turn command if stable for required frames
                if stable_direction_counter >= stability_requirement or abs(final_offset) > 0.3:
                    current_turn_cmd = new_turn_cmd
                    last_turn_direction = new_turn_cmd
                else:
                    current_turn_cmd = last_turn_direction
                
            else:
                # No line detected - search
                robot_status = f"Searching for Black Line ({search_counter})"
                search_counter += 1
                
                # Search behavior
                current_turn_cmd, current_speed_cmd = search_behavior(search_counter, last_known_offset)
                current_steering = 0.0
                
                # Reset search counter if too high
                if search_counter > 40:
                    search_counter = 0
            
            # Send commands to ESP32
            esp_comm.send_command(current_speed_cmd, current_turn_cmd)
            
            # Draw visual overlays
            draw_line_overlays(display_frame, roi_data, detected_lines, final_offset, confidence)
            draw_car_arrow(display_frame)
            draw_status_panel(display_frame)
            
            # Add debug view in corner
            debug_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
            debug_small = cv2.resize(debug_frame, (CAM_W//4, CAM_H//4))
            display_frame[10:10+CAM_H//4, CAM_W-CAM_W//4-10:CAM_W-10] = debug_small
            
            # Connection status
            connection_color = (0, 255, 0) if esp_comm.sock else (0, 0, 255)
            connection_text = "ESP32: CONNECTED" if esp_comm.sock else "ESP32: DISCONNECTED"
            cv2.putText(display_frame, connection_text, (10, CAM_H - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, connection_color, 2)
            
            # Update output frame
            with frame_lock:
                output_frame_flask = display_frame.copy()
            
            # Performance monitoring
            loop_time = time.time() - loop_start
            fps = 1.0 / loop_time if loop_time > 0 else 0
            fps_deque.append(fps)
            fps_current = sum(fps_deque) / len(fps_deque) if fps_deque else 0
            
            frame_count += 1
            if frame_count % 60 == 0:
                logger.info(f"📊 FPS: {fps_current:.1f} | Status: {robot_status} | "
                           f"Lines: {lines_detected} | Confidence: {confidence_score:.1%} | "
                           f"Offset: {current_line_offset:.3f} | Cmd: {current_speed_cmd}-{current_turn_cmd}")
    
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down...")
        robot_status = "Shutting Down"
    except Exception as e:
        logger.error(f"💥 Error: {e}", exc_info=True)
        robot_status = "Error"
    finally:
        logger.info("🧹 Cleanup...")
        try:
            esp_comm.close()
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass
        logger.info("✅ Cleanup complete")

if __name__ == "__main__":
    main()