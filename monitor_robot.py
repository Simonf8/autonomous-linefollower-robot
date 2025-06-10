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
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from enum import Enum

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("YOLO11n available for object detection")
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available. Install with: pip install ultralytics")

# Try to import text-to-speech
try:
    import subprocess
    import os
    import queue
    # Check if Piper TTS is available
    if os.path.exists('./piper/piper') and os.path.exists('./voices/en_US-lessac-medium.onnx'):
        TTS_AVAILABLE = True
        TTS_ENGINE = 'piper'
        print("Piper TTS available")
    else:
        try:
            import pyttsx3
            TTS_AVAILABLE = True
            TTS_ENGINE = 'pyttsx3'
            print("pyttsx3 TTS available")
        except ImportError:
            TTS_AVAILABLE = False
            TTS_ENGINE = None
except ImportError:
    TTS_AVAILABLE = False
    TTS_ENGINE = None
    print("Text-to-speech not available")

# Configuration
ESP32_IP = '192.168.128.117'  
ESP32_PORT = 1234
CAMERA_WIDTH, CAMERA_HEIGHT = 730, 420
CAMERA_FPS = 10  # Increased for better responsiveness

# Enhanced line detection parameters
BLACK_THRESHOLD = 80
ADAPTIVE_THRESHOLD_ENABLED = True
BLUR_SIZE = 5
MIN_CONTOUR_AREA = 50
MORPH_KERNEL_SIZE = 3

# Multi-zone detection with better ratios
ZONE_BOTTOM_HEIGHT = 0.30   # Bottom 30% for primary line following
ZONE_MIDDLE_HEIGHT = 0.25   # Middle 25% for predictive tracking
ZONE_TOP_HEIGHT = 0.45      # Top 45% for early object detection

# Corner detection improvements
CORNER_DETECTION_ENABLED = True
CORNER_ANGLE_THRESHOLD = 25  # Reduced from 35 - less sensitive
CORNER_SMOOTHING_FACTOR = 0.7
CORNER_PREDICTION_FRAMES = 5

# Object detection parameters
OBJECT_DETECTION_ENABLED = True
USE_YOLO = True
YOLO_MODEL_SIZE = "yolo11n.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.4
YOLO_NMS_THRESHOLD = 0.45

# Enhanced PID with feedforward
KP = 0.30  # Reduced from 0.40 for less aggressive response
KI = 0.005  # Reduced from 0.008 to prevent windup
KD = 0.20  # Reduced from 0.30 for less jerkiness
KF = 0.10  # Reduced feedforward gain
MAX_INTEGRAL = 2.0  # Reduced from 2.5

# Smooth curve parameters
CURVE_LOOKAHEAD_DISTANCE = 0.8  # meters
CURVE_SMOOTHING_POINTS = 20
BEZIER_CURVE_ENABLED = True

# Robot physical parameters
ROBOT_WIDTH = 0.20  # meters
ROBOT_LENGTH = 0.25  # meters
CAMERA_OFFSET = 0.05  # meters from front

# Commands for ESP32
class RobotCommand(Enum):
    FORWARD = 'FORWARD'
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'
    STOP = 'STOP'
    GENTLE_LEFT = 'GENTLE_LEFT'
    GENTLE_RIGHT = 'GENTLE_RIGHT'
    SHARP_LEFT = 'SHARP_LEFT'
    SHARP_RIGHT = 'SHARP_RIGHT'

# Avoidance states
class AvoidanceState(Enum):
    NONE = 'none'
    PLANNING = 'planning'
    EXECUTING = 'executing'
    RETURNING = 'returning'

@dataclass
class DetectedObject:
    position: Tuple[float, float]
    size: Tuple[float, float]
    confidence: float
    class_name: str
    distance: float
    angle: float

@dataclass
class PathPoint:
    x: float
    y: float
    heading: float
    curvature: float

# Global state
robot_state = {
    'status': 'Initializing',
    'line_detected': False,
    'line_offset': 0.0,
    'confidence': 0.0,
    'command': RobotCommand.STOP,
    'fps': 0.0,
    'esp_connected': False,
    'avoidance_state': AvoidanceState.NONE,
    'objects': [],
    'path': [],
    'stats': {
        'uptime': 0,
        'frames': 0,
        'corners': 0,
        'objects_avoided': 0
    }
}

# Thread-safe frame buffer
frame_buffer = {
    'frame': None,
    'lock': threading.Lock()
}

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("EnhancedLineFollower")

# -----------------------------------------------------------------------------
# --- Image Processing Pipeline ---
# -----------------------------------------------------------------------------

class ImageProcessor:
    def __init__(self):
        self.adaptive_threshold = BLACK_THRESHOLD
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
        )
        
    def preprocess(self, frame):
        """Enhanced preprocessing with adaptive thresholding"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (BLUR_SIZE, BLUR_SIZE), 0)
        
        # Adaptive thresholding if enabled
        if ADAPTIVE_THRESHOLD_ENABLED:
            # Calculate histogram to adjust threshold
            hist = cv2.calcHist([blurred], [0], None, [256], [0, 256])
            # Find the valley between black line and background
            self.adaptive_threshold = self._find_threshold_valley(hist)
        
        # Binary threshold
        _, binary = cv2.threshold(blurred, self.adaptive_threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations to clean up
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.morph_kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.morph_kernel)
        
        return binary, blurred
    
    def _find_threshold_valley(self, hist):
        """Find optimal threshold using valley detection"""
        hist_smooth = cv2.GaussianBlur(hist, (1, 5), 0).flatten()
        
        # Find local minima between peaks
        valleys = []
        for i in range(10, 200):  # Search range
            if hist_smooth[i] < hist_smooth[i-1] and hist_smooth[i] < hist_smooth[i+1]:
                valleys.append((i, hist_smooth[i]))
        
        if valleys:
            # Return the most prominent valley
            best_valley = min(valleys, key=lambda x: x[1])
            return int(best_valley[0] * 0.9)  # Slightly lower for safety
        
        return BLACK_THRESHOLD

    def detect_line_features(self, binary_roi, zone_name="unknown"):
        """Enhanced line detection with feature extraction"""
        if binary_roi.size == 0:
            return None
        
        height, width = binary_roi.shape
        contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter and analyze contours
        line_features = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA:
                continue
            
            # Fit a line to the contour
            if len(contour) >= 5:
                (vx, vy, x, y) = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                angle = float(np.arctan2(vy[0], vx[0]) * 180 / np.pi)
            else:
                angle = 0.0
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = x + w // 2
                cy = y + h // 2
            
            # Calculate confidence based on multiple factors
            aspect_ratio = w / h if h > 0 else 0
            solidity = area / (w * h) if w * h > 0 else 0
            
            # Line-like features have high aspect ratio and good solidity
            if zone_name == "bottom":
                confidence = min(1.0, (area / (width * height * 0.1)) * solidity)
            else:
                confidence = min(1.0, (area / (width * height * 0.15)) * solidity * 0.8)
            
            line_features.append({
                'center': (cx, cy),
                'angle': angle,
                'confidence': confidence,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'contour': contour,
                'bbox': (x, y, w, h)
            })
        
        if not line_features:
            return None
        
        # Select the best line feature
        best_feature = max(line_features, key=lambda f: f['confidence'] * f['area'])
        return best_feature

# -----------------------------------------------------------------------------
# --- Path Planning and Smoothing ---
# -----------------------------------------------------------------------------

class PathPlanner:
    def __init__(self):
        self.path_history = deque(maxlen=50)
        self.bezier_points = []
        
    def generate_bezier_curve(self, start, control1, control2, end, num_points=20):
        """Generate smooth Bezier curve for path planning"""
        t = np.linspace(0, 1, num_points)
        curve_points = []
        
        for i in range(num_points):
            # Cubic Bezier formula
            point = (1-t[i])**3 * start + \
                   3*(1-t[i])**2*t[i] * control1 + \
                   3*(1-t[i])*t[i]**2 * control2 + \
                   t[i]**3 * end
            curve_points.append(point)
        
        return np.array(curve_points)
    
    def plan_avoidance_path(self, obstacle: DetectedObject, current_pos, target_pos):
        """Plan smooth avoidance path using Bezier curves"""
        # Calculate avoidance waypoints
        avoidance_distance = max(obstacle.size[0], obstacle.size[1]) * 2 + ROBOT_WIDTH
        
        # Determine avoidance direction (go around the side with more space)
        if obstacle.position[0] > 0:  # Obstacle on right, go left
            avoidance_side = -1
        else:  # Obstacle on left, go right
            avoidance_side = 1
        
        # Create control points for Bezier curve
        control1 = current_pos + np.array([0.2, avoidance_side * avoidance_distance * 0.5])
        control2 = target_pos + np.array([-0.2, avoidance_side * avoidance_distance * 0.5])
        
        # Generate smooth path
        path = self.generate_bezier_curve(current_pos, control1, control2, target_pos)
        
        return path
    
    def smooth_path(self, raw_points, smoothing_factor=0.5):
        """Apply path smoothing using moving average"""
        if len(raw_points) < 3:
            return raw_points
        
        smoothed = []
        for i in range(len(raw_points)):
            if i == 0 or i == len(raw_points) - 1:
                smoothed.append(raw_points[i])
            else:
                prev_point = raw_points[i-1]
                curr_point = raw_points[i]
                next_point = raw_points[i+1]
                
                smooth_point = (
                    smoothing_factor * curr_point +
                    (1 - smoothing_factor) * 0.5 * (prev_point + next_point)
                )
                smoothed.append(smooth_point)
        
        return smoothed

# -----------------------------------------------------------------------------
# --- Enhanced PID Controller ---
# -----------------------------------------------------------------------------

class EnhancedPID:
    def __init__(self, kp, ki, kd, kf, max_integral=5.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.kf = kf  # Feedforward gain
        self.max_integral = max_integral
        
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        
        # Anti-windup and performance tracking
        self.integral_active = True
        self.performance_window = deque(maxlen=50)
        
    def calculate(self, error, feedforward=0.0):
        """Calculate PID output with optional feedforward"""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        dt = max(dt, 0.001)
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        if self.integral_active:
            self.integral += error * dt
            self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        i_term = self.ki * self.integral
        
        # Derivative term with filtering
        derivative = (error - self.previous_error) / dt
        # Simple low-pass filter on derivative
        if hasattr(self, 'filtered_derivative'):
            self.filtered_derivative = 0.7 * self.filtered_derivative + 0.3 * derivative
        else:
            self.filtered_derivative = derivative
        d_term = self.kd * self.filtered_derivative
        
        # Feedforward term
        f_term = self.kf * feedforward
        
        # Total output
        output = p_term + i_term + d_term + f_term
        
        # Anti-windup: disable integral if output is saturated
        if abs(output) > 1.0:
            self.integral_active = False
        else:
            self.integral_active = True
        
        # Track performance
        self.performance_window.append(abs(error))
        
        self.previous_error = error
        return np.clip(output, -1.0, 1.0)
    
    def get_performance_metric(self):
        """Get current controller performance"""
        if len(self.performance_window) > 0:
            return np.mean(self.performance_window)
        return 0.0
    
    def reset(self):
        """Reset controller state"""
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        self.integral_active = True

# -----------------------------------------------------------------------------
# --- Robot Control System ---
# -----------------------------------------------------------------------------

class RobotController:
    def __init__(self, esp_ip, esp_port):
        self.esp_ip = esp_ip
        self.esp_port = esp_port
        self.socket = None
        self.connected = False
        self.last_command = None
        self.last_command_time = 0
        
        self.pid = EnhancedPID(KP, KI, KD, KF, MAX_INTEGRAL)
        self.image_processor = ImageProcessor()
        self.path_planner = PathPlanner()
        
        # State tracking
        self.line_history = deque(maxlen=10)
        self.command_history = deque(maxlen=5)
        self.connection_failures = 0
        self.last_connection_check = 0
        
    def connect(self):
        """Establish connection to ESP32"""
        try:
            if self.socket:
                self.socket.close()
            
            self.socket = socket.create_connection((self.esp_ip, self.esp_port), timeout=5)
            self.socket.settimeout(1.0)  # Reduced timeout to 1 second
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)  # Enable keepalive
            self.socket.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)  # Disable Nagle's algorithm
            self.connected = True
            robot_state['esp_connected'] = True
            logger.info(f"🔗 Connected to ESP32 at {self.esp_ip}:{self.esp_port}")
            return True
        except Exception as e:
            self.connected = False
            robot_state['esp_connected'] = False
            logger.error(f"❌ ESP32 connection failed: {e}")
            return False
    
    def send_command(self, command: RobotCommand):
        """Send command to ESP32 with error handling"""
        if not self.connected and not self.connect():
            logger.error("❌ Failed to connect to ESP32")
            robot_state['status'] = "ESP32 DISCONNECTED"
            return False
        
        try:
            # Send command with timestamp for debugging
            timestamp = time.time()
            cmd_str = f"{command.value}\n"
            
            # Use sendall with explicit error handling
            self.socket.sendall(cmd_str.encode())
            
            # Log all commands for debugging the stopping issue
            logger.info(f"📤 SENT: {command.value} at {timestamp:.3f}")
            
            self.last_command = command
            self.last_command_time = timestamp
            robot_state['command'] = command
            
            # Update status to show last successful command
            if robot_state['line_detected']:
                robot_state['status'] = f"Following line - {command.value}"
            else:
                robot_state['status'] = f"Searching - {command.value}"
                
            return True
            
        except socket.timeout:
            logger.error("⏰ ESP32 socket timeout - reconnecting")
            self.connected = False
            robot_state['esp_connected'] = False
            robot_state['status'] = "ESP32 TIMEOUT"
            return False
                    
        except (socket.error, ConnectionResetError, BrokenPipeError) as e:
            logger.error(f"🔌 ESP32 socket error: {e}")
            self.connected = False
            robot_state['esp_connected'] = False
            robot_state['status'] = f"ESP32 ERROR: {str(e)[:20]}"
            return False
                
        except Exception as e:
            logger.error(f"💥 Unexpected ESP32 error: {e}")
            self.connected = False
            robot_state['esp_connected'] = False
            robot_state['status'] = f"UNKNOWN ERROR: {str(e)[:20]}"
            return False
    
    def check_connection_health(self):
        """Periodically check connection health"""
        current_time = time.time()
        if current_time - self.last_connection_check > 5.0:  # Check every 5 seconds
            self.last_connection_check = current_time
            
            if self.connected:
                try:
                    # Send a test ping to verify connection
                    self.socket.send(b'')  # Empty send to test connection
                except:
                    logger.warning("Connection health check failed, reconnecting...")
                    self.connected = False
                    robot_state['esp_connected'] = False
                    self.connect()
    
    def process_frame(self, frame):
        """Main processing pipeline"""
        # Check connection health periodically - DISABLED to prevent delays
        # self.check_connection_health()
        
        # Preprocess image
        binary, blurred = self.image_processor.preprocess(frame)
        
        height, width = binary.shape
        
        # Define zones
        zones = {
            'bottom': (int(height * (1 - ZONE_BOTTOM_HEIGHT)), height),
            'middle': (int(height * (1 - ZONE_BOTTOM_HEIGHT - ZONE_MIDDLE_HEIGHT)), 
                      int(height * (1 - ZONE_BOTTOM_HEIGHT))),
            'top': (int(height * (1 - ZONE_BOTTOM_HEIGHT - ZONE_MIDDLE_HEIGHT - ZONE_TOP_HEIGHT)),
                   int(height * (1 - ZONE_BOTTOM_HEIGHT - ZONE_MIDDLE_HEIGHT)))
        }
        
        # Extract ROIs
        bottom_roi = binary[zones['bottom'][0]:zones['bottom'][1], :]
        middle_roi = binary[zones['middle'][0]:zones['middle'][1], :]
        
        # Detect line features
        bottom_features = self.image_processor.detect_line_features(bottom_roi, "bottom")
        middle_features = self.image_processor.detect_line_features(middle_roi, "middle")
        
        # Determine primary line position
        line_detected = False
        line_position = None
        confidence = 0.0
        
        # Try bottom zone first (most reliable)
        if bottom_features and bottom_features['confidence'] > 0.15:  # Even more permissive
            line_position = bottom_features['center'][0]
            confidence = bottom_features['confidence']
            line_detected = True
        # If no bottom line, try middle zone
        elif middle_features and middle_features['confidence'] > 0.2:  # More permissive  
            line_position = middle_features['center'][0]
            confidence = middle_features['confidence'] * 0.9  # Less penalty
            line_detected = True
        # If still no line but we have ANY features, try to use them
        elif bottom_features and bottom_features['confidence'] > 0.05:
            line_position = bottom_features['center'][0]
            confidence = bottom_features['confidence'] * 0.5
            line_detected = True
            robot_state['status'] = "Weak line signal - following"
        elif middle_features and middle_features['confidence'] > 0.1:
            line_position = middle_features['center'][0]
            confidence = middle_features['confidence'] * 0.5
            line_detected = True
            robot_state['status'] = "Weak line signal - following"
        
        # Update line history for smoothing
        if line_detected:
            self.line_history.append(line_position)
            # Use weighted average of recent positions
            if len(self.line_history) >= 3:
                weights = np.array([0.5, 0.3, 0.2])[-len(self.line_history):]
                positions = list(self.line_history)[-len(weights):]
                line_position = np.average(positions, weights=weights)
        
        # Calculate offset and steering
        if line_detected:
            # Reset line lost tracking when line is found
            if hasattr(self, 'line_lost_time'):
                self.line_lost_time = 0
            
            center_x = width / 2
            offset = (line_position - center_x) / center_x
            robot_state['line_offset'] = offset
            robot_state['confidence'] = confidence
            robot_state['status'] = f"Following line (confidence: {confidence:.2f})"
            
            # Calculate steering with PID
            steering_error = -offset
            
            # Add feedforward for corners
            feedforward = 0.0
            if middle_features and bottom_features:
                # Predict curvature from line angle difference
                angle_diff = middle_features['angle'] - bottom_features['angle']
                if abs(angle_diff) > CORNER_ANGLE_THRESHOLD:
                    feedforward = np.sign(angle_diff) * 0.3
                    robot_state['status'] = f"Corner detected ({angle_diff:.1f}°)"
                    robot_state['stats']['corners'] += 1
            
            steering = self.pid.calculate(steering_error, feedforward)
            
            # Convert steering to command
            command = self._steering_to_command(steering, confidence)
            
        else:
            # Track line loss duration
            current_time = time.time()
            if not hasattr(self, 'line_lost_time') or self.line_lost_time == 0:
                self.line_lost_time = current_time
            
            line_lost_duration = current_time - self.line_lost_time
            
            # If line lost for less than 1 second, use predictive following
            if line_lost_duration < 1.0 and len(self.line_history) > 0:
                # Use last known line position and direction for predictive following
                last_position = self.line_history[-1]
                center_x = width / 2
                offset = (last_position - center_x) / center_x
                
                # Continue with same steering as before but reduced magnitude
                steering_error = -offset * 0.7  # Reduced steering for prediction
                steering = self.pid.calculate(steering_error, 0)
                command = self._steering_to_command(steering, 0.3)  # Lower confidence
                
                robot_state['line_detected'] = False
                robot_state['confidence'] = 0.3  # Show we're using prediction
                robot_state['status'] = f"Predictive following ({line_lost_duration:.1f}s)"
                
            else:
                # Line not detected for too long - search mode
                robot_state['line_detected'] = False
                robot_state['confidence'] = 0.0
                robot_state['status'] = f"Searching for line... ({line_lost_duration:.1f}s)"
                command = self._search_for_line()
        
        robot_state['line_detected'] = line_detected
        
        # Send command
        self.send_command(command)
        
        # Return processed data for visualization
        return {
            'binary': binary,
            'zones': zones,
            'bottom_features': bottom_features,
            'middle_features': middle_features,
            'line_position': line_position if line_detected else None,
            'confidence': confidence
        }
    
    def _steering_to_command(self, steering, confidence):
        """Convert steering value to robot command"""
        # Dead zone
        if abs(steering) < 0.03:  # Reduced dead zone for more responsive turning
            return RobotCommand.FORWARD
        
        # Variable turn rates based on steering magnitude
        if abs(steering) > 0.5:  # Reduced from 0.6 for earlier sharp turns
            return RobotCommand.SHARP_LEFT if steering > 0 else RobotCommand.SHARP_RIGHT
        elif abs(steering) > 0.25:  # Reduced from 0.3 for earlier turning
            return RobotCommand.LEFT if steering > 0 else RobotCommand.RIGHT
        else:
            return RobotCommand.GENTLE_LEFT if steering > 0 else RobotCommand.GENTLE_RIGHT
    
    def _search_for_line(self):
        """Intelligent search pattern when line is lost"""
        # If we have recent line history, use it to predict direction
        if len(self.line_history) > 0:
            last_position = self.line_history[-1]
            center = 365  # Camera center
            
            # If line was significantly off center, search in that direction
            if abs(last_position - center) > 50:
                if last_position < center - 50:  # Line was on left
                    return RobotCommand.GENTLE_LEFT
                else:  # Line was on right
                    return RobotCommand.GENTLE_RIGHT
            else:
                # Line was near center, continue forward briefly
                return RobotCommand.FORWARD
        
        # No history - continue forward (most conservative)
        return RobotCommand.FORWARD

# -----------------------------------------------------------------------------
# --- Flask Web Interface ---
# -----------------------------------------------------------------------------

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Line Following Robot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: #fff;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .video-feed {
            width: 100%;
            max-width: 800px;
            margin: 0 auto;
            display: block;
            border-radius: 10px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .stat-card {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
        }
        .stat-label {
            color: #888;
            margin-top: 5px;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-left: 10px;
        }
        .status-online { background: #4CAF50; }
        .status-offline { background: #f44336; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Enhanced Line Following Robot
                <span id="esp-status" class="status-indicator status-offline"></span>
            </h1>
        </div>
        
        <img src="/video_feed" class="video-feed" alt="Robot Camera Feed">
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value" id="status">Initializing</div>
                <div class="stat-label">Status</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="command">STOP</div>
                <div class="stat-label">Command</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="confidence">0%</div>
                <div class="stat-label">Line Confidence</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="fps">0.0</div>
                <div class="stat-label">FPS</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="offset">0.00</div>
                <div class="stat-label">Line Offset</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="corners">0</div>
                <div class="stat-label">Corners Detected</div>
            </div>
        </div>
    </div>
    
    <script>
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status').textContent = data.status;
                    document.getElementById('command').textContent = data.command;
                    document.getElementById('confidence').textContent = 
                        Math.round(data.confidence * 100) + '%';
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('offset').textContent = data.offset.toFixed(2);
                    document.getElementById('corners').textContent = data.stats.corners;
                    
                    const espIndicator = document.getElementById('esp-status');
                    espIndicator.className = 'status-indicator status-' + 
                        (data.esp_connected ? 'online' : 'offline');
                });
        }
        
        setInterval(updateStatus, 500);
        updateStatus();
    </script>
</body>
</html>
    ''')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_buffer['lock']:
                if frame_buffer['frame'] is not None:
                    frame = frame_buffer['frame'].copy()
                else:
                    frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.03)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    return jsonify({
        'status': robot_state['status'],
        'command': robot_state['command'].value,
        'offset': robot_state['line_offset'],
        'confidence': robot_state['confidence'],
        'fps': robot_state['fps'],
        'esp_connected': robot_state['esp_connected'],
        'line_detected': robot_state['line_detected'],
        'stats': robot_state['stats']
    })

# -----------------------------------------------------------------------------
# --- Visualization ---
# -----------------------------------------------------------------------------

def draw_visualization(frame, processing_data):
    """Draw enhanced visualization on frame"""
    if not processing_data:
        return frame
    
    height, width = frame.shape[:2]
    viz_frame = frame.copy()
    
    # Draw zones
    zones = processing_data.get('zones', {})
    zone_colors = {
        'bottom': (0, 255, 255),
        'middle': (0, 255, 0),
        'top': (255, 0, 0)
    }
    
    for zone_name, (y1, y2) in zones.items():
        color = zone_colors.get(zone_name, (255, 255, 255))
        cv2.rectangle(viz_frame, (0, y1), (width, y2), color, 2)
        cv2.putText(viz_frame, zone_name.upper(), (5, y1 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw detected line
    if processing_data.get('line_position') is not None:
        line_x = int(processing_data['line_position'])
        cv2.line(viz_frame, (line_x, 0), (line_x, height), (255, 0, 255), 3)
        
        # Draw offset indicator
        center_x = width // 2
        cv2.line(viz_frame, (center_x, height - 50), (line_x, height - 50), (255, 255, 0), 5)
    
    # Draw status info
    status_text = [
        f"Status: {robot_state['status']}",
        f"Command: {robot_state['command'].value}",
        f"Confidence: {robot_state['confidence']:.2f}",
        f"FPS: {robot_state['fps']:.1f}"
    ]
    
    for i, text in enumerate(status_text):
        cv2.putText(viz_frame, text, (10, 30 + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return viz_frame

# -----------------------------------------------------------------------------
# --- Main Application ---
# -----------------------------------------------------------------------------

def main():
    """Main application entry point"""
    logger.info("Starting Enhanced Line Following Robot")
    
    # Initialize robot controller
    controller = RobotController(ESP32_IP, ESP32_PORT)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open camera 0, trying camera 1")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            logger.error("Failed to open any camera")
            return
    
    # Configure camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Get actual settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Camera initialized: {actual_width}x{actual_height}")
    
    # Start Flask server in background
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False), daemon=True)
    flask_thread.start()
    logger.info("Web interface available at http://localhost:5000")
    
    # Connect to ESP32
    controller.connect()
    
    # FPS tracking
    fps_counter = 0
    fps_timer = time.time()
    
    # Main loop
    try:
        robot_state['stats']['start_time'] = time.time()
        
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                continue
            
            # Process frame
            processing_data = controller.process_frame(frame)
            
            # Draw visualization
            viz_frame = draw_visualization(frame, processing_data)
            
            # Update frame buffer
            with frame_buffer['lock']:
                frame_buffer['frame'] = viz_frame
            
            # Update FPS
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                robot_state['fps'] = fps_counter
                fps_counter = 0
                fps_timer = time.time()
            
            # Update stats
            robot_state['stats']['frames'] += 1
            robot_state['stats']['uptime'] = time.time() - robot_state['stats']['start_time']
            
            # Small delay to prevent CPU overload - reduced for better responsiveness
            time.sleep(0.005)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        # Cleanup
        controller.send_command(RobotCommand.STOP)
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Cleanup complete")

if __name__ == "__main__":
    main()