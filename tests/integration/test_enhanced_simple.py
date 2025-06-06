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

# Try to import YOLO - fallback gracefully if not available
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
    print("✅ YOLO available - enhanced detection enabled")
except ImportError:
    print("⚠️ YOLO not available - using improved contour detection")
    YOLO_AVAILABLE = False

# -----------------------------------------------------------------------------
# --- CONFIGURATION ---
# -----------------------------------------------------------------------------
ESP32_IP = '192.168.53.117'  # Change this to your ESP32's IP address
ESP32_PORT = 1234
CAMERA_WIDTH, CAMERA_HEIGHT = 320, 240
CAMERA_FPS = 15

# Enhanced image processing parameters
BLACK_THRESHOLD = 60
BLUR_SIZE = 5
MIN_CONTOUR_AREA = 150

# Multi-zone detection
ZONE_BOTTOM_HEIGHT = 0.35   # Bottom 35% for line detection
ZONE_MIDDLE_HEIGHT = 0.25   # Middle 25% for prediction
ZONE_TOP_HEIGHT = 0.40      # Top 40% for obstacles

# Improved obstacle detection parameters
OBSTACLE_DETECTION_ENABLED = True
OBSTACLE_MIN_AREA = 1500  # Larger minimum area
OBSTACLE_MIN_WIDTH_RATIO = 0.15  # Minimum width to be considered blocking
OBSTACLE_MAX_DISTANCE_RATIO = 0.4  # Maximum distance from center to care about
AVOIDANCE_DURATION = 10  # Frames to avoid
AVOIDANCE_COOLDOWN = 20  # Frames before can avoid again

# Advanced filtering parameters
OBSTACLE_CONFIDENCE_THRESHOLD = 0.6
MOTION_DETECTION_ENABLED = True
MOTION_THRESHOLD = 1000  # Pixels that must change for motion

# PID parameters - enhanced for better response
KP = 0.8
KI = 0.03
KD = 0.15
MAX_INTEGRAL = 4.0

# Commands
COMMANDS = {'FORWARD': 'FORWARD', 'LEFT': 'LEFT', 'RIGHT': 'RIGHT', 'STOP': 'STOP', 
           'AVOID_LEFT': 'AVOID_LEFT', 'AVOID_RIGHT': 'AVOID_RIGHT'}

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

# Enhanced obstacle tracking
obstacle_detected = False
obstacle_tracker = {'bbox': None, 'confidence': 0.0, 'frames_detected': 0}
avoidance_counter = 0
avoidance_direction = None
avoidance_cooldown_counter = 0
last_frame = None  # For motion detection

# Statistics
robot_stats = {
    'uptime': 0,
    'total_frames': 0,
    'obstacles_detected': 0,
    'avoidance_maneuvers': 0,
    'false_positives_filtered': 0,
    'start_time': time.time()
}

# Logging
logging.basicConfig(level=logging.INFO, 
                   format='[%(asctime)s] %(levelname)s: %(message)s', 
                   datefmt='%H:%M:%S')
logger = logging.getLogger("EnhancedLineFollower")

# -----------------------------------------------------------------------------
# --- ENHANCED OBSTACLE DETECTION ---
# -----------------------------------------------------------------------------
class EnhancedObstacleDetector:
    def __init__(self):
        self.yolo_model = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.obstacle_history = deque(maxlen=5)
        
        if YOLO_AVAILABLE:
            try:
                logger.info("🔄 Loading YOLO model...")
                self.yolo_model = YOLO('yolov5n.pt')  # Nano version for speed
                # Warm up
                dummy_img = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
                _ = self.yolo_model(dummy_img, verbose=False)
                logger.info("✅ YOLO model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ YOLO loading failed: {e}")
                self.yolo_model = None
        
        logger.info(f"🔍 Obstacle detector initialized (YOLO: {'Yes' if self.yolo_model else 'No'})")
    
    def detect_obstacles(self, frame):
        """Enhanced obstacle detection with multiple methods"""
        obstacles = []
        
        # Method 1: YOLO detection (if available)
        if self.yolo_model:
            yolo_obstacles = self._yolo_detection(frame)
            obstacles.extend(yolo_obstacles)
        
        # Method 2: Enhanced contour detection
        contour_obstacles = self._enhanced_contour_detection(frame)
        obstacles.extend(contour_obstacles)
        
        # Method 3: Motion-based detection
        if MOTION_DETECTION_ENABLED:
            motion_obstacles = self._motion_detection(frame)
            obstacles.extend(motion_obstacles)
        
        # Filter and combine obstacles
        filtered_obstacles = self._filter_obstacles(obstacles, frame)
        
        return filtered_obstacles
    
    def _yolo_detection(self, frame):
        """YOLO-based obstacle detection"""
        try:
            # Relevant obstacle classes for line following robot
            obstacle_classes = [0, 1, 2, 3, 5, 7, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
            
            results = self.yolo_model(frame, conf=0.4, iou=0.5, verbose=False)
            obstacles = []
            
            if results and len(results) > 0:
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes
                        for i in range(len(boxes)):
                            cls = int(boxes.cls[i])
                            conf = float(boxes.conf[i])
                            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                            
                            if cls in obstacle_classes and conf > 0.4:
                                width = x2 - x1
                                height = y2 - y1
                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                
                                # Check if in obstacle detection zone
                                if center_y < frame.shape[0] * ZONE_TOP_HEIGHT:
                                    obstacles.append({
                                        'type': 'yolo',
                                        'class': cls,
                                        'confidence': conf,
                                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                        'center': (center_x, center_y),
                                        'area': width * height,
                                        'width_ratio': width / frame.shape[1]
                                    })
            
            return obstacles
            
        except Exception as e:
            logger.debug(f"YOLO detection error: {e}")
            return []
    
    def _enhanced_contour_detection(self, frame):
        """Improved contour-based obstacle detection"""
        try:
            height, width = frame.shape[:2]
            
            # Focus on obstacle detection zone
            detect_height = int(height * ZONE_TOP_HEIGHT)
            roi = frame[0:detect_height, :]
            
            # Enhanced preprocessing
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Apply multiple filtering techniques
            # 1. Bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # 2. Adaptive thresholding for varying lighting
            adaptive_thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                   cv2.THRESH_BINARY, 11, 2)
            
            # 3. Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            obstacles = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area > OBSTACLE_MIN_AREA:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate properties
                    center_x = x + w/2
                    center_y = y + h/2
                    width_ratio = w / width
                    aspect_ratio = h / w if w > 0 else 0
                    
                    # Filter based on shape and position
                    distance_from_center = abs(center_x - width/2) / (width/2)
                    
                    # Enhanced filtering criteria
                    if (width_ratio > OBSTACLE_MIN_WIDTH_RATIO and 
                        distance_from_center < OBSTACLE_MAX_DISTANCE_RATIO and
                        aspect_ratio > 0.3 and aspect_ratio < 3.0 and  # Reasonable aspect ratio
                        center_y > detect_height * 0.2):  # Not too high in frame
                        
                        # Calculate confidence based on size and shape
                        size_confidence = min(area / (width * detect_height * 0.1), 1.0)
                        shape_confidence = 1.0 - abs(aspect_ratio - 1.0)  # Prefer square-ish objects
                        position_confidence = 1.0 - distance_from_center
                        
                        overall_confidence = (size_confidence * 0.5 + 
                                            shape_confidence * 0.3 + 
                                            position_confidence * 0.2)
                        
                        if overall_confidence > OBSTACLE_CONFIDENCE_THRESHOLD:
                            obstacles.append({
                                'type': 'contour',
                                'confidence': overall_confidence,
                                'bbox': (x, y, x + w, y + h),
                                'center': (center_x, center_y),
                                'area': area,
                                'width_ratio': width_ratio,
                                'aspect_ratio': aspect_ratio
                            })
            
            return obstacles
            
        except Exception as e:
            logger.debug(f"Contour detection error: {e}")
            return []
    
    def _motion_detection(self, frame):
        """Motion-based obstacle detection"""
        global last_frame
        
        try:
            if last_frame is None:
                last_frame = frame.copy()
                return []
            
            # Calculate frame difference
            diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 
                              cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY))
            
            # Threshold and find motion areas
            _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(frame)
            
            # Combine motion detection with background subtraction
            combined_mask = cv2.bitwise_and(motion_mask, fg_mask)
            
            # Find contours in motion areas
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            obstacles = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area > MOTION_THRESHOLD:
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w/2
                    center_y = y + h/2
                    
                    # Only consider motion in obstacle detection zone
                    if center_y < frame.shape[0] * ZONE_TOP_HEIGHT:
                        obstacles.append({
                            'type': 'motion',
                            'confidence': min(area / 5000, 1.0),
                            'bbox': (x, y, x + w, y + h),
                            'center': (center_x, center_y),
                            'area': area,
                            'width_ratio': w / frame.shape[1]
                        })
            
            last_frame = frame.copy()
            return obstacles
            
        except Exception as e:
            logger.debug(f"Motion detection error: {e}")
            return []
    
    def _filter_obstacles(self, obstacles, frame):
        """Filter and combine obstacles from different detection methods"""
        if not obstacles:
            return []
        
        # Remove duplicates by checking overlap
        filtered = []
        for obstacle in obstacles:
            is_duplicate = False
            bbox = obstacle['bbox']
            
            for existing in filtered:
                existing_bbox = existing['bbox']
                
                # Calculate overlap
                overlap = self._calculate_overlap(bbox, existing_bbox)
                if overlap > 0.3:  # 30% overlap threshold
                    # Keep the one with higher confidence
                    if obstacle['confidence'] > existing['confidence']:
                        filtered.remove(existing)
                        filtered.append(obstacle)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(obstacle)
        
        # Filter by position relevance for line following
        relevant_obstacles = []
        frame_center = frame.shape[1] / 2
        
        for obstacle in filtered:
            center_x = obstacle['center'][0]
            distance_from_center = abs(center_x - frame_center) / frame_center
            
            # Only consider obstacles that could affect the robot's path
            if (distance_from_center < OBSTACLE_MAX_DISTANCE_RATIO and
                obstacle['confidence'] > OBSTACLE_CONFIDENCE_THRESHOLD):
                relevant_obstacles.append(obstacle)
        
        # Sort by threat level (closer to center and larger = higher threat)
        relevant_obstacles.sort(key=lambda x: x['confidence'] * (1 - abs(x['center'][0] - frame_center) / frame_center), 
                               reverse=True)
        
        return relevant_obstacles

    def _calculate_overlap(self, bbox1, bbox2):
        """Calculate overlap percentage between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x_overlap = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
        y_overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
        intersection = x_overlap * y_overlap
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

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
    """Enhanced line detection with multi-zone analysis"""
    height, width = frame.shape[:2]
    
    # Define detection zones
    bottom_start = int(height * (1 - ZONE_BOTTOM_HEIGHT))
    middle_start = int(height * (1 - ZONE_BOTTOM_HEIGHT - ZONE_MIDDLE_HEIGHT))
    
    # Extract zones
    bottom_roi = frame[bottom_start:height, :]
    middle_roi = frame[middle_start:bottom_start, :]
    
    # Process bottom zone (primary)
    line_x_bottom, conf_bottom = _detect_line_in_roi(bottom_roi)
    
    # Process middle zone (prediction)
    line_x_middle, conf_middle = _detect_line_in_roi(middle_roi)
    
    # Determine best line position
    if conf_bottom > 0.3:
        return line_x_bottom, conf_bottom
    elif conf_middle > 0.4:
        return line_x_middle, conf_middle * 0.8  # Reduce confidence for prediction
    elif conf_bottom > 0.1:
        return line_x_bottom, conf_bottom
    else:
        return None, 0.0

def _detect_line_in_roi(roi):
    """Detect line in a specific ROI"""
    if roi.size == 0:
        return None, 0.0
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Enhanced preprocessing
    blurred = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)
    
    # Threshold
    _, binary = cv2.threshold(blurred, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, 0.0
    
    # Find the most line-like contour
    best_contour = None
    best_score = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area >= MIN_CONTOUR_AREA:
            # Calculate line-like properties
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            
            if width > 0 and height > 0:
                aspect_ratio = max(width, height) / min(width, height)
                
                # Prefer elongated shapes (more line-like)
                if aspect_ratio > 2.0:  # Line should be elongated
                    score = area * (aspect_ratio / 10.0)  # Reward elongated shapes
                    
                    if score > best_score:
                        best_score = score
                        best_contour = contour
    
    if best_contour is None:
        return None, 0.0
    
    # Get center of best contour
    M = cv2.moments(best_contour)
    if M["m00"] == 0:
        return None, 0.0
    
    cx = int(M["m10"] / M["m00"])
    
    # Calculate confidence
    area = cv2.contourArea(best_contour)
    roi_height, roi_width = roi.shape[:2]
    confidence = min(area / (roi_width * roi_height * 0.1), 1.0)
    
    return cx, confidence

# -----------------------------------------------------------------------------
# --- MOVEMENT CONTROL ---
# -----------------------------------------------------------------------------
def get_movement_command(steering, obstacles):
    """Enhanced movement control with smart obstacle avoidance"""
    global avoidance_counter, avoidance_direction, avoidance_cooldown_counter
    global obstacle_tracker, robot_stats
    
    # Update cooldown
    if avoidance_cooldown_counter > 0:
        avoidance_cooldown_counter -= 1
    
    # Check for immediate threats
    immediate_threats = [obs for obs in obstacles 
                        if abs(obs['center'][0] - CAMERA_WIDTH/2) < CAMERA_WIDTH * 0.25]
    
    # Start new avoidance if threat detected and not in cooldown
    if immediate_threats and avoidance_counter <= 0 and avoidance_cooldown_counter <= 0:
        threat = immediate_threats[0]  # Take the first (highest confidence) threat
        
        avoidance_counter = AVOIDANCE_DURATION
        avoidance_cooldown_counter = AVOIDANCE_COOLDOWN
        robot_stats['avoidance_maneuvers'] += 1
        
        # Determine avoidance direction
        threat_center = threat['center'][0]
        frame_center = CAMERA_WIDTH / 2
        
        if threat_center < frame_center:
            avoidance_direction = 'RIGHT'
            return COMMANDS['AVOID_RIGHT']
        else:
            avoidance_direction = 'LEFT'
            return COMMANDS['AVOID_LEFT']
    
    # Continue current avoidance
    if avoidance_counter > 0:
        avoidance_counter -= 1
        if avoidance_direction == 'LEFT':
            return COMMANDS['AVOID_LEFT']
        elif avoidance_direction == 'RIGHT':
            return COMMANDS['AVOID_RIGHT']
    
    # Normal line following with enhanced steering
    if abs(steering) < 0.08:  # Smaller deadzone for more responsive steering
        return COMMANDS['FORWARD']
    elif steering < -0.15:  # Strong right turn needed
        return COMMANDS['RIGHT']
    elif steering > 0.15:   # Strong left turn needed
        return COMMANDS['LEFT']
    elif steering < 0:      # Gentle right turn
        return COMMANDS['RIGHT']
    else:                   # Gentle left turn
        return COMMANDS['LEFT']

# -----------------------------------------------------------------------------
# --- VISUALIZATION ---
# -----------------------------------------------------------------------------
def draw_debug_info(frame, line_x=None, obstacles=None, line_confidence=0.0):
    """Enhanced debug visualization"""
    height, width = frame.shape[:2]
    
    # Draw zone boundaries
    bottom_start = int(height * (1 - ZONE_BOTTOM_HEIGHT))
    middle_start = int(height * (1 - ZONE_BOTTOM_HEIGHT - ZONE_MIDDLE_HEIGHT))
    top_end = int(height * ZONE_TOP_HEIGHT)
    
    # Zone rectangles
    cv2.rectangle(frame, (0, bottom_start), (width, height), (0, 255, 255), 2)
    cv2.putText(frame, "LINE ZONE", (5, bottom_start + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    cv2.rectangle(frame, (0, middle_start), (width, bottom_start), (0, 200, 255), 1)
    cv2.putText(frame, "PREDICTION", (5, middle_start + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    
    cv2.rectangle(frame, (0, 0), (width, top_end), (255, 100, 100), 2)
    cv2.putText(frame, "OBSTACLE ZONE", (5, 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
    
    # Draw obstacles with enhanced information
    if obstacles:
        for i, obstacle in enumerate(obstacles):
            x1, y1, x2, y2 = obstacle['bbox']
            
            # Color coding based on threat level and detection type
            threat_level = 1.0 - abs(obstacle['center'][0] - width/2) / (width/2)
            
            if obstacle.get('type') == 'yolo':
                color = (0, 0, 255)  # Red for YOLO detections
            elif obstacle.get('type') == 'motion':
                color = (255, 0, 255)  # Magenta for motion
            else:
                color = (0, 165, 255)  # Orange for contour
            
            # Adjust color intensity based on threat
            if threat_level > 0.7:
                thickness = 3
            else:
                thickness = 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Enhanced labels
            label = f"{obstacle.get('type', 'unk').upper()[:4]}: {obstacle['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw threat indicator
            if threat_level > 0.5:
                cv2.putText(frame, "⚠️ THREAT", (x1, y2 + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Draw center line
    center_x = width // 2
    cv2.line(frame, (center_x, 0), (center_x, height), (200, 200, 200), 1)
    
    # Draw detected line with enhanced visualization
    if line_x is not None:
        line_y = bottom_start + int(ZONE_BOTTOM_HEIGHT * height / 2)
        cv2.circle(frame, (line_x, line_y), 8, (0, 255, 0), -1)
        cv2.line(frame, (center_x, line_y), (line_x, line_y), (255, 0, 255), 3)
        
        # Draw offset arc for better visualization
        offset_angle = np.arctan2(line_x - center_x, 50) * 180 / np.pi
        cv2.putText(frame, f"Offset: {line_offset:.2f}", (line_x - 40, line_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    
    # Enhanced status information
    status_color = (0, 255, 0) if line_detected else (0, 0, 255)
    if obstacles and any(abs(obs['center'][0] - width/2) < width * 0.25 for obs in obstacles):
        status_color = (0, 100, 255)  # Orange for obstacle threat
    
    info_lines = [
        f"Status: {robot_status}",
        f"Line: {line_offset:.2f} (C:{line_confidence:.2f})",
        f"Command: {turn_command}",
        f"FPS: {current_fps:.1f}",
        f"Obstacles: {len(obstacles) if obstacles else 0}",
        f"Detection: {'YOLO+' if YOLO_AVAILABLE else ''}Enhanced"
    ]
    
    for i, line in enumerate(info_lines):
        y_pos = 20 + (i * 18)
        cv2.putText(frame, line, (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, 
                    status_color if i == 0 else (255, 255, 255), 1)
    
    # Avoidance status
    if avoidance_counter > 0:
        cv2.putText(frame, f"AVOIDING {avoidance_direction} ({avoidance_counter})", 
                   (width//2 - 80, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    # Performance indicators
    cv2.putText(frame, f"Avoided: {robot_stats.get('avoidance_maneuvers', 0)}", 
               (width - 120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

# -----------------------------------------------------------------------------
# --- FLASK WEB INTERFACE ---
# -----------------------------------------------------------------------------
app = Flask(__name__)

@app.route('/')
def index():
    return """<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Line Follower Robot</title>
    <style>
        body { font-family: Arial; background: #1a1a1a; color: white; padding: 20px; }
        .dashboard { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
        .video-section { background: #333; padding: 20px; border-radius: 10px; }
        .stats-section { background: #333; padding: 20px; border-radius: 10px; }
        .video-feed { width: 100%; border-radius: 8px; }
        .stat-item { margin: 8px 0; padding: 8px; background: #444; border-radius: 5px; }
        .status-online { color: #4CAF50; }
        .status-offline { color: #f44336; }
        .status-warning { color: #FF9800; }
        h1 { text-align: center; color: #4CAF50; }
        h3 { color: #81C784; }
        .performance-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
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
                    document.getElementById('avoidance-count').textContent = data.avoidance_maneuvers;
                    document.getElementById('esp-status').className = data.esp_connected ? 'status-online' : 'status-offline';
                    document.getElementById('esp-status').textContent = data.esp_connected ? 'Connected' : 'Disconnected';
                    
                    // Update status indicator
                    const statusElement = document.getElementById('robot-status');
                    if (data.status.includes('obstacle') || data.status.includes('avoid')) {
                        statusElement.className = 'status-warning';
                    } else if (data.line_detected) {
                        statusElement.className = 'status-online';
                    } else {
                        statusElement.className = 'status-offline';
                    }
                });
        }
        setInterval(updateStats, 500);
        updateStats();
    </script>
</head>
<body>
    <h1>🤖 Enhanced Line Follower Robot</h1>
    <p style="text-align: center; color: #81C784;">Advanced Obstacle Detection & Avoidance System</p>
    <div class="dashboard">
        <div class="video-section">
            <h3>📷 Live Camera Feed with Enhanced Detection</h3>
            <img src="/video_feed" class="video-feed" alt="Robot Camera Feed">
        </div>
        <div class="stats-section">
            <h3>📊 Robot Status</h3>
            <div class="stat-item">Status: <span id="robot-status" class="status-offline">Loading...</span></div>
            <div class="stat-item">Line Offset: <span id="line-offset">0.00</span></div>
            <div class="stat-item">Line Confidence: <span id="confidence">0%</span></div>
            <div class="stat-item">Current Command: <span id="command">STOP</span></div>
            <div class="stat-item">FPS: <span id="fps">0.0</span></div>
            
            <h3 style="margin-top: 20px;">🛡️ Safety & Detection</h3>
            <div class="stat-item">Obstacles Detected: <span id="obstacles">0</span></div>
            <div class="stat-item">Avoidance Maneuvers: <span id="avoidance-count">0</span></div>
            <div class="stat-item">ESP32 Status: <span id="esp-status" class="status-offline">Disconnected</span></div>
            
            <h3 style="margin-top: 20px;">⚙️ Detection Methods</h3>
            <div class="stat-item">YOLO Detection: <span style="color: #4CAF50;">""" + ("Available" if YOLO_AVAILABLE else "Fallback Mode") + """</span></div>
            <div class="stat-item">Contour Analysis: <span style="color: #4CAF50;">Enhanced</span></div>
            <div class="stat-item">Motion Detection: <span style="color: #4CAF50;">Active</span></div>
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
        'total_frames': robot_stats.get('total_frames', 0),
        'yolo_available': YOLO_AVAILABLE
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
    global line_detected, current_fps, confidence, esp_connection
    global obstacle_detected, robot_stats
    
    logger.info("🚀 Starting Enhanced Line Follower Robot")
    logger.info(f"🔍 Detection Mode: {'YOLO + Enhanced' if YOLO_AVAILABLE else 'Enhanced Contour + Motion'}")
    robot_status = "Initializing enhanced systems"
    
    # Initialize enhanced obstacle detector
    obstacle_detector = EnhancedObstacleDetector()
    
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
    logger.info("🌐 Enhanced web dashboard available at http://localhost:5000")
    
    # Performance tracking
    fps_history = deque(maxlen=10)
    offset_history = deque(maxlen=3)
    search_counter = 0
    
    logger.info("✅ Enhanced robot ready! Starting intelligent line detection...")
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
            
            # Enhanced obstacle detection
            obstacles = obstacle_detector.detect_obstacles(frame)
            if obstacles:
                robot_stats['obstacles_detected'] += 1
                obstacle_detected = True
            else:
                obstacle_detected = False
            
            # Enhanced line detection
            line_x, line_confidence = detect_line(frame)
            confidence = line_confidence
            
            if line_x is not None and confidence > 0.2:
                line_detected = True
                search_counter = 0
                
                # Calculate offset with smoothing
                center_x = CAMERA_WIDTH / 2
                raw_offset = (line_x - center_x) / center_x
                offset_history.append(raw_offset)
                
                line_offset = sum(offset_history) / len(offset_history) if offset_history else 0.0
                
                # Enhanced PID control
                steering_error = -line_offset
                steering_value = pid.calculate(steering_error)
                
                # Smart movement control with obstacle avoidance
                turn_command = get_movement_command(steering_value, obstacles)
                
                # Enhanced status reporting
                if obstacles:
                    threat_count = sum(1 for obs in obstacles 
                                     if abs(obs['center'][0] - CAMERA_WIDTH/2) < CAMERA_WIDTH * 0.25)
                    if threat_count > 0:
                        robot_status = f"⚠️ {threat_count} threat(s) detected - avoiding (C:{confidence:.2f})"
                    else:
                        robot_status = f"🔍 {len(obstacles)} object(s) monitored (C:{confidence:.2f})"
                else:
                    robot_status = f"✅ Clear path - following line (C:{confidence:.2f})"
                
            else:
                line_detected = False
                search_counter += 1
                
                if obstacles:
                    robot_status = "⚠️ Line lost - obstacles present"
                    turn_command = COMMANDS['STOP']
                else:
                    if search_counter < 10:
                        robot_status = "🔍 Brief line gap - continuing"
                    elif search_counter < 30:
                        robot_status = f"🔄 Searching for line ({search_counter})"
                    else:
                        robot_status = "🛑 Line completely lost"
                        turn_command = COMMANDS['STOP']
            
            # Send command to ESP32
            if esp_connection:
                esp_connection.send_command(turn_command)
            
            # Enhanced visualization
            draw_debug_info(display_frame, line_x, obstacles, confidence)
            
            # Update output frame
            with frame_lock:
                output_frame = display_frame.copy()
            
            # Performance calculation
            processing_time = time.time() - start_time
            if processing_time > 0:
                fps_history.append(1.0 / processing_time)
                current_fps = sum(fps_history) / len(fps_history)
            
            # Enhanced logging
            if frame_count % 60 == 0:
                obstacle_status = f"{len(obstacles)} detected" if obstacles else "clear"
                logger.info(f"📊 Status: {robot_status[:30]}... | FPS: {current_fps:.1f} | "
                           f"Command: {turn_command} | ESP32: {'✅' if esp_connected else '❌'} | "
                           f"Obstacles: {obstacle_status} | Detection: {'YOLO+' if YOLO_AVAILABLE else 'Enhanced'}")
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        logger.info("🛑 Stopping enhanced robot (Ctrl+C pressed)")
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}", exc_info=True)
    finally:
        # Cleanup
        if esp_connection:
            esp_connection.send_command(COMMANDS['STOP'])
            esp_connection.close()
        
        if cap.isOpened():
            cap.release()
        
        logger.info("✅ Enhanced robot stopped cleanly")

if __name__ == "__main__":
    main() 