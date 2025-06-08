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


try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("YOLO11n available for object detection")
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available. Install with: pip install ultralytics")
    print("Falling back to basic computer vision detection")


ESP32_IP = '192.168.2.21'  
ESP32_PORT = 1234
CAMERA_WIDTH, CAMERA_HEIGHT = 730, 420
CAMERA_FPS = 5

BLACK_THRESHOLD = 60  # Higher values detect darker lines
BLUR_SIZE = 5
MIN_CONTOUR_AREA = 100  # Minimum area to be considered a line

# Multi-zone detection parameters
ZONE_BOTTOM_HEIGHT = 0.25   # Bottom 20% for primary line following
ZONE_MIDDLE_HEIGHT = 0.20   # Middle 25% for corner prediction
ZONE_TOP_HEIGHT = 0.45      # Top 45% for early object detection (much larger!)

# Corner detection parameters
CORNER_DETECTION_ENABLED = True
CORNER_CONFIDENCE_BOOST = 1.2
CORNER_CIRCULARITY_THRESHOLD = 0.4  # Lower values indicate corners
CORNER_PREDICTION_THRESHOLD = 0.3   # Confidence needed for corner warning

# Object detection parameters1
OBJECT_DETECTION_ENABLED = True  # Enable obstacle detection and avoidance
USE_YOLO = True  # Use YOLO11n for accurate object detection (recommended)
USE_SMART_AVOIDANCE = True  # Use smart avoidance with live mapping and learning

# YOLO Configuration
YOLO_MODEL_SIZE = "yolo11n.pt"  # Nano model for speed (yolo11s.pt, yolo11m.pt for more accuracy)
YOLO_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for object detection
YOLO_CLASSES_TO_AVOID = [0, 39, 41, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]  # person, bottle, cup, bowl, etc.

# Legacy CV-based detection parameters (fallback if YOLO not available)
OBJECT_SIZE_THRESHOLD = 800    # Minimum contour area 
OBJECT_WIDTH_THRESHOLD = 0.20  # Object width ratio to trigger avoidance
OBJECT_HEIGHT_THRESHOLD = 0.15  # Object height ratio to trigger avoidance
OBJECT_AVOIDANCE_DISTANCE = 35  # How many frames to remember object position (much longer avoidance)
OBJECT_MIN_ASPECT_RATIO = 0.3  # Shape filtering
OBJECT_MAX_ASPECT_RATIO = 3.0  # Shape filtering
OBJECT_LINE_BLOCKING_THRESHOLD = 0.7  # Distance from line to trigger avoidance (very lenient for early detection)

# Adaptive PID controller values with auto-tuning
KP = 0.35  # Lower initial proportional gain to reduce overshoot
KI = 0.005 # Lower integral gain
KD = 0.25  # Higher derivative gain for better dampening
MAX_INTEGRAL = 2.0  # Lower integral windup limit

# Auto-tuning parameters
PID_LEARNING_RATE = 0.0005  # Slower learning to avoid aggressive changes
PID_ADAPTATION_WINDOW = 100  # Longer window for more stable adaptation
PERFORMANCE_THRESHOLD = 0.12  # Tighter performance target

# Commands for ESP32 - GENTLE PROGRESSIVE AVOIDANCE
COMMANDS = {'FORWARD': 'FORWARD', 'LEFT': 'LEFT', 'RIGHT': 'RIGHT', 'STOP': 'STOP', 
           'AVOID_LEFT': 'LEFT', 'AVOID_RIGHT': 'RIGHT'}  # Use gentle turns for avoidance
SPEED = 'SLOW'  # Default speed
ROBOT_SPEED_M_S = 0.1  # Approximate robot speed in m/s for distance estimation

# Enhanced obstacle mapping parameters
OBSTACLE_MAP_SIZE = 50  # Remember last 50 obstacles
OBSTACLE_MEMORY_TIMEOUT = 30.0  # Forget obstacles after 30 seconds
MIN_OBSTACLE_WIDTH = 0.1  # Minimum width ratio to be significant
MIN_OBSTACLE_HEIGHT = 0.08  # Minimum height ratio to be significant

# Gentle avoidance parameters - roundabout style smooth curve
GENTLE_TURN_DURATION = 6     # Shorter turns to stay closer to line
GENTLE_CLEAR_DURATION = 2    # About 1 second forward movement
GENTLE_RETURN_DURATION = 4   # Shorter return - less time turning backls


# Dynamic path calculation
SAFETY_MARGIN = 1.5           # Multiply obstacle size by this for safety
PATH_CALCULATION_ENABLED = True

# Steering parameters
STEERING_DEADZONE = 0.12  # Ignore small errors (slightly reduced for more responsiveness)
MAX_STEERING = 0.96 # Maximum steering value (slightly increased for better responsiveness)

# Learning state
obstacle_memory = {}  # Learning-based obstacle memory
performance_history = deque(maxlen=PID_ADAPTATION_WINDOW)  # PID performance tracking
learned_maneuvers = {}  # Successful avoidance patterns

# Enhanced 3-phase obstacle avoidance state with mapping
object_detected = False
object_position = 0.0
current_obstacle = None       # Current obstacle being avoided
obstacle_map = {}            # Persistent obstacle memory
avoidance_phase = 'none'     # 'none', 'mapping', 'avoiding', 'clearing', 'returning'
avoidance_side = 'none'      # 'left', 'right'
avoidance_frame_count = 0
avoidance_duration = 0
planned_path = None          # Calculated path around obstacle
corner_warning = False
corner_prediction_frames = 0
object_detection_frames = 0
OBJECT_DETECTION_PERSISTENCE = 0  # Immediate detection for faster avoidance

# Dynamic avoidance parameters
MAX_TURN_DURATION = 20          # Maximum frames to turn if object still visible
AVOIDANCE_CLEAR_DURATION = 12   # Phase 2: Move forward to clear object  
AVOIDANCE_RETURN_DURATION = 15  # Phase 3: Turn back to find line
OBJECT_CLEAR_THRESHOLD = 3      # Frames without seeing object to consider it cleared

# Avoidance phase durations
AVOIDANCE_TURN_FRAMES = 8      # Frames to turn away from object
AVOIDANCE_CLEAR_FRAMES = 12    # Frames to go around object  
AVOIDANCE_RETURN_FRAMES = 10   # Frames to turn back toward line

# Add distance estimation parameters near the top with other parameters
# Distance estimation using monocular vision
CAMERA_FOCAL_LENGTH = 200.0  # Estimated focal length in pixels (you may need to calibrate this)
CAMERA_HEIGHT = 0.12  # Camera height from ground in meters (adjust for your robot)
CAMERA_ANGLE = 5  # Camera tilt angle in degrees (positive = looking down)

# Known object sizes for distance estimation (in meters)
KNOWN_OBJECT_SIZES = {
    'person': 1.7,      # Average person height
    'car': 1.5,         # Average car height  
    'bicycle': 1.0,     # Average bicycle height
    'bottle': 0.25,     # Average bottle height
    'cup': 0.10,        # Average cup height
    'chair': 0.85,      # Average chair height
    'default': 0.3      # Default object height for unknown objects
}

# Distance-based avoidance thresholds
SAFE_DISTANCE = 2.0       # Minimum safe distance in meters - start avoidance earlier
WARNING_DISTANCE = 3.0    # Warning distance in meters - detect threats earlier
EMERGENCY_DISTANCE = 1.2  # Emergency stop distance - increased to avoid getting too close

# Avoidance path planning
MIN_AVOIDANCE_ANGLE = 30   # Minimum avoidance angle in degrees
MAX_AVOIDANCE_ANGLE = 90   # Maximum avoidance angle in degrees

# Frame averaging for stable distance estimation
DISTANCE_HISTORY_SIZE = 5  # Number of frames to average distance over

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
    'corner_count': 0,
    'objects_detected': 0,
    'avoidance_maneuvers': 0
}

# YOLO model (initialized in main)
yolo_model = None

# Smart avoidance system instance
smart_avoidance = None

# PID controller instance (for dashboard access)
pid_controller = None

# Count frames during clearing phase
clearing_frame_count = 0
# Last number of clearing frames
last_avoidance_clear_frames = 0
# Last estimated distance (m) cleared during avoidance
last_avoidance_clear_distance = 0.0

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
class AdaptivePID:
    def __init__(self, kp, ki, kd, max_integral=5.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        
        # Learning and adaptation
        self.error_history = deque(maxlen=20)
        self.output_history = deque(maxlen=20)
        self.performance_score = 0.0
        self.adaptation_counter = 0
        
    def calculate(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        dt = max(dt, 0.001)
        
        # Store error for learning
        self.error_history.append(abs(error))
        
        # Calculate PID terms
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        derivative = (error - self.previous_error) / dt
        
        p_term = self.kp * error
        i_term = self.ki * self.integral
        d_term = self.kd * derivative
        output = p_term + i_term + d_term
        
        # Store output for learning
        self.output_history.append(abs(output))
        
        # Auto-tune parameters
        self._adapt_parameters()
        
        self.previous_error = error
        return np.clip(output, -MAX_STEERING, MAX_STEERING)
    
    def _adapt_parameters(self):
        if len(self.error_history) < 20:
            return
            
        self.adaptation_counter += 1
        if self.adaptation_counter % 50 != 0:  # Adapt every 50 iterations for stability
            return
            
        # Calculate performance metrics
        avg_error = sum(self.error_history) / len(self.error_history)
        error_variance = np.var(list(self.error_history))
        recent_errors = list(self.error_history)[-5:]
        trend = sum(recent_errors) / len(recent_errors)
        
        # More conservative adaptation to prevent overshoot
        if avg_error > PERFORMANCE_THRESHOLD:
            if error_variance > 0.08:  # High oscillation - reduce overshoot
                self.kd += PID_LEARNING_RATE * 2  # Increase dampening more aggressively
                self.kp *= 0.95  # Reduce proportional gain
                self.ki *= 0.9   # Reduce integral to prevent windup
            elif trend > avg_error * 1.2:  # Error increasing - be more conservative
                self.kp *= 0.98
            else:  # Steady state error - gentle increase
                self.kp += PID_LEARNING_RATE * 0.5
                
        elif avg_error < PERFORMANCE_THRESHOLD * 0.3:  # Very good performance
            if error_variance < 0.02:  # Very stable
                self.kp += PID_LEARNING_RATE * 0.3  # Small increase
        
        # Conservative bounds to prevent overshoot
        self.kp = np.clip(self.kp, 0.1, 0.8)   # Lower max KP
        self.ki = np.clip(self.ki, 0.001, 0.03) # Lower max KI
        self.kd = np.clip(self.kd, 0.1, 0.6)   # Higher min/max KD for better dampening
    
    def get_params(self):
        return self.kp, self.ki, self.kd
    
    def reset(self):
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

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
            logger.info(f"Connected to ESP32 at {self.ip}:{self.port}")
            return True
        except Exception as e:
            esp_connected = False
            self.connection_attempts += 1
            if self.connection_attempts % 10 == 1:  # Log every 10 attempts
                logger.error(f"Failed to connect to ESP32 (attempt {self.connection_attempts}): {e}")
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
# --- Enhanced Multi-Zone Image Processing ---
# -----------------------------------------------------------------------------
def detect_line_in_roi(roi, zone_name="unknown"):
    """Helper function to detect line in a specific ROI with zone-specific parameters"""
    if roi.size == 0:
        return None, 0.0, None
        
    # Find contours in the ROI
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, 0.0, None
    
    # Filter contours based on zone
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if zone_name == "bottom" and area >= MIN_CONTOUR_AREA:
            valid_contours.append(contour)
        elif zone_name == "middle" and area >= MIN_CONTOUR_AREA * 0.7:  # Slightly smaller threshold for middle
            valid_contours.append(contour)
        elif zone_name == "top" and area >= OBJECT_SIZE_THRESHOLD:  # Much larger threshold for objects
            valid_contours.append(contour)
    
    if not valid_contours:
        return None, 0.0, None
    
    # Find the largest contour
    largest_contour = max(valid_contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # Get the center of the contour
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None, 0.0, None
    
    # Calculate x-position of detection center
    cx = int(M["m10"] / M["m00"])
    
    # Calculate confidence based on contour area and zone
    height, width = roi.shape
    base_confidence = min(area / (width * height * 0.1), 1.0)
    
    # Zone-specific confidence adjustments
    if zone_name == "top":
        # For object detection, check if it's significant enough to avoid
        bounding_rect = cv2.boundingRect(largest_contour)
        object_width_ratio = bounding_rect[2] / width
        if object_width_ratio > OBJECT_WIDTH_THRESHOLD:
            base_confidence *= 1.5  # Boost confidence for significant objects
    elif zone_name == "middle":
        # For corner prediction, analyze shape
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < CORNER_CIRCULARITY_THRESHOLD:
                base_confidence *= CORNER_CONFIDENCE_BOOST
    
    return cx, base_confidence, largest_contour

def detect_objects_with_yolo(frame, top_zone_bbox, line_position=None):
    """Use YOLO11n to detect objects in the top zone"""
    global yolo_model
    
    if not YOLO_AVAILABLE or yolo_model is None:
        return []
    
    x1, y1, x2, y2 = top_zone_bbox
    top_zone = frame[y1:y2, x1:x2]
    
    if top_zone.size == 0:
        return []
    
    try:
        # Run YOLO inference on the top zone
        results = yolo_model(top_zone, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)
        
        objects = []
        frame_width = x2 - x1
        frame_height = y2 - y1
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Get bounding box coordinates
                    x, y, w, h = box.xywh[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Check if this is a class we want to avoid
                    if class_id not in YOLO_CLASSES_TO_AVOID:
                        continue
                    
                    # Calculate object properties
                    width_ratio = w / frame_width
                    height_ratio = h / frame_height
                    
                    # Object center position relative to zone (-1.0 to 1.0)
                    center_x = x
                    relative_pos = (center_x - frame_width/2) / (frame_width/2)
                    
                    # Check if object is blocking the robot's path
                    if line_position is not None:
                        # Convert line position to top zone coordinates
                        line_x_in_zone = line_position - x1
                        line_relative_pos = (line_x_in_zone - frame_width/2) / (frame_width/2)
                        distance_from_line = abs(relative_pos - line_relative_pos)
                        
                        # Object must be near the line path to be considered blocking
                        if distance_from_line > OBJECT_LINE_BLOCKING_THRESHOLD:
                            continue
                    else:
                        # If no line detected, only consider objects near center
                        if abs(relative_pos) > 0.4:
                            continue
                    
                    # Get class name for debugging
                    class_name = yolo_model.names[class_id] if hasattr(yolo_model, 'names') else f"class_{class_id}"
                    
                    # Create object data for distance estimation
                    object_data = {
                        'position': relative_pos,
                        'width_ratio': width_ratio,
                        'height_ratio': height_ratio,
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'bbox': (int(x - w/2), int(y - h/2), int(w), int(h)),
                        'distance_from_line': distance_from_line if line_position is not None else abs(relative_pos)
                    }
                    
                    # Calculate distance using distance estimator (higher confidence for YOLO)
                    estimated_distance = distance_estimator.get_object_distance(
                        object_data, frame_height, frame_width, object_id=f"yolo_{class_id}_{int(center_x)}_{int(y)}"
                    )
                    
                    # Assess threat level
                    threat_level = distance_estimator.assess_threat_level(estimated_distance, relative_pos)
                    
                    # Add distance and threat info to object data
                    object_data.update({
                        'estimated_distance': estimated_distance,
                        'threat_level': threat_level,
                        'distance_confidence': 0.8  # Higher confidence for YOLO detection
                    })
                    
                    print(f"🔍 YOLO detected: {class_name} (conf: {confidence:.2f}, pos: {relative_pos:.2f})")
                    
                    # Enhanced logging with distance info
                    if estimated_distance:
                        print(f"    Distance: {estimated_distance:.2f}m, Threat: {threat_level}")
                    
                    # Only add objects that pose some threat or are close
                    if threat_level in ['warning', 'danger', 'emergency'] or (estimated_distance and estimated_distance < WARNING_DISTANCE):
                        objects.append(object_data)
                    elif estimated_distance is None:
                        # Keep objects where we can't estimate distance (safety fallback)
                        objects.append(object_data)
        
        return objects
        
    except Exception as e:
        print(f"YOLO detection error: {e}")
        return []

def detect_objects_in_zone(binary_roi, width, line_position=None):
    """Detect objects specifically in the top zone that might block the line path"""
    if binary_roi.size == 0:
        return []
        
    height = binary_roi.shape[0]
    contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Debug: Log how many contours found in top zone
    if len(contours) > 0:
        print(f"DEBUG: Found {len(contours)} contours in top zone")
    
    objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < OBJECT_SIZE_THRESHOLD:
            continue
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate ratios
        width_ratio = w / width
        height_ratio = h / height
        aspect_ratio = w / h if h > 0 else 0
        
        # Less restrictive filtering to detect real objects
        # 1. Check minimum size requirements
        if width_ratio < OBJECT_WIDTH_THRESHOLD or height_ratio < OBJECT_HEIGHT_THRESHOLD:
            continue
            
        # 2. Check aspect ratio (allow wide range for containers, boxes, etc.)
        if aspect_ratio < OBJECT_MIN_ASPECT_RATIO or aspect_ratio > OBJECT_MAX_ASPECT_RATIO:
            continue
        
        # 3. Calculate object center position
        center_x = x + w/2
        relative_pos = (center_x - width/2) / (width/2)
        
        # 4. Check if object is in the robot's path (more lenient)
        if line_position is not None:
            line_relative_pos = (line_position - width/2) / (width/2)
            distance_from_line = abs(relative_pos - line_relative_pos)
            
            # Be more lenient - object just needs to be somewhat near the path
            if distance_from_line > OBJECT_LINE_BLOCKING_THRESHOLD:
                continue
        else:
            # If no line detected, consider objects in broader center area
            if abs(relative_pos) > 0.6:
                continue
            
        # 5. Additional validation - check if object is substantial enough
        # Calculate minimum expected area for a real object
        min_expected_area = (width * 0.15) * (height * 0.10)  # 15% x 10% of frame
        if area < min_expected_area:
            continue
            
        # Check that the object has reasonable density (not just scattered pixels)
        contour_perimeter = cv2.arcLength(contour, True)
        if contour_perimeter > 0:
            compactness = (4 * 3.14159 * area) / (contour_perimeter * contour_perimeter)
            if compactness < 0.1:  # Too scattered/thin to be a real object
                continue
        else:
            compactness = 0.0
        
        # Log detected object for debugging
        print(f"DEBUG: Valid object - Area: {area}, Width: {width_ratio:.2f}, Height: {height_ratio:.2f}, Aspect: {aspect_ratio:.2f}, Pos: {relative_pos:.2f}, Compact: {compactness:.2f}")
        
        # Create object data for distance estimation
        object_data = {
            'position': relative_pos,
            'width_ratio': width_ratio,
            'height_ratio': height_ratio,
            'aspect_ratio': aspect_ratio,
            'area': area,
            'contour': contour,
            'bbox': (x, y, w, h),
            'distance_from_line': distance_from_line if line_position is not None else abs(relative_pos),
            'class_name': 'default'  # Traditional CV doesn't have class info
        }
        
        # Calculate distance using distance estimator
        estimated_distance = distance_estimator.get_object_distance(
            object_data, height, width, object_id=f"cv_{center_x}_{y}"
        )
        
        # Assess threat level
        threat_level = distance_estimator.assess_threat_level(estimated_distance, relative_pos)
        
        # Add distance and threat info to object data
        object_data.update({
            'estimated_distance': estimated_distance,
            'threat_level': threat_level,
            'distance_confidence': 0.6  # Medium confidence for CV-based detection
        })
        
        # Only add objects that pose some threat or are close enough to matter
        if threat_level in ['warning', 'danger', 'emergency'] or (estimated_distance and estimated_distance < WARNING_DISTANCE):
            print(f"DEBUG: Threat object - Distance: {estimated_distance:.2f}m, Threat: {threat_level}")
            objects.append(object_data)
        elif estimated_distance is None:
            # Keep objects where we can't estimate distance (safety fallback)
            print(f"DEBUG: Unknown distance object - keeping for safety")
            objects.append(object_data)
    
    return objects

def process_image_multi_zone(frame):
    """Enhanced image processing with multi-zone detection"""
    global corner_warning, corner_prediction_frames, object_detected, object_position
    
    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)
    
    # Create binary image for line detection
    _, binary = cv2.threshold(blurred, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    
    # Create inverted binary for object detection (objects are usually lighter than lines)
    # Balanced threshold to avoid false positives from shadows/lighting
    _, binary_inverted = cv2.threshold(blurred, BLACK_THRESHOLD + 40, 255, cv2.THRESH_BINARY)
    
    height, width = binary.shape
    
    # Define zones
    bottom_start = int(height * (1 - ZONE_BOTTOM_HEIGHT))
    middle_start = int(height * (1 - ZONE_BOTTOM_HEIGHT - ZONE_MIDDLE_HEIGHT))
    top_start = int(height * (1 - ZONE_BOTTOM_HEIGHT - ZONE_MIDDLE_HEIGHT - ZONE_TOP_HEIGHT))
    
    # Extract ROIs for each zone
    bottom_roi = binary[bottom_start:height, :]
    middle_roi = binary[middle_start:bottom_start, :]
    top_roi = binary_inverted[top_start:middle_start, :]  # Use inverted for object detection
    
    # Process each zone
    # 1. Bottom zone - Primary line following
    bottom_x, bottom_confidence, bottom_contour = detect_line_in_roi(bottom_roi, "bottom")
    
    # 2. Middle zone - Corner prediction
    middle_x, middle_confidence, middle_contour = detect_line_in_roi(middle_roi, "middle")
    
    # 3. Top zone - Object detection (only if enabled)
    detected_objects = []
    if OBJECT_DETECTION_ENABLED:
        if USE_YOLO and YOLO_AVAILABLE and yolo_model is not None:
            # Use YOLO for accurate object detection
            top_zone_bbox = (0, top_start, width, middle_start)
            line_pos = bottom_x if bottom_x is not None and bottom_confidence > 0.3 else None
            detected_objects = detect_objects_with_yolo(frame, top_zone_bbox, line_pos)
        else:
            # Fallback to traditional computer vision detection
            if bottom_x is not None and bottom_confidence > 0.3:
                detected_objects = detect_objects_in_zone(top_roi, width, bottom_x)
            else:
                detected_objects = detect_objects_in_zone(top_roi, width, None)
    
    # Update corner warning
    if middle_confidence > CORNER_PREDICTION_THRESHOLD:
        if bottom_x is not None and middle_x is not None:
            shift = abs(middle_x - bottom_x)
            if shift > width * 0.15:  # Significant shift indicates upcoming corner
                corner_warning = True
                corner_prediction_frames += 1
            else:
                corner_prediction_frames = max(0, corner_prediction_frames - 1)
        else:
            corner_prediction_frames = max(0, corner_prediction_frames - 1)
    else:
        corner_prediction_frames = max(0, corner_prediction_frames - 1)
    
    if corner_prediction_frames <= 0:
        corner_warning = False
    
    # Update object detection for 3-PHASE NAVIGATION
    global object_detection_frames, avoidance_phase
    
    if detected_objects and avoidance_phase == 'none':
        # Only start new avoidance if not already in progress
        # Find the most significant object (largest and most centered)
        def object_significance(obj):
            if 'area' in obj:
                # Traditional CV detection
                return obj['area'] * (1 - abs(obj['position']))
            else:
                # YOLO detection - use bounding box area equivalent
                area_equivalent = obj['width_ratio'] * obj['height_ratio'] * 10000  # Scale for comparison
                return area_equivalent * (1 - abs(obj['position']))
        
        main_object = max(detected_objects, key=object_significance)
        
        # SMART detection - check if this is a significant new obstacle
        obstacle_id = obstacle_mapper._generate_obstacle_id(main_object)
        
        if not object_detected:  # First time detecting this object
            robot_stats['objects_detected'] += 1
            
            # Add to obstacle memory for tracking
            # Use bottom_x since line_x isn't determined yet
            line_position_for_obstacle = bottom_x if bottom_x is not None else None
            obstacle_mapper.add_obstacle(main_object, line_position_for_obstacle)
            
            if 'class_name' in main_object:
                # YOLO object with size info
                logger.info(f"🗺️ SMART DETECTION: {main_object['class_name']} "
                           f"(conf: {main_object['confidence']:.2f}, "
                           f"size: {main_object['width_ratio']:.2f}x{main_object['height_ratio']:.2f}, "
                           f"pos: {main_object['position']:.2f}) - Calculating optimal path!")
            else:
                # Traditional CV object
                logger.info(f"🗺️ SMART DETECTION: Object "
                           f"(size: {main_object.get('width_ratio', 'N/A')}x{main_object.get('height_ratio', 'N/A')}, "
                           f"pos: {main_object['position']:.2f}, area: {main_object.get('area', 'N/A')}) - Mapping geometry!")
        
        object_detected = True
        object_position = main_object['position']
        object_detection_frames = 1
        
    elif avoidance_phase == 'none':
        # No objects detected and not in avoidance
        if object_detected:
            logger.info("✅ No more objects detected")
        object_detected = False
        object_position = 0.0
        object_detection_frames = 0
    
    # During avoidance phases, keep object_detected True to maintain avoidance
    elif avoidance_phase != 'none':
        object_detected = True  # Keep avoidance active during navigation phases
    else:
        # When not in avoidance and no objects detected, ensure object_detected is False
        if not detected_objects:
            object_detected = False
    
    # Determine primary line position
    line_x = None
    confidence = 0.0
    primary_roi = None
    
    # During avoidance, use whole frame for line detection
    if avoidance_phase != 'none':
        # Check all zones for line during avoidance
        if bottom_confidence > 0.1:
            line_x = bottom_x
            confidence = bottom_confidence
            primary_roi = bottom_roi
        elif middle_confidence > 0.1:
            line_x = middle_x
            confidence = middle_confidence * 0.9
            primary_roi = middle_roi
        # Could add top zone line detection here if needed
    else:
        # Normal mode - use zone priorities
        if bottom_confidence > 0.3:
            # Strong bottom detection - use it
            line_x = bottom_x
            confidence = bottom_confidence
            primary_roi = bottom_roi
        elif middle_confidence > 0.4 and corner_warning:
            # Corner prediction mode - use middle detection
            line_x = middle_x
            confidence = middle_confidence * 0.8  # Reduce confidence for prediction
            primary_roi = middle_roi
        elif bottom_confidence > 0.1:
            # Weak bottom detection - still usable
            line_x = bottom_x
            confidence = bottom_confidence
            primary_roi = bottom_roi
    
    return {
        'line_x': line_x,
        'confidence': confidence,
        'primary_roi': primary_roi,
        'bottom_detection': (bottom_x, bottom_confidence),
        'middle_detection': (middle_x, middle_confidence),
        'detected_objects': detected_objects,
        'zones': {
            'bottom': (bottom_start, height),
            'middle': (middle_start, bottom_start),
            'top': (top_start, middle_start)
        }
    }

# -----------------------------------------------------------------------------
# --- Enhanced Movement Control with Object Avoidance ---
# -----------------------------------------------------------------------------
class SmartAvoidanceSystem:
    def __init__(self):
        self.maneuver_success = {}  # Learning from successful maneuvers
        self.current_strategy = 'none'
        self.strategy_start_time = 0
        self.learned_patterns = []
    
    def learn_successful_maneuver(self, obstacle_type, maneuver_sequence, success_score):
        """Learn from successful obstacle avoidance"""
        key = f"{obstacle_type}_{len(maneuver_sequence)}"
        
        if key not in self.maneuver_success:
            self.maneuver_success[key] = []
        
        self.maneuver_success[key].append({
            'sequence': maneuver_sequence,
            'score': success_score,
            'timestamp': time.time()
        })
        
        # Keep only best performing maneuvers
        self.maneuver_success[key] = sorted(
            self.maneuver_success[key], 
            key=lambda x: x['score'], 
            reverse=True
        )[:5]
    
    def get_smart_avoidance_command(self, detected_objects, line_offset, line_detected):
        """Intelligent avoidance based on mapping and learning"""
        if not detected_objects:
            if self.current_strategy != 'none':
                # Evaluate if last strategy was successful
                if line_detected and abs(line_offset) < 0.2:
                    success_score = 1.0 - abs(line_offset)
                    # Learn from this success (simplified)
                    pass
                self.current_strategy = 'none'
            return None
        
        main_obstacle = detected_objects[0]
        obstacle_pos = main_obstacle['position']
        
        # Check learned patterns first
        best_maneuver = self._get_best_learned_maneuver(main_obstacle)
        if best_maneuver:
            return best_maneuver
        
        # Intelligent decision based on mapping
        if abs(obstacle_pos) < 0.1:  # Center obstacle
            if self._path_clear_side('left'):
                self.current_strategy = 'left_sweep'
                return 'AVOID_LEFT'
            elif self._path_clear_side('right'):
                self.current_strategy = 'right_sweep'
                return 'AVOID_RIGHT'
        elif obstacle_pos < 0:  # Left side obstacle
            self.current_strategy = 'right_sweep'
            return 'AVOID_RIGHT'
        else:  # Right side obstacle
            self.current_strategy = 'left_sweep'
            return 'AVOID_LEFT'
        
        return 'FORWARD'  # Default safe action
    
    def _get_best_learned_maneuver(self, obstacle):
        """Get best learned maneuver for similar obstacles"""
        obstacle_type = obstacle.get('class_name', 'unknown')
        
        for key, maneuvers in self.maneuver_success.items():
            if obstacle_type in key and maneuvers:
                best = maneuvers[0]  # Highest scoring maneuver
                if best['score'] > 0.7:  # High confidence threshold
                    return best['sequence'][0] if best['sequence'] else None
        return None
    
    def _path_clear_side(self, side):
        """Check if path is historically clear on given side"""
        return True  # Simplified - always assume clear

smart_avoidance = SmartAvoidanceSystem()

# -----------------------------------------------------------------------------
# --- Distance Estimation System ---
# -----------------------------------------------------------------------------
class DistanceEstimator:
    def __init__(self):
        self.distance_history = {}  # Store distance history for each object
        self.focal_length = CAMERA_FOCAL_LENGTH
        self.camera_height = CAMERA_HEIGHT
        self.camera_angle = math.radians(CAMERA_ANGLE)
        
    def estimate_distance_by_height(self, object_data, frame_height):
        """
        Estimate distance using known object height and pinhole camera model
        Distance = (Known_Height * Focal_Length) / Pixel_Height
        """
        if 'bbox' not in object_data:
            return None
            
        # Get object class name for height lookup
        class_name = object_data.get('class_name', 'default')
        known_height = KNOWN_OBJECT_SIZES.get(class_name, KNOWN_OBJECT_SIZES['default'])
        
        # Get pixel height of object
        if 'height_ratio' in object_data:
            pixel_height = object_data['height_ratio'] * frame_height
        elif 'bbox' in object_data:
            bbox = object_data['bbox']
            pixel_height = bbox[3]  # Height from bbox
        else:
            return None
            
        if pixel_height <= 0:
            return None
            
        # Calculate distance using pinhole camera model
        distance = (known_height * self.focal_length) / pixel_height
        
        # Apply camera height and angle corrections for ground-based objects
        if class_name in ['person', 'car', 'bicycle', 'chair']:
            # Correct for camera height and viewing angle
            distance = distance * math.cos(self.camera_angle)
            
        return max(distance, 0.1)  # Minimum 10cm distance
    
    def estimate_distance_by_ground_plane(self, object_data, frame_height, frame_width):
        """
        Estimate distance using ground plane assumption
        For objects on the ground, y-coordinate relates to distance
        """
        if 'bbox' not in object_data:
            return None
            
        bbox = object_data['bbox']
        x, y, w, h = bbox
        
        # Object bottom y-coordinate (where it touches ground)
        object_bottom_y = y + h
        
        # Convert to normalized coordinates
        bottom_ratio = object_bottom_y / frame_height
        
        # Objects lower in frame are closer
        # This is a simplified model - you may need to calibrate
        if bottom_ratio > 0.8:  # Very bottom of frame
            return 0.5 + (1.0 - bottom_ratio) * 2.0  # 0.5-0.9m
        elif bottom_ratio > 0.6:  # Middle-bottom
            return 1.0 + (0.8 - bottom_ratio) * 5.0  # 1.0-2.0m
        else:  # Upper part of frame
            return 3.0 + (0.6 - bottom_ratio) * 10.0  # 3.0+m
    
    def estimate_distance_combined(self, object_data, frame_height, frame_width):
        """
        Combine multiple distance estimation methods for better accuracy
        """
        distances = []
        
        # Method 1: Height-based estimation
        dist_height = self.estimate_distance_by_height(object_data, frame_height)
        if dist_height and 0.1 < dist_height < 50.0:  # Reasonable range
            distances.append(dist_height)
            
        # Method 2: Ground plane estimation
        dist_ground = self.estimate_distance_by_ground_plane(object_data, frame_height, frame_width)
        if dist_ground and 0.1 < dist_ground < 50.0:
            distances.append(dist_ground)
            
        if not distances:
            return None
            
        # Take weighted average (prefer height-based if available)
        if len(distances) == 1:
            return distances[0]
        else:
            # Weight height-based method more heavily for known objects
            class_name = object_data.get('class_name', 'default')
            if class_name in KNOWN_OBJECT_SIZES:
                return 0.7 * distances[0] + 0.3 * distances[1]  # Height weighted
            else:
                return 0.4 * distances[0] + 0.6 * distances[1]  # Ground weighted
    
    def update_distance_history(self, object_id, distance):
        """
        Maintain rolling average of distances for stability
        """
        if object_id not in self.distance_history:
            self.distance_history[object_id] = deque(maxlen=DISTANCE_HISTORY_SIZE)
            
        self.distance_history[object_id].append(distance)
        
        # Return smoothed distance
        return sum(self.distance_history[object_id]) / len(self.distance_history[object_id])
    
    def get_object_distance(self, object_data, frame_height, frame_width, object_id=None):
        """
        Main method to get distance to an object
        """
        distance = self.estimate_distance_combined(object_data, frame_height, frame_width)
        
        if distance is None:
            return None
            
        # Apply smoothing if object ID provided
        if object_id is not None:
            distance = self.update_distance_history(object_id, distance)
            
        return distance
    
    def assess_threat_level(self, distance, object_position):
        """
        Assess threat level based on distance and position
        Returns: 'safe', 'warning', 'danger', 'emergency'
        """
        if distance is None:
            return 'unknown'
            
        # Consider both distance and lateral position
        lateral_threat = abs(object_position)  # How much in path (0=center, 1=edge)
        
        if distance > WARNING_DISTANCE:
            return 'safe'
        elif distance > SAFE_DISTANCE:
            if lateral_threat < 0.3:  # Directly in path
                return 'warning'
            else:
                return 'safe'
        elif distance > EMERGENCY_DISTANCE:
            if lateral_threat < 0.5:
                return 'danger'
            else:
                return 'warning'
        else:
            return 'emergency'

# -----------------------------------------------------------------------------
# --- Intelligent Obstacle Mapping and Path Planning System ---
# -----------------------------------------------------------------------------
class ObstacleMapper:
    def __init__(self):
        self.obstacle_memory = {}  # Persistent obstacle memory
        self.current_obstacles = []  # Currently detected obstacles
        self.last_cleanup = time.time()
        
    def add_obstacle(self, obstacle_data, line_position=None):
        """Add or update an obstacle in memory with geometry tracking"""
        obstacle_id = self._generate_obstacle_id(obstacle_data)
        current_time = time.time()
        
        # Calculate obstacle geometry
        width_ratio = obstacle_data.get('width_ratio', 0)
        height_ratio = obstacle_data.get('height_ratio', 0)
        position = obstacle_data.get('position', 0)
        
        # Enhanced obstacle info with geometry
        obstacle_info = {
            'position': position,
            'width_ratio': width_ratio,
            'height_ratio': height_ratio,
            'first_seen': current_time,
            'last_seen': current_time,
            'confidence': obstacle_data.get('confidence', 0.5),
            'class_name': obstacle_data.get('class_name', 'unknown'),
            'line_position_when_detected': line_position,
            'avoidance_count': self.obstacle_memory.get(obstacle_id, {}).get('avoidance_count', 0),
            'estimated_real_width': self._estimate_real_size(width_ratio, height_ratio),
            'significance_score': self._calculate_significance(obstacle_data)
        }
        
        self.obstacle_memory[obstacle_id] = obstacle_info
        return obstacle_id
    
    def calculate_avoidance_path(self, obstacle_data, line_position=None):
        """Calculate optimal path around obstacle based on its geometry"""
        if not PATH_CALCULATION_ENABLED:
            return self._simple_avoidance_decision(obstacle_data)
        
        width = obstacle_data.get('width_ratio', 0.2)
        height = obstacle_data.get('height_ratio', 0.15)
        position = obstacle_data.get('position', 0)
        
        # Calculate obstacle boundaries with safety margin
        obstacle_left = position - (width * SAFETY_MARGIN) / 2
        obstacle_right = position + (width * SAFETY_MARGIN) / 2
        
        # Determine best side to avoid (prefer side with more space)
        left_space = abs(obstacle_left - (-1.0))  # Space on left side
        right_space = abs(1.0 - obstacle_right)   # Space on right side
        
        # Calculate path parameters based on obstacle size
        estimated_clear_distance = max(width * SAFETY_MARGIN, 0.3)  # Minimum safe distance
        estimated_clear_frames = int(estimated_clear_distance * 25)  # Rough estimate
        
        # Choose side and calculate path
        if left_space > right_space:
            side = 'left'
            turn_magnitude = min(abs(obstacle_left), 0.7)  # Limit turn strength
        else:
            side = 'right' 
            turn_magnitude = min(abs(obstacle_right), 0.7)
        
        path_plan = {
            'side': side,
            'turn_magnitude': turn_magnitude,
            'estimated_clear_frames': max(estimated_clear_frames, GENTLE_CLEAR_DURATION),
            'obstacle_width': width,
            'obstacle_height': height,
            'safety_margin_used': SAFETY_MARGIN,
            'confidence': min(obstacle_data.get('confidence', 0.5) * 1.2, 1.0)
        }
        
        logger.info(f"PATH CALCULATED: {side} side, magnitude: {turn_magnitude:.2f}, "
                   f"clear frames: {estimated_clear_frames}, obstacle size: {width:.2f}x{height:.2f}")
        
        return path_plan
    
    def is_obstacle_cleared(self, original_obstacle, current_detections):
        """Determine if we've successfully navigated around the obstacle"""
        if not current_detections:
            return True
        
        original_pos = original_obstacle.get('position', 0)
        original_width = original_obstacle.get('width_ratio', 0.2)
        
        # Check if any current detection matches our original obstacle
        for detection in current_detections:
            detection_pos = detection.get('position', 0)
            position_diff = abs(detection_pos - original_pos)
            
            # If we see something in similar position, obstacle not cleared
            if position_diff < original_width:
                return False
        
        return True
    
    def cleanup_old_obstacles(self):
        """Remove obstacles that haven't been seen recently"""
        current_time = time.time()
        if current_time - self.last_cleanup < 5.0:  # Cleanup every 5 seconds
            return
        
        expired_obstacles = []
        for obstacle_id, obstacle_info in self.obstacle_memory.items():
            if current_time - obstacle_info['last_seen'] > OBSTACLE_MEMORY_TIMEOUT:
                expired_obstacles.append(obstacle_id)
        
        for obstacle_id in expired_obstacles:
            del self.obstacle_memory[obstacle_id]
            
        self.last_cleanup = current_time
        
        if expired_obstacles:
            logger.info(f"🧹 Cleaned up {len(expired_obstacles)} old obstacles from memory")
    
    def _generate_obstacle_id(self, obstacle_data):
        """Generate a unique ID for obstacle tracking"""
        pos = obstacle_data.get('position', 0)
        width = obstacle_data.get('width_ratio', 0)
        height = obstacle_data.get('height_ratio', 0)
        
        # Create ID based on position and size (rounded for stability)
        return f"{round(pos, 1)}_{round(width, 2)}_{round(height, 2)}"
    
    def _estimate_real_size(self, width_ratio, height_ratio):
        """Estimate real-world size category"""
        total_size = width_ratio * height_ratio
        
        if total_size > 0.15:
            return "large"
        elif total_size > 0.08:
            return "medium"
        else:
            return "small"
    
    def _calculate_significance(self, obstacle_data):
        """Calculate how significant this obstacle is for avoidance"""
        width = obstacle_data.get('width_ratio', 0)
        height = obstacle_data.get('height_ratio', 0)
        position = obstacle_data.get('position', 0)
        confidence = obstacle_data.get('confidence', 0.5)
        
        # Size significance
        size_score = (width * height) * 5  # Bigger = more significant
        
        # Position significance (center is more significant)
        position_score = 1.0 - abs(position)  # Closer to center = more significant
        
        # Confidence significance
        confidence_score = confidence
        
        total_significance = (size_score + position_score + confidence_score) / 3
        return min(total_significance, 1.0)
    
    def _simple_avoidance_decision(self, obstacle_data):
        """Simple left/right decision when path calculation is disabled"""
        position = obstacle_data.get('position', 0)
        
        return {
            'side': 'left' if position > 0 else 'right',
            'turn_magnitude': 0.5,
            'estimated_clear_frames': GENTLE_CLEAR_DURATION,
            'confidence': 0.7
        }
    
    def get_memory_stats(self):
        """Get statistics about obstacle memory"""
        return {
            'total_obstacles': len(self.obstacle_memory),
            'recent_obstacles': len([o for o in self.obstacle_memory.values() 
                                   if time.time() - o['last_seen'] < 10]),
            'avg_significance': sum(o.get('significance_score', 0) 
                                  for o in self.obstacle_memory.values()) / 
                                max(len(self.obstacle_memory), 1)
        }

# Initialize obstacle mapper
obstacle_mapper = ObstacleMapper()
distance_estimator = DistanceEstimator()

def get_turn_command_with_avoidance(steering, avoid_objects=False, line_detected_now=False, line_offset_now=0.0, detected_objects_list=None):
    """SIMPLE 3-STEP AVOIDANCE: Turn right → Go forward → Turn left back to line"""
    global avoidance_phase, avoidance_side, avoidance_frame_count, avoidance_duration
    
    # SIMPLE OBSTACLE AVOIDANCE - NO COMPLEX MAPPING
    if avoid_objects and (detected_objects_list or avoidance_phase != 'none'):
        
        # START SIMPLE AVOIDANCE
        if avoidance_phase == 'none' and detected_objects_list:
            logger.info("🚨 OBSTACLE DETECTED - Starting simple avoidance: RIGHT → FORWARD → LEFT")
            avoidance_phase = 'turning_right'
            avoidance_duration = 3  # Turn right for 3 frames (0.6 seconds)
            return COMMANDS['AVOID_RIGHT']
        
        # STEP 1: Turn RIGHT away from obstacle  
        elif avoidance_phase == 'turning_right':
            avoidance_duration -= 1
            if avoidance_duration > 0:
                logger.info(f"STEP 1: Turning RIGHT away from obstacle ({avoidance_duration} frames left)")
                return COMMANDS['AVOID_RIGHT']
            else:
                # Go to step 2
                avoidance_phase = 'going_forward'
                avoidance_duration = 6  # Go forward for 6 frames (~1.2 seconds)
                logger.info("STEP 2: Going FORWARD to clear obstacle")
                return COMMANDS['FORWARD']
        
        # STEP 2: Go FORWARD past obstacle
        elif avoidance_phase == 'going_forward':
            avoidance_duration -= 1
            if avoidance_duration > 0:
                logger.info(f"STEP 2: Going FORWARD past obstacle ({avoidance_duration} frames left)")
                return COMMANDS['FORWARD']
            else:
                # Go to step 3
                avoidance_phase = 'turning_left'
                avoidance_duration = 6  # Turn left for 6 frames to get back to line
                logger.info("STEP 3: Turning LEFT back to line")
                return COMMANDS['AVOID_LEFT']
        
        # STEP 3: Turn LEFT back to line
        elif avoidance_phase == 'turning_left':
            # Check if we found the line early
            if line_detected_now and abs(line_offset_now) < 0.4:
                logger.info("✅ LINE FOUND! Avoidance complete - back to line following")
                avoidance_phase = 'none'
                return COMMANDS['FORWARD']
            
            avoidance_duration -= 1
            if avoidance_duration > 0:
                logger.info(f"STEP 3: Turning LEFT back to line ({avoidance_duration} frames left)")
                return COMMANDS['AVOID_LEFT']
            else:
                # Avoidance complete
                logger.info("✅ Simple avoidance complete - resuming line search")
                avoidance_phase = 'none'
                return COMMANDS['FORWARD']
    
    # NORMAL LINE FOLLOWING BEHAVIOR
    if abs(steering) < STEERING_DEADZONE:
        return COMMANDS['FORWARD']
    
    if abs(steering) > 0.45:
        return COMMANDS['RIGHT'] if steering < 0 else COMMANDS['LEFT']
    
    return COMMANDS['RIGHT'] if steering < 0 else COMMANDS['LEFT']

def opposite_direction(direction):
    """Helper function to get opposite direction"""
    return 'left' if direction == 'right' else 'right'

def get_turn_command_without_avoidance(steering):
    """Get normal turn command without avoidance logic"""
    if abs(steering) < STEERING_DEADZONE:
        return COMMANDS['FORWARD']
    
    if abs(steering) > 0.45:
        return COMMANDS['RIGHT'] if steering < 0 else COMMANDS['LEFT']
    
    return COMMANDS['RIGHT'] if steering < 0 else COMMANDS['LEFT']

# -----------------------------------------------------------------------------
# --- Enhanced Visualization ---
# -----------------------------------------------------------------------------
def draw_debug_info(frame, detection_data):
    """Draw comprehensive debug information including all zones and detections"""
    height, width = frame.shape[:2]
    
    if detection_data is None:
        return
    
    zones = detection_data.get('zones', {})
    detected_objects = detection_data.get('detected_objects', [])
    bottom_detection = detection_data.get('bottom_detection', (None, 0))
    middle_detection = detection_data.get('middle_detection', (None, 0))
    line_x = detection_data.get('line_x')
    
    # Draw zone boundaries
    if 'bottom' in zones:
        bottom_start, bottom_end = zones['bottom']
        cv2.rectangle(frame, (0, bottom_start), (width, bottom_end), (0, 255, 255), 2)
        cv2.putText(frame, "BOTTOM: Line Following", (5, bottom_start + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    if 'middle' in zones:
        middle_start, middle_end = zones['middle']
        cv2.rectangle(frame, (0, middle_start), (width, middle_end), (0, 255, 128), 2)
        cv2.putText(frame, "MIDDLE: Corner Prediction", (5, middle_start + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 128), 1)
    
    if 'top' in zones:
        top_start, top_end = zones['top']
        cv2.rectangle(frame, (0, top_start), (width, top_end), (128, 255, 0), 2)
        cv2.putText(frame, "TOP: Object Detection", (5, top_start + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 255, 0), 1)
    
    # Draw center line
    center_x = width // 2
    cv2.line(frame, (center_x, 0), (center_x, height), (200, 200, 200), 1)
    
    # Draw detected objects
    for obj in detected_objects:
        bbox = obj['bbox']
        x, y, w, h = bbox
        # Adjust y coordinate for top zone
        y_adjusted = zones.get('top', (0, 0))[0] + y
        
        # Draw object label (YOLO class name if available, otherwise generic)
        if 'class_name' in obj:
            label = f"{obj['class_name']} ({obj['confidence']:.2f})"
        else:
            label = "OBJECT"
        
        # Add distance and threat level to label
        if 'estimated_distance' in obj and obj['estimated_distance']:
            label += f" {obj['estimated_distance']:.1f}m"
        if 'threat_level' in obj:
            threat = obj['threat_level']
            threat_colors = {
                'safe': (0, 255, 0),       # Green
                'warning': (0, 255, 255),  # Yellow  
                'danger': (0, 165, 255),   # Orange
                'emergency': (0, 0, 255)   # Red
            }
            label += f" [{threat.upper()}]"
            bbox_color = threat_colors.get(threat, (0, 0, 255))
        else:
            bbox_color = (0, 0, 255)
        
        # Draw bounding box with threat-level color
        cv2.rectangle(frame, (x, y_adjusted), (x + w, y_adjusted + h), bbox_color, 2)
        
        cv2.putText(frame, label, (x, y_adjusted - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, bbox_color, 1)
        
        # Draw warning indicator
        cv2.putText(frame, "OBSTACLE DETECTED", (width//2 - 100, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw object position indicator
        obj_center_x = int(width/2 + obj['position'] * width/2)
        cv2.circle(frame, (obj_center_x, 60), 8, (0, 0, 255), -1)
    
    # Draw corner warning
    if corner_warning:
        cv2.putText(frame, "CORNER AHEAD", (width//2 - 80, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Draw line detections
    if bottom_detection[0] is not None:
        bottom_x, bottom_conf = bottom_detection
        bottom_y = zones.get('bottom', (height-50, height))[0] + 25
        cv2.circle(frame, (bottom_x, bottom_y), 8, (0, 255, 255), -1)
        cv2.putText(frame, f"B:{bottom_conf:.2f}", (bottom_x - 20, bottom_y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    if middle_detection[0] is not None:
        middle_x, middle_conf = middle_detection
        middle_y = zones.get('middle', (height//2, height//2))[0] + 15
        cv2.circle(frame, (middle_x, middle_y), 6, (0, 255, 128), -1)
        cv2.putText(frame, f"M:{middle_conf:.2f}", (middle_x - 20, middle_y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 128), 1)
    
    # Draw primary line detection
    if line_x is not None:
        primary_y = zones.get('bottom', (height-50, height))[0] + 25
        cv2.circle(frame, (line_x, primary_y), 12, (255, 0, 255), 3)
        cv2.line(frame, (center_x, primary_y), (line_x, primary_y), (255, 0, 255), 3)
        
        # Draw offset information
        offset_text = f"Offset: {line_offset:.2f}"
        cv2.putText(frame, offset_text, (line_x - 40, primary_y - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    
    # Draw status information
    status_color = (0, 255, 0) if line_detected else (0, 0, 255)
    info_lines = [
        f"Status: {robot_status}",
        f"Offset: {line_offset:.2f}",
        f"Confidence: {confidence:.2f}",
        f"Command: {turn_command}",
        f"FPS: {current_fps:.1f}",
        f"Objects: {len(detected_objects)}"
    ]
    
    for i, line in enumerate(info_lines):
        y_pos = 20 + (i * 20)
        cv2.putText(frame, line, (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color if i == 0 else (255, 255, 255), 1)
    
    # Draw avoidance status
    if avoidance_phase != 'none':
        phase_text = f"AVOIDANCE: {avoidance_phase.upper()} {avoidance_side.upper()}"
        cv2.putText(frame, phase_text, (width//2 - 120, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    # Draw smart avoidance status
    if smart_avoidance.current_strategy != 'none':
        strategy_text = f"SMART AVOIDANCE: {smart_avoidance.current_strategy.upper()}"
        cv2.putText(frame, strategy_text, (width//2 - 120, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 2)
    
    # Draw direction arrow with enhanced colors for avoidance
    arrow_y = height - 30
    arrow_color = (0, 255, 0)  # Green for forward
    arrow_text = "FORWARD"
    
    if turn_command == COMMANDS['LEFT']:
        arrow_color = (0, 255, 255)  # Yellow for left
        arrow_text = "LEFT"
    elif turn_command == COMMANDS['RIGHT']:
        arrow_color = (255, 255, 0)  # Cyan for right
        arrow_text = "RIGHT"
    elif turn_command == COMMANDS['AVOID_LEFT']:
        arrow_color = (255, 0, 255)  # Magenta for avoid left
        arrow_text = "AVOID LEFT"
    elif turn_command == COMMANDS['AVOID_RIGHT']:
        arrow_color = (255, 0, 255)  # Magenta for avoid right
        arrow_text = "AVOID RIGHT"
    elif turn_command == COMMANDS['STOP']:
        arrow_color = (0, 0, 255)  # Red for stop
        arrow_text = "STOP"
    
    cv2.putText(frame, arrow_text, (width // 2 - 50, arrow_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, arrow_color, 2)
    
    # Draw offset line
    if line_x is not None and 'zones' in detection_data:
        bottom_y = detection_data['zones'].get('bottom', (height-50, height))[0] + 25
        offset_text = f"Offset: {line_offset:.2f}"
        cv2.putText(frame, offset_text, (line_x - 40, bottom_y - 15),
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
    <title>Smart Line Follower Robot Dashboard</title>
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
        
        .smart-features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .feature-section {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .feature-section h3 {
            margin-bottom: 15px;
            color: #fff;
            font-size: 1.2em;
        }
        
        .pid-display {
            display: flex;
            justify-content: space-around;
            gap: 15px;
        }
        
        .pid-param {
            text-align: center;
            padding: 10px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            flex: 1;
        }
        
        .param-label {
            display: block;
            font-size: 0.9em;
            opacity: 0.8;
            margin-bottom: 5px;
        }
        
        .param-value {
            display: block;
            font-size: 1.4em;
            font-weight: bold;
            color: #4CAF50;
        }
        
        .performance-display {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        
        #performance-chart {
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 8px;
            background: #222;
        }
        
        .performance-stats {
            flex: 1;
        }
        
        .performance-stats div {
            margin-bottom: 10px;
            padding: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
        }
        
        .learning-display div {
            margin-bottom: 8px;
            padding: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
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
            <h1>Smart Line Follower Robot</h1>
            <div>Control Dashboard <span id="status-dot" class="status-indicator status-offline"></span></div>
        </div>
        
        <div class="dashboard">
            <div class="video-section">
                <h3 style="margin-bottom: 15px;">Live Camera Feed</h3>
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
        
        <div class="smart-features">
            <div class="feature-section">
                <h3>Adaptive PID Controller</h3>
                <div class="pid-display">
                    <div class="pid-param">
                        <span class="param-label">KP:</span>
                        <span class="param-value" id="pid-kp">0.500</span>
                    </div>
                    <div class="pid-param">
                        <span class="param-label">KI:</span>
                        <span class="param-value" id="pid-ki">0.010</span>
                    </div>
                    <div class="pid-param">
                        <span class="param-label">KD:</span>
                        <span class="param-value" id="pid-kd">0.100</span>
                    </div>
                </div>
            </div>
            
            <div class="feature-section">
                <h3>Performance Monitor</h3>
                <div class="performance-display">
                    <canvas id="performance-chart" width="250" height="150"></canvas>
                    <div class="performance-stats">
                        <div>Average Error: <span id="avg-performance">0.00</span></div>
                        <div>Current Strategy: <span id="current-strategy">none</span></div>
                    </div>
                </div>
            </div>
            
            <div class="feature-section">
                <h3>Learning Progress</h3>
                <div class="learning-display">
                    <div>Learned Maneuvers: <span id="learned-maneuvers">0</span></div>
                    <div id="maneuver-list"></div>
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
                    
                    // Update PID parameters
                    if (data.pid_params) {
                        document.getElementById('pid-kp').textContent = data.pid_params.kp.toFixed(3);
                        document.getElementById('pid-ki').textContent = data.pid_params.ki.toFixed(3);
                        document.getElementById('pid-kd').textContent = data.pid_params.kd.toFixed(3);
                    }
                    
                    // Update smart avoidance info
                    if (data.smart_avoidance) {
                        document.getElementById('avg-performance').textContent = data.smart_avoidance.performance_avg.toFixed(3);
                        document.getElementById('current-strategy').textContent = data.smart_avoidance.strategy;
                        document.getElementById('learned-maneuvers').textContent = data.smart_avoidance.learned_maneuvers;
                    }
                    
                    // Update connection statuses
                    const statusDot = document.getElementById('status-dot');
                    const espStatus = document.getElementById('esp-status');
                    const cameraStatus = document.getElementById('camera-status');
                    
                    if (data.esp_connected) {
                        statusDot.className = 'status-indicator status-online';
                        espStatus.innerHTML = 'Connected';
                    } else {
                        statusDot.className = 'status-indicator status-offline';
                        espStatus.innerHTML = 'Disconnected';
                    }
                    
                    // Camera status based on recent updates
                    const timeSinceUpdate = Date.now() - lastUpdateTime;
                    if (timeSinceUpdate < 2000) {
                        cameraStatus.innerHTML = 'Active';
                    } else {
                        cameraStatus.innerHTML = 'No Signal';
                    }
                    
                    lastUpdateTime = Date.now();
                })
                .catch(error => {
                    console.error('Error fetching status:', error);
                    document.getElementById('robot-status').textContent = 'Connection Error';
                    document.getElementById('status-dot').className = 'status-indicator status-offline';
                });
        }
        
        // Update performance visualization
        let performanceData = [];
        function updatePerformance() {
            fetch('/api/learning')
                .then(response => response.json())
                .then(data => {
                    // Draw performance trend chart
                    const canvas = document.getElementById('performance-chart');
                    const ctx = canvas.getContext('2d');
                    
                    // Clear canvas
                    ctx.fillStyle = '#222';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    
                    // Update performance data
                    if (data.performance_trend && data.performance_trend.length > 0) {
                        performanceData = data.performance_trend;
                        
                        // Draw performance trend line
                        ctx.strokeStyle = '#4CAF50';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        
                        const maxY = Math.max(...performanceData, 0.1);
                        performanceData.forEach((value, index) => {
                            const x = (index / performanceData.length) * canvas.width;
                            const y = canvas.height - (value / maxY) * canvas.height;
                            
                            if (index === 0) {
                                ctx.moveTo(x, y);
                            } else {
                                ctx.lineTo(x, y);
                            }
                        });
                        ctx.stroke();
                        
                        // Draw grid lines
                        ctx.strokeStyle = '#444';
                        ctx.lineWidth = 1;
                        for (let i = 0; i < 5; i++) {
                            const y = (i / 4) * canvas.height;
                            ctx.beginPath();
                            ctx.moveTo(0, y);
                            ctx.lineTo(canvas.width, y);
                            ctx.stroke();
                        }
                    }
                    
                    // Update learning progress
                    const maneuverList = document.getElementById('maneuver-list');
                    maneuverList.innerHTML = '';
                    data.learned_maneuvers.forEach(maneuver => {
                        const div = document.createElement('div');
                        div.innerHTML = `${maneuver.type}: ${maneuver.attempts} attempts (${(maneuver.best_score * 100).toFixed(1)}% success)`;
                        maneuverList.appendChild(div);
                    });
                })
                .catch(error => {
                    console.error('Error fetching performance data:', error);
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
        setInterval(updateStatus, 500);      // Update every 500ms
        setInterval(updatePerformance, 2000); // Update performance every 2 seconds
        setInterval(updateUptime, 1000);     // Update uptime every second
        
        // Initial load
        updateStatus();
        updatePerformance();
        updateUptime();
        
        // Handle image load events
        const videoFeed = document.querySelector('.video-feed');
        videoFeed.addEventListener('load', () => {
            document.getElementById('camera-status').innerHTML = 'Active';
        });
        
        videoFeed.addEventListener('error', () => {
            document.getElementById('camera-status').innerHTML = 'Error';
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
    global robot_stats, smart_avoidance
    
    # Calculate uptime
    current_time = time.time()
    if 'start_time' not in robot_stats:
        robot_stats['start_time'] = current_time
    
    uptime = current_time - robot_stats['start_time']
    
    # Get PID parameters from global controller
    pid_params = {'kp': KP, 'ki': KI, 'kd': KD}  # Default values
    if pid_controller is not None:
        try:
            pid_params['kp'], pid_params['ki'], pid_params['kd'] = pid_controller.get_params()
        except:
            pass
    
    # Get smart avoidance status and obstacle mapping stats
    avoidance_info = {
        'strategy': smart_avoidance.current_strategy,
        'learned_maneuvers': len(smart_avoidance.maneuver_success),
        'performance_avg': sum(performance_history) / len(performance_history) if performance_history else 0
    }
    
    # Get obstacle mapping statistics
    obstacle_stats = obstacle_mapper.get_memory_stats()
    obstacle_info = {
        'total_mapped': obstacle_stats['total_obstacles'],
        'recent_obstacles': obstacle_stats['recent_obstacles'],
        'avg_significance': obstacle_stats['avg_significance'],
        'current_phase': avoidance_phase,
        'planned_path_side': planned_path.get('side', 'none') if planned_path else 'none',
        'planned_path_estimated_clear_frames': planned_path.get('estimated_clear_frames', 0) if planned_path else 0,
        'last_avoidance_clear_distance': last_avoidance_clear_distance
    }
    
    # Cast values to native types to ensure JSON serializability
    response = {
        'status': robot_status,
        'command': turn_command,
        'offset': float(line_offset),
        'fps': float(current_fps),
        'confidence': float(confidence),
        'line_detected': bool(line_detected),
        'esp_connected': bool(esp_connected),
        'uptime': float(uptime),
        'total_frames': int(robot_stats.get('total_frames', 0)),
        'corner_count': int(robot_stats.get('corner_count', 0)),
        'objects_detected': int(robot_stats.get('objects_detected', 0)),
        'avoidance_maneuvers': int(robot_stats.get('avoidance_maneuvers', 0)),
        'corner_warning': bool(corner_warning),
        'object_detected': bool(object_detected),
        'smart_avoidance': {
            'strategy': avoidance_info['strategy'],
            'learned_maneuvers': int(avoidance_info['learned_maneuvers']),
            'performance_avg': float(avoidance_info['performance_avg'])
        },
        'obstacle_mapping': {
            'total_mapped': int(obstacle_info['total_mapped']),
            'recent_obstacles': int(obstacle_info['recent_obstacles']),
            'avg_significance': float(obstacle_info['avg_significance']),
            'current_phase': obstacle_info['current_phase'],
            'planned_path_side': obstacle_info['planned_path_side'],
            'planned_path_estimated_clear_frames': int(obstacle_info['planned_path_estimated_clear_frames']),
            'last_avoidance_clear_distance': float(obstacle_info['last_avoidance_clear_distance'])
        },
        'pid_params': {
            'kp': float(pid_params['kp']),
            'ki': float(pid_params['ki']),
            'kd': float(pid_params['kd'])
        }
    }
    return jsonify(response)

@app.route('/api/learning')
def api_learning():
    """Get learning and performance data"""
    global smart_avoidance, performance_history
    
    # Get learning data
    learning_data = []
    for maneuver_type, maneuvers in smart_avoidance.maneuver_success.items():
        if maneuvers:
            best_score = max(m['score'] for m in maneuvers)
            learning_data.append({
                'type': maneuver_type,
                'attempts': len(maneuvers),
                'best_score': best_score
            })
    
    # Get recent performance trend
    recent_performance = list(performance_history)[-20:] if len(performance_history) >= 20 else list(performance_history)
    
    return jsonify({
        'learned_maneuvers': learning_data,
        'current_strategy': smart_avoidance.current_strategy,
        'performance_trend': recent_performance
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
    global line_detected, current_fps, confidence, esp_connection, robot_stats, yolo_model, smart_avoidance
    
    logger.info("Starting Smart Line Follower Robot")
    robot_status = "Starting camera"
    robot_stats['start_time'] = time.time()
    
    # Initialize YOLO model if available and enabled
    if USE_YOLO and YOLO_AVAILABLE and OBJECT_DETECTION_ENABLED:
        try:
            logger.info(f"Loading YOLO11n model: {YOLO_MODEL_SIZE}")
            robot_status = "Loading YOLO model"
            yolo_model = YOLO(YOLO_MODEL_SIZE)
            logger.info("YOLO11n model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            logger.info("Falling back to traditional computer vision detection")
            yolo_model = None
    else:
        if not OBJECT_DETECTION_ENABLED:
            logger.info("Object detection disabled")
        elif not USE_YOLO:
            logger.info("Using traditional computer vision detection")
        elif not YOLO_AVAILABLE:
            logger.info("YOLO not available, using traditional detection")
    
    # Initialize smart avoidance system
    if OBJECT_DETECTION_ENABLED:
        logger.info("Initializing smart avoidance with live mapping")
        robot_status = "Loading smart avoidance"
    else:
        logger.info("Object avoidance disabled")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Try camera 0 first
    if not cap.isOpened():
        logger.warning("Camera 0 failed, trying camera 1")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            logger.error("Failed to open any camera")
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
    
    # Initialize adaptive PID controller
    global pid_controller
    pid = AdaptivePID(KP, KI, KD, MAX_INTEGRAL)
    pid_controller = pid
    
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
    
    # History for smoothing (balanced for responsive yet smooth following)
    offset_history = deque(maxlen=3)
    steering_history = deque(maxlen=3)
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
            
            # Process image with multi-zone detection
            detection_data = process_image_multi_zone(frame)
            line_x = detection_data['line_x']
            confidence = detection_data['confidence']
            detected_objects = detection_data['detected_objects']
            
            # Create a copy for visualization
            display_frame = frame.copy()
            
            # Track performance for learning
            performance_history.append(abs(line_offset) if line_detected else 1.0)
            
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
                
                # Use weighted average of recent offsets for stability
                if len(offset_history) > 0:
                    # Give more weight to recent measurements
                    weights = [0.5, 0.3, 0.2][:len(offset_history)]
                    weighted_sum = sum(w * offset for w, offset in zip(weights, reversed(offset_history)))
                    weight_total = sum(weights[:len(offset_history)])
                    line_offset = weighted_sum / weight_total
                else:
                    line_offset = raw_offset
                
                # Remember this offset as last known good position
                last_known_good_offset = line_offset
                
                # Check for corner (enhanced with prediction)
                if abs(line_offset) > 0.4 and confidence > 0.3:
                    corner_detected_count += 1
                    
                    # Adjust status based on corner detection duration
                    if corner_detected_count > 5:
                        robot_status = f"Taking corner ({corner_detected_count})"
                        
                        # For sharp corners, exaggerate the steering to make tighter turns
                        if abs(line_offset) > 0.5:
                            line_offset *= 1.5  # Increase turning response
                    else:
                        robot_status = f" Corner detected ({corner_detected_count})"
                elif corner_warning:
                    robot_status = f"Corner predicted ahead"
                else:
                    if corner_detected_count > 0:
                        corner_detected_count -= 1
                    else:
                        corner_detected_count = 0
                        
                    # Enhanced status with intelligent obstacle mapping phases
                    if avoidance_phase == 'mapping':
                        robot_status = f"MAPPING: Analyzing obstacle geometry, gentle turn {avoidance_side}"
                    elif avoidance_phase == 'clearing':
                        robot_status = f"ROUNDABOUT: Initial curve forward ({avoidance_duration} frames)"
                    elif avoidance_phase == 'curving':
                        robot_status = f"ROUNDABOUT: Continuing curve around obstacle ({avoidance_duration} frames)"
                    elif avoidance_phase == 'completing':
                        robot_status = f"ROUNDABOUT: Completing curve forward ({avoidance_duration} frames)"
                    elif avoidance_phase == 'returning':
                        if confidence > 0.3:
                            robot_status = f"RETURNING: Line detected! Smart transition (offset: {line_offset:.2f})"
                        else:
                            robot_status = f"RETURNING: Gentle turn {opposite_direction(avoidance_side)} to find line"
                    elif object_detected:
                        robot_status = f"Object mapped - Calculating smart path"
                    elif detected_objects:
                        robot_status = f"Obstacle candidate detected"
                    else:
                        robot_status = f"Following line (C:{confidence:.2f})"
                
                # Calculate steering using PID controller
                # Negative offset means line is to the left, so invert for steering
                steering_error = -line_offset
                raw_steering = pid.calculate(steering_error)
                
                # Apply light exponential smoothing for responsive control
                alpha = 0.7  # Smoothing factor (slightly increased for more responsiveness)
                if hasattr(main, 'last_steering'):
                    steering_value = alpha * raw_steering + (1 - alpha) * main.last_steering
                else:
                    steering_value = raw_steering
                main.last_steering = steering_value
                
                # Add to steering history for light additional smoothing
                steering_history.append(steering_value)
                
                # Use simple average of recent steering values (more responsive)
                if len(steering_history) > 0:
                    avg_steering = sum(steering_history) / len(steering_history)
                else:
                    avg_steering = steering_value
                
                # Convert steering to command with DYNAMIC NAVIGATION
                should_avoid = OBJECT_DETECTION_ENABLED and object_detected
                if should_avoid and avoidance_phase != 'none':
                    if avoidance_phase == 'turning':
                        logger.info(f"DYNAMIC NAVIGATION! Phase: {avoidance_phase.upper()} - frame {avoidance_duration}")
                    else:
                        logger.info(f"DYNAMIC NAVIGATION! Phase: {avoidance_phase.upper()} - {avoidance_duration} frames left")
                
                turn_command = get_turn_command_with_avoidance(avg_steering, 
                                                             avoid_objects=should_avoid,
                                                             line_detected_now=True,
                                                             line_offset_now=line_offset,
                                                             detected_objects_list=detected_objects)
                
                logger.debug(f"Line at x={line_x}, offset={line_offset:.2f}, steering={steering_value:.2f}")
                
            else:
                # Line not detected
                line_detected = False
                # If we're in an avoidance sequence, always defer to FSM
                if avoidance_phase != 'none':
                    turn_command = get_turn_command_with_avoidance(
                        0.0,
                        avoid_objects=True,
                        line_detected_now=False,
                        line_offset_now=0.0,
                        detected_objects_list=detected_objects if 'detected_objects' in locals() else []
                    )
                    robot_status = f"AVOIDING: {avoidance_phase}"
                else:
                    # Regular search mode
                    search_counter += 1
                    if search_counter < 5:
                        robot_status = "🔍 Searching for line (brief gap)"
                    elif search_counter < 15:
                        if last_known_good_offset < 0:
                            turn_command = COMMANDS['RIGHT']
                            robot_status = "Searching right (last seen left)"
                        else:
                            turn_command = COMMANDS['LEFT']
                            robot_status = "Searching left (last seen right)"
                    elif search_counter < 30:
                        turn_command = COMMANDS['RIGHT'] if turn_command == COMMANDS['LEFT'] else COMMANDS['LEFT']
                        robot_status = f"Searching opposite direction ({search_counter})"
                    elif search_counter < 45:
                        should_avoid = OBJECT_DETECTION_ENABLED and object_detected
                        if should_avoid:
                            logger.info(f"AVOIDANCE DURING SEARCH! Duration: {avoidance_duration} frames")
                        turn_command = get_turn_command_with_avoidance(
                            0.0, avoid_objects=should_avoid,
                            line_detected_now=False, line_offset_now=0.0,
                            detected_objects_list=detected_objects if 'detected_objects' in locals() else []
                        )
                        if turn_command in [COMMANDS['AVOID_LEFT'], COMMANDS['AVOID_RIGHT']]:
                            robot_status = f"AVOIDING while searching - {avoidance_duration} frames left"
                        elif turn_command == COMMANDS['FORWARD']:
                            robot_status = "Moving forward to find line"
                        else:
                            robot_status = "Searching for line"
                    else:
                        should_avoid = OBJECT_DETECTION_ENABLED and object_detected
                        if should_avoid:
                            turn_command = get_turn_command_with_avoidance(
                                0.0, avoid_objects=should_avoid,
                                line_detected_now=False, line_offset_now=0.0,
                                detected_objects_list=detected_objects if 'detected_objects' in locals() else []
                            )
                            robot_status = "🔍 Line lost but still avoiding object"
                        else:
                            turn_command = COMMANDS['STOP']
                            robot_status = "Line lost - stopped"
                        if search_counter > 60:
                            search_counter = 0
                            last_known_good_offset = 0.0
                            pid.reset()
                            offset_history.clear()
                            steering_history.clear()
                            logger.info("Search pattern reset")
            
            # Send command to ESP32 with avoidance logging
            if turn_command in [COMMANDS['AVOID_LEFT'], COMMANDS['AVOID_RIGHT']]:
                logger.info(f"SENDING AVOIDANCE COMMAND: {turn_command}")
            
            if esp_connection:
                success = esp_connection.send_command(turn_command)
                if not success and frame_count % 30 == 0:  # Log occasionally
                    logger.warning("ESP32 communication failed")
            
            # Draw enhanced debug information
            draw_debug_info(display_frame, detection_data)
            
            # Update output frame for web interface
            with frame_lock:
                output_frame = display_frame.copy()
            
            # Calculate FPS
            processing_time = time.time() - start_time
            if processing_time > 0:
                fps_history.append(1.0 / processing_time)
                current_fps = sum(fps_history) / len(fps_history)
            
            # Log status periodically
            if frame_count % 120 == 0:  # Every 4 seconds to reduce noise
                kp, ki, kd = pid.get_params()
                smart_status = ""
                if smart_avoidance.current_strategy != 'none':
                    smart_status = f" | Strategy: {smart_avoidance.current_strategy}"
                
                esp32_status = "Connected" if esp_connected else "Disconnected"
                logger.info(f"Status: {robot_status} | FPS: {current_fps:.1f} | "
                           f"PID: {kp:.3f}/{ki:.3f}/{kd:.3f} | ESP32: {esp32_status}"
                           f" | Objects: {len(detected_objects) if 'detected_objects' in locals() else 0}{smart_status}")
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        logger.info(" Stopping robot (Ctrl+C pressed)")
        robot_status = "Shutting down"
    except Exception as e:
        logger.error(f" Unexpected error: {e}", exc_info=True)
        robot_status = f"Error: {str(e)}"
    finally:
        # Clean up
        robot_status = "Cleaning up"
        
        if esp_connection:
            logger.info("Stopping robot and disconnecting ESP32")
            esp_connection.send_command(COMMANDS['STOP'])
            time.sleep(0.2)  # Give time for stop command
            esp_connection.close()
        
        if 'cap' in locals() and cap.isOpened():
            cap.release()
            logger.info("Camera released")
        
        logger.info(" Robot stopped cleanly")

if __name__ == "__main__":
    main()