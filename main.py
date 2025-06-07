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

# YOLO11n for object detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO11n available for object detection")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available. Install with: pip install ultralytics")
    print("   Falling back to basic computer vision detection")

# -----------------------------------------------------------------------------
# --- CONFIGURATION FOR BLACK LINE FOLLOWING ---
# -----------------------------------------------------------------------------
# ESP32 Configuration - UPDATE THIS TO MATCH YOUR ESP32's IP
ESP32_IP = '192.168.2.21'  # Change this to your ESP32's IP address
ESP32_PORT = 1234
CAMERA_WIDTH, CAMERA_HEIGHT = 320, 240
CAMERA_FPS = 5

# Image processing parameters
BLACK_THRESHOLD = 60  # Higher values detect darker lines
BLUR_SIZE = 5
MIN_CONTOUR_AREA = 100  # Minimum area to be considered a line

# Multi-zone detection parameters
ZONE_BOTTOM_HEIGHT = 0.25   # Bottom 25% for primary line following
ZONE_MIDDLE_HEIGHT = 0.25   # Middle 25% for corner prediction
ZONE_TOP_HEIGHT = 0.30      # Top 30% for object detection

# Corner detection parameters
CORNER_DETECTION_ENABLED = True
CORNER_CONFIDENCE_BOOST = 1.2
CORNER_CIRCULARITY_THRESHOLD = 0.4  # Lower values indicate corners
CORNER_PREDICTION_THRESHOLD = 0.3   # Confidence needed for corner warning

# Object detection parameters
OBJECT_DETECTION_ENABLED = True  # Enable obstacle detection and avoidance
USE_YOLO = True  # Use YOLO11n for accurate object detection (recommended)

# YOLO Configuration
YOLO_MODEL_SIZE = "yolo11n.pt"  # Nano model for speed (yolo11s.pt, yolo11m.pt for more accuracy)
YOLO_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for object detection
YOLO_CLASSES_TO_AVOID = [0, 39, 41, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]  # person, bottle, cup, bowl, etc.

# Legacy CV-based detection parameters (fallback if YOLO not available)
OBJECT_SIZE_THRESHOLD = 800    # Minimum contour area 
OBJECT_WIDTH_THRESHOLD = 0.20  # Object width ratio to trigger avoidance
OBJECT_HEIGHT_THRESHOLD = 0.15  # Object height ratio to trigger avoidance
OBJECT_AVOIDANCE_DISTANCE = 10  # How many frames to remember object position (longer avoidance)
OBJECT_MIN_ASPECT_RATIO = 0.3  # Shape filtering
OBJECT_MAX_ASPECT_RATIO = 3.0  # Shape filtering
OBJECT_LINE_BLOCKING_THRESHOLD = 0.5  # Distance from line to trigger avoidance (more lenient for YOLO)

# Simple PID controller values
KP = 0.6  # Proportional gain
KI = 0.02  # Integral gain 
KD = 0.1   # Derivative gain
MAX_INTEGRAL = 5.0  # Prevent integral windup

# Commands for ESP32
COMMANDS = {'FORWARD': 'FORWARD', 'LEFT': 'LEFT', 'RIGHT': 'RIGHT', 'STOP': 'STOP', 
           'AVOID_LEFT': 'AVOID_LEFT', 'AVOID_RIGHT': 'AVOID_RIGHT'}
SPEED = 'SLOW'  # Default speed

# Steering parameters
STEERING_DEADZONE = 0.1  # Ignore small errors
MAX_STEERING = 1.0  # Maximum steering value

# Object avoidance state
object_detected = False
object_position = 0.0  # -1.0 (left) to 1.0 (right)
object_avoidance_counter = 0
avoidance_side = None  # 'left' or 'right'
corner_warning = False
corner_prediction_frames = 0
object_detection_frames = 0  # Counter for consecutive object detections
OBJECT_DETECTION_PERSISTENCE = 2  # Frames needed for object confirmation (faster response for YOLO)

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
                    
                    objects.append({
                        'position': relative_pos,
                        'width_ratio': width_ratio,
                        'height_ratio': height_ratio,
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'bbox': (int(x - w/2), int(y - h/2), int(w), int(h)),
                        'distance_from_line': distance_from_line if line_position is not None else abs(relative_pos)
                    })
                    
                    print(f"🔍 YOLO detected: {class_name} (conf: {confidence:.2f}, pos: {relative_pos:.2f})")
        
        return objects
        
    except Exception as e:
        print(f"❌ YOLO detection error: {e}")
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
            
        objects.append({
            'position': relative_pos,
            'width_ratio': width_ratio,
            'height_ratio': height_ratio,
            'aspect_ratio': aspect_ratio,
            'area': area,
            'contour': contour,
            'bbox': (x, y, w, h),
            'distance_from_line': distance_from_line if line_position is not None else abs(relative_pos)
        })
    
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
    
    # Update object detection with temporal filtering
    global object_detection_frames
    
    if detected_objects:
        # Find the most significant object (largest and most centered)
        # For YOLO objects, use width_ratio * height_ratio as area equivalent
        def object_significance(obj):
            if 'area' in obj:
                # Traditional CV detection
                return obj['area'] * (1 - abs(obj['position']))
            else:
                # YOLO detection - use bounding box area equivalent
                area_equivalent = obj['width_ratio'] * obj['height_ratio'] * 10000  # Scale for comparison
                return area_equivalent * (1 - abs(obj['position']))
        
        main_object = max(detected_objects, key=object_significance)
        object_detection_frames += 1
        
        # Only confirm object detection after multiple consecutive frames
        if object_detection_frames >= OBJECT_DETECTION_PERSISTENCE:
            if not object_detected:  # First time confirming this object
                robot_stats['objects_detected'] += 1
                if 'class_name' in main_object:
                    # YOLO object
                    logger.info(f"🚨 YOLO Object confirmed! {main_object['class_name']} (conf: {main_object['confidence']:.2f}, pos: {main_object['position']:.2f})")
                else:
                    # Traditional CV object
                    logger.info(f"🚨 Object confirmed! Position: {main_object['position']:.2f}, Area: {main_object['area']}")
            object_detected = True
            object_position = main_object['position']
        else:
            # Still building up confidence, don't trigger avoidance yet
            object_detected = False
            object_position = 0.0
            logger.debug(f"🔍 Object candidate detected ({object_detection_frames}/{OBJECT_DETECTION_PERSISTENCE})")
    else:
        object_detection_frames = max(0, object_detection_frames - 1)
        if object_detection_frames <= 0:
            object_detected = False
            object_position = 0.0
    
    # Determine primary line position
    line_x = None
    confidence = 0.0
    primary_roi = None
    
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
def get_turn_command_with_avoidance(steering, avoid_objects=False, avoidance_direction=None):
    """Convert steering value to turn command with object avoidance"""
    global object_avoidance_counter, avoidance_side
    
    # Object avoidance takes priority
    if avoid_objects and object_detected:
        object_avoidance_counter = OBJECT_AVOIDANCE_DISTANCE
        
        # Determine avoidance direction based on object position
        # Be very aggressive about avoiding detected objects
        if object_position < -0.05:  # Object slightly left of center
            avoidance_side = 'right'
            robot_stats['avoidance_maneuvers'] += 1
            logger.info(f"🚨 AVOIDING RIGHT - Object at position {object_position:.2f}")
            return COMMANDS['AVOID_RIGHT']
        elif object_position > 0.05:  # Object slightly right of center
            avoidance_side = 'left'
            robot_stats['avoidance_maneuvers'] += 1
            logger.info(f"🚨 AVOIDING LEFT - Object at position {object_position:.2f}")
            return COMMANDS['AVOID_LEFT']
        else:  # Object directly in center
            # For center objects, default to right avoidance
            avoidance_side = 'right'
            robot_stats['avoidance_maneuvers'] += 1
            logger.info(f"🚨 AVOIDING RIGHT (CENTER) - Object at position {object_position:.2f}")
            return COMMANDS['AVOID_RIGHT']
    
    # Continue avoidance maneuver for a few frames
    if object_avoidance_counter > 0:
        object_avoidance_counter -= 1
        logger.debug(f"🔄 Continuing avoidance {avoidance_side} (frames left: {object_avoidance_counter})")
        if avoidance_side == 'left':
            return COMMANDS['AVOID_LEFT']
        elif avoidance_side == 'right':
            return COMMANDS['AVOID_RIGHT']
    
    # Normal steering behavior
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
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y_adjusted), (x + w, y_adjusted + h), (0, 0, 255), 2)
        
        # Draw object label (YOLO class name if available, otherwise generic)
        if 'class_name' in obj:
            label = f"{obj['class_name']} ({obj['confidence']:.2f})"
        else:
            label = "OBJECT"
        
        cv2.putText(frame, label, (x, y_adjusted - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw warning indicator
        cv2.putText(frame, "🚨 OBSTACLE DETECTED", (width//2 - 100, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw object position indicator
        obj_center_x = int(width/2 + obj['position'] * width/2)
        cv2.circle(frame, (obj_center_x, 60), 8, (0, 0, 255), -1)
    
    # Draw corner warning
    if corner_warning:
        cv2.putText(frame, "🔄 CORNER AHEAD", (width//2 - 80, 80),
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
    if object_avoidance_counter > 0:
        cv2.putText(frame, f"AVOIDING {avoidance_side.upper()}", (width//2 - 80, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
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
        'corner_count': robot_stats.get('corner_count', 0),
        'objects_detected': robot_stats.get('objects_detected', 0),
        'avoidance_maneuvers': robot_stats.get('avoidance_maneuvers', 0),
        'corner_warning': corner_warning,
        'object_detected': object_detected
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
    global line_detected, current_fps, confidence, esp_connection, robot_stats, yolo_model
    
    logger.info("🚀 Starting Simple Line Follower Robot")
    robot_status = "Starting camera"
    robot_stats['start_time'] = time.time()
    
    # Initialize YOLO model if available and enabled
    if USE_YOLO and YOLO_AVAILABLE and OBJECT_DETECTION_ENABLED:
        try:
            logger.info(f"🤖 Loading YOLO11n model: {YOLO_MODEL_SIZE}")
            robot_status = "Loading YOLO model"
            yolo_model = YOLO(YOLO_MODEL_SIZE)
            logger.info("✅ YOLO11n model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            logger.info("📉 Falling back to traditional computer vision detection")
            yolo_model = None
    else:
        if not OBJECT_DETECTION_ENABLED:
            logger.info("🚫 Object detection disabled")
        elif not USE_YOLO:
            logger.info("📉 Using traditional computer vision detection")
        elif not YOLO_AVAILABLE:
            logger.info("⚠️ YOLO not available, using traditional detection")
    
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
            
            # Process image with multi-zone detection
            detection_data = process_image_multi_zone(frame)
            line_x = detection_data['line_x']
            confidence = detection_data['confidence']
            detected_objects = detection_data['detected_objects']
            
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
                
                # Check for corner (enhanced with prediction)
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
                elif corner_warning:
                    robot_status = f"🔮 Corner predicted ahead"
                else:
                    if corner_detected_count > 0:
                        corner_detected_count -= 1
                    else:
                        corner_detected_count = 0
                        
                    # Enhanced status with object detection
                    if object_detected:
                        robot_status = f"🚨 AVOIDING OBSTACLE - Object detected!"
                    elif detected_objects:
                        robot_status = f"⚠️ Object candidate detected (building confidence)"
                    else:
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
                
                # Convert steering to command with object avoidance
                should_avoid = OBJECT_DETECTION_ENABLED and object_detected
                if should_avoid:
                    logger.debug(f"🔍 Object avoidance active! Object detected: {object_detected}, Position: {object_position:.2f}")
                
                turn_command = get_turn_command_with_avoidance(avg_steering, 
                                                             avoid_objects=should_avoid,
                                                             avoidance_direction=avoidance_side)
                
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
            
            # Log status periodically with enhanced info
            if frame_count % 60 == 0:  # Every 2 seconds at 30fps
                logger.info(f"📊 Status: {robot_status} | FPS: {current_fps:.1f} | "
                           f"Command: {turn_command} | ESP32: {'✅' if esp_connected else '❌'} | "
                           f"Objects: {len(detected_objects) if 'detected_objects' in locals() else 0} | "
                           f"Corner Warning: {'⚠️' if corner_warning else '✅'}")
            
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