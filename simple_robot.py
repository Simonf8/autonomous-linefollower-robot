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

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("YOLO11n available for object detection")
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available. Install with: pip install ultralytics")
    print("Falling back to basic computer vision detection")

# Try to import text-to-speech (Piper TTS)
try:
    import subprocess
    import os
    import queue
    # Check if Piper TTS is available
    if os.path.exists('./piper/piper') and os.path.exists('./voices/en_US-lessac-medium.onnx'):
        TTS_AVAILABLE = True
        TTS_ENGINE = 'piper'
        print("Piper TTS available - High quality neural speech")
    else:
        # Fallback to pyttsx3 if available
        import pyttsx3
        TTS_AVAILABLE = True
        TTS_ENGINE = 'pyttsx3'
        print("pyttsx3 TTS available")
except ImportError:
    TTS_AVAILABLE = False
    TTS_ENGINE = None
    print("Text-to-speech not available. Install piper or pyttsx3")

ESP32_IP = '192.168.2.21'  
ESP32_PORT = 1234
CAMERA_WIDTH, CAMERA_HEIGHT = 730, 420
CAMERA_FPS = 5

BLACK_THRESHOLD = 90  # Higher values detect darker lines
BLUR_SIZE = 5
MIN_CONTOUR_AREA = 40  # Minimum area to be considered a line

# Multi-zone detection parameters
ZONE_BOTTOM_HEIGHT = 0.25   # Bottom 25% for primary line following
ZONE_MIDDLE_HEIGHT = 0.20   # Middle 20% for corner prediction
ZONE_TOP_HEIGHT = 0.45      # Top 45% for early object detection

# Corner detection parameters
CORNER_DETECTION_ENABLED = True
CORNER_CONFIDENCE_BOOST = 1.2
CORNER_CIRCULARITY_THRESHOLD = 0.4  # Lower values indicate corners
CORNER_PREDICTION_THRESHOLD = 0.3   # Confidence needed for corner warning

# Object detection parameters
OBJECT_DETECTION_ENABLED = True  # Enable obstacle detection and avoidance
USE_YOLO = True  # Use YOLO11n for accurate object detection (recommended)
USE_SMART_AVOIDANCE = True  # Use smart avoidance with live mapping and learning

# YOLO Configuration
YOLO_MODEL_SIZE = "yolo11n.pt"  # Nano model for speed
YOLO_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for object detection
YOLO_CLASSES_TO_AVOID = [0, 39, 41, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]  # person, bottle, cup, bowl, etc.

# Legacy CV-based detection parameters (fallback if YOLO not available)
OBJECT_SIZE_THRESHOLD = 800    # Minimum contour area 
OBJECT_WIDTH_THRESHOLD = 0.20  # Object width ratio to trigger avoidance
OBJECT_HEIGHT_THRESHOLD = 0.15  # Object height ratio to trigger avoidance
OBJECT_AVOIDANCE_DISTANCE = 35  # How many frames to remember object position
OBJECT_MIN_ASPECT_RATIO = 0.3  # Shape filtering
OBJECT_MAX_ASPECT_RATIO = 3.0  # Shape filtering
OBJECT_LINE_BLOCKING_THRESHOLD = 0.7  # Distance from line to trigger avoidance

# Adaptive PID controller values with auto-tuning
KP = 0.35  # Proportional gain
KI = 0.005 # Integral gain
KD = 0.25  # Derivative gain
MAX_INTEGRAL = 2.0  # Integral windup limit

# Auto-tuning parameters
PID_LEARNING_RATE = 0.0005  # Learning rate for PID adaptation
PID_ADAPTATION_WINDOW = 100  # Window for PID adaptation
PERFORMANCE_THRESHOLD = 0.12  # Performance threshold

# Commands for ESP32 - SMOOTH CURVE AVOIDANCE
COMMANDS = {
    'FORWARD': 'FORWARD', 
    'LEFT': 'LEFT', 
    'RIGHT': 'RIGHT', 
    'STOP': 'STOP',
    'CURVE_LEFT': 'LEFT',      # Smooth curve left
    'CURVE_RIGHT': 'RIGHT',    # Smooth curve right
    'GENTLE_LEFT': 'LEFT',     # Gentle turn left
    'GENTLE_RIGHT': 'RIGHT'    # Gentle turn right
}
SPEED = 'SLOW'  # Default speed
ROBOT_SPEED_M_S = 0.1  # Approximate robot speed in m/s

# Enhanced obstacle mapping parameters
OBSTACLE_MAP_SIZE = 50  # Remember last 50 obstacles
OBSTACLE_MEMORY_TIMEOUT = 30.0  # Forget obstacles after 30 seconds
MIN_OBSTACLE_WIDTH = 0.1  # Minimum width ratio to be significant
MIN_OBSTACLE_HEIGHT = 0.08  # Minimum height ratio to be significant

# C-SHAPED TURN AVOIDANCE PARAMETERS - Like a C-shaped road curve
CURVE_AVOIDANCE_ENABLED = True
CURVE_DETECTION_DISTANCE = 2.0    # Start curve when obstacle is 2m away
CURVE_RADIUS_MULTIPLIER = 2.5     # How wide the C-curve should be
CURVE_SMOOTHNESS_FACTOR = 0.80    # Smoothness of the curve (0.0-1.0)
CURVE_FORWARD_BIAS = 0.85         # INCREASED: Much more forward motion during curve
CURVE_RETURN_SENSITIVITY = 0.25   # How quickly to return to line
C_CURVE_SHARPNESS = 1.2           # How sharp the C-curve should be (reduced for more forward motion)
C_CURVE_DEPTH = 1.8               # How deep into the C-curve to go
C_CURVE_FORWARD_EXTENSION = 1.4   # How much extra forward motion after clearing obstacle

# Dynamic path calculation
SAFETY_MARGIN = 1.5           # Multiply obstacle size by this for safety
PATH_CALCULATION_ENABLED = True

# Steering parameters
STEERING_DEADZONE = 0.12  # Ignore small errors
MAX_STEERING = 0.12       # Maximum steering value

# Learning state
obstacle_memory = {}  # Learning-based obstacle memory
performance_history = deque(maxlen=PID_ADAPTATION_WINDOW)  # PID performance tracking
learned_maneuvers = {}  # Successful avoidance patterns

# Enhanced smooth curve avoidance state
object_detected = False
object_position = 0.0
current_obstacle = None       # Current obstacle being avoided
obstacle_map = {}            # Persistent obstacle memory
avoidance_phase = 'none'     # 'none', 'curve_out', 'curve_around', 'curve_back'
avoidance_side = 'none'      # 'left', 'right'
avoidance_progress = 0.0     # Progress through the curve (0.0 to 1.0)
curve_center_offset = 0.0    # How far we've curved from the line
planned_path = None          # Calculated path around obstacle
corner_warning = False
corner_prediction_frames = 0
object_detection_frames = 0

# Distance estimation parameters
CAMERA_FOCAL_LENGTH = 200.0  # Estimated focal length in pixels
CAMERA_HEIGHT = 0.12         # Camera height from ground in meters
CAMERA_ANGLE = 5             # Camera tilt angle in degrees

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
SAFE_DISTANCE = 2.0       # Minimum safe distance in meters
WARNING_DISTANCE = 3.0    # Warning distance in meters
EMERGENCY_DISTANCE = 1.2  # Emergency stop distance

# Speech Settings for Piper TTS
SPEECH_ENABLED = True          # Enable/disable speech announcements
SPEECH_RATE = 150             # Speech rate (words per minute)
SPEECH_VOLUME = 0.8           # Speech volume (0.0 to 1.0)
ANNOUNCE_INTERVAL = 3.0       # Minimum seconds between similar announcements

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

# Speech manager instance
speech_manager = None

# Curve avoidance state
last_line_position = 0.0
curve_start_position = 0.0
target_curve_offset = 0.0

# -----------------------------------------------------------------------------
# --- Logging Setup ---
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, 
                   format='[%(asctime)s] %(levelname)s: %(message)s', 
                   datefmt='%H:%M:%S')
logger = logging.getLogger("SmoothLineFollower")

# -----------------------------------------------------------------------------
# --- Speech Manager (Piper TTS) ---
# -----------------------------------------------------------------------------
class SpeechManager:
    def __init__(self):
        self.engine = None
        self.speech_queue = queue.Queue()
        self.running = False
        self.last_announcements = {}
        self.tts_engine_type = TTS_ENGINE
        
        if TTS_AVAILABLE and SPEECH_ENABLED:
            try:
                if TTS_ENGINE == 'piper':
                    # Piper TTS setup
                    self.piper_path = './piper/piper'
                    self.voice_model = './voices/en_US-lessac-medium.onnx'
                    self.engine = True  # Flag to indicate piper is ready
                    logger.info("Piper TTS system initialized successfully")
                elif TTS_ENGINE == 'pyttsx3':
                    # pyttsx3 setup (fallback)
                    import pyttsx3
                    self.engine = pyttsx3.init()
                    self.engine.setProperty('rate', SPEECH_RATE)
                    self.engine.setProperty('volume', SPEECH_VOLUME)
                    
                    # Try to set voice (prefer female voice if available)
                    voices = self.engine.getProperty('voices')
                    if voices:
                        for voice in voices:
                            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                                self.engine.setProperty('voice', voice.id)
                                break
                    
                    logger.info("pyttsx3 speech system initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize speech system: {e}")
                self.engine = None
    
    def start(self):
        if self.engine:
            self.running = True
            self.thread = threading.Thread(target=self._speech_worker, daemon=True)
            self.thread.start()
            logger.info(f"{self.tts_engine_type} speech system started")
    
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
    
    def announce(self, message, category="general", force=False):
        """Add announcement to queue with category-based throttling"""
        if not self.engine or not SPEECH_ENABLED:
            return
        
        current_time = time.time()
        
        # Check if we should throttle this announcement
        if not force and category in self.last_announcements:
            if current_time - self.last_announcements[category] < ANNOUNCE_INTERVAL:
                return
        
        # Add to queue and update last announcement time
        self.speech_queue.put(message)
        self.last_announcements[category] = current_time
        logger.info(f"Queued announcement: {message}")
    
    def _speak_with_piper(self, message):
        """Use Piper TTS to speak message"""
        try:
            # Create temporary audio file
            temp_audio = "/tmp/robot_speech.wav"
            
            # Generate speech with Piper
            process = subprocess.run([
                self.piper_path,
                '--model', self.voice_model,
                '--output_file', temp_audio
            ], input=message, text=True, capture_output=True, timeout=10)
            
            if process.returncode == 0:
                # Play the generated audio
                subprocess.run(['aplay', temp_audio], check=True, capture_output=True)
                # Clean up temp file
                os.remove(temp_audio)
                return True
            else:
                logger.error(f"Piper TTS failed: {process.stderr}")
                return False
        except Exception as e:
            logger.error(f"Piper TTS error: {e}")
            return False
    
    def _speak_with_pyttsx3(self, message):
        """Use pyttsx3 to speak message"""
        try:
            self.engine.say(message)
            self.engine.runAndWait()
            return True
        except Exception as e:
            logger.error(f"pyttsx3 error: {e}")
            return False
    
    def _speech_worker(self):
        """Background thread to handle speech synthesis"""
        while self.running:
            try:
                # Get message from queue with timeout
                message = self.speech_queue.get(timeout=1.0)
                if message and self.engine:
                    if self.tts_engine_type == 'piper':
                        self._speak_with_piper(message)
                    elif self.tts_engine_type == 'pyttsx3':
                        self._speak_with_pyttsx3(message)
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Speech worker error: {e}")

# -----------------------------------------------------------------------------
# --- Smooth Curve Mathematics ---
# -----------------------------------------------------------------------------
class CShapedCurveCalculator:
    """Calculate C-shaped curves for obstacle avoidance"""
    
    @staticmethod
    def calculate_curve_parameters(obstacle_position, obstacle_width, line_position=0.0):
        """
        Calculate parameters for a C-shaped curve around an obstacle
        Returns: (curve_radius, curve_direction, target_offset)
        """
        # Determine which side to make the C-curve
        if obstacle_position > 0:  # Obstacle on right, C-curve left
            curve_direction = 'left'
            target_offset = -(abs(obstacle_position) + obstacle_width * SAFETY_MARGIN + 0.4) * C_CURVE_DEPTH
        else:  # Obstacle on left, C-curve right
            curve_direction = 'right'
            target_offset = (abs(obstacle_position) + obstacle_width * SAFETY_MARGIN + 0.4) * C_CURVE_DEPTH
        
        # Calculate curve radius for C-shape (wider than normal curves)
        base_radius = max(obstacle_width * CURVE_RADIUS_MULTIPLIER, 1.0)
        curve_radius = base_radius * (1.2 + abs(obstacle_position)) * C_CURVE_SHARPNESS
        
        return curve_radius, curve_direction, target_offset
    
    @staticmethod
    def calculate_curve_steering(progress, curve_radius, target_offset, current_offset):
        """
        Calculate steering for C-shaped curve progression with STRONG forward bias
        progress: 0.0 to 1.0 (0 = start curve, 1 = end curve)
        """
        # C-shaped curve with four phases for better forward progress
        # Phase 1 (0.0-0.2): Quick curve out from line 
        # Phase 2 (0.2-0.6): Forward movement around obstacle with slight curve
        # Phase 3 (0.6-0.8): Continue forward while maintaining clearance  
        # Phase 4 (0.8-1.0): Quick curve back towards line search area
        
        if progress <= 0.2:
            # Quick curve out - get clear of obstacle fast
            phase_progress = progress / 0.2
            smooth_progress = CShapedCurveCalculator._c_curve_function(phase_progress)
            desired_offset = target_offset * smooth_progress * 0.8
            forward_emphasis = 0.9  # Strong forward motion even while curving out
            
        elif progress <= 0.6:
            # Forward movement around obstacle - prioritize moving FORWARD
            phase_progress = (progress - 0.2) / 0.4
            # Maintain most of the offset but focus on forward progress
            desired_offset = target_offset * 0.9  # Stay clear of obstacle
            forward_emphasis = 0.95  # VERY strong forward motion
            
        elif progress <= 0.8:
            # Continue forward past obstacle - ensure we're well clear
            phase_progress = (progress - 0.6) / 0.2
            # Gradually reduce offset as we move past obstacle
            desired_offset = target_offset * (0.9 - 0.3 * phase_progress)
            forward_emphasis = 0.92  # Still very forward-focused
            
        else:
            # Quick curve back to search area ahead
            phase_progress = (progress - 0.8) / 0.2
            smooth_progress = CShapedCurveCalculator._c_curve_function(phase_progress)
            # Return to center-ish area to search for line ahead
            desired_offset = target_offset * 0.6 * (1.0 - smooth_progress)
            forward_emphasis = 0.88  # Maintain forward motion while returning
        
        # Calculate steering to achieve desired offset
        offset_error = desired_offset - current_offset
        
        # Apply C-curve steering with reduced sharpness for more forward motion
        steering = offset_error * CURVE_SMOOTHNESS_FACTOR * (C_CURVE_SHARPNESS * 0.7)
        
        # Strong forward component - this is key for moving past the obstacle
        forward_component = CURVE_FORWARD_BIAS * forward_emphasis
        
        return np.clip(steering, -0.8, 0.8), forward_component
    
    @staticmethod
    def _smooth_step(x):
        """Smooth step function for natural acceleration/deceleration"""
        # Smoothstep function: 3x² - 2x³
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)
    
    @staticmethod
    def _c_curve_function(x):
        """C-curve specific function for sharper transitions"""
        # Sigmoid-like function for sharper C-curve transitions
        x = np.clip(x, 0.0, 1.0)
        # Modified sigmoid: steeper at start and end, smoother in middle
        return 1.0 / (1.0 + np.exp(-8.0 * (x - 0.5)))
    
    @staticmethod
    def calculate_return_to_line_steering(current_offset, line_offset):
        """Calculate steering to smoothly return to the line"""
        # Combine curve completion with line following
        total_offset = current_offset + line_offset
        
        # Use gentler return steering
        return_steering = -total_offset * CURVE_RETURN_SENSITIVITY
        
        return np.clip(return_steering, -0.8, 0.8)

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
                    
                    # Create object data
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
                    
                    print(f"🔍 YOLO detected: {class_name} (conf: {confidence:.2f}, pos: {relative_pos:.2f})")
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
        
        # Check minimum size requirements
        if width_ratio < OBJECT_WIDTH_THRESHOLD or height_ratio < OBJECT_HEIGHT_THRESHOLD:
            continue
            
        # Check aspect ratio
        if aspect_ratio < OBJECT_MIN_ASPECT_RATIO or aspect_ratio > OBJECT_MAX_ASPECT_RATIO:
            continue
        
        # Calculate object center position
        center_x = x + w/2
        relative_pos = (center_x - width/2) / (width/2)
        
        # Check if object is in the robot's path
        if line_position is not None:
            line_relative_pos = (line_position - width/2) / (width/2)
            distance_from_line = abs(relative_pos - line_relative_pos)
            
            if distance_from_line > OBJECT_LINE_BLOCKING_THRESHOLD:
                continue
        else:
            if abs(relative_pos) > 0.6:
                continue
        
        # Create object data
        object_data = {
            'position': relative_pos,
            'width_ratio': width_ratio,
            'height_ratio': height_ratio,
            'aspect_ratio': aspect_ratio,
            'area': area,
            'contour': contour,
            'bbox': (x, y, w, h),
            'distance_from_line': distance_from_line if line_position is not None else abs(relative_pos),
            'class_name': 'default'
        }
        
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
    
    # Create inverted binary for object detection
    _, binary_inverted = cv2.threshold(blurred, BLACK_THRESHOLD + 40, 255, cv2.THRESH_BINARY)
    
    height, width = binary.shape
    
    # Define zones
    bottom_start = int(height * (1 - ZONE_BOTTOM_HEIGHT))
    middle_start = int(height * (1 - ZONE_BOTTOM_HEIGHT - ZONE_MIDDLE_HEIGHT))
    top_start = int(height * (1 - ZONE_BOTTOM_HEIGHT - ZONE_MIDDLE_HEIGHT - ZONE_TOP_HEIGHT))
    
    # Extract ROIs for each zone
    bottom_roi = binary[bottom_start:height, :]
    middle_roi = binary[middle_start:bottom_start, :]
    top_roi = binary_inverted[top_start:middle_start, :]
    
    # Process each zone
    # 1. Bottom zone - Primary line following
    bottom_x, bottom_confidence, bottom_contour = detect_line_in_roi(bottom_roi, "bottom")
    
    # 2. Middle zone - Corner prediction
    middle_x, middle_confidence, middle_contour = detect_line_in_roi(middle_roi, "middle")
    
    # 3. Top zone - Object detection
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
    
    # Update object detection
    if detected_objects and avoidance_phase == 'none':
        # Find the most significant object
        def object_significance(obj):
            if 'area' in obj:
                return obj['area'] * (1 - abs(obj['position']))
            else:
                area_equivalent = obj['width_ratio'] * obj['height_ratio'] * 10000
                return area_equivalent * (1 - abs(obj['position']))
        
        main_object = max(detected_objects, key=object_significance)
        
        if not object_detected:
            robot_stats['objects_detected'] += 1
            
            if 'class_name' in main_object:
                logger.info(f"🗺️ OBSTACLE DETECTED: {main_object['class_name']} "
                           f"(conf: {main_object['confidence']:.2f}, "
                           f"pos: {main_object['position']:.2f}) - Starting smooth curve!")
            else:
                logger.info(f"🗺️ OBSTACLE DETECTED: Object "
                           f"(pos: {main_object['position']:.2f}) - Starting smooth curve!")
        
        object_detected = True
        object_position = main_object['position']
        
    elif not detected_objects and avoidance_phase == 'none':
        if object_detected:
            logger.info("✅ No more objects detected")
        object_detected = False
        object_position = 0.0
    
    # Determine primary line position
    line_x = None
    confidence = 0.0
    primary_roi = None
    
    # During avoidance, check all zones for line
    if avoidance_phase != 'none':
        if bottom_confidence > 0.1:
            line_x = bottom_x
            confidence = bottom_confidence
            primary_roi = bottom_roi
        elif middle_confidence > 0.1:
            line_x = middle_x
            confidence = middle_confidence * 0.9
            primary_roi = middle_roi
    else:
        # Normal mode - use zone priorities
        if bottom_confidence > 0.3:
            line_x = bottom_x
            confidence = bottom_confidence
            primary_roi = bottom_roi
        elif middle_confidence > 0.4 and corner_warning:
            line_x = middle_x
            confidence = middle_confidence * 0.8
            primary_roi = middle_roi
        elif bottom_confidence > 0.1:
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
# --- Smooth Curve Obstacle Avoidance ---
# -----------------------------------------------------------------------------
def get_c_shaped_curve_command(steering, detected_objects=None, line_detected_now=False, line_offset_now=0.0):
    """C-SHAPED CURVE AVOIDANCE: Like driving through a C-shaped road curve"""
    global avoidance_phase, avoidance_side, avoidance_progress, curve_center_offset
    global target_curve_offset, last_line_position, curve_start_position
    
    # Initialize C-curve calculator
    curve_calc = CShapedCurveCalculator()
    
    if CURVE_AVOIDANCE_ENABLED and detected_objects and avoidance_phase == 'none':
        # START SMOOTH CURVE AVOIDANCE
        main_object = detected_objects[0]  # Most significant object
        
        # Calculate curve parameters
        curve_radius, curve_direction, target_offset = curve_calc.calculate_curve_parameters(
            main_object['position'], 
            main_object.get('width_ratio', 0.2),
            line_offset_now
        )
        
        # Initialize curve state
        avoidance_phase = 'curve_out'
        avoidance_side = curve_direction
        avoidance_progress = 0.0
        curve_center_offset = 0.0
        target_curve_offset = target_offset
        last_line_position = line_offset_now
        curve_start_position = line_offset_now
        
        logger.info(f"🌊 STARTING C-SHAPED CURVE: {curve_direction} side, "
                   f"target offset: {target_offset:.2f}, radius: {curve_radius:.2f}")
        
        robot_stats['avoidance_maneuvers'] += 1
        return COMMANDS[f'CURVE_{curve_direction.upper()}']
    
    elif avoidance_phase != 'none':
        # CONTINUE SMOOTH CURVE AVOIDANCE
        
        # Update progress (increment based on processing speed)
        # Slower progression for more forward movement
        progress_increment = 0.012  # Slower progression for more time to move forward
        avoidance_progress = min(avoidance_progress + progress_increment, 1.0)
        
        # Calculate smooth curve steering
        curve_steering, forward_component = curve_calc.calculate_curve_steering(
            avoidance_progress, 
            1.0,  # normalized radius
            target_curve_offset, 
            curve_center_offset
        )
        
        # Update our position in the curve
        curve_center_offset += curve_steering * 0.1  # Simulate movement
        
        # Determine C-curve phase (updated for new 4-phase system)
        if avoidance_progress <= 0.2:
            avoidance_phase = 'c_curve_out'
            phase_name = "C-CURVE OUT"
        elif avoidance_progress <= 0.6:
            avoidance_phase = 'c_curve_forward'
            phase_name = "C-CURVE FORWARD"
        elif avoidance_progress <= 0.8:
            avoidance_phase = 'c_curve_clear'
            phase_name = "C-CURVE CLEARING"
        else:
            avoidance_phase = 'c_curve_search'
            phase_name = "C-CURVE SEARCH"
        
        # Check if we should complete the curve
        if avoidance_progress >= 0.9:  # Start looking for completion earlier
            if line_detected_now and abs(line_offset_now) < 0.4:
                # Successfully found line ahead after passing obstacle
                logger.info("✅ C-SHAPED CURVE COMPLETED - Line found ahead!")
                avoidance_phase = 'none'
                avoidance_progress = 0.0
                curve_center_offset = 0.0
                
                # Smooth transition back to line following
                transition_steering = curve_calc.calculate_return_to_line_steering(
                    curve_center_offset, line_offset_now
                )
                
                if abs(transition_steering) < STEERING_DEADZONE:
                    return COMMANDS['FORWARD']
                else:
                    return COMMANDS['LEFT'] if transition_steering > 0 else COMMANDS['RIGHT']
        
        # Extended curve with forward search if no line found        
        if avoidance_progress >= 1.0:
            if avoidance_progress >= 1.4:  # Much longer extension for forward search
                logger.info("⚠️ C-SHAPED CURVE COMPLETED - Switching to forward search")
                avoidance_phase = 'none'
                avoidance_progress = 0.0
                curve_center_offset = 0.0
                # Continue forward to search for line ahead
                return COMMANDS['FORWARD']
            else:
                # Continue the curve with forward emphasis while searching
                logger.debug(f"🔍 C-CURVE EXTENDED: Searching ahead ({avoidance_progress:.2f})")
                # Force more forward movement during extended search
                if int(avoidance_progress * 50) % 3 == 0:  # Mostly forward movement
                    return COMMANDS['FORWARD']
                else:
                    return COMMANDS['LEFT'] if curve_steering > 0 else COMMANDS['RIGHT']
        
        # Generate command based on curve steering with STRONG forward bias
        logger.debug(f"🌊 {phase_name}: progress={avoidance_progress:.2f}, "
                    f"steering={curve_steering:.2f}, offset={curve_center_offset:.2f}, forward={forward_component:.2f}")
        
        # Convert curve steering to command with heavy forward emphasis
        steering_threshold = 0.4  # Higher threshold = more forward movement
        
        # During forward phases, prioritize forward movement even more
        if avoidance_phase in ['c_curve_forward', 'c_curve_clear']:
            forward_ratio = 0.8  # 80% forward movement during these phases
            if int(avoidance_progress * 100) % 10 < (forward_ratio * 10):
                return COMMANDS['FORWARD']
        
        # For all phases, use forward-biased steering
        if abs(curve_steering) < 0.15:  # Very small steering = forward
            return COMMANDS['FORWARD']
        elif abs(curve_steering) < steering_threshold:
            # Light steering - alternate with forward for smooth motion
            if int(avoidance_progress * 100) % 4 < 3:  # 75% forward, 25% turn
                return COMMANDS['FORWARD']
            else:
                return COMMANDS['LEFT'] if curve_steering > 0 else COMMANDS['RIGHT']
        else:
            # Stronger steering needed
            if int(avoidance_progress * 100) % 3 < 2:  # Still 66% forward movement
                return COMMANDS['FORWARD']
            else:
                return COMMANDS['LEFT'] if curve_steering > 0 else COMMANDS['RIGHT']
    
    # NORMAL LINE FOLLOWING BEHAVIOR
    if abs(steering) < STEERING_DEADZONE:
        return COMMANDS['FORWARD']
    
    if abs(steering) > 0.45:
        return COMMANDS['RIGHT'] if steering < 0 else COMMANDS['LEFT']
    
    return COMMANDS['RIGHT'] if steering < 0 else COMMANDS['LEFT']

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
    
    # Draw detected objects with smooth curve visualization
    for obj in detected_objects:
        bbox = obj['bbox']
        x, y, w, h = bbox
        y_adjusted = zones.get('top', (0, 0))[0] + y
        
        # Draw object bounding box
        cv2.rectangle(frame, (x, y_adjusted), (x + w, y_adjusted + h), (0, 0, 255), 2)
        
        # Draw object label
        if 'class_name' in obj:
            label = f"{obj['class_name']} ({obj['confidence']:.2f})"
        else:
            label = "OBJECT"
        
        cv2.putText(frame, label, (x, y_adjusted - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Draw curve path visualization
        if avoidance_phase != 'none':
            # Draw planned curve path
            curve_points = []
            for i in range(0, 101, 5):  # 0 to 100% in 5% steps
                progress = i / 100.0
                curve_calc = CShapedCurveCalculator()
                curve_steering, _ = curve_calc.calculate_curve_steering(
                    progress, 1.0, target_curve_offset, 0.0
                )
                
                # Convert to screen coordinates
                curve_x = center_x + int(curve_steering * width * 0.3)
                curve_y = height - int(progress * height * 0.6)
                curve_points.append((curve_x, curve_y))
            
            # Draw curve path
            for i in range(len(curve_points) - 1):
                cv2.line(frame, curve_points[i], curve_points[i + 1], (255, 255, 0), 2)
            
            # Draw current position on curve
            if avoidance_progress > 0:
                current_curve_x = center_x + int(curve_center_offset * width * 0.3)
                current_curve_y = height - int(avoidance_progress * height * 0.6)
                cv2.circle(frame, (current_curve_x, current_curve_y), 8, (255, 0, 255), -1)
        
        # Draw obstacle warning
        cv2.putText(frame, "C-SHAPED CURVE AVOIDANCE", (width//2 - 120, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
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
    
    # Draw C-shaped curve avoidance status
    if avoidance_phase != 'none':
        if avoidance_phase == 'c_curve_out':
            phase_text = f"C-CURVE: OUT ({avoidance_progress:.1%})"
        elif avoidance_phase == 'c_curve_forward':
            phase_text = f"C-CURVE: FORWARD ({avoidance_progress:.1%})"
        elif avoidance_phase == 'c_curve_clear':
            phase_text = f"C-CURVE: CLEARING ({avoidance_progress:.1%})"
        elif avoidance_phase == 'c_curve_search':
            phase_text = f"C-CURVE: SEARCHING ({avoidance_progress:.1%})"
        else:
            phase_text = f"C-CURVE: {avoidance_phase.upper()}"
        
        cv2.putText(frame, phase_text, (width//2 - 150, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw progress bar
        bar_width = 200
        bar_height = 10
        bar_x = width//2 - bar_width//2
        bar_y = 130
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        # Progress
        progress_width = int(bar_width * avoidance_progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 255), -1)
    
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
    
    # Draw direction arrow with enhanced colors
    arrow_y = height - 30
    arrow_color = (0, 255, 0)  # Green for forward
    arrow_text = "FORWARD"
    
    if turn_command == COMMANDS['LEFT']:
        arrow_color = (0, 255, 255)  # Yellow for left
        arrow_text = "LEFT"
    elif turn_command == COMMANDS['RIGHT']:
        arrow_color = (255, 255, 0)  # Cyan for right
        arrow_text = "RIGHT"
    elif turn_command == COMMANDS['CURVE_LEFT']:
        arrow_color = (255, 0, 255)  # Magenta for curve left
        arrow_text = "CURVE LEFT"
    elif turn_command == COMMANDS['CURVE_RIGHT']:
        arrow_color = (255, 0, 255)  # Magenta for curve right
        arrow_text = "CURVE RIGHT"
    elif turn_command == COMMANDS['STOP']:
        arrow_color = (0, 0, 255)  # Red for stop
        arrow_text = "STOP"
    
    cv2.putText(frame, arrow_text, (width // 2 - 50, arrow_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, arrow_color, 2)

# -----------------------------------------------------------------------------
# --- Flask Web Interface ---
# -----------------------------------------------------------------------------
app = Flask(__name__)

@app.route('/')
def index():
    """Serve the modern dashboard with smooth curve visualization"""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Line Follower Robot - Smooth Curve Avoidance</title>
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
        .cmd-curve_left { background: linear-gradient(45deg, #9C27B0, #E91E63); }
        .cmd-curve_right { background: linear-gradient(45deg, #9C27B0, #E91E63); }
        
        .curve-visualization {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            margin-top: 20px;
        }
        
        .curve-progress {
            width: 100%;
            height: 20px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .curve-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #FF9800, #FFC107, #4CAF50);
            border-radius: 10px;
            transition: width 0.3s ease;
        }
        
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
            <h1>🌊 C-Shaped Curve Line Follower</h1>
            <div>Advanced Navigation Dashboard <span id="status-dot" class="status-indicator status-offline"></span></div>
        </div>
        
        <div class="dashboard">
            <div class="video-section">
                <h3 style="margin-bottom: 15px;">🎥 Live Camera Feed with Curve Visualization</h3>
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
        
        <div class="curve-visualization" id="curve-section" style="display: none;">
            <h3>🌊 C-Shaped Curve Avoidance Progress</h3>
            <div class="curve-progress">
                <div class="curve-progress-fill" id="curve-progress-bar" style="width: 0%"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.9em; margin-top: 10px;">
                <span>Phase: <span id="curve-phase">None</span></span>
                <span>Progress: <span id="curve-progress-text">0%</span></span>
                <span>Side: <span id="curve-side">None</span></span>
            </div>
        </div>
        
        <div class="smart-features">
            <div class="feature-section">
                <h3>🎯 Adaptive PID Controller</h3>
                <div class="pid-display">
                    <div class="pid-param">
                        <span class="param-label">KP:</span>
                        <span class="param-value" id="pid-kp">0.350</span>
                    </div>
                    <div class="pid-param">
                        <span class="param-label">KI:</span>
                        <span class="param-value" id="pid-ki">0.005</span>
                    </div>
                    <div class="pid-param">
                        <span class="param-label">KD:</span>
                        <span class="param-value" id="pid-kd">0.250</span>
                    </div>
                </div>
            </div>
            
            <div class="feature-section">
                <h3>🚀 Navigation Statistics</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
                        <div style="font-size: 0.8em; opacity: 0.8;">Objects Detected:</div>
                        <div style="font-size: 1.3em; font-weight: bold;" id="objects-detected">0</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
                        <div style="font-size: 0.8em; opacity: 0.8;">Avoidance Maneuvers:</div>
                        <div style="font-size: 1.3em; font-weight: bold;" id="avoidance-maneuvers">0</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
                        <div style="font-size: 0.8em; opacity: 0.8;">Corner Count:</div>
                        <div style="font-size: 1.3em; font-weight: bold;" id="corner-count">0</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
                        <div style="font-size: 0.8em; opacity: 0.8;">Total Frames:</div>
                        <div style="font-size: 1.3em; font-weight: bold;" id="total-frames">0</div>
                    </div>
                </div>
            </div>
            
            <div class="feature-section">
                <h3>🧠 Smart Features Status</h3>
                <div style="display: flex; flex-direction: column; gap: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span>YOLO Detection:</span>
                        <span id="yolo-status" style="color: #4CAF50;">●</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span>C-Shaped Curves:</span>
                        <span style="color: #4CAF50;">● Active</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span>Speech System:</span>
                        <span id="speech-status" style="color: #4CAF50;">●</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span>Corner Prediction:</span>
                        <span id="corner-warning" style="color: #666;">● Ready</span>
                    </div>
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
                    commandDisplay.className = 'command-display cmd-' + command.toLowerCase().replace('_', '_');
                    
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
                    
                    // Update statistics
                    document.getElementById('objects-detected').textContent = data.objects_detected || 0;
                    document.getElementById('avoidance-maneuvers').textContent = data.avoidance_maneuvers || 0;
                    document.getElementById('corner-count').textContent = data.corner_count || 0;
                    document.getElementById('total-frames').textContent = data.total_frames || 0;
                    
                    // Update curve avoidance visualization
                    const curveSection = document.getElementById('curve-section');
                    if (data.avoidance_phase && data.avoidance_phase !== 'none') {
                        curveSection.style.display = 'block';
                        document.getElementById('curve-phase').textContent = data.avoidance_phase || 'Unknown';
                        document.getElementById('curve-progress-text').textContent = Math.round((data.avoidance_progress || 0) * 100) + '%';
                        document.getElementById('curve-progress-bar').style.width = (data.avoidance_progress || 0) * 100 + '%';
                        document.getElementById('curve-side').textContent = data.avoidance_side || 'None';
                    } else {
                        curveSection.style.display = 'none';
                    }
                    
                    // Update feature status indicators
                    const yoloStatus = document.getElementById('yolo-status');
                    if (data.yolo_available) {
                        yoloStatus.textContent = '● Active';
                        yoloStatus.style.color = '#4CAF50';
                    } else {
                        yoloStatus.textContent = '● Disabled';
                        yoloStatus.style.color = '#666';
                    }
                    
                    const speechStatus = document.getElementById('speech-status');
                    if (data.speech_enabled) {
                        speechStatus.textContent = '● ' + (data.tts_engine || 'Active');
                        speechStatus.style.color = '#4CAF50';
                    } else {
                        speechStatus.textContent = '● Disabled';
                        speechStatus.style.color = '#666';
                    }
                    
                    const cornerWarning = document.getElementById('corner-warning');
                    if (data.corner_warning) {
                        cornerWarning.textContent = '● Warning!';
                        cornerWarning.style.color = '#FF9800';
                    } else {
                        cornerWarning.textContent = '● Ready';
                        cornerWarning.style.color = '#666';
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
        setInterval(updateUptime, 1000);     // Update uptime every second
        
        // Initial load
        updateStatus();
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
    global robot_stats, avoidance_phase, avoidance_progress, avoidance_side
    
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
        'avoidance_phase': avoidance_phase,
        'avoidance_progress': float(avoidance_progress),
        'avoidance_side': avoidance_side,
        'pid_params': {
            'kp': float(pid_params['kp']),
            'ki': float(pid_params['ki']),
            'kd': float(pid_params['kd'])
        },
        'yolo_available': bool(YOLO_AVAILABLE and yolo_model is not None),
        'speech_enabled': bool(SPEECH_ENABLED and TTS_AVAILABLE and speech_manager and speech_manager.engine is not None),
        'tts_engine': TTS_ENGINE if TTS_AVAILABLE else 'None'
    }
    return jsonify(response)

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
    global line_detected, current_fps, confidence, esp_connection, robot_stats, yolo_model, speech_manager
    
    logger.info("🌊 Starting Smart Line Follower Robot with C-Shaped Curve Avoidance")
    robot_status = "Starting camera"
    robot_stats['start_time'] = time.time()
    
    # Initialize YOLO model if available and enabled
    if USE_YOLO and YOLO_AVAILABLE and OBJECT_DETECTION_ENABLED:
        try:
            logger.info(f"Loading YOLO11n model: {YOLO_MODEL_SIZE}")
            robot_status = "Loading YOLO model"
            yolo_model = YOLO(YOLO_MODEL_SIZE)
            logger.info("✅ YOLO11n model loaded successfully")
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
    
    # Initialize C-shaped curve avoidance system
    if OBJECT_DETECTION_ENABLED and CURVE_AVOIDANCE_ENABLED:
        logger.info("🌊 Initializing C-shaped curve avoidance system")
        robot_status = "Loading C-shaped curve system"
    else:
        logger.info("Object avoidance disabled or using traditional avoidance")
    
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
    
    # Initialize and start speech system
    speech_manager = SpeechManager()
    speech_manager.start()
    
    # Start web interface in a separate thread
    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()
    logger.info("🌐 Web dashboard available at http://localhost:5000")
    
    # FPS calculation
    fps_history = deque(maxlen=30)
    search_counter = 0
    
    # History for smoothing
    offset_history = deque(maxlen=3)
    steering_history = deque(maxlen=3)
    last_known_good_offset = 0.0
    corner_detected_count = 0
    
    # Speech announcement tracking
    last_line_detected = False
    last_avoidance_announced = None
    
    logger.info("✅ Robot ready! Starting C-shaped curve line detection...")
    robot_status = "Ready - Searching for line"
    speech_manager.announce("Smart Line Follower Robot with C-shaped curve avoidance ready!", "startup", force=True)
    
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
                
                # Announce line detection if just found
                if not last_line_detected:
                    speech_manager.announce("Line detected. Following path.", "line_found")
                last_line_detected = True
                
                # Calculate line position relative to center (-1.0 to 1.0)
                center_x = frame.shape[1] / 2
                raw_offset = (line_x - center_x) / center_x
                
                # Add to history for smoothing
                offset_history.append(raw_offset)
                
                # Use weighted average of recent offsets for stability
                if len(offset_history) > 0:
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
                    robot_stats['corner_count'] = robot_stats.get('corner_count', 0) + 1
                    
                    if corner_detected_count > 5:
                        robot_status = f"Taking corner ({corner_detected_count})"
                        # For sharp corners, exaggerate the steering
                        if abs(line_offset) > 0.5:
                            line_offset *= 1.3  # Increase turning response
                    else:
                        robot_status = f"Corner detected ({corner_detected_count})"
                elif corner_warning:
                    robot_status = f"Corner predicted ahead"
                else:
                    if corner_detected_count > 0:
                        corner_detected_count -= 1
                    else:
                        corner_detected_count = 0
                        
                    # Enhanced status with C-shaped curve avoidance phases
                    if avoidance_phase == 'c_curve_out':
                        robot_status = f"🌊 C-CURVE: Curving out ({avoidance_progress:.1%})"
                    elif avoidance_phase == 'c_curve_forward':
                        robot_status = f"🌊 C-CURVE: Moving forward around obstacle ({avoidance_progress:.1%})"
                    elif avoidance_phase == 'c_curve_clear':
                        robot_status = f"🌊 C-CURVE: Clearing obstacle ({avoidance_progress:.1%})"
                    elif avoidance_phase == 'c_curve_search':
                        robot_status = f"🌊 C-CURVE: Searching for line ahead ({avoidance_progress:.1%})"
                    elif object_detected:
                        robot_status = f"Obstacle detected - Starting C-shaped curve"
                        if last_avoidance_announced != 'detected':
                            speech_manager.announce("Obstacle detected! Starting C-shaped curve avoidance.", "obstacle", force=True)
                            last_avoidance_announced = 'detected'
                    elif detected_objects:
                        robot_status = f"Obstacle candidate detected"
                    else:
                        robot_status = f"Following line (C:{confidence:.2f})"
                
                # Calculate steering using PID controller
                steering_error = -line_offset  # Negative offset means line is to the left
                raw_steering = pid.calculate(steering_error)
                
                # Apply light exponential smoothing for responsive control
                alpha = 0.7  # Smoothing factor
                if hasattr(main, 'last_steering'):
                    steering_value = alpha * raw_steering + (1 - alpha) * main.last_steering
                else:
                    steering_value = raw_steering
                main.last_steering = steering_value
                
                # Add to steering history for additional smoothing
                steering_history.append(steering_value)
                
                # Use simple average of recent steering values
                if len(steering_history) > 0:
                    avg_steering = sum(steering_history) / len(steering_history)
                else:
                    avg_steering = steering_value
                
                # Convert steering to command with C-SHAPED CURVE AVOIDANCE
                should_avoid = CURVE_AVOIDANCE_ENABLED and OBJECT_DETECTION_ENABLED and object_detected
                
                turn_command = get_c_shaped_curve_command(avg_steering, 
                                                         detected_objects=detected_objects if should_avoid else None,
                                                         line_detected_now=True,
                                                         line_offset_now=line_offset)
                
                logger.debug(f"Line at x={line_x}, offset={line_offset:.2f}, steering={steering_value:.2f}")
                
            else:
                # Line not detected
                line_detected = False
                
                # Announce line loss if just lost
                if last_line_detected:
                    speech_manager.announce("Line lost. Searching for path.", "line_lost")
                last_line_detected = False
                
                # If we're in a C-shaped curve avoidance sequence, continue it
                if avoidance_phase != 'none':
                    turn_command = get_c_shaped_curve_command(
                        0.0,
                        detected_objects=detected_objects if OBJECT_DETECTION_ENABLED else None,
                        line_detected_now=False,
                        line_offset_now=0.0
                    )
                    robot_status = f"🌊 C-SHAPED CURVE: {avoidance_phase} (no line)"
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
                        should_avoid = CURVE_AVOIDANCE_ENABLED and OBJECT_DETECTION_ENABLED and object_detected
                        turn_command = get_c_shaped_curve_command(
                            0.0, 
                            detected_objects=detected_objects if should_avoid else None,
                            line_detected_now=False, 
                            line_offset_now=0.0
                        )
                        robot_status = "Searching while avoiding obstacles"
                    else:
                        should_avoid = CURVE_AVOIDANCE_ENABLED and OBJECT_DETECTION_ENABLED and object_detected
                        if should_avoid:
                            turn_command = get_c_shaped_curve_command(
                                0.0, 
                                detected_objects=detected_objects,
                                line_detected_now=False, 
                                line_offset_now=0.0
                            )
                            robot_status = "🔍 Line lost but still in C-shaped curve"
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
            
            # Send command to ESP32
            if esp_connection:
                success = esp_connection.send_command(turn_command)
                if not success and frame_count % 30 == 0:  # Log occasionally
                    logger.warning("ESP32 communication failed")
            
            # Draw enhanced debug information with curve visualization
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
            if frame_count % 120 == 0:  # Every 4 seconds
                kp, ki, kd = pid.get_params()
                curve_status = ""
                if avoidance_phase != 'none':
                    curve_status = f" | Curve: {avoidance_phase} ({avoidance_progress:.1%})"
                
                esp32_status = "Connected" if esp_connected else "Disconnected"
                logger.info(f"Status: {robot_status} | FPS: {current_fps:.1f} | "
                           f"PID: {kp:.3f}/{ki:.3f}/{kd:.3f} | ESP32: {esp32_status}"
                           f" | Objects: {len(detected_objects) if 'detected_objects' in locals() else 0}{curve_status}")
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        logger.info("🛑 Stopping robot (Ctrl+C pressed)")
        robot_status = "Shutting down"
        speech_manager.announce("Shutting down C-shaped curve robot.", "shutdown", force=True)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}", exc_info=True)
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
        
        if speech_manager:
            speech_manager.stop()
            logger.info("Speech system stopped")
        
        logger.info("🌊 C-shaped curve robot stopped cleanly")

if __name__ == "__main__":
    main()