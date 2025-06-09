#!/usr/bin/env python3
"""
Advanced Line Following Robot with C-Shaped Curve Obstacle Avoidance

This script controls a robot to follow a line, using advanced computer vision
techniques. It features multi-zone image analysis, adaptive PID control for
smooth steering, and a sophisticated C-shaped curve algorithm for intelligent
obstacle avoidance.

Features:
- Multi-Zone Vision: Divides camera view for line following, corner prediction,
  and object detection.
- Obstacle Avoidance:
  - Uses YOLOv8n for high-accuracy object detection (if available).
  - Fallback to classic computer vision for obstacle detection.
  - Implements a C-shaped curve maneuver for smooth, intelligent avoidance.
- Adaptive PID Control: Auto-tunes PID parameters for optimal steering response.
- Text-to-Speech (TTS): Provides audible status updates using Piper (high-quality)
  or pyttsx3 (fallback).
- Web Dashboard: A comprehensive Flask-based web UI for real-time monitoring of
  the camera feed, robot status, and performance metrics.
- Robust Communication: Manages a stable socket connection to an ESP32 for
  motor control.

Setup and Dependencies:
1. Install Python dependencies:
   pip install opencv-python numpy flask ultralytics pyttsx3

2. (Optional, for high-quality TTS) Setup Piper TTS:
   - Create a 'piper' directory.
   - Download the Piper executable into it.
   - Create a 'voices' directory.
   - Download a voice model (e.g., en_US-lessac-medium.onnx) into it.

3. (Optional, for Piper TTS on Linux) Ensure 'aplay' is installed:
   sudo apt-get install alsa-utils

4. Run the script:
   python3 your_script_name.py
"""

import cv2
import numpy as np
import socket
import time
import logging
import threading
import json
import os
import subprocess
import queue
import shutil
from collections import deque
from flask import Flask, Response, jsonify

# --- Dependency Availability Checks ---

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

# --- Configuration ---

class Config:
    """
    Central configuration class to hold all static parameters.
    """
    class Network:
        ESP32_IP = '192.168.2.21'
        ESP32_PORT = 1234
        WEB_SERVER_PORT = 5000

    class Camera:
        WIDTH = 640
        HEIGHT = 480
        FPS = 30
        BUFFER_SIZE = 1 # Lower buffer for reduced latency

    class Vision:
        # Pre-processing
        BLUR_SIZE = 5
        BLACK_THRESHOLD = 80 # Threshold for detecting the line

        # Multi-zone detection heights (as percentage of frame height)
        ZONE_BOTTOM_HEIGHT = 0.25  # Primary line following
        ZONE_MIDDLE_HEIGHT = 0.20  # Corner prediction
        ZONE_TOP_HEIGHT = 0.45     # Object detection

        # Contour filtering
        MIN_LINE_CONTOUR_AREA = 50

    class CornerDetection:
        ENABLED = True
        CONFIDENCE_BOOST = 1.2    # How much to boost confidence for corner-like shapes
        CIRCULARITY_THRESHOLD = 0.5 # Lower values are less circular (more corner-like)
        PREDICTION_THRESHOLD = 0.3  # Confidence needed to trigger a corner warning

    class ObjectDetection:
        ENABLED = True
        USE_YOLO = True # Master switch to use YOLO if available
        USE_CV_FALLBACK = True # Use basic CV if YOLO is off or unavailable

        # YOLO specific settings
        class YOLO:
            MODEL_PATH = "yolov8n.pt" # Nano model for speed
            CONFIDENCE_THRESHOLD = 0.45
            # Classes to avoid (COCO model: 0=person, 39=bottle, 41=cup, etc.)
            CLASSES_TO_AVOID = list(range(39, 80)) + [0]

        # Classic Computer Vision (CV) fallback settings
        class CV:
            OBJECT_SIZE_THRESHOLD = 800     # Min contour area to be considered an obstacle
            OBJECT_WIDTH_THRESHOLD = 0.20   # Min width ratio to trigger avoidance
            OBJECT_HEIGHT_THRESHOLD = 0.15  # Min height ratio to trigger avoidance
            OBJECT_ASPECT_RATIO_RANGE = (0.3, 3.0) # Filter shapes
            OBJECT_LINE_BLOCKING_THRESHOLD = 0.7 # How close to the line path an object must be

    class Control:
        # Commands sent to the ESP32
        COMMANDS = {
            'FORWARD': 'FORWARD', 'LEFT': 'LEFT', 'RIGHT': 'RIGHT', 'STOP': 'STOP',
            'CURVE_LEFT': 'LEFT', 'CURVE_RIGHT': 'RIGHT' # Mapped to basic turns
        }
        # Steering parameters
        STEERING_DEADZONE = 0.08 # Ignore very small steering errors
        MAX_STEERING = 1.0       # Max steering output

    class PID:
        # Initial PID gains
        KP = 0.40
        KI = 0.008
        KD = 0.28
        MAX_INTEGRAL = 2.0 # Anti-windup limit

        # Auto-tuning parameters
        ADAPTATION_ENABLED = True
        LEARNING_RATE = 0.0005
        ADAPTATION_WINDOW = 100 # Number of frames to average for performance
        PERFORMANCE_THRESHOLD = 0.12 # Error threshold to trigger tuning

    class Avoidance:
        # C-Shaped Curve maneuver parameters
        ENABLED = True
        CURVE_RADIUS_MULTIPLIER = 2.2   # How wide the curve is relative to obstacle size
        CURVE_DEPTH_MULTIPLIER = 1.5    # How "deep" the C-shape is
        CURVE_SHARPNESS = 1.4           # Exaggerates the sharpness of the curve
        SAFETY_MARGIN = 1.5             # Safety buffer around the obstacle's width
        PROGRESS_RATE = 0.25            # Speed of maneuver progress (units per second)

    class Speech:
        ENABLED = True
        USE_PIPER = True # Use Piper TTS if available
        # Check for Piper executable and a default voice model
        PIPER_PATH = './piper/piper'
        VOICE_MODEL_PATH = './voices/en_US-lessac-medium.onnx'
        PIPER_AVAILABLE = os.path.exists(PIPER_PATH) and os.path.exists(VOICE_MODEL_PATH)
        # Check for 'aplay' command needed by Piper implementation
        APLAY_AVAILABLE = shutil.which('aplay') is not None
        # pyttsx3 fallback settings
        SPEECH_RATE = 160
        SPEECH_VOLUME = 0.9
        # Cooldown between announcements of the same type (seconds)
        ANNOUNCE_INTERVAL = 5.0

# --- State Management ---

class RobotState:
    """Holds the current dynamic state of the robot."""
    def __init__(self):
        self.status = "Initializing"
        self.command = Config.Control.COMMANDS['STOP']
        self.line_offset = 0.0
        self.steering_value = 0.0
        self.confidence = 0.0
        self.line_detected = False
        self.esp_connected = False
        self.fps = 0.0
        self.corner_warning = False
        self.object_detected = False
        self.stats = {
            'uptime_start': time.time(), 'total_frames': 0, 'lost_frames': 0,
            'corner_count': 0, 'objects_detected': 0, 'avoidance_maneuvers': 0
        }
        self.pid_params = {'kp': Config.PID.KP, 'ki': Config.PID.KI, 'kd': Config.PID.KD}

class AvoidanceState:
    """Holds the state for the C-shaped curve avoidance maneuver."""
    def __init__(self):
        self.phase = 'none' # 'none', 'curve_out', 'curve_around', 'curve_back'
        self.side = 'none'  # 'left' or 'right'
        self.progress = 0.0 # 0.0 to 1.0
        self.current_offset = 0.0
        self.target_offset = 0.0
        self.current_obstacle = None

# --- Global Variables (for cross-thread communication) ---
output_frame = None
frame_lock = threading.Lock()
logger = logging.getLogger("LineFollowerRobot")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)-8s] %(message)s',
    datefmt='%H:%M:%S'
)


# --- Core Modules ---

class SpeechManager:
    """Manages Text-to-Speech (TTS) announcements in a separate thread."""
    def __init__(self):
        self.engine = None
        self.tts_engine_type = None
        self.speech_queue = queue.Queue()
        self.running = False
        self.last_announcements = {}
        self._initialize_engine()

    def _initialize_engine(self):
        """Initializes the best available TTS engine."""
        if not Config.Speech.ENABLED:
            logger.info("Speech system is disabled in config.")
            return

        # Prefer Piper for high-quality, local, neural speech
        if Config.Speech.USE_PIPER and Config.Speech.PIPER_AVAILABLE:
            if not Config.Speech.APLAY_AVAILABLE:
                logger.warning("Piper TTS configured, but 'aplay' command not found. Speech will be disabled.")
                return
            self.tts_engine_type = 'piper'
            self.engine = True # Use a simple flag for Piper
            logger.info("SpeechManager: Piper TTS engine initialized.")
        # Fallback to pyttsx3
        elif PYTTSX3_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', Config.Speech.SPEECH_RATE)
                self.engine.setProperty('volume', Config.Speech.SPEECH_VOLUME)
                self.tts_engine_type = 'pyttsx3'
                logger.info("SpeechManager: pyttsx3 engine initialized.")
            except Exception as e:
                logger.error(f"SpeechManager: Failed to initialize pyttsx3: {e}")
                self.engine = None
        else:
            logger.warning("SpeechManager: No TTS engines available (Piper or pyttsx3).")

    def start(self):
        """Starts the speech worker thread."""
        if not self.engine:
            return
        self.running = True
        thread = threading.Thread(target=self._speech_worker, daemon=True)
        thread.start()
        logger.info(f"Speech worker started with '{self.tts_engine_type}' engine.")

    def stop(self):
        """Stops the speech worker thread."""
        self.running = False

    def announce(self, message, category="general", force=False):
        """Adds a message to the speech queue, with throttling."""
        if not self.engine:
            return

        current_time = time.time()
        # Throttle announcements to avoid spamming
        if not force and category in self.last_announcements:
            if current_time - self.last_announcements[category] < Config.Speech.ANNOUNCE_INTERVAL:
                return

        self.speech_queue.put(message)
        self.last_announcements[category] = current_time
        logger.info(f"Speech: Queued '{message}'")

    def _speech_worker(self):
        """The background thread that processes the speech queue."""
        while self.running:
            try:
                message = self.speech_queue.get(timeout=1.0)
                logger.info(f"Speech: Speaking '{message}'")
                if self.tts_engine_type == 'piper':
                    self._speak_with_piper(message)
                elif self.tts_engine_type == 'pyttsx3':
                    self._speak_with_pyttsx3(message)
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Speech worker error: {e}")
        logger.info("Speech worker stopped.")

    def _speak_with_piper(self, message):
        """Generates and plays audio using Piper TTS."""
        try:
            # Using a temporary file for the audio output
            with open("/tmp/robot_speech.wav", "w") as f:
                subprocess.run(
                    [Config.Speech.PIPER_PATH, '--model', Config.Speech.VOICE_MODEL_PATH, '--output_file', f.name],
                    input=message, text=True, check=True, capture_output=True, timeout=10
                )
                # Play the generated audio file using aplay
                subprocess.run(['aplay', f.name], check=True, capture_output=True, timeout=10)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"Piper TTS execution failed: {e.stderr.decode() if e.stderr else str(e)}")
        except FileNotFoundError:
             logger.error("Piper or aplay command not found. Ensure they are in your system's PATH.")
        except Exception as e:
            logger.error(f"An unexpected error occurred with Piper TTS: {e}")

    def _speak_with_pyttsx3(self, message):
        """Speaks a message using the pyttsx3 library."""
        try:
            self.engine.say(message)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"pyttsx3 error: {e}")

class CShapedCurveCalculator:
    """
    Calculates the parameters and steering for a smooth, C-shaped
    obstacle avoidance maneuver.
    """
    @staticmethod
    def calculate_curve_parameters(obstacle_pos, obstacle_width):
        """
        Determines the direction and target offset for the C-curve.

        Args:
            obstacle_pos (float): The obstacle's horizontal position (-1.0 to 1.0).
            obstacle_width (float): The obstacle's width as a ratio of the frame width.

        Returns:
            tuple: (avoidance_side, target_offset)
        """
        safety_buffer = obstacle_width * Config.Avoidance.SAFETY_MARGIN
        base_offset = safety_buffer + 0.4 # Additional fixed buffer

        if obstacle_pos > 0: # Obstacle on the right, so curve left
            side = 'left'
            target_offset = -base_offset * Config.Avoidance.CURVE_DEPTH_MULTIPLIER
        else: # Obstacle on the left, so curve right
            side = 'right'
            target_offset = base_offset * Config.Avoidance.CURVE_DEPTH_MULTIPLIER

        return side, target_offset

    @staticmethod
    def _smoothstep(x):
        """A smooth transition function (hermite interpolation)."""
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    @staticmethod
    def get_curve_steering(progress, target_offset, current_offset):
        """
        Calculates the required steering value based on the maneuver's progress.

        The curve is divided into three phases for a distinct "C" shape:
        1. Curve Out (0% - 25%): A sharp turn away from the line.
        2. Curve Around (25% - 75%): A wider, sustained turn around the obstacle.
        3. Curve Back (75% - 100%): A sharp turn back towards the line.

        Args:
            progress (float): The current progress of the maneuver (0.0 to 1.0).
            target_offset (float): The maximum desired lateral offset from the line.
            current_offset (float): The robot's current estimated lateral offset.

        Returns:
            float: The calculated steering value.
        """
        if progress <= 0.25: # Phase 1: Curve Out
            phase_progress = progress / 0.25
            desired_offset = target_offset * CShapedCurveCalculator._smoothstep(phase_progress) * 0.8
        elif progress <= 0.75: # Phase 2: Curve Around
            desired_offset = target_offset
        else: # Phase 3: Curve Back
            phase_progress = (progress - 0.75) / 0.25
            desired_offset = target_offset * (1.0 - CShapedCurveCalculator._smoothstep(phase_progress))

        # Calculate steering needed to move from current to desired offset
        offset_error = desired_offset - current_offset
        steering = offset_error * Config.Avoidance.CURVE_SHARPNESS

        return np.clip(steering, -Config.Control.MAX_STEERING, Config.Control.MAX_STEERING)


class AdaptivePID:
    """
    An adaptive PID controller that fine-tunes its gains based on performance.
    """
    def __init__(self, kp, ki, kd, max_integral):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.max_integral = max_integral
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        self.error_history = deque(maxlen=Config.PID.ADAPTATION_WINDOW)

    def calculate(self, error):
        """
        Calculates the PID output for a given error.

        Args:
            error (float): The error term (e.g., negative line offset).

        Returns:
            float: The calculated control output (steering value).
        """
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        if dt <= 0: return self.prev_error * self.kd # Avoid division by zero

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        i_term = self.ki * self.integral

        # Derivative term
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative

        # Total output
        output = p_term + i_term + d_term

        # Update state
        self.prev_error = error
        self.error_history.append(abs(error))

        if Config.PID.ADAPTATION_ENABLED:
            self._adapt_parameters()

        return np.clip(output, -Config.Control.MAX_STEERING, Config.Control.MAX_STEERING)

    def _adapt_parameters(self):
        """Heuristic-based auto-tuning of PID gains."""
        if len(self.error_history) < Config.PID.ADAPTATION_WINDOW:
            return

        avg_error = np.mean(self.error_history)
        error_variance = np.var(self.error_history)

        # If error is large, the system is not tracking well
        if avg_error > Config.PID.PERFORMANCE_THRESHOLD:
            # If variance is high, it's oscillating -> increase D to dampen
            if error_variance > 0.05:
                self.kd += Config.PID.LEARNING_RATE * 2
                self.kp *= 0.98 # Slightly reduce P to prevent overshoot
            # If variance is low, it's a steady state error -> increase P
            else:
                self.kp += Config.PID.LEARNING_RATE
        # If error is very low, the system is stable
        elif avg_error < Config.PID.PERFORMANCE_THRESHOLD * 0.5:
            # If variance is also low, we can be more responsive -> slightly decrease D
            if error_variance < 0.01:
                self.kd -= Config.PID.LEARNING_RATE * 0.5

        # Clamp values to prevent them from becoming unstable
        self.kp = np.clip(self.kp, 0.1, 1.0)
        self.ki = np.clip(self.ki, 0.0, 0.05)
        self.kd = np.clip(self.kd, 0.1, 0.8)

    def get_params(self):
        """Returns the current PID gains."""
        return self.kp, self.ki, self.kd

    def reset(self):
        """Resets the PID controller's state."""
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()


class ESP32Connection:
    """Handles communication with the ESP32 motor controller."""
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.socket = None
        self.last_command = None
        self.is_connected = False
        self.reconnect_delay = 5 # seconds
        self.last_attempt_time = 0

    def connect(self):
        """Attempts to establish a socket connection."""
        if self.is_connected or time.time() - self.last_attempt_time < self.reconnect_delay:
            return

        self.last_attempt_time = time.time()
        logger.info(f"ESP32: Attempting to connect to {self.ip}:{self.port}...")
        try:
            if self.socket: self.socket.close()
            self.socket = socket.create_connection((self.ip, self.port), timeout=2)
            self.socket.settimeout(0.5)
            self.is_connected = True
            logger.info("ESP32: Connection successful.")
        except Exception as e:
            self.is_connected = False
            logger.error(f"ESP32: Connection failed: {e}")
            self.socket = None

    def send_command(self, command):
        """
        Sends a command to the ESP32 if connected.

        Args:
            command (str): The command to send (e.g., 'FORWARD').
        """
        if not self.is_connected:
            self.connect() # Attempt to reconnect if not connected
            return

        # Only send if the command has changed to reduce network traffic
        if command == self.last_command:
            return

        try:
            full_command = f"{command}\n"
            self.socket.sendall(full_command.encode())
            self.last_command = command
            logger.debug(f"ESP32: Sent '{command}'")
        except (socket.timeout, socket.error) as e:
            logger.error(f"ESP32: Send command failed: {e}")
            self.is_connected = False
            self.socket = None
            self.last_command = None # Force resend after reconnect

    def close(self):
        """Closes the connection gracefully."""
        if self.socket:
            try:
                self.send_command(Config.Control.COMMANDS['STOP'])
                time.sleep(0.1)
                self.socket.close()
            except Exception as e:
                logger.error(f"ESP32: Error while closing connection: {e}")
            finally:
                self.socket = None
                self.is_connected = False
                logger.info("ESP32: Connection closed.")


# --- Vision and Control Logic ---

def process_image(frame, yolo_model):
    """
    Analyzes a single camera frame to detect the line, corners, and obstacles.

    Args:
        frame (np.ndarray): The input image frame from the camera.
        yolo_model (YOLO, optional): The initialized YOLO model.

    Returns:
        dict: A dictionary containing all vision processing results.
    """
    height, width, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (Config.Vision.BLUR_SIZE, Config.Vision.BLUR_SIZE), 0)
    _, binary_line = cv2.threshold(blurred, Config.Vision.BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    # --- Define Zones ---
    h = height
    zone_bottom_y = int(h * (1 - Config.Vision.ZONE_BOTTOM_HEIGHT))
    zone_middle_y = int(h * (1 - Config.Vision.ZONE_BOTTOM_HEIGHT - Config.Vision.ZONE_MIDDLE_HEIGHT))
    zone_top_y = int(h * (1 - Config.Vision.ZONE_BOTTOM_HEIGHT - Config.Vision.ZONE_MIDDLE_HEIGHT - Config.Vision.ZONE_TOP_HEIGHT))

    # --- ROIs for each zone ---
    roi_bottom = binary_line[zone_bottom_y:h, :]
    roi_middle = binary_line[zone_middle_y:zone_bottom_y, :]
    roi_top_frame = frame[zone_top_y:zone_middle_y, :]

    # --- Process each zone ---
    line_x, line_confidence = _detect_line_in_roi(roi_bottom)
    corner_x, corner_confidence = _detect_line_in_roi(roi_middle)

    # --- Object Detection ---
    detected_objects = []
    if Config.ObjectDetection.ENABLED:
        use_yolo_now = Config.ObjectDetection.USE_YOLO and YOLO_AVAILABLE and yolo_model is not None
        if use_yolo_now:
            detected_objects = _detect_objects_yolo(roi_top_frame, yolo_model)
        elif Config.ObjectDetection.USE_CV_FALLBACK:
            # Invert threshold for object detection (detect dark objects on light background)
            _, binary_objects = cv2.threshold(blurred[zone_top_y:zone_middle_y, :], Config.Vision.BLACK_THRESHOLD, 255, cv2.THRESH_BINARY)
            detected_objects = _detect_objects_cv(binary_objects)

    return {
        'line_x': line_x, 'line_confidence': line_confidence,
        'corner_x': corner_x, 'corner_confidence': corner_confidence,
        'detected_objects': detected_objects,
        'zones': { # For visualization
            'bottom': (zone_bottom_y, h), 'middle': (zone_middle_y, zone_bottom_y),
            'top': (zone_top_y, zone_middle_y)
        }
    }

def _detect_line_in_roi(roi):
    """Finds the largest contour in a region of interest (ROI) and its center."""
    if roi.size == 0: return None, 0.0
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, 0.0

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    if area < Config.Vision.MIN_LINE_CONTOUR_AREA:
        return None, 0.0

    M = cv2.moments(largest_contour)
    if M["m00"] == 0: return None, 0.0

    cx = int(M["m10"] / M["m00"])
    confidence = min(area / (roi.shape[0] * roi.shape[1] * 0.1), 1.0)
    return cx, confidence

def _detect_objects_yolo(roi_frame, model):
    """Detects objects using a YOLO model."""
    objects = []
    try:
        results = model.predict(roi_frame, conf=Config.ObjectDetection.YOLO.CONFIDENCE_THRESHOLD, verbose=False)
        for res in results:
            for box in res.boxes:
                if int(box.cls[0]) in Config.ObjectDetection.YOLO.CLASSES_TO_AVOID:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w = x2 - x1
                    center_x = x1 + w / 2
                    objects.append({
                        'position': (center_x - roi_frame.shape[1] / 2) / (roi_frame.shape[1] / 2),
                        'width_ratio': w / roi_frame.shape[1],
                        'class_name': model.names[int(box.cls[0])],
                        'confidence': float(box.conf[0]),
                        'bbox': (int(x1), int(y1), int(w), int(y2-y1))
                    })
    except Exception as e:
        logger.error(f"YOLO detection error: {e}")
    return objects

def _detect_objects_cv(roi_binary):
    """Detects objects using classic computer vision (contour analysis)."""
    objects = []
    contours, _ = cv2.findContours(roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = roi_binary.shape

    for contour in contours:
        if cv2.contourArea(contour) < Config.ObjectDetection.CV.OBJECT_SIZE_THRESHOLD:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        width_ratio = w / width
        aspect_ratio = w / h if h > 0 else 0
        min_ar, max_ar = Config.ObjectDetection.CV.OBJECT_ASPECT_RATIO_RANGE

        if width_ratio > Config.ObjectDetection.CV.OBJECT_WIDTH_THRESHOLD and min_ar < aspect_ratio < max_ar:
            center_x = x + w / 2
            objects.append({
                'position': (center_x - width / 2) / (width / 2),
                'width_ratio': width_ratio,
                'class_name': 'CV_Object',
                'confidence': 1.0,
                'bbox': (x, y, w, h)
            })
    return objects

def update_robot_logic(state, avoidance, vision_data, pid, dt):
    """
    Main logic loop to determine the robot's next command based on vision input.
    This function modifies the state and avoidance objects directly.
    """
    # --- Line and Corner Detection Logic ---
    state.line_x = vision_data['line_x']
    state.confidence = vision_data['line_confidence']
    state.line_detected = state.line_x is not None and state.confidence > 0.2

    # Corner warning logic
    if Config.CornerDetection.ENABLED:
        shift = abs(vision_data['corner_x'] - state.line_x) if state.line_detected and vision_data['corner_x'] else 0
        state.corner_warning = vision_data['corner_confidence'] > Config.CornerDetection.PREDICTION_THRESHOLD and shift > Config.Camera.WIDTH * 0.15

    # --- Obstacle Detection and Avoidance State ---
    state.object_detected = bool(vision_data.get('detected_objects'))
    most_significant_object = max(vision_data['detected_objects'], key=lambda o: o['width_ratio']) if state.object_detected else None

    # --- C-Shaped Curve Avoidance State Machine ---
    if Config.Avoidance.ENABLED:
        is_avoiding = avoidance.phase != 'none'

        # State: Not avoiding, but an object is now detected -> START AVOIDANCE
        if not is_avoiding and most_significant_object:
            avoidance.phase = 'curve_out'
            avoidance.progress = 0.0
            avoidance.current_offset = state.line_offset
            avoidance.current_obstacle = most_significant_object
            avoidance.side, avoidance.target_offset = CShapedCurveCalculator.calculate_curve_parameters(
                most_significant_object['position'], most_significant_object['width_ratio']
            )
            state.stats['avoidance_maneuvers'] += 1
            state.status = f"Obstacle! Start C-Curve {avoidance.side}"
            speech_manager.announce(f"Obstacle detected. Curving {avoidance.side}.", "obstacle", force=True)

        # State: Currently avoiding -> CONTINUE AVOIDANCE
        elif is_avoiding:
            # Update progress based on time
            avoidance.progress = min(avoidance.progress + Config.Avoidance.PROGRESS_RATE * dt, 1.0)

            # Determine phase based on progress
            if avoidance.progress < 0.25: avoidance.phase = 'curve_out'
            elif avoidance.progress < 0.75: avoidance.phase = 'curve_around'
            else: avoidance.phase = 'curve_back'

            state.status = f"C-Curve: {avoidance.phase} {avoidance.side} ({avoidance.progress:.0%})"

            # Calculate the steering for the curve
            curve_steering = CShapedCurveCalculator.get_curve_steering(
                avoidance.progress, avoidance.target_offset, avoidance.current_offset
            )
            state.steering_value = curve_steering
            avoidance.current_offset += state.steering_value * dt * 5 # Update our position estimate

            # Check for completion condition
            line_reacquired = state.line_detected and abs(state.line_offset) < 0.4
            if avoidance.progress >= 1.0 and line_reacquired:
                state.status = "C-Curve Complete!"
                speech_manager.announce("Path is clear.", "obstacle")
                avoidance.__init__() # Reset avoidance state

    # --- Standard Line Following (if not avoiding) ---
    if avoidance.phase == 'none':
        if state.line_detected:
            state.status = "Following Line"
            center_x = Config.Camera.WIDTH / 2
            state.line_offset = (state.line_x - center_x) / center_x
            state.steering_value = pid.calculate(-state.line_offset) # PID error is negative offset
        else:
            state.status = "Line Lost! Searching..."
            pid.reset()
            # Simple search: turn in the direction the line was last seen
            state.steering_value = 0.7 if state.line_offset > 0 else -0.7
            if abs(state.line_offset) < 0.1: # If last seen in center, stop
                state.steering_value = 0
                state.command = Config.Control.COMMANDS['STOP']

    # --- Final Command Generation (if not stopped) ---
    if state.command != Config.Control.COMMANDS['STOP'] or not state.line_detected:
        if abs(state.steering_value) < Config.Control.STEERING_DEADZONE:
            state.command = Config.Control.COMMANDS['FORWARD']
        elif state.steering_value > 0:
            state.command = Config.Control.COMMANDS['LEFT']
        else:
            state.command = Config.Control.COMMANDS['RIGHT']

    # Update state with current PID params for the dashboard
    state.pid_params['kp'], state.pid_params['ki'], state.pid_params['kd'] = pid.get_params()

def draw_visualization(frame, state, avoidance, vision_data):
    """Draws debug information and visualizations onto the frame."""
    h, w, _ = frame.shape
    center_x = w // 2

    # Draw zones
    zones = vision_data.get('zones', {})
    cv2.rectangle(frame, (0, zones['top'][0]), (w-1, zones['top'][1]), (255, 50, 50), 2)
    cv2.putText(frame, "Object Zone", (10, zones['top'][0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 50), 1)
    cv2.rectangle(frame, (0, zones['middle'][0]), (w-1, zones['middle'][1]), (50, 255, 50), 2)
    cv2.putText(frame, "Corner Zone", (10, zones['middle'][0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 1)
    cv2.rectangle(frame, (0, zones['bottom'][0]), (w-1, zones['bottom'][1]), (50, 50, 255), 2)
    cv2.putText(frame, "Line Zone", (10, zones['bottom'][0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 255), 1)

    # Draw line detection
    if state.line_detected:
        line_y = zones['bottom'][0] + 20
        cv2.circle(frame, (state.line_x, line_y), 10, (255, 0, 255), -1)
        cv2.line(frame, (center_x, line_y), (state.line_x, line_y), (255, 0, 255), 2)

    # Draw detected objects
    for obj in vision_data.get('detected_objects', []):
        x, y, bw, bh = obj['bbox']
        y_abs = y + zones['top'][0]
        label = f"{obj['class_name']} ({obj['confidence']:.2f})"
        cv2.rectangle(frame, (x, y_abs), (x + bw, y_abs + bh), (0, 0, 255), 2)
        cv2.putText(frame, label, (x, y_abs - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw C-Curve Avoidance visualization
    if avoidance.phase != 'none':
        progress_text = f"AVOIDING ({avoidance.side}): {avoidance.phase} {avoidance.progress:.0%}"
        cv2.putText(frame, progress_text, (center_x - 150, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # Draw a progress bar for the maneuver
        bar_y = h - 50
        cv2.rectangle(frame, (center_x - 100, bar_y), (center_x + 100, bar_y + 15), (80, 80, 80), -1)
        progress_width = int(200 * avoidance.progress)
        cv2.rectangle(frame, (center_x - 100, bar_y), (center_x - 100 + progress_width, bar_y + 15), (0, 255, 255), -1)

    # Draw main status text overlay
    info_text = [
        f"Status: {state.status}",
        f"Command: {state.command}",
        f"FPS: {state.fps:.1f}",
        f"Offset: {state.line_offset:.2f} | Steer: {state.steering_value:.2f}",
        f"Confidence: {state.confidence:.2f}",
    ]
    for i, text in enumerate(info_text):
        cv2.putText(frame, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return frame


# --- Web Server ---
app = Flask(__name__)

# Global reference to the robot's state, to be populated in main()
robot_state = None
avoidance_state = None

@app.route('/')
def index():
    """Serves the main web dashboard."""
    # This HTML is kept as a string for single-file simplicity, as in the original.
    # For larger projects, using Flask's render_template() is recommended.
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Line Follower Robot Dashboard</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }
        .container { max-width: 1400px; margin: auto; display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
        .card { background-color: #1e1e1e; border-radius: 12px; padding: 20px; border: 1px solid #333; }
        h1, h2, h3 { color: #fff; border-bottom: 2px solid #03a9f4; padding-bottom: 10px; margin-top: 0; }
        img.video-feed { width: 100%; border-radius: 8px; background-color: #000; }
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .stat-item { background-color: #2a2a2a; padding: 15px; border-radius: 8px; text-align: center; }
        .stat-label { font-size: 0.9em; color: #aaa; }
        .stat-value { font-size: 1.8em; font-weight: bold; color: #4dd0e1; margin-top: 5px; }
        .status-dot { height: 12px; width: 12px; border-radius: 50%; display: inline-block; vertical-align: middle; margin-left: 8px; }
        .online { background-color: #4caf50; }
        .offline { background-color: #f44336; }
        #command-display { font-size: 1.5em; font-weight: bold; text-transform: uppercase; padding: 10px; border-radius: 5px; margin-top: 5px; }
        #pid-params { display: flex; justify-content: space-around; }
        #avoidance-progress-bar { width: 100%; background-color: #444; border-radius: 5px; overflow: hidden; height: 25px; }
        #avoidance-progress-fill { width: 0%; height: 100%; background-color: #ff9800; transition: width 0.2s; }
        @media (max-width: 900px) { .container { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="main-content">
            <div class="card">
                <h2>🎥 Live Camera Feed</h2>
                <img src="/video_feed" class="video-feed" alt="Video Feed">
            </div>
            <div class="card">
                <h3>🧠 Avoidance Maneuver</h3>
                <div id="avoidance-status">Status: Not Active</div>
                <div id="avoidance-progress-bar"><div id="avoidance-progress-fill"></div></div>
            </div>
        </div>
        <aside class="sidebar">
            <div class="card">
                <h2>📊 Robot Status</h2>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-label">System Status</div>
                        <div class="stat-value" id="robot-status">...</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">ESP32</div>
                        <div class="stat-value" id="esp-status">Offline <span class="status-dot offline"></span></div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Command</div>
                        <div class="stat-value" id="command-display">STOP</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">FPS</div>
                        <div class="stat-value" id="fps">0.0</div>
                    </div>
                </div>
            </div>
            <div class="card">
                <h3>🧭 Navigation</h3>
                 <div class="stats-grid">
                    <div class="stat-item"><div class="stat-label">Line Offset</div><div class="stat-value" id="offset">0.00</div></div>
                    <div class="stat-item"><div class="stat-label">Confidence</div><div class="stat-value" id="confidence">0%</div></div>
                 </div>
            </div>
            <div class="card">
                <h3>⚙️ Adaptive PID</h3>
                <div id="pid-params" class="stats-grid">
                    <div class="stat-item"><div class="stat-label">P</div><div class="stat-value" id="pid-kp">0.0</div></div>
                    <div class="stat-item"><div class="stat-label">I</div><div class="stat-value" id="pid-ki">0.0</div></div>
                    <div class="stat-item"><div class="stat-label">D</div><div class="stat-value" id="pid-kd">0.0</div></div>
                </div>
            </div>
        </aside>
    </div>

    <script>
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('robot-status').textContent = data.status;
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('command-display').textContent = data.command;
                    document.getElementById('offset').textContent = data.offset.toFixed(2);
                    document.getElementById('confidence').textContent = `${(data.confidence * 100).toFixed(0)}%`;

                    const espStatus = document.getElementById('esp-status');
                    espStatus.innerHTML = data.esp_connected ? 'Online <span class="status-dot online"></span>' : 'Offline <span class="status-dot offline"></span>';

                    document.getElementById('pid-kp').textContent = data.pid_params.kp.toFixed(3);
                    document.getElementById('pid-ki').textContent = data.pid_params.ki.toFixed(3);
                    document.getElementById('pid-kd').textContent = data.pid_params.kd.toFixed(3);
                    
                    const avoidanceStatus = document.getElementById('avoidance-status');
                    const avoidanceFill = document.getElementById('avoidance-progress-fill');
                    if (data.avoidance_phase !== 'none') {
                        avoidanceStatus.textContent = `Phase: ${data.avoidance_phase} (${data.avoidance_side})`;
                        avoidanceFill.style.width = `${data.avoidance_progress * 100}%`;
                    } else {
                        avoidanceStatus.textContent = 'Status: Not Active';
                        avoidanceFill.style.width = '0%';
                    }
                })
                .catch(err => console.error("Failed to fetch status:", err));
        }
        setInterval(updateStatus, 500);
        window.onload = updateStatus;
    </script>
</body>
</html>
"""

@app.route('/video_feed')
def video_feed():
    """Streams the processed video frames."""
    def generate():
        while True:
            with frame_lock:
                if output_frame is not None:
                    frame = output_frame
                else:
                    # Create a placeholder frame if none is available
                    frame = np.zeros((Config.Camera.HEIGHT, Config.Camera.WIDTH, 3), dtype=np.uint8)
                    cv2.putText(frame, "No Feed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1/Config.Camera.FPS) # Cap the stream rate
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    """Provides the robot's status as a JSON object."""
    if not robot_state or not avoidance_state:
        return jsonify({'error': 'Robot not initialized'})
        
    return jsonify({
        'status': robot_state.status,
        'command': robot_state.command,
        'offset': robot_state.line_offset,
        'fps': robot_state.fps,
        'confidence': robot_state.confidence,
        'esp_connected': robot_state.esp_connected,
        'pid_params': robot_state.pid_params,
        'avoidance_phase': avoidance_state.phase,
        'avoidance_side': avoidance_state.side,
        'avoidance_progress': avoidance_state.progress
    })

def run_web_server():
    """Runs the Flask web server in a separate thread."""
    try:
        app.run(host='0.0.0.0', port=Config.Network.WEB_SERVER_PORT, debug=False)
    except Exception as e:
        logger.error(f"Web server failed to start: {e}")

# --- Main Application ---

def main():
    """The main entry point of the robot application."""
    global output_frame, robot_state, avoidance_state, speech_manager

    # --- Initialization ---
    robot_state = RobotState()
    avoidance_state = AvoidanceState()
    speech_manager = SpeechManager()
    speech_manager.start()
    speech_manager.announce("Robot systems initializing.", "startup", force=True)

    # Initialize YOLO model
    yolo_model = None
    if Config.ObjectDetection.USE_YOLO:
        if YOLO_AVAILABLE:
            try:
                logger.info(f"Loading YOLO model: {Config.ObjectDetection.YOLO.MODEL_PATH}")
                yolo_model = YOLO(Config.ObjectDetection.YOLO.MODEL_PATH)
                logger.info("YOLO model loaded successfully.")
                speech_manager.announce("YOLO object detection is active.", "init")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}. Falling back to CV.")
                yolo_model = None
        else:
            logger.warning("YOLO is enabled in config, but 'ultralytics' package is not installed.")

    # Initialize PID controller and ESP32 connection
    pid = AdaptivePID(Config.PID.KP, Config.PID.KI, Config.PID.KD, Config.PID.MAX_INTEGRAL)
    esp_conn = ESP32Connection(Config.Network.ESP32_IP, Config.Network.ESP32_PORT)

    # Initialize Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.warning("Camera at index 0 not found, trying index 1.")
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        logger.critical("Could not open any camera. Exiting.")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.Camera.WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.Camera.HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, Config.Camera.FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, Config.Camera.BUFFER_SIZE)
    logger.info("Camera initialized successfully.")

    # Start the web server thread
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()
    logger.info(f"Web dashboard running at http://0.0.0.0:{Config.Network.WEB_SERVER_PORT}")
    
    speech_manager.announce("Initialization complete. Starting main loop.", "startup", force=True)

    # --- Main Loop ---
    fps_history = deque(maxlen=30)
    last_loop_time = time.time()
    try:
        while True:
            # Calculate delta time (dt) for time-based calculations
            current_time = time.time()
            dt = current_time - last_loop_time
            last_loop_time = current_time
            
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame from camera.")
                robot_state.stats['lost_frames'] += 1
                time.sleep(0.1)
                continue
            
            robot_state.stats['total_frames'] += 1

            # Core Logic
            vision_data = process_image(frame, yolo_model)
            update_robot_logic(robot_state, avoidance_state, vision_data, pid, dt)
            
            # Communication
            esp_conn.send_command(robot_state.command)
            robot_state.esp_connected = esp_conn.is_connected
            
            # Visualization
            display_frame = draw_visualization(frame.copy(), robot_state, avoidance_state, vision_data)
            
            # Update shared frame for web stream
            with frame_lock:
                output_frame = display_frame
            
            # Update FPS
            fps_history.append(1 / dt if dt > 0 else 0)
            robot_state.fps = np.mean(fps_history)

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user (Ctrl+C).")
        speech_manager.announce("Shutting down.", "shutdown", force=True)
    except Exception as e:
        logger.critical(f"An unhandled exception occurred in the main loop: {e}", exc_info=True)
    finally:
        # --- Cleanup ---
        logger.info("Cleaning up resources...")
        cap.release()
        esp_conn.close()
        speech_manager.stop()
        logger.info("Robot stopped cleanly.")

if __name__ == "__main__":
    main()