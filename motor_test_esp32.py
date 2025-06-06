#!/usr/bin/env python3

import cv2
import numpy as np
import socket
import time
import logging
import threading
from collections import deque
from flask import Flask, Response, render_template_string

# -----------------------------------------------------------------------------
# --- CONFIGURATION ---
# -----------------------------------------------------------------------------
# ESP32 Configuration
ESP32_IP = '192.168.53.117'  # Change this to your ESP32's IP address
ESP32_PORT = 1234
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 15

# Line following parameters
BLACK_THRESHOLD = 60  # Higher values detect darker lines
BLUR_SIZE = 5
MIN_CONTOUR_AREA = 100  # Minimum area to be considered a line

# Object detection parameters
OBJECT_DETECTION_ENABLED = True  # Set to False to disable object detection
OBJECT_AREA_THRESHOLD = 800  # Minimum contour area for objects
OBJECT_AVOIDANCE_DURATION = 15  # Frames to perform avoidance
OBJECT_AVOIDANCE_DIRECTION = 'RIGHT'  # Default direction to avoid obstacles

# PID controller values
KP = 0.6  # Proportional gain
KI = 0.02  # Integral gain
KD = 0.1  # Derivative gain
MAX_INTEGRAL = 5.0  # Prevent integral windup

# Commands for ESP32
COMMANDS = {'FORWARD': 'FORWARD', 'LEFT': 'LEFT', 'RIGHT': 'RIGHT', 'STOP': 'STOP'}
STEERING_DEADZONE = 0.1  # Ignore small errors

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
obstacle_detected = False
obstacle_avoid_counter = 0
object_detected_area = None

# -----------------------------------------------------------------------------
# --- Logging Setup ---
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, 
                   format='🤖 [%(asctime)s] %(levelname)s: %(message)s', 
                   datefmt='%H:%M:%S')
logger = logging.getLogger("LineFollower")

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
        self.connect()
    
    def connect(self):
        if self.socket:
            try: 
                self.socket.close()
            except: 
                pass
            self.socket = None
        
        try:
            self.socket = socket.create_connection((self.ip, self.port), timeout=2)
            self.socket.settimeout(0.2)
            logger.info(f"✅ Connected to ESP32 at {self.ip}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to ESP32: {e}")
            self.socket = None
            return False
    
    def send_command(self, command):
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
            logger.error(f"❌ Error sending command to ESP32: {e}")
            self.socket = None
            return False
    
    def close(self):
        if self.socket:
            try:
                self.send_command("STOP")
                time.sleep(0.1)
                self.socket.close()
            except:
                pass
            self.socket = None
            logger.info("🔌 Disconnected from ESP32")

# -----------------------------------------------------------------------------
# --- Image Processing ---
# -----------------------------------------------------------------------------
def detect_line(frame):
    """Detect black line in image and return line position"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)
    
    # Threshold the image to identify black regions
    _, binary = cv2.threshold(blurred, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    
    # Focus on bottom part of the image (where the line is most visible)
    height, width = binary.shape
    bottom_height = int(height * 0.3)  # Bottom 30% of the image
    bottom_roi = binary[height - bottom_height:height, :]
    
    # Middle part for corner detection
    middle_height = int(height * 0.2)  # Middle 20% of the image
    middle_roi = binary[height - bottom_height - middle_height:height - bottom_height, :]
    
    # Find contours in both ROIs
    bottom_contours, _ = cv2.findContours(bottom_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    middle_contours, _ = cv2.findContours(middle_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process bottom ROI (primary detection)
    if bottom_contours:
        # Find the largest contour
        largest_contour = max(bottom_contours, key=cv2.contourArea)
        
        # Check if contour is large enough
        area = cv2.contourArea(largest_contour)
        if area >= MIN_CONTOUR_AREA:
            # Get the center of the contour
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                # Calculate confidence based on size
                confidence = min(area / (width * bottom_height * 0.1), 1.0)
                return cx, bottom_roi, confidence
    
    # If no line found in bottom ROI, check middle ROI
    if middle_contours:
        largest_contour = max(middle_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area >= MIN_CONTOUR_AREA:
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                confidence = min(area / (width * middle_height * 0.1), 1.0) * 0.8  # Lower confidence
                return cx, middle_roi, confidence
    
    # No line detected
    return None, None, 0.0

def detect_obstacle(frame):
    """Detect obstacles in the path"""
    if not OBJECT_DETECTION_ENABLED:
        return False, None
    
    # Only use top 60% of the image for obstacle detection
    height, width = frame.shape[:2]
    detect_height = int(height * 0.6)
    obstacle_roi = frame[0:detect_height, :]
    
    # Convert to grayscale
    gray = cv2.cvtColor(obstacle_roi, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find dark objects
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Clean up with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by size
    large_contours = [c for c in contours if cv2.contourArea(c) > OBJECT_AREA_THRESHOLD]
    
    if not large_contours:
        return False, None
    
    # Get the largest contour
    largest_contour = max(large_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Check if in the path (near center and in lower half of detection zone)
    center_x = width // 2
    object_center_x = x + w // 2
    
    if abs(object_center_x - center_x) < width * 0.3 and y + h > detect_height * 0.5:
        return True, (x, y, w, h)
    
    return False, None

# -----------------------------------------------------------------------------
# --- Movement Control ---
# -----------------------------------------------------------------------------
def get_turn_command(steering):
    """Convert steering value to turn command"""
    if abs(steering) < STEERING_DEADZONE:
        return COMMANDS['FORWARD']
    elif steering < 0:  # Negative steering turns right
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
    cv2.putText(frame, "Line Detection", (5, bottom_top + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Middle ROI (corner detection)
    middle_top = bottom_top - middle_height
    cv2.rectangle(frame, (0, middle_top), (width, bottom_top), (0, 200, 255), 2)
    
    # Draw object detection area if enabled
    if OBJECT_DETECTION_ENABLED:
        detect_height = int(height * 0.6)
        cv2.rectangle(frame, (0, 0), (width, detect_height), (255, 100, 100), 1)
        cv2.putText(frame, "Object Detection", (5, 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
    
    # Draw detected obstacle
    if obstacle_detected and object_detected_area:
        x, y, w, h = object_detected_area
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "OBSTACLE", (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw center line
    center_x = width // 2
    cv2.line(frame, (center_x, 0), (center_x, height), (200, 200, 200), 1)
    
    # Draw detected line position
    if line_x is not None and not obstacle_detected:
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
    if obstacle_detected:
        status_color = (0, 100, 255)  # Orange for obstacle
        
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

# Simple HTML template for web interface
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>Line Follower Robot</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial; background: #222; color: white; text-align: center; }
        .container { margin: 20px auto; max-width: 800px; }
        h1 { color: #00ffff; }
        .video-feed { width: 100%; border: 2px solid #444; border-radius: 5px; }
        .status { margin-top: 20px; padding: 10px; background: #333; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Line Follower Robot with Obstacle Avoidance</h1>
        <img src="{{ url_for('video_feed') }}" class="video-feed">
        <div class="status">
            <h2>Robot Status: <span id="status">Unknown</span></h2>
            <p>Command: <span id="command">None</span></p>
        </div>
    </div>
    
    <script>
        // Update status periodically
        setInterval(function() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status').textContent = data.status;
                    document.getElementById('command').textContent = data.command;
                });
        }, 500);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return {
        'status': robot_status,
        'command': turn_command,
        'offset': line_offset,
        'fps': current_fps,
        'confidence': confidence,
        'obstacle': obstacle_detected
    }

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
        
        ret, jpeg = cv2.imencode('.jpg', frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(1/30)  # Limit to 30 FPS for web streaming

def run_flask_server():
    logger.info("🌐 Starting web server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)

# -----------------------------------------------------------------------------
# --- Main Application ---
# -----------------------------------------------------------------------------
def main():
    global output_frame, line_offset, steering_value, turn_command, robot_status
    global line_detected, current_fps, confidence, esp_connection
    global obstacle_detected, obstacle_avoid_counter, object_detected_area
    
    logger.info("🚀 Starting Line Follower Robot with Obstacle Avoidance")
    robot_status = "Starting camera"
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("❌ Failed to open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    
    # Get actual camera resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"📷 Camera initialized: {actual_width}x{actual_height}")
    
    # Initialize PID controller
    pid = SimplePID(KP, KI, KD, MAX_INTEGRAL)
    
    # Connect to ESP32
    robot_status = "Connecting to ESP32"
    esp_connection = ESP32Connection(ESP32_IP, ESP32_PORT)
    
    # Start web interface in a separate thread
    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()
    
    # FPS calculation
    fps_history = deque(maxlen=10)
    search_counter = 0
    
    # History for smoothing
    offset_history = deque(maxlen=3)
    steering_history = deque(maxlen=2)
    last_known_good_offset = 0.0
    
    logger.info("🤖 Robot ready! Starting line detection...")
    robot_status = "Ready"
    
    try:
        while True:
            # Measure processing time for FPS calculation
            start_time = time.time()
            
            # Capture frame from camera
            ret, frame = cap.read()
            if not ret:
                logger.warning("⚠️ Failed to capture frame")
                robot_status = "Camera error"
                time.sleep(0.1)
                continue
            
            # Create a copy for visualization
            display_frame = frame.copy()
            
            # First check for obstacles
            if OBJECT_DETECTION_ENABLED:
                obstacle_detected_now, obstacle_area = detect_obstacle(frame)
                
                # Handle obstacle detection state
                if obstacle_detected_now:
                    obstacle_detected = True
                    obstacle_avoid_counter = OBJECT_AVOIDANCE_DURATION
                    object_detected_area = obstacle_area
                    robot_status = "Obstacle detected! Avoiding..."
                    logger.info("🚧 Obstacle detected! Starting avoidance")
                elif obstacle_detected:
                    # Continue avoidance for a set duration
                    obstacle_avoid_counter -= 1
                    if obstacle_avoid_counter <= 0:
                        obstacle_detected = False
                        object_detected_area = None
                        robot_status = "Returning to line following"
                        logger.info("✅ Obstacle avoidance completed")
                    else:
                        robot_status = f"Avoiding obstacle ({obstacle_avoid_counter})"
            
            # Process image to find line if not in obstacle avoidance mode
            if not obstacle_detected:
                line_x, roi, detection_confidence = detect_line(frame)
                confidence = detection_confidence
                
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
                        robot_status = "Taking corner"
                        
                        # For sharp corners, exaggerate the steering
                        if abs(line_offset) > 0.5:
                            line_offset *= 1.5  # Increase turning response
                    else:
                        robot_status = f"Following line (C:{confidence:.2f})"
                    
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
                    
                    logger.debug(f"🎯 Line at x={line_x}, offset={line_offset:.2f}, steering={steering_value:.2f}")
                    
                else:
                    # Line not detected - search mode
                    line_detected = False
                    search_counter += 1
                    
                    if search_counter < 5:
                        # Keep last command briefly (helps with short line gaps)
                        robot_status = "Searching for line (brief)"
                    elif search_counter < 15:
                        # Try turning based on last known position
                        if last_known_good_offset < 0:
                            turn_command = COMMANDS['RIGHT']
                            robot_status = "Searching right (last seen left)"
                        else:
                            turn_command = COMMANDS['LEFT']
                            robot_status = "Searching left (last seen right)"
                    elif search_counter < 30:
                        # Switch direction
                        if turn_command == COMMANDS['LEFT']:
                            turn_command = COMMANDS['RIGHT']
                        else:
                            turn_command = COMMANDS['LEFT']
                        robot_status = f"Searching opposite ({search_counter})"
                    elif search_counter < 45:
                        # Try moving forward a bit
                        turn_command = COMMANDS['FORWARD']
                        robot_status = "Moving forward to find line"
                    else:
                        # Stop and reset if line completely lost
                        turn_command = COMMANDS['STOP']
                        robot_status = "Line lost - stopped"
                        
                        # Reset search after a pause
                        if search_counter > 60:
                            search_counter = 0
                            last_known_good_offset = 0.0
                            pid.reset()
                            offset_history.clear()
                            steering_history.clear()
            
            else:
                # Object avoidance mode
                if object_detected_area:
                    x, y, w, h = object_detected_area
                    center_x = frame.shape[1] // 2
                    object_center_x = x + w // 2
                    
                    # Determine which way to turn based on obstacle position
                    if obstacle_avoid_counter > OBJECT_AVOIDANCE_DURATION - 5:
                        # First phase: Stop briefly
                        turn_command = COMMANDS['STOP']
                        robot_status = "Obstacle detected - stopping"
                    elif obstacle_avoid_counter > OBJECT_AVOIDANCE_DURATION // 2:
                        # Second phase: Turn away from obstacle
                        if object_center_x < center_x:
                            # Obstacle is on the left, turn right
                            turn_command = COMMANDS['RIGHT']
                            robot_status = "Avoiding obstacle - turning right"
                        else:
                            # Obstacle is on the right, turn left
                            turn_command = COMMANDS['LEFT']
                            robot_status = "Avoiding obstacle - turning left"
                    else:
                        # Third phase: Move forward to pass the obstacle
                        turn_command = COMMANDS['FORWARD']
                        robot_status = "Passing obstacle"
                else:
                    # Fallback: Use default avoidance direction
                    turn_command = COMMANDS[OBJECT_AVOIDANCE_DIRECTION]
                    robot_status = "Avoiding obstacle - default direction"
            
            # Send command to ESP32
            if esp_connection:
                esp_connection.send_command(turn_command)
            
            # Draw debug information
            draw_debug_info(display_frame, line_x, roi, confidence)
            
            # Update output frame for web interface
            with frame_lock:
                output_frame = display_frame.copy()
            
            # Calculate FPS
            processing_time = time.time() - start_time
            fps_history.append(1.0 / max(processing_time, 0.001))
            current_fps = sum(fps_history) / len(fps_history)
            
            # Log status periodically
            if len(fps_history) % 30 == 0:
                obstacle_status = "OBSTACLE DETECTED" if obstacle_detected else "No obstacles"
                logger.info(f"📊 Status: {robot_status} | {obstacle_status} | FPS: {current_fps:.1f}")
            
    except KeyboardInterrupt:
        logger.info("🛑 Stopping robot (Ctrl+C pressed)")
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
    finally:
        # Clean up
        if esp_connection:
            esp_connection.send_command(COMMANDS['STOP'])
            esp_connection.close()
        
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        
        logger.info("✅ Robot stopped")

if __name__ == "__main__":
    main()