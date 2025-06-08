#!/usr/bin/env python3

import cv2
import numpy as np
import socket
import time
import logging
import threading
from collections import deque
from flask import Flask, Response, render_template_string, jsonify
import queue

# Try to import text-to-speech
try:
    import pyttsx3
    TTS_AVAILABLE = True
    print("Text-to-speech available")
except ImportError:
    TTS_AVAILABLE = False
    print("Text-to-speech not available. Install with: pip install pyttsx3")

# ESP32 Connection Settings
ESP32_IP = '192.168.2.21'
ESP32_PORT = 1234

# Camera Settings
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
CAMERA_FPS = 10

# Line Detection Settings - Adaptive for lighting
LINE_THRESHOLD_LOW = 40    # Lower threshold for bright conditions
LINE_THRESHOLD_HIGH = 80   # Higher threshold for dark conditions
BLUR_SIZE = 7
MIN_LINE_AREA = 150
ADAPTIVE_THRESHOLD = True  # Use adaptive thresholding for better lighting handling

# Object Detection Settings - Simple OpenCV based
OBJECT_THRESHOLD = 100     # Threshold for object detection
MIN_OBJECT_AREA = 1000     # Minimum area to be considered an object
OBJECT_MIN_WIDTH = 60      # Minimum object width in pixels
OBJECT_MIN_HEIGHT = 40     # Minimum object height in pixels

# Avoidance Settings - Simple 3-step process
AVOIDANCE_TURN_FRAMES = 8      # Frames to turn away from object
AVOIDANCE_FORWARD_FRAMES = 15   # Frames to go forward past object
AVOIDANCE_RETURN_FRAMES = 12    # Frames to turn back to line

# Speech Settings
SPEECH_ENABLED = True          # Enable/disable speech announcements
SPEECH_RATE = 150             # Speech rate (words per minute)
SPEECH_VOLUME = 0.8           # Speech volume (0.0 to 1.0)
ANNOUNCE_INTERVAL = 3.0       # Minimum seconds between similar announcements

# PID Settings - Simple and stable
KP = 0.4
KI = 0.01
KD = 0.2
MAX_STEERING = 1.0

# Robot Commands
FORWARD = 'FORWARD'
LEFT = 'LEFT'
RIGHT = 'RIGHT'
STOP = 'STOP'

# Global Variables
current_frame = None
frame_lock = threading.Lock()
robot_status = "Starting"
line_offset = 0.0
line_detected = False
object_detected = False
avoidance_phase = 'none'  # 'none', 'turning', 'forward', 'returning'
avoidance_counter = 0
esp_connected = False
current_fps = 0.0

# Speech system variables
speech_queue = queue.Queue()
speech_engine = None
last_announcement = {}  # Track last announcement times
speech_thread = None

# Logging setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger("OpenCVLineFollower")

class SpeechManager:
    def __init__(self):
        self.engine = None
        self.speech_queue = queue.Queue()
        self.running = False
        self.last_announcements = {}
        
        if TTS_AVAILABLE and SPEECH_ENABLED:
            try:
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
                
                logger.info("Speech system initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize speech system: {e}")
                self.engine = None
    
    def start(self):
        if self.engine:
            self.running = True
            self.thread = threading.Thread(target=self._speech_worker, daemon=True)
            self.thread.start()
            logger.info("Speech system started")
    
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
    
    def _speech_worker(self):
        """Background thread to handle speech synthesis"""
        while self.running:
            try:
                # Get message from queue with timeout
                message = self.speech_queue.get(timeout=1.0)
                if message and self.engine:
                    self.engine.say(message)
                    self.engine.runAndWait()
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Speech error: {e}")

# Initialize speech manager
speech_manager = SpeechManager()

class SimplePID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0.0
        self.integral = 0.0
        
    def calculate(self, error):
        self.integral += error
        self.integral = max(-5.0, min(5.0, self.integral))  # Limit integral windup
        
        derivative = error - self.previous_error
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        self.previous_error = error
        return max(-MAX_STEERING, min(MAX_STEERING, output))
    
    def reset(self):
        self.previous_error = 0.0
        self.integral = 0.0

class ESP32Controller:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.socket = None
        self.last_command = None
        
    def connect(self):
        global esp_connected
        try:
            if self.socket:
                self.socket.close()
            self.socket = socket.create_connection((self.ip, self.port), timeout=2)
            esp_connected = True
            logger.info(f"Connected to ESP32 at {self.ip}:{self.port}")
            return True
        except Exception as e:
            esp_connected = False
            logger.error(f"ESP32 connection failed: {e}")
            return False
    
    def send_command(self, command):
        if not self.socket and not self.connect():
            return False
        
        try:
            if command != self.last_command:
                self.socket.sendall(f"{command}\n".encode())
                self.last_command = command
                logger.info(f"Sent: {command}")
            return True
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            esp_connected = False
            self.socket = None
            return False
    
    def close(self):
        if self.socket:
            try:
                self.send_command(STOP)
                time.sleep(0.1)
                self.socket.close()
            except:
                pass
            self.socket = None

def detect_line_adaptive(frame):
    """Detect line using adaptive thresholding for better lighting handling"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)
    
    # Calculate average brightness to adapt threshold
    avg_brightness = np.mean(blurred)
    
    if ADAPTIVE_THRESHOLD:
        # Use adaptive threshold based on local area
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY_INV, 21, 10)
    else:
        # Use dynamic threshold based on brightness
        if avg_brightness > 120:  # Bright conditions
            threshold = LINE_THRESHOLD_LOW
        elif avg_brightness < 60:  # Dark conditions  
            threshold = LINE_THRESHOLD_HIGH
        else:  # Normal conditions
            threshold = (LINE_THRESHOLD_LOW + LINE_THRESHOLD_HIGH) // 2
        
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
    
    return binary, avg_brightness

def find_line_center(binary_frame):
    """Find the center of the line in the bottom portion of the frame"""
    height, width = binary_frame.shape
    
    # Focus on the bottom 30% of the frame for line detection
    roi_height = int(height * 0.3)
    roi = binary_frame[height - roi_height:height, :]
    
    # Find contours
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, 0.0
    
    # Find the largest contour (should be the line)
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    if area < MIN_LINE_AREA:
        return None, 0.0
    
    # Calculate the center of the contour
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None, 0.0
    
    cx = int(M["m10"] / M["m00"])
    
    # Calculate confidence based on area
    max_possible_area = roi_height * width * 0.1  # 10% of ROI area
    confidence = min(area / max_possible_area, 1.0)
    
    return cx, confidence

def detect_objects_simple(frame, binary_frame):
    """Simple object detection using OpenCV"""
    height, width = frame.shape[:2]
    
    # Focus on the top 60% of the frame for object detection
    roi_height = int(height * 0.6)
    roi = binary_frame[0:roi_height, :]
    
    # Invert the binary image for object detection (objects are usually lighter)
    roi_inverted = cv2.bitwise_not(roi)
    
    # Find contours
    contours, _ = cv2.findContours(roi_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_OBJECT_AREA:
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by size
        if w < OBJECT_MIN_WIDTH or h < OBJECT_MIN_HEIGHT:
            continue
        
        # Calculate position relative to center (-1 to 1)
        center_x = x + w // 2
        relative_pos = (center_x - width // 2) / (width // 2)
        
        # Only consider objects somewhat near the center path
        if abs(relative_pos) > 0.6:
            continue
        
        objects.append({
            'bbox': (x, y, w, h),
            'position': relative_pos,
            'area': area
        })
    
    return objects

def simple_avoidance_fsm(objects_detected, line_found):
    """Simple 3-step avoidance: Turn Right -> Go Forward -> Turn Left back to line"""
    global avoidance_phase, avoidance_counter, robot_status
    
    # Start avoidance if object detected and not already avoiding
    if objects_detected and avoidance_phase == 'none':
        avoidance_phase = 'turning'
        avoidance_counter = AVOIDANCE_TURN_FRAMES
        robot_status = "AVOIDING: Turning right away from object"
        logger.info("OBJECT DETECTED - Starting avoidance sequence")
        speech_manager.announce("Object detected! Starting avoidance maneuver.", "obstacle", force=True)
        return RIGHT
    
    # Execute avoidance sequence
    if avoidance_phase == 'turning':
        avoidance_counter -= 1
        if avoidance_counter > 0:
            robot_status = f"AVOIDING: Turning right ({avoidance_counter} frames left)"
            return RIGHT
        else:
            avoidance_phase = 'forward'
            avoidance_counter = AVOIDANCE_FORWARD_FRAMES
            robot_status = "AVOIDING: Going forward past object"
            speech_manager.announce("Moving forward to clear obstacle.", "avoidance")
            return FORWARD
    
    elif avoidance_phase == 'forward':
        avoidance_counter -= 1
        if avoidance_counter > 0:
            robot_status = f"AVOIDING: Going forward ({avoidance_counter} frames left)"
            return FORWARD
        else:
            avoidance_phase = 'returning'
            avoidance_counter = AVOIDANCE_RETURN_FRAMES
            robot_status = "AVOIDING: Turning left back to line"
            speech_manager.announce("Turning back to find the line.", "returning")
            return LEFT
    
    elif avoidance_phase == 'returning':
        # Check if we found the line early
        if line_found:
            avoidance_phase = 'none'
            robot_status = "Line found! Avoidance complete"
            logger.info("Avoidance complete - line reacquired")
            speech_manager.announce("Line reacquired! Resuming line following.", "success", force=True)
            return FORWARD
        
        avoidance_counter -= 1
        if avoidance_counter > 0:
            robot_status = f"AVOIDING: Turning left ({avoidance_counter} frames left)"
            return LEFT
        else:
            avoidance_phase = 'none'
            robot_status = "Avoidance complete - searching for line"
            logger.info("Avoidance sequence complete")
            speech_manager.announce("Avoidance maneuver complete.", "complete")
            return FORWARD
    
    return None

def get_robot_command(line_center, frame_width, steering_value, objects_detected):
    """Determine robot command based on line position and objects"""
    global robot_status, line_detected, object_detected
    
    prev_line_detected = line_detected
    line_detected = line_center is not None
    object_detected = len(objects_detected) > 0
    
    # Check avoidance first
    avoidance_command = simple_avoidance_fsm(object_detected, line_detected)
    if avoidance_command:
        return avoidance_command
    
    # Normal line following
    if line_center is not None:
        robot_status = "Following line"
        
        # Announce line detection if we just found it
        if not prev_line_detected:
            speech_manager.announce("Line detected. Following path.", "line_found")
        
        # Simple steering logic with announcements for sharp turns
        if abs(steering_value) < 0.1:
            return FORWARD
        elif steering_value > 0.3:
            if abs(steering_value) > 0.6:
                speech_manager.announce("Sharp left turn ahead.", "sharp_turn")
            return LEFT
        elif steering_value < -0.3:
            if abs(steering_value) > 0.6:
                speech_manager.announce("Sharp right turn ahead.", "sharp_turn")
            return RIGHT
        else:
            return FORWARD
    else:
        robot_status = "Searching for line"
        
        # Announce line lost if we just lost it
        if prev_line_detected:
            speech_manager.announce("Line lost. Searching for path.", "line_lost")
        
        return STOP

def draw_debug_info(frame, line_center, objects, steering_value, brightness):
    """Draw debug information on the frame"""
    height, width = frame.shape[:2]
    
    # Draw center line
    cv2.line(frame, (width//2, 0), (width//2, height), (255, 255, 255), 1)
    
    # Draw line detection area
    roi_start = int(height * 0.7)
    cv2.rectangle(frame, (0, roi_start), (width, height), (0, 255, 0), 2)
    cv2.putText(frame, "LINE DETECTION", (10, roi_start + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw object detection area
    roi_end = int(height * 0.6)
    cv2.rectangle(frame, (0, 0), (width, roi_end), (0, 0, 255), 2)
    cv2.putText(frame, "OBJECT DETECTION", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Draw detected line
    if line_center is not None:
        line_y = int(height * 0.85)
        cv2.circle(frame, (line_center, line_y), 10, (255, 0, 255), -1)
        cv2.line(frame, (width//2, line_y), (line_center, line_y), (255, 0, 255), 3)
        
        # Show offset
        offset_text = f"Offset: {line_offset:.2f}"
        cv2.putText(frame, offset_text, (line_center - 50, line_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    
    # Draw detected objects
    for obj in objects:
        x, y, w, h = obj['bbox']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, f"OBJECT", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Draw status information
    info_y = 50
    cv2.putText(frame, f"Status: {robot_status}", (10, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Steering: {steering_value:.2f}", (10, info_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Brightness: {brightness:.0f}", (10, info_y + 45),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, info_y + 65),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw avoidance status
    if avoidance_phase != 'none':
        cv2.putText(frame, f"AVOIDANCE: {avoidance_phase.upper()}", 
                   (width//2 - 100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

# Flask Web Interface
app = Flask(__name__)

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>OpenCV Line Follower</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
            .container { max-width: 1000px; margin: 0 auto; }
            .video-container { text-align: center; margin: 20px 0; }
            img { border: 2px solid #333; border-radius: 10px; }
            .status { background: white; padding: 20px; border-radius: 10px; margin: 10px 0; }
            .status h3 { margin: 0 0 10px 0; color: #333; }
            .info { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
            .stat { background: #e0e0e0; padding: 10px; border-radius: 5px; text-align: center; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>OpenCV Line Follower Robot</h1>
            <div class="video-container">
                <img src="/video_feed" alt="Robot Camera Feed" style="max-width: 100%; height: auto;">
            </div>
            <div class="status">
                <h3>Robot Status</h3>
                <div class="info">
                    <div class="stat">
                        <strong>Status:</strong><br>
                        <span id="status">Loading...</span>
                    </div>
                    <div class="stat">
                        <strong>Line Detected:</strong><br>
                        <span id="line-detected">No</span>
                    </div>
                    <div class="stat">
                        <strong>Object Detected:</strong><br>
                        <span id="object-detected">No</span>
                    </div>
                                         <div class="stat">
                         <strong>ESP32:</strong><br>
                         <span id="esp-status">Disconnected</span>
                     </div>
                     <div class="stat">
                         <strong>Speech:</strong><br>
                         <span id="speech-status">Disabled</span>
                     </div>
                 </div>
             </div>
         </div>
        
        <script>
            function updateStatus() {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('status').textContent = data.status;
                                                 document.getElementById('line-detected').textContent = data.line_detected ? 'Yes' : 'No';
                         document.getElementById('object-detected').textContent = data.object_detected ? 'Yes' : 'No';
                         document.getElementById('esp-status').textContent = data.esp_connected ? 'Connected' : 'Disconnected';
                         document.getElementById('speech-status').textContent = data.speech_enabled ? 'Active' : 'Disabled';
                    })
                    .catch(error => console.error('Error:', error));
            }
            
            setInterval(updateStatus, 1000);
            updateStatus();
        </script>
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        while True:
            with frame_lock:
                if current_frame is not None:
                    frame = current_frame.copy()
                else:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "No Camera Feed", (200, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.1)
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    return jsonify({
        'status': robot_status,
        'line_detected': line_detected,
        'object_detected': object_detected,
        'esp_connected': esp_connected,
        'line_offset': line_offset,
        'fps': current_fps,
        'avoidance_phase': avoidance_phase,
        'speech_enabled': SPEECH_ENABLED and TTS_AVAILABLE and speech_manager.engine is not None
    })

def run_web_server():
    logger.info("Starting web server at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)

def main():
    global current_frame, line_offset, robot_status, current_fps
    
    logger.info("Starting OpenCV Line Follower Robot")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    logger.info(f"Camera initialized: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    
    # Initialize PID controller
    pid = SimplePID(KP, KI, KD)
    
    # Initialize ESP32 controller
    esp32 = ESP32Controller(ESP32_IP, ESP32_PORT)
    esp32.connect()
    
    # Start web server
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()
    
    # Start speech system
    speech_manager.start()
    speech_manager.announce("OpenCV Line Follower Robot initialized and ready!", "startup", force=True)
    
    # FPS calculation
    fps_history = deque(maxlen=10)
    
    logger.info("Robot ready!")
    robot_status = "Ready"
    
    try:
        while True:
            start_time = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                continue
            
            # Detect line with adaptive thresholding
            binary_frame, brightness = detect_line_adaptive(frame)
            line_center, confidence = find_line_center(binary_frame)
            
            # Detect objects
            detected_objects = detect_objects_simple(frame, binary_frame)
            
            # Calculate steering
            if line_center is not None:
                center_x = frame.shape[1] // 2
                line_offset = (line_center - center_x) / center_x
                steering_value = pid.calculate(-line_offset)  # Negative because we want to steer opposite to offset
            else:
                steering_value = 0.0
                line_offset = 0.0
            
            # Get robot command
            command = get_robot_command(line_center, frame.shape[1], steering_value, detected_objects)
            
            # Send command to ESP32
            esp32.send_command(command)
            
            # Draw debug information
            display_frame = frame.copy()
            draw_debug_info(display_frame, line_center, detected_objects, steering_value, brightness)
            
            # Update shared frame for web interface
            with frame_lock:
                current_frame = display_frame
            
            # Calculate FPS
            frame_time = time.time() - start_time
            if frame_time > 0:
                fps_history.append(1.0 / frame_time)
                current_fps = sum(fps_history) / len(fps_history)
            
            # Small delay
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        logger.info("Stopping robot...")
        robot_status = "Stopping"
        speech_manager.announce("Shutting down robot.", "shutdown", force=True)
    
    finally:
        esp32.send_command(STOP)
        esp32.close()
        cap.release()
        speech_manager.stop()
        logger.info("Robot stopped cleanly")

if __name__ == "__main__":
    main() 