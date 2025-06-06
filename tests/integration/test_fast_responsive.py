#!/usr/bin/env python3

import cv2
import numpy as np
import socket
import time
import logging
import threading
from collections import deque
from flask import Flask, Response, jsonify

# -----------------------------------------------------------------------------
# --- FAST & RESPONSIVE CONFIGURATION ---
# -----------------------------------------------------------------------------
ESP32_IP = '192.168.53.117'  # Change this to your ESP32's IP address
ESP32_PORT = 1234

# Optimized for SPEED - smaller resolution for faster processing
CAMERA_WIDTH, CAMERA_HEIGHT = 240, 180  # Reduced from 320x240
CAMERA_FPS = 30  # Higher FPS target

# Fast image processing - simplified parameters
BLACK_THRESHOLD = 65
BLUR_SIZE = 3  # Smaller blur for speed
MIN_CONTOUR_AREA = 80  # Lower threshold for responsiveness

# Simplified zones for speed
LINE_ZONE_HEIGHT = 0.4    # Bottom 40% for line detection
OBSTACLE_ZONE_HEIGHT = 0.6  # Top 60% for obstacles

# Fast obstacle detection - very simple
OBSTACLE_MIN_AREA = 800   # Smaller for faster detection
OBSTACLE_WIDTH_THRESHOLD = 0.2
AVOIDANCE_FRAMES = 6      # Shorter avoidance for responsiveness

# Aggressive PID for fast response
KP = 1.0  # Higher for faster response
KI = 0.01  # Minimal integral
KD = 0.05  # Minimal derivative

# Commands
COMMANDS = {'FORWARD': 'FORWARD', 'LEFT': 'LEFT', 'RIGHT': 'RIGHT', 'STOP': 'STOP', 
           'AVOID_LEFT': 'AVOID_LEFT', 'AVOID_RIGHT': 'AVOID_RIGHT'}

# Global variables
output_frame = None
frame_lock = threading.Lock()
line_offset = 0.0
turn_command = COMMANDS['STOP']
robot_status = "Fast Mode Initializing"
line_detected = False
current_fps = 0.0
confidence = 0.0
esp_connection = None
esp_connected = False

# Fast obstacle tracking
obstacle_detected = False
avoidance_counter = 0
avoidance_direction = None

# Minimal statistics for speed
stats = {'frames': 0, 'obstacles': 0, 'avoidances': 0, 'start_time': time.time()}

# Minimal logging for speed
logging.basicConfig(level=logging.WARNING)  # Reduced logging
logger = logging.getLogger("FastRobot")

# -----------------------------------------------------------------------------
# --- FAST PID CONTROLLER ---
# -----------------------------------------------------------------------------
class FastPID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        
    def calculate(self, error):
        # Simplified PID for speed
        self.integral += error
        self.integral = np.clip(self.integral, -2.0, 2.0)  # Fast clipping
        
        derivative = error - self.prev_error
        self.prev_error = error
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return np.clip(output, -1.0, 1.0)

# -----------------------------------------------------------------------------
# --- FAST ESP32 CONNECTION ---
# -----------------------------------------------------------------------------
class FastESP32:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.socket = None
        self.last_cmd = None
        self.connect()
    
    def connect(self):
        global esp_connected
        try:
            if self.socket:
                self.socket.close()
            self.socket = socket.create_connection((self.ip, self.port), timeout=1)
            self.socket.settimeout(0.1)  # Very fast timeout
            esp_connected = True
            return True
        except:
            esp_connected = False
            self.socket = None
            return False
    
    def send(self, cmd):
        if not self.socket and not self.connect():
            return False
        try:
            if cmd != self.last_cmd:
                self.socket.sendall(f"{cmd}\n".encode())
                self.last_cmd = cmd
            return True
        except:
            self.socket = None
            return False

# -----------------------------------------------------------------------------
# --- LIGHTNING FAST LINE DETECTION ---
# -----------------------------------------------------------------------------
def fast_line_detect(frame):
    """Ultra-fast line detection optimized for speed"""
    height, width = frame.shape[:2]
    
    # Only process bottom zone for speed
    line_start = int(height * (1 - LINE_ZONE_HEIGHT))
    roi = frame[line_start:height, :]
    
    # Super fast processing
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Skip blur for speed, direct threshold
    _, binary = cv2.threshold(gray, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    
    # Fast contour detection
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, 0.0
    
    # Just take the largest contour - no complex analysis
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    
    if area < MIN_CONTOUR_AREA:
        return None, 0.0
    
    # Fast center calculation
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, 0.0
    
    cx = int(M["m10"] / M["m00"])
    confidence = min(area / 1000, 1.0)  # Simple confidence
    
    return cx, confidence

# -----------------------------------------------------------------------------
# --- ULTRA-FAST OBSTACLE DETECTION ---
# -----------------------------------------------------------------------------
def fast_obstacle_detect(frame):
    """Super simple and fast obstacle detection"""
    height, width = frame.shape[:2]
    
    # Only check top zone
    obstacle_end = int(height * OBSTACLE_ZONE_HEIGHT)
    roi = frame[0:obstacle_end, :]
    
    # Super simple processing
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Simple threshold - no fancy filtering
    _, binary = cv2.threshold(gray, BLACK_THRESHOLD + 30, 255, cv2.THRESH_BINARY)
    
    # Fast contour check
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    obstacles = []
    center_x = width // 2
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > OBSTACLE_MIN_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Simple center check
            obj_center = x + w // 2
            distance_from_center = abs(obj_center - center_x) / center_x
            
            # Simple filtering
            if (w / width > OBSTACLE_WIDTH_THRESHOLD and 
                distance_from_center < 0.3):
                obstacles.append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'center': obj_center,
                    'threat': distance_from_center < 0.2
                })
    
    return obstacles

# -----------------------------------------------------------------------------
# --- FAST MOVEMENT CONTROL ---
# -----------------------------------------------------------------------------
def fast_movement(steering, obstacles):
    """Lightning fast movement decisions"""
    global avoidance_counter, avoidance_direction, stats
    
    # Check for immediate threats
    threats = [obs for obs in obstacles if obs['threat']]
    
    # Start avoidance if threat detected
    if threats and avoidance_counter <= 0:
        threat = threats[0]
        avoidance_counter = AVOIDANCE_FRAMES
        stats['avoidances'] += 1
        
        # Simple direction decision
        if threat['center'] < CAMERA_WIDTH // 2:
            avoidance_direction = 'RIGHT'
            return COMMANDS['AVOID_RIGHT']
        else:
            avoidance_direction = 'LEFT'
            return COMMANDS['AVOID_LEFT']
    
    # Continue avoidance
    if avoidance_counter > 0:
        avoidance_counter -= 1
        return COMMANDS[f'AVOID_{avoidance_direction}'] if avoidance_direction else COMMANDS['FORWARD']
    
    # Fast steering decisions
    if abs(steering) < 0.05:
        return COMMANDS['FORWARD']
    return COMMANDS['LEFT'] if steering > 0 else COMMANDS['RIGHT']

# -----------------------------------------------------------------------------
# --- MINIMAL VISUALIZATION ---
# -----------------------------------------------------------------------------
def fast_draw(frame, line_x=None, obstacles=None):
    """Minimal drawing for maximum speed"""
    height, width = frame.shape[:2]
    
    # Draw center line only
    center_x = width // 2
    cv2.line(frame, (center_x, 0), (center_x, height), (100, 100, 100), 1)
    
    # Draw line detection if found
    if line_x is not None:
        line_y = int(height * 0.8)
        cv2.circle(frame, (line_x, line_y), 4, (0, 255, 0), -1)
        cv2.line(frame, (center_x, line_y), (line_x, line_y), (255, 0, 255), 2)
    
    # Draw obstacles with minimal info
    if obstacles:
        for obs in obstacles:
            color = (0, 0, 255) if obs['threat'] else (0, 255, 255)
            cv2.rectangle(frame, (obs['x'], obs['y']), 
                         (obs['x'] + obs['w'], obs['y'] + obs['h']), color, 1)
    
    # Minimal status
    status_color = (0, 255, 0) if line_detected else (0, 0, 255)
    cv2.putText(frame, f"FAST: {robot_status}", (5, 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
    cv2.putText(frame, f"FPS: {current_fps:.1f}", (5, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"CMD: {turn_command}", (5, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

# -----------------------------------------------------------------------------
# --- MINIMAL FLASK INTERFACE ---
# -----------------------------------------------------------------------------
app = Flask(__name__)

@app.route('/')
def index():
    return """<!DOCTYPE html>
<html>
<head><title>FAST Robot</title>
<style>
body{font-family:Arial;background:#000;color:#0f0;padding:20px;text-align:center}
.video{border:2px solid #0f0;border-radius:10px}
.stats{margin:20px;font-size:18px}
.fast{color:#ff0;font-weight:bold}
</style>
<script>
setInterval(()=>{
fetch('/api/status').then(r=>r.json()).then(d=>{
document.getElementById('status').textContent=d.status;
document.getElementById('fps').textContent=d.fps.toFixed(1);
document.getElementById('cmd').textContent=d.command;
});
},200);
</script>
</head>
<body>
<h1>⚡ <span class="fast">FAST</span> Line Follower Robot</h1>
<img src="/video_feed" class="video" width="480">
<div class="stats">
Status: <span id="status">Loading...</span><br>
FPS: <span id="fps">0</span><br>
Command: <span id="cmd">STOP</span>
</div>
</body>
</html>"""

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    return jsonify({
        'status': robot_status,
        'command': turn_command,
        'offset': line_offset,
        'fps': current_fps,
        'esp_connected': esp_connected
    })

def generate_frames():
    global output_frame, frame_lock
    while True:
        with frame_lock:
            if output_frame is not None:
                frame = output_frame.copy()
            else:
                frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(1/60)  # Fast web streaming

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)

# -----------------------------------------------------------------------------
# --- MAIN FAST LOOP ---
# -----------------------------------------------------------------------------
def main():
    global output_frame, line_offset, turn_command, robot_status
    global line_detected, current_fps, confidence, esp_connection, stats
    
    print("⚡ Starting LIGHTNING FAST Line Follower Robot")
    robot_status = "FAST MODE READY"
    
    # Initialize camera with optimized settings
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera failed")
        return
    
    # Optimize camera for SPEED
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
    cap.set(cv2.CAP_PROP_EXPOSURE, -7)  # Fast exposure
    
    print(f"📷 Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
    
    # Fast PID controller
    pid = FastPID(KP, KI, KD)
    
    # ESP32 connection
    esp_connection = FastESP32(ESP32_IP, ESP32_PORT)
    
    # Start minimal web interface
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("🌐 Fast dashboard: http://localhost:5000")
    
    # Speed tracking
    fps_times = deque(maxlen=5)  # Small buffer for speed
    last_log = time.time()
    
    print("⚡ FAST ROBOT READY - MAXIMUM SPEED MODE!")
    robot_status = "FAST - Line Following"
    
    try:
        frame_count = 0
        while True:
            loop_start = time.time()
            
            # Fast frame capture
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            stats['frames'] = frame_count
            
            # Lightning fast line detection
            line_x, conf = fast_line_detect(frame)
            confidence = conf
            
            if line_x is not None and conf > 0.15:  # Lower threshold for speed
                line_detected = True
                
                # Fast offset calculation
                center_x = CAMERA_WIDTH / 2
                line_offset = (line_x - center_x) / center_x
                
                # Fast steering
                steering = pid.calculate(-line_offset)
                
                robot_status = "FAST - Following"
            else:
                line_detected = False
                steering = 0
                robot_status = "FAST - Searching"
            
            # Ultra-fast obstacle detection
            obstacles = fast_obstacle_detect(frame)
            if obstacles:
                stats['obstacles'] += 1
            
            # Lightning fast movement decision
            turn_command = fast_movement(steering, obstacles)
            
            # Send command immediately
            if esp_connection:
                esp_connection.send(turn_command)
            
            # Minimal drawing
            fast_draw(frame, line_x, obstacles)
            
            # Update output
            with frame_lock:
                output_frame = frame
            
            # Fast FPS calculation
            loop_time = time.time() - loop_start
            if loop_time > 0:
                fps_times.append(1.0 / loop_time)
                current_fps = sum(fps_times) / len(fps_times)
            
            # Minimal logging
            if time.time() - last_log > 2:  # Every 2 seconds
                print(f"⚡ FPS: {current_fps:.1f} | Status: {robot_status} | Frames: {frame_count}")
                last_log = time.time()
            
            # NO DELAYS - maximum speed!
            
    except KeyboardInterrupt:
        print("⚡ Fast robot stopped")
    finally:
        if esp_connection:
            esp_connection.send(COMMANDS['STOP'])
        cap.release()
        print("⚡ Cleanup complete")

if __name__ == "__main__":
    main() 