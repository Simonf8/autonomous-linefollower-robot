#!/usr/bin/env python3

import cv2
import numpy as np
import socket
import time
import logging
import sys
import math
from collections import deque
from flask import Flask, Response, render_template_string
import threading
from scipy import ndimage
from sklearn.cluster import DBSCAN






# -----------------------------------------------------------------------------
# --- CONFIGURATION FOR BLACK LINE FOLLOWING ---
# -----------------------------------------------------------------------------
# ESP32 Configuration - UPDATE THIS TO MATCH YOUR ESP32's IP
ESP32_IP = '192.168.53.117'  # Change this to your ESP32's actual IP address
ESP32_PORT = 1234
REQUESTED_CAM_W, REQUESTED_CAM_H = 320, 240
REQUESTED_CAM_FPS = 30  # Increased from 20 to 30 for faster decisions

# Enhanced Canny parameters
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_THRESHOLD = 25
HOUGH_MIN_LINE_LENGTH = 20
HOUGH_MAX_LINE_GAP = 15

# Improved thresholding
BLACK_THRESHOLD = 60  # Reduced from 80 to 60 - more strict for true black
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C = 8  # Increased from 5 to 8 - more strict adaptive thresholding

# Grid-based analysis parameters
GRID_ROWS = 6
GRID_COLS = 8
MIN_BLACK_PIXELS_RATIO = 0.35

# Optimized ROI zones - focus more on immediate area ahead
ROI_ZONES = [
    {"height_ratio": 0.30, "top_offset": 0.65, "weight": 4.0},  # Primary zone - closer and larger
    {"height_ratio": 0.25, "top_offset": 0.45, "weight": 2.5},  # Secondary zone
    {"height_ratio": 0.20, "top_offset": 0.25, "weight": 1.0},  # Tertiary zone
]

# Improved PID parameters for better turning response
PID_KP = 1.2   # Increased for more responsive steering
PID_KI = 0.02  # Slightly increased for steady-state accuracy
PID_KD = 0.3   # Increased damping
PID_INT_MAX = 0.4  # Increased limit

SPEEDS = {'FAST':'F', 'NORMAL':'N', 'SLOW':'S', 'TURN':'T', 'STOP':'H'}

STEERING_DEADZONE = 0.03   # More sensitive for quicker turn response
OFFSET_THRESHOLDS = {'PERFECT':0.08, 'GOOD':0.20, 'MODERATE':0.35, 'LARGE':0.50}  # More forgiving thresholds

# Median filter parameters
MEDIAN_FILTER_SIZE = 5
GAUSSIAN_BLUR_SIZE = 7

# -----------------------------------------------------------------------------
# --- Global Variables ---
# -----------------------------------------------------------------------------
CAM_W, CAM_H = REQUESTED_CAM_W, REQUESTED_CAM_H
output_frame_flask = None
frame_lock = threading.Lock()
current_line_angle, current_line_offset, current_steering = 0.0, 0.0, 0.0
current_speed_cmd, current_turn_cmd = SPEEDS['STOP'], "FORWARD"
lines_detected, robot_status, fps_current, confidence_score = 0, "Initializing", 0.0, 0.0
search_memory = deque(maxlen=50)
esp_comm = None
line_history = deque(maxlen=5)  # Reduced from 10 to 5 for faster response
# Shorter history for faster response
offset_history = deque(maxlen=3)  # Reduced from 7 to 3 for quicker decisions

# -----------------------------------------------------------------------------
# --- Logging Setup ---
# -----------------------------------------------------------------------------
# Set to logging.DEBUG for more verbose output if needed for deep diagnosis
logging.basicConfig(level=logging.INFO, format='🤖 [%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("BlackLineFollower")

# -----------------------------------------------------------------------------
# --- Enhanced PID Controller ---
# -----------------------------------------------------------------------------
class EnhancedPIDController:
    def __init__(self, kp, ki, kd, integral_max):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_max = integral_max
        self.prev_error, self.integral = 0.0, 0.0
        self.last_time = time.time()
        self.error_history = deque(maxlen=10)
        self.derivative_filter = deque(maxlen=3)
        
    def calculate(self, error, dt=None):
        current_time = time.time()
        dt = max(current_time - self.last_time if dt is None else dt, 1e-3)
        
        # Add error to history for median filtering
        self.error_history.append(error)
        
        # Use median filter for error smoothing
        if len(self.error_history) >= 5:
            smoothed_error = np.median(list(self.error_history)[-5:])
        else:
            smoothed_error = error
            
        # Integral with windup protection
        self.integral += smoothed_error * dt
        self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)
        
        # Derivative with filtering
        derivative = (smoothed_error - self.prev_error) / dt
        self.derivative_filter.append(derivative)
        filtered_derivative = np.mean(list(self.derivative_filter)) if len(self.derivative_filter) > 1 else derivative
        
        # PID output
        output = (self.kp * smoothed_error) + (self.ki * self.integral) + (self.kd * filtered_derivative)
        
        self.prev_error, self.last_time = smoothed_error, current_time
        return np.clip(output, -1.0, 1.0)
    
    def reset(self):
        self.prev_error, self.integral = 0.0, 0.0
        self.error_history.clear()
        self.derivative_filter.clear()
        self.last_time = time.time()
        logger.info("🔄 Enhanced PID Controller Reset")

# -----------------------------------------------------------------------------
# --- ESP32 Communication ---
# -----------------------------------------------------------------------------
class ESP32Communicator:
    def __init__(self, ip, port):
        self.ip, self.port = ip, port
        self.sock = None
        self.last_command = None
        self.connect_attempts, self.max_connect_attempts = 0, 5
        self.last_heartbeat = time.time()
        self.connect()
    
    def connect(self):
        if self.sock:
            try: 
                self.sock.close()
            except: 
                pass
            self.sock = None
        
        try:
            self.sock = socket.create_connection((self.ip, self.port), timeout=1)  # Reduced timeout
            self.sock.settimeout(0.1)  # Much faster socket timeout for real-time control
            logger.info(f" ESP32 connected: {self.ip}:{self.port}")
            self.connect_attempts = 0
            self.last_command = None
            return True
        except Exception as e:
            logger.error(f" ESP32 connection failed (Attempt {self.connect_attempts+1}): {e}")
            self.sock = None
            self.connect_attempts += 1
            return False
    
    def send_command(self, speed_cmd, turn_cmd):
        if not self.sock:
            if self.connect_attempts >= self.max_connect_attempts:
                if time.time() - self.last_heartbeat > 2:  # Reduced from 5 to 2 seconds
                    self.connect_attempts = 0
                    if not self.connect(): 
                        return False
                    self.last_heartbeat = time.time()
                else: 
                    return False
            elif not self.connect(): 
                return False
        
        try:
            # Convert old format to new simple command format
            if speed_cmd == 'H':  # Stop command
                simple_command = "STOP"
            else:
                # Use the turn command directly for new format
                simple_command = turn_cmd  # 'FORWARD', 'LEFT', or 'RIGHT'
            
            # Send simple command to ESP32
            command = f"{simple_command}\n"
            self.sock.sendall(command.encode())
            self.last_command = command
            # Removed debug logging to reduce overhead
            
            return True
        except socket.timeout:
            logger.error(" ESP32 Send Timeout")
            self.sock = None
            self.connect_attempts = 0
            return False
        except socket.error as e:
            logger.error(f" ESP32 Socket Error: {e}")
            self.sock = None
            self.connect_attempts = 0
            return False
        except Exception as e:
            logger.error(f"ESP32 General Error: {e}")
            self.sock = None
            self.connect_attempts = 0
            return False
    
    def close(self):
        if self.sock:
            try:
                stop_command = "STOP\n"  # Use new simple format
                logger.info(f"Sending STOP command to ESP32: {stop_command.strip()}")
                self.sock.sendall(stop_command.encode())
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error sending stop command: {e}")
            finally:
                try:
                    self.sock.close()
                except:
                    pass
                self.sock = None
                logger.info("🔌 ESP32 socket closed.")

# -----------------------------------------------------------------------------
# --- FAST IMAGE PROCESSING FUNCTIONS ---
# -----------------------------------------------------------------------------
def fast_line_detection(frame):
    """
    Ultra-fast line detection focused on speed over complexity.
    Processes only the bottom portion of the image for maximum speed.
    """
    h, w = frame.shape[:2]
    
    # Focus only on bottom 40% of image for speed
    roi_start = int(h * 0.6)
    roi = frame[roi_start:h, :]
    
    # Convert to grayscale and threshold in one step
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Simple binary threshold - much faster than adaptive
    _, binary = cv2.threshold(gray, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours - fastest method for line center
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, 0.0, []
    
    # Find largest contour (assumed to be the line)
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # Skip tiny contours (noise)
    if area < 150:
        return None, 0.0, []
    
    # Calculate centroid
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None, 0.0, []
    
    cx = int(M["m10"] / M["m00"])
    
    # Calculate normalized offset
    center_x = w // 2
    offset = (cx - center_x) / (w // 2)
    
    # Simple confidence based on contour area
    max_expected_area = w * (h - roi_start) * 0.3  # 30% of ROI
    confidence = min(area / max_expected_area, 1.0)
    
    # Visualization data
    viz_data = [{'contour': largest_contour, 'cx': cx, 'cy': roi_start + 20}]
    
    return offset, confidence, viz_data

# -----------------------------------------------------------------------------
# --- MOVEMENT CONTROL HELPER ---
# -----------------------------------------------------------------------------
def get_turn_command(steering_output):
    if abs(steering_output) < STEERING_DEADZONE: 
        return 'FORWARD'
    elif steering_output < -STEERING_DEADZONE: 
        return 'RIGHT'  # Fixed: negative steering should turn right
    elif steering_output > STEERING_DEADZONE: 
        return 'LEFT'   # Fixed: positive steering should turn left
    else: 
        return 'FORWARD'

# -----------------------------------------------------------------------------
# --- VISUALS & FLASK ---
# -----------------------------------------------------------------------------
def draw_car_arrow(display_frame):
    height, width = display_frame.shape[:2]; arrow_base_y = int(height*0.8); arrow_center_x = width//2
    arrow_color = (0,255,0); arrow_text = "FWD"; arrow_angle_deg = current_steering * 20
    if current_turn_cmd == 'LEFT': arrow_color=(0,255,255); arrow_text=f"LEFT ({current_steering:.2f})"; arrow_angle_deg = -25+(current_steering*30)
    elif current_turn_cmd == 'RIGHT': arrow_color=(255,255,0); arrow_text=f"RIGHT ({current_steering:.2f})"; arrow_angle_deg = 25+(current_steering*30)
    arrow_len=40; end_x=int(arrow_center_x+arrow_len*math.sin(math.radians(arrow_angle_deg))); end_y=int(arrow_base_y-arrow_len*math.cos(math.radians(arrow_angle_deg)))
    cv2.arrowedLine(display_frame, (arrow_center_x,arrow_base_y), (end_x,end_y), arrow_color, 3, tipLength=0.4)
    text_size=cv2.getTextSize(arrow_text, cv2.FONT_HERSHEY_SIMPLEX,0.5,1)[0]; text_x=arrow_center_x-text_size[0]//2
    cv2.putText(display_frame, arrow_text, (text_x, arrow_base_y+20), cv2.FONT_HERSHEY_SIMPLEX,0.5,arrow_color,1)

def draw_line_overlays(display_frame, roi_config_list, viz_data, offset, confidence):
    h, w = display_frame.shape[:2]; center_x = w//2
    
    # Draw simple ROI indicator
    roi_start = int(h * 0.6)
    cv2.rectangle(display_frame, (5, roi_start), (w-5, h-5), (255,100,0), 1)
    cv2.putText(display_frame, "ROI", (10, roi_start + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,100,0), 1)
    
    # Draw detected line contour and center
    if viz_data:
        for data in viz_data:
            contour = data['contour']
            cx, cy = data['cx'], data['cy']
            # Draw contour
            cv2.drawContours(display_frame, [contour], -1, (0,255,255), 2, offset=(0, roi_start))
            # Draw center point
            cv2.circle(display_frame, (cx, cy + roi_start), 8, (255,0,0), -1)
            # Draw line from center to detected center
            cv2.line(display_frame, (center_x, cy + roi_start), (cx, cy + roi_start), (0,255,0), 2)
            # Show offset text
            cv2.putText(display_frame, f"CX:{cx}", (cx-20, cy + roi_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
    
    # Draw center line
    cv2.line(display_frame,(center_x,0),(center_x,h),(200,200,200),1,cv2.LINE_AA)
    
    # Draw offset indicator
    if offset is not None and abs(offset) <= 2.0:  # Validate offset
        target_x = int(center_x + (offset * (w / 4)))
        target_x = max(0, min(target_x, w-1))  # Clamp to screen bounds
        target_y = int(h * 0.8)
        
        cv2.circle(display_frame,(target_x, target_y),10,(0,0,255),-1)
        cv2.line(display_frame,(center_x, target_y),(target_x, target_y),(255,0,255),3)
        # Show offset value
        cv2.putText(display_frame, f"Offset: {offset:.2f}", (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

def draw_status_panel(display_frame):
    h,w=display_frame.shape[:2]; pw,ph=220,155; overlay=display_frame.copy()
    cv2.rectangle(overlay,(5,5),(pw,ph),(0,0,0),-1); cv2.addWeighted(overlay,0.7,display_frame,0.3,0,display_frame)
    cv2.rectangle(display_frame,(5,5),(pw,ph),(0,255,255),1); y,lh=20,16
    stat_clr=(0,255,0) if "Follow" in robot_status else ((255,0,0) if "Error" in robot_status else (255,255,0))
    cv2.putText(display_frame,f"St: {robot_status[:20]}",(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,stat_clr,1); y+=lh
    cv2.putText(display_frame,f"Lines: {lines_detected}",(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,0),1); y+=lh
    cv2.putText(display_frame,f"Offset: {current_line_offset:.2f}",(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1); y+=lh
    cv2.putText(display_frame,f"Steer: {current_steering:.2f}",(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,255),1); y+=lh
    cv2.putText(display_frame,f"Conf: {confidence_score:.2f}",(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1); y+=lh
    smap={v:k for k,v in SPEEDS.items()}; sdisp=smap.get(current_speed_cmd,"?").title(); tdisp="Fwd" if current_turn_cmd=="FORWARD" else("Left" if current_turn_cmd=="LEFT" else "Right")
    cv2.putText(display_frame,f"Cmd: {sdisp}/{tdisp}",(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),1); y+=lh
    cv2.putText(display_frame,f"FPS: {fps_current:.1f}",(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)

app = Flask(__name__)
HTML_TEMPLATE = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>🤖 Black Line Follower Dashboard</title><style>:root{--bg-color:#1a1c20;--card-bg:#25282e;--text-color:#e0e0e0;--primary-accent:#00ffff;--secondary-accent:#ff6b35;--success-green:#4caf50;--warning-yellow:#ffeb3b;--error-red:#f44336;--border-color:#3a3f47}body{margin:0;font-family:'Roboto',Arial,sans-serif;background-color:var(--bg-color);color:var(--text-color);display:flex;flex-direction:column;align-items:center;padding:10px;font-size:14px}.container{display:grid;grid-template-columns:2fr 1fr;gap:20px;width:100%;max-width:1200px}header{grid-column:1 / -1;text-align:center;margin-bottom:10px}header h1{font-size:2em;color:var(--primary-accent);margin:0}#esp-status{font-weight:700;padding:5px 10px;border-radius:5px;display:inline-block;margin-top:5px}.card{background-color:var(--card-bg);border-radius:8px;padding:15px;box-shadow:0 2px 10px rgba(0,0,0,.2);border:1px solid var(--border-color)}.video-feed{width:100%;height:auto;border-radius:6px;display:block;background-color:#000}.status-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}.status-item{padding:10px;border-radius:6px;background-color:#2c3038}.status-item h3{margin:0 0 5px;font-size:.9em;color:var(--primary-accent);text-transform:uppercase}.status-item .value{font-size:1.5em;font-weight:700;color:var(--text-color)}.status-item .unit{font-size:.8em;color:#aaa;margin-left:5px}.connected{color:var(--success-green);background-color:rgba(76,175,80,.1);border:1px solid var(--success-green)}.disconnected{color:var(--error-red);background-color:rgba(244,67,54,.1);border:1px solid var(--error-red)}.progress-bar-container{width:100%;height:8px;background-color:#3a3f47;border-radius:4px;overflow:hidden;margin-top:5px}.progress-bar-fill{height:100%;background-color:var(--success-green);width:0%;transition:width .2s ease-in-out}@media (max-width:768px){.container{grid-template-columns:1fr}header h1{font-size:1.5em}.status-item .value{font-size:1.2em}}</style></head><body><header><h1>Line Follower Control</h1><div id="esp-status" class="disconnected">ESP32: Connecting...</div></header><div class="container"><div class="card video-container"><img src="{{ url_for('video_feed') }}" class="video-feed" alt="Camera Feed"></div><div class="card status-grid"><div class="status-item"><h3>Robot Status</h3><span class="value" id="robot-status">N/A</span></div><div class="status-item"><h3>Lines Found</h3><span class="value" id="lines-detected">0</span></div><div class="status-item"><h3>Confidence</h3><span class="value" id="confidence">0.00</span><div class="progress-bar-container"><div class="progress-bar-fill" id="confidence-bar"></div></div></div><div class="status-item"><h3>Offset</h3><span class="value" id="line-offset">0.000</span></div><div class="status-item"><h3>Steering</h3><span class="value" id="steering">0.000</span></div><div class="status-item"><h3>Speed Cmd</h3><span class="value" id="speed-cmd">N/A</span></div><div class="status-item"><h3>Turn Cmd</h3><span class="value" id="turn-cmd">N/A</span></div><div class="status-item"><h3>FPS</h3><span class="value" id="fps">0.0</span></div></div></div><script>const speedMap={'F':'FAST','N':'NORMAL','S':'SLOW','T':'TURN','H':'STOP'},turnMap={'FORWARD':'FWD','LEFT':'LEFT','RIGHT':'RIGHT'};function updateStatus(){fetch('/status').then(e=>e.json()).then(e=>{document.getElementById('robot-status').textContent=e.robot_status||'N/A';document.getElementById('lines-detected').textContent=e.lines_detected||0;const t=e.confidence||0;document.getElementById('confidence').textContent=t.toFixed(2),document.getElementById('confidence-bar').style.width=100*t+'%';document.getElementById('line-offset').textContent=(e.line_offset||0).toFixed(3);document.getElementById('steering').textContent=(e.steering||0).toFixed(3);document.getElementById('speed-cmd').textContent=speedMap[e.speed_cmd]||e.speed_cmd||'N/A';document.getElementById('turn-cmd').textContent=turnMap[e.turn_cmd]||e.turn_cmd||'N/A';document.getElementById('fps').textContent=(e.fps||0).toFixed(1);const n=document.getElementById('esp-status');e.robot_status&&e.robot_status.toLowerCase().includes('error')?(n.className='disconnected',n.textContent='ESP32: Error/Disconnected'):e.esp_connected?(n.className='connected',n.textContent='ESP32: Connected'):(n.className='disconnected',n.textContent='ESP32: Disconnected')}).catch(e=>console.error('Error fetching status:',e))}setInterval(updateStatus,300),updateStatus();</script></body></html>"""
@app.route('/')
def index(): return render_template_string(HTML_TEMPLATE)
@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/status')
def status_endpoint():
    global esp_comm
    return {'line_angle':current_line_angle,'line_offset':current_line_offset,'steering':current_steering,'speed_cmd':current_speed_cmd,'turn_cmd':current_turn_cmd,'lines_detected':lines_detected,'robot_status':robot_status,'fps':fps_current,'confidence':confidence_score,'esp_connected':esp_comm.sock is not None if esp_comm else False}
def generate_frames():
    global output_frame_flask, frame_lock
    while True:
        with frame_lock: frame_to_send = output_frame_flask.copy() if output_frame_flask is not None else np.zeros((CAM_H,CAM_W,3),dtype=np.uint8)
        if output_frame_flask is None: cv2.putText(frame_to_send,"No Feed",(CAM_W//2-50,CAM_H//2),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        ret, jpeg_buffer = cv2.imencode('.jpg', frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if ret: yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpeg_buffer.tobytes()+b'\r\n')
        time.sleep(1/max(REQUESTED_CAM_FPS, 10))
def run_flask():
    logger.info("🌐 Starting Flask server on http://0.0.0.0:5000")
    try: app.run(host='0.0.0.0',port=5000,debug=False,use_reloader=False,threaded=True)
    except Exception as e: logger.error(f"Flask server failed: {e}")

# -----------------------------------------------------------------------------
# --- MAIN APPLICATION ---
# -----------------------------------------------------------------------------
def main():
    global CAM_W, CAM_H, output_frame_flask, frame_lock, current_line_angle, current_line_offset, current_steering
    global current_speed_cmd, current_turn_cmd, lines_detected, robot_status, fps_current, confidence_score
    global search_memory, esp_comm

    logger.info("🚀 Starting Line Following Robot...")
    robot_status = "Init Cam"
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): logger.error("❌ CAM FAILED"); robot_status="Cam Err"; return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,REQUESTED_CAM_W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,REQUESTED_CAM_H); cap.set(cv2.CAP_PROP_FPS,REQUESTED_CAM_FPS)
    CAM_W,CAM_H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"📷 Cam: {CAM_W}x{CAM_H} @ {cap.get(cv2.CAP_PROP_FPS):.1f}FPS")

    robot_status = "Init ESP32"
    pid_controller = EnhancedPIDController(PID_KP,PID_KI,PID_KD,PID_INT_MAX)
    esp_comm = ESP32Communicator(ESP32_IP,ESP32_PORT)

    flask_thread = threading.Thread(target=run_flask,daemon=True); flask_thread.start()
    fps_deque = deque(maxlen=REQUESTED_CAM_FPS); frame_count=0; search_counter=0; last_known_offset=0.0
    logger.info("🤖 Robot READY!")
    logger.info(f"🎛️ Turn Thresholds: DEADZONE={STEERING_DEADZONE}, PID_KP={PID_KP}")
    
    # Send a quick test sequence to verify ESP32 turning
    logger.info("🧪 Testing ESP32 turn commands...")
    if esp_comm:
        test_commands = [('S', 'LEFT'), ('S', 'RIGHT'), ('S', 'FORWARD')]
        for speed, turn in test_commands:
            esp_comm.send_command(speed, turn)
            logger.info(f"📡 Test sent: {turn}")  # Log the actual command being sent
            time.sleep(0.5)
        logger.info("✅ Turn command test complete")

    # Clean startup - removed workaround logging


    try:
        while True:
            loop_start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret: logger.warning("⚠️ No frame"); robot_status="No Frame"; time.sleep(0.1); continue
            if frame.shape[1]!=CAM_W or frame.shape[0]!=CAM_H: frame=cv2.resize(frame,(CAM_W,CAM_H))

            display_frame = frame.copy()
            
            # Use fast line detection
            final_offset, confidence, viz_data = fast_line_detection(frame)
            
            # Set angle to 0 for simple detection (not calculating angle)
            final_angle = 0.0
            lines_detected = 1 if final_offset is not None else 0
            
            # Validate and smooth offset using short history to reduce jitter
            if final_offset is not None:
                # Validate offset is reasonable before adding to history
                if abs(final_offset) <= 5.0 and np.isfinite(final_offset):
                    offset_history.append(final_offset)
                else:
                    # Silently skip invalid offsets
                    final_offset = None
                    
            if offset_history:
                smoothed_offset = float(np.mean(offset_history))
                # Final safety clamp
                smoothed_offset = np.clip(smoothed_offset, -5.0, 5.0)
            else:
                smoothed_offset = 0.0
            current_line_offset = smoothed_offset
            current_line_angle=final_angle if final_angle is not None else 0.0
            lines_detected=len(viz_data) if viz_data else 0
            confidence_score = confidence

            # --- Control Logic ---
            if final_offset is not None and confidence > 0.2:  # Reduced from 0.5 to 0.2 for faster response
                robot_status = f"Follow (C:{confidence:.2f})"
                search_counter=0; search_memory.append(smoothed_offset); last_known_offset=smoothed_offset
                steering_error = -smoothed_offset
                current_steering = pid_controller.calculate(steering_error)
                abs_offset = abs(smoothed_offset)

                # Simplified speed control for faster decisions
                if abs_offset < 0.15:
                    current_speed_cmd = SPEEDS['NORMAL']  # Use NORMAL speed when line is well centered
                else: 
                    current_speed_cmd = SPEEDS['SLOW']    # SLOW when corrections needed
                
                # Use consistent turn command logic
                current_turn_cmd = get_turn_command(current_steering)
                
                # Debug output every 15 frames for better visibility
                if frame_count % 15 == 0:
                    logger.info(f"🎯 TURN DEBUG: offset={final_offset:.3f}, smooth={smoothed_offset:.3f}, steer={current_steering:.3f}, turn={current_turn_cmd}, deadzone={STEERING_DEADZONE}")

            elif final_offset is not None and confidence > 0.1:  # Reduced from 0.3 to 0.1 for faster weak signal response
                robot_status = f"Weak Sig (C:{confidence:.2f})"
                search_counter=0; last_known_offset=smoothed_offset
                steering_error = -smoothed_offset * 0.7  # Increased response from 0.5 to 0.7
                current_steering = pid_controller.calculate(steering_error)
                current_speed_cmd = SPEEDS['SLOW']  # Keep slow for weak signals
                current_turn_cmd = get_turn_command(current_steering)
            else: 
                robot_status = f"Search...({search_counter})"
                current_steering=0.0; pid_controller.reset()
                search_counter+=1
                # Faster, more aggressive search pattern
                if search_counter < 5: current_turn_cmd='LEFT' if last_known_offset<0 else 'RIGHT'; current_speed_cmd=SPEEDS['SLOW']
                elif search_counter < 10: current_turn_cmd='RIGHT' if last_known_offset<0 else 'LEFT'; current_speed_cmd=SPEEDS['SLOW']
                elif search_counter < 20: current_turn_cmd='LEFT' if (search_counter//3)%2==0 else 'RIGHT'; current_speed_cmd=SPEEDS['SLOW']
                else:
                    current_turn_cmd='FORWARD'; current_speed_cmd=SPEEDS['SLOW']
                    if search_counter > 25: search_counter=0  # Reset faster

            # Removed slow speed logging for cleaner output

            # Send command to ESP32 with verification
            if esp_comm: 
                cmd_sent = esp_comm.send_command(current_speed_cmd, current_turn_cmd)
                if not cmd_sent and frame_count % 30 == 0:
                    logger.warning(f"⚠️ ESP32 Command failed: {current_speed_cmd}-{current_turn_cmd}")
            else:
                logger.error("❌ ESP32 communicator not available")

            # --- Visuals & Update ---
            draw_line_overlays(display_frame, ROI_ZONES, viz_data, smoothed_offset if final_offset is not None else None, confidence)
            draw_car_arrow(display_frame); draw_status_panel(display_frame)
            
            # Simplified connection indicator
            conn_txt="ESP:OK" if esp_comm and esp_comm.sock else "ESP:NO"
            cv2.putText(display_frame,conn_txt,(10,CAM_H-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0) if esp_comm and esp_comm.sock else (0,0,255),1)
            with frame_lock: output_frame_flask = display_frame.copy()

            # --- Performance & Simplified Logging ---
            processing_time = time.perf_counter()-loop_start_time
            fps_deque.append(1.0/processing_time if processing_time > 0 else REQUESTED_CAM_FPS)
            fps_current = sum(fps_deque)/len(fps_deque) if fps_deque else 0.0
            
            frame_count+=1
            # Reduced logging frequency for better performance - every 30 seconds instead of 10
            if frame_count % (int(REQUESTED_CAM_FPS*30))==0:  
                logger.info(f"Status: {robot_status} | FPS:{fps_current:.1f} | Off:{current_line_offset:.2f} | Cmd:{current_speed_cmd}-{current_turn_cmd}")
                if confidence_score<0.1 and ("Follow" in robot_status or "Weak" in robot_status): 
                    logger.warning(f"LOW CONFIDENCE: {confidence_score:.2f}")
    
    except KeyboardInterrupt: logger.info("Shutdown by user.")
    except Exception as e: logger.error(f"MAIN LOOP ERROR: {e}",exc_info=True)
    finally:
        logger.info("Cleaning up...")
        if esp_comm: esp_comm.close()
        if 'cap' in locals() and cap.isOpened(): cap.release()
        cv2.destroyAllWindows()
        logger.info("Cleanup done. Exiting.")

if __name__ == "__main__":
    main()