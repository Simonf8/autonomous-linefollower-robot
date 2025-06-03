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
ESP32_IP = '192.168.53.117'
ESP32_PORT = 1234
REQUESTED_CAM_W, REQUESTED_CAM_H = 320, 240
REQUESTED_CAM_FPS = 20

# Enhanced Canny parameters
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_THRESHOLD = 25
HOUGH_MIN_LINE_LENGTH = 20
HOUGH_MAX_LINE_GAP = 15

# Improved thresholding
BLACK_THRESHOLD = 80
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C = 5

# Grid-based analysis parameters
GRID_ROWS = 6
GRID_COLS = 8
MIN_BLACK_PIXELS_RATIO = 0.15

# Enhanced ROI zones with more granular control
ROI_ZONES = [
    {"height_ratio": 0.25, "top_offset": 0.70, "weight": 3.0},  # Primary zone
    {"height_ratio": 0.20, "top_offset": 0.50, "weight": 2.0},  # Secondary zone
    {"height_ratio": 0.15, "top_offset": 0.35, "weight": 1.0},  # Tertiary zone
]

# Improved PID parameters
PID_KP = 1.2
PID_KI = 0.05
PID_KD = 0.35
PID_INT_MAX = 0.8

SPEEDS = {'FAST':'F', 'NORMAL':'N', 'SLOW':'S', 'TURN':'T', 'STOP':'H'}

STEERING_DEADZONE = 0.08
OFFSET_THRESHOLDS = {'PERFECT':0.05, 'GOOD':0.15, 'MODERATE':0.30, 'LARGE':0.50}

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
line_history = deque(maxlen=10)  # For temporal consistency

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
            self.sock = socket.create_connection((self.ip, self.port), timeout=2)
            self.sock.settimeout(0.5)
            logger.info(f"✅ ESP32 connected: {self.ip}:{self.port}")
            self.connect_attempts = 0
            self.last_command = None
            return True
        except Exception as e:
            logger.error(f"❌ ESP32 connection failed (Attempt {self.connect_attempts+1}): {e}")
            self.sock = None
            self.connect_attempts += 1
            return False
    
    def send_command(self, speed_cmd, turn_cmd):
        if not self.sock:
            if self.connect_attempts >= self.max_connect_attempts:
                if time.time() - self.last_heartbeat > 5:
                    logger.info("Attempting ESP32 reconnect...")
                    self.connect_attempts = 0
                    if not self.connect(): 
                        return False
                    self.last_heartbeat = time.time()
                else: 
                    return False
            elif not self.connect(): 
                return False
        
        try:
            # Create command string
            command = f"{speed_cmd}:{turn_cmd}\n"
            
            # Only send if command changed
            if self.last_command != command:
                self.sock.sendall(command.encode())
                self.last_command = command
                logger.debug(f"📡 Sent to ESP32: {command.strip()}")
            
            return True
        except socket.timeout:
            logger.error("💥 ESP32 Send Timeout")
            self.sock = None
            self.connect_attempts = 0
            return False
        except socket.error as e:
            logger.error(f"💥 ESP32 Socket Error: {e}")
            self.sock = None
            self.connect_attempts = 0
            return False
        except Exception as e:
            logger.error(f"💥 ESP32 General Error: {e}")
            self.sock = None
            self.connect_attempts = 0
            return False
    
    def close(self):
        if self.sock:
            try:
                stop_command = f"{SPEEDS['STOP']}:FORWARD\n"
                logger.info(f"Sending STOP command to ESP32: {stop_command.strip()}")
                self.sock.sendall(stop_command.encode())
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"⚠️ Error sending stop command: {e}")
            finally:
                try:
                    self.sock.close()
                except:
                    pass
                self.sock = None
                logger.info("🔌 ESP32 socket closed.")

# -----------------------------------------------------------------------------
# --- Enhanced Image Processing Functions ---
# -----------------------------------------------------------------------------
def apply_median_filter(image, kernel_size=MEDIAN_FILTER_SIZE):
    """Apply median filter to reduce noise"""
    return cv2.medianBlur(image, kernel_size)

def enhance_contrast_adaptive(image):
    """Enhanced contrast using adaptive histogram equalization"""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    return clahe.apply(image)

def preprocess_for_black_lines(frame):
    """Enhanced preprocessing with median filtering and noise reduction"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply median filter to reduce noise
    filtered = apply_median_filter(gray, MEDIAN_FILTER_SIZE)
    
    # Enhanced contrast
    enhanced = enhance_contrast_adaptive(filtered)
    
    # Gaussian blur for smoothing
    blurred = cv2.GaussianBlur(enhanced, (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), 0)
    
    # Handle bright spots (reflections)
    bright_mask_val, bright_mask = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(bright_mask) > 0:
        blurred = cv2.inpaint(blurred, bright_mask, 3, cv2.INPAINT_TELEA)
    
    # Multiple thresholding approaches
    _, binary_simple = cv2.threshold(blurred, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    binary_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C)
    
    # Combine thresholding results
    combined = cv2.bitwise_or(binary_simple, binary_adaptive)
    
    # Morphological operations for noise cleanup
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Close small gaps
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
    # Remove small noise
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)
    
    return cleaned

def analyze_grid_based_line(binary_image):
    """Grid-based line analysis for better line detection"""
    h, w = binary_image.shape
    grid_h, grid_w = h // GRID_ROWS, w // GRID_COLS
    
    line_cells = []
    
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            y1, y2 = row * grid_h, min((row + 1) * grid_h, h)
            x1, x2 = col * grid_w, min((col + 1) * grid_w, w)
            
            cell = binary_image[y1:y2, x1:x2]
            black_pixels = np.sum(cell > 0)
            total_pixels = cell.size
            
            if total_pixels > 0:
                black_ratio = black_pixels / total_pixels
                if black_ratio > MIN_BLACK_PIXELS_RATIO:
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    line_cells.append({
                        'center_x': center_x,
                        'center_y': center_y,
                        'confidence': black_ratio,
                        'row': row,
                        'col': col
                    })
    
    return line_cells

def fit_line_from_cells(line_cells, image_width):
    """Fit a line through detected grid cells"""
    if len(line_cells) < 2:
        return None, None, 0.0
    
    # Extract points
    points = np.array([(cell['center_x'], cell['center_y']) for cell in line_cells])
    weights = np.array([cell['confidence'] for cell in line_cells])
    
    # Weighted linear regression
    try:
        # Fit line: y = mx + b
        A = np.vstack([points[:, 0], np.ones(len(points))]).T
        coeffs = np.linalg.lstsq(A, points[:, 1], rcond=None)[0]
        slope, intercept = coeffs
        
        # Calculate angle
        angle = np.degrees(np.arctan(slope))
        angle = angle + 90  # Convert to vertical reference
        
        # Calculate center offset at bottom of image
        bottom_y = len(line_cells) * 40  # Approximate bottom
        center_x_at_bottom = (bottom_y - intercept) / slope if slope != 0 else points[-1, 0]
        offset = (center_x_at_bottom - image_width/2) / (image_width/2)
        
        # Calculate confidence based on number of cells and their confidence
        confidence = min(len(line_cells) / 10.0, 1.0) * np.mean(weights)
        
        return offset, angle, confidence
        
    except np.linalg.LinAlgError:
        return None, None, 0.0

def extract_roi_zones(frame):
    h, w = frame.shape[:2]
    rois = []
    for z in ROI_ZONES:
        rt, rh = int(h * z["top_offset"]), int(h * z["height_ratio"])
        rt = max(0, min(rt, h-1))
        rh = max(1, min(rh, h-rt))
        rois.append({
            'roi': frame[rt:rt+rh, :], 
            'top': rt, 
            'height': rh, 
            'weight': z["weight"]
        })
    return rois

def detect_black_lines_enhanced(roi_data, full_binary_image):
    """Enhanced line detection combining traditional and grid-based methods"""
    best_offset, best_angle, highest_weighted_conf = None, None, 0
    all_lines_info = []
    
    # Traditional Hough line detection
    for roi_info in roi_data:
        roi, weight, roi_top_abs = roi_info['roi'], roi_info['weight'], roi_info['top']
        if roi.size == 0: 
            continue
            
        roi_h, roi_w = roi.shape
        
        # Enhanced edge detection
        edges1 = cv2.Canny(roi, CANNY_LOW, CANNY_HIGH)
        edges2 = cv2.Canny(roi, CANNY_LOW//2, CANNY_HIGH//2)
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                               max(HOUGH_THRESHOLD//2, 8),
                               minLineLength=max(HOUGH_MIN_LINE_LENGTH//2, 10),
                               maxLineGap=HOUGH_MAX_LINE_GAP*2)
        
        if lines is None: 
            continue
            
        valid_lines = []
        for line_seg in lines:
            x1, y1, x2, y2 = line_seg[0]
            length = np.hypot(x2-x1, y2-y1)
            if length < 8: 
                continue
                
            angle = np.degrees(np.arctan2(y2-y1, x2-x1))
            angle = angle + 180 if angle < 0 else angle
            
            # More lenient angle filter for better detection
            if abs(angle - 90) > 60: 
                continue
                
            cx = (x1 + x2) / 2
            pos_score = 1.0 - (abs(cx - roi_w/2) / (roi_w/2))
            len_score = min(length / (roi_h * 0.6), 1.0)
            total_score = (pos_score * 0.6 + len_score * 0.4) * length
            
            valid_lines.append({
                'coords': (x1, y1, x2, y2),
                'angle': angle,
                'center_x': cx,
                'score': total_score,
                'roi_top_offset': roi_top_abs
            })
        
        if valid_lines:
            valid_lines.sort(key=lambda x: x['score'], reverse=True)
            best_zone_lines = valid_lines[:min(3, len(valid_lines))]
            
            if best_zone_lines:
                avg_cx = np.mean([l['center_x'] for l in best_zone_lines])
                
                # Weighted angle calculation
                sx = sum(l['score'] * math.cos(math.radians(l['angle'])) for l in best_zone_lines)
                sy = sum(l['score'] * math.sin(math.radians(l['angle'])) for l in best_zone_lines)
                avg_angle_rad = math.atan2(sy, sx)
                current_avg_angle = math.degrees(avg_angle_rad)
                current_avg_angle = current_avg_angle + 180 if current_avg_angle < 0 else current_avg_angle
                
                norm_offset = (avg_cx - roi_w/2) / (roi_w/2)
                conf_roi = (min(len(best_zone_lines) / 2.0, 1.0)) * weight
                
                if conf_roi > highest_weighted_conf:
                    highest_weighted_conf = conf_roi
                    best_offset, best_angle = norm_offset, current_avg_angle
                    all_lines_info = best_zone_lines
    
    # Grid-based analysis as backup/enhancement
    grid_cells = analyze_grid_based_line(full_binary_image)
    grid_offset, grid_angle, grid_conf = fit_line_from_cells(grid_cells, full_binary_image.shape[1])
    
    # Combine results - prefer traditional if confident, otherwise use grid
    if highest_weighted_conf > 0.3:
        final_conf = min(highest_weighted_conf + grid_conf * 0.3, 1.0)
        final_offset, final_angle = best_offset, best_angle
    elif grid_conf > 0.2:
        final_conf = grid_conf
        final_offset, final_angle = grid_offset, grid_angle
        all_lines_info = []  # Grid method doesn't provide line segments
    else:
        final_conf = max(highest_weighted_conf, grid_conf) if highest_weighted_conf or grid_conf else 0.0
        final_offset, final_angle = best_offset, best_angle
    
    return final_offset, final_angle, all_lines_info, final_conf

# -----------------------------------------------------------------------------
# --- MOVEMENT CONTROL HELPER ---
# -----------------------------------------------------------------------------
def get_turn_command(steering_output):
    if abs(steering_output)<STEERING_DEADZONE: return 'FORWARD'
    elif steering_output<-0.05: return 'LEFT'
    elif steering_output>0.05: return 'RIGHT'
    else: return 'FORWARD'

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

def draw_line_overlays(display_frame, roi_config_list, detected_lines, offset, confidence):
    h, w = display_frame.shape[:2]; center_x = w//2
    for i, zone_cfg in enumerate(roi_config_list):
        color=(255,100,0)
        abs_roi_top = int(h * zone_cfg["top_offset"])
        abs_roi_height = int(h * zone_cfg["height_ratio"])
        cv2.rectangle(display_frame,(5, abs_roi_top),(w-5, abs_roi_top + abs_roi_height),color,1)
        cv2.putText(display_frame,f"ROI{i+1}",(10, abs_roi_top + 15),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1)
    if detected_lines:
        for line_info in detected_lines: 
            x1,y1,x2,y2 = line_info['coords']
            roi_abs_top_of_this_line = line_info['roi_top_offset'] 
            y1a, y2a = y1 + roi_abs_top_of_this_line, y2 + roi_abs_top_of_this_line 
            cv2.line(display_frame,(int(x1),int(y1a)),(int(x2),int(y2a)),(255,255,255),4)
            cv2.line(display_frame,(int(x1),int(y1a)),(int(x2),int(y2a)),(0,255,255),2)
    cv2.line(display_frame,(center_x,0),(center_x,h),(200,200,200),1,cv2.LINE_AA)
    if offset is not None and roi_config_list:
        primary_roi_abs_top = int(h * roi_config_list[0]["top_offset"])
        primary_roi_abs_height = int(h * roi_config_list[0]["height_ratio"])
        primary_roi_cy = primary_roi_abs_top + primary_roi_abs_height // 2
        target_x = int(center_x + (offset * (w / 4)))
        cv2.circle(display_frame,(target_x, primary_roi_cy),10,(0,0,255),-1)
        cv2.circle(display_frame,(target_x, primary_roi_cy),12,(255,255,255),1)
        cv2.line(display_frame,(center_x, primary_roi_cy),(target_x, primary_roi_cy),(255,0,255),2)

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
    cap = cv2.VideoCapture(1)
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

    # PYTHON WORKAROUND FOR 'S' (SLOW) COMMAND STALLING THE ROBOT
    # Set this to True IF AND ONLY IF you cannot modify your ESP32 firmware to make 'S' speed effective.
    # This will make the robot use 'NORMAL' speed instead of 'SLOW' when following,
    # which might be less smooth but will prevent stalling if 'S' is too weak.
    USE_NORMAL_INSTEAD_OF_STALLING_SLOW = False
    if USE_NORMAL_INSTEAD_OF_STALLING_SLOW:
        logger.warning("⚠️ PYTHON WORKAROUND ENABLED: Using NORMAL speed instead of potentially stalling SLOW speed.")


    try:
        while True:
            loop_start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret: logger.warning("⚠️ No frame"); robot_status="No Frame"; time.sleep(0.1); continue
            if frame.shape[1]!=CAM_W or frame.shape[0]!=CAM_H: frame=cv2.resize(frame,(CAM_W,CAM_H))

            display_frame = frame.copy()
            processed_frame = preprocess_for_black_lines(frame)
            roi_zones_data = extract_roi_zones(processed_frame)
            final_offset, final_angle, detected_line_segments, confidence = detect_black_lines_enhanced(roi_zones_data, processed_frame)
            
            # Changed to INFO for better visibility of this crucial debug line without setting global DEBUG
            logger.info(f"Detection: Off={final_offset:.2f if final_offset else None}, Ang={final_angle:.1f if final_angle else None}, Segs={len(detected_line_segments if detected_line_segments else [])}, Conf={confidence:.2f}")

            current_line_offset=final_offset if final_offset is not None else 0.0
            current_line_angle=final_angle if final_angle is not None else 0.0
            lines_detected=len(detected_line_segments) if detected_line_segments else 0
            confidence_score = confidence

            # --- Control Logic ---
            if final_offset is not None and confidence > 0.15: 
                logger.info("Decision: FOLLOW line") # Changed to INFO for visibility
                robot_status = f"Follow (C:{confidence:.2f})"
                search_counter=0; search_memory.append(final_offset); last_known_offset=final_offset
                steering_error = -final_offset
                current_steering = pid_controller.calculate(steering_error)
                abs_offset = abs(final_offset)

                if confidence > 0.6 and abs_offset < OFFSET_THRESHOLDS['PERFECT']:
                    current_speed_cmd = SPEEDS['FAST']
                elif confidence > 0.4 and abs_offset < OFFSET_THRESHOLDS['GOOD']:
                    current_speed_cmd = SPEEDS['NORMAL']
                elif abs_offset < OFFSET_THRESHOLDS['MODERATE']:
                    current_speed_cmd = SPEEDS['SLOW']
                    if USE_NORMAL_INSTEAD_OF_STALLING_SLOW: current_speed_cmd = SPEEDS['NORMAL']
                else: 
                    current_speed_cmd = SPEEDS['SLOW']
                    if USE_NORMAL_INSTEAD_OF_STALLING_SLOW: current_speed_cmd = SPEEDS['NORMAL']
                
                if abs(current_steering) < STEERING_DEADZONE: current_turn_cmd='FORWARD'
                elif current_steering < -0.05: current_turn_cmd='LEFT'
                elif current_steering > 0.05: current_turn_cmd='RIGHT'
                else: current_turn_cmd='FORWARD'

            elif final_offset is not None and confidence > 0.05: 
                logger.info("Decision: WEAK SIGNAL") # Changed to INFO
                robot_status = f"Weak Sig (C:{confidence:.2f})"
                search_counter=0; last_known_offset=final_offset
                steering_error = -final_offset * 0.7
                current_steering = pid_controller.calculate(steering_error)
                current_speed_cmd = SPEEDS['SLOW']
                if USE_NORMAL_INSTEAD_OF_STALLING_SLOW: current_speed_cmd = SPEEDS['NORMAL']
                current_turn_cmd = get_turn_command(current_steering)
            else: 
                logger.info(f"Decision: SEARCH (PrevOff: {last_known_offset:.2f}, Cnt: {search_counter})") # Changed to INFO
                robot_status = f"Search...({search_counter})"
                current_steering=0.0; pid_controller.reset()
                search_counter+=1
                if search_counter < 7: current_turn_cmd='LEFT' if last_known_offset<0 else 'RIGHT'; current_speed_cmd=SPEEDS['NORMAL']
                elif search_counter < 14: current_turn_cmd='RIGHT' if last_known_offset<0 else 'LEFT'; current_speed_cmd=SPEEDS['NORMAL']
                elif search_counter < 28: current_turn_cmd='LEFT' if (search_counter//4)%2==0 else 'RIGHT'; current_speed_cmd=SPEEDS['SLOW']
                else:
                    current_turn_cmd='FORWARD'; current_speed_cmd=SPEEDS['SLOW']
                    if search_counter > 40: search_counter=0

            if current_speed_cmd == SPEEDS['SLOW'] and (robot_status.startswith("Follow") or robot_status.startswith("Weak Sig")):
                logger.info(f"INFO: Intending to move SLOW. Status: {robot_status}, Offset: {current_line_offset:.2f}, Conf: {confidence_score:.2f}")

            if esp_comm: esp_comm.send_command(current_speed_cmd, current_turn_cmd)

            # --- Visuals & Update ---
            draw_line_overlays(display_frame, ROI_ZONES, detected_line_segments, final_offset, confidence)
            draw_car_arrow(display_frame); draw_status_panel(display_frame)
            if processed_frame is not None and processed_frame.ndim==2:
                dbg_frm_clr=cv2.cvtColor(processed_frame,cv2.COLOR_GRAY2BGR); dbg_sml=cv2.resize(dbg_frm_clr,(CAM_W//4,CAM_H//4))
                try: 
                    display_frame[10:10+CAM_H//4, CAM_W-CAM_W//4-10:CAM_W-10] = dbg_sml
                except ValueError as e: logger.warning(f"Could not place debug view: {e}")

            conn_txt="ESP:OK" if esp_comm and esp_comm.sock else "ESP:NO"
            cv2.putText(display_frame,conn_txt,(10,CAM_H-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0) if esp_comm and esp_comm.sock else (0,0,255),1)
            with frame_lock: output_frame_flask = display_frame.copy()

            # --- Performance & Periodic Logging ---
            processing_time = time.perf_counter()-loop_start_time
            fps_deque.append(1.0/processing_time if processing_time > 0 else REQUESTED_CAM_FPS)
            fps_current = sum(fps_deque)/len(fps_deque) if fps_deque else 0.0
            
            frame_count+=1
            if frame_count % (int(REQUESTED_CAM_FPS*2))==0: 
                logger.info(f"St:{robot_status}, FPS:{fps_current:.1f}, Off:{current_line_offset:.2f}, Str:{current_steering:.2f}, Conf:{confidence_score:.2f}, Cmd:{current_speed_cmd}-{current_turn_cmd}, Lines:{lines_detected}")
                if lines_detected==0 and "Search" not in robot_status: logger.warning(f"WARN: NO LINE! Check THRESH({BLACK_THRESHOLD}), lighting, line quality.")
                elif confidence_score<0.3 and ("Follow" in robot_status or "Weak" in robot_status) : logger.warning(f"WARN: LOW CONF ({confidence_score:.2f}) while trying to follow!")
    
    except KeyboardInterrupt: logger.info("🛑 Shutdown by user.")
    except Exception as e: logger.error(f"💥 MAIN LOOP ERROR: {e}",exc_info=True)
    finally:
        logger.info("🧹 Cleaning up...")
        if esp_comm: esp_comm.close()
        if 'cap' in locals() and cap.isOpened(): cap.release()
        cv2.destroyAllWindows()
        logger.info("✅ Cleanup done. Exiting.")

if __name__ == "__main__":
    main()