#!/usr/bin/env python3

import cv2
import numpy as np
import socket
import time
import logging
import sys
import math
import json
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from collections import deque
from pathlib import Path
from flask import Flask, Response, render_template_string, jsonify, request
from scipy import ndimage
from sklearn.cluster import DBSCAN
import yaml

# -----------------------------------------------------------------------------
# --- CONFIGURATION MANAGEMENT ---
# -----------------------------------------------------------------------------

@dataclass
class VisionConfig:
    """Vision processing configuration"""
    cam_width: int = 320
    cam_height: int = 240
    cam_fps: int = 20
    cam_index: int = 1
    
    # Preprocessing
    median_filter_size: int = 5
    gaussian_blur_size: int = 7
    black_threshold: int = 80
    adaptive_block_size: int = 11
    adaptive_c: int = 5
    
    # Edge detection
    canny_low: int = 50
    canny_high: int = 150
    
    # Hough line detection
    hough_threshold: int = 25
    hough_min_line_length: int = 20
    hough_max_line_gap: int = 15
    
    # Grid analysis
    grid_rows: int = 6
    grid_cols: int = 8
    min_black_pixels_ratio: float = 0.15
    
    # ROI zones
    roi_zones: List[Dict] = None

@dataclass
class ControlConfig:
    """Control system configuration"""
    # PID parameters
    pid_kp: float = 1.2
    pid_ki: float = 0.05
    pid_kd: float = 0.35
    pid_integral_max: float = 0.8
    
    # Adaptive PID
    enable_adaptive_pid: bool = True
    pid_kp_range: Tuple[float, float] = (0.8, 2.0)
    pid_adaptation_rate: float = 0.1
    
    # Steering
    steering_deadzone: float = 0.08
    max_steering_rate: float = 0.5  # Maximum steering change per frame
    
    # Speed control
    speed_zones: Dict[str, Dict] = None
    
    # Search behavior
    search_timeout: int = 40
    search_oscillation_amplitude: float = 0.3

@dataclass
class SystemConfig:
    """System configuration"""
    esp32_ip: str = '192.168.53.117'
    esp32_port: int = 1234
    flask_port: int = 5000
    
    # Logging
    log_level: str = 'INFO'
    enable_data_logging: bool = True
    log_file_path: str = 'robot_data.log'
    
    # Performance
    enable_threading: bool = True
    frame_skip_ratio: float = 0.0  # Skip frames when processing is slow
    max_processing_time: float = 0.1  # Target processing time per frame

class ConfigManager:
    """Manages configuration loading and saving"""
    
    def __init__(self, config_file: str = 'robot_config.yaml'):
        self.config_file = Path(config_file)
        self.vision = VisionConfig()
        self.control = ControlConfig()
        self.system = SystemConfig()
        self.load_config()
        self._setup_defaults()
    
    def _setup_defaults(self):
        """Set up default values that can't be in dataclass"""
        if self.vision.roi_zones is None:
            self.vision.roi_zones = [
                {"height_ratio": 0.25, "top_offset": 0.70, "weight": 3.0},
                {"height_ratio": 0.20, "top_offset": 0.50, "weight": 2.0},
                {"height_ratio": 0.15, "top_offset": 0.35, "weight": 1.0},
            ]
        
        if self.control.speed_zones is None:
            self.control.speed_zones = {
                'PERFECT': {'threshold': 0.05, 'speed': 'FAST'},
                'GOOD': {'threshold': 0.15, 'speed': 'NORMAL'},
                'MODERATE': {'threshold': 0.30, 'speed': 'SLOW'},
                'LARGE': {'threshold': 0.50, 'speed': 'SLOW'},
            }
    
    def load_config(self):
        """Load configuration from YAML file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                if 'vision' in data:
                    self.vision = VisionConfig(**data['vision'])
                if 'control' in data:
                    self.control = ControlConfig(**data['control'])
                if 'system' in data:
                    self.system = SystemConfig(**data['system'])
                    
                logging.info(f"✅ Configuration loaded from {self.config_file}")
            except Exception as e:
                logging.warning(f"⚠️ Failed to load config: {e}, using defaults")
    
    def save_config(self):
        """Save current configuration to YAML file"""
        try:
            data = {
                'vision': self.vision.__dict__,
                'control': self.control.__dict__,
                'system': self.system.__dict__,
            }
            with open(self.config_file, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            logging.info(f"✅ Configuration saved to {self.config_file}")
        except Exception as e:
            logging.error(f"❌ Failed to save config: {e}")

# -----------------------------------------------------------------------------
# --- PERFORMANCE MONITORING ---
# -----------------------------------------------------------------------------

class PerformanceMonitor:
    """Monitor system performance and provide optimization suggestions"""
    
    def __init__(self, history_size: int = 100):
        self.processing_times = deque(maxlen=history_size)
        self.fps_history = deque(maxlen=history_size)
        self.memory_usage = deque(maxlen=50)
        self.detection_confidence = deque(maxlen=history_size)
        
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_optimization_check = time.time()
    
    def update(self, processing_time: float, fps: float, confidence: float):
        """Update performance metrics"""
        self.processing_times.append(processing_time)
        self.fps_history.append(fps)
        self.detection_confidence.append(confidence)
        self.frame_count += 1
        
        # Check for optimization opportunities every 10 seconds
        if time.time() - self.last_optimization_check > 10:
            self._check_optimization_opportunities()
            self.last_optimization_check = time.time()
    
    def _check_optimization_opportunities(self):
        """Suggest optimizations based on performance data"""
        if len(self.processing_times) < 10:
            return
        
        avg_processing_time = np.mean(list(self.processing_times)[-20:])
        avg_fps = np.mean(list(self.fps_history)[-20:])
        avg_confidence = np.mean(list(self.detection_confidence)[-20:])
        
        # Suggest optimizations
        if avg_processing_time > 0.15:
            logging.warning("🐌 Processing time high, consider reducing resolution or enabling frame skipping")
        
        if avg_fps < 15:
            logging.warning("📉 Low FPS detected, system may be overloaded")
        
        if avg_confidence < 0.3:
            logging.warning("🎯 Low detection confidence, check lighting or line quality")
    
    def get_stats(self) -> Dict:
        """Get current performance statistics"""
        if not self.processing_times:
            return {}
        
        return {
            'avg_processing_time': np.mean(list(self.processing_times)),
            'avg_fps': np.mean(list(self.fps_history)),
            'avg_confidence': np.mean(list(self.detection_confidence)),
            'frame_count': self.frame_count,
            'dropped_frames': self.dropped_frames,
            'efficiency': (self.frame_count - self.dropped_frames) / max(self.frame_count, 1)
        }

# -----------------------------------------------------------------------------
# --- ENHANCED LINE DETECTION WITH KALMAN FILTERING ---
# -----------------------------------------------------------------------------

class KalmanLineFilter:
    """Kalman filter for smooth line tracking"""
    
    def __init__(self):
        self.initialized = False
        self.state = np.zeros(4)  # [offset, offset_velocity, angle, angle_velocity]
        self.covariance = np.eye(4) * 1000
        
        # Process noise (how much we expect the line to change)
        self.process_noise = np.diag([0.1, 0.5, 0.1, 0.5])
        
        # Measurement noise (how much we trust our observations)
        self.measurement_noise = np.diag([0.2, 0.3])
        
        # State transition model (constant velocity)
        self.transition = np.array([
            [1, 1, 0, 0],  # offset += offset_velocity
            [0, 1, 0, 0],  # offset_velocity unchanged
            [0, 0, 1, 1],  # angle += angle_velocity
            [0, 0, 0, 1]   # angle_velocity unchanged
        ])
        
        # Observation model
        self.observation = np.array([
            [1, 0, 0, 0],  # observe offset
            [0, 0, 1, 0]   # observe angle
        ])
    
    def predict(self):
        """Predict next state"""
        if not self.initialized:
            return None, None
        
        # Predict state
        self.state = self.transition @ self.state
        
        # Predict covariance
        self.covariance = (self.transition @ self.covariance @ self.transition.T) + self.process_noise
        
        return self.state[0], self.state[2]  # offset, angle
    
    def update(self, offset: float, angle: float, confidence: float):
        """Update filter with new measurement"""
        measurement = np.array([offset, angle])
        
        if not self.initialized:
            self.state = np.array([offset, 0, angle, 0])
            self.initialized = True
            return offset, angle
        
        # Adjust measurement noise based on confidence
        adjusted_noise = self.measurement_noise * (2.0 - confidence)
        
        # Calculate Kalman gain
        innovation_covariance = (self.observation @ self.covariance @ self.observation.T) + adjusted_noise
        kalman_gain = self.covariance @ self.observation.T @ np.linalg.inv(innovation_covariance)
        
        # Update state
        innovation = measurement - (self.observation @ self.state)
        self.state = self.state + (kalman_gain @ innovation)
        
        # Update covariance
        self.covariance = (np.eye(4) - kalman_gain @ self.observation) @ self.covariance
        
        return self.state[0], self.state[2]  # filtered offset, angle

# -----------------------------------------------------------------------------
# --- THREADED IMAGE PROCESSOR ---
# -----------------------------------------------------------------------------

class ThreadedImageProcessor:
    """Process images in separate thread for better performance"""
    
    def __init__(self, config: VisionConfig, max_queue_size: int = 2):
        self.config = config
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.thread = None
        
        # Pre-create morphological kernels for better performance
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Pre-create CLAHE object
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    
    def start(self):
        """Start processing thread"""
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        logging.info("🧵 Image processing thread started")
    
    def stop(self):
        """Stop processing thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
    
    def process_frame_async(self, frame: np.ndarray) -> bool:
        """Queue frame for processing (non-blocking)"""
        try:
            self.input_queue.put_nowait(frame.copy())
            return True
        except queue.Full:
            return False
    
    def get_result(self) -> Optional[Tuple]:
        """Get processing result (non-blocking)"""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _process_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)
                result = self._process_frame(frame)
                
                try:
                    self.output_queue.put_nowait(result)
                except queue.Full:
                    # Remove oldest result to make room
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait(result)
                    except queue.Empty:
                        pass
                        
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"❌ Image processing error: {e}")
    
    def _process_frame(self, frame: np.ndarray) -> Tuple:
        """Process single frame"""
        # Enhanced preprocessing with optimizations
        processed_frame = self._preprocess_optimized(frame)
        roi_zones_data = self._extract_roi_zones(processed_frame)
        final_offset, final_angle, detected_line_segments, confidence = self._detect_lines_enhanced(roi_zones_data, processed_frame)
        
        return final_offset, final_angle, detected_line_segments, confidence, processed_frame
    
    def _preprocess_optimized(self, frame: np.ndarray) -> np.ndarray:
        """Optimized preprocessing pipeline"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply median filter to reduce noise
        filtered = cv2.medianBlur(gray, self.config.median_filter_size)
        
        # Enhanced contrast using pre-created CLAHE
        enhanced = self.clahe.apply(filtered)
        
        # Gaussian blur for smoothing
        blurred = cv2.GaussianBlur(enhanced, (self.config.gaussian_blur_size, self.config.gaussian_blur_size), 0)
        
        # Handle bright spots (reflections) - optimized
        bright_mask = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)[1]
        if cv2.countNonZero(bright_mask) > 0:
            blurred = cv2.inpaint(blurred, bright_mask, 3, cv2.INPAINT_TELEA)
        
        # Multiple thresholding approaches
        binary_simple = cv2.threshold(blurred, self.config.black_threshold, 255, cv2.THRESH_BINARY_INV)[1]
        binary_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY_INV, self.config.adaptive_block_size, self.config.adaptive_c)
        
        # Combine thresholding results
        combined = cv2.bitwise_or(binary_simple, binary_adaptive)
        
        # Morphological operations using pre-created kernels
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self.kernel_close)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, self.kernel_open)
        
        return cleaned
    
    def _extract_roi_zones(self, frame: np.ndarray) -> List[Dict]:
        """Extract ROI zones from frame"""
        h, w = frame.shape[:2]
        rois = []
        for z in self.config.roi_zones:
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
    
    def _detect_lines_enhanced(self, roi_data: List[Dict], full_binary_image: np.ndarray) -> Tuple:
        """Enhanced line detection with multiple methods"""
        best_offset, best_angle, highest_weighted_conf = None, None, 0
        all_lines_info = []
        
        # Traditional Hough line detection
        for roi_info in roi_data:
            roi, weight, roi_top_abs = roi_info['roi'], roi_info['weight'], roi_info['top']
            if roi.size == 0: 
                continue
                
            roi_h, roi_w = roi.shape
            
            # Enhanced edge detection
            edges1 = cv2.Canny(roi, self.config.canny_low, self.config.canny_high)
            edges2 = cv2.Canny(roi, self.config.canny_low//2, self.config.canny_high//2)
            edges = cv2.bitwise_or(edges1, edges2)
            
            # Hough line detection
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                   max(self.config.hough_threshold//2, 8),
                                   minLineLength=max(self.config.hough_min_line_length//2, 10),
                                   maxLineGap=self.config.hough_max_line_gap*2)
            
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
        grid_cells = self._analyze_grid_based_line(full_binary_image)
        grid_offset, grid_angle, grid_conf = self._fit_line_from_cells(grid_cells, full_binary_image.shape[1])
        
        # Combine results
        if highest_weighted_conf > 0.3:
            final_conf = min(highest_weighted_conf + grid_conf * 0.3, 1.0)
            final_offset, final_angle = best_offset, best_angle
        elif grid_conf > 0.2:
            final_conf = grid_conf
            final_offset, final_angle = grid_offset, grid_angle
            all_lines_info = []
        else:
            final_conf = max(highest_weighted_conf, grid_conf) if highest_weighted_conf or grid_conf else 0.0
            final_offset, final_angle = best_offset, best_angle
        
        return final_offset, final_angle, all_lines_info, final_conf
    
    def _analyze_grid_based_line(self, binary_image: np.ndarray) -> List[Dict]:
        """Grid-based line analysis"""
        h, w = binary_image.shape
        grid_h, grid_w = h // self.config.grid_rows, w // self.config.grid_cols
        
        line_cells = []
        
        for row in range(self.config.grid_rows):
            for col in range(self.config.grid_cols):
                y1, y2 = row * grid_h, min((row + 1) * grid_h, h)
                x1, x2 = col * grid_w, min((col + 1) * grid_w, w)
                
                cell = binary_image[y1:y2, x1:x2]
                black_pixels = np.sum(cell > 0)
                total_pixels = cell.size
                
                if total_pixels > 0:
                    black_ratio = black_pixels / total_pixels
                    if black_ratio > self.config.min_black_pixels_ratio:
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
    
    def _fit_line_from_cells(self, line_cells: List[Dict], image_width: int) -> Tuple[Optional[float], Optional[float], float]:
        """Fit line from grid cells"""
        if len(line_cells) < 2:
            return None, None, 0.0
        
        points = np.array([(cell['center_x'], cell['center_y']) for cell in line_cells])
        weights = np.array([cell['confidence'] for cell in line_cells])
        
        try:
            A = np.vstack([points[:, 0], np.ones(len(points))]).T
            coeffs = np.linalg.lstsq(A, points[:, 1], rcond=None)[0]
            slope, intercept = coeffs
            
            angle = np.degrees(np.arctan(slope))
            angle = angle + 90
            
            bottom_y = len(line_cells) * 40
            center_x_at_bottom = (bottom_y - intercept) / slope if slope != 0 else points[-1, 0]
            offset = (center_x_at_bottom - image_width/2) / (image_width/2)
            
            confidence = min(len(line_cells) / 10.0, 1.0) * np.mean(weights)
            
            return offset, angle, confidence
            
        except np.linalg.LinAlgError:
            return None, None, 0.0

# -----------------------------------------------------------------------------
# --- ADAPTIVE PID CONTROLLER ---
# -----------------------------------------------------------------------------

class AdaptivePIDController:
    """Enhanced PID controller with adaptive parameters and advanced filtering"""
    
    def __init__(self, config: ControlConfig):
        self.config = config
        self.kp = config.pid_kp
        self.ki = config.pid_ki
        self.kd = config.pid_kd
        self.integral_max = config.pid_integral_max
        
        # State variables
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        
        # Enhanced filtering
        self.error_history = deque(maxlen=15)
        self.derivative_history = deque(maxlen=5)
        self.output_history = deque(maxlen=10)
        
        # Adaptive parameters
        self.performance_metric = deque(maxlen=50)
        self.last_adaptation_time = time.time()
        
        # Anti-windup
        self.integral_active = True
        
        # Derivative filtering (low-pass filter)
        self.derivative_filter_alpha = 0.1
        self.filtered_derivative = 0.0
    
    def calculate(self, error: float, dt: Optional[float] = None) -> float:
        """Calculate PID output with adaptive parameters"""
        current_time = time.time()
        dt = max(current_time - self.last_time if dt is None else dt, 1e-3)
        
        # Store error for analysis
        self.error_history.append(error)
        
        # Adaptive parameter adjustment
        if self.config.enable_adaptive_pid:
            self._adapt_parameters()
        
        # Proportional term with gain scheduling
        proportional = self.kp * error
        
        # Integral term with conditional integration
        if self.integral_active and abs(error) < 0.5:  # Only integrate for small errors
            self.integral += error * dt
            self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)
        elif abs(error) > 0.7:  # Reset integral for large errors
            self.integral *= 0.8
        
        integral_term = self.ki * self.integral
        
        # Derivative term with advanced filtering
        derivative = (error - self.prev_error) / dt
        
        # Apply low-pass filter to derivative
        self.filtered_derivative = (self.derivative_filter_alpha * derivative + 
                                  (1 - self.derivative_filter_alpha) * self.filtered_derivative)
        
        self.derivative_history.append(self.filtered_derivative)
        
        # Use median of recent derivatives for robustness
        if len(self.derivative_history) >= 3:
            derivative_term = self.kd * np.median(list(self.derivative_history)[-3:])
        else:
            derivative_term = self.kd * self.filtered_derivative
        
        # Calculate total output
        output = proportional + integral_term + derivative_term
        
        # Output limiting and rate limiting
        if self.output_history:
            max_change = self.config.max_steering_rate * dt
            output = np.clip(output, 
                           self.output_history[-1] - max_change,
                           self.output_history[-1] + max_change)
        
        output = np.clip(output, -1.0, 1.0)
        
        # Store for next iteration
        self.output_history.append(output)
        self.prev_error = error
        self.last_time = current_time
        
        # Update performance metric
        self._update_performance_metric(error, output)
        
        return output
    
    def _adapt_parameters(self):
        """Adapt PID parameters based on performance"""
        if time.time() - self.last_adaptation_time < 2.0:  # Adapt every 2 seconds
            return
            
        if len(self.performance_metric) < 20:
            return
        
        # Calculate recent performance
        recent_performance = np.mean(list(self.performance_metric)[-10:])
        older_performance = np.mean(list(self.performance_metric)[-20:-10])
        
        # Adapt Kp based on performance trend
        if recent_performance > older_performance * 1.1:  # Performance getting worse
            self.kp = max(self.config.pid_kp_range[0], 
                         self.kp - self.config.pid_adaptation_rate)
        elif recent_performance < older_performance * 0.9:  # Performance improving
            self.kp = min(self.config.pid_kp_range[1], 
                         self.kp + self.config.pid_adaptation_rate)
        
        self.last_adaptation_time = time.time()
        logging.debug(f"🎛️ Adapted PID: Kp={self.kp:.3f}")
    
    def _update_performance_metric(self, error: float, output: float):
        """Update performance metric for adaptation"""
        # Performance metric combines error magnitude and output stability
        error_component = abs(error)
        
        # Output stability (penalize rapid changes)
        if len(self.output_history) > 1:
            output_stability = abs(output - self.output_history[-2])
        else:
            output_stability = 0
        
        performance = error_component + 0.5 * output_stability
        self.performance_metric.append(performance)
    
    def reset(self):
        """Reset PID controller state"""
        self.prev_error = 0.0
        self.integral = 0.0
        self.filtered_derivative = 0.0
        self.error_history.clear()
        self.derivative_history.clear()
        self.output_history.clear()
        self.performance_metric.clear()
        self.last_time = time.time()
        logging.info("🔄 Adaptive PID Controller Reset")

# -----------------------------------------------------------------------------
# --- ENHANCED ESP32 COMMUNICATION ---
# -----------------------------------------------------------------------------

class EnhancedESP32Communicator:
    """Enhanced ESP32 communication with better error handling and monitoring"""
    
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.sock = None
        self.last_command = None
        
        # Connection management
        self.connect_attempts = 0
        self.max_connect_attempts = 5
        self.reconnect_delay = 5.0
        self.last_reconnect_attempt = 0
        
        # Monitoring
        self.commands_sent = 0
        self.connection_errors = 0
        self.last_successful_send = 0
        self.response_times = deque(maxlen=20)
        
        # Command queuing for reliability
        self.command_queue = deque(maxlen=10)
        self.command_lock = threading.Lock()
        
        self.connect()
    
    def connect(self) -> bool:
        """Connect to ESP32 with improved error handling"""
        current_time = time.time()
        
        # Rate limit reconnection attempts
        if current_time - self.last_reconnect_attempt < self.reconnect_delay:
            return False
        
        self.last_reconnect_attempt = current_time
        
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        
        try:
            self.sock = socket.create_connection((self.ip, self.port), timeout=3)
            self.sock.settimeout(1.0)
            logging.info(f"✅ ESP32 connected: {self.ip}:{self.port}")
            self.connect_attempts = 0
            self.last_command = None
            return True
            
        except Exception as e:
            self.connection_errors += 1
            logging.error(f"❌ ESP32 connection failed (Attempt {self.connect_attempts+1}): {e}")
            self.sock = None
            self.connect_attempts += 1
            return False
    
    def send_command_reliable(self, speed_cmd: str, turn_cmd: str) -> bool:
        """Send command with reliability features"""
        command = f"{speed_cmd}:{turn_cmd}\n"
        
        with self.command_lock:
            # Add to queue for retry capability
            self.command_queue.append({
                'command': command,
                'timestamp': time.time(),
                'attempts': 0
            })
        
        return self._send_immediate(command)
    
    def _send_immediate(self, command: str) -> bool:
        """Send command immediately"""
        if not self.sock:
            if self.connect_attempts >= self.max_connect_attempts:
                return False
            if not self.connect():
                return False
        
        try:
            send_start = time.perf_counter()
            
            # Only send if command changed or significant time passed
            if (self.last_command != command or 
                time.time() - self.last_successful_send > 1.0):
                
                self.sock.sendall(command.encode())
                self.last_command = command
                self.last_successful_send = time.time()
                
                # Record response time
                response_time = time.perf_counter() - send_start
                self.response_times.append(response_time)
                
                self.commands_sent += 1
                logging.debug(f"📡 Sent to ESP32: {command.strip()}")
            
            return True
            
        except socket.timeout:
            logging.error("💥 ESP32 Send Timeout")
            self._handle_connection_error()
            return False
            
        except socket.error as e:
            logging.error(f"💥 ESP32 Socket Error: {e}")
            self._handle_connection_error()
            return False
            
        except Exception as e:
            logging.error(f"💥 ESP32 General Error: {e}")
            self._handle_connection_error()
            return False
    
    def _handle_connection_error(self):
        """Handle connection errors and prepare for reconnection"""
        self.connection_errors += 1
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        
        # Reset connection attempts if too many errors
        if self.connection_errors % 10 == 0:
            self.connect_attempts = 0
            logging.warning(f"🔄 Resetting ESP32 connection after {self.connection_errors} errors")
    
    def get_stats(self) -> Dict:
        """Get communication statistics"""
        avg_response_time = np.mean(list(self.response_times)) if self.response_times else 0
        
        return {
            'connected': self.sock is not None,
            'commands_sent': self.commands_sent,
            'connection_errors': self.connection_errors,
            'avg_response_time_ms': avg_response_time * 1000,
            'queue_size': len(self.command_queue),
            'last_successful_send': self.last_successful_send
        }
    
    def close(self):
        """Close connection with proper cleanup"""
        if self.sock:
            try:
                # Send stop command before closing
                stop_command = "H:FORWARD\n"
                logging.info(f"Sending STOP command to ESP32: {stop_command.strip()}")
                self.sock.sendall(stop_command.encode())
                time.sleep(0.1)
            except Exception as e:
                logging.error(f"⚠️ Error sending stop command: {e}")
            finally:
                try:
                    self.sock.close()
                except:
                    pass
                self.sock = None
                logging.info("🔌 ESP32 socket closed.")

# -----------------------------------------------------------------------------
# --- DATA LOGGING SYSTEM ---
# -----------------------------------------------------------------------------

class DataLogger:
    """Log robot data for analysis and debugging"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.enabled = config.enable_data_logging
        self.log_file = None
        self.data_buffer = deque(maxlen=100)
        self.last_flush = time.time()
        
        if self.enabled:
            self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging file and format"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_filename = f"robot_data_{timestamp}.csv"
            self.log_file = open(log_filename, 'w')
            
            # Write header
            header = ("timestamp,frame_count,offset,angle,confidence,steering_output,"
                     "speed_cmd,turn_cmd,processing_time_ms,fps,line_count,"
                     "robot_status,esp32_connected\n")
            self.log_file.write(header)
            
            logging.info(f"📊 Data logging started: {log_filename}")
            
        except Exception as e:
            logging.error(f"❌ Failed to setup data logging: {e}")
            self.enabled = False
    
    def log_data(self, data: Dict):
        """Log robot data point"""
        if not self.enabled:
            return
        
        try:
            # Add timestamp
            data['timestamp'] = time.time()
            
            # Buffer the data
            self.data_buffer.append(data)
            
            # Flush buffer periodically
            if time.time() - self.last_flush > 5.0:
                self._flush_buffer()
        
        except Exception as e:
            logging.error(f"❌ Data logging error: {e}")
    
    def _flush_buffer(self):
        """Flush buffered data to file"""
        if not self.log_file or not self.data_buffer:
            return
        
        try:
            while self.data_buffer:
                data = self.data_buffer.popleft()
                
                # Format data row
                row = (f"{data.get('timestamp', 0):.3f},"
                      f"{data.get('frame_count', 0)},"
                      f"{data.get('offset', 0):.4f},"
                      f"{data.get('angle', 0):.2f},"
                      f"{data.get('confidence', 0):.3f},"
                      f"{data.get('steering_output', 0):.4f},"
                      f"{data.get('speed_cmd', 'H')},"
                      f"{data.get('turn_cmd', 'FORWARD')},"
                      f"{data.get('processing_time_ms', 0):.2f},"
                      f"{data.get('fps', 0):.1f},"
                      f"{data.get('line_count', 0)},"
                      f"{data.get('robot_status', 'Unknown')},"
                      f"{data.get('esp32_connected', False)}\n")
                
                self.log_file.write(row)
            
            self.log_file.flush()
            self.last_flush = time.time()
            
        except Exception as e:
            logging.error(f"❌ Error flushing data buffer: {e}")
    
    def close(self):
        """Close logging and flush remaining data"""
        if self.enabled and self.log_file:
            self._flush_buffer()
            self.log_file.close()
            logging.info("📊 Data logging closed")

# -----------------------------------------------------------------------------
# --- ENHANCED VISUAL INTERFACE ---
# -----------------------------------------------------------------------------

def draw_enhanced_overlays(display_frame: np.ndarray, roi_zones: List[Dict], 
                          detected_lines: List, offset: Optional[float], 
                          confidence: float, kalman_prediction: Optional[Tuple] = None):
    """Draw enhanced visual overlays"""
    h, w = display_frame.shape[:2]
    center_x = w // 2
    
    # Draw ROI zones with confidence-based coloring
    for i, zone_cfg in enumerate(roi_zones):
        # Color based on confidence
        if confidence > 0.7:
            color = (0, 255, 0)  # Green for high confidence
        elif confidence > 0.4:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (0, 0, 255)  # Red for low confidence
            
        abs_roi_top = int(h * zone_cfg["top_offset"])
        abs_roi_height = int(h * zone_cfg["height_ratio"])
        
        # Draw zone rectangle
        cv2.rectangle(display_frame, (5, abs_roi_top), 
                     (w-5, abs_roi_top + abs_roi_height), color, 2)
        
        # Draw zone label with weight
        label = f"ROI{i+1} (w:{zone_cfg['weight']:.1f})"
        cv2.putText(display_frame, label, (10, abs_roi_top + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw detected lines with thickness based on score
    if detected_lines:
        for line_info in detected_lines:
            x1, y1, x2, y2 = line_info['coords']
            roi_abs_top = line_info['roi_top_offset']
            y1a, y2a = y1 + roi_abs_top, y2 + roi_abs_top
            
            # Line thickness based on score
            thickness = max(2, min(6, int(line_info.get('score', 100) / 50)))
            
            cv2.line(display_frame, (int(x1), int(y1a)), 
                    (int(x2), int(y2a)), (255, 255, 255), thickness + 2)
            cv2.line(display_frame, (int(x1), int(y1a)), 
                    (int(x2), int(y2a)), (0, 255, 255), thickness)
    
    # Draw center line
    cv2.line(display_frame, (center_x, 0), (center_x, h), (200, 200, 200), 1, cv2.LINE_AA)
    
    # Draw target point and prediction
    if offset is not None and roi_zones:
        primary_roi_top = int(h * roi_zones[0]["top_offset"])
        primary_roi_height = int(h * roi_zones[0]["height_ratio"])
        primary_roi_cy = primary_roi_top + primary_roi_height // 2
        
        # Current target
        target_x = int(center_x + (offset * (w / 4)))
        cv2.circle(display_frame, (target_x, primary_roi_cy), 8, (0, 0, 255), -1)
        cv2.circle(display_frame, (target_x, primary_roi_cy), 10, (255, 255, 255), 2)
        
        # Connection line
        cv2.line(display_frame, (center_x, primary_roi_cy), 
                (target_x, primary_roi_cy), (255, 0, 255), 2)
        
        # Kalman prediction (if available)
        if kalman_prediction:
            pred_offset, pred_angle = kalman_prediction
            pred_x = int(center_x + (pred_offset * (w / 4)))
            cv2.circle(display_frame, (pred_x, primary_roi_cy - 20), 6, (255, 128, 0), -1)
            cv2.putText(display_frame, "PRED", (pred_x - 15, primary_roi_cy - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 128, 0), 1)

def draw_enhanced_status_panel(display_frame: np.ndarray, robot_status: str, 
                              performance_stats: Dict, esp32_stats: Dict,
                              confidence: float, adaptive_pid_kp: float):
    """Draw enhanced status panel with more information"""
    h, w = display_frame.shape[:2]
    panel_width, panel_height = 280, 200
    
    # Create semi-transparent overlay
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (5, 5), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, display_frame, 0.2, 0, display_frame)
    
    # Panel border
    cv2.rectangle(display_frame, (5, 5), (panel_width, panel_height), (0, 255, 255), 2)
    
    y, line_height = 25, 18
    
    # Status with color coding
    status_color = (0, 255, 0) if "Follow" in robot_status else \
                  ((255, 0, 0) if "Error" in robot_status else (255, 255, 0))
    cv2.putText(display_frame, f"Status: {robot_status[:25]}", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
    y += line_height
    
    # Performance metrics
    cv2.putText(display_frame, f"FPS: {performance_stats.get('avg_fps', 0):.1f}", 
               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    y += line_height
    
    cv2.putText(display_frame, f"Proc: {performance_stats.get('avg_processing_time', 0)*1000:.1f}ms", 
               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    y += line_height
    
    cv2.putText(display_frame, f"Confidence: {confidence:.3f}", 
               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    y += line_height
    
    # Adaptive PID info
    cv2.putText(display_frame, f"PID Kp: {adaptive_pid_kp:.3f}", 
               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 255), 1)
    y += line_height
    
    # ESP32 status
    esp32_color = (0, 255, 0) if esp32_stats.get('connected', False) else (0, 0, 255)
    cv2.putText(display_frame, f"ESP32: {'OK' if esp32_stats.get('connected', False) else 'ERR'}", 
               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, esp32_color, 1)
    y += line_height
    
    cv2.putText(display_frame, f"Cmds: {esp32_stats.get('commands_sent', 0)}", 
               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y += line_height
    
    # Response time
    cv2.putText(display_frame, f"Resp: {esp32_stats.get('avg_response_time_ms', 0):.1f}ms", 
               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

# To be continued with Flask app and main function... 