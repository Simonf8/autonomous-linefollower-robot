#!/usr/bin/env python3

import cv2
import numpy as np
import socket
import time
import logging
from ultralytics import YOLO
from flask import Flask, render_template_string, Response
from flask_socketio import SocketIO
import threading
import base64
import os
import pygame
import requests
import json
from pathlib import Path

# HTML template for web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Line Follower Robot Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="/static/dashboard.css">
</head>
<body>
    <div class="container">
        <div class="video-feed">
            <h2>Camera Feed</h2>
            <img id="camera-feed" src="" alt="Robot Camera Feed">
            <canvas id="path-canvas" class="path-canvas" width="600" height="400"></canvas>
        </div>
        
        <div class="status-panel">
            <h2>Robot Status</h2>
            
            <div id="line-status" class="status-item">
                <div class="status-indicator inactive"></div>
                Line Status: Not Detected
            </div>
            
            <div class="status-item">
                <h3>Position</h3>
                <div class="position-grid">
                    <div class="position-item">
                        <div class="position-label">X</div>
                        <div class="position-value" id="position-x">0.00</div>
                    </div>
                    <div class="position-item">
                        <div class="position-label">Y</div>
                        <div class="position-value" id="position-y">0.00</div>
                    </div>
                    <div class="position-item">
                        <div class="position-label">Angle</div>
                        <div class="position-value" id="position-theta">0.00</div>
                    </div>
                </div>
            </div>
            
            <div id="command" class="status-item">
                Command: None
            </div>
            
            <div class="status-item">
                Objects Detected: <span id="objects-count">0</span>
            </div>
            
            <div class="status-item">
                Uptime: <span id="uptime">00:00:00</span>
            </div>
            
            <div class="status-item">
                <h3>Performance Metrics</h3>
                <canvas id="metrics-chart" width="300" height="200"></canvas>
            </div>
            
            <div class="controls-info">
                <h3>Manual Controls</h3>
                <p>W - Forward</p>
                <p>A - Left</p>
                <p>D - Right</p>
                <p>S - Stop</p>
                <p>Q - Turn Around</p>
            </div>
        </div>
    </div>
    <script src="/static/dashboard.js"></script>
</body>
</html>
'''

class ESP32Interface:
    """Handles communication with ESP32"""
    
    VALID_COMMANDS = [
        'FORWARD', 'LEFT', 'RIGHT', 'STOP', 'BACKWARD',
        'SLIGHT_LEFT', 'SLIGHT_RIGHT', 'TURN_AROUND',
        'EMERGENCY_LEFT', 'EMERGENCY_RIGHT'
    ]
    
    def __init__(self, ip_address, port=1234):
        self.ip_address = ip_address
        self.port = port
        self.socket = None
        self.connected = False
        self.logger = logging.getLogger(__name__)
        self.line_position = 0
        self.line_detected = False
    
    def connect(self):
        """Establish connection to ESP32"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.ip_address, self.port))
            self.connected = True
            self.logger.info(f"Connected to ESP32 at {self.ip_address}:{self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to ESP32: {e}")
            self.connected = False
            return False
    
    def reconnect(self):
        """Attempt to reconnect to ESP32"""
        self.logger.info("Attempting to reconnect to ESP32...")
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.socket = None
        return self.connect()
    
    def send_command(self, command):
        """Send command to ESP32 and receive line sensor data"""
        if not command in self.VALID_COMMANDS:
            self.logger.error(f"Invalid command: {command}")
            return False
        
        if not self.connected:
            if not self.reconnect():
                return False
        
        try:
            # Send command
            self.socket.send(command.encode('utf-8'))
            
            # Receive line sensor data
            data = self.socket.recv(64).decode('utf-8').strip()
            if data:
                try:
                    position, detected = map(float, data.split(','))
                    self.line_position = position
                    self.line_detected = bool(detected)
                    print(f"ESP32 Data - Position: {position:.2f}, Detected: {detected}")
                except:
                    self.logger.error("Failed to parse line sensor data")
                    print(f"ESP32 Raw Data: '{data}'")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to communicate with ESP32: {e}")
            self.connected = False
            return False
    
    def close(self):
        """Close connection to ESP32"""
        if self.socket:
            try:
                self.send_command('STOP')
                self.socket.close()
            except:
                pass
        self.socket = None
        self.connected = False

class PIDController:
    """Advanced PID controller for smooth line following"""
    
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, output_limits=(-1.0, 1.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
    
    def update(self, error):
        """Update PID controller with new error value"""
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0.0:
            return 0.0
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term with windup protection
        self.integral += error * dt
        self.integral = max(min(self.integral, 10.0), -10.0)  # Clamp integral
        integral = self.ki * self.integral
        
        # Derivative term
        derivative = self.kd * (error - self.last_error) / dt
        
        # Calculate output
        output = proportional + integral + derivative
        
        # Apply output limits
        output = max(min(output, self.output_limits[1]), self.output_limits[0])
        
        # Update for next iteration
        self.last_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        """Reset PID controller state"""
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

class ScaleEstimator:
    """Advanced scale estimation using multiple methods"""
    
    def __init__(self):
        self.scale_history = []
        self.ground_plane_height = 1.2  # Camera height in meters
        
    def estimate_scale(self, pts1, pts2, R, t, camera_matrix):
        """Estimate scale using multiple methods and fusion"""
        # Method 1: Ground plane assumption
        scale_ground = self._estimate_scale_ground_plane(pts1, pts2, camera_matrix)
        
        # Method 2: Feature displacement analysis
        scale_displacement = self._estimate_scale_displacement(pts1, pts2)
        
        # Method 3: Historical scale smoothing
        scale_smoothed = self._smooth_scale_estimate(scale_ground, scale_displacement)
        
        return scale_smoothed
    
    def _estimate_scale_ground_plane(self, pts1, pts2, camera_matrix):
        """Estimate scale assuming features are on ground plane"""
        # Calculate average feature height in image
        avg_y1 = np.mean(pts1[:, 1])
        avg_y2 = np.mean(pts2[:, 1])
        
        # Convert to normalized coordinates
        cy = camera_matrix[1, 2]
        fy = camera_matrix[1, 1]
        
        # Estimate distance to ground plane
        distance = self.ground_plane_height * fy / (avg_y1 - cy)
        
        # Scale based on expected movement
        expected_movement = 0.1  # meters per frame at normal speed
        scale = expected_movement / max(np.linalg.norm(pts2 - pts1, axis=1).mean(), 1.0)
        
        return max(0.01, min(scale, 2.0))  # Clamp scale
    
    def _estimate_scale_displacement(self, pts1, pts2):
        """Estimate scale based on feature displacement patterns"""
        displacements = np.linalg.norm(pts2 - pts1, axis=1)
        median_displacement = np.median(displacements)
        
        # Empirical scale based on typical robot movement
        scale = 0.05 / max(median_displacement, 1.0)
        return max(0.01, min(scale, 2.0))
    
    def _smooth_scale_estimate(self, scale1, scale2):
        """Smooth scale estimates using history"""
        # Weighted average of methods
        current_scale = 0.6 * scale1 + 0.4 * scale2
        
        # Add to history
        self.scale_history.append(current_scale)
        if len(self.scale_history) > 10:
            self.scale_history.pop(0)
        
        # Return smoothed scale
        if len(self.scale_history) > 3:
            return np.median(self.scale_history[-5:])  # Median of recent estimates
        else:
            return current_scale

class LineSearchStrategy:
    """Intelligent line search using memory and prediction"""
    
    def __init__(self):
        self.search_state = "spiral"  # spiral, backtrack, predict
        self.search_start_time = time.time()
        self.search_direction = 1  # 1 for right, -1 for left
        self.last_known_line_position = None
        self.search_attempts = 0
    
    def get_search_command(self, current_position, path_history, tracking_quality):
        """Get intelligent search command based on robot state"""
        search_time = time.time() - self.search_start_time
        
        # Strategy 1: Quick spiral search (first 3 seconds)
        if search_time < 3.0:
            return self._spiral_search()
        
        # Strategy 2: Backtrack to last known good position (3-8 seconds)
        elif search_time < 8.0 and len(path_history) > 5:
            return self._backtrack_search(current_position, path_history)
        
        # Strategy 3: Predictive search based on path pattern (8+ seconds)
        else:
            return self._predictive_search(path_history)
    
    def _spiral_search(self):
        """Execute spiral search pattern"""
        # Alternate between left and right with increasing duration
        cycle_time = (time.time() - self.search_start_time) % 2.0
        
        if cycle_time < 1.0:
            return "RIGHT" if self.search_direction > 0 else "LEFT"
        else:
            self.search_direction *= -1  # Switch direction
            return "LEFT" if self.search_direction > 0 else "RIGHT"
    
    def _backtrack_search(self, current_position, path_history):
        """Backtrack to last known good position"""
        if len(path_history) < 2:
            return "BACKWARD"
        
        # Calculate direction to previous position
        last_pos = path_history[-5]  # Go back 5 positions
        current_pos = current_position[:2]  # x, y only
        
        dx = last_pos[0] - current_pos[0]
        dy = last_pos[1] - current_pos[1]
        
        # Simple direction calculation
        if abs(dx) > abs(dy):
            return "RIGHT" if dx > 0 else "LEFT"
        else:
            return "FORWARD" if dy > 0 else "BACKWARD"
    
    def _predictive_search(self, path_history):
        """Predict line direction based on path pattern"""
        if len(path_history) < 10:
            return "FORWARD"  # Default forward search
        
        # Analyze path curvature to predict line direction
        recent_path = path_history[-10:]
        
        # Calculate average direction change
        direction_changes = []
        for i in range(1, len(recent_path)):
            dx = recent_path[i][0] - recent_path[i-1][0]
            dy = recent_path[i][1] - recent_path[i-1][1]
            angle = np.arctan2(dy, dx)
            direction_changes.append(angle)
        
        if len(direction_changes) > 1:
            avg_curve = np.mean(np.diff(direction_changes))
            if avg_curve > 0.1:
                return "RIGHT"  # Path was curving right
            elif avg_curve < -0.1:
                return "LEFT"   # Path was curving left
        
        return "FORWARD"  # Continue straight
    
    def reset(self):
        """Reset search strategy"""
        self.search_state = "spiral"
        self.search_start_time = time.time()
        self.search_direction = 1
        self.search_attempts = 0

class VisualOdometry:
    """Advanced visual odometry with scale estimation and drift correction"""
    
    def __init__(self, camera_matrix=None):
        if camera_matrix is None:
            self.camera_matrix = np.array([
                [500, 0, 320],
                [0, 500, 240],
                [0, 0, 1]
            ])
        else:
            self.camera_matrix = camera_matrix
            
        # Initialize feature detector
        self.orb = cv2.ORB_create(
            nfeatures=1000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            patchSize=31,
            fastThreshold=20
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # State variables
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None
        self.position = np.zeros(3)  # [x, y, theta]
        self.tracking_quality = 1.0
    
    def process_frame(self, frame):
        """Advanced frame processing with scale estimation and drift correction"""
        # Convert to grayscale and apply preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)  # Reduce noise
        
        # Detect features with adaptive thresholding
        kp, des = self.orb.detectAndCompute(gray, None)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            self.scale_estimator = ScaleEstimator()
            return self.position, 1.0
        
        # Advanced feature matching with ratio test
        if des is not None and self.prev_des is not None:
            matches = self._robust_feature_matching(self.prev_des, des)
            
            if len(matches) > 12:  # Increased threshold for better accuracy
                # Get matched keypoints
                pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp[m.trainIdx].pt for m in matches])
                
                # Calculate Essential matrix with improved RANSAC
                E, mask = cv2.findEssentialMat(
                    pts1, pts2, self.camera_matrix,
                    method=cv2.RANSAC,
                    prob=0.9999,
                    threshold=0.5,
                    maxIters=1000
                )
                
                if E is not None and mask is not None:
                    # Filter matches using RANSAC mask
                    good_pts1 = pts1[mask.ravel() == 1]
                    good_pts2 = pts2[mask.ravel() == 1]
                    
                    if len(good_pts1) > 8:
                        # Recover pose
                        _, R, t, pose_mask = cv2.recoverPose(
                            E, good_pts1, good_pts2, self.camera_matrix
                        )
                        
                        # Advanced scale estimation
                        scale = self.scale_estimator.estimate_scale(
                            good_pts1, good_pts2, R, t, self.camera_matrix
                        )
                        
                        # Update position with drift correction
                        self._update_position_with_correction(R, t, scale)
                        
                        # Update tracking quality with multiple factors
                        self.tracking_quality = self._calculate_tracking_quality(
                            len(good_pts1), np.mean(mask), scale
                        )
        
        # Update previous frame data
        self.prev_frame = gray
        self.prev_kp = kp
        self.prev_des = des
        
        return self.position, self.tracking_quality
    
    def _robust_feature_matching(self, des1, des2):
        """Robust feature matching with ratio test and cross-check"""
        # Use BF matcher for reliability (avoid FLANN crashes)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            matches = bf.knnMatch(des1, des2, k=2)
        except cv2.error:
            # Fallback to simple matching if knn fails
            matches = [[m] for m in bf.match(des1, des2)]
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:  # Lowe's ratio test
                    good_matches.append(m)
        
        return good_matches
    
    def _update_position_with_correction(self, R, t, scale):
        """Update position with drift correction and smoothing"""
        # Apply scale to translation
        scaled_t = t * scale
        
        # Convert rotation matrix to euler angle
        rotation_angle = np.arctan2(R[1, 0], R[0, 0])
        
        # Apply exponential smoothing to reduce noise
        alpha = 0.7  # Smoothing factor
        self.position[0] = alpha * (self.position[0] + scaled_t[0]) + (1-alpha) * self.position[0]
        self.position[1] = alpha * (self.position[1] + scaled_t[1]) + (1-alpha) * self.position[1]
        self.position[2] = alpha * rotation_angle + (1-alpha) * self.position[2]
        
        # Drift correction using IMU-like approach (if available)
        if hasattr(self, 'drift_corrector'):
            self.position = self.drift_corrector.correct(self.position)
    
    def _calculate_tracking_quality(self, num_matches, inlier_ratio, scale):
        """Calculate comprehensive tracking quality metric"""
        # Multiple quality factors
        match_quality = min(num_matches / 100.0, 1.0)  # Based on number of matches
        inlier_quality = inlier_ratio  # Based on RANSAC inliers
        scale_quality = 1.0 - min(abs(scale - 1.0), 1.0)  # Penalize extreme scales
        
        # Weighted combination
        overall_quality = (0.4 * match_quality + 
                          0.4 * inlier_quality + 
                          0.2 * scale_quality)
        
        return max(0.0, min(overall_quality, 1.0))
    
    def draw_debug(self, frame):
        """Draw debug visualization"""
        if self.prev_kp is not None and len(self.prev_kp) > 0:
            # Only draw top 20 best features to reduce clutter
            sorted_kp = sorted(self.prev_kp, key=lambda x: x.response, reverse=True)[:20]
            cv2.drawKeypoints(
                frame,
                sorted_kp,
                frame,
                color=(0, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
            )
        
        # Draw position and heading
        height, width = frame.shape[:2]
        center = (width // 2, height // 2)
        
        # Draw robot position
        cv2.circle(frame, center, 5, (0, 255, 0), -1)
        
        # Draw heading arrow
        arrow_length = 50
        end_point = (
            int(center[0] + arrow_length * np.cos(self.position[2])),
            int(center[1] + arrow_length * np.sin(self.position[2]))
        )
        cv2.arrowedLine(frame, center, end_point, (0, 255, 0), 2)

class VoiceSystem:
    """Robust voice system with multiple TTS fallbacks"""
    
    def __init__(self):
        # Initialize pygame mixer for audio playback
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            print("Audio system initialized successfully")
        except Exception as e:
            print(f"Audio system failed: {e}")
        
        self.cache_dir = Path("voice_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.is_playing = False
        self.tts_method = None
        
        # Try to initialize TTS systems in order of preference
        if self._init_bark():
            print("Bark TTS system ready!")
        elif self._init_pyttsx3():
            print("pyttsx3 TTS system ready!")
        elif self._init_espeak():
            print("espeak TTS system ready!")
        else:
            print("No TTS system available - voice disabled")
        
        # Voice messages
        self.voices = {
            'obstacle': "Obstacle detected. Initiating turn around maneuver.",
            'line_lost': "Line signal lost. Activating search protocol.",
            'line_found': "Line detected. Resuming navigation control.",
            'startup': "Autonomous navigation system activated.",
            'turn_complete': "Avoidance maneuver complete. Resuming line following.",
            'object_detected': "Critical obstacle detected. Executing emergency avoidance.",
            'tracking_lost': "Visual tracking degraded. Switching to backup navigation.",
            'high_precision': "High precision mode engaged."
        }
        
        self.audio_cache = {}
    
    def _init_bark(self):
        """Try to initialize Bark TTS"""
        try:
            from bark import SAMPLE_RATE, generate_audio, preload_models
            from scipy.io.wavfile import write as write_wav
            
            self.SAMPLE_RATE = SAMPLE_RATE
            self.generate_audio = generate_audio
            self.write_wav = write_wav
            self.tts_method = 'bark'
            
            # Preload models (optional - comment out for faster startup)
            # preload_models()
            return True
        except Exception as e:
            print(f"Bark TTS not available: {e}")
            return False
    
    def _init_pyttsx3(self):
        """Try to initialize pyttsx3 TTS"""
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.8)
            self.tts_method = 'pyttsx3'
            return True
        except Exception as e:
            print(f"pyttsx3 TTS not available: {e}")
            return False
    
    def _init_espeak(self):
        """Try to initialize espeak TTS"""
        try:
            import subprocess
            result = subprocess.run(['espeak', '--version'], capture_output=True, check=True)
            self.tts_method = 'espeak'
            return True
        except Exception as e:
            print(f"espeak TTS not available: {e}")
            return False
        
    def _get_cached_audio(self, text, voice_preset):
        """Get cached audio file or generate new one"""
        cache_key = f"{text}_{voice_preset}"
        cache_file = self.cache_dir / f"{abs(hash(cache_key))}.wav"
        
        if cache_file.exists():
            return str(cache_file)
            
        # Generate audio using Bark
        try:
            print(f"Generating voice: {text[:50]}...")
            
            # Generate audio with Bark
            audio_array = self.generate_audio(text, history_prompt=voice_preset)
            
            # Save to file
            self.write_wav(str(cache_file), self.SAMPLE_RATE, audio_array)
            print(f"Voice generated and cached: {cache_file.name}")
            return str(cache_file)
                
        except Exception as e:
            logging.error(f"Failed to generate audio with Bark: {e}")
            return None
    
    def play_sound(self, event_or_text):
        """Play a sound for a specific event or custom text"""
        if self.is_playing or not self.tts_method:
            return  # Don't interrupt current playback or if no TTS
        
        # Handle both predefined events and custom text
        if event_or_text in self.voices:
            text = self.voices[event_or_text]
        else:
            text = str(event_or_text)  # Use the text directly
        
        try:
            self.is_playing = True
            
            if self.tts_method == 'bark':
                self._play_bark_tts(text)
            elif self.tts_method == 'pyttsx3':
                self._play_pyttsx3_tts(text)
            elif self.tts_method == 'espeak':
                self._play_espeak_tts(text)
                
        except Exception as e:
            print(f"Failed to play TTS audio: {e}")
            self.is_playing = False
    
    def _play_bark_tts(self, text):
        """Play TTS using Bark"""
        try:
            audio_array = self.generate_audio(text, history_prompt="v2/en_speaker_6")
            temp_file = self.cache_dir / f"temp_{int(time.time())}.wav"
            self.write_wav(str(temp_file), self.SAMPLE_RATE, audio_array)
            
            pygame.mixer.music.load(str(temp_file))
            pygame.mixer.music.play()
            
            def cleanup():
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                self.is_playing = False
                temp_file.unlink(missing_ok=True)
            
            threading.Thread(target=cleanup, daemon=True).start()
        except Exception as e:
            print(f"Bark TTS failed: {e}")
            self.is_playing = False
    
    def _play_pyttsx3_tts(self, text):
        """Play TTS using pyttsx3"""
        try:
            def speak():
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                self.is_playing = False
            
            threading.Thread(target=speak, daemon=True).start()
        except Exception as e:
            print(f"pyttsx3 TTS failed: {e}")
            self.is_playing = False
    
    def _play_espeak_tts(self, text):
        """Play TTS using espeak"""
        try:
            def speak():
                import subprocess
                subprocess.run(['espeak', text], check=True)
                self.is_playing = False
            
            threading.Thread(target=speak, daemon=True).start()
        except Exception as e:
            print(f"espeak TTS failed: {e}")
            self.is_playing = False
    
    def add_custom_voice(self, event_name, text, voice_preset="v2/en_speaker_6"):
        """Add a custom voice line"""
        self.voices[event_name] = {
            'text': text,
            'voice_preset': voice_preset
        }

class Robot:
    """Main robot control class"""
    
    def __init__(self, esp32_ip, esp32_port=1234):
        # Initialize Flask app and SocketIO
        self.app = Flask(__name__, static_folder='../static')
        self.socketio = SocketIO(self.app)
        self.app.route('/')(self.index)
        
        # Add manual command handler
        @self.socketio.on('manual_command')
        def handle_manual_command(command):
            if command in self.esp32.VALID_COMMANDS:
                self.send_command(command)
                logging.info(f"Manual command executed: {command}")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize components
        self.esp32 = ESP32Interface(esp32_ip, esp32_port)
        self.vo = VisualOdometry()
        self.yolo = YOLO("yolo11n.pt")  # Use YOLO11n model
        
        # Initialize voice system (optional)
        try:
            self.voice = VoiceSystem()
        except ImportError as e:
            logging.warning(f"Voice system disabled: {e}")
            self.voice = None
        
        # State variables
        self.position = np.zeros(3)
        self.tracking_quality = 1.0
        self.detected_objects = []
        self.last_command = None
        self.last_command_time = time.time()
        self.line_status = False  # Track if line was previously detected
        self.tracking_status = True  # Track if visual tracking is good
        self.precision_mode = False  # Track if in high precision mode
        
        # Path tracking
        self.path_history = []
        
        # Start web server thread
        self.web_thread = threading.Thread(target=self._run_web_server)
        self.web_thread.daemon = True
        self.web_thread.start()
        
        # Play startup sound
        if self.voice:
            self.voice.play_sound('startup')
    
    def index(self):
        """Serve the web interface"""
        return render_template_string(HTML_TEMPLATE)
    
    def _run_web_server(self):
        """Run the web server in a separate thread"""
        self.socketio.run(self.app, host='0.0.0.0', port=5000)
    
    def _encode_frame(self, frame):
        """Convert OpenCV frame to base64 for web display"""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    
    def _emit_update(self, frame):
        """Emit updates to web clients"""
        try:
            # Limit update frequency to prevent payload issues
            current_time = time.time()
            if not hasattr(self, '_last_emit_time'):
                self._last_emit_time = 0
            
            if current_time - self._last_emit_time < 0.2:  # 5 FPS max
                return
                
            self._last_emit_time = current_time
            
            frame_data = self._encode_frame(frame)
            self.socketio.emit('robot_update', {
                'frame': frame_data,
                'data': {
                    'position': self.position.tolist(),
                    'line_position': self.esp32.line_position,
                    'line_detected': self.esp32.line_detected,
                    'detected_objects': len(self.detected_objects),  # Just count, not full data
                    'last_command': self.last_command,
                    'tracking_quality': self.tracking_quality
                }
            })
        except Exception as e:
            logging.error(f"Failed to emit update: {e}")
    
    def process_frame(self, frame):
        """Process a single camera frame"""
        # 1. Update position using visual odometry
        self.position, self.tracking_quality = self.vo.process_frame(frame)
        
        # Monitor tracking quality and provide feedback
        if self.tracking_quality > 0.8 and not self.precision_mode:
            if self.voice:
                self.voice.play_sound('high_precision')
            self.precision_mode = True
        elif self.tracking_quality < 0.3 and self.tracking_status:
            if self.voice:
                self.voice.play_sound('tracking_lost')
            self.tracking_status = False
        elif self.tracking_quality > 0.5:
            self.tracking_status = True
            self.precision_mode = False
        
        if self.tracking_quality > 0.5:
            self.path_history.append((self.position[0], self.position[1]))
            if len(self.path_history) > 100:
                self.path_history.pop(0)
        
        # 2. Detect objects using YOLO
        results = self.yolo(frame)
        self.detected_objects = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = box.cls[0]
                
                if conf > 0.5:
                    self.detected_objects.append({
                        'class': int(cls),
                        'confidence': float(conf),
                        'box': (int(x1), int(y1), int(x2), int(y2))
                    })
        
        # 3. Draw visualizations
        self.draw_visualizations(frame)
        
        # 4. Emit update to web clients
        self._emit_update(frame)
        
        return frame
    
    def draw_visualizations(self, frame):
        """Draw debug visualizations"""
        # Draw visual odometry debug info
        self.vo.draw_debug(frame)
        
        # Draw path history (disabled to reduce visual clutter)
        # if len(self.path_history) > 1:
        #     points = np.array(self.path_history, dtype=np.int32)
        #     cv2.polylines(frame, [points], False, (0, 255, 255), 2)
        
        # Draw detected objects
        for obj in self.detected_objects:
            x1, y1, x2, y2 = obj['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Class {obj['class']}: {obj['confidence']:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        # Draw line following status
        cv2.putText(
            frame,
            f"Line: {'Detected' if self.esp32.line_detected else 'Lost'} "
            f"Position: {self.esp32.line_position:.2f}",
            (10, frame.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
        
        # Draw current command
        if self.last_command:
            cv2.putText(
                frame,
                f"Command: {self.last_command}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
    
    def calculate_control(self):
        """Advanced control with PID, obstacle avoidance, and adaptive behavior"""
        # Check for obstacles first - priority over everything else
        obstacle_result = self._analyze_obstacles_advanced()
        if obstacle_result and obstacle_result.get('action') == 'TURN_AROUND':
            return obstacle_result['action']
        
        # Adaptive PID line following when line is detected
        if self.esp32.line_detected:
            if not self.line_status:
                if self.voice:
                    self.voice.play_sound('line_found')
                self.line_status = True
            return self._pid_line_following()
        else:
            # Line lost - handle search behavior
            if self.line_status:
                if self.voice:
                    self.voice.play_sound('line_lost')
                self.line_status = False
            return self._handle_line_loss()  # Use improved line loss handling
    
    def _analyze_obstacles_advanced(self):
        """Enhanced obstacle detection with 180° turn on object detection"""
        if not self.detected_objects:
            return None
            
        # Initialize object detection state tracking
        if not hasattr(self, 'last_object_time'):
            self.last_object_time = 0
            
        current_time = time.time()
        
        for obj in self.detected_objects:
            x1, y1, x2, y2 = obj['box']
            width, height = x2 - x1, y2 - y1
            center_x = (x1 + x2) / 2
            
            # Estimate distance using object size
            estimated_distance = self._estimate_distance(width, height, obj['class'])
            
            # Check if object is in collision path
            robot_path_width = 120  # pixels - wider detection zone
            frame_center = 320  # assuming 640px width
            
            if (abs(center_x - frame_center) < robot_path_width and 
                estimated_distance < 2.0 and  # Increased detection distance
                y2 > 150 and  # Object is in relevant part of frame
                current_time - self.last_object_time > 5.0):  # Avoid repeated detections
                
                # Get object class name for voice
                class_names = {0: 'person', 1: 'bicycle', 2: 'car', 15: 'cat', 16: 'dog'}
                obj_name = class_names.get(obj['class'], f'object class {obj["class"]}')
                
                # Voice announcement
                if hasattr(self, 'voice') and self.voice:
                    self.voice.play_sound(f"Obstacle detected: {obj_name}. Performing 180 degree turn.")
                
                print(f"OBSTACLE DETECTED: {obj_name} at distance {estimated_distance:.2f}m - TURNING AROUND")
                self.last_object_time = current_time
                
                return {
                    'action': 'TURN_AROUND',
                    'distance': estimated_distance,
                    'position': center_x,
                    'object_type': obj_name,
                    'confidence': obj['confidence']
                }
        return None
    
    def _estimate_distance(self, width, height, obj_class):
        """Estimate distance to object based on size and class"""
        # Known object sizes (in meters) - this should be calibrated
        object_sizes = {
            0: 0.6,   # person
            1: 0.3,   # bicycle wheel
            2: 1.8,   # car width
            # Add more object classes as needed
        }
        
        if obj_class in object_sizes:
            real_size = object_sizes[obj_class]
            focal_length = 500  # camera focal length in pixels
            distance = (real_size * focal_length) / width
            return max(0.1, min(distance, 10.0))  # clamp between 0.1-10m
        
        # Fallback: use empirical size-distance relationship
        return max(0.5, 5.0 - (width / 100))
    
    def _execute_smart_avoidance(self, obstacle):
        """Execute intelligent obstacle avoidance maneuver"""
        if self.voice:
            self.voice.play_sound('object_detected')
        
        # Choose avoidance strategy based on obstacle position and distance
        if obstacle['distance'] < 0.8:  # Very close - emergency maneuver
            if obstacle['position'] < 320:  # Object on left
                return "EMERGENCY_RIGHT"
            else:  # Object on right
                return "EMERGENCY_LEFT"
        else:  # Planned avoidance
            return "TURN_AROUND"
    
    def _pid_line_following(self):
        """Responsive but smooth PID controller for line following"""
        if not hasattr(self, 'pid_controller'):
            self.pid_controller = PIDController(kp=0.7, ki=0.0, kd=0.25)  # More responsive
        
        error = self.esp32.line_position
        line_detected = self.esp32.line_detected
        
        # If no line detected, initiate search behavior
        if not line_detected:
            return self._handle_line_loss()
        
        # Reset line loss tracking when line is found
        if hasattr(self, 'line_lost_time'):
            delattr(self, 'line_lost_time')
            delattr(self, 'search_direction')
            print("LINE FOUND! Resuming normal following")
        
        # Moderate smoothing for responsiveness
        if not hasattr(self, 'error_history'):
            self.error_history = [0.0, 0.0, 0.0]  # 3-point smoothing
        
        self.error_history.append(error)
        self.error_history.pop(0)
        smooth_error = sum(self.error_history) / len(self.error_history)
        
        control_output = self.pid_controller.update(smooth_error)
        
        # Responsive but smooth control
        if abs(smooth_error) < 0.08:  # Wider center zone - less micro-adjustments
            return "FORWARD"
        elif smooth_error > 0.7:  # Sharp turns when moderately far off
            return "RIGHT" 
        elif smooth_error < -0.7:
            return "LEFT"
        elif smooth_error > 0.12:  # More responsive slight turns
            return "SLIGHT_RIGHT"
        elif smooth_error < -0.12:
            return "SLIGHT_LEFT"
        else:
            return "FORWARD"
    
    def _handle_line_loss(self):
        """Handle when robot loses the line - search by spinning"""
        current_time = time.time()
        
        # Initialize line loss tracking
        if not hasattr(self, 'line_lost_time'):
            self.line_lost_time = current_time
            # Determine search direction based on last known position
            if hasattr(self, 'error_history') and self.error_history:
                last_error = self.error_history[-1]
                self.search_direction = "RIGHT" if last_error > 0 else "LEFT"
            else:
                self.search_direction = "LEFT"  # Default search direction
            print(f"LINE LOST! Starting search in direction: {self.search_direction}")
        
        # Search for line by spinning
        search_duration = current_time - self.line_lost_time
        
        if search_duration < 3.0:  # Search for 3 seconds in one direction
            return self.search_direction
        elif search_duration < 6.0:  # Then try the other direction
            opposite_direction = "LEFT" if self.search_direction == "RIGHT" else "RIGHT"
            return opposite_direction
        else:
            # If still no line after 6 seconds, continue spinning
            return self.search_direction
    
    def _intelligent_line_search(self):
        """Intelligent line search using visual odometry and memory"""
        if not hasattr(self, 'search_strategy'):
            self.search_strategy = LineSearchStrategy()
        
        return self.search_strategy.get_search_command(
            self.position, 
            self.path_history,
            self.tracking_quality
        )
    
    def _simple_line_search(self):
        """Simple line search based on last known position"""
        last_position = self.esp32.line_position
        
        # Search in the direction where line was last seen
        if last_position > 0.3:  # Line was to the right
            return "RIGHT"
        elif last_position < -0.3:  # Line was to the left  
            return "LEFT"
        else:
            # Line was center, do a slow search pattern
            import time
            search_time = time.time() % 4  # 4 second cycle
            if search_time < 2:
                return "SLIGHT_LEFT"
            else:
                return "SLIGHT_RIGHT"
    
    def send_command(self, command):
        """Send command to ESP32 with rate limiting"""
        current_time = time.time()
        
        # Only send new commands if different from last command
        # or if more than 100ms has passed
        if (command != self.last_command or 
            current_time - self.last_command_time > 0.1):
            
            if self.esp32.send_command(command):
                # Add voice feedback for turn around completion
                if command == "TURN_AROUND" and self.last_command != "TURN_AROUND":
                    # Play turn complete sound after a delay
                    def delayed_voice():
                        time.sleep(3)  # Wait for turn to complete
                        if self.voice:
                            self.voice.play_sound('turn_complete')
                    threading.Thread(target=delayed_voice, daemon=True).start()
                
                self.last_command = command
                self.last_command_time = current_time
                logging.debug(f"Sent command: {command}")
            else:
                logging.error("Failed to send command to ESP32")
    
    def run(self):
        """Main control loop"""
        try:
            logging.info("Starting robot control loop")
            while True:
                # 1. Get camera frame
                ret, frame = self.cap.read()
                if not ret:
                    logging.error("Failed to get camera frame")
                    continue
                
                # 2. Process frame
                processed_frame = self.process_frame(frame)
                
                # 3. Calculate and send control command
                command = self.calculate_control()
                
                # Debug line sensor data with detailed analysis
                pos = self.esp32.line_position
                detected = self.esp32.line_detected
                
                if detected:
                    if pos < -0.5:
                        direction = "LINE FAR LEFT - should turn LEFT"
                    elif pos < -0.2:
                        direction = "LINE SLIGHT LEFT - should turn SLIGHT LEFT"
                    elif abs(pos) < 0.15:
                        direction = "LINE CENTERED - should go FORWARD"  
                    elif pos > 0.2:
                        direction = "LINE SLIGHT RIGHT - should turn SLIGHT RIGHT"
                    elif pos > 0.5:
                        direction = "LINE FAR RIGHT - should turn RIGHT"
                    else:
                        direction = "LINE CENTERED"
                else:
                    direction = "NO LINE DETECTED"
                    
                print(f"Position: {pos:.2f}, {direction}, Command: {command}")
                
                self.send_command(command)
                
                # 4. Display frame (disabled for headless operation)
                # cv2.imshow('Robot View', processed_frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            logging.info("Shutting down...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        logging.info("Cleaning up...")
        self.send_command("STOP")
        self.cap.release()
        cv2.destroyAllWindows()
        self.esp32.close()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create and run robot
    # Replace with your ESP32's IP address
    robot = Robot(esp32_ip="192.168.128.117")
    robot.run() 