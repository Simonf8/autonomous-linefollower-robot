#!/usr/bin/env python3

import cv2
import numpy as np
import socket
import time
import logging
from flask import Flask, render_template_string, Response, jsonify
import threading
import subprocess
import os
from pathlib import Path
import pygame

# HTML template for web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Robot Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: white;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
                }
        .camera-feed {
            width: 100%;
            max-width: 640px;
            margin: 0 auto 20px;
            display: block;
            border-radius: 8px;
            border: 2px solid #4CAF50;
        }
        .status {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .status-card {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .status-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #4CAF50;
            margin-bottom: 5px;
        }
        .status-label {
            color: #888;
            font-size: 0.9em;
        }
        .autonomous-info {
            margin-top: 20px;
            text-align: center;
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
        }
        .autonomous-info h3 {
            margin: 0 0 10px 0;
            color: #4CAF50;
        }
        .action-display {
            margin: 20px 0;
            text-align: center;
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            border: 2px solid #4CAF50;
        }
        .action-display h3 {
            margin: 0 0 10px 0;
            color: #4CAF50;
        }
        .action-text {
            font-size: 1.2em;
            font-weight: bold;
            color: #fff;
            padding: 10px;
            background: #1a1a1a;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Line Follower Robot</h1>
        </div>
        
        <img src="/video_feed" class="camera-feed" alt="Robot Camera Feed">
        
        <div class="status">
            <div class="status-card">
                <div class="status-value" id="robot-status">Running</div>
                <div class="status-label">Status</div>
            </div>
            
            <div class="status-card">
                <div class="status-value" id="command">FORWARD</div>
                <div class="status-label">Current Command</div>
            </div>
            
            <div class="status-card">
                <div class="status-value" id="line-detected">Yes</div>
                <div class="status-label">Line Detected</div>
            </div>
            
            <div class="status-card">
                <div class="status-value" id="position">0.00</div>
                <div class="status-label">Line Position</div>
            </div>
            
            <div class="status-card">
                <div class="status-value" id="obstacle">No</div>
                <div class="status-label">Obstacle Detected</div>
            </div>
            </div>
            
        <div class="action-display">
            <h3>Current Action</h3>
            <div class="action-text" id="current-action">Initializing...</div>
            </div>
            
        <div class="autonomous-info">
            <h3>Autonomous Line Following</h3>
            <p>Robot follows BLACK lines using ESP32 sensors and avoids obstacles with camera</p>
            </div>
        </div>
    
    <script>
        // Real-time status updates for autonomous robot
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update all status fields
                    document.getElementById('robot-status').textContent = data.status;
                    document.getElementById('command').textContent = data.command;
                    document.getElementById('line-detected').textContent = data.line_detected ? 'Yes' : 'No';
                    document.getElementById('obstacle').textContent = data.obstacle_detected ? 'YES!' : 'No';
                    document.getElementById('position').textContent = data.position.toFixed(2);
                    document.getElementById('current-action').textContent = data.current_action;
                    
                    // Color coding for better visibility
                    const lineCard = document.getElementById('line-detected');
                    lineCard.style.color = data.line_detected ? '#4CAF50' : '#f44336';
                    
                    const obstacleCard = document.getElementById('obstacle');
                    obstacleCard.style.color = data.obstacle_detected ? '#f44336' : '#4CAF50';
                    obstacleCard.style.fontWeight = data.obstacle_detected ? 'bold' : 'normal';
                    
                    const positionCard = document.getElementById('position');
                    const pos = data.position;
                    if (Math.abs(pos) < 0.1) {
                        positionCard.style.color = '#4CAF50'; // Green for centered
                    } else if (Math.abs(pos) < 0.5) {
                        positionCard.style.color = '#FFC107'; // Yellow for slight offset
                    } else {
                        positionCard.style.color = '#f44336'; // Red for major offset
                    }
                })
                .catch(error => {
                    console.error('Error fetching status:', error);
                    document.getElementById('robot-status').textContent = 'Connection Error';
                });
        }
        
        // Fast updates for real-time feedback
        setInterval(updateStatus, 500);
        updateStatus(); // Initial call
    </script>
</body>
</html>'''

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
            self.socket.settimeout(10.0)  # Longer timeout
            # Enable keep-alive to maintain connection
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            # Disable Nagle's algorithm for faster response
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.socket.connect((self.ip_address, self.port))
            self.connected = True
            self.logger.info(f"Connected to ESP32 at {self.ip_address}:{self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to ESP32: {e}")
            self.connected = False
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
                self.socket = None
            return False
    
    def reconnect(self):
        """Attempt to reconnect to ESP32"""
        # Don't spam reconnection attempts
        if hasattr(self, '_last_reconnect_time'):
            if time.time() - self._last_reconnect_time < 2.0:
                return False
        
        self.logger.info("Attempting to reconnect to ESP32...")
        self._last_reconnect_time = time.time()
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.socket = None
        self.connected = False
        time.sleep(1.0)  # Longer delay before reconnection
        return self.connect()
    
    def send_command(self, command):
        """Send command to ESP32 and receive line sensor data"""
        if not command in self.VALID_COMMANDS:
            self.logger.error(f"Invalid command: {command}")
            return False
        
        # Only try reconnection if really needed
        if not self.connected:
            if not self.reconnect():
                return False
        
        try:
            # Send command with reasonable timeout
            self.socket.settimeout(2.0)
            self.socket.send(command.encode('utf-8'))
            
            # Receive line sensor data with timeout
            self.socket.settimeout(2.0)
            data = self.socket.recv(64).decode('utf-8').strip()
            if data:
                try:
                    position, detected = map(float, data.split(','))
                    self.line_position = position
                    # ESP32 sends: 0 = line detected, 1 = no line detected
                    self.line_detected = (detected == 0.0)
                    # Only print occasionally to reduce spam
                    if hasattr(self, '_last_print_time'):
                        if time.time() - self._last_print_time > 3.0:
                            print(f"ESP32 Data - Position: {position:.2f}, Detected: {detected}")
                            self._last_print_time = time.time()
                    else:
                        self._last_print_time = time.time()
                except Exception as parse_e:
                    self.logger.error(f"Failed to parse line sensor data: {parse_e}")
                    print(f"ESP32 Raw Data: '{data}'")
            
            return True
        except socket.timeout:
            # Don't disconnect on timeout - ESP32 might just be busy
            return False
        except Exception as e:
            self.logger.error(f"Failed to communicate with ESP32: {e}")
            self.connected = False
            # Close the broken socket
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
                self.socket = None
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

#class VoiceSystem:
#    """Robust voice system with multiple TTS fallbacks"""
#    
#    def __init__(self):
#        # Initialize pygame mixer for audio playback
#        try:
#            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
#            print("Audio system initialized successfully")
#        except Exception as e:
#            print(f"Audio system failed: {e}")
#        
#        self.cache_dir = Path("voice_cache")
#        self.cache_dir.mkdir(exist_ok=True)
#        self.is_playing = False
#        self.tts_method = None
#        
#        # Try to initialize TTS systems in order of preference
#        if self._init_bark():
#            print("Bark TTS system ready!")
#        elif self._init_pyttsx3():
#            print("pyttsx3 TTS system ready!")
#        elif self._init_espeak():
#            print("espeak TTS system ready!")
#        else:
#            print("No TTS system available - voice disabled")
#        
#        # Voice messages with SAVAGE your mom jokes
#        self.savage_mom_jokes = [
#            "Your mom is so big, she shows up on satellite navigation as a permanent obstacle!",
#            "Your mom is so slow, she makes this robot look like Formula 1!",
#            "Your mom is so wide, she blocks more sensors than a solar eclipse!",
#            "Your mom is so heavy, she caused a seismic shift in my line detection algorithm!",
#            "Your mom is so large, NASA classified her as a small moon!",
#            "Your mom is so massive, she has her own gravitational field affecting my gyroscope!",
#            "Your mom is so huge, she appears in every frame of my camera feed!",
#            "Your mom is so enormous, she triggered my emergency avoidance protocols from 3 miles away!",
#            "Your mom is so gigantic, she makes elephants look like ants in my object detection!",
#            "Your mom is so colossal, she broke my distance sensors just by existing!",
#            "Your mom is so immense, she caused a buffer overflow in my memory just thinking about her size!",
#            "Your mom is so vast, she's visible from the International Space Station!",
#            "Your mom is so tremendous, she has her own zip code in my mapping system!",
#            "Your mom is so monumental, she appears as a mountain range in my terrain analysis!",
#            "Your mom is so gargantuan, she makes blue whales jealous of her size!"
#        ]
#        
#        self.voices = {
#            'obstacle': "Obstacle detected. Initiating turn around maneuver.",
#            'line_lost': "Line signal lost. Activating search protocol.",
#            'line_found': "Line detected. Resuming navigation control.",
#            'startup': "Autonomous navigation system activated. Ready to roast some moms!",
#            'turn_complete': "Avoidance maneuver complete. Resuming line following.",
#            'object_detected': "Critical obstacle detected. Executing emergency avoidance.",
#            'tracking_lost': "Visual tracking degraded. Switching to backup navigation.",
#            'high_precision': "High precision mode engaged.",
#            'savage_roast': "Time for a savage roast!"
#        }
#        
#        self.audio_cache = {}
#    
#    def _init_bark(self):
#        """Try to initialize Bark TTS"""
#        try:
#                from bark import SAMPLE_RATE, generate_audio, preload_models
#            from scipy.io.wavfile import write as write_wav
#            
#            self.SAMPLE_RATE = SAMPLE_RATE
#            self.generate_audio = generate_audio
#            self.write_wav = write_wav
#            self.tts_method = 'bark'
#            
#            # Preload models (optional - comment out for faster startup)
#            # preload_models()
#            return True
#        except ImportError:
#            print("Bark TTS not available: 'bark' package not installed")
#            return False
#        except Exception as e:
#            print(f"Bark TTS not available: {e}")
#            return False
#    
#    def _init_pyttsx3(self):
#        """Try to initialize pyttsx3 TTS"""
#        try:
#            import pyttsx3
#            self.tts_engine = pyttsx3.init()
#            self.tts_engine.setProperty('rate', 150)
#            self.tts_engine.setProperty('volume', 0.8)
#            self.tts_method = 'pyttsx3'
#            return True
#        except Exception as e:
#            print(f"pyttsx3 TTS not available: {e}")
#            return False
#    
#    def _init_espeak(self):
#        """Try to initialize espeak TTS"""
#        try:
#            import subprocess
#            result = subprocess.run(['espeak', '--version'], capture_output=True, check=True)
#            self.tts_method = 'espeak'
#            return True
#        except Exception as e:
#            print(f"espeak TTS not available: {e}")
#            return False
#        
#    def _get_cached_audio(self, text, voice_preset):
#        """Get cached audio file or generate new one"""
#        cache_key = f"{text}_{voice_preset}"
#        cache_file = self.cache_dir / f"{abs(hash(cache_key))}.wav"
#        
#        if cache_file.exists():
#            return str(cache_file)
#            
#        # Generate audio using Bark
#        try:
#            print(f"Generating voice: {text[:50]}...")
#            
#            # Generate audio with Bark
#            audio_array = self.generate_audio(text, history_prompt=voice_preset)
#            
#            # Save to file
#            self.write_wav(str(cache_file), self.SAMPLE_RATE, audio_array)
#            print(f"Voice generated and cached: {cache_file.name}")
#            return str(cache_file)
#                
#        except Exception as e:
#            logging.error(f"Failed to generate audio with Bark: {e}")
#            return None
#    
#    def play_sound(self, event_or_text):
#        """Play a sound for a specific event or custom text"""
#        if self.is_playing or not self.tts_method:
#            return  # Don't interrupt current playback or if no TTS
#        
#        # Handle both predefined events and custom text
#        if event_or_text in self.voices:
#            text = self.voices[event_or_text]
#        else:
#            text = str(event_or_text)  # Use the text directly
#        
#        try:
#            self.is_playing = True
#            
#            if self.tts_method == 'bark':
#                self._play_bark_tts(text)
#            elif self.tts_method == 'pyttsx3':
#                self._play_pyttsx3_tts(text)
#            elif self.tts_method == 'espeak':
#                self._play_espeak_tts(text)
#                
#        except Exception as e:
#            print(f"Failed to play TTS audio: {e}")
#            self.is_playing = False
#    
#    def _play_bark_tts(self, text):
#        """Play TTS using Bark"""
#        try:
#            audio_array = self.generate_audio(text, history_prompt="v2/en_speaker_6")
#            temp_file = self.cache_dir / f"temp_{int(time.time())}.wav"
#            self.write_wav(str(temp_file), self.SAMPLE_RATE, audio_array)
#            
#            pygame.mixer.music.load(str(temp_file))
#            pygame.mixer.music.play()
#            
#            def cleanup():
#                while pygame.mixer.music.get_busy():
#                    time.sleep(0.1)
#                self.is_playing = False
#            temp_file.unlink(missing_ok=True)
#            
#            threading.Thread(target=cleanup, daemon=True).start()
#        except Exception as e:
#            print(f"Bark TTS failed: {e}")
#            self.is_playing = False
#    
#    def _play_pyttsx3_tts(self, text):
#        """Play TTS using pyttsx3"""
#        try:
#            def speak():
#                self.tts_engine.say(text)
#                self.tts_engine.runAndWait()
#                self.is_playing = False
#            
#            threading.Thread(target=speak, daemon=True).start()
#        except Exception as e:
#            print(f"pyttsx3 TTS failed: {e}")
#            self.is_playing = False
#    
#    def _play_espeak_tts(self, text):
#        """Play TTS using espeak"""
#        try:
#            def speak():
#                import subprocess
#                subprocess.run(['espeak', text], check=True)
#                self.is_playing = False
#            
#            threading.Thread(target=speak, daemon=True).start()
#        except Exception as e:
#            print(f"espeak TTS failed: {e}")
#            self.is_playing = False
#    
#    def get_savage_roast(self):
#        """Get a random savage your mom joke"""
#        import random
#        return random.choice(self.savage_mom_jokes)
#    
#    def speak_with_roast(self, main_text):
#        """Speak main text followed by a savage roast"""
#        roast = self.get_savage_roast()
#        combined_text = f"{main_text} {roast}"
#        self.play_sound(combined_text)
#    
#    def add_custom_voice(self, event_name, text, voice_preset="v2/en_speaker_6"):
#        """Add a custom voice line"""
##        self.voices[event_name] = {
#            'text': text,
#            'voice_preset': voice_preset
#        }
#
class Robot:
    """Simple robot control class for line following"""
    
    def __init__(self, esp32_ip, esp32_port=1234):
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.route('/')(self.index)
        self.app.route('/video_feed')(self.video_feed)
        self.app.route('/api/status')(self.api_status)
        
        # Initialize camera for line detection
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)  # Try camera 1 if 0 fails
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize ESP32 connection
        self.esp32 = ESP32Interface(esp32_ip, esp32_port)
        
        # Simple state variables
        self.last_command = "STOP"
        self.last_command_time = time.time()
        self.line_detected = False
        self.obstacle_detected = False
        self.current_action = "Initializing"
        
        # Initialize advanced voice system with savage roasts
        # self.voice_system = VoiceSystem()
        self.tts_enabled = self.check_tts_available()
        
        # Start web server thread
        self.web_thread = threading.Thread(target=self._run_web_server)
        self.web_thread.daemon = True
        self.web_thread.start()
        
        logging.info("Robot initialized - simple mode")
        
        # Start with a savage roast announcement
        #self.speak_savage_roast(" roast robot activated and ready to insult your moms while following lines. Starting up because")
    
    def index(self):
        """Serve the web interface"""
        return render_template_string(HTML_TEMPLATE)
    
    def check_tts_available(self):
        """Check if text-to-speech is available"""
        try:
            # Check for espeak
            subprocess.run(['espeak', '--version'], capture_output=True, check=True)
            logging.info("Text-to-speech (espeak) available")
            return True
        except:
            try:
                # Check for festival
                subprocess.run(['festival', '--version'], capture_output=True, check=True)
                logging.info("Text-to-speech (festival) available")
                return True
            except:
                logging.warning("No text-to-speech system found")
                return False
    
    def speak(self, text):
        """Enhanced text-to-speech with optional savage roasts"""
        if not self.tts_enabled:
            return
        
        # 30% chance to add a savage roast to any speech
        import random
        if random.random() < 0.3:  # 30% chance
            roast = self.voice_system.get_savage_roast()
            text = f"{text} Also, {roast}"
        
        try:
            # Use espeak for fast, clear speech
            subprocess.Popen(['espeak', '-s', '150', '-v', 'en', text], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            try:
                # Fallback to festival
                subprocess.Popen(['festival', '--tts'], 
                               input=text.encode(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except:
                pass
    
    def speak_savage_roast(self, main_text):
        """Guaranteed savage roast with main text"""
        if not self.tts_enabled:
            return
        
        roast = self.voice_system.get_savage_roast()
        combined_text = f"{main_text} {roast}"
        
        try:
            subprocess.Popen(['espeak', '-s', '150', '-v', 'en', combined_text], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            try:
                subprocess.Popen(['festival', '--tts'], 
                               input=combined_text.encode(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except:
                pass
    
    def api_status(self):
        """API endpoint for real-time robot status"""
        return jsonify({
            'status': 'Autonomous' if self.esp32.connected else 'Disconnected',
            'command': self.last_command,
            'line_detected': self.line_detected,
            'obstacle_detected': getattr(self, 'obstacle_detected', False),
            'position': getattr(self.esp32, 'line_position', 0.0),
            'current_action': getattr(self, 'current_action', 'Unknown'),
            'esp_connected': self.esp32.connected
        })
    
    def _run_web_server(self):
        """Run the simple web server"""
        self.app.run(host='0.0.0.0', port=5001, debug=False)
    
    def send_command(self, command):
        """Send command to ESP32"""
        if self.esp32.send_command(command):
            self.last_command = command
            logging.info(f"Command sent: {command}")
            return True
        return False
    
    def video_feed(self):
        """Video streaming route"""
        return Response(self.generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def generate_frames(self):
        """Generate camera frames for streaming"""
        while True:
            ret, frame = self.cap.read()
            if ret:
                # Process frame for line detection
                self.process_frame(frame)
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                break
    
    def process_frame(self, frame):
        """Process frame for OBJECT DETECTION only - not line following!"""
        height, width = frame.shape[:2]
        
        # Much smaller detection area to avoid false positives
        center_region = frame[height//2:3*height//4, 2*width//5:3*width//5]
        
        # More sophisticated obstacle detection
        gray = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
        
        # Use edge detection for better obstacle recognition
        edges = cv2.Canny(gray, 50, 150)
        edge_count = np.sum(edges > 0)
        total_pixels = gray.shape[0] * gray.shape[1]
        edge_ratio = edge_count / total_pixels
        
        # Also check for very dark objects (like walls)
        dark_threshold = np.mean(gray) - 50
        dark_pixels = np.sum(gray < dark_threshold)
        dark_ratio = dark_pixels / total_pixels
        
        # More sensitive obstacle detection
        obstacle_detected = (edge_ratio > 0.1 or dark_ratio > 0.2)
        
        if obstacle_detected:
            # Prevent spam - only trigger if enough time has passed
            current_time = time.time()
            if not hasattr(self, '_last_obstacle_time'):
                self._last_obstacle_time = 0
            
            if current_time - self._last_obstacle_time > 3.0:  # 3 second cooldown
                logging.info("MAJOR OBSTACLE DETECTED! Making 180° turn")
                self.current_action = "OBSTACLE! Making 180° turn"
                #self.speak_savage_roast("YOUR MOM DETECTED, turning around.")
                self.send_command("TURN_AROUND")
                self._last_obstacle_time = current_time
                self.obstacle_detected = True
            
            # Add visual indicator on frame
            cv2.rectangle(frame, (2*width//5, height//2), (3*width//5, 3*height//4), (0, 0, 255), 3)
            cv2.putText(frame, "OBSTACLE!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            self.obstacle_detected = False
        
        return frame
    
    def smart_line_following(self):
        """FAST line following using ESP32 sensor data"""
        # Get line data from ESP32
        line_detected = self.esp32.line_detected
        position = self.esp32.line_position  # -1.0 (left) to +1.0 (right)
        
        current_time = time.time()
        
        if line_detected:
            # Line following mode - FAST and direct
            self.line_detected = True
            error = position
            
            # Direct control based on error - no smoothing
            abs_error = abs(error)
            
            if abs_error < 0.15:  # Close to center - wider zone
                command = "FORWARD"
                self.current_action = "Following line"
            elif abs_error < 0.4:  # Medium correction - includes 0.5
                # FIXED: Corrected direction logic
                command = "SLIGHT_RIGHT" if error > 0 else "SLIGHT_LEFT"
                self.current_action = "Correcting position"
            else:  # Big correction needed
                # FIXED: Corrected direction logic  
                command = "RIGHT" if error > 0 else "LEFT"
                self.current_action = "Major turn"
            
            self.send_command(command)
            
        else:
            # No line detected - FAST search
            self.line_detected = False
            command = "RIGHT"  # Just spin right to search
            self.current_action = "Searching for line"
            self.send_command(command)
    
    def calculate_control(self):
        """Ultra-simplified control focused only on line following"""
        # Direct line following - no voice, no delays
        if self.esp32.line_detected:
            self.line_status = True
            return self._pid_line_following()
        else:
            self.line_status = False
            return self._handle_line_loss()
    
    def _pid_line_following(self):
        """Responsive but smooth PID controller for line following"""
        if not hasattr(self, 'pid_controller'):
            self.pid_controller = PIDController(kp=0.5, ki=0.0, kd=0.15)  # Smoother PID
        
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
        
        # Enhanced smoothing for silky smooth movement
        if not hasattr(self, 'error_history'):
            self.error_history = [0.0, 0.0, 0.0, 0.0, 0.0]  # 5-point smoothing for ultra smooth
        
        self.error_history.append(error)
        self.error_history.pop(0)
        smooth_error = sum(self.error_history) / len(self.error_history)
        
        control_output = self.pid_controller.update(smooth_error)
        
        # FIXED: Corrected control direction logic
        # Positive error = line to RIGHT of robot = turn LEFT to follow
        # Negative error = line to LEFT of robot = turn RIGHT to follow
        if abs(smooth_error) < 0.06:  # Very tight center zone for super smooth following
            return "FORWARD"
        elif smooth_error > 0.5:  # Line far to the right - turn LEFT sharply
            return "LEFT"
        elif smooth_error < -0.5:  # Line far to the left - turn RIGHT sharply
            return "RIGHT"
        elif smooth_error > 0.08:  # Line slightly right - turn LEFT gently
            return "SLIGHT_LEFT"
        elif smooth_error < -0.08:  # Line slightly left - turn RIGHT gently
            return "SLIGHT_RIGHT"
        else:
            return "FORWARD"
    
    def _handle_line_loss(self):
        """Handle when robot loses the line - simplified search"""
        current_time = time.time()
        
        if not hasattr(self, 'line_lost_time'):
            self.line_lost_time = current_time
            self.search_direction = "LEFT"  # Default search direction
        
        search_duration = current_time - self.line_lost_time
        
        if search_duration < 1.0:  # Search for 1 second in one direction (faster)
            return self.search_direction
        elif search_duration < 2.0:  # Then try the other direction (faster)
            return "RIGHT" if self.search_direction == "LEFT" else "LEFT"
        else:
            # Reset search every 3 seconds to prevent getting stuck
            if search_duration > 3.0:
                self.line_lost_time = current_time
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
        """Send command to ESP32 - simplified for speed"""
        current_time = time.time()
        
        if (command != self.last_command or 
            current_time - self.last_command_time > 0.1):
            
            if self.esp32.send_command(command):
                self.last_command = command
                self.last_command_time = current_time
    
    def run(self):
        """Smart autonomous robot with line following + obstacle avoidance"""
        try:
            logging.info("Starting smart autonomous robot control")
            
            # Check camera for obstacle detection
            camera_available = self.cap and self.cap.isOpened()
            if camera_available:
                logging.info("Camera available for obstacle detection")
            else:
                logging.warning("No camera - obstacle detection disabled")
                
            # Connect to ESP32
            if not self.esp32.connect():
                logging.error("Failed to connect to ESP32")
                return
                
            logging.info("ESP32 connected - starting smart line following")
            
            frame_count = 0
            
            while True:
                frame_count += 1
                
                # Primary control: Smart line following using ESP32 sensors
                self.smart_line_following()
                
                # Secondary: Obstacle detection using camera (every 3rd frame)
                if camera_available and frame_count % 3 == 0:
                    ret, frame = self.cap.read()
                    if ret:
                        self.process_frame(frame)  # Check for obstacles
                
                # Faster control loop for quicker response
                time.sleep(0.08)  # ~12Hz - faster response, still manageable for ESP32
                
        except KeyboardInterrupt:
            logging.info("Shutting down...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        logging.info("Cleaning up...")
        self.send_command("STOP")
        if self.cap:
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