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

# Optional: YOLO object detection (Ultralytics)
try:
    from ultralytics import YOLO  # pip install ultralytics
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# YOLO settings
USE_YOLO = True  # Toggle to enable/disable YOLO at runtime
YOLO_MODEL_PATH = "yolo11n.pt"  # Nano model for speed
YOLO_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detection

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
    """IMPROVED: Handles communication with ESP32 with heartbeat and message queuing"""
    
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
        
        # IMPROVED: Sensor fusion system
        self.line_position = 0
        self.line_detected = False
        self.sensor_confidence = 0.0
        self.position_history = []
        self.confidence_history = []
        self.fusion_window = 5
        
        # IMPROVED: Communication reliability
        self.message_queue = []
        self.last_heartbeat_time = time.time()
        self.last_sensor_time = time.time()
        self.heartbeat_interval = 2.0  # Expect heartbeat every 2 seconds
        self.connection_stable = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
    
    def connect(self):
        """IMPROVED: Establish connection to ESP32 with better reliability"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)  # Reasonable timeout
            # Enable keep-alive to maintain connection
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            # Disable Nagle's algorithm for faster response
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.socket.connect((self.ip_address, self.port))
            
            # IMPROVED: Set non-blocking mode after connection
            self.socket.settimeout(0.1)  # Short timeout for non-blocking operation
            
            self.connected = True
            self.connection_stable = True
            self.reconnect_attempts = 0
            self.last_heartbeat_time = time.time()
            self.message_queue.clear()
            
            self.logger.info(f"Connected to ESP32 at {self.ip_address}:{self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to ESP32: {e}")
            self.connected = False
            self.connection_stable = False
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
                self.socket = None
            return False
    
    def reconnect(self):
        """IMPROVED: Attempt to reconnect with exponential backoff"""
        # Don't spam reconnection attempts
        if hasattr(self, '_last_reconnect_time'):
            if time.time() - self._last_reconnect_time < (2.0 ** min(self.reconnect_attempts, 4)):
                return False
        
        self.reconnect_attempts += 1
        if self.reconnect_attempts > self.max_reconnect_attempts:
            self.logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
            return False
        
        self.logger.info(f"Attempting to reconnect to ESP32 (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
        self._last_reconnect_time = time.time()
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.socket = None
        self.connected = False
        self.connection_stable = False
        
        # Clear message queue on reconnection
        self.message_queue.clear()
        
        return self.connect()
    
    def send_command(self, command):
        """IMPROVED: Send command with non-blocking communication and message queuing"""
        if not command in self.VALID_COMMANDS:
            self.logger.error(f"Invalid command: {command}")
            return False
        
        # Check connection health
        if not self._check_connection_health():
            if not self.reconnect():
                return False
        
        try:
            # Add command to queue for reliable delivery
            self.message_queue.append({
                'command': command,
                'timestamp': time.time(),
                'attempts': 0
            })
            
            # Process message queue (non-blocking)
            return self._process_message_queue()
            
        except Exception as e:
            self.logger.error(f"Failed to queue command: {e}")
            self.connection_stable = False
            return False
    
    def _check_connection_health(self):
        """Check if connection is healthy based on heartbeat"""
        current_time = time.time()
        
        # Check for heartbeat timeout
        if current_time - self.last_heartbeat_time > self.heartbeat_interval * 2:
            self.logger.warning("Heartbeat timeout - connection may be unstable")
            self.connection_stable = False
            return False
        
        return self.connected and self.connection_stable
    
    def _process_message_queue(self):
        """Process queued messages with non-blocking I/O"""
        if not self.message_queue or not self.connected:
            return False
        
        try:
            # Process one message at a time to avoid blocking
            message = self.message_queue[0]
            
            # Send command with timeout
            self.socket.settimeout(0.1)
            self.socket.send(message['command'].encode('utf-8'))
            
            # Try to receive response (sensor data or heartbeat)
            self._receive_data()
            
            # Remove processed message
            self.message_queue.pop(0)
            return True
            
        except socket.timeout:
            # Timeout is normal for non-blocking operation
            message['attempts'] += 1
            if message['attempts'] > 3:
                self.logger.warning(f"Command {message['command']} failed after 3 attempts")
                self.message_queue.pop(0)
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to process message queue: {e}")
            self.connected = False
            self.connection_stable = False
            return False
    
    def _receive_data(self):
        """IMPROVED: Non-blocking data reception with sensor fusion"""
        try:
            # Receive data with short timeout
            self.socket.settimeout(0.05)
            data = self.socket.recv(128).decode('utf-8').strip()
            
            if not data:
                return
            
            # Process different types of messages
            for line in data.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if line == "HEARTBEAT":
                    self.last_heartbeat_time = time.time()
                    self.connection_stable = True
                elif ',' in line:
                    # Sensor data: position,detected
                    try:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            position = float(parts[0])
                            detected = float(parts[1])
                            
                            # Apply sensor fusion
                            self._update_sensor_fusion(position, detected == 0.0)
                            
                            self.last_sensor_time = time.time()
                    except ValueError as e:
                        self.logger.debug(f"Failed to parse sensor data: {line}")
                        
        except socket.timeout:
            # Normal for non-blocking operation
            pass
        except Exception as e:
            self.logger.debug(f"Data reception error: {e}")
    
    def _update_sensor_fusion(self, position, detected):
        """IMPROVED: Unified sensor fusion for position and detection"""
        current_time = time.time()
        
        # Calculate confidence based on data freshness and consistency
        confidence = 1.0 if detected else 0.5
        
        # Add to history
        self.position_history.append({
            'position': position,
            'confidence': confidence,
            'timestamp': current_time
        })
        
        # Maintain history size
        if len(self.position_history) > self.fusion_window:
            self.position_history.pop(0)
        
        # Apply temporal fusion
        if len(self.position_history) >= 2:
            # Weighted average based on confidence and recency
            total_weight = 0
            weighted_sum = 0
            
            for i, data in enumerate(self.position_history):
                # Recent data gets higher weight
                age = current_time - data['timestamp']
                recency_weight = max(0.1, 1.0 - age * 0.5)  # Decay over 2 seconds
                
                # Confidence weight
                conf_weight = data['confidence']
                
                # Combined weight
                weight = recency_weight * conf_weight
                
                weighted_sum += data['position'] * weight
                total_weight += weight
            
            if total_weight > 0:
                fused_position = weighted_sum / total_weight
            else:
                fused_position = position
        else:
            fused_position = position
        
        # Update state
        self.line_position = fused_position
        self.line_detected = detected
        self.sensor_confidence = confidence
        
        # Debug output (reduced frequency)
        if not hasattr(self, '_last_fusion_debug'):
            self._last_fusion_debug = 0
        if current_time - self._last_fusion_debug > 2.0:
            self.logger.debug(f"Sensor fusion: pos={fused_position:.3f}, detected={detected}, conf={confidence:.2f}")
            self._last_fusion_debug = current_time
    
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
        
        # YOLO model (optional)
        self.use_yolo = USE_YOLO and YOLO_AVAILABLE
        self.yolo_model = None

        if self.use_yolo:
            try:
                logging.info(f"Loading YOLO model: {YOLO_MODEL_PATH}")
                self.yolo_model = YOLO(YOLO_MODEL_PATH)
                logging.info("YOLO model loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load YOLO model: {e}")
                self.use_yolo = False

        # Enhanced 3-phase obstacle avoidance state with mapping
        self.object_detected = False
        self.object_position = 0.0
        self.current_obstacle = None       # Current obstacle being avoided
        self.obstacle_map = {}            # Persistent obstacle memory
        self.avoidance_phase = 'none'     # 'none', 'turnaround', 'search_after_turn', 'return_to_line'
        self.turnaround_direction = 'right'  # 'left' or 'right' - which way to turn around
        self.avoidance_duration = 0
        self.turnaround_start_time = 0.0    # Track actual time for turnaround verification
        self.planned_path = None          # Calculated path around obstacle
        self.corner_warning = False
        self.corner_prediction_frames = 0
        self.object_detection_frames = 0
        self.OBJECT_DETECTION_PERSISTENCE = 0  # Immediate detection for faster avoidance

        # Dynamic avoidance parameters
        self.MAX_TURN_DURATION = 20          # Maximum frames to turn if object still visible
        self.AVOIDANCE_CLEAR_DURATION = 12   # Phase 2: Move forward to clear object  
        self.AVOIDANCE_RETURN_DURATION = 15  # Phase 3: Turn back to find line
        self.OBJECT_CLEAR_THRESHOLD = 3      # Frames without seeing object to consider it cleared

        # 180-DEGREE TURNAROUND AVOIDANCE - ENHANCED WITH RETURN-TO-LINE
        # PROBLEM: Robot not completing full 180° turn AND not finding line afterward
        # SOLUTION: Dramatically increased duration + systematic line recovery
        self.TURNAROUND_FRAMES = 200        # MUCH longer - 40 seconds at 5 FPS to ensure full turn
        self.TURNAROUND_SEARCH_FRAMES = 50  # Initial search frames after turnaround
        self.TURNAROUND_IGNORE_LINE = True  # Ignore line detection during turnaround to prevent early exit
        self.TURNAROUND_COMMAND = 'RIGHT'   # Use strong RIGHT command for turnaround

        # NEW: Enhanced return-to-line system after turnaround
        self.RETURN_TO_LINE_PHASES = {
            'initial_search': 30,      # 6 seconds: Quick search in likely directions
            'systematic_sweep': 60,    # 12 seconds: Left-right sweep pattern
            'spiral_search': 80,       # 16 seconds: Expanding spiral search
            'recovery_mode': 100       # 20 seconds: Aggressive recovery attempts
        }

        # Return-to-line state variables
        self.return_phase = 'none'          # Current return phase
        self.return_phase_counter = 0       # Frames left in current phase
        self.last_line_direction = 0.0      # Remember which way line was going before obstacle
        self.pre_turnaround_position = 0.0  # Line position before we started avoiding
        self.search_pattern_step = 0        # Current step in search pattern
        self.successful_returns = 0         # Track success rate for learning

        # Alternative settings if robot still doesn't turn enough:
        # TURNAROUND_FRAMES = 300  # 60 seconds - try this if 200 isn't enough
        # TURNAROUND_FRAMES = 400  # 80 seconds - last resort
    
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
        """Send command to ESP32 - simplified for speed"""
        current_time = time.time()
        
        if (command != self.last_command or 
            current_time - self.last_command_time > 0.1):
            
            if self.esp32.send_command(command):
                self.last_command = command
                self.last_command_time = current_time

    def run(self):
        """ENHANCED: Non-blocking autonomous robot with obstacle avoidance and return-to-line recovery"""
        try:
            logging.info("Starting enhanced autonomous robot control with obstacle avoidance")
            
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
                
            logging.info("ESP32 connected - starting enhanced line following with obstacle avoidance")
            
            frame_count = 0
            last_control_time = time.time()
            last_camera_time = time.time()
            control_interval = 0.05  # 20Hz control loop (much faster)
            camera_interval = 0.1    # 10Hz camera processing
            
            while True:
                current_time = time.time()
                frame_count += 1
                
                # ENHANCED: High-frequency control loop (20Hz) with obstacle avoidance
                if current_time - last_control_time >= control_interval:
                    # Always try to receive sensor data (non-blocking)
                    self.esp32._receive_data()
                    
                    # Primary control: Enhanced line following with obstacle avoidance
                    self.smart_line_following()
                    last_control_time = current_time
                
                # ENHANCED: Camera processing for obstacle detection (10Hz)
                if camera_available and (current_time - last_camera_time >= camera_interval):
                    ret, frame = self.cap.read()
                    if ret:
                        self.process_frame(frame)  # Check for obstacles and update avoidance state
                    last_camera_time = current_time
                
                # ENHANCED: Much shorter sleep for responsiveness
                time.sleep(0.01)  # 1ms sleep - much more responsive
                
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

    def _detect_objects_yolo(self, frame):
        """Return True if any object is detected with confidence above threshold"""
        if not self.use_yolo or self.yolo_model is None:
            return False

        # Run inference (suppress verbose)
        try:
            results = self.yolo_model(frame, conf=0.5, verbose=False)  # Use 0.5 confidence threshold
            if not results:
                return False

            result = results[0]  # Single image
            # If no boxes => no detection
            if result.boxes is None or len(result.boxes) == 0:
                return False

            return True  # At least one object detected
        except Exception as e:
            logging.error(f"YOLO inference error: {e}")
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
        """Process camera frame for obstacle detection"""
        if not self.use_yolo:
            return
        
        try:
            # Run YOLO detection
            results = self.yolo_model(frame)
            
            # Process detections
            detected_objects = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class and confidence
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Check if it's a class we want to avoid and confidence is high enough
                        if cls in self.classes_to_avoid and conf > 0.5:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            # Calculate object position (-1.0 to 1.0, left to right)
                            frame_width = frame.shape[1]
                            object_center_x = (x1 + x2) / 2
                            object_position = (object_center_x / frame_width) * 2 - 1
                            
                            detected_objects.append({
                                'class': cls,
                                'confidence': conf,
                                'position': object_position,
                                'bbox': (x1, y1, x2, y2)
                            })
            
            # Update obstacle detection state
            if detected_objects:
                self.object_detected = True
                # Use the most confident detection
                best_detection = max(detected_objects, key=lambda x: x['confidence'])
                self.object_position = best_detection['position']
                
                # Remember line state before starting avoidance
                if self.avoidance_phase == 'none' and self.esp32.line_detected:
                    self.pre_turnaround_position = self.esp32.line_position
                    self.last_line_direction = self.esp32.line_position
                    logging.info(f"🎯 OBSTACLE DETECTED! Remembering line position: {self.pre_turnaround_position:.2f}")
            else:
                self.object_detected = False
                
        except Exception as e:
            logging.error(f"Error in obstacle detection: {e}")

    def get_turn_command_with_avoidance(self, steering, line_detected_now=False, line_offset_now=0.0):
        """Enhanced 180-degree turnaround with systematic return-to-line recovery"""
        
        # START 180-DEGREE TURNAROUND when obstacle detected
        if self.object_detected and self.avoidance_phase == 'none':
            self.turnaround_direction = 'right'  # Always turn right for consistency
            self.avoidance_phase = 'turnaround'
            self.avoidance_duration = self.TURNAROUND_FRAMES
            self.turnaround_start_time = time.time()
            
            logging.info(f"🚨 OBSTACLE DETECTED - Starting 180-DEGREE TURNAROUND! Duration: {self.TURNAROUND_FRAMES} frames")
            logging.info(f"📍 Pre-turnaround line position: {self.pre_turnaround_position:.2f}")
            
            return 'RIGHT'  # Start turnaround immediately
        
        # PHASE 1: Complete 180-degree turnaround (IGNORE LINE DETECTION)
        elif self.avoidance_phase == 'turnaround':
            if self.avoidance_duration > 0:
                self.avoidance_duration -= 1
                
                # Log progress every 20 frames
                if self.avoidance_duration % 20 == 0:
                    progress_percent = ((self.TURNAROUND_FRAMES - self.avoidance_duration) / self.TURNAROUND_FRAMES) * 100
                    elapsed_time = time.time() - self.turnaround_start_time
                    logging.info(f"🔄 TURNAROUND PROGRESS: {progress_percent:.0f}% - {elapsed_time:.1f}s elapsed")
                
                return 'RIGHT'  # Force continuous right turn
            else:
                # Turnaround complete - start systematic line recovery
                total_time = time.time() - self.turnaround_start_time
                self.avoidance_phase = 'return_to_line'
                self.return_phase = 'initial_search'
                self.return_phase_counter = self.RETURN_TO_LINE_PHASES['initial_search']
                self.search_pattern_step = 0
                
                logging.info(f"✅ TURNAROUND COMPLETE - Total time: {total_time:.1f}s")
                logging.info(f"🔍 Starting systematic line recovery - Phase: {self.return_phase}")
                
                return 'FORWARD'
        
        # PHASE 2: Systematic return-to-line recovery
        elif self.avoidance_phase == 'return_to_line':
            # Check if we found the line
            if line_detected_now:
                logging.info(f"✅ LINE FOUND during {self.return_phase}! Recovery successful")
                self.avoidance_phase = 'none'
                self.return_phase = 'none'
                self.successful_returns += 1
                
                # Resume normal line following
                if abs(line_offset_now) < 0.1:
                    return 'FORWARD'
                else:
                    return 'LEFT' if line_offset_now > 0 else 'RIGHT'
            
            # Execute current return phase
            self.return_phase_counter -= 1
            
            if self.return_phase == 'initial_search':
                return self._initial_search_pattern()
            elif self.return_phase == 'systematic_sweep':
                return self._systematic_sweep_pattern()
            elif self.return_phase == 'spiral_search':
                return self._spiral_search_pattern()
            elif self.return_phase == 'recovery_mode':
                return self._recovery_mode_pattern()
            
            # Check if current phase is complete
            if self.return_phase_counter <= 0:
                self._advance_to_next_return_phase()
                
            return 'FORWARD'  # Default action
        
        # NORMAL LINE FOLLOWING BEHAVIOR
        if abs(steering) < 0.15:  # Deadzone
            return 'FORWARD'
        elif steering > 0.4:
            return 'LEFT'
        elif steering < -0.4:
            return 'RIGHT'
        else:
            return 'LEFT' if steering > 0 else 'RIGHT'

    def _initial_search_pattern(self):
        """Phase 1: Quick search in likely directions based on pre-turnaround memory"""
        step = self.search_pattern_step % 20
        self.search_pattern_step += 1
        
        # Use memory of where line was before turnaround
        if self.pre_turnaround_position > 0.2:  # Line was to the right
            if step < 10:
                return 'RIGHT'  # Search right first
            else:
                return 'LEFT'   # Then search left
        elif self.pre_turnaround_position < -0.2:  # Line was to the left
            if step < 10:
                return 'LEFT'   # Search left first
            else:
                return 'RIGHT'  # Then search right
        else:  # Line was center - alternate search
            return 'RIGHT' if step < 10 else 'LEFT'

    def _systematic_sweep_pattern(self):
        """Phase 2: Left-right sweep pattern with increasing amplitude"""
        step = self.search_pattern_step % 40
        self.search_pattern_step += 1
        
        if step < 15:
            return 'LEFT'
        elif step < 20:
            return 'FORWARD'
        elif step < 35:
            return 'RIGHT'
        else:
            return 'FORWARD'

    def _spiral_search_pattern(self):
        """Phase 3: Expanding spiral search pattern"""
        step = self.search_pattern_step % 60
        self.search_pattern_step += 1
        
        if step < 10:
            return 'FORWARD'
        elif step < 20:
            return 'RIGHT'
        elif step < 30:
            return 'FORWARD'
        elif step < 45:
            return 'LEFT'
        elif step < 55:
            return 'FORWARD'
        else:
            return 'RIGHT'

    def _recovery_mode_pattern(self):
        """Phase 4: Aggressive recovery attempts"""
        step = self.search_pattern_step % 30
        self.search_pattern_step += 1
        
        # More aggressive movements
        if step < 8:
            return 'LEFT'
        elif step < 12:
            return 'FORWARD'
        elif step < 20:
            return 'RIGHT'
        elif step < 24:
            return 'FORWARD'
        else:
            return 'LEFT'

    def _advance_to_next_return_phase(self):
        """Advance to the next phase of return-to-line recovery"""
        if self.return_phase == 'initial_search':
            self.return_phase = 'systematic_sweep'
            self.return_phase_counter = self.RETURN_TO_LINE_PHASES['systematic_sweep']
            logging.info("🔍 Advancing to systematic sweep phase")
        elif self.return_phase == 'systematic_sweep':
            self.return_phase = 'spiral_search'
            self.return_phase_counter = self.RETURN_TO_LINE_PHASES['spiral_search']
            logging.info("🔍 Advancing to spiral search phase")
        elif self.return_phase == 'spiral_search':
            self.return_phase = 'recovery_mode'
            self.return_phase_counter = self.RETURN_TO_LINE_PHASES['recovery_mode']
            logging.info("🔍 Advancing to recovery mode phase")
        else:
            # Recovery complete - give up and resume normal operation
            logging.warning("⚠️ All recovery phases exhausted - resuming normal operation")
            self.avoidance_phase = 'none'
            self.return_phase = 'none'
        
        self.search_pattern_step = 0  # Reset pattern step for new phase

    def smart_line_following(self):
        """ENHANCED: Line following with obstacle avoidance and systematic return-to-line recovery"""
        # Get fused sensor data from ESP32
        line_detected = self.esp32.line_detected
        position = self.esp32.line_position  # -1.0 (left) to +1.0 (right)
        confidence = self.esp32.sensor_confidence
        
        current_time = time.time()
        
        # PRIORITY 1: Handle obstacle avoidance phases
        if self.avoidance_phase != 'none':
            # Calculate steering for avoidance system
            if line_detected and confidence > 0.3:
                steering = position  # Use actual line position when available
            else:
                steering = 0.0  # No line detected
            
            # Get avoidance command
            command = self.get_turn_command_with_avoidance(
                steering, 
                line_detected_now=line_detected, 
                line_offset_now=position
            )
            
            # Update status based on avoidance phase
            if self.avoidance_phase == 'turnaround':
                progress = ((self.TURNAROUND_FRAMES - self.avoidance_duration) / self.TURNAROUND_FRAMES) * 100
                self.current_action = f"🔄 TURNAROUND: {progress:.0f}% complete"
            elif self.avoidance_phase == 'return_to_line':
                self.current_action = f"🔍 RECOVERY: {self.return_phase} phase ({self.return_phase_counter} frames left)"
            
            self.send_command(command)
            return
        
        # PRIORITY 2: Normal line following when no avoidance needed
        if line_detected and confidence > 0.3:  # Use confidence threshold
            # Line following mode - IMPROVED with confidence-based control
            self.line_detected = True
            error = position
            
            # IMPROVED: Confidence-based control sensitivity
            # Higher confidence = more aggressive control
            # Lower confidence = more conservative control
            confidence_factor = min(confidence, 1.0)
            
            # IMPROVED: Better control logic with confidence weighting
            abs_error = abs(error)
            
            # Adjust thresholds based on confidence
            center_threshold = 0.05 * (2.0 - confidence_factor)  # Wider when less confident
            gentle_threshold = 0.2 * (2.0 - confidence_factor)
            moderate_threshold = 0.5 * (2.0 - confidence_factor)
            
            # Perfect center detection - when ESP32 reports very low error
            if abs_error < center_threshold:
                command = "FORWARD"
                self.current_action = f"Center (conf: {confidence:.2f})"
            elif abs_error < gentle_threshold:  # Close to center - gentle correction
                if error > 0:
                    command = "SLIGHT_RIGHT"  # Line slightly to right, turn right gently
                    self.current_action = f"Gentle right (conf: {confidence:.2f})"
                else:
                    command = "SLIGHT_LEFT"   # Line slightly to left, turn left gently
                    self.current_action = f"Gentle left (conf: {confidence:.2f})"
            elif abs_error < moderate_threshold:  # Medium offset - moderate correction
                if error > 0:
                    command = "RIGHT"  # Line to right, turn right
                    self.current_action = f"Turn right (conf: {confidence:.2f})"
                else:
                    command = "LEFT"   # Line to left, turn left
                    self.current_action = f"Turn left (conf: {confidence:.2f})"
            else:  # Large offset - sharp correction needed
                if error > 0:
                    command = "RIGHT"  # Line far to right, turn right sharply
                    self.current_action = f"Sharp right (conf: {confidence:.2f})"
                else:
                    command = "LEFT"   # Line far to left, turn left sharply
                    self.current_action = f"Sharp left (conf: {confidence:.2f})"
            
            # Check for obstacle avoidance override
            if self.object_detected:
                # Start avoidance - this will be handled in next iteration
                self.current_action = f"🚨 OBSTACLE DETECTED - Starting avoidance"
                command = self.get_turn_command_with_avoidance(
                    error, 
                    line_detected_now=line_detected, 
                    line_offset_now=position
                )
            
            # Send command
            self.send_command(command)
            
            # Debug output every 3 seconds to avoid spam
            if not hasattr(self, '_last_debug_time'):
                self._last_debug_time = 0
            if current_time - self._last_debug_time > 3.0:
                print(f"LINE FOLLOW: pos={position:.3f}, conf={confidence:.2f}, cmd={command}")
                self._last_debug_time = current_time
            
        else:
            # No line detected or low confidence - IMPROVED search pattern
            self.line_detected = False
            
            # Check if we should start obstacle avoidance during search
            if self.object_detected:
                command = self.get_turn_command_with_avoidance(
                    0.0, 
                    line_detected_now=False, 
                    line_offset_now=0.0
                )
                self.current_action = f"🚨 OBSTACLE during search - Starting avoidance"
                self.send_command(command)
                return
            
            # Use sensor fusion history to guide search
            if hasattr(self.esp32, 'position_history') and self.esp32.position_history:
                # Get recent position trend
                recent_positions = [data['position'] for data in self.esp32.position_history[-3:]]
                if recent_positions:
                    avg_recent_pos = sum(recent_positions) / len(recent_positions)
                    
                    if avg_recent_pos > 0.3:  # Line was trending right
                        command = "RIGHT"
                        self.current_action = f"Search right (trend: {avg_recent_pos:.2f})"
                    elif avg_recent_pos < -0.3:  # Line was trending left
                        command = "LEFT"
                        self.current_action = f"Search left (trend: {avg_recent_pos:.2f})"
                    else:
                        # Line was near center, do time-based alternating search
                        search_time = int(current_time) % 6  # 6 second cycle
                        if search_time < 3:
                            command = "RIGHT"
                            self.current_action = "Search right (center trend)"
                        else:
                            command = "LEFT"
                            self.current_action = "Search left (center trend)"
                else:
                    # No trend data, default search
                    command = "RIGHT"
                    self.current_action = "Default search right"
            else:
                # No history, default right search
                command = "RIGHT"
                self.current_action = "Default search right"
            
            self.send_command(command)

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