#!/usr/bin/env python3

import cv2
import numpy as np
import time
import math
from typing import Tuple, Optional, Dict, List
from collections import deque

try:
    from picamera2 import Picamera2
except ImportError:
    print("Warning: picamera2 library not found. Camera functionality will be limited.")
    Picamera2 = None

# This is needed by the mixin
from pid import PIDController
from adaptive_pid import AdaptivePIDController
from line_memory_buffer import LineMemoryBuffer 

# Shared configuration
LINE_FOLLOW_SPEED = 45

class CameraObstacleAvoidance:
    """
    Camera-based obstacle avoidance and corner detection system.
    Uses computer vision to detect obstacles ahead and identify corners.
    """
    
    def __init__(self, debug=False):
        self.debug = debug
        
        # Debug frame optimization
        self.debug_frame_counter = 0
        self.debug_frame_skip = 2  
        
        # Obstacle detection parameters
        self.depth_threshold = 50  
        self.min_obstacle_area = 300  # Minimum contour area to consider as obstacle
        self.obstacle_distance_threshold = 0.3  # Distance threshold (normalized)
        
        # Corner detection parameters
        self.corner_angle_threshold = 60  # Degrees to detect corner
        self.corner_line_length = 50  # Minimum line length for corner detection
        self.corner_cooldown = 2.0  # Seconds between corner detections
        self.last_corner_time = 0
        
        # ROI configuration (focus on forward view)
        self.obstacle_roi_height_ratio = 0.6  # Use top 60% of frame for obstacle detection
        self.corner_roi_height_ratio = 0.4   # Use bottom 40% for corner detection
        
        # Detection history for smoothing
        self.obstacle_history = deque(maxlen=5)
        self.corner_history = deque(maxlen=3)
        self.frames_processed = 0
        
        # Movement state
        self.last_obstacle_direction = None
        self.obstacle_avoidance_active = False
        self.corner_turn_active = False
        
    def detect_obstacles_and_corners(self, frame: np.ndarray) -> Dict:
        """
        Detect obstacles ahead and corners using computer vision.
        
        Args:
            frame: Input camera frame (BGR)
            
        Returns:
            Dictionary with detection results:
            - obstacle_detected: bool
            - obstacle_direction: str ('left', 'right', 'center', None)
            - obstacle_distance: float (0.0 to 1.0, normalized)
            - corner_detected: bool
            - corner_direction: str ('left', 'right', None)
            - corner_angle: float (angle in degrees)
            - avoidance_action: str (recommended action)
            - processed_frame: frame with debug visualization (if debug=True)
        """
        if frame is None:
            return self._empty_result()
        
        height, width = frame.shape[:2]
        self.frames_processed += 1
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use a simple global binary threshold. This is more effective for
        # detecting a solid black line on a lighter background.
        # Pixels darker than 100 will be considered the line.
        _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

        # Use morphological closing to connect any small gaps in the line
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Detect obstacles in the upper portion of the frame
        obstacle_result = self._detect_obstacles(closed, width, height)
        
        # Detect corners in the lower portion of the frame
        corner_result = self._detect_corners(closed, width, height)
        
        # Combine results
        result = {
            'obstacle_detected': obstacle_result['detected'],
            'obstacle_direction': obstacle_result['direction'],
            'obstacle_distance': obstacle_result['distance'],
            'corner_detected': corner_result['detected'],
            'corner_direction': corner_result['direction'],
            'corner_angle': corner_result['angle'],
            'avoidance_action': self._determine_action(obstacle_result, corner_result),
            'frame_number': self.frames_processed
        }
        
        # Create debug visualization if requested
        if self.debug:
            self.debug_frame_counter += 1
            if self.debug_frame_counter % (self.debug_frame_skip + 1) == 0:
                result['processed_frame'] = self._create_debug_frame(
                    frame, blurred, obstacle_result, corner_result, result
                )
            else:
                result['processed_frame'] = frame
        else:
            result['processed_frame'] = frame
        
        return result
    
    def _detect_obstacles(self, gray_frame: np.ndarray, width: int, height: int) -> Dict:
        """
        Detect obstacles in the forward view using edge detection and contour analysis.
        """
        # Define ROI for obstacle detection (upper portion of frame)
        roi_height = int(height * self.obstacle_roi_height_ratio)
        obstacle_roi = gray_frame[0:roi_height, :]
        
        # Apply edge detection to find obstacles
        edges = cv2.Canny(obstacle_roi, 50, 150, apertureSize=3)
        
        # Morphological operations to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours (potential obstacles)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacle_detected = False
        obstacle_direction = None
        obstacle_distance = 1.0  # Far away
        largest_obstacle = None
        
        if contours:
            # Filter contours by area
            valid_obstacles = [c for c in contours if cv2.contourArea(c) >= self.min_obstacle_area]
            
            if valid_obstacles:
                # Find the largest obstacle (closest/most significant)
                largest_obstacle = max(valid_obstacles, key=cv2.contourArea)
                
                # Calculate obstacle properties
                x, y, w, h = cv2.boundingRect(largest_obstacle)
                obstacle_center_x = x + w // 2
                obstacle_area = cv2.contourArea(largest_obstacle)
                
                # Determine direction relative to frame center
                frame_center = width // 2
                if obstacle_center_x < frame_center - width * 0.2:
                    obstacle_direction = 'left'
                elif obstacle_center_x > frame_center + width * 0.2:
                    obstacle_direction = 'right'
                else:
                    obstacle_direction = 'center'
                
                # Estimate distance based on obstacle size and position
                # Larger obstacles or those lower in frame are closer
                max_area = roi_height * width * 0.3  # 30% of ROI area
                size_factor = min(1.0, obstacle_area / max_area)
                position_factor = (roi_height - y) / roi_height  # Lower in frame = closer
                
                obstacle_distance = 1.0 - (size_factor * 0.7 + position_factor * 0.3)
                obstacle_distance = max(0.0, min(1.0, obstacle_distance))
                
                # Consider it detected if distance is below threshold
                if obstacle_distance < self.obstacle_distance_threshold:
                    obstacle_detected = True
        
        return {
            'detected': obstacle_detected,
            'direction': obstacle_direction,
            'distance': obstacle_distance,
            'contour': largest_obstacle,
            'roi': edges
        }
    
    def _detect_corners(self, gray_frame: np.ndarray, width: int, height: int) -> Dict:
        """
        Detect corners ahead using line detection and angle analysis.
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_corner_time < self.corner_cooldown:
            return {'detected': False, 'direction': None, 'angle': 0.0, 'lines': []}
        
        # Define ROI for corner detection (lower portion of frame)
        roi_start = int(height * (1.0 - self.corner_roi_height_ratio))
        corner_roi = gray_frame[roi_start:height, :]
        
        # Apply edge detection
        edges = cv2.Canny(corner_roi, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=self.corner_line_length, maxLineGap=10)
        
        corner_detected = False
        corner_direction = None
        corner_angle = 0.0
        detected_lines = []
        
        if lines is not None and len(lines) >= 2:
            # Convert lines to angles
            line_angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                line_angles.append(angle)
                detected_lines.append((x1, y1 + roi_start, x2, y2 + roi_start))
            
            # Find pairs of lines that form corners
            for i in range(len(line_angles)):
                for j in range(i + 1, len(line_angles)):
                    angle_diff = abs(line_angles[i] - line_angles[j])
                    
                    # Normalize angle difference to 0-90 degrees
                    if angle_diff > 90:
                        angle_diff = 180 - angle_diff
                    
                    # Check if lines form a corner
                    if angle_diff > self.corner_angle_threshold:
                        corner_detected = True
                        corner_angle = angle_diff
                        
                        # Determine corner direction based on line orientations
                        avg_angle = (line_angles[i] + line_angles[j]) / 2
                        if avg_angle < -10:
                            corner_direction = 'right'
                        elif avg_angle > 10:
                            corner_direction = 'left'
                        else:
                            # Determine based on line positions
                            line1 = lines[i][0]
                            line2 = lines[j][0]
                            avg_x = (line1[0] + line1[2] + line2[0] + line2[2]) / 4
                            corner_direction = 'left' if avg_x < width // 2 else 'right'
                        
                        self.last_corner_time = current_time
                        break
                
                if corner_detected:
                    break
        
        return {
            'detected': corner_detected,
            'direction': corner_direction,
            'angle': corner_angle,
            'lines': detected_lines,
            'roi': edges
        }
    
    def _determine_action(self, obstacle_result: Dict, corner_result: Dict) -> str:
        """
        Determine the recommended avoidance action based on detections.
        """
        obstacle_detected = obstacle_result['detected']
        corner_detected = corner_result['detected']
        
        # Priority: Obstacle avoidance first, then corner navigation
        if obstacle_detected:
            obstacle_dir = obstacle_result['direction']
            distance = obstacle_result['distance']
            
            if distance < 0.15:  # Very close obstacle
                if obstacle_dir == 'center':
                    return 'stop_and_backup'
                elif obstacle_dir == 'left':
                    return 'turn_right_sharp'
                else:  # right
                    return 'turn_left_sharp'
            elif distance < 0.25:  # Close obstacle
                if obstacle_dir == 'center':
                    return 'turn_around'
                elif obstacle_dir == 'left':
                    return 'turn_right'
                else:  # right
                    return 'turn_left'
            else:  # Moderate distance
                if obstacle_dir == 'left':
                    return 'steer_right'
                elif obstacle_dir == 'right':
                    return 'steer_left'
                else:
                    return 'slow_down'
        
        elif corner_detected:
            corner_dir = corner_result['direction']
            angle = corner_result['angle']
            
            if angle > 60:  # Sharp corner
                return f'turn_{corner_dir}_sharp'
            else:  # Gentle corner
                return f'turn_{corner_dir}_gentle'
        
        else:
            return 'continue_forward'
    
    def get_motor_speeds(self, detection_result: Dict, base_speed: int = 40) -> Tuple[int, int, int, int]:
        """
        Convert detection results to motor speeds for obstacle avoidance and corner navigation.
        
        Args:
            detection_result: Result from detect_obstacles_and_corners()
            base_speed: Base forward speed (0-100)
            
        Returns:
            Tuple of motor speeds (fl, fr, bl, br)
        """
        action = detection_result['avoidance_action']
        
        # Motor speed mappings for different actions
        if action == 'stop_and_backup':
            return (-base_speed//2, -base_speed//2, -base_speed//2, -base_speed//2)
        
        elif action == 'turn_right_sharp':
            turn_speed = base_speed // 2
            return (turn_speed, -turn_speed, turn_speed, -turn_speed)
        
        elif action == 'turn_left_sharp':
            turn_speed = base_speed // 2
            return (-turn_speed, turn_speed, -turn_speed, turn_speed)
        
        elif action == 'turn_right':
            turn_speed = base_speed // 3
            return (turn_speed, -turn_speed//2, turn_speed, -turn_speed//2)
        
        elif action == 'turn_left':
            turn_speed = base_speed // 3
            return (-turn_speed//2, turn_speed, -turn_speed//2, turn_speed)
        
        elif action == 'steer_right':
            return (base_speed, base_speed//2, base_speed, base_speed//2)
        
        elif action == 'steer_left':
            return (base_speed//2, base_speed, base_speed//2, base_speed)
        
        elif action == 'slow_down':
            slow_speed = base_speed // 2
            return (slow_speed, slow_speed, slow_speed, slow_speed)
        
        elif action == 'turn_around':
            turn_speed = base_speed // 2
            return (-turn_speed, turn_speed, -turn_speed, turn_speed)
        
        elif action in ['turn_left_gentle', 'turn_right_gentle']:
            direction = 1 if 'right' in action else -1
            turn_speed = base_speed // 4
            forward_speed = base_speed * 3 // 4
            
            # Gentle turn while moving forward
            left_speed = forward_speed - (direction * turn_speed)
            right_speed = forward_speed + (direction * turn_speed)
            return (left_speed, right_speed, left_speed, right_speed)
        
        else:  # continue_forward
            return (base_speed, base_speed, base_speed, base_speed)
    
    def _create_debug_frame(self, frame: np.ndarray, enhanced: np.ndarray, 
                           obstacle_result: Dict, corner_result: Dict, 
                           detection_result: Dict) -> np.ndarray:
        """
        Create debug visualization frame showing obstacle and corner detection.
        """
        debug_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw ROI boundaries
        obstacle_roi_end = int(height * self.obstacle_roi_height_ratio)
        corner_roi_start = int(height * (1.0 - self.corner_roi_height_ratio))
        
        cv2.line(debug_frame, (0, obstacle_roi_end), (width, obstacle_roi_end), (255, 0, 0), 2)
        cv2.line(debug_frame, (0, corner_roi_start), (width, corner_roi_start), (0, 255, 0), 2)
        
        # Draw obstacle detection
        if obstacle_result['detected']:
            contour = obstacle_result['contour']
            if contour is not None:
                cv2.drawContours(debug_frame, [contour], -1, (0, 0, 255), 3)
                
                # Draw obstacle info
                x, y, w, h = cv2.boundingRect(contour)
                obstacle_dir = obstacle_result['direction']
                distance = obstacle_result['distance']
                
                cv2.putText(debug_frame, f"OBSTACLE: {obstacle_dir}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(debug_frame, f"Dist: {distance:.2f}", 
                           (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw corner detection
        if corner_result['detected']:
            lines = corner_result['lines']
            for line in lines:
                x1, y1, x2, y2 = line
                cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            
            corner_dir = corner_result['direction']
            angle = corner_result['angle']
            cv2.putText(debug_frame, f"CORNER: {corner_dir} ({angle:.1f}°)", 
                       (10, corner_roi_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw action and status
        action = detection_result['avoidance_action']
        frame_num = detection_result['frame_number']
        
        cv2.putText(debug_frame, f"Action: {action}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_frame, f"Frame: {frame_num}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw center line
        cv2.line(debug_frame, (width//2, 0), (width//2, height), (128, 128, 128), 1)
        
        return debug_frame
    
    def _empty_result(self) -> Dict:
        """Return empty result when frame is None."""
        return {
            'obstacle_detected': False,
            'obstacle_direction': None,
            'obstacle_distance': 1.0,
            'corner_detected': False,
            'corner_direction': None,
            'corner_angle': 0.0,
            'avoidance_action': 'continue_forward',
            'processed_frame': None,
            'frame_number': self.frames_processed
        } 

class CameraLineFollower:
    """
    Camera-based line following system for autonomous robot navigation.
    Detects black lines and provides steering corrections for line following.
    """
    
    def __init__(self, camera_index=1, width=720, height=480, fps=30, debug=False):
        self.debug = debug
        
        # Enhanced line detection parameters for better centering accuracy
        self.BLACK_THRESHOLD = 80  # Threshold for detecting black lines
        self.BLUR_SIZE = (5, 5)
        self.MIN_CONTOUR_AREA = 500
        self.MIN_LINE_WIDTH = 10
        self.MAX_LINE_WIDTH = 200
        
        # Enhanced centering parameters
        self.CENTERING_PRECISION_MODE = True  # Enable high-precision centering
        self.MULTI_THRESHOLD_DETECTION = True  # Use multiple thresholds for better line detection
        
        # Morphological operations kernel
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # LOOK-AHEAD ZONE CONFIGURATION
        # Far zone: For detecting upcoming intersections and obstacles
        self.FAR_ZONE_START = 0.10    # Top 10% to 40% for look-ahead
        self.FAR_ZONE_END = 0.40
        
        # Near zone: For actual line following (where robot currently is)
        self.NEAR_ZONE_START = 0.60   # 60% to 80% for current position
        self.NEAR_ZONE_END = 0.80
        
        # Middle zone (40% to 60%) is ignored - transition area
        
        # Look-ahead timing parameters
        self.FRAMES_TO_INTERSECTION = 15  # Frames between detection and arrival
        self.intersection_countdown = 0
        self.upcoming_intersection_type = None  # 'left', 'right', 'T', or None
        
        # Box detection look-ahead
        self.box_detected_ahead = False
        self.box_countdown = 0
        self.FRAMES_TO_BOX = 20  # Frames between box detection and arrival
        
        # Box approach parameters
        self.box_approach_active = False
        self.box_in_position = False
        self.BOX_PICKUP_ZONE = 0.90  # Box should be in bottom 10% of frame for pickup
        
        # Blue and yellow box detection - primarily blue
        self.BOX_COLOR_RANGES = [
            # Blue ranges (primary color)
            (np.array([100, 100, 50]), np.array([130, 255, 255])),  # Standard blue
            (np.array([90, 50, 50]), np.array([120, 255, 255])),    # Wider blue range
            # Yellow ranges (secondary color)
            (np.array([20, 100, 100]), np.array([35, 255, 255])),   # Yellow
            (np.array([15, 50, 50]), np.array([40, 255, 255])),     # Wider yellow range
        ]
        self.MIN_BOX_AREA = 500  # Minimum area to consider as box
        
        # Path correction parameters
        self.path_correction_enabled = True
        self.turn_direction_pattern = ['right', 'right', 'left', 'right']  # Example pattern
        self.current_turn_index = 0
        self.intersections_seen = 0
        
        # Region of Interest (ROI) settings - DEPRECATED but kept for compatibility
        self.ROI_START_RATIO = 0.30   # Start ROI at 30% down the frame 
        self.ARM_EXCLUSION_RATIO = 0.80  # Bottom 20% is arm area
        
        # ADAPTIVE THRESHOLDING CONFIGURATION
        # Enable/disable different thresholding methods
        self.ADAPTIVE_THRESH_ENABLED = True
        self.SIMPLE_THRESH_ENABLED = True  
        self.HSV_THRESH_ENABLED = True
        
        # Adaptive threshold parameters - highly configurable
        self.ADAPTIVE_METHOD = cv2.ADAPTIVE_THRESH_GAUSSIAN_C  # or cv2.ADAPTIVE_THRESH_MEAN_C
        self.ADAPTIVE_THRESH_TYPE = cv2.THRESH_BINARY_INV     # Inverted for black lines
        self.ADAPTIVE_BLOCK_SIZE = 11      # Size of neighborhood area (must be odd)
        self.ADAPTIVE_C_CONSTANT = 5       # Constant subtracted from mean/gaussian
        
        # Alternative adaptive parameters for different lighting conditions
        self.ADAPTIVE_PARAMS_BRIGHT = {
            'block_size': 15,
            'c_constant': 8,
            'method': cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        }
        
        self.ADAPTIVE_PARAMS_DIM = {
            'block_size': 9,
            'c_constant': 3,
            'method': cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        }
        
        self.ADAPTIVE_PARAMS_NORMAL = {
            'block_size': 11,
            'c_constant': 5,
            'method': cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        }
        
        # Current adaptive parameters (can be switched dynamically)
        self.current_adaptive_params = self.ADAPTIVE_PARAMS_NORMAL
        
        # Auto-adaptation based on frame brightness
        self.AUTO_ADAPTIVE_ENABLED = True
        self.BRIGHTNESS_THRESHOLD_BRIGHT = 120  # Average brightness above this = bright conditions
        self.BRIGHTNESS_THRESHOLD_DIM = 80      # Average brightness below this = dim conditions
        
        # Simple threshold parameters
        self.SIMPLE_THRESHOLD_VALUE = 60       # Fixed threshold value
        self.SIMPLE_THRESH_TYPE = cv2.THRESH_BINARY_INV
        
        # ARM FILTERING CONFIGURATION
        # Enable/disable purple arm filtering
        self.ARM_FILTERING_ENABLED = False
        
        # Purple color range in HSV for arm detection and filtering
        # Expanded ranges to better catch the bright purple arm
        self.ARM_PURPLE_LOWER = np.array([120, 50, 50])   # Lower HSV bound for purple
        self.ARM_PURPLE_UPPER = np.array([160, 255, 255]) # Upper HSV bound for purple
        
        # Alternative purple ranges (you can enable multiple ranges if needed)
        self.ARM_PURPLE_RANGES = [
            (np.array([120, 50, 50]), np.array([160, 255, 255])),   # Primary purple range
            (np.array([140, 30, 30]), np.array([170, 255, 200])),   # Extended purple range
            (np.array([140, 80, 100]), np.array([170, 255, 255])),  # Bright magenta range
            (np.array([145, 100, 150]), np.array([165, 255, 255])), # Very bright purple
        ]
        
        # Minimum area for arm detection (pixels) - reduced to catch smaller arm regions
        self.ARM_MIN_AREA = 30
        
        # Maximum area ratio for arm (to avoid filtering large purple obstacles)
        self.ARM_MAX_AREA_RATIO = 0.4  # Max 50% of ROI area
        
        # Arm position constraints (where we expect the arm to appear) - expanded
        self.ARM_EXPECTED_REGION = {
            'bottom_ratio': 0.5,    # Arm typically appears in bottom 50% of ROI
            'center_tolerance': 0.6  # Arm typically appears within 60% of center
        }
        
        # Line following parameters - REDUCED for immediate response
        self.center_offset_history = []
        self.history_size = 3  # Reduced to 3 for faster response to changes
        
        # Stuck detection to reset bias when not making progress
        self.stuck_detection_history = []
        self.stuck_detection_size = 6  # Reduced from 10 to 6 for faster detection
        self.stuck_threshold = 0.03  # Reduced threshold for earlier detection
        self.last_reset_time = 0  # Track when we last reset to prevent spam
        
        # Corner detection parameters
        self.CORNER_THRESHOLD = 0.4  # Line offset ratio to detect corner
        self.SHARP_CORNER_THRESHOLD = 0.6
        self.corner_detected = False
        self.corner_direction = None
        self.corner_confidence = 0.0
        
        # Line loss recovery
        self.line_lost_counter = 0
        self.MAX_LINE_LOST_FRAMES = 10
        self.search_direction = 0  # -1 for left, 1 for right, 0 for center
        self.last_turn_correction = 0  # Track last turn correction for recovery
        
        # Performance tracking
        self.last_detection_time = 0
        self.detection_fps = 0
        
        # Enhanced Adaptive PID controller for IMMEDIATE response to any drift
        # Tuned for aggressive correction as soon as robot moves off center
        self.pid = AdaptivePIDController(
            base_kp=2.0, base_ki=0.08, base_kd=0.6,  # Increased all gains for immediate response
            output_limits=(-60, 60),  # Increased output limits for stronger corrections
            sample_time=0.033,  # ~30 FPS
            debug=self.debug
        )
        
        # Detection history for smoothing
        self.line_offset_history = deque(maxlen=5)
        
        # Temporal filtering for noise reduction
        self.mask_history = deque(maxlen=3)
        
        # Result cache to avoid re-computation
        self.last_frame_hash = None
        self.last_result = None
        
        # Corner stopping system
        self.corner_stop_enabled = True
        self.corner_stop_duration = 0.2  # Stop for 0.2 seconds at each corner (minimal pause)
        self.corner_stop_start_time = None
        self.is_stopping_at_corner = False
        self.last_intersection_state = False
        self.corner_count = 0
        self.corners_passed = []  # Track which corners we've passed
        self.last_intersection_time = 0  # Time when we last detected an intersection
        self.intersection_cooldown = 0.5  # Very short cooldown - only prevent duplicate detections at same intersection
        self.intersection_detected_for_main = False  # Signal to main controller that intersection was detected
        
        self.max_line_width = 0.7 # Max line width as a ratio of ROI width
        self.intersection_solidity_threshold = 0.4 # VERY LOW threshold - catch any complex shapes
        self.intersection_aspect_ratio_threshold = 0.6 # VERY LOW threshold - catch any wide shapes  
        self.intersection_area_threshold = 400 # VERY LOW area threshold - catch small intersections
        
        self.canny_high = 150
        self.binary_threshold = 60
        
        # Line detection parameters
        self.min_line_area = 20  # Minimum contour area to be considered a line (reduced for better sensitivity)
        
        # Camera properties
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.camera_initialized = False
        self.frames_processed = 0
        
        # Speed smoothing for more consistent movement
        self.speed_history = deque(maxlen=3)  # Track last 3 speeds for smoothing
        self.last_base_speed = 60  # Track last base speed
        self.last_calculated_speed = 60  # Initialize last calculated speed
        
        # Line Memory Buffer for lookahead with memory
        self.line_memory_buffer = LineMemoryBuffer(
            buffer_size=20,
            max_prediction_time=2.0,
            debug=self.debug
        )

    def initialize_camera(self) -> bool:
        """Initializes the camera using picamera2."""
        if not Picamera2:
            print("ERROR: Cannot initialize camera, picamera2 library not available.")
            return False
        if self.camera_initialized:
            return True
        
        try:
            print("Initializing camera for line following...")
            self.cap = Picamera2()
            config = self.cap.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.cap.configure(config)
            self.cap.start()
            self.camera_initialized = True
            print("Camera initialized successfully.")
            # Allow some time for sensor to warm up
            time.sleep(1.0)
            return True
        except Exception as e:
            print(f"CRITICAL: Failed to initialize camera: {e}")
            self.cap = None
            self.camera_initialized = False
            return False

    def release_camera(self):
        """Releases the camera resource."""
        if self.cap and self.camera_initialized:
            try:
                print("Releasing camera...")
                self.cap.stop()
                self.cap.close()
                print("Camera released.")
            except Exception as e:
                print(f"Error releasing camera: {e}")
        self.cap = None
        self.camera_initialized = False
    
    def get_camera_frame(self) -> Optional[np.ndarray]:
        """Captures and returns a single frame from the camera."""
        if not self.camera_initialized or not self.cap:
            return None
        
        try:
            frame = self.cap.capture_array()
            return frame
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None

    def detect_line_with_lookahead(self, frame: Optional[np.ndarray] = None) -> Dict:
        """
        Simplified line detection with look-ahead strategy.
        Uses two zones: far zone for upcoming events, near zone for current line following.
        """
        self.frames_processed += 1
        
        if frame is None:
            frame = self.get_camera_frame()
            if frame is None:
                return self._get_fallback_result()

        height, width, _ = frame.shape
        
        # Define zones
        far_zone_start = int(height * self.FAR_ZONE_START)
        far_zone_end = int(height * self.FAR_ZONE_END)
        near_zone_start = int(height * self.NEAR_ZONE_START)
        near_zone_end = int(height * self.NEAR_ZONE_END)
        
        # Extract zones
        far_zone = frame[far_zone_start:far_zone_end, :]
        near_zone = frame[near_zone_start:near_zone_end, :]
        
        # Process near zone for line following
        near_zone_result = self._process_zone_simple(near_zone, zone_type='near')
        
        # Process far zone for intersection detection
        far_zone_result = self._process_zone_simple(far_zone, zone_type='far')
        
        # Update countdowns
        if self.intersection_countdown > 0:
            self.intersection_countdown -= 1
            if self.debug and self.intersection_countdown % 5 == 0:
                print(f"Intersection arriving in {self.intersection_countdown} frames")
        
        if self.box_countdown > 0:
            self.box_countdown -= 1
            
        # Check for new intersection in far zone
        if far_zone_result.get('is_intersection', False) and self.intersection_countdown == 0:
            self.intersection_countdown = self.FRAMES_TO_INTERSECTION
            self.upcoming_intersection_type = far_zone_result.get('intersection_type', 'T')
            if self.debug:
                print(f"Intersection detected ahead! Type: {self.upcoming_intersection_type}")
        
        # Check for box detection
        box_in_far = far_zone_result.get('box_detected', False)
        box_in_near = near_zone_result.get('box_detected', False)
        
        # Handle box approach logic
        if box_in_far and not self.box_approach_active:
            # Start box approach when first detected in far zone
            self.box_approach_active = True
            if self.debug:
                print("Box detected ahead! Starting approach...")
        
        # Check if box is in pickup position (very bottom of near zone)
        if box_in_near and near_zone_result.get('box_info', {}).get('in_pickup_position', False):
            self.box_in_position = True
            if self.debug:
                box_y_ratio = near_zone_result['box_info']['bottom_y_ratio']
                print(f"Box in pickup position! Bottom at {box_y_ratio:.2f} of near zone")
        else:
            self.box_in_position = False
        
        # Build combined result
        result = {
            'line_detected': near_zone_result.get('line_detected', False),
            'line_offset': near_zone_result.get('line_offset', 0.0),
            'line_center_x': near_zone_result.get('line_center_x', width // 2),
            'line_confidence': near_zone_result.get('confidence', 0.0),
            'intersection_ahead': self.intersection_countdown > 0,
            'intersection_countdown': self.intersection_countdown,
            'intersection_now': self.intersection_countdown == 1,  # Next frame will be at intersection
            'upcoming_intersection_type': self.upcoming_intersection_type,
            'far_zone_intersection': far_zone_result.get('is_intersection', False),
            'box_detected': box_in_near or box_in_far,
            'box_in_far': box_in_far,
            'box_in_near': box_in_near,
            'box_approach_active': self.box_approach_active,
            'box_in_position': self.box_in_position,
            'box_info': near_zone_result.get('box_info', {}) if box_in_near else far_zone_result.get('box_info', {}),
            'status': 'box_approach' if self.box_approach_active else ('line_following' if near_zone_result.get('line_detected') else 'line_lost')
        }
        
        # Update memory buffer with current detection for future predictions
        robot_state = self._get_default_robot_state()
        self.line_memory_buffer.update_line_detection(result, robot_state, {})
        
        # Create debug visualization
        if self.debug:
            result['processed_frame'] = self._draw_lookahead_debug(
                frame, near_zone_result, far_zone_result, 
                near_zone_start, near_zone_end, far_zone_start, far_zone_end
            )
        
        return result
    
    def _process_zone_simple(self, zone: np.ndarray, zone_type: str = 'near') -> Dict:
        """
        Simple zone processing for either near (line following) or far (intersection detection).
        Also detects boxes for pickup.
        """
        if zone is None or zone.shape[0] == 0:
            return {'line_detected': False, 'is_intersection': False, 'box_detected': False}
            
        # Convert to grayscale and threshold for line detection
        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, self.BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        
        # Basic morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours for line
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Also check for box (yellow object)
        box_detected, box_info = self._detect_box_in_zone(zone)
        
        if not contours:
            return {
                'line_detected': False, 
                'is_intersection': False,
                'box_detected': box_detected,
                'box_info': box_info
            }
        
        # Filter small contours
        min_area = 100 if zone_type == 'near' else 200
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        if not valid_contours:
            return {
                'line_detected': False, 
                'is_intersection': False,
                'box_detected': box_detected,
                'box_info': box_info
            }
        
        # Get largest contour
        largest_contour = max(valid_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        zone_height, zone_width = zone.shape[:2]
        
        # Calculate line center
        line_center_x = x + w // 2
        line_offset = (line_center_x - zone_width // 2) / (zone_width // 2)
        
        # Simple intersection detection
        is_intersection = False
        intersection_type = None
        
        if zone_type == 'far':
            # Check for intersection patterns
            width_ratio = w / zone_width
            height_ratio = h / zone_height
            
            # T-intersection: very wide line
            if width_ratio > 0.7:
                is_intersection = True
                intersection_type = 'T'
            # Multiple large contours might indicate intersection
            elif len(valid_contours) >= 2:
                second_largest = sorted(valid_contours, key=cv2.contourArea, reverse=True)[1]
                if cv2.contourArea(second_largest) > cv2.contourArea(largest_contour) * 0.5:
                    is_intersection = True
                    intersection_type = 'split'
        
        return {
            'line_detected': True,
            'line_center_x': line_center_x,
            'line_offset': line_offset,
            'confidence': 1.0,
            'is_intersection': is_intersection,
            'intersection_type': intersection_type,
            'contour': largest_contour,
            'bbox': (x, y, w, h),
            'box_detected': box_detected,
            'box_info': box_info
        }
    
    def _detect_box_in_zone(self, zone: np.ndarray) -> Tuple[bool, Dict]:
        """
        Detect blue/yellow box in the given zone.
        Returns (box_detected, box_info)
        """
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
        
        # Create combined mask for all color ranges
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        # Add each color range to the combined mask
        for lower, upper in self.BOX_COLOR_RANGES:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, {}
        
        # Find largest colored contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < self.MIN_BOX_AREA:
            return False, {}
        
        # Get box properties
        x, y, w, h = cv2.boundingRect(largest_contour)
        zone_height, zone_width = zone.shape[:2]
        
        # Calculate box position relative to zone
        box_center_x = x + w // 2
        box_center_y = y + h // 2
        box_bottom_y = y + h
        
        # Check if box is in pickup position (bottom of frame) - just a bit more contact needed
        # Use two thresholds: close enough for approach, and very close for pickup
        close_to_pickup = (box_bottom_y / zone_height) > 0.78  # Start push a bit earlier
        in_pickup_position = (box_bottom_y / zone_height) > 0.995  # Must be at absolute edge of frame (just 2cm more)
        
        # Determine dominant color (blue or yellow)
        roi = hsv[y:y+h, x:x+w]
        blue_pixels = 0
        yellow_pixels = 0
        
        # Count blue pixels
        for lower, upper in self.BOX_COLOR_RANGES[:2]:  # First two are blue ranges
            mask = cv2.inRange(roi, lower, upper)
            blue_pixels += cv2.countNonZero(mask)
        
        # Count yellow pixels
        for lower, upper in self.BOX_COLOR_RANGES[2:]:  # Last two are yellow ranges
            mask = cv2.inRange(roi, lower, upper)
            yellow_pixels += cv2.countNonZero(mask)
        
        dominant_color = 'blue' if blue_pixels > yellow_pixels else 'yellow'
        
        return True, {
            'bbox': (x, y, w, h),
            'area': area,
            'center': (box_center_x, box_center_y),
            'center_x_ratio': box_center_x / zone_width,  # Add horizontal position ratio
            'bottom_y_ratio': box_bottom_y / zone_height,
            'close_to_pickup': close_to_pickup,  # Add intermediate threshold
            'in_pickup_position': in_pickup_position,
            'contour': largest_contour,
            'dominant_color': dominant_color,
            'color_ratio': blue_pixels / (blue_pixels + yellow_pixels + 1)  # Avoid division by zero
        }
    
    def _draw_lookahead_debug(self, frame, near_result, far_result, 
                              near_start, near_end, far_start, far_end):
        """Draw debug overlay showing both zones."""
        debug_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw zone boundaries
        cv2.line(debug_frame, (0, far_start), (width, far_start), (255, 255, 0), 2)
        cv2.line(debug_frame, (0, far_end), (width, far_end), (255, 255, 0), 2)
        cv2.putText(debug_frame, "FAR ZONE (Look-ahead)", (10, far_start + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.line(debug_frame, (0, near_start), (width, near_start), (0, 255, 0), 2)
        cv2.line(debug_frame, (0, near_end), (width, near_end), (0, 255, 0), 2)
        cv2.putText(debug_frame, "NEAR ZONE (Current)", (10, near_start + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw center line
        cv2.line(debug_frame, (width//2, 0), (width//2, height), (128, 128, 128), 1)
        
        # Draw detected lines in each zone
        if near_result.get('line_detected'):
            x, y, w, h = near_result['bbox']
            cv2.rectangle(debug_frame, (x, y + near_start), (x + w, y + h + near_start), 
                         (0, 255, 0), 2)
            center_x = x + w // 2
            cv2.circle(debug_frame, (center_x, near_start + (near_end - near_start)//2), 
                      5, (0, 255, 0), -1)
        
        if far_result.get('line_detected'):
            x, y, w, h = far_result['bbox']
            cv2.rectangle(debug_frame, (x, y + far_start), (x + w, y + h + far_start), 
                         (255, 255, 0), 2)
            if far_result.get('is_intersection'):
                cv2.putText(debug_frame, f"INTERSECTION: {far_result.get('intersection_type', 'unknown')}", 
                           (x, y + far_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw box detection
        if near_result.get('box_detected'):
            box_info = near_result['box_info']
            x, y, w, h = box_info['bbox']
            cv2.rectangle(debug_frame, (x, y + near_start), (x + w, y + h + near_start), 
                         (0, 165, 255), 3)  # Orange for box
            cv2.putText(debug_frame, f"BOX {box_info['bottom_y_ratio']:.2f}", 
                       (x, y + near_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            if box_info.get('in_pickup_position'):
                cv2.putText(debug_frame, "PICKUP POSITION!", 
                           (x, y + h + near_start + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if far_result.get('box_detected'):
            box_info = far_result['box_info']
            x, y, w, h = box_info['bbox']
            cv2.rectangle(debug_frame, (x, y + far_start), (x + w, y + h + far_start), 
                         (0, 165, 255), 2)  # Orange for box
            cv2.putText(debug_frame, "BOX AHEAD", 
                       (x, y + far_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Draw status
        status_y = 30
        cv2.putText(debug_frame, f"Line Offset: {near_result.get('line_offset', 0):.3f}", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.intersection_countdown > 0:
            cv2.putText(debug_frame, f"INTERSECTION IN: {self.intersection_countdown} frames", 
                       (10, status_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return debug_frame

    def detect_line(self, frame: Optional[np.ndarray] = None, robot_state: Dict = None, encoder_counts: Dict = None) -> Dict:
        """
        Main line detection function with memory buffer integration.
        Takes a raw camera frame, processes it, and returns line following data.
        Includes lookahead capability using buffered line positions when direct detection fails.
        """
        self.frames_processed += 1
        
        if frame is None:
            frame = self.get_camera_frame()
            if frame is None:
                return self._get_fallback_result()

        height, width, _ = frame.shape
        
        # 1. Define and Extract Region of Interest (ROI)
        roi_start_y = int(height * self.ROI_START_RATIO)
        roi_end_y = int(height * self.ARM_EXCLUSION_RATIO)
        roi = frame[roi_start_y:roi_end_y, :]

        # 2. Preprocess ROI to get a clean binary mask
        final_mask = self._preprocess_roi(roi)

        # 3. Find line and intersection information from the mask
        line_info = self._find_line_in_roi(final_mask)
        
        # 4. Calculate final motor control parameters
        result = self._calculate_line_following_params(line_info, width, roi_start_y)
        if line_info:
            result.update(line_info)

        # 5. Update line memory buffer with current detection and robot state
        if robot_state is None:
            robot_state = self._get_default_robot_state()
        
        self.line_memory_buffer.update_line_detection(result, robot_state, encoder_counts)
        
        # 6. If direct line detection failed, try to get prediction from memory buffer
        if not result.get('line_detected', False) or result.get('line_confidence', 0) < 0.3:
            predicted_result = self.line_memory_buffer.get_predicted_line_state(width)
            if predicted_result:
                # Merge predicted result with current result, keeping some original data
                result.update(predicted_result)
                result['using_prediction'] = True
                
                # Check if we're in severe offset recovery mode
                line_offset = result.get('line_offset', 0)
                if abs(line_offset) > 0.4:
                    result['severe_offset_recovery'] = True
                    if self.debug:
                        print(f"CAMERA LINE FOLLOWER: Severe offset recovery mode - offset: {line_offset:.3f}")
                
                if self.debug:
                    print(f"CAMERA LINE FOLLOWER: Using predicted line state - conf: {predicted_result['line_confidence']:.2f}")
            else:
                result['using_prediction'] = False
        else:
            result['using_prediction'] = False

        # 7. Add buffer status to result for monitoring
        buffer_status = self.line_memory_buffer.get_buffer_status()
        result['buffer_status'] = buffer_status
        
        # 8. Create debug view
        if self.debug:
            result['processed_frame'] = self._draw_debug_overlay(frame, result, roi_start_y, final_mask)

        return result

    def _preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        Takes the raw ROI frame and converts it into a clean, binary mask 
        of the line. This is the heart of the vision pipeline.
        """
        # 1. Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply a Gaussian blur to reduce noise before thresholding
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Very sensitive threshold - detect anything darker than 180 as black line
        _, binary_roi = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)

        # 4. Apply arm filtering if enabled
        if self.ARM_FILTERING_ENABLED:
            binary_roi = self._filter_out_robot_arm(roi, binary_roi)

        # --- SCANLINE-BASED GAP FILLING ---
        # This approach fills gaps between left and right edges on each row
        height, width = binary_roi.shape
        filled_roi = binary_roi.copy()
        
        # Process each row to find and fill gaps between edges
        for y in range(height):
            row = binary_roi[y, :]
            white_pixels = np.where(row == 255)[0]
            
            if len(white_pixels) >= 2:
                # Find leftmost and rightmost white pixels
                left_edge = white_pixels[0]
                right_edge = white_pixels[-1]
                
                # Only fill if there's a reasonable gap (not too wide to be noise)
                gap_width = right_edge - left_edge
                if gap_width > 10 and gap_width < width * 0.8:  # Reasonable lane width
                    # Fill the entire span between edges
                    filled_roi[y, left_edge:right_edge+1] = 255
        
        # --- ENHANCED CORNER CLEANUP ---
        # Remove corner blocks more aggressively
        
        # Find contours after gap filling
        contours, _ = cv2.findContours(filled_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) >= 1:
            # Sort contours by area - largest is likely the main lane
            contours_by_area = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Remove ALL smaller contours that could be corner artifacts
            for i, contour in enumerate(contours_by_area):
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter out very small contours (noise) - reduced threshold for more sensitivity
                min_contour_area = 25  # Reduced from 50 to 25 to catch smaller intersection features
                if area < min_contour_area:
                    cv2.fillPoly(filled_roi, [contour], 0)
        
        # Apply morphological closing to smooth connections
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        filled_roi = cv2.morphologyEx(filled_roi, cv2.MORPH_CLOSE, kernel_vertical)
        
        # Final horizontal closing to ensure solid lanes
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        final_roi = cv2.morphologyEx(filled_roi, cv2.MORPH_CLOSE, kernel_horizontal)

        return final_roi

    def _apply_temporal_filter(self) -> np.ndarray:
        """
        Applies a temporal filter to the mask history to smooth out noise.
        """
        if len(self.mask_history) < 2:
            return self.mask_history[-1]
        
        # Calculate the average of the last two masks
        last_mask = self.mask_history[-1]
        second_last_mask = self.mask_history[-2]
        
        averaged_mask = (last_mask + second_last_mask) / 2
        
        return averaged_mask

    def _filter_out_robot_arm(self, roi: np.ndarray, binary_roi: np.ndarray) -> np.ndarray:
        """
        Filters out the robot arm from the binary mask.
        
        Args:
            roi: The original RGB frame
            binary_roi: The binary mask of the line
            
        Returns:
            Filtered binary mask with the robot arm removed
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for the robot arm
        lower_arm = np.array([0, 0, 0])
        upper_arm = np.array([180, 255, 75])
        
        # Create a mask for the robot arm
        arm_mask = cv2.inRange(hsv, lower_arm, upper_arm)
        
        # Invert the arm mask
        arm_mask_inv = cv2.bitwise_not(arm_mask)
        
        # Apply the inverted arm mask to the binary mask
        filtered_binary_roi = cv2.bitwise_and(binary_roi, arm_mask_inv)
        
        return filtered_binary_roi

    def _find_line_in_roi(self, binary_roi: np.ndarray) -> Optional[Dict]:
        """
        Analyzes the binary mask of the ROI to find the line and detect intersections.
        This uses a simplified, robust logic for intersection detection.
        """
        contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Filter out very small contours (noise) - reduced threshold for more sensitivity
        min_contour_area = 25  # Reduced from 50 to 25 to catch smaller intersection features
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

        if not valid_contours:
            return None

        # Sort by area, largest first
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
        
        roi_h, roi_w = binary_roi.shape
        is_intersection = False

        # --- Case 1: Multiple contours detected ---
        # This could indicate an intersection, but let's be more selective
        if len(valid_contours) >= 2:
            line1_contour = valid_contours[0]
            line2_contour = valid_contours[1]
            
            # Check if both contours are significant in size (lowered threshold)
            area1 = cv2.contourArea(line1_contour)
            area2 = cv2.contourArea(line2_contour)
            min_intersection_area = roi_w * roi_h * 0.02  # Reduced from 0.05 to 0.02 - more sensitive
            
            # Only consider it an intersection if both contours are substantial
            if area1 > min_intersection_area and area2 > min_intersection_area:
                is_intersection = True
            
            # Combine the two largest contours for analysis
            combined_contour = np.vstack([line1_contour, line2_contour])
            M = cv2.moments(combined_contour)
            x, y, w, h = cv2.boundingRect(combined_contour)
            area = cv2.contourArea(combined_contour)

        # --- Case 2: One large contour detected ---
        else:
            largest_contour = valid_contours[0]
            M = cv2.moments(largest_contour)
            x, y, w, h = cv2.boundingRect(largest_contour)
            area = cv2.contourArea(largest_contour)
            
            # More sensitive intersection logic for detecting T-intersections at grid cells
            # Lowered thresholds to catch more intersections
            is_wide = w > roi_w * 0.5  # Reduced from 0.8 to 0.5 - more sensitive
            is_tall = h > roi_h * 0.5  # Reduced from 0.8 to 0.5 - more sensitive
            
            # For a T-intersection, we need significant width OR height (not both)
            # This catches horizontal T-bars, vertical T-bars, and cross intersections
            is_t_intersection = is_wide or is_tall or (w > roi_w * 0.6) or (h > roi_h * 0.6)
            
            if is_t_intersection:
                is_intersection = True
        
        if M["m00"] == 0:
            return None
        
        # Enhanced line center calculation for better self-correction
        if self.CENTERING_PRECISION_MODE:
            cx, cy = self._calculate_precise_line_center(binary_roi, M, valid_contours[0], is_intersection)
        else:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        # If it's an intersection, use the bottom-slice logic to find a stable navigation point
        if is_intersection:
            sub_roi_h = roi_h // 3
            sub_roi_y_start = roi_h - sub_roi_h
            bottom_slice = binary_roi[sub_roi_y_start:, :]
            M_bottom = cv2.moments(bottom_slice)
            if M_bottom["m00"] > 0:
                bottom_cx = int(M_bottom["m10"] / M_bottom["m00"])
                # Use weighted average of intersection center and bottom slice center
                cx = int(0.7 * bottom_cx + 0.3 * cx)  # Prioritize bottom slice for stability

        return {
            'center': (cx, cy),
            'contour': valid_contours[0], # Return the largest for drawing
            'contours': valid_contours,
            'bbox': (x, y, w, h),
            'area': area,
            'confidence': 1.0, # Simplified confidence
            'is_at_intersection': is_intersection
        }
    
    def _calculate_precise_line_center(self, binary_roi: np.ndarray, moments: Dict, 
                                     main_contour: np.ndarray, is_intersection: bool) -> Tuple[int, int]:
        """
        Calculate precise line center using multiple methods for better self-correction.
        Combines centroid, contour fitting, and edge analysis for maximum accuracy.
        """
        roi_h, roi_w = binary_roi.shape
        
        # Method 1: Standard centroid from moments
        centroid_cx = int(moments["m10"] / moments["m00"])
        centroid_cy = int(moments["m01"] / moments["m00"])
        
        # Method 2: Contour-based center using minimum enclosing rectangle
        rect = cv2.minAreaRect(main_contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=int)  # Fixed: replaced np.int0 with np.array(dtype=int)
        rect_cx = int(rect[0][0])  # Center x of the rotated rectangle
        rect_cy = int(rect[0][1])  # Center y of the rotated rectangle
        
        # Method 3: Horizontal scanning for line center (most reliable for line following)
        scan_centers = []
        scan_start_y = max(0, roi_h // 4)  # Start from 25% down
        scan_end_y = min(roi_h, roi_h * 3 // 4)  # End at 75% down
        
        for y in range(scan_start_y, scan_end_y, 2):  # Sample every 2 pixels
            row = binary_roi[y, :]
            white_pixels = np.where(row > 128)[0]  # Find white pixels in this row
            
            if len(white_pixels) > 5:  # Need minimum pixels for reliable center
                # Calculate weighted center of this row
                left_edge = white_pixels[0]
                right_edge = white_pixels[-1]
                row_center = (left_edge + right_edge) // 2
                scan_centers.append(row_center)
        
        # Calculate average scan center if we have enough data
        if len(scan_centers) >= 3:
            scan_cx = int(np.median(scan_centers))  # Use median for robustness
        else:
            scan_cx = centroid_cx  # Fallback to centroid
        
        # Method 4: Edge-based center calculation for very precise centering
        # Apply edge detection to find line boundaries
        edges = cv2.Canny(binary_roi, 50, 150)
        
        # Find leftmost and rightmost edges in the middle section of ROI
        middle_y_start = roi_h // 3
        middle_y_end = roi_h * 2 // 3
        middle_section = edges[middle_y_start:middle_y_end, :]
        
        edge_centers = []
        for y in range(0, middle_section.shape[0], 3):  # Sample every 3 pixels
            row_edges = np.where(middle_section[y, :] > 0)[0]
            if len(row_edges) >= 2:
                # Find leftmost and rightmost significant edges
                left_edge = row_edges[0]
                right_edge = row_edges[-1]
                if right_edge - left_edge > 10:  # Minimum line width
                    edge_center = (left_edge + right_edge) // 2
                    edge_centers.append(edge_center)
        
        if len(edge_centers) >= 2:
            edge_cx = int(np.mean(edge_centers))
        else:
            edge_cx = centroid_cx  # Fallback
        
        # Combine all methods with weighted average for final result
        if is_intersection:
            # For intersections, prioritize scan-based method (most stable)
            final_cx = int(0.5 * scan_cx + 0.3 * centroid_cx + 0.2 * edge_cx)
            final_cy = centroid_cy  # Use standard centroid for y
        else:
            # For normal line following, use more balanced combination
            final_cx = int(0.4 * scan_cx + 0.3 * edge_cx + 0.2 * centroid_cx + 0.1 * rect_cx)
            final_cy = int(0.8 * centroid_cy + 0.2 * rect_cy)
        
        # Clamp to valid coordinates
        final_cx = max(0, min(roi_w - 1, final_cx))
        final_cy = max(0, min(roi_h - 1, final_cy))
        
        if self.debug and abs(final_cx - centroid_cx) > 5:
            print(f"PRECISE CENTERING: Adjusted center from {centroid_cx} to {final_cx} "
                  f"(scan: {scan_cx}, edge: {edge_cx}, rect: {rect_cx})")
        
        return final_cx, final_cy
    
    def _calculate_line_following_params(self, line_info: Optional[Dict], 
                                       frame_width: int, roi_start_y: int) -> Dict:
        """Calculate line following parameters."""
        if line_info is None:
            self.line_lost_counter += 1
            return {
                'line_detected': False,
                'line_center_x': frame_width // 2,
                'line_offset': 0.0,
                'line_confidence': 0.0,
                'is_at_intersection': False,
                'status': 'line_lost'
            }
        
        # Line found - reset lost counter
        self.line_lost_counter = 0
        
        # Calculate line center in full frame coordinates
        line_center_x = line_info['center'][0]
        
        # Calculate offset from frame center (-1.0 to 1.0)
        frame_center = frame_width // 2
        raw_offset = (line_center_x - frame_center) / (frame_width // 2)
        raw_offset = max(-1.0, min(1.0, raw_offset))  # Clamp to valid range
        
        # Enhanced smoothing with self-correction bias and stuck detection
        self.center_offset_history.append(raw_offset)
        if len(self.center_offset_history) > self.history_size:
            self.center_offset_history.pop(0)
        
        # Stuck detection - track if we're making progress
        self.stuck_detection_history.append(abs(raw_offset))
        if len(self.stuck_detection_history) > self.stuck_detection_size:
            self.stuck_detection_history.pop(0)
        
        # Check if we're stuck (offset not decreasing over time)
        is_stuck = False
        current_time = time.time()
        reset_cooldown = 1.5  # Minimum 1.5 seconds between resets
        
        if len(self.stuck_detection_history) >= self.stuck_detection_size and (current_time - self.last_reset_time) > reset_cooldown:
            recent_avg = sum(self.stuck_detection_history[-3:]) / 3  # Use last 3 samples
            older_avg = sum(self.stuck_detection_history[:3]) / 3   # Compare to first 3 samples
            # If recent offset is not significantly smaller than older offset, we're stuck
            if recent_avg >= older_avg - self.stuck_threshold and recent_avg > 0.25:  # Lower threshold
                is_stuck = True
                self.last_reset_time = current_time  # Update reset time
                if self.debug:
                    print(f"STUCK DETECTED: Recent avg {recent_avg:.3f} vs older avg {older_avg:.3f} - resetting bias system")
        
        # Use adaptive weighted average that prioritizes correction for large offsets
        if len(self.center_offset_history) > 1:
            # For large offsets, give MORE weight to recent measurements for faster correction
            abs_raw_offset = abs(raw_offset)
            if abs_raw_offset > 0.4:  # Large offset - prioritize immediate correction
                weights = [0.05, 0.1, 0.2, 0.65]  # Heavy weight on most recent
                correction_boost = 1.3  # Boost the correction signal
            elif abs_raw_offset > 0.2:  # Medium offset - moderate correction bias
                weights = [0.1, 0.15, 0.25, 0.5]  # Moderate weight on recent
                correction_boost = 1.1
            else:  # Small offset - normal smoothing
                weights = [0.1, 0.2, 0.3, 0.4]  # Standard weights
                correction_boost = 1.0
            
            weights = weights[-len(self.center_offset_history):]  # Adjust for actual history length
            total_weight = sum(weights)
            smoothed_offset = sum(offset * weight for offset, weight in zip(self.center_offset_history, weights)) / total_weight
            
            # Apply correction boost for large offsets
            smoothed_offset *= correction_boost
            
            # Add additional self-correction bias - if consistently off-center, increase correction TOWARDS center
            # BUT: Reset bias system if we're stuck to prevent getting trapped in correction loops
            if is_stuck:
                # Clear history to reset the bias system
                self.center_offset_history = self.center_offset_history[-1:]  # Keep only latest measurement
                smoothed_offset = raw_offset  # Use raw offset without bias
                # Also reset PID to prevent windup
                self.pid.reset()
                if self.debug:
                    print("BIAS RESET: Cleared correction history and PID integral due to stuck detection")
            elif len(self.center_offset_history) >= 2:  # React faster with only 2 samples
                recent_offsets = self.center_offset_history[-2:]  # Use last 2 instead of 3 for faster response
                avg_recent = sum(recent_offsets) / len(recent_offsets)
                
                # IMMEDIATE BIAS CORRECTION: Add extra correction bias for ANY consistent drift
                if abs(avg_recent) > 0.05:  # Much lower threshold - catch ANY drift from center
                    bias_factor = min(0.25, abs(avg_recent) * 0.8)  # Stronger bias factor for immediate correction
                    # FIXED: Bias should be OPPOSITE to the offset to correct towards center
                    correction_bias = bias_factor * (-1 if avg_recent > 0 else 1)
                    smoothed_offset += correction_bias
                    
                    if self.debug and abs(correction_bias) > 0.03:
                        print(f"IMMEDIATE BIAS: Adding strong bias {correction_bias:.3f} to counter drift {avg_recent:.3f}")
        else:
            smoothed_offset = raw_offset
        
        # Clamp final offset to valid range
        smoothed_offset = max(-1.0, min(1.0, smoothed_offset))
        
        # Calculate turn angle (enhanced proportional control)
        turn_angle = smoothed_offset * 45
        
        return {
            'line_detected': True,
            'line_center_x': line_center_x,
            'line_offset': smoothed_offset,
            'raw_offset': raw_offset,  # Include raw offset for debugging
            'line_confidence': line_info['confidence'],
            'status': 'line_following',
            'is_at_intersection': line_info.get('is_at_intersection', False),
            'solidity': line_info.get('solidity', 1.0),
            'aspect_ratio': line_info.get('aspect_ratio', 0.0),
            'area': line_info.get('area', 0),
            'contour': line_info.get('contour'),
            'contours': line_info.get('contours')
        }
    
    def _draw_debug_overlay(self, frame: np.ndarray, result: Dict, 
                          roi_start_y: int, binary_roi: np.ndarray) -> np.ndarray:
        """
        Draw debug overlay on the frame.
        MODIFIED: Now shows the binary ROI view for better debugging of line detection.
        """
        height, width = frame.shape[:2]

        # Create a BGR version of the binary ROI to draw colored lines on
        binary_roi_bgr = cv2.cvtColor(binary_roi, cv2.COLOR_GRAY2BGR)
        
        # Create a full-size black frame
        debug_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Calculate ROI end position (excluding arm area)
        roi_end_y = int(height * self.ARM_EXCLUSION_RATIO)
        
        # Place the binary ROI onto the full-size frame in the correct position
        debug_frame[roi_start_y:roi_end_y, :] = binary_roi_bgr

        # Draw frame center line for reference (only in ROI area)
        center_x = width // 2
        cv2.line(debug_frame, (center_x, roi_start_y), (center_x, roi_end_y), (0, 0, 255), 1) # Red line
        
        # Draw arm exclusion zone boundary
        cv2.line(debug_frame, (0, roi_end_y), (width, roi_end_y), (0, 255, 255), 2) # Yellow line
        cv2.putText(debug_frame, "ARM EXCLUSION ZONE", (10, roi_end_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if result.get('line_detected') and result.get('contours'):
            # Draw all detected contours in blue
            cv2.drawContours(debug_frame, result['contours'], -1, (255, 0, 0), 2, offset=(0, roi_start_y))

            # Draw detected line center in green
            line_center_x = result.get('line_center_x')
            if line_center_x is not None:
                # For the y-coordinate, let's find an average y from the main contour's bounding box
                # to place the center dot correctly.
                contour = result.get('contour')
                center_y_in_roi = frame.shape[0] // 2 
                if contour is not None:
                    _, y_box, _, h_box = cv2.boundingRect(contour)
                    center_y_in_roi = y_box + h_box // 2

                cv2.circle(debug_frame, (line_center_x, center_y_in_roi + roi_start_y), 8, (0, 255, 0), -1)

        # Display detection info
        solidity = result.get('solidity', -1.0)
        aspect_ratio = result.get('aspect_ratio', -1.0)
        area = result.get('area', 0)
        confidence = result.get('line_confidence', 0.0)
        
        # Get buffer status for display
        buffer_status = result.get('buffer_status', {})
        using_prediction = result.get('using_prediction', False)
        
        # Get PID performance metrics for display
        pid_metrics = self.get_pid_performance_metrics()
        
        stats_text = [
            f"Offset: {result.get('line_offset', 0.0):.3f} | Raw: {result.get('raw_offset', 0.0):.3f}",
            f"S: {solidity:.2f} | AR: {aspect_ratio:.2f} | Area: {area}",
            f"Conf: {confidence:.2f} | Pred: {'YES' if using_prediction else 'NO'}",
            f"Status: {result.get('status', 'unknown')}",
            f"Precision Mode: {'ON' if self.CENTERING_PRECISION_MODE else 'OFF'}",
            f"Buffer: {buffer_status.get('buffer_size', 0)}/{buffer_status.get('max_buffer_size', 0)}",
            f"PID: kp={pid_metrics.get('kp', 0):.2f} ki={pid_metrics.get('ki', 0):.3f} kd={pid_metrics.get('kd', 0):.2f}",
            f"PID Out: {pid_metrics.get('output', 0):.1f} | Int: {pid_metrics.get('integral', 0):.1f}",
            f"Intersection: {result.get('is_at_intersection', False)}",
            f"Arm Filter: {'ON' if self.ARM_FILTERING_ENABLED else 'OFF'}"
        ]

        for i, text in enumerate(stats_text):
            cv2.putText(debug_frame, text, (10, height - 140 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return debug_frame
    
    def _empty_result(self) -> Dict:
        """Return empty result when no frame is provided."""
        return {
            'line_detected': False,
            'line_center_x': 0,
            'line_offset': 0.0,
            'line_confidence': 0.0,
            'corner_detected': False,
            'corner_direction': None,
            'corner_confidence': 0.0,
            'turn_angle': 0.0,
            'status': 'no_frame'
        }
    
    def _get_fallback_result(self) -> Dict:
        """Get fallback result when no camera frame is available."""
        # Try to get prediction from memory buffer if available
        predicted_result = self.line_memory_buffer.get_predicted_line_state()
        if predicted_result:
            predicted_result['status'] = 'predicted_no_camera'
            predicted_result['using_prediction'] = True
            return predicted_result
        else:
            # Return basic failure result
            return {
                'line_detected': False,
                'line_center_x': self.width // 2 if hasattr(self, 'width') else 160,
                'line_offset': 0.0,
                'line_confidence': 0.0,
                'is_at_intersection': False,
                'status': 'no_camera',
                'using_prediction': False,
                'buffer_status': self.line_memory_buffer.get_buffer_status()
            }
    
    def _get_default_robot_state(self) -> Dict:
        """Get default robot state when none is provided."""
        return {
            'position': (0, 0),
            'direction': 'N',
            'motor_speeds': {'left': 0, 'right': 0},
            'timestamp': time.time()
        }
    
    def update_corner_stopping(self, result: Dict):
        """
        Update corner stopping logic - MUST be called every iteration.
        This is separate from get_motor_speeds so the timer runs even when main controller overrides motors.
        """
        current_time = time.time()
        is_at_intersection = result.get('is_at_intersection', False)

        # Corner stopping logic - SIMPLIFIED AND FIXED
        if self.corner_stop_enabled:
            # FIRST: Handle active corner stopping
            if self.is_stopping_at_corner and self.corner_stop_start_time is not None:
                elapsed_time = current_time - self.corner_stop_start_time
                if elapsed_time >= self.corner_stop_duration:
                    # Time's up - stop the corner stopping
                    print(f"CORNER STOP: Finished stopping at intersection {self.corner_count} after {elapsed_time:.1f}s - signaling main controller for turn")
                    self.is_stopping_at_corner = False
                    self.corner_stop_start_time = None
                    # Reset intersection tracking to current state
                    self.last_intersection_state = is_at_intersection
                    # Signal main controller that it's time to make a turn decision
                    self.intersection_detected_for_main = True
                else:
                    # Still stopping
                    print(f"CORNER STOP: Still stopping at intersection {self.corner_count} - {elapsed_time:.1f}/{self.corner_stop_duration}s")
            
            # SECOND: Detect new intersections (only when not currently stopping)
            if not self.is_stopping_at_corner:
                # Check cooldown - don't detect new intersections too soon after the last one
                time_since_last_intersection = current_time - self.last_intersection_time
                
                # Look for intersection PASSING detection - stop after we pass the T-shape
                # We want to detect: was seeing intersection → now not seeing intersection
                intersection_just_passed = (self.last_intersection_state and not is_at_intersection)
                
                if intersection_just_passed and time_since_last_intersection > self.intersection_cooldown:
                    # We just passed an intersection - start stopping
                    self.is_stopping_at_corner = True
                    self.corner_stop_start_time = current_time
                    self.corner_count += 1
                    self.corners_passed.append(current_time)
                    self.last_intersection_time = current_time
                    self.intersection_detected_for_main = True  # Signal to main controller
                    print(f"INTERSECTION {self.corner_count} PASSED! Stopping for {self.corner_stop_duration} seconds...")
                elif is_at_intersection and time_since_last_intersection <= self.intersection_cooldown:
                    # Intersection detected but still in cooldown period - reduce spam
                    pass  # Removed debug print to reduce log noise
                elif is_at_intersection:
                    # Just for debugging - should not reach here
                    pass  # Removed debug print to reduce log noise
                
                # Update intersection state for next iteration
                self.last_intersection_state = is_at_intersection

    def set_path_to_destination(self, turn_sequence: List[str]):
        """
        Set the turn sequence to reach the destination.
        
        Args:
            turn_sequence: List of turn directions ['left', 'right', 'straight'] for each intersection
        
        Example:
            # To go right, right, left, right at intersections:
            camera_line_follower.set_path_to_destination(['right', 'right', 'left', 'right'])
        """
        self.turn_direction_pattern = turn_sequence
        self.current_turn_index = 0
        self.intersections_seen = 0
        if self.debug:
            print(f"Path updated: {turn_sequence}")
    
    def get_next_turn_direction(self) -> str:
        """Get the next turn direction from the pattern."""
        if not self.path_correction_enabled or not self.turn_direction_pattern:
            return 'right'  # Default turn
        
        if self.current_turn_index < len(self.turn_direction_pattern):
            direction = self.turn_direction_pattern[self.current_turn_index]
            return direction
        else:
            # Pattern complete, default to right
            return 'right'
    
    def update_path_dynamically(self, current_position: Tuple[int, int], 
                               destination: Tuple[int, int], 
                               current_direction: str = 'N'):
        """
        Dynamically update the path based on current position and destination.
        
        Args:
            current_position: (x, y) current grid position
            destination: (x, y) destination grid position
            current_direction: Current facing direction ('N', 'S', 'E', 'W')
        """
        # Simple pathfinding logic - can be made more sophisticated
        dx = destination[0] - current_position[0]
        dy = destination[1] - current_position[1]
        
        new_path = []
        
        # Determine turn sequence based on position difference
        # This is a simple example - you can make it more sophisticated
        if current_direction == 'N':  # Facing North
            if dx > 0:  # Need to go East
                new_path.append('right')
            elif dx < 0:  # Need to go West
                new_path.append('left')
            else:  # dx == 0
                if dy < 0:  # Need to go South (turn around)
                    new_path.append('right')  # or 'left', then another turn
                else:
                    new_path.append('straight')
        elif current_direction == 'E':  # Facing East
            if dy > 0:  # Need to go North
                new_path.append('left')
            elif dy < 0:  # Need to go South
                new_path.append('right')
            else:  # dy == 0
                if dx < 0:  # Need to go West (turn around)
                    new_path.append('right')
                else:
                    new_path.append('straight')
        # Add similar logic for 'S' and 'W' directions
        
        # Update the path
        self.set_path_to_destination(new_path)
        
        if self.debug:
            print(f"Path dynamically updated from {current_position} to {destination}")
            print(f"New path: {new_path}")
    
    def get_path_status(self) -> Dict:
        """Get current path navigation status."""
        return {
            'path_correction_enabled': self.path_correction_enabled,
            'turn_pattern': self.turn_direction_pattern,
            'current_turn_index': self.current_turn_index,
            'intersections_seen': self.intersections_seen,
            'next_turn': self.get_next_turn_direction() if self.current_turn_index < len(self.turn_direction_pattern) else None,
            'turns_remaining': len(self.turn_direction_pattern) - self.current_turn_index
        }
    
    def get_motor_speeds_lookahead(self, result: Dict, base_speed: int = 60) -> Tuple[int, int, int, int]:
        """
        Simplified motor control using look-ahead strategy.
        Returns motor speeds and action recommendation.
        """
        # PRIORITY 1: Box pickup positioning
        if result.get('box_approach_active', False):
            if result.get('box_in_position', False):
                # Box is at the very bottom of frame - STOP for pickup
                if self.debug:
                    print("BOX IN PICKUP POSITION - STOP")
                # Signal that we're ready for pickup
                result['ready_for_pickup'] = True
                return (0, 0, 0, 0)
            elif result.get('box_in_near', False):
                # Box is in near zone but not at bottom yet - move forward EXTREMELY aggressively
                box_info = result.get('box_info', {})
                box_y_ratio = box_info.get('bottom_y_ratio', 0.5)
                close_to_pickup = box_info.get('close_to_pickup', False)
                
                # Controlled aggressive approach - push firmly but not too hard
                if box_y_ratio < 0.6:
                    # Box is far, move at good speed
                    approach_speed = int(base_speed * 0.8)
                elif box_y_ratio < 0.8:
                    # Box is getting closer, moderate speed
                    approach_speed = int(base_speed * 0.6)
                elif close_to_pickup and box_y_ratio < 0.995:
                    # Box is close to pickup zone, push a bit more firmly to close the 2cm gap
                    approach_speed = int(base_speed * 0.55)
                    print(f"CLOSING GAP: Box at {box_y_ratio:.3f}, closing final gap at speed {approach_speed}")
                else:
                    # Box is very close, final push with just a bit more force to close that 2cm
                    approach_speed = int(base_speed * 0.45)
                    print(f"FINAL 2CM PUSH: Box at {box_y_ratio:.3f}, final 2cm push at speed {approach_speed}")
                
                approach_speed = max(30, approach_speed)  # Reasonable minimum speed
                
                if self.debug:
                    print(f"Box approach - moving forward at speed {approach_speed}, box at {box_y_ratio:.2f}")
                
                # Still try to stay centered on line if visible
                line_offset = result.get('line_offset', 0.0)
                turn_factor = line_offset * 0.3  # Gentle corrections
                
                left_speed = int(approach_speed * (1 + turn_factor))
                right_speed = int(approach_speed * (1 - turn_factor))
                
                return (left_speed, right_speed, left_speed, right_speed)
            else:
                # Box detected in far zone - continue normal speed but prepare
                if self.debug:
                    print("Box detected ahead - approaching at normal speed")
                # Continue with normal line following below
        
        # PRIORITY 2: Intersection handling
        if result.get('intersection_now', False) and not result.get('box_approach_active', False):
            # Get the next turn direction from the pattern
            turn_direction = self.get_next_turn_direction()
            turn_speed = base_speed // 2
            
            if self.debug:
                print(f"EXECUTING INTERSECTION TURN - {turn_direction.upper()}")
            
            # Update turn index for next intersection
            self.current_turn_index += 1
            self.intersections_seen += 1
            
            # Execute turn based on direction
            if turn_direction == 'right':
                return (turn_speed, -turn_speed, turn_speed, -turn_speed)
            elif turn_direction == 'left':
                return (-turn_speed, turn_speed, -turn_speed, turn_speed)
            elif turn_direction == 'straight':
                # Go straight through intersection
                return (base_speed, base_speed, base_speed, base_speed)
            else:
                # Default to right turn if unknown direction
                return (turn_speed, -turn_speed, turn_speed, -turn_speed)
        
        # PRIORITY 3: Normal line following
        line_offset = result.get('line_offset', 0.0)
        line_detected = result.get('line_detected', False)
        
        if not line_detected:
            # Use memory buffer to predict where line might be
            if hasattr(self, 'line_memory_buffer'):
                # Try to get predicted line position from memory
                predicted_result = self.line_memory_buffer.get_predicted_line_state(320)  # Assuming 320 width
                if predicted_result and predicted_result.get('line_detected', False):
                    # Use predicted line position for recovery
                    predicted_offset = predicted_result.get('line_offset', 0.0)
                    if self.debug:
                        print(f"Using memory buffer prediction: offset={predicted_offset:.3f}")
                    
                    # Turn towards predicted line position
                    turn_intensity = min(50, abs(predicted_offset) * 100)
                    turn_direction = 1 if predicted_offset > 0 else -1
                    turn_speed = int(turn_intensity * turn_direction)
                    
                    recovery_speed = base_speed // 3
                    left_speed = int(recovery_speed + turn_speed)
                    right_speed = int(recovery_speed - turn_speed)
                    
                    self.line_lost_counter += 1
                    return (left_speed, right_speed, left_speed, right_speed)
            
            # Fallback to simple alternating search if no memory available
            if self.line_lost_counter < 10:
                search_dir = 1 if (self.line_lost_counter // 3) % 2 == 0 else -1
                turn_speed = base_speed // 3
                self.line_lost_counter += 1
                return (turn_speed * search_dir, -turn_speed * search_dir, 
                       turn_speed * search_dir, -turn_speed * search_dir)
            else:
                # Stop if line lost for too long
                return (0, 0, 0, 0)
        
        # Reset line lost counter
        self.line_lost_counter = 0
        
        # Simple proportional control for line following
        turn_factor = line_offset * 0.5  # Simple P control
        
        # Calculate differential speeds
        left_speed = int(base_speed * (1 + turn_factor))
        right_speed = int(base_speed * (1 - turn_factor))
        
        # Clamp speeds
        left_speed = max(-100, min(100, left_speed))
        right_speed = max(-100, min(100, right_speed))
        
        # Slow down if intersection is approaching (but not during box approach)
        if result.get('intersection_ahead', False) and not result.get('box_approach_active', False):
            countdown = result.get('intersection_countdown', 0)
            if countdown < 5:  # Start slowing down 5 frames before intersection
                slow_factor = countdown / 5.0  # Linear slowdown
                left_speed = int(left_speed * (0.5 + 0.5 * slow_factor))
                right_speed = int(right_speed * (0.5 + 0.5 * slow_factor))
                if self.debug:
                    print(f"Slowing for intersection: {slow_factor:.2f}")
        
        return (left_speed, right_speed, left_speed, right_speed)
    
    def get_motor_speeds(self, result: Dict, base_speed: int = 60) -> Tuple[int, int, int, int]:
        """
        Enhanced motor speed calculation with aggressive line centering.
        Prioritizes getting back to line center when robot drifts off.
        """
        # Update corner stopping logic first
        self.update_corner_stopping(result)
        
        # If we're stopping at a corner, return zero speeds
        if self.is_stopping_at_corner:
            return 0, 0, 0, 0

        line_offset = result.get('line_offset', 0.0)
        raw_offset = result.get('raw_offset', line_offset)
        line_confidence = result.get('line_confidence', 0.0)

        if result.get('status') == 'line_lost' or line_confidence < 0.2:
            # Enhanced line recovery mechanism with intelligent search patterns
            if self.line_lost_counter < 2:  # Reduced from 3 to 2
                # Immediate response: try to continue in last known direction with increased intensity
                if hasattr(self, 'last_turn_correction') and abs(self.last_turn_correction) > 5:
                    recovery_turn = self.last_turn_correction * 0.8  # More aggressive continuation
                else:
                    recovery_turn = 0
            elif self.line_lost_counter < 5:  # Reduced from 8 to 5
                # Smart search: look in direction of last known offset
                if hasattr(self, 'last_known_offset') and abs(self.last_known_offset) > 0.1:
                    # Search in direction where line was last seen
                    search_direction = 1 if self.last_known_offset > 0 else -1
                    search_intensity = min(35, 15 + self.line_lost_counter * 3)
                    recovery_turn = search_intensity * search_direction
                else:
                    # Standard alternating search but more intensive
                    search_intensity = min(35, self.line_lost_counter * 4)
                    recovery_turn = search_intensity * (1 if (self.line_lost_counter // 2) % 2 == 0 else -1)
            elif self.line_lost_counter < 10:  # Reduced from 20 to 10
                # Wider search pattern
                search_intensity = min(50, 20 + self.line_lost_counter * 2)
                recovery_turn = search_intensity * (1 if (self.line_lost_counter // 3) % 2 == 0 else -1)
            else:
                # If still lost after 10 frames, stop and try memory buffer prediction
                return 0, 0, 0, 0
            
            # Apply recovery movement with dynamic speed
            recovery_speed = max(25, base_speed // 2)  # Higher minimum speed for better recovery
            left_speed = int(recovery_speed + recovery_turn)
            right_speed = int(recovery_speed - recovery_turn)
            
            # Increment lost counter
            self.line_lost_counter += 1
            
            return left_speed, right_speed, left_speed, right_speed
        
        # Reset line lost counter since we found the line
        self.line_lost_counter = 0
        
        # Store current offset for future recovery reference
        self.last_known_offset = line_offset
        
        # Prepare robot state for adaptive PID
        robot_state = {
            'motor_speeds': {'left': base_speed, 'right': base_speed},  # Approximate current speeds
            'timestamp': time.time()
        }
        
        # Get additional parameters for adaptive PID
        using_prediction = result.get('using_prediction', False)
        
        # Calculate the turning correction using the enhanced adaptive PID controller
        turn_correction = self.pid.update(
            error=line_offset,
            robot_state=robot_state,
            line_confidence=line_confidence,
            using_prediction=using_prediction
        )
        
        # Store last turn correction for recovery mechanism
        self.last_turn_correction = turn_correction
        
        # Improved speed control that maintains enough speed for effective correction
        is_severe_recovery = result.get('severe_offset_recovery', False)
        abs_offset = abs(line_offset)
        
        if is_severe_recovery:
            # Severe offset recovery: moderate speed reduction but keep enough speed for correction
            speed_reduction_factor = 0.7  # Reduce to 70% speed but maintain correction ability
        elif abs_offset > 0.4:  # Large offset - need speed to execute correction
            speed_reduction_factor = 0.85  # Only slight reduction to maintain correction power
        elif abs_offset > 0.3:  # Medium offset - moderate reduction
            speed_reduction_factor = 0.95  # Reduce to 95% speed  
        elif using_prediction:
            speed_reduction_factor = 0.95  # Slightly slower when using predictions
        else:
            speed_reduction_factor = 1.0  # Full speed for well-centered line following
        
        target_speed = int(base_speed * max(0.6, speed_reduction_factor))  # Higher minimum speed for effective correction
        
        # Speed smoothing - gradually adjust to target speed for consistency
        self.speed_history.append(target_speed)
        if len(self.speed_history) > 1:
            # Use weighted average of recent speeds
            weights = [0.3, 0.4, 0.3]  # Give most weight to current, some to recent
            weights = weights[-len(self.speed_history):]
            total_weight = sum(weights)
            current_speed = int(sum(speed * weight for speed, weight in zip(self.speed_history, weights)) / total_weight)
        else:
            current_speed = target_speed
        
        # Limit speed changes to prevent sudden jumps
        max_speed_change = 10  # Maximum speed change per frame
        if hasattr(self, 'last_calculated_speed'):
            speed_diff = current_speed - self.last_calculated_speed
            if abs(speed_diff) > max_speed_change:
                current_speed = self.last_calculated_speed + (max_speed_change if speed_diff > 0 else -max_speed_change)
        
        self.last_calculated_speed = current_speed
        
        # Enhanced turn correction limiting that allows stronger corrections for large offsets
        confidence_factor = max(0.4, line_confidence)  # Scale from 0.4 to 1.0
        
        # IMMEDIATE RESPONSE LIMITS - allow strong corrections for any drift from center
        if abs_offset < 0.03:  # Truly centered - gentle limits
            max_turn_correction = current_speed * 0.25  # 25% of speed for true center
        elif abs_offset < 0.08:  # ANY drift from center - strong correction power
            max_turn_correction = current_speed * 0.55  # 55% of speed for immediate response
        elif abs_offset < 0.15:  # Small drift - very strong correction power
            max_turn_correction = current_speed * 0.7   # 70% of speed for quick recovery
        elif abs_offset < 0.25:  # Medium drift - maximum correction power
            max_turn_correction = current_speed * 0.85  # 85% of speed to get back to center
        elif abs_offset > 0.35:  # Large offset - ultra-maximum correction power
            max_turn_correction = current_speed * 0.95  # 95% of speed for emergency correction
        else:  # Normal drift - strong correction
            max_turn_correction = current_speed * (0.6 + 0.2 * confidence_factor)  # 60% to 80% of speed
        
        # The adaptive PID already limits its output, but we add a secondary limit for safety
        turn_correction = max(-max_turn_correction, min(max_turn_correction, turn_correction))
        
        # Calculate motor speeds with smoother differential - FIXED DIRECTION
        left_speed = int(current_speed + turn_correction)
        right_speed = int(current_speed - turn_correction)
        
        # Ensure motor speeds stay within reasonable bounds
        left_speed = max(-100, min(100, left_speed))
        right_speed = max(-100, min(100, right_speed))
        
        # Debug output for large offsets to troubleshoot stopping issues
        if self.debug and abs_offset > 0.3:
            print(f"MOTOR SPEEDS: Base={base_speed} Target={target_speed} Current={current_speed} | Turn={turn_correction:.1f} | Motors: L={left_speed} R={right_speed}")

        return left_speed, right_speed, left_speed, right_speed
    
    def get_corner_status(self) -> Dict:
        """Get current corner stopping status for debugging/monitoring."""
        current_stop_time = 0
        if self.is_stopping_at_corner and self.corner_stop_start_time:
            current_stop_time = time.time() - self.corner_stop_start_time
            
        return {
            'corner_stop_enabled': self.corner_stop_enabled,
            'is_stopping_at_corner': self.is_stopping_at_corner,
            'corner_count': self.corner_count,
            'corners_passed': len(self.corners_passed),
            'current_stop_time': current_stop_time,
            'stop_duration': self.corner_stop_duration,
            'intersection_detected_for_main': self.intersection_detected_for_main
        }
    
    def clear_intersection_signal(self):
        """Clear the intersection detection signal after main controller handles it."""
        self.intersection_detected_for_main = False
    
    def reset_box_approach(self):
        """Reset box approach state after successful pickup."""
        self.box_approach_active = False
        self.box_in_position = False
        self.box_countdown = 0
        if self.debug:
            print("Box approach state reset")
    
    def clear_line_memory_buffer(self):
        """Clear the line memory buffer - useful for mission restart."""
        self.line_memory_buffer.clear_buffer()
        
    def reset_pid_controller(self):
        """Reset the adaptive PID controller - useful for mission restart."""
        self.pid.reset()
        
    def get_line_memory_status(self) -> Dict:
        """Get current line memory buffer status for monitoring."""
        return self.line_memory_buffer.get_buffer_status()
        
    def get_line_trend_analysis(self) -> Dict:
        """Get line movement trend analysis from the buffer."""
        return self.line_memory_buffer.get_line_trend()
    
    def get_pid_status(self) -> Dict:
        """Get current adaptive PID controller status."""
        return self.pid.get_status()
    
    def get_pid_performance_metrics(self) -> Dict:
        """Get detailed PID performance metrics for debugging."""
        return self.pid.get_performance_metrics()


class CameraLineFollowingMixin:
    """
    A mixin class to add camera-based line following capabilities to a robot controller.
    """
    def init_camera_line_following(self, camera_index=1, width=320, height=240, fps=30, debug=True):
        """
        Initializes the camera line follower system.
        Note: This creates the object, but the camera must be initialized separately
        by calling self.camera_line_follower.initialize_camera() in the main run loop.
        """
        print("Initializing Camera Line Following Mixin...")
        self.camera_line_follower = CameraLineFollower(
            camera_index=camera_index,
            width=width,
            height=height,
            fps=fps,
            debug=debug
        )
        self.line_pid = PIDController(kp=0.4, ki=0.01, kd=0.1, output_limits=(-LINE_FOLLOW_SPEED, LINE_FOLLOW_SPEED))
        self.camera_line_result = {}

    def follow_line_with_camera(self, frame, base_speed=LINE_FOLLOW_SPEED):
        """
        Detects the line in the given frame and sends motor commands.
        
        Args:
            frame: The camera frame to process.
            base_speed: The base speed for the robot.
        """
        if frame is None or not hasattr(self, 'camera_line_follower'):
            return

        # Detect the line and get parameters
        self.camera_line_result = self.camera_line_follower.detect_line(frame)
        
        # Get motor speeds based on detection result
        fl, fr, bl, br = self.camera_line_follower.get_motor_speeds(
            self.camera_line_result,
            base_speed=base_speed
        )
        
        # Send motor speeds
        self.motor_controller.send_motor_speeds(fl, fr, bl, br)
    
    def follow_line_with_lookahead(self, frame, base_speed=LINE_FOLLOW_SPEED):
        """
        Uses the simplified look-ahead strategy for line following.
        
        Args:
            frame: The camera frame to process.
            base_speed: The base speed for the robot.
            
        Returns:
            dict: Result containing line detection info and intersection status
        """
        if frame is None or not hasattr(self, 'camera_line_follower'):
            return {'status': 'no_camera'}

        # Use the new look-ahead detection
        self.camera_line_result = self.camera_line_follower.detect_line_with_lookahead(frame)
        
        # Get motor speeds using look-ahead strategy
        fl, fr, bl, br = self.camera_line_follower.get_motor_speeds_lookahead(
            self.camera_line_result,
            base_speed=base_speed
        )
        
        # Send motor speeds
        self.motor_controller.send_motor_speeds(fl, fr, bl, br)
        
        return self.camera_line_result

    def get_camera_line_status(self) -> Dict:
        """Return the latest line detection result."""
        if not hasattr(self, 'camera_line_result'):
            return {
                'line_detected': False,
                'line_offset': 0.0,
                'confidence': 0.0,
                'status': 'not_initialized'
            }
        
        return {
            'line_detected': self.camera_line_result.get('line_detected', False),
            'line_offset': self.camera_line_result.get('line_offset', 0.0),
            'confidence': self.camera_line_result.get('line_confidence', 0.0),
            'corner_detected': self.camera_line_result.get('corner_detected', False),
            'corner_direction': self.camera_line_result.get('corner_direction', None),
            'turn_angle': self.camera_line_result.get('turn_angle', 0.0),
            'status': self.camera_line_result.get('status', 'unknown')
        } 