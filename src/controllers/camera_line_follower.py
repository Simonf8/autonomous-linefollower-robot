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
        self.corner_angle_threshold = 30  # Degrees to detect corner
        self.corner_line_length = 50  # Minimum line length for corner detection
        self.corner_cooldown = 3.0  # Seconds between corner detections
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
    
    def __init__(self, camera_index=1, width=320, height=240, fps=30, debug=False):
        self.debug = debug
        
        # Line detection parameters
        self.BLACK_THRESHOLD = 80  # Threshold for detecting black lines
        self.BLUR_SIZE = (5, 5)
        self.MIN_CONTOUR_AREA = 500
        self.MIN_LINE_WIDTH = 10
        self.MAX_LINE_WIDTH = 200
        
        # Morphological operations kernel
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Region of Interest (ROI) settings - ignore bottom 25% where arm is
        self.ROI_HEIGHT_RATIO = 0.5  # Use middle 50% of frame for line detection
        self.ROI_START_RATIO = 0.25   # Start ROI at 25% down from top (ignore top 25%)
        self.ARM_EXCLUSION_RATIO = 0.75  # Bottom 25% is arm area - completely ignored
        
        # ARM FILTERING CONFIGURATION
        # Enable/disable purple arm filtering
        self.ARM_FILTERING_ENABLED = True
        
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
        self.ARM_MIN_AREA = 50
        
        # Maximum area ratio for arm (to avoid filtering large purple obstacles)
        self.ARM_MAX_AREA_RATIO = 0.5  # Max 50% of ROI area
        
        # Arm position constraints (where we expect the arm to appear) - expanded
        self.ARM_EXPECTED_REGION = {
            'bottom_ratio': 0.5,    # Arm typically appears in bottom 50% of ROI
            'center_tolerance': 0.6  # Arm typically appears within 60% of center
        }
        
        # Line following parameters
        self.center_offset_history = []
        self.history_size = 5  # Smooth over last 5 measurements
        
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
        
        # Performance tracking
        self.last_detection_time = 0
        self.detection_fps = 0
        
        # PID controller for smooth line following
        self.pid = PIDController(kp=1.2, ki=0.1, kd=0.8, output_limits=(-100, 100))
        self.strafe_gain = 30.0 # Proportional gain for sideways correction
        
        # Detection history for smoothing
        self.line_offset_history = deque(maxlen=5)
        
        # Result cache to avoid re-computation
        self.last_frame_hash = None
        self.last_result = None
        
        self.max_line_width = 0.8 # Max line width as a ratio of ROI width
        self.intersection_solidity_threshold = 0.85 # Lower value means more complex shape
        self.intersection_aspect_ratio_threshold = 0.8 # If width is > 80% of height, might be a corner
        
        self.canny_high = 150
        self.binary_threshold = 70
        
        # Line detection parameters
        self.min_line_area = 30  # Minimum contour area to be considered a line (reduced for better sensitivity)
        
        # Camera properties
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.camera_initialized = False
        self.frames_processed = 0

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

    def detect_line(self, frame: Optional[np.ndarray] = None) -> Dict:
        """
        Analyzes a camera frame to find the line, its offset, and intersections.
        If a frame is provided, it's used. Otherwise, captures a new frame.
        """
        if frame is None:
            frame = self.get_camera_frame()
        
        if frame is None:
            # If still no frame, return an empty result
            return self._empty_result()
            
        height, width, _ = frame.shape
        self.frames_processed += 1
        
        # Define ROI for line detection (exclude bottom 25% where arm is)
        roi_start_y = int(height * self.ROI_START_RATIO)
        roi_end_y = int(height * self.ARM_EXCLUSION_RATIO)  # Stop before arm area
        roi = frame[roi_start_y:roi_end_y, :]
        
        # Preprocess the ROI
        processed_roi = self._preprocess_roi(roi)
        
        # Detect line in the ROI
        line_info = self._find_line_in_roi(processed_roi)
        
        # Calculate line following parameters and merge results
        result = self._calculate_line_following_params(line_info, width, roi_start_y)
        if line_info:
            result.update(line_info) # Add solidity, aspect_ratio etc. to the main result
        
        # Add debug visualization if enabled
        if self.debug:
            result['processed_frame'] = self._draw_debug_overlay(frame, result, roi_start_y, processed_roi)
        
        # Update FPS calculation based on actual processing time
        current_time = time.time()
        if self.last_detection_time > 0:
            time_diff = current_time - self.last_detection_time
            if time_diff > 0:
                self.detection_fps = 1.0 / time_diff
        self.last_detection_time = current_time
        
        return result
    
    def _preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        Applies a series of filters to the ROI to isolate the line using color masking.
        Simplified approach since arm area is completely excluded from ROI.
        """
        # Convert to HSV color space for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define range for black color in HSV - balanced to detect black lines but avoid brown surfaces
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 80, 100])  # Increased thresholds to better detect black lines
        
        # Create a mask that isolates the black parts of the image
        black_mask = cv2.inRange(hsv, lower_black, upper_black)
        
        # Additional approach: Use both adaptive and simple thresholding on grayscale
        # This helps with reflections and uneven lighting
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold to find dark regions - balanced sensitivity
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5
        )
        
        # Also apply a simple threshold to exclude brown/wooden surfaces
        _, simple_thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        
        # Combine all three methods: HSV + adaptive + simple threshold
        # Only keep pixels that are dark in ALL methods
        temp_mask = cv2.bitwise_and(black_mask, adaptive_thresh)
        combined_mask = cv2.bitwise_and(temp_mask, simple_thresh)
        
        # Morphological operations to connect broken line segments
        # 1. Remove small noise
        small_kernel = np.ones((2, 2), np.uint8)
        opened_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, small_kernel)
        
        # 2. Connect line segments with directional kernels - much larger kernels to fill gaps
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        horizontal_closed = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, horizontal_kernel)
        
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 25))
        vertical_closed = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, vertical_kernel)
        
        # 3. Combine both directional closings
        directional_closed = cv2.bitwise_or(horizontal_closed, vertical_closed)
        
        # 4. Very aggressive closing to fill large gaps in the middle
        large_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
        gap_filled = cv2.morphologyEx(directional_closed, cv2.MORPH_CLOSE, large_kernel)
        
        # 5. Additional horizontal closing specifically for line gaps
        line_connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 10))
        gap_filled = cv2.morphologyEx(gap_filled, cv2.MORPH_CLOSE, line_connect_kernel)
        
        # 5. Final dilation to ensure line segments are thick enough
        dilate_kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.dilate(gap_filled, dilate_kernel, iterations=2)
        
        return final_mask
    
    def _filter_purple_arm(self, hsv_roi: np.ndarray, black_mask: np.ndarray, roi_shape: tuple) -> np.ndarray:
        """
        Filters out purple arm regions from the black mask to prevent interference.
        
        Args:
            hsv_roi: HSV version of the ROI
            black_mask: Current black line mask
            roi_shape: Shape of the ROI (height, width, channels)
            
        Returns:
            Filtered black mask with arm regions removed
        """
        roi_height, roi_width = roi_shape[:2]
        
        # Create combined purple mask from all defined ranges
        purple_mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
        
        for lower_purple, upper_purple in self.ARM_PURPLE_RANGES:
            range_mask = cv2.inRange(hsv_roi, lower_purple, upper_purple)
            purple_mask = cv2.bitwise_or(purple_mask, range_mask)
        
        # Find purple contours (potential arm regions)
        purple_contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter purple contours to identify likely arm regions
        arm_regions = []
        for contour in purple_contours:
            area = cv2.contourArea(contour)
            
            # Skip if too small or too large
            if area < self.ARM_MIN_AREA:
                continue
            if area > (roi_width * roi_height * self.ARM_MAX_AREA_RATIO):
                continue
            
            # Get contour properties
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Check if in expected arm region (bottom portion, near center)
            expected_bottom_y = roi_height * (1.0 - self.ARM_EXPECTED_REGION['bottom_ratio'])
            center_tolerance = roi_width * self.ARM_EXPECTED_REGION['center_tolerance']
            roi_center_x = roi_width // 2
            
            is_in_bottom_region = center_y >= expected_bottom_y
            is_near_center = abs(center_x - roi_center_x) <= center_tolerance
            
            # Consider it an arm if it meets position criteria
            if is_in_bottom_region and is_near_center:
                arm_regions.append({
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'center': (center_x, center_y)
                })
        
        # Remove arm regions from the black mask
        filtered_mask = black_mask.copy()
        
        # Additional aggressive filtering: remove any purple pixels from the bottom region
        bottom_region_start = int(roi_height * 0.7)  # Bottom 30% of ROI
        bottom_purple_mask = purple_mask[bottom_region_start:, :]
        
        if np.sum(bottom_purple_mask) > 0:  # If there are purple pixels in bottom region
            # Create a mask to remove all purple from bottom region
            bottom_filter_mask = np.ones((roi_height, roi_width), dtype=np.uint8) * 255
            bottom_filter_mask[bottom_region_start:, :] = cv2.bitwise_not(purple_mask[bottom_region_start:, :])
            filtered_mask = cv2.bitwise_and(filtered_mask, bottom_filter_mask)
            
            if self.debug:
                purple_pixel_count = np.sum(bottom_purple_mask > 0)
                print(f"Filtered {purple_pixel_count} purple pixels from bottom region")
        
        # Additional filtering: Remove black pixels that are too close to purple regions (arm edges)
        # Dilate the purple mask to create a "no-black-zone" around purple areas
        purple_exclusion_kernel = np.ones((15, 15), np.uint8)
        purple_exclusion_mask = cv2.dilate(purple_mask, purple_exclusion_kernel, iterations=1)
        
        # Remove any black pixels that are within the purple exclusion zone
        filtered_mask = cv2.bitwise_and(filtered_mask, cv2.bitwise_not(purple_exclusion_mask))
        
        if self.debug and np.sum(purple_exclusion_mask) > 0:
            excluded_pixels = np.sum(purple_exclusion_mask > 0)
            print(f"Excluded {excluded_pixels} pixels near purple regions (arm edges)")
        
        # Process individual arm regions
        for arm_region in arm_regions:
            # Create a mask for this arm region with some padding
            arm_mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
            cv2.drawContours(arm_mask, [arm_region['contour']], -1, 255, -1)
            
            # Dilate the arm mask more aggressively to ensure complete removal
            dilate_kernel = np.ones((7, 7), np.uint8)
            arm_mask_dilated = cv2.dilate(arm_mask, dilate_kernel, iterations=2)
            
            # Remove this arm region from the black mask
            filtered_mask = cv2.bitwise_and(filtered_mask, cv2.bitwise_not(arm_mask_dilated))
            
            if self.debug:
                print(f"Filtered arm region: area={arm_region['area']}, center={arm_region['center']}")
        
        return filtered_mask
    
    def set_arm_filtering(self, enabled: bool):
        """Enable or disable purple arm filtering."""
        self.ARM_FILTERING_ENABLED = enabled
        if self.debug:
            status = "enabled" if enabled else "disabled"
            print(f"Purple arm filtering {status}")
    
    def configure_arm_colors(self, purple_ranges: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Configure the purple color ranges for arm detection.
        
        Args:
            purple_ranges: List of (lower_hsv, upper_hsv) tuples defining purple color ranges
        """
        self.ARM_PURPLE_RANGES = purple_ranges
        if self.debug:
            print(f"Updated arm color ranges: {len(purple_ranges)} ranges configured")
    
    def get_arm_filtering_status(self) -> Dict:
        """Get current arm filtering configuration and status."""
        return {
            'enabled': self.ARM_FILTERING_ENABLED,
            'purple_ranges': len(self.ARM_PURPLE_RANGES),
            'min_area': self.ARM_MIN_AREA,
            'max_area_ratio': self.ARM_MAX_AREA_RATIO,
            'expected_region': self.ARM_EXPECTED_REGION
        }
    
    def _find_line_in_roi(self, binary_roi: np.ndarray) -> Optional[Dict]:
        """
        Finds the largest contour in the binary ROI, treating it as the line.
        """
        contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            if self.debug:
                print("DEBUG: No contours found in binary ROI")
            return None

        # --- Find the largest contour in the ROI ---
        # The central filter was removed because it was preventing the detection
        # of wide T-junctions, which are essential for cornering. By considering
        # all contours, we can correctly identify the full shape of an intersection.
        largest_contour = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest_contour)

        if self.debug:
            print(f"DEBUG: Found {len(contours)} contours, largest area: {largest_area}, min_threshold: {self.min_line_area}")

        # Lower the minimum area threshold to be more sensitive to small line segments
        if largest_area < self.min_line_area:
            if self.debug:
                print(f"DEBUG: Largest contour area ({largest_area}) below threshold ({self.min_line_area})")
            
            # Fallback: Try to combine multiple smaller contours that might be line segments
            valid_contours = [c for c in contours if cv2.contourArea(c) >= 10]  # Very small minimum
            if len(valid_contours) >= 2:
                if self.debug:
                    print(f"DEBUG: Trying fallback with {len(valid_contours)} smaller contours")
                
                # Find the centroid of all valid contours combined
                total_area = sum(cv2.contourArea(c) for c in valid_contours)
                if total_area >= self.min_line_area:
                    # Calculate weighted centroid
                    total_mx = total_my = 0
                    for contour in valid_contours:
                        M = cv2.moments(contour)
                        if M["m00"] > 0:
                            total_mx += M["m10"]
                            total_my += M["m01"]
                    
                    if total_mx > 0 and total_my > 0:
                        # Use the largest contour but with combined centroid
                        cx = int(total_mx / total_area)
                        cy = int(total_my / total_area)
                        
                        # Update the largest contour center
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        roi_height, roi_width = binary_roi.shape
                        aspect_ratio = w / float(h) if h > 0 else 0
                        area_ratio = total_area / (roi_width * roi_height)
                        confidence = min(1.0, area_ratio * 10 + min(aspect_ratio / 3, 1.0) * 0.5)
                        
                        solidity = float(total_area) / cv2.contourArea(cv2.convexHull(largest_contour)) if cv2.contourArea(cv2.convexHull(largest_contour)) > 0 else 1.0
                        is_complex_shape = solidity < self.intersection_solidity_threshold
                        is_wide_shape = aspect_ratio > self.intersection_aspect_ratio_threshold
                        
                        if self.debug:
                            print(f"DEBUG: Fallback successful - combined area: {total_area}, center: ({cx}, {cy})")
                        
                        return {
                            'center': (cx, cy),
                            'contour': largest_contour,
                            'bbox': (x, y, w, h),
                            'area': total_area,
                            'confidence': confidence,
                            'is_at_intersection': is_complex_shape or is_wide_shape,
                            'solidity': solidity,
                            'aspect_ratio': aspect_ratio
                        }
            
            return None

        # --- Proceed with the largest contour ---
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            if self.debug:
                print("DEBUG: Contour has zero moments")
            return None
        
        # Line center
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate confidence based on contour properties
        roi_height, roi_width = binary_roi.shape
        area_ratio = cv2.contourArea(largest_contour) / (roi_width * roi_height)
        aspect_ratio = w / float(h) if h > 0 else 0
        
        # Higher confidence for larger, more line-like contours
        confidence = min(1.0, area_ratio * 10 + min(aspect_ratio / 3, 1.0) * 0.5)
        
        # --- Intersection Detection ---
        # A corner or T-junction is more "square" and less "solid" than a straight line.
        solidity = float(cv2.contourArea(largest_contour)) / cv2.contourArea(cv2.convexHull(largest_contour)) if cv2.contourArea(cv2.convexHull(largest_contour)) > 0 else 1.0
        
        # Check if the shape is complex (not solid) or wide (like a T-junction)
        is_complex_shape = solidity < self.intersection_solidity_threshold
        is_wide_shape = aspect_ratio > self.intersection_aspect_ratio_threshold
        
        result = {
            'center': (cx, cy),
            'contour': largest_contour,
            'bbox': (x, y, w, h),
            'area': cv2.contourArea(largest_contour),
            'confidence': confidence,
            'is_at_intersection': is_complex_shape or is_wide_shape,
            'solidity': solidity,
            'aspect_ratio': aspect_ratio
        }
        
        return result
    
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
        
        # Smooth the offset using history
        self.center_offset_history.append(raw_offset)
        if len(self.center_offset_history) > self.history_size:
            self.center_offset_history.pop(0)
        
        smoothed_offset = sum(self.center_offset_history) / len(self.center_offset_history)
        
        # Calculate turn angle (simple proportional control)
        turn_angle = smoothed_offset * 45  # Max 45 degrees turn
        
        return {
            'line_detected': True,
            'line_center_x': line_center_x,
            'line_offset': smoothed_offset,
            'line_confidence': line_info['confidence'],
            'status': 'line_following'
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

        if result.get('line_detected') and result.get('contour') is not None:
            # Draw the detected contour in blue
            contour = result['contour']
            cv2.drawContours(debug_frame, [contour], -1, (255, 0, 0), 2, offset=(0, roi_start_y))

            # Draw detected line center in green
            if result.get('center'):
                line_x, line_y = result['center']
                cv2.circle(debug_frame, (line_x, line_y + roi_start_y), 8, (0, 255, 0), -1)

        # Display detection info
        solidity = result.get('solidity', -1.0)
        aspect_ratio = result.get('aspect_ratio', -1.0)
        area = result.get('area', 0)
        confidence = result.get('line_confidence', 0.0)
        stats_text = [
            f"Offset: {result.get('line_offset', 0.0):.3f}",
            f"S: {solidity:.2f}",
            f"AR: {aspect_ratio:.2f}",
            f"Area: {area}",
            f"Conf: {confidence:.2f}",
            f"Intersection: {result.get('is_at_intersection', False)}",
            f"Status: {result.get('status', 'unknown')}",
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
    
    def get_motor_speeds(self, result: Dict, base_speed: int = 60) -> Tuple[int, int, int, int]:
        """
        Calculates motor speeds based on the line detection result.
        'line_offset' is the key parameter used for PID control.
        A positive offset means the line is to the right of center.
        A negative offset means the line is to the left of center.
        """
        line_offset = result.get('line_offset', 0.0)
        line_confidence = result.get('line_confidence', 0.0)

        if result.get('status') == 'line_lost' or line_confidence < 0.2:
            # If line is lost or confidence is too low, stop.
            return 0, 0, 0, 0
            
        # --- Omni-wheel control logic ---
        # 1. Rotational correction to stay on the line (PID)
        omega = self.pid.update(line_offset)
        
        # 2. Sideways (strafing) correction for fine-tuning position
        vy = self.strafe_gain * line_offset
        
        # Combine forward, sideways, and rotational movements using the correct
        # kinematic model for an X-configured omni-drive.
        # vx = base_speed (forward)
        fl = int(base_speed - vy + omega)
        fr = int(base_speed + vy - omega)
        bl = int(base_speed + vy + omega)
        br = int(base_speed - vy - omega)

        return fl, fr, bl, br


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
