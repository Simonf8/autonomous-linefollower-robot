#!/usr/bin/env python3

import cv2
import numpy as np
import time
from typing import Tuple, Optional, Dict, List
from collections import deque

class CameraLineFollower:
    """
    Camera-based line detection and following system.
    Replaces ESP32 hardware sensors with computer vision.
    """
    
    def __init__(self, debug=False):
        self.debug = debug
        
        # Line detection parameters
        self.black_threshold = 80
        self.blur_size = (5, 5)
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.min_contour_area = 200
        
        # ROI configuration (focus on bottom portion of frame for line following)
        self.roi_height_ratio = 0.4  # Use bottom 40% of frame
        self.roi_width_ratio = 1.0   # Use full width
        
        # Line following state
        self.line_history = deque(maxlen=5)  # Smooth line position over frames
        self.confidence_history = deque(maxlen=3)
        self.last_known_line_x = None
        self.frames_without_line = 0
        
        # Intersection detection
        self.intersection_threshold = 0.7  # How much of frame width to consider intersection
        self.intersection_cooldown = 2.0   # Seconds between intersection detections
        self.last_intersection_time = 0
        
        # Enhanced PID parameters like simple_robot.py
        self.kp = 0.30  # Proportional gain (reduced for less aggressive response)
        self.ki = 0.005  # Integral gain
        self.kd = 0.20  # Derivative gain
        self.max_turn_correction = 0.8  # Maximum turn correction
        
        # PID state
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        self.max_integral = 2.4
        
    def detect_line(self, frame: np.ndarray) -> Dict:
        """
        Detect line using proven multi-zone detection approach.
        
        Args:
            frame: Input camera frame (BGR)
            
        Returns:
            Dictionary with line detection results:
            - line_detected: bool
            - line_position: int (x-coordinate of line center, or None)
            - line_offset: float (-1.0 to 1.0, relative to frame center)
            - confidence: float (0.0 to 1.0)
            - intersection_detected: bool
            - processed_frame: frame with debug visualization (if debug=True)
        """
        if frame is None:
            return self._empty_result()
        
        height, width = frame.shape[:2]
        
        # Enhanced preprocessing like simple_robot.py
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Binary threshold (inverted for black line detection)
        _, binary = cv2.threshold(blurred, self.black_threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Multi-zone detection like simple_robot.py
        # Bottom zone (most reliable for line following)
        bottom_height = int(height * 0.30)  # Bottom 30%
        bottom_roi = binary[height - bottom_height:height, :]
        
        # Middle zone (for predictive tracking)
        middle_height = int(height * 0.25)  # Middle 25%
        middle_start = height - bottom_height - middle_height
        middle_roi = binary[middle_start:height - bottom_height, :]
        
        # Detect line features in each zone
        bottom_features = self._detect_line_features(bottom_roi, "bottom", width)
        middle_features = self._detect_line_features(middle_roi, "middle", width)
        
        # Determine primary line position (prioritize bottom zone)
        line_detected = False
        line_position = None
        confidence = 0.0
        zone_used = None
        
        # Try bottom zone first (most reliable)
        if bottom_features and bottom_features['confidence'] > 0.15:
            line_position = bottom_features['center'][0]
            confidence = bottom_features['confidence']
            line_detected = True
            zone_used = "bottom"
        # If no bottom line, try middle zone  
        elif middle_features and middle_features['confidence'] > 0.2:
            line_position = middle_features['center'][0]
            confidence = middle_features['confidence'] * 0.9
            line_detected = True
            zone_used = "middle"
        # If still no line but we have ANY features, try to use them
        elif bottom_features and bottom_features['confidence'] > 0.05:
            line_position = bottom_features['center'][0]
            confidence = bottom_features['confidence'] * 0.5
            line_detected = True
            zone_used = "bottom_weak"
        elif middle_features and middle_features['confidence'] > 0.1:
            line_position = middle_features['center'][0]
            confidence = middle_features['confidence'] * 0.5
            line_detected = True
            zone_used = "middle_weak"
        
        line_result = {}
        
        if line_detected:
            # Reset frames without line counter
            self.frames_without_line = 0
            
            # Calculate offset from center (-1.0 to +1.0)
            center_x = width / 2
            offset = (line_position - center_x) / center_x
            offset = max(-1.0, min(1.0, offset))
            
            # Smooth the offset using history
            self.line_history.append(offset)
            if len(self.line_history) >= 3:
                # Use weighted average like simple_robot.py
                weights = np.array([0.5, 0.3, 0.2])[-len(self.line_history):]
                positions = list(self.line_history)[-len(weights):]
                smoothed_offset = np.average(positions, weights=weights)
            else:
                smoothed_offset = offset
            
            line_result = {
                'line_detected': True,
                'line_position': line_position,
                'line_offset': smoothed_offset,
                'confidence': confidence,
                'zone_used': zone_used,
                'contour': bottom_features['contour'] if zone_used.startswith('bottom') else middle_features['contour'],
                'center': (line_position, height - bottom_height // 2)  # Approximate center
            }
        else:
            # No line detected
            self.frames_without_line += 1
            line_result = self._no_line_result()
        
        # Check for intersection
        intersection_detected = self._detect_intersection(binary, width)
        line_result['intersection_detected'] = intersection_detected
        
        # Create debug visualization if requested
        if self.debug:
            line_result['processed_frame'] = self._create_debug_frame_multi_zone(
                frame, binary, bottom_roi, middle_roi, bottom_features, middle_features, line_result, intersection_detected
            )
        else:
            line_result['processed_frame'] = frame
        
        return line_result
    
    def _detect_line_features(self, roi: np.ndarray, zone_name: str, frame_width: int) -> Optional[Dict]:
        """Detect line features in ROI using contour analysis like simple_robot.py"""
        if roi.size == 0:
            return None
        
        height, width = roi.shape
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter and analyze contours
        line_features = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # MIN_CONTOUR_AREA
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate center using moments
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = x + w // 2
                cy = y + h // 2
            
            # Calculate confidence based on area and shape
            aspect_ratio = w / h if h > 0 else 0
            solidity = area / (w * h) if w * h > 0 else 0
            
            # Line-like features scoring
            if zone_name == "bottom":
                confidence = min(1.0, (area / (width * height * 0.1)) * solidity)
            else:
                confidence = min(1.0, (area / (width * height * 0.15)) * solidity * 0.8)
            
            line_features.append({
                'center': (cx, cy),
                'confidence': confidence,
                'area': area,
                'contour': contour,
                'bbox': (x, y, w, h)
            })
        
        if not line_features:
            return None
        
        # Select the best line feature
        best_feature = max(line_features, key=lambda f: f['confidence'] * f['area'])
        return best_feature

    def _process_contours(self, contours: List, frame_width: int, roi_offset_y: int) -> Dict:
        """Process contours to find the main line."""
        if not contours:
            self.frames_without_line += 1
            return self._no_line_result()
        
        # Filter contours by area
        valid_contours = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]
        
        if not valid_contours:
            self.frames_without_line += 1
            return self._no_line_result()
        
        # Find the largest contour (assume it's the main line)
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # Calculate center of the line
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            self.frames_without_line += 1
            return self._no_line_result()
        
        # Line center coordinates
        line_x = int(M["m10"] / M["m00"])
        line_y = int(M["m01"] / M["m00"]) + roi_offset_y
        
        # Calculate confidence based on contour area
        contour_area = cv2.contourArea(largest_contour)
        roi_area = frame_width * (roi_offset_y)
        confidence = min(1.0, contour_area / (roi_area * 0.1))
        
        # Calculate line offset from center (-1.0 to 1.0)
        center_x = frame_width // 2
        line_offset = (line_x - center_x) / center_x
        
        # Update history for smoothing
        self.line_history.append(line_x)
        self.confidence_history.append(confidence)
        self.last_known_line_x = line_x
        self.frames_without_line = 0
        
        # Use smoothed position if we have enough history
        if len(self.line_history) >= 3:
            smoothed_x = int(np.mean(list(self.line_history)[-3:]))
            smoothed_offset = (smoothed_x - center_x) / center_x
        else:
            smoothed_offset = line_offset
        
        return {
            'line_detected': True,
            'line_position': line_x,
            'line_offset': smoothed_offset,
            'confidence': confidence,
            'contour': largest_contour,
            'center': (line_x, line_y)
        }
    
    def _detect_intersection(self, binary_roi: np.ndarray, frame_width: int) -> bool:
        """
        Detect intersections by looking for wide horizontal line segments.
        """
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_intersection_time < self.intersection_cooldown:
            return False
        
        height, width = binary_roi.shape
        
        # Scan horizontal lines in the middle portion of ROI
        scan_start = height // 4
        scan_end = 3 * height // 4
        
        for y in range(scan_start, scan_end, 2):  # Sample every 2 pixels
            row = binary_roi[y, :]
            white_pixels = np.sum(row > 0)
            
            # If this row has a significant portion of white pixels, it might be an intersection
            if white_pixels > frame_width * self.intersection_threshold:
                # Double-check by looking at nearby rows
                nearby_rows = binary_roi[max(0, y-2):min(height, y+3), :]
                avg_white_ratio = np.mean(nearby_rows > 0)
                
                if avg_white_ratio > self.intersection_threshold:
                    self.last_intersection_time = current_time
                    return True
        
        return False
    
    def _no_line_result(self) -> Dict:
        """Return result when no line is detected."""
        return {
            'line_detected': False,
            'line_position': None,
            'line_offset': 0.0,
            'confidence': 0.0,
            'contour': None,
            'center': None
        }
    
    def _empty_result(self) -> Dict:
        """Return empty result for invalid input."""
        return {
            'line_detected': False,
            'line_position': None,
            'line_offset': 0.0,
            'confidence': 0.0,
            'intersection_detected': False,
            'processed_frame': None,
            'contour': None,
            'center': None
        }
    
    def _create_debug_frame(self, frame: np.ndarray, roi_offset_y: int, 
                           binary_roi: np.ndarray, line_result: Dict, 
                           intersection_detected: bool) -> np.ndarray:
        """Create debug visualization frame."""
        debug_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw ROI boundary
        cv2.rectangle(debug_frame, (0, roi_offset_y), (width, height), (255, 255, 0), 2)
        
        # Draw line detection
        if line_result['line_detected'] and line_result['contour'] is not None:
            # Adjust contour coordinates for full frame
            adjusted_contour = line_result['contour'].copy()
            adjusted_contour[:, :, 1] += roi_offset_y
            
            cv2.drawContours(debug_frame, [adjusted_contour], -1, (0, 255, 0), 2)
            
            # Draw line center
            if line_result['center']:
                cv2.circle(debug_frame, line_result['center'], 5, (255, 0, 0), -1)
            
            # Draw center line and detected line
            center_x = width // 2
            line_x = line_result['line_position']
            cv2.line(debug_frame, (center_x, 0), (center_x, height), (0, 255, 255), 1)
            cv2.line(debug_frame, (line_x, roi_offset_y), (line_x, height), (255, 0, 255), 2)
        
        # Draw intersection indicator
        if intersection_detected:
            cv2.putText(debug_frame, "INTERSECTION!", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw status information
        status_text = [
            f"Line: {'YES' if line_result['line_detected'] else 'NO'}",
            f"Confidence: {line_result['confidence']:.2f}",
            f"Offset: {line_result['line_offset']:.2f}",
            f"Frames w/o line: {self.frames_without_line}"
        ]
        
        for i, text in enumerate(status_text):
            cv2.putText(debug_frame, text, (10, 60 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return debug_frame
    
    def calculate_steering(self, line_result: Dict) -> float:
        """
        Calculate steering using enhanced PID control like simple_robot.py
        
        Args:
            line_result: Result from detect_line()
            
        Returns:
            Steering value (-1.0 to 1.0, negative=left, positive=right)
        """
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        dt = max(dt, 0.001)  # Prevent division by zero
        
        if not line_result['line_detected']:
            # No line detected - reset PID and return gentle search
            self.previous_error = 0.0
            self.integral = 0.0
            return 0.0
        
        # Line detected - calculate error
        error = line_result['line_offset']  # Already normalized to -1.0 to 1.0
        confidence = line_result['confidence']
        
        # PID calculations
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with windup protection
        self.integral += error * dt
        self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative
        
        # Total steering output
        steering = p_term + i_term + d_term
        
        # Scale by confidence (less confident = more conservative steering)
        steering *= confidence
        
        # Clamp to maximum turn correction
        steering = max(-self.max_turn_correction, min(self.max_turn_correction, steering))
        
        # Update for next iteration
        self.previous_error = error
        
        return steering
    
    def get_motor_speeds(self, line_result: Dict, base_speed: int = 50) -> Tuple[int, int, int, int]:
        """
        Convert line detection to motor speeds using enhanced steering like simple_robot.py
        
        Args:
            line_result: Result from detect_line()
            base_speed: Base forward speed (0-100)
            
        Returns:
            Tuple of motor speeds (fl, fr, bl, br)
        """
        steering = self.calculate_steering(line_result)
        
        # Variable turn rates based on steering magnitude (like simple_robot.py)
        abs_steering = abs(steering)
        
        # Determine turn type and speeds
        if abs_steering < 0.03:  # Dead zone - go straight
            return (base_speed, base_speed, base_speed, base_speed)
        
        elif abs_steering > 0.5:  # Sharp turn
            if not line_result['line_detected']:
                # If no line, stop
                return (0, 0, 0, 0)
            # Sharp turn - differential steering
            turn_speed = int(base_speed * 0.8)  # Reduce speed for sharp turns
            if steering > 0:  # Turn right
                return (turn_speed, int(turn_speed * 0.3), turn_speed, int(turn_speed * 0.3))
            else:  # Turn left
                return (int(turn_speed * 0.3), turn_speed, int(turn_speed * 0.3), turn_speed)
        
        elif abs_steering > 0.25:  # Medium turn
            # Medium turn - gentle differential
            turn_correction = steering * base_speed * 0.6
            left_speed = int(base_speed - turn_correction)
            right_speed = int(base_speed + turn_correction)
            
            # Clamp speeds
            left_speed = max(0, min(100, left_speed))
            right_speed = max(0, min(100, right_speed))
            
            return (left_speed, right_speed, left_speed, right_speed)
        
        else:  # Gentle turn
            # Gentle turn - small correction
            turn_correction = steering * base_speed * 0.3
            left_speed = int(base_speed - turn_correction)
            right_speed = int(base_speed + turn_correction)
            
            # Clamp speeds
            left_speed = max(int(base_speed * 0.7), min(100, left_speed))
            right_speed = max(int(base_speed * 0.7), min(100, right_speed))
            
            return (left_speed, right_speed, left_speed, right_speed)
    
    def _create_debug_frame_multi_zone(self, frame: np.ndarray, binary: np.ndarray, 
                                     bottom_roi: np.ndarray, middle_roi: np.ndarray,
                                     bottom_features: Optional[Dict], middle_features: Optional[Dict],
                                     line_result: Dict, intersection_detected: bool) -> np.ndarray:
        """Create debug visualization for multi-zone detection"""
        debug_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw zone boundaries
        bottom_height = int(height * 0.30)
        middle_height = int(height * 0.25)
        middle_start = height - bottom_height - middle_height
        
        # Bottom zone (yellow)
        cv2.rectangle(debug_frame, (0, height - bottom_height), (width, height), (0, 255, 255), 2)
        cv2.putText(debug_frame, "BOTTOM", (5, height - bottom_height + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Middle zone (cyan)
        cv2.rectangle(debug_frame, (0, middle_start), (width, height - bottom_height), (255, 255, 0), 2)
        cv2.putText(debug_frame, "MIDDLE", (5, middle_start + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw detected features
        if bottom_features:
            cx, cy = bottom_features['center']
            cv2.circle(debug_frame, (cx, cy + height - bottom_height), 8, (0, 255, 255), -1)
            cv2.putText(debug_frame, f"B:{bottom_features['confidence']:.2f}", 
                       (cx + 10, cy + height - bottom_height), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        if middle_features:
            cx, cy = middle_features['center']
            cv2.circle(debug_frame, (cx, cy + middle_start), 8, (255, 255, 0), -1)
            cv2.putText(debug_frame, f"M:{middle_features['confidence']:.2f}", 
                       (cx + 10, cy + middle_start), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Draw line center and offset
        if line_result['line_detected']:
            line_x = line_result['line_position']
            center_x = width // 2
            
            # Draw line center
            cv2.line(debug_frame, (line_x, 0), (line_x, height), (255, 0, 255), 2)
            
            # Draw offset visualization
            cv2.line(debug_frame, (center_x, height - 30), (line_x, height - 30), (0, 255, 0), 3)
            
            # Status text
            zone = line_result.get('zone_used', 'unknown')
            offset = line_result['line_offset']
            conf = line_result['confidence']
            
            cv2.putText(debug_frame, f"Zone: {zone}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(debug_frame, f"Offset: {offset:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(debug_frame, f"Conf: {conf:.3f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(debug_frame, "NO LINE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Intersection indicator
        if intersection_detected:
            cv2.putText(debug_frame, "INTERSECTION!", (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return debug_frame 