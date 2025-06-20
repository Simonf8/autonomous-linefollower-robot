#!/usr/bin/env python3

import cv2
import numpy as np
from collections import deque

class CameraLineDetector:
    """
    Detects the line position from a camera frame using a bird's-eye view
    transformation and provides a smoothed output.
    """
    def __init__(self, width, height, src_pts, dst_pts, smoothing_window=5, manual_threshold=100):
        self.width = width
        self.height = height
        
        # Bird's-eye view transformation
        self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Image processing kernels
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
        # Detection history for smoothing
        self.positions_history = deque(maxlen=smoothing_window)

        # Manual threshold value
        self.manual_threshold = manual_threshold

    def set_threshold(self, value: int):
        """Set the manual threshold value for line detection."""
        self.manual_threshold = int(value)
        print(f"Line detection threshold set to: {self.manual_threshold}")

    def _transform_to_bev(self, frame):
        """Transform the frame to a bird's-eye view."""
        return cv2.warpPerspective(frame, self.M, (self.width, self.height), flags=cv2.INTER_LINEAR)

    def _preprocess(self, frame):
        """Preprocess the frame for line detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # A bilateral filter is effective at smoothing while preserving edges.
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Use manual thresholding instead of adaptive
        _, binary = cv2.threshold(
            blurred, self.manual_threshold, 255, cv2.THRESH_BINARY_INV
        )
        
        # Clean up noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.morph_kernel)
        return binary

    def _find_line_center(self, binary_bev):
        """Finds the line center from the bottom half of the bird's-eye view image."""
        # Focus on the bottom half of the BEV image, which is closest to the robot
        roi = binary_bev[self.height // 2:, :]
        
        # Summing the pixels vertically gives a histogram of the line's position
        histogram = np.sum(roi, axis=0)
        
        if np.sum(histogram) < 1000: # Threshold for detecting any line
            return None, 0.0

        # The line center is the weighted average (centroid) of the histogram
        line_center_px = np.average(np.arange(len(histogram)), weights=histogram)
        
        # Confidence can be estimated from the strength of the signal
        confidence = np.sum(histogram) / (roi.shape[0] * roi.shape[1] * 255 * 0.1)
        
        return line_center_px, min(1.0, confidence)

    def detect(self, frame):
        """
        Detect the line, smooth the result, and return its normalized position.
        
        Returns:
            - normalized_position: Smoothed line position (-1.0 to 1.0) or None if not found.
            - confidence: Confidence of the detection (0.0 to 1.0).
            - debug_image: Enhanced debug image with navigation overlay for visualization.
        """
        bev_frame = self._transform_to_bev(frame)
        binary_bev = self._preprocess(bev_frame)
        
        center_px, confidence = self._find_line_center(binary_bev)
        
        # If a line is detected with sufficient confidence, update history
        if center_px is not None and confidence > 0.1:
            self.positions_history.append(center_px)
        
        # Create enhanced debug image for better navigation visualization
        debug_image = self._create_navigation_overlay(binary_bev, center_px, confidence)
        
        # If we have historical data, use the smoothed average
        if self.positions_history:
            smoothed_center_px = np.mean(self.positions_history)
            # Normalize to the range -1.0 to 1.0
            normalized_position = (smoothed_center_px - self.width / 2) / (self.width / 2)
            return normalized_position, confidence, debug_image
        
        return None, 0.0, debug_image

    def _create_navigation_overlay(self, binary_bev, center_px, confidence):
        """
        Create an enhanced debug image with navigation information.
        This shows the robot's perspective and navigation cues.
        """
        # Convert binary to color for better visualization
        debug_image = cv2.cvtColor(binary_bev, cv2.COLOR_GRAY2BGR)
        
        # Draw center line (where robot should be)
        center_line_x = self.width // 2
        cv2.line(debug_image, (center_line_x, 0), (center_line_x, self.height), (0, 255, 0), 2)
        
        # Draw detection zones
        left_zone = self.width // 4
        right_zone = 3 * self.width // 4
        cv2.line(debug_image, (left_zone, 0), (left_zone, self.height), (255, 255, 0), 1)
        cv2.line(debug_image, (right_zone, 0), (right_zone, self.height), (255, 255, 0), 1)
        
        # If line is detected, show its position and direction
        if center_px is not None:
            # Draw detected line center
            cv2.line(debug_image, (int(center_px), 0), (int(center_px), self.height), (0, 0, 255), 3)
            
            # Calculate and show error direction
            error_px = center_px - center_line_x
            if abs(error_px) > 10:  # Only show if significant error
                # Draw arrow showing correction direction
                arrow_start_y = self.height - 50
                arrow_end_x = center_line_x + (error_px * 0.5)  # Scale down for visibility
                cv2.arrowedLine(debug_image, 
                              (center_line_x, arrow_start_y), 
                              (int(arrow_end_x), arrow_start_y), 
                              (0, 255, 255), 3)
            
            # Add text information
            error_text = f"Error: {error_px:.1f}px"
            confidence_text = f"Conf: {confidence:.2f}"
            cv2.putText(debug_image, error_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(debug_image, confidence_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # No line detected
            cv2.putText(debug_image, "NO LINE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add robot direction indicator (bottom center)
        robot_indicator_y = self.height - 20
        cv2.circle(debug_image, (center_line_x, robot_indicator_y), 10, (255, 0, 255), -1)
        cv2.putText(debug_image, "ROBOT", (center_line_x - 25, robot_indicator_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return debug_image 