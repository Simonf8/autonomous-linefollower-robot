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
            - debug_image: The binary bird's-eye view image for visualization.
        """
        bev_frame = self._transform_to_bev(frame)
        binary_bev = self._preprocess(bev_frame)
        
        center_px, confidence = self._find_line_center(binary_bev)
        
        # If a line is detected with sufficient confidence, update history
        if center_px is not None and confidence > 0.1:
            self.positions_history.append(center_px)
        
        # If we have historical data, use the smoothed average
        if self.positions_history:
            smoothed_center_px = np.mean(self.positions_history)
            # Normalize to the range -1.0 to 1.0
            normalized_position = (smoothed_center_px - self.width / 2) / (self.width / 2)
            return normalized_position, confidence, binary_bev
        
        return None, 0.0, binary_bev 