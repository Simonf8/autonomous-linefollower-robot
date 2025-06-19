#!/usr/bin/env python3

import cv2
import numpy as np

class CameraLineDetector:
    """Detects the line position from a camera frame."""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # Zone configuration
        self.zone_bottom_height = 0.4  # Use bottom 40% for primary detection
        self.zone_middle_height = 0.3  # Middle 30% for prediction

    def preprocess(self, frame):
        """Preprocess the frame for line detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Improve contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)
        
        # Adaptive thresholding is generally robust
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Clean up noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.morph_kernel)
        return binary

    def find_line_in_roi(self, roi):
        """Finds the line center in a region of interest."""
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, 0.0

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        area = cv2.contourArea(largest_contour)
        if area < 100:  # Min area to be considered a line
            return None, 0.0
            
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None, 0.0
            
        cx = int(M["m10"] / M["m00"])
        
        # Confidence based on area
        confidence = min(1.0, area / (roi.shape[1] * roi.shape[0] * 0.5))
        
        return cx, confidence

    def detect(self, frame):
        """Detect the line and return its normalized position."""
        binary_image = self.preprocess(frame)
        
        # Define detection zones
        bottom_y1 = int(self.height * (1 - self.zone_bottom_height))
        middle_y1 = int(self.height * (1 - self.zone_bottom_height - self.zone_middle_height))
        middle_y2 = bottom_y1

        # Extract ROIs
        bottom_roi = binary_image[bottom_y1:self.height, :]
        middle_roi = binary_image[middle_y1:middle_y2, :]

        # Detect line in zones
        bottom_cx, bottom_conf = self.find_line_in_roi(bottom_roi)
        middle_cx, middle_conf = self.find_line_in_roi(middle_roi)
        
        line_position_px = None
        
        if bottom_conf > 0.1:
            line_position_px = bottom_cx
        elif middle_conf > 0.1:
            line_position_px = middle_cx
            
        if line_position_px is not None:
            # Normalize to -1.0 to 1.0
            normalized_position = (line_position_px - self.width / 2) / (self.width / 2)
            return normalized_position, max(bottom_conf, middle_conf), binary_image
        
        return None, 0.0, binary_image 