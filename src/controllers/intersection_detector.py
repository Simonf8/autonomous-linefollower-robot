import cv2
import numpy as np
from typing import Optional, Tuple

class IntersectionDetector:
    """
    Detects line intersections from a camera frame to aid in position tracking.
    """
    def __init__(self, debug=False):
        self.debug = debug
        # Configuration for intersection detection
        self.black_threshold = 80
        self.blur_size = (5, 5)
        self.morph_kernel = np.ones((5, 5), np.uint8)
        self.min_contour_area = 1000
        
        self.intersection_line_width_threshold = 0.6  

    def detect(self, frame: np.ndarray) -> Optional[str]:
        """
        Detects intersections in the given camera frame.

        Args:
            frame: The input camera frame (BGR).

        Returns:
            The type of intersection detected (e.g., 'intersection'), or None.
        """
        if frame is None:
            return None

        # Pre-process the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_size, 0)
        _, binary = cv2.threshold(blurred, self.black_threshold, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.morph_kernel)

        height, width = binary.shape

        # Find the main line contour
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < self.min_contour_area:
            return None
            
        contour_mask = np.zeros_like(binary)
        cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
        
        
        scan_y_start = height // 4
        scan_y_end = int(height * 0.75)

        for y in range(scan_y_start, scan_y_end):
            row = contour_mask[y, :]
            line_indices = np.where(row > 0)[0]
            
            if len(line_indices) > 0:
                line_width = line_indices[-1] - line_indices[0]
                line_width_ratio = line_width / width
                
                if line_width_ratio > self.intersection_line_width_threshold:
                    if self.debug:
                        print(f"Intersection detected at y={y} with width ratio {line_width_ratio:.2f}")
                    return "intersection"
            
        return None

    def draw_debug_info(self, frame: np.ndarray, intersection_type: Optional[str]):
        """Draws debug information on the frame."""
        if intersection_type:
            cv2.putText(frame, f"Event: {intersection_type.upper()}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame 
    
    def get_intersection_info(self, frame: np.ndarray):
        """get intersection info from the frame"""
        if frame is None:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self)
        _, binary = cv2.threshold(blurres, self.black_threshold, 255, cv2.THRESH_BINNARY)
        binary = cv2.morphologyEx(binary, )
    
    def center_of_mass(self, frame, binary):
        """get the center of mass of the binary image"""
        moments = cv2.moments(binary)
        if moments['m00'] == 0:
            return None
        
        cx = int(moments['m10'])/moments['m00']
        cy = int(moments['m01'])/moments['m00']
        return cx, cy
    
    def get_intersection_info(self, frame: np.ndarray):
        """get intersection info from the frame"""
        if frame is None:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blursize, 0)
        _, binary = cv2.threshold(blurred, self.blackthreshld, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, self.morph_kernel, self.morph_kernel)
        
        
    