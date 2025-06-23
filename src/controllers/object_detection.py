#!/usr/bin/env python3

import cv2
import numpy as np
import logging
from ultralytics import YOLO
from typing import List, Tuple, Optional, Dict

class ObjectDetector:
    """Handles YOLO-based object detection for packages and obstacles."""
    
    def __init__(self, model_path: str = 'yolo11n.pt', confidence_threshold: float = 0.5):
        """
        Initialize object detector.
        
        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.yolo_model = None
        self.latest_frame = None
        
        # Object categories
        self.package_objects = {'box'}
        self.obstacle_objects = {
            'bottle', 'cup', 'bowl', 'banana', 'apple', 'orange', 'broccoli',
            'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'potted plant',
            'vase', 'sports ball', 'baseball bat', 'skateboard', 'surfboard', 
            'tennis racket', 'chair', 'dining table', 'couch', 'bed'
        }
        self.ignore_objects = {'tie', 'necktie'}
        
        self._initialize_model(model_path)
    
    def _initialize_model(self, model_path: str):
        """Initialize YOLO model."""
        try:
            self.yolo_model = YOLO(model_path)
        except Exception:
            self.yolo_model = None
    
    def detect_objects(self, frame: np.ndarray) -> Dict:
        """
        Detect objects in frame and categorize them.
        
        Args:
            frame: Input camera frame
            
        Returns:
            Dictionary with detection results
        """
        if self.yolo_model is None:
            return {
                'packages': [],
                'obstacles': [],
                'package_detected': False,
                'obstacle_detected': False
            }
        
        self.latest_frame = frame.copy()
        
        try:
            # Run YOLO detection
            results = self.yolo_model(frame, verbose=False)
            
            packages = []
            obstacles = []
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        confidence = float(box.conf[0])
                        
                        if confidence > self.confidence_threshold:
                            class_id = int(box.cls[0])
                            class_name = self.yolo_model.names[class_id].lower()
                            bbox = box.xyxy[0].cpu().numpy().astype(int)
                            
                            detection = {
                                'class_name': class_name,
                                'confidence': confidence,
                                'bbox': bbox,  # [x1, y1, x2, y2]
                                'center': self._get_bbox_center(bbox)
                            }
                            
                            # Categorize detection
                            if class_name in self.package_objects:
                                packages.append(detection)
                            elif class_name in self.obstacle_objects:
                                obstacles.append(detection)
                            elif class_name not in self.ignore_objects:
                                # Unknown object - treat as obstacle for safety
                                obstacles.append(detection)
            
            return {
                'packages': packages,
                'obstacles': obstacles,
                'package_detected': len(packages) > 0,
                'obstacle_detected': len(obstacles) > 0
            }
            
        except Exception:
            return {
                'packages': [],
                'obstacles': [],
                'package_detected': False,
                'obstacle_detected': False
            }
    
    def _get_bbox_center(self, bbox: np.ndarray) -> Tuple[int, int]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return (center_x, center_y)
    
    def draw_detections(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """
        Draw detection overlays on frame.
        
        Args:
            frame: Input frame
            detections: Detection results from detect_objects()
            
        Returns:
            Frame with overlays drawn
        """
        annotated_frame = frame.copy()
        
        # Draw packages in cyan
        for package in detections['packages']:
            bbox = package['bbox']
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = f"PACKAGE: {package['class_name']} ({package['confidence']:.2f})"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw obstacles in red
        for obstacle in detections['obstacles']:
            bbox = obstacle['bbox']
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"OBSTACLE: {obstacle['class_name']} ({obstacle['confidence']:.2f})"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return annotated_frame
    
    def get_closest_package(self, detections: Dict, frame_center: Tuple[int, int] = None) -> Optional[Dict]:
        """
        Get the package closest to frame center.
        
        Args:
            detections: Detection results
            frame_center: Center point of frame (default: calculate from frame)
            
        Returns:
            Closest package detection or None
        """
        packages = detections['packages']
        if not packages:
            return None
        
        if frame_center is None:
            frame_center = (320, 240)  # Default center for 640x480
        
        closest_package = None
        min_distance = float('inf')
        
        for package in packages:
            center = package['center']
            distance = np.sqrt((center[0] - frame_center[0])**2 + (center[1] - frame_center[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_package = package
        
        return closest_package
    
    def is_package_centered(self, package: Dict, frame_center: Tuple[int, int] = None, tolerance: int = 50) -> bool:
        """
        Check if a package is centered in the frame.
        
        Args:
            package: Package detection
            frame_center: Center point of frame
            tolerance: Pixel tolerance for "centered"
            
        Returns:
            True if package is centered within tolerance
        """
        if frame_center is None:
            frame_center = (320, 240)
        
        package_center = package['center']
        distance = np.sqrt((package_center[0] - frame_center[0])**2 + (package_center[1] - frame_center[1])**2)
        
        return distance <= tolerance

class LineObstacleDetector:
    """
    Detects obstacles directly on the line path using color and contour analysis.
    This is a simpler, faster alternative to YOLO for on-line obstacles.
    """
    def __init__(self, debug=False):
        self.debug = debug
        # Line detection parameters
        self.black_threshold = 80
        self.blur_size = (5, 5)
        self.min_line_contour_area = 100
        # Obstacle detection parameters
        self.obstacle_color_threshold = 100  # How bright an object is to be an obstacle
        self.min_obstacle_area = 500         # Min area to be considered a blocking obstacle

    def detect(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detects an obstacle if it is physically blocking the detected line path.

        Args:
            frame: The input camera frame (BGR).

        Returns:
            A dictionary with obstacle info if detected, otherwise None.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_size, 0)
        
        # 1. Find the main line
        _, line_mask = cv2.threshold(blurred, self.black_threshold, 255, cv2.THRESH_BINARY_INV)
        line_contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not line_contours:
            return None # No line found, so can't detect obstacle on it

        main_line_contour = max(line_contours, key=cv2.contourArea)
        if cv2.contourArea(main_line_contour) < self.min_line_contour_area:
            return None # Line is too small to be reliable

        # 2. Identify potential obstacles (bright areas)
        _, obstacle_mask = cv2.threshold(blurred, self.obstacle_color_threshold, 255, cv2.THRESH_BINARY)

        # 3. Find where obstacles intersect with the line path
        line_area_mask = np.zeros_like(gray)
        cv2.drawContours(line_area_mask, [main_line_contour], -1, 255, -1)
        
        blocking_mask = cv2.bitwise_and(obstacle_mask, line_area_mask)
        
        # 4. Find the contours of the blocking objects
        blocking_contours, _ = cv2.findContours(blocking_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not blocking_contours:
            return None # No obstacles are blocking the line

        # Find the largest blocking object
        largest_blocker = max(blocking_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_blocker)

        if area > self.min_obstacle_area:
            M = cv2.moments(largest_blocker)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                if self.debug:
                    print(f"Blocking obstacle detected at ({cx}, {cy}) with area {area}")

                return {
                    "area": area,
                    "center": (cx, cy),
                    "contour": largest_blocker
                }

        return None

class PathShapeDetector:
    """Detects path shapes and upcoming turns using computer vision."""
    
    def __init__(self, src_points: np.ndarray, dst_points: np.ndarray):
        """
        Initialize path shape detector.
        
        Args:
            src_points: Source points for perspective transform
            dst_points: Destination points for perspective transform
        """
        self.perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    def detect_path_shape(self, frame: np.ndarray) -> str:
        """
        Analyze frame to detect upcoming path shapes.
        
        Args:
            frame: Input camera frame
            
        Returns:
            Path shape: "STRAIGHT", "CORNER_LEFT", "CORNER_RIGHT", or "UNKNOWN"
        """
        try:
            # Apply perspective transform for bird's eye view
            warped_img = cv2.warpPerspective(frame, self.perspective_matrix, 
                                           (frame.shape[1], frame.shape[0]))
            
            # Convert to grayscale and find edges
            gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, 
                                  minLineLength=50, maxLineGap=20)
            
            if lines is None:
                return "STRAIGHT"
            
            # Analyze line angles
            left_lines, right_lines = [], []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue  # Skip vertical lines
                
                angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
                
                # Classify lines based on angle
                if -80 < angle < -10:
                    left_lines.append(line)
                elif 10 < angle < 80:
                    right_lines.append(line)
            
            # Determine path shape based on line distribution
            if len(right_lines) > len(left_lines) + 5:
                return "CORNER_LEFT"
            elif len(left_lines) > len(right_lines) + 5:
                return "CORNER_RIGHT"
            
            return "STRAIGHT"
            
        except Exception:
            return "UNKNOWN" 