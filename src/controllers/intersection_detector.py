import cv2
import numpy as np
from typing import Optional, Tuple, List
import math

class IntersectionDetector:
    """
    Detects line intersections and corners from a camera frame to aid in position tracking.
    Combines vision detection with encoder data for improved accuracy.
    """
    def __init__(self, debug=False, cell_size_m=0.11):
        self.debug = debug
        self.cell_size_m = cell_size_m
        
        # Configuration for intersection detection
        self.black_threshold = 80
        self.blur_size = (5, 5)
        self.morph_kernel = np.ones((5, 5), np.uint8)
        self.min_contour_area = 1000
        
        # An intersection is detected if a horizontal line's width is >60% of the frame's width
        self.intersection_line_width_threshold = 0.6
        
        # Corner detection parameters
        self.corner_detection_enabled = True
        self.min_corner_angle = 45  # Minimum angle to consider as a corner (degrees)
        self.max_corner_angle = 135  # Maximum angle to consider as a corner (degrees)
        self.corner_line_length_threshold = 50  # Minimum line length for corner detection
        
        # Grid and encoder integration
        self.encoder_data = [0, 0, 0, 0]  # FL, FR, BL, BR encoder values
        self.last_encoder_data = [0, 0, 0, 0]
        self.grid_position = (0, 0)  # Current estimated grid position
        
        # Corner confidence tracking
        self.corner_confidence_threshold = 0.7
        self.consecutive_corner_detections = 0
        self.min_consecutive_detections = 3

    def update_encoder_data(self, encoder_ticks: List[int]):
        """
        Update encoder data from ESP32.
        
        Args:
            encoder_ticks: List of 4 encoder values [FL, FR, BL, BR]
        """
        if len(encoder_ticks) == 4:
            self.last_encoder_data = self.encoder_data.copy()
            self.encoder_data = encoder_ticks.copy()

    def detect(self, frame: np.ndarray) -> Optional[str]:
        """
        Detects intersections and corners in the given camera frame.

        Args:
            frame: The input camera frame (BGR).

        Returns:
            The type of event detected ('intersection', 'left_corner', 'right_corner'), or None.
        """
        if frame is None:
            return None

        # Pre-process the image
        processed_frame = self._preprocess_frame(frame)
        
        # First check for intersections (existing functionality)
        intersection_result = self._detect_intersection(processed_frame)
        if intersection_result:
            return intersection_result
        
        # Then check for corners if enabled
        if self.corner_detection_enabled:
            corner_result = self._detect_corners(processed_frame, frame)
            if corner_result:
                return corner_result
                
        return None

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess the frame for line detection.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Binary processed frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_size, 0)
        _, binary = cv2.threshold(blurred, self.black_threshold, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.morph_kernel)
        return binary

    def _detect_intersection(self, binary_frame: np.ndarray) -> Optional[str]:
        """
        Detect intersection using the existing algorithm.
        
        Args:
            binary_frame: Preprocessed binary frame
            
        Returns:
            'intersection' if detected, None otherwise
        """
        height, width = binary_frame.shape

        # Find the main line contour
        contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < self.min_contour_area:
            return None
            
        contour_mask = np.zeros_like(binary_frame)
        cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
        
        # Scan the upper-middle part of the image for a wide horizontal line segment
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

    def _detect_corners(self, binary_frame: np.ndarray, original_frame: np.ndarray) -> Optional[str]:
        """
        Detect corners using line detection and angle analysis.
        
        Args:
            binary_frame: Preprocessed binary frame
            original_frame: Original color frame for debugging
            
        Returns:
            'left_corner', 'right_corner', or None
        """
        # Use HoughLinesP to detect line segments
        lines = cv2.HoughLinesP(
            binary_frame,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=self.corner_line_length_threshold,
            maxLineGap=20
        )
        
        if lines is None or len(lines) < 2:
            self.consecutive_corner_detections = 0
            return None
        
        # Analyze line angles and positions
        corner_type = self._analyze_lines_for_corners(lines, binary_frame.shape, original_frame)
        
        if corner_type:
            self.consecutive_corner_detections += 1
            if self.consecutive_corner_detections >= self.min_consecutive_detections:
                if self.debug:
                    print(f"Corner confirmed: {corner_type} (confidence: {self.consecutive_corner_detections})")
                return corner_type
        else:
            self.consecutive_corner_detections = 0
            
        return None

    def _analyze_lines_for_corners(self, lines: np.ndarray, frame_shape: Tuple[int, int], 
                                  original_frame: np.ndarray) -> Optional[str]:
        """
        Analyze detected lines to determine corner type.
        
        Args:
            lines: Detected line segments from HoughLinesP
            frame_shape: Shape of the frame (height, width)
            original_frame: Original frame for debug visualization
            
        Returns:
            Corner type or None
        """
        height, width = frame_shape
        frame_center_x = width // 2
        
        # Group lines by angle
        horizontal_lines = []
        vertical_lines = []
        diagonal_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angle = abs(angle)
            
            # Classify line by angle
            if angle < 15 or angle > 165:  # Horizontal
                horizontal_lines.append((x1, y1, x2, y2, angle))
            elif 75 < angle < 105:  # Vertical
                vertical_lines.append((x1, y1, x2, y2, angle))
            elif 30 < angle < 60 or 120 < angle < 150:  # Diagonal
                diagonal_lines.append((x1, y1, x2, y2, angle))
        
        # Debug visualization
        if self.debug and len(horizontal_lines) > 0 and len(vertical_lines) > 0:
            debug_frame = original_frame.copy()
            for x1, y1, x2, y2, _ in horizontal_lines:
                cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for horizontal
            for x1, y1, x2, y2, _ in vertical_lines:
                cv2.line(debug_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for vertical
            
        # Detect corner based on line intersections and positions
        corner_type = self._determine_corner_type(horizontal_lines, vertical_lines, 
                                                 diagonal_lines, frame_center_x, height)
        
        # Combine with encoder data for validation
        if corner_type:
            corner_type = self._validate_corner_with_encoders(corner_type)
            
        return corner_type

    def _determine_corner_type(self, horizontal_lines: List, vertical_lines: List, 
                              diagonal_lines: List, frame_center_x: int, frame_height: int) -> Optional[str]:
        """
        Determine corner type based on line analysis.
        
        Args:
            horizontal_lines: List of horizontal line segments
            vertical_lines: List of vertical line segments
            diagonal_lines: List of diagonal line segments
            frame_center_x: X coordinate of frame center
            frame_height: Height of the frame
            
        Returns:
            Corner type or None
        """
        # Look for L-shaped patterns (horizontal + vertical lines)
        if len(horizontal_lines) > 0 and len(vertical_lines) > 0:
            # Find the most prominent vertical line
            main_vertical = max(vertical_lines, key=lambda line: abs(line[3] - line[1]))  # Longest vertical line
            v_x = (main_vertical[0] + main_vertical[2]) // 2  # Average X position of vertical line
            
            # Determine corner direction based on vertical line position
            if v_x < frame_center_x * 0.8:  # Vertical line on the left side
                return "left_corner"
            elif v_x > frame_center_x * 1.2:  # Vertical line on the right side
                return "right_corner"
        
        # Look for diagonal patterns that might indicate corners
        if len(diagonal_lines) > 0:
            # Analyze diagonal line positions and angles
            for x1, y1, x2, y2, angle in diagonal_lines:
                line_center_x = (x1 + x2) // 2
                
                # Check if diagonal line suggests a corner
                if 30 < angle < 60:  # Right-leaning diagonal
                    if line_center_x < frame_center_x:
                        return "left_corner"
                elif 120 < angle < 150:  # Left-leaning diagonal  
                    if line_center_x > frame_center_x:
                        return "right_corner"
        
        return None

    def _validate_corner_with_encoders(self, detected_corner: str) -> Optional[str]:
        """
        Validate detected corner with encoder data.
        
        Args:
            detected_corner: Corner type detected by vision
            
        Returns:
            Validated corner type or None
        """
        # Calculate encoder differences to detect turning motion
        encoder_diff = [
            self.encoder_data[i] - self.last_encoder_data[i] 
            for i in range(4)
        ]
        
        # Calculate if robot is turning based on encoder differences
        # For omni-wheel: FL, FR, BL, BR
        left_side_motion = (encoder_diff[0] + encoder_diff[2]) / 2  # FL + BL
        right_side_motion = (encoder_diff[1] + encoder_diff[3]) / 2  # FR + BR
        
        turn_direction = None
        turn_threshold = 5  # Minimum encoder difference to consider as turning
        
        if abs(left_side_motion - right_side_motion) > turn_threshold:
            if left_side_motion < right_side_motion:
                turn_direction = "left"
            else:
                turn_direction = "right"
        
        # Validate vision detection with encoder data
        if turn_direction is None:
            # No significant turning detected by encoders
            return None
        
        if detected_corner == "left_corner" and turn_direction == "left":
            if self.debug:
                print(f"Corner validated: {detected_corner} matches encoder turn direction {turn_direction}")
            return detected_corner
        elif detected_corner == "right_corner" and turn_direction == "right":
            if self.debug:
                print(f"Corner validated: {detected_corner} matches encoder turn direction {turn_direction}")
            return detected_corner
        else:
            if self.debug:
                print(f"Corner validation failed: vision={detected_corner}, encoders={turn_direction}")
            return None

    def update_grid_position(self, grid_x: int, grid_y: int):
        """
        Update the current grid position for comparison with detected events.
        
        Args:
            grid_x: X coordinate in grid
            grid_y: Y coordinate in grid
        """
        self.grid_position = (grid_x, grid_y)

    def compare_with_grid(self, pathfinder_grid: np.ndarray, detected_event: str) -> bool:
        """
        Compare detected event with expected grid layout.
        
        Args:
            pathfinder_grid: The maze grid from pathfinder
            detected_event: The detected event type
            
        Returns:
            True if event matches expected grid layout
        """
        if not detected_event or not hasattr(self, 'grid_position'):
            return False
            
        grid_x, grid_y = self.grid_position
        grid_height, grid_width = pathfinder_grid.shape
        
        # Check bounds
        if not (0 <= grid_x < grid_width and 0 <= grid_y < grid_height):
            return False
        
        # For corners, check if we're approaching a turn in the grid
        if detected_event in ["left_corner", "right_corner"]:
            return self._validate_corner_with_grid(pathfinder_grid, grid_x, grid_y, detected_event)
        elif detected_event == "intersection":
            return self._validate_intersection_with_grid(pathfinder_grid, grid_x, grid_y)
            
        return False

    def _validate_corner_with_grid(self, grid: np.ndarray, x: int, y: int, corner_type: str) -> bool:
        """
        Validate corner detection with grid layout.
        
        Args:
            grid: Pathfinder grid
            x, y: Current grid position
            corner_type: Detected corner type
            
        Returns:
            True if corner matches grid expectations
        """
        # Check surrounding cells to see if a corner is expected
        directions = {
            'up': (0, -1),
            'down': (0, 1), 
            'left': (-1, 0),
            'right': (1, 0)
        }
        
        adjacent_paths = []
        for direction, (dx, dy) in directions.items():
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < grid.shape[1] and 0 <= new_y < grid.shape[0] and 
                grid[new_y, new_x] == 0):  # 0 = path, 1 = obstacle
                adjacent_paths.append(direction)
        
        # A corner should have exactly 2 adjacent paths forming an L-shape
        if len(adjacent_paths) == 2:
            if corner_type == "left_corner":
                # Left corner: expect paths going up/down and left, or up and left/right
                expected_patterns = [
                    ('up', 'left'), ('down', 'left'),
                    ('up', 'right'), ('left', 'right')
                ]
            else:  # right_corner
                # Right corner: expect paths going up/down and right, or down and left/right  
                expected_patterns = [
                    ('up', 'right'), ('down', 'right'),
                    ('down', 'left'), ('left', 'right')
                ]
            
            for pattern in expected_patterns:
                if all(direction in adjacent_paths for direction in pattern):
                    if self.debug:
                        print(f"Grid validation: {corner_type} matches pattern {pattern}")
                    return True
        
        return False

    def _validate_intersection_with_grid(self, grid: np.ndarray, x: int, y: int) -> bool:
        """
        Validate intersection detection with grid layout.
        
        Args:
            grid: Pathfinder grid
            x, y: Current grid position
            
        Returns:
            True if intersection matches grid expectations
        """
        # Count adjacent path cells
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
        path_count = 0
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < grid.shape[1] and 0 <= new_y < grid.shape[0] and 
                grid[new_y, new_x] == 0):  # 0 = path
                path_count += 1
        
        # An intersection should have 3 or 4 adjacent paths
        is_intersection = path_count >= 3
        
        if self.debug and is_intersection:
            print(f"Grid validation: intersection at ({x},{y}) has {path_count} adjacent paths")
            
        return is_intersection

    def draw_debug_info(self, frame: np.ndarray, intersection_type: Optional[str]):
        """Draws debug information on the frame."""
        if intersection_type:
            # Color coding for different event types
            if intersection_type == "intersection":
                color = (0, 0, 255)  # Red
            elif intersection_type == "left_corner":
                color = (255, 0, 0)  # Blue
            elif intersection_type == "right_corner":
                color = (0, 255, 0)  # Green
            else:
                color = (0, 255, 255)  # Yellow
                
            cv2.putText(frame, f"Event: {intersection_type.upper()}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Show encoder validation info if available
            if hasattr(self, 'consecutive_corner_detections'):
                cv2.putText(frame, f"Confidence: {self.consecutive_corner_detections}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Show current grid position
        if hasattr(self, 'grid_position'):
            cv2.putText(frame, f"Grid: {self.grid_position}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
        return frame 