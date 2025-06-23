import cv2
import time
import numpy as np
import math

from intersection_detector import IntersectionDetector
from line_obstacle_detector import LineObstacleDetector
from camera_line_follower import CameraLineFollower

# Constants for perspective transform - this should be tuned for the specific camera setup
# These points define a trapezoid in the source image that will be mapped to a rectangle
SRC_PTS = np.float32([[100, 220], [540, 220], [640, 300], [0, 300]])
DST_PTS = np.float32([[0, 0], [640, 0], [640, 240], [0, 240]])
TRANSFORM_MATRIX = cv2.getPerspectiveTransform(SRC_PTS, DST_PTS)

class Perception:
    """Handles all vision-based perception tasks."""
    
    def __init__(self, features: dict):
        self.features = features
        self.intersection_detector = None
        self.line_obstacle_detector = None
        self.object_detector = None # Placeholder for future object detection
        
        self.camera_line_follower = CameraLineFollower(debug=self.features['DEBUG_VISUALIZATION_ENABLED'])
        self.latest_line_result = None
        self.debug_frame = None

        self._setup_detectors()

    def _setup_detectors(self):
        """Initialize vision systems based on feature flags."""
        if not self.features['VISION_SYSTEM_ENABLED']:
            print("Vision system is disabled.")
            return

        print("Initializing perception detectors...")
        if self.features['INTERSECTION_CORRECTION_ENABLED']:
            self.intersection_detector = IntersectionDetector(debug=self.features['DEBUG_VISUALIZATION_ENABLED'])
        
        if self.features['OBSTACLE_AVOIDANCE_ENABLED']:
            self.line_obstacle_detector = LineObstacleDetector(debug=self.features['DEBUG_VISUALIZATION_ENABLED'])

    def estimate_pose_from_grid(self, frame, last_known_cell, cell_size_m):
        """
        Estimates the robot's precise (x, y, heading) pose from the grid lines.
        """
        # 1. Get a top-down view of the scene
        h, w = frame.shape[:2]
        warped_img = cv2.warpPerspective(frame, TRANSFORM_MATRIX, (w, h))
        
        # 2. Detect lines using Hough Transform
        gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=50, maxLineGap=10)

        if lines is None:
            return None # No lines detected

        # 3. Filter and classify lines into horizontal and vertical
        horz_lines, vert_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            if angle < 10 or angle > 170: # Horizontal
                horz_lines.append(line[0])
            elif angle > 80 and angle < 100: # Vertical
                vert_lines.append(line[0])

        if not horz_lines or not vert_lines:
            return None # Need both horizontal and vertical lines for a pose

        # 4. Geometric calculations
        # Find the average y of horizontal lines and average x of vertical lines
        # This gives us the center of the visible grid intersection
        avg_horz_y = np.mean([line[1] for line in horz_lines])
        avg_vert_x = np.mean([line[0] for line in vert_lines])
        
        # The center of the image is the robot's camera position
        img_center_x, img_center_y = w / 2, h / 2

        # Calculate offset in pixels from the center of the image to the grid intersection
        pixel_offset_x = avg_vert_x - img_center_x
        pixel_offset_y = avg_horz_y - img_center_y

        # TODO: This pixel-to-meter conversion needs calibration
        # For now, use a simple ratio based on the warped view's known size
        pixels_per_meter_x = w / (2 * cell_size_m) 
        pixels_per_meter_y = h / (1.5 * cell_size_m)

        offset_m_x = pixel_offset_x / pixels_per_meter_x
        offset_m_y = pixel_offset_y / pixels_per_meter_y

        # Calculate heading from a dominant vertical line's angle
        dominant_vert_line = max(vert_lines, key=lambda l: (l[3]-l[1])**2)
        vx1, vy1, vx2, vy2 = dominant_vert_line
        heading_rad = math.atan2(vy2 - vy1, vx2 - vx1) - math.pi / 2
        # Normalize heading to be within [-pi, pi]
        heading_rad = (heading_rad + np.pi) % (2 * np.pi) - np.pi

        # Combine with last known cell to get world pose
        pose_x = last_known_cell[0] * cell_size_m - offset_m_x
        pose_y = last_known_cell[1] * cell_size_m - offset_m_y

        return (pose_x, pose_y, heading_rad)

    def process_frame(self, frame, state: str, current_cell, estimated_heading) -> tuple:
        """
        Run all perception tasks on a single camera frame.
        Returns a tuple of (processed_frame, event_detected).
        An event could be an obstacle, an intersection, etc.
        """
        if not self.features['VISION_SYSTEM_ENABLED'] or frame is None:
            return frame, None

        processed_frame = frame.copy()
        event = None

        # --- Obstacle Detection ---
        if self.features['OBSTACLE_AVOIDANCE_ENABLED'] and self.line_obstacle_detector and state == "path_following":
            if self.line_obstacle_detector.detect(processed_frame):
                event = {"type": "obstacle"}

        # --- Line and Intersection Detection ---
        line_result = self.camera_line_follower.detect_line(processed_frame)
        self.latest_line_result = line_result
        self.debug_frame = line_result.get('processed_frame', processed_frame)
        
        # This can be expanded to return more detailed intersection events
        if line_result.get('intersection_detected', False):
             event = {"type": "intersection"}

        return processed_frame, event

    def get_line_following_speeds(self, speed):
        """Get motor speeds for line following from the latest frame result."""
        if self.latest_line_result:
            return self.camera_line_follower.get_motor_speeds(self.latest_line_result, speed)
        return 0, 0, 0, 0 