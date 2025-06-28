import cv2
import numpy as np
import threading
import time
from typing import Tuple, Optional, Dict, List
import math
from picamera2 import Picamera2

class PreciseMazeLocalizer:
    def __init__(self, maze, start_pos: Tuple[int, int], camera_width: int, camera_height: int, camera_fps: int, start_direction: str = 'N'):
        # Your exact maze
        self.maze = maze
        
        self.current_pos = start_pos
        self.current_direction = start_direction
        
        # Initialize camera
        self.cap = None # Will be Picamera2 instance
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.latest_frame = None
        
        # Movement detection - adjusted for line following
        self.prev_frame = None
        self.movement_threshold = 30  # Lower threshold
        self.stationary_frames = 0
        self.max_stationary_frames = 30  # Allow more stationary frames
        self.blur_threshold = 100.0 # New: tunable threshold for blur detection
        
        # Corner distance detection
        self.close_corner_threshold = 0.7
        self.far_corner_threshold = 0.3
        
        # Create precise signatures for your maze
        self.corner_signatures = self.create_precise_signatures()
        
        # Confidence tracking
        self.position_confidence = 1.0
        self.min_confidence = 0.6
        self.search_mode = 'normal' # 'normal' or 'corridor'
        
        # New: Stability tracking for position
        self.position_candidate = None
        self.candidate_confidence = 0.0
        self.candidate_stability_counter = 0
        self.STABILITY_THRESHOLD = 3 # Require 3 consecutive good frames
        
        # Status tracking
        self.last_status = {
            'status': 'initialized',
            'position': self.current_pos,
            'direction': self.current_direction,
            'confidence': self.position_confidence,
            'scene_type': 'unknown',
            'is_moving': False,
            'message': 'System initialized'
        }
        
        # Threading
        self.running = False
        self.localization_thread = None
        self.frame_lock = threading.Lock()
        self.initialization_frames = camera_fps # Grace period before localizing
        
    def initialize_camera(self):
        """Initializes the Raspberry Pi Camera Module."""
        if self.cap is None:
            try:
                self.cap = Picamera2()
                config = self.cap.create_preview_configuration(
                    main={"size": (self.camera_width, self.camera_height), "format": "RGB888"}
                )
                self.cap.configure(config)
                self.cap.start()
                # Allow the camera to warm up
                time.sleep(1.0) 
                print(f"Successfully opened PiCamera with {self.camera_width}x{self.camera_height} resolution.")
                return True
            except Exception as e:
                print(f"Error: Could not open PiCamera. {e}")
                self.cap = None
                return False
        return True

    def create_precise_signatures(self) -> Dict:
        """Create precise corner signatures for each valid position in YOUR maze"""
        signatures = {}
        
        for y in range(len(self.maze)):
            for x in range(len(self.maze[0])):
                if self.maze[y][x] == 0:  # Valid path
                    for direction in ['N', 'S', 'E', 'W']:
                        sig = self.analyze_maze_position(x, y, direction)
                        signatures[(x, y, direction)] = sig
        
        return signatures
    
    def analyze_maze_position(self, x: int, y: int, direction: str) -> Dict:
        """Analyze what should be visible from position (x,y) facing direction"""
        signature = {
            'scene_type': 'unknown',
            'corner_ahead_left': False,
            'corner_ahead_right': False,
            'wall_ahead': False,
            'opening_left': False,
            'opening_right': False,
            'intersection_type': 'none',
            'unique_features': []
        }
        
        # Get surrounding positions based on direction
        if direction == 'N':
            ahead = (x, y-1)
            left = (x-1, y)
            right = (x+1, y)
        elif direction == 'S':
            ahead = (x, y+1)
            left = (x+1, y)
            right = (x-1, y)
        elif direction == 'E':
            ahead = (x+1, y)
            left = (x, y-1)
            right = (x, y+1)
        elif direction == 'W':
            ahead = (x-1, y)
            left = (x, y+1)
            right = (x, y-1)
        
        # Check each position
        wall_ahead = self.is_wall_at(ahead)
        opening_left = not self.is_wall_at(left)
        opening_right = not self.is_wall_at(right)
        
        signature['wall_ahead'] = wall_ahead
        signature['opening_left'] = opening_left
        signature['opening_right'] = opening_right
        
        # Determine scene type
        if not wall_ahead and opening_left and opening_right:
            signature['scene_type'] = 'intersection'
            signature['intersection_type'] = '4way'
        elif not wall_ahead and (opening_left or opening_right):
            signature['scene_type'] = 'T_junction'
            signature['intersection_type'] = 'T'
        elif wall_ahead and (opening_left or opening_right):
            signature['scene_type'] = 'L_turn'
            signature['intersection_type'] = 'L'
        elif wall_ahead and not opening_left and not opening_right:
            signature['scene_type'] = 'dead_end'
        else:
            signature['scene_type'] = 'corridor'
            
        return signature
    
    def is_wall_at(self, pos: Tuple[int, int]) -> bool:
        """Check if position is a wall or out of bounds"""
        x, y = pos
        if x < 0 or x >= len(self.maze[0]) or y < 0 or y >= len(self.maze):
            return True
        return self.maze[y][x] == 1
    
    def detect_movement(self, current_frame: np.ndarray) -> bool:
        """Detect if robot is moving using frame difference"""
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return False
        
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        diff = cv2.absdiff(self.prev_frame, current_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Count changed pixels
        movement_pixels = cv2.countNonZero(thresh)
        is_moving = movement_pixels > self.movement_threshold
        
        if not is_moving:
            self.stationary_frames += 1
        else:
            self.stationary_frames = 0
            
        self.prev_frame = current_gray
        return is_moving
    
    def detect_blur(self, image: np.ndarray) -> bool:
        """Detects if an image is blurry using the variance of the Laplacian.
        
        Args:
            image: The input image (should be grayscale).

        Returns:
            True if the image is blurry, False otherwise.
        """
        # Compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        return laplacian_var < self.blur_threshold

    def detect_scene_with_precision(self) -> Optional[Dict]:
        """Precisely detect current scene using line detection instead of 3D corners"""
        if not self.cap:
            return {
                'status': 'error',
                'confidence': 0.0,
                'message': 'Camera not available'
            }
        
        frame = self.cap.capture_array()
        if frame is None:
            return {
                'status': 'error',
                'confidence': 0.0,
                'message': 'Failed to read camera frame'
            }
        
        with self.frame_lock:
            self.latest_frame = frame.copy()

        # Check if robot is moving
        is_moving = self.detect_movement(frame)
        
        # If stationary for too long, don't update position
        if self.stationary_frames > self.max_stationary_frames:
            return {
                'status': 'stationary',
                'confidence': 0.0,
                'is_moving': False,
                'message': 'Robot stationary - no position update'
            }
        
        # Convert to grayscale for line detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check for blur
        if self.detect_blur(gray):
            return {
                'status': 'error',
                'confidence': 0.0,
                'is_moving': is_moving,
                'message': 'Frame is too blurry for localization'
            }

        # Detect lines instead of corners
        scene = self.classify_scene_by_lines(gray)
        scene['is_moving'] = is_moving
        scene['status'] = 'ok'
        
        return scene
    
    def classify_scene_by_lines(self, gray_frame: np.ndarray) -> Dict:
        """Classify scene based on line detection for 2D line following"""
        height, width = gray_frame.shape
        
        # Focus on the bottom half of the image where lines are most visible
        roi_start = height // 2
        roi = gray_frame[roi_start:, :]
        
        # Threshold to detect black lines on white background
        # THRESH_BINARY_INV makes black lines appear as white in binary image
        _, binary = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up the binary image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours (lines)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                'scene_type': 'corridor',
                'wall_ahead': False,
                'opening_left': False,
                'opening_right': False,
                'intersection_type': 'none',
                'confidence': 0.3
            }
        
        # Analyze the largest contour (main line)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate line properties
        roi_height, roi_width = roi.shape
        line_width_ratio = w / roi_width
        line_height_ratio = h / roi_height
        
        # Determine scene type based on line shape
        scene = {
            'scene_type': 'corridor',
            'corner_ahead_left': False,
            'corner_ahead_right': False,
            'wall_ahead': False,
            'opening_left': False,
            'opening_right': False,
            'intersection_type': 'none',
            'confidence': 0.8
        }
        
        # If line is very wide, it might be an intersection
        if line_width_ratio > 0.7:
            scene['scene_type'] = 'intersection'
            scene['intersection_type'] = '4way'
            scene['opening_left'] = True
            scene['opening_right'] = True
        # If line is moderately wide, might be T-junction
        elif line_width_ratio > 0.5:
            scene['scene_type'] = 'T_junction'
            scene['intersection_type'] = 'T'
            # Determine which side has opening based on line position
            line_center_x = x + w // 2
            if line_center_x < roi_width // 3:
                scene['opening_right'] = True
            elif line_center_x > 2 * roi_width // 3:
                scene['opening_left'] = True
            else:
                scene['opening_left'] = True  # Default assumption
        # If line is very short in height, might be approaching dead end
        elif line_height_ratio < 0.3:
            scene['scene_type'] = 'dead_end'
            scene['wall_ahead'] = True
        
        return scene
    
    def localize_with_confidence(self) -> Dict:
        """Simplified localization for line following - focus on movement detection"""
        observed_scene = self.detect_scene_with_precision()
        if observed_scene['status'] in ['error']:
            self.last_status.update({
                'message': observed_scene['message'],
                'status': 'tracking_lost'
            })
            return self.last_status

        # If robot is stationary, don't change position but still report status
        if observed_scene['status'] == 'stationary':
            self.last_status.update({
                'message': 'Robot stationary - maintaining position',
                'status': 'stationary',
                'scene_type': observed_scene.get('scene_type', 'unknown'),
                'is_moving': False
            })
            return self.last_status

        # For line following, we'll be less strict about exact scene matching
        # Just use the detected scene type for navigation decisions
        current_key = (self.current_pos[0], self.current_pos[1], self.current_direction)
        
        # Simple confidence based on scene detection quality
        if observed_scene.get('scene_type') == 'corridor':
            match_confidence = 0.8
        elif observed_scene.get('scene_type') in ['T_junction', 'intersection']:
            match_confidence = 0.9
        else:
            match_confidence = 0.6

        # Always maintain reasonable confidence for line following
        if match_confidence >= 0.5:  # Lower threshold for line following
            self.position_confidence = match_confidence
        else:
            self.position_confidence *= 0.95  # Slow decay

        self.last_status.update({
            'status': 'tracking' if match_confidence >= 0.5 else 'tracking_lost',
            'position': self.current_pos,
            'direction': self.current_direction,
            'confidence': self.position_confidence,
            'scene_type': observed_scene.get('scene_type', 'unknown'),
            'is_moving': observed_scene.get('is_moving', False),
            'message': f"Line following mode - confidence: {match_confidence:.2f}"
        })
        
        return self.last_status
    
    def compare_scenes_precisely(self, observed: Dict, expected: Dict) -> float:
        """Compare observed scene with expected signature for a more detailed confidence score."""
        scene_score = 0
        if observed.get('scene_type') == expected.get('scene_type'):
            scene_score += 1
        if observed.get('intersection_type') == expected.get('intersection_type'):
            scene_score += 1

        feature_score = 0
        features = ['wall_ahead', 'opening_left', 'opening_right']
        for feature in features:
            if observed.get(feature) == expected.get(feature):
                feature_score += 1
        
        # Normalize scores (max scene_score=2, max feature_score=3)
        normalized_scene = scene_score / 2.0
        normalized_feature = feature_score / 3.0

        # Combine with weighting
        confidence = (normalized_scene * 0.6) + (normalized_feature * 0.4)
        return confidence

    def _get_neighboring_keys(self, current_key: Tuple[int, int, str]) -> List[Tuple[int, int, str]]:
        """Get all valid signature keys for the current cell and its adjacent cells."""
        x, y, current_direction = current_key
        neighbor_keys = []

        # Add all directions for the current cell
        for d in ['N', 'S', 'E', 'W']:
            neighbor_keys.append((x, y, d))

        # Add all directions for all valid adjacent cells
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            # Check if the new position is within bounds and is a path
            if 0 <= ny < len(self.maze) and 0 <= nx < len(self.maze[0]) and self.maze[ny][nx] == 0:
                for d in ['N', 'S', 'E', 'W']:
                    neighbor_keys.append((nx, ny, d))
        
        return list(set(neighbor_keys)) # Use set to remove duplicates
    
    def update_direction_after_turn(self, turn_direction: str):
        """Manually update the robot's direction after a pivot turn."""
        if turn_direction == 'left':
            turn_map = {'N': 'W', 'W': 'S', 'S': 'E', 'E': 'N'}
            self.current_direction = turn_map[self.current_direction]
        elif turn_direction == 'right':
            turn_map = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}
            self.current_direction = turn_map[self.current_direction]
        print(f"Direction updated after turn. New direction: {self.current_direction}")

    def start_localization(self):
        """Start the localization thread."""
        if not self.running:
            self.running = True
            self.localization_thread = threading.Thread(target=self._localization_loop, daemon=True)
            self.localization_thread.start()
            print("Visual localization thread started.")
    
    def stop_localization(self):
        """Stop the localization thread."""
        self.running = False
        if self.localization_thread:
            self.localization_thread.join()
        if self.cap:
            self.cap.stop()
            self.cap.close()
    
    def _localization_loop(self):
        """The main loop for the localization thread."""
        while self.running:
            if self.initialization_frames > 0:
                self.initialization_frames -= 1
                time.sleep(1 / self.camera_fps)
                continue

            localization_result = self.localize_with_confidence()
            
            if localization_result:
                # New Stability Logic:
                # Check if the new result matches the current candidate.
                if (self.position_candidate and 
                    localization_result['position'] == self.position_candidate and
                    localization_result['direction'] == self.last_status['direction']): # Ensure direction is also stable
                    
                    self.candidate_stability_counter += 1
                else:
                    # New candidate found, reset counter.
                    self.position_candidate = localization_result['position']
                    self.candidate_stability_counter = 1

                # If the candidate has been stable for long enough, update the official position.
                if self.candidate_stability_counter >= self.STABILITY_THRESHOLD:
                    if self.current_pos != self.position_candidate:
                        print(f"Position lock updated: {self.current_pos} -> {self.position_candidate}")
                        self.current_pos = self.position_candidate
                    
                    # Also update direction if it has changed and is stable
                    if self.current_direction != localization_result['direction']:
                         self.current_direction = localization_result['direction']

                    # Update confidence and other status info from the latest good result
                    self.position_confidence = localization_result['confidence']

                # Always update the last_status for real-time feedback, but don't change the official position
                # until it's stable.
                self.last_status.update({
                    'status': localization_result.get('status', 'unknown'),
                    'confidence': localization_result.get('confidence', 0),
                    'position': self.current_pos, # Report the STABLE position
                    'direction': self.current_direction, # Report the STABLE direction
                    'scene_type': localization_result.get('scene_type', 'unknown'),
                    'is_moving': localization_result.get('is_moving', False),
                    'message': localization_result.get('message', '')
                })

            time.sleep(1 / self.camera_fps)
            
    def get_status(self) -> Dict:
        """Get the latest status of the localizer."""
        with self.frame_lock:
            status = self.last_status.copy()
            status.update({
                'current_position': self.current_pos,
                'current_direction': self.current_direction,
                'position_confidence': self.position_confidence,
                'stationary_frames': self.stationary_frames,
                'is_stationary': self.stationary_frames > self.max_stationary_frames
            })
            return status
    
    def get_camera_frame(self):
        """Get current camera frame for video feed"""
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        
        # If no frame is available from the localization thread, capture one directly
        if self.cap:
            try:
                frame = self.cap.capture_array()
                return frame
            except Exception as e:
                print(f"Failed to capture camera frame: {e}")
        
        return None

    def get_current_cell(self):
        """Get current cell from the localizer"""
        return self.current_pos

    def get_pose(self):
        """Get pose (x, y, heading) from the localizer"""
        x, y = self.current_pos
        
        direction_to_heading = {'N': 90, 'E': 0, 'S': 270, 'W': 180} # Degrees
        heading_deg = direction_to_heading.get(self.current_direction, 0)
        heading_rad = math.radians(heading_deg)

        return (x, y, heading_rad)

    def nudge_position_forward(self):
        """Manually advance the current position by one cell based on the current direction."""
        
        dx, dy = 0, 0
        if self.current_direction == 'N': dy = -1
        elif self.current_direction == 'S': dy = 1
        elif self.current_direction == 'E': dx = 1
        elif self.current_direction == 'W': dx = -1

        new_pos = (self.current_pos[0] + dx, self.current_pos[1] + dy)
        
        # Boundary check to ensure we don't go off the map
        if (0 <= new_pos[1] < len(self.maze) and 
            0 <= new_pos[0] < len(self.maze[0]) and
            self.maze[new_pos[1]][new_pos[0]] == 0):
            
            self.current_pos = new_pos
            
            # Reset the stability filter to prevent it from getting stuck on the old cell
            self.position_candidate = self.current_pos
            self.candidate_stability_counter = 0

            print(f"Dead reckoning nudge: Position updated to {self.current_pos}")
            self.last_status['message'] = "Position nudged by dead reckoning"
        else:
            print(f"WARN: Dead reckoning nudge to {new_pos} ignored; out of bounds or wall.")

    def set_search_mode(self, mode: str):
        """Set the localization search mode to 'normal' or 'corridor'."""
        if mode in ['normal', 'corridor']:
            if self.search_mode != mode:
                print(f"Localizer search mode set to: {mode}")
                self.search_mode = mode
        else:
            print(f"WARN: Invalid search mode '{mode}' requested.")
            
    def _get_key_ahead(self, current_key: Tuple[int, int, str]) -> Optional[Tuple[int, int, str]]:
        """Calculates the grid key for the cell directly in front of the current key."""
        x, y, direction = current_key
        dx, dy = 0, 0
        if direction == 'N': dy = -1
        elif direction == 'S': dy = 1
        elif direction == 'E': dx = 1
        elif direction == 'W': dx = -1
        
        next_pos = (x + dx, y + dy)

        # Check bounds
        if (0 <= next_pos[1] < len(self.maze) and 
            0 <= next_pos[0] < len(self.maze[0])):
            return (next_pos[0], next_pos[1], direction)
        return None