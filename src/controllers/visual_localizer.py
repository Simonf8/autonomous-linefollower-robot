import cv2
import numpy as np
import threading
import time
from typing import Tuple, Optional, Dict, List
import math

class PreciseMazeLocalizer:
    def __init__(self, maze, start_pos: Tuple[int, int], camera_width: int, camera_height: int, camera_fps: int, start_direction: str = 'N'):
        # Your exact maze
        self.maze = maze
        
        self.current_pos = start_pos
        self.current_direction = start_direction
        
        # Initialize camera
        self.cap = None # Will be initialized in initialize_camera
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.latest_frame = None
        
        # Movement detection
        self.prev_frame = None
        self.movement_threshold = 50
        self.stationary_frames = 0
        self.max_stationary_frames = 10
        self.blur_threshold = 100.0 # New: tunable threshold for blur detection
        
        # Corner distance detection
        self.close_corner_threshold = 0.7
        self.far_corner_threshold = 0.3
        
        # Create precise signatures for your maze
        self.corner_signatures = self.create_precise_signatures()
        
        # Confidence tracking
        self.position_confidence = 1.0
        self.min_confidence = 0.85
        
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
        
    def initialize_camera(self, camera_index=0):
        """Initializes the camera with a given index."""
        if self.cap is None:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                print(f"Warning: Could not open camera at index {camera_index}")
                self.cap = None
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
            
            print(f"Successfully opened camera at index {camera_index} with {self.camera_width}x{self.camera_height} resolution at {self.camera_fps} FPS")
            return True
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
        """Precisely detect current scene with movement and distance awareness"""
        if not self.cap or not self.cap.isOpened():
            return {
                'status': 'error',
                'confidence': 0.0,
                'message': 'Camera not available'
            }
        
        ret, frame = self.cap.read()
        if not ret:
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
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # New: Blur detection
        if self.detect_blur(gray):
            return {
                'status': 'error',
                'confidence': 0.0,
                'is_moving': is_moving,
                'message': 'Frame is too blurry for localization'
            }

        # Use FAST for faster corner detection
        fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        keypoints = fast.detect(gray, None)
        
        # Get corner coordinates from keypoints
        corner_points = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]

        # Draw keypoints for debugging if needed
        # frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0,255,0))
        # with self.frame_lock:
        #     self.latest_frame = frame_with_keypoints
        
        # Analyze corner distances
        distance_analysis = self.analyze_corner_distance(corner_points, frame.shape[0])
        
        # Determine scene type based on corner patterns
        scene = self.classify_scene_precisely(corner_points, frame.shape, distance_analysis)
        scene['is_moving'] = is_moving
        scene['distance_analysis'] = distance_analysis
        
        return scene
    
    def analyze_corner_distance(self, corner_points: List[Tuple[int, int]], frame_height: int) -> Dict:
        """Analyze if corners are close (immediate) or far (ahead)"""
        if not corner_points:
            return {'close_corners': 0, 'far_corners': 0, 'corner_distance': 'none'}
        
        close_threshold = frame_height * self.close_corner_threshold
        far_threshold = frame_height * self.far_corner_threshold
        
        close_corners = sum(1 for _, y in corner_points if y > close_threshold)
        far_corners = sum(1 for _, y in corner_points if y < far_threshold)
        
        if close_corners > far_corners:
            distance = 'close'
        elif far_corners > close_corners:
            distance = 'far'
        else:
            distance = 'medium'
            
        return {
            'close_corners': close_corners,
            'far_corners': far_corners,
            'corner_distance': distance
        }
    
    def classify_scene_precisely(self, corner_points: List[Tuple[int, int]], 
                                frame_shape: Tuple[int, int], 
                                distance_analysis: Dict) -> Dict:
        """Precisely classify the scene based on corner distribution"""
        height, width = frame_shape[:2]
        
        scene = {
            'scene_type': 'unknown',
            'corner_ahead_left': False,
            'corner_ahead_right': False,
            'wall_ahead': False,
            'opening_left': False,
            'opening_right': False,
            'intersection_type': 'none',
            'confidence': 0.0
        }
        
        if not corner_points:
            scene['scene_type'] = 'corridor'
            scene['confidence'] = 0.8
            return scene
        
        # Divide frame into precise regions
        left_third = width // 3
        right_third = 2 * width // 3
        top_half = height // 2
        
        # Count corners in each region
        corners_left = [(x, y) for x, y in corner_points if x < left_third]
        corners_right = [(x, y) for x, y in corner_points if x > right_third]
        corners_center = [(x, y) for x, y in corner_points if left_third <= x <= right_third]
        corners_top = [(x, y) for x, y in corner_points if y < top_half]
        
        # Adjust confidence based on corner distance
        if distance_analysis['corner_distance'] == 'far':
            scene['confidence'] = 0.4
        else:
            scene['confidence'] = 0.9
        
        # Classification logic
        if len(corners_center) > 4 and len(corners_top) > 3:
            scene['scene_type'] = 'dead_end'
            scene['wall_ahead'] = True
        elif len(corner_points) < 3:
            scene['scene_type'] = 'intersection'
            scene['intersection_type'] = '4way'
        elif len(corners_left) > 2 and len(corners_center) > 2:
            scene['scene_type'] = 'L_turn'
            scene['corner_ahead_left'] = True
            scene['opening_right'] = True
        elif len(corners_right) > 2 and len(corners_center) > 2:
            scene['scene_type'] = 'L_turn'
            scene['corner_ahead_right'] = True
            scene['opening_left'] = True
        elif len(corners_center) > 2 and (len(corners_left) > 1 or len(corners_right) > 1):
            scene['scene_type'] = 'T_junction'
            scene['intersection_type'] = 'T'
            scene['opening_left'] = len(corners_left) < 2
            scene['opening_right'] = len(corners_right) < 2
        else:
            scene['scene_type'] = 'corridor'
            
        return scene
    
    def localize_with_confidence(self) -> Dict:
        """Localize position with confidence tracking"""
        observed_scene = self.detect_scene_with_precision()
        if not observed_scene:
            return {
                'status': 'error',
                'confidence': 0.0,
                'message': 'Failed to analyze scene'
            }
        
        # Handle special cases
        if observed_scene.get('status') == 'stationary':
            return observed_scene
        
        if observed_scene.get('status') == 'error':
            return observed_scene
        
        # If corners are too far ahead, reduce confidence
        if observed_scene.get('distance_analysis', {}).get('corner_distance') == 'far':
            return {
                'status': 'corners_too_far',
                'confidence': 0.2,
                'is_moving': observed_scene.get('is_moving', False),
                'scene_type': observed_scene.get('scene_type', 'unknown'),
                'message': 'Corners detected too far ahead - position uncertain'
            }
        
        # Match against known signatures
        best_matches = []
        for (x, y, direction), expected in self.corner_signatures.items():
            match_score = self.compare_scenes_precisely(observed_scene, expected)
            if match_score > 0.8:
                best_matches.append(((x, y, direction), match_score))
        
        if best_matches:
            best_matches.sort(key=lambda x: x[1], reverse=True)
            best_pos = best_matches[0][0]
            confidence = best_matches[0][1]
            
            # Update position only if confidence is high enough
            if confidence > self.min_confidence:
                self.current_pos = (best_pos[0], best_pos[1])
                self.current_direction = best_pos[2]
                self.position_confidence = confidence
                
                return {
                    'status': 'localized',
                    'position': self.current_pos,
                    'direction': self.current_direction,
                    'confidence': confidence,
                    'scene_type': observed_scene['scene_type'],
                    'is_moving': observed_scene.get('is_moving', False),
                    'message': f'Localized at ({self.current_pos[0]}, {self.current_pos[1]})'
                }
        
        return {
            'status': 'uncertain',
            'confidence': 0.0,
            'scene_type': observed_scene.get('scene_type', 'unknown'),
            'is_moving': observed_scene.get('is_moving', False),
            'message': 'Cannot confidently determine position'
        }
    
    def compare_scenes_precisely(self, observed: Dict, expected: Dict) -> float:
        """Precisely compare observed scene with expected signature"""
        score = 0.0
        total_weight = 0.0
        
        # Scene type match (high weight)
        if observed.get('scene_type') == expected.get('scene_type'):
            score += 3.0
        total_weight += 3.0
        
        # Specific feature matches
        features = ['corner_ahead_left', 'corner_ahead_right', 'wall_ahead', 
                   'opening_left', 'opening_right']
        
        for feature in features:
            if observed.get(feature) == expected.get(feature):
                score += 1.0
            total_weight += 1.0
        
        # Intersection type match
        if observed.get('intersection_type') == expected.get('intersection_type'):
            score += 2.0
        total_weight += 2.0
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def start_localization(self):
        """Start the localization thread"""
        if not self.running:
            self.running = True
            self.localization_thread = threading.Thread(target=self._localization_loop)
            self.localization_thread.daemon = True
            self.localization_thread.start()
    
    def stop_localization(self):
        """Stop the localization thread"""
        self.running = False
        if self.localization_thread:
            self.localization_thread.join()
        if self.cap:
            self.cap.release()
    
    def _localization_loop(self):
        """Main loop for the localization thread."""
        while self.running:
            if self.initialization_frames > 0:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        with self.frame_lock:
                            self.latest_frame = frame.copy()
                self.initialization_frames -= 1
                time.sleep(1 / self.camera_fps)
                continue

            result = self.localize_with_confidence()

            # Update status based on result
            if result and result.get('status') == 'success':
                with self.frame_lock:
                    self.current_pos = result['position']
                    self.current_direction = result['direction']
                    self.position_confidence = result['confidence']

                self.last_status.update({
                    'status': 'tracking',
                    'position': self.current_pos,
                    'direction': self.current_direction,
                    'confidence': self.position_confidence,
                    'scene_type': result.get('scene', {}).get('scene_type', 'unknown'),
                    'message': 'Position updated'
                })
            elif result:
                # Handle other statuses like 'stationary', 'error', 'blurry'
                self.last_status.update({
                    'status': result.get('status', 'unknown'),
                    'position': self.current_pos,
                    'direction': self.current_direction,
                    'confidence': result.get('confidence', self.position_confidence),
                    'message': result.get('message', 'No message')
                })

            time.sleep(1 / self.camera_fps) # Control loop rate
    
    def get_status(self) -> Dict:
        """Get the current status of the localizer."""
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