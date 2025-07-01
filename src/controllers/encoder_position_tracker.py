import time
import math
from typing import Tuple, Dict, Optional

class EncoderPositionTracker:
    """Tracks the robot's position (grid cell) and orientation using wheel encoders with camera assistance."""

    def __init__(self, maze: list, start_pos: Tuple[int, int], motor_controller,
                 start_direction: str = 'N', cell_size_m: float = 0.11,
                 wheel_circumference_m: float = 0.204, ticks_per_revolution: int = 960,
                 debug: bool = False):

        self.maze = maze
        self.motor_controller = motor_controller
        self.cell_size_m = cell_size_m
        self.start_direction = start_direction
        self.debug = debug

        # Robot physical parameters
        self.TICKS_PER_REVOLUTION = ticks_per_revolution
        self.WHEEL_CIRCUMFERENCE_M = wheel_circumference_m
        self.METERS_PER_TICK = self.WHEEL_CIRCUMFERENCE_M / self.TICKS_PER_REVOLUTION

        # State variables
        self.current_pos: Tuple[int, int] = start_pos
        self.current_direction: str = start_direction
        self.distance_since_last_cell: float = 0.0
        self.last_encoder_counts: Dict[str, int] = {'fl': 0, 'fr': 0, 'bl': 0, 'br': 0}
        self.is_moving = False

        self.running = False
        self.last_update_time = time.time()
        
        # Camera-assisted tracking
        self.camera_line_result: Optional[Dict] = None
        self.last_intersection_detection = False
        self.intersection_detection_cooldown = 0.0
        self.INTERSECTION_COOLDOWN_S = 2.0  # Minimum time between intersection detections
        self.expected_intersections = self._calculate_expected_intersections()
        self.intersection_count = 0
        
        # Position correction parameters
        self.position_confidence = 1.0
        self.encoder_drift_threshold = 0.3  # Maximum allowed drift before correction
        
        # Reset encoders at the start
        if self.motor_controller:
            self.motor_controller.reset_encoders()
            self.last_encoder_counts = self.motor_controller.get_encoder_counts()
        else:
            self.last_encoder_counts = {'fl': 0, 'fr': 0, 'bl': 0, 'br': 0}

    def _calculate_expected_intersections(self) -> int:
        """Calculate expected number of intersections on the path."""
        # This is a simplified calculation - in a real implementation,
        # you'd calculate this based on the planned path
        return 0

    def _handle_intersection_detection(self):
        """
        Handles the logic when the camera detects an intersection.
        This simply resets the distance tracker, using the intersection as a hard sync point.
        """
        self.intersection_count += 1
        
        if not self.is_moving:
            return

        if self.debug:
            print(f"CAMERA CORRECTION: Intersection detected at {self.current_pos}. Resetting distance tracker.")
        
        # Reset the distance traveled since the last cell.
        # The camera-detected intersection is now our new "zero" point for this cell.
        self.distance_since_last_cell = 0.0
        self.position_confidence = min(1.0, self.position_confidence + 0.1)

    def set_camera_line_result(self, camera_result: Dict):
        """Update the tracker with the latest camera line detection result."""
        self.camera_line_result = camera_result
        
        # Check for intersection detection
        if camera_result and camera_result.get('is_at_intersection', False):
            current_time = time.time()
            
            # Only process if we're not in cooldown period
            if current_time - self.intersection_detection_cooldown > self.INTERSECTION_COOLDOWN_S:
                if not self.last_intersection_detection:  # Rising edge detection
                    self._handle_intersection_detection()
                    self.intersection_detection_cooldown = current_time
                
                self.last_intersection_detection = True
            else:
                # Still seeing an intersection, but in cooldown
                pass
        else:
            self.last_intersection_detection = False

    def start(self):
        """Starts the position tracking."""
        print("Hybrid Encoder+Camera Position Tracker started.")
        self.running = True
        self.last_update_time = time.time()
        # Reset state at the start of a mission
        self.distance_since_last_cell = 0.0
        self.intersection_count = 0
        self.position_confidence = 1.0
        if self.motor_controller:
            self.motor_controller.reset_encoders()
            self.last_encoder_counts = self.motor_controller.get_encoder_counts()

    def stop(self):
        """Stops the position tracking."""
        self.running = False
        print("Hybrid Position Tracker stopped.")

    def update_position(self):
        """
        Updates the robot's traveled distance based on encoder ticks.
        If the distance exceeds the cell size, it updates the robot's grid position.
        This method should be called regularly in the main control loop.
        """
        if not self.running or not self.motor_controller:
            return

        current_counts = self.motor_controller.get_encoder_counts()
        
        # Calculate delta ticks for each wheel. This assumes forward motion.
        # For pure forward motion, all deltas should be positive.
        delta_ticks = {w: current_counts[w] - self.last_encoder_counts[w] for w in current_counts}
        self.last_encoder_counts = current_counts
        
        # For 2-wheel differential drive: average the left and right wheel ticks.
        # This is a simplification for straight forward motion.
        # We only consider positive ticks to count forward movement.
        forward_ticks = [delta_ticks['left'], delta_ticks['right']]
        avg_delta_ticks = sum(t for t in forward_ticks if t > 0) / 2 if any(t > 0 for t in forward_ticks) else 0

        distance_moved = avg_delta_ticks * self.METERS_PER_TICK
        self.distance_since_last_cell += distance_moved

        # Check if we have crossed into a new cell
        if self.distance_since_last_cell >= self.cell_size_m:
            if self.debug:
                print(f"ENCODER: Cell crossing triggered! Dist: {self.distance_since_last_cell:.3f}m")
            self._advance_cell()
            self.distance_since_last_cell = 0.0 # Reset distance

    def _advance_cell(self):
        """Move one cell forward in the current direction based on encoder distance."""
        # This is now a fallback method when no camera correction is available
        dx, dy = 0, 0
        if self.current_direction == 'N': dy = -1
        elif self.current_direction == 'S': dy = 1
        elif self.current_direction == 'E': dx = 1
        elif self.current_direction == 'W': dx = -1

        new_pos = (self.current_pos[0] + dx, self.current_pos[1] + dy)

        if (0 <= new_pos[1] < len(self.maze) and
            0 <= new_pos[0] < len(self.maze[0]) and
            self.maze[new_pos[1]][new_pos[0]] == 0):
            if self.debug:
                print(f"Encoder Tracker: Moved from {self.current_pos} -> {new_pos}")
            self.current_pos = new_pos
            # Since this is encoder-based, slightly decrease confidence over time
            self.position_confidence = max(0.5, self.position_confidence - 0.02)
        else:
            if self.debug:
                print(f"Encoder Tracker: WARN: Advance to {new_pos} blocked by wall or boundary.")
            # We hit a wall according to encoders, majorly reduce confidence
            self.position_confidence = max(0.0, self.position_confidence - 0.5)

    def get_current_cell(self) -> Tuple[int, int]:
        return self.current_pos

    def update_direction_after_turn(self, turn: str):
        """Updates the robot's orientation after a turn, e.g., 'left' or 'right'."""
        turn_map_right = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}
        turn_map_left = {'N': 'W', 'W': 'S', 'S': 'E', 'E': 'N'}

        if turn == 'left':
            self.current_direction = turn_map_left[self.current_direction]
        elif turn == 'right':
            self.current_direction = turn_map_right[self.current_direction]
        
        if self.debug:
            print(f"Encoder Tracker: Direction updated to {self.current_direction} after {turn} turn.")

    def get_pose(self) -> Tuple[float, float, float]:
        """Returns the robot's current pose (x, y, heading) in meters and radians."""
        x = self.current_pos[0] * self.cell_size_m
        y = self.current_pos[1] * self.cell_size_m
        dir_to_rad = {'N': math.pi / 2, 'E': 0, 'S': -math.pi / 2, 'W': math.pi}
        heading_rad = dir_to_rad.get(self.current_direction, 0)
        return x, y, heading_rad

    def get_status(self) -> Dict:
        """Provides status information for the UI."""
        return {
            'status': 'HYBRID_TRACKING' if self.running else 'IDLE',
            'confidence': self.position_confidence,
            'current_position': self.current_pos,
            'current_direction': self.current_direction,
            'scene_type': 'hybrid_encoder_camera',
            'message': f"Dist: {self.distance_since_last_cell:.2f}m, Intersections: {self.intersection_count}, Conf: {self.position_confidence:.2f}"
        }

    def set_moving(self, moving: bool):
        """Inform the tracker if the robot is supposed to be moving."""
        self.is_moving = moving
        if not moving:
            # When stopping, it's good practice to get a final encoder reading
            # to prevent drift calculations on the next start.
            if self.motor_controller:
                self.last_encoder_counts = self.motor_controller.get_encoder_counts()

    # --- Compatibility methods to avoid breaking main.py ---
    def initialize_camera(self): return True
    def stop_localization(self): self.stop()
    def get_camera_frame(self): return None
    def set_search_mode(self, mode): pass
    def force_movement_detection(self): self.set_moving(True)
    def nudge_position_forward(self): self._advance_cell() 