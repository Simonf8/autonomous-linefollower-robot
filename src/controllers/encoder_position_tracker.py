import time
import math
from typing import Tuple, Dict

class EncoderPositionTracker:
    """Tracks the robot's position (grid cell) and orientation using wheel encoders."""

    def __init__(self, maze: list, start_pos: Tuple[int, int], motor_controller,
                 start_direction: str = 'N', cell_size_m: float = 0.11,
                 wheel_circumference_m: float = 0.204, ticks_per_revolution: int = 960):

        self.maze = maze
        self.motor_controller = motor_controller
        self.cell_size_m = cell_size_m
        self.start_direction = start_direction

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
        
        # Reset encoders at the start
        if self.motor_controller:
            self.motor_controller.reset_encoders()
            self.last_encoder_counts = self.motor_controller.get_encoder_counts()
        else:
            self.last_encoder_counts = {'fl': 0, 'fr': 0, 'bl': 0, 'br': 0}


    def start(self):
        """Starts the position tracking."""
        print("Encoder Position Tracker started.")
        self.running = True
        self.last_update_time = time.time()
        # Reset state at the start of a mission
        self.distance_since_last_cell = 0.0
        if self.motor_controller:
            self.motor_controller.reset_encoders()
            self.last_encoder_counts = self.motor_controller.get_encoder_counts()

    def stop(self):
        """Stops the position tracking."""
        self.running = False
        print("Encoder Position Tracker stopped.")

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
        
        # A simple model for straight forward motion: average the two front wheel ticks.
        # This is a simplification and might need adjustment based on robot kinematics.
        # We only consider positive ticks to count forward movement.
        forward_ticks = [delta_ticks['fl'], delta_ticks['fr']]
        avg_delta_ticks = sum(t for t in forward_ticks if t > 0) / 2 if any(t > 0 for t in forward_ticks) else 0

        distance_moved = avg_delta_ticks * self.METERS_PER_TICK
        self.distance_since_last_cell += distance_moved

        # Check if we have crossed into a new cell
        if self.distance_since_last_cell >= self.cell_size_m:
            print(f"Cell crossing triggered! Dist: {self.distance_since_last_cell:.3f}m")
            self._advance_cell()
            self.distance_since_last_cell = 0 # Reset distance

    def _advance_cell(self):
        """Move one cell forward in the current direction."""
        dx, dy = 0, 0
        if self.current_direction == 'N': dy = -1
        elif self.current_direction == 'S': dy = 1
        elif self.current_direction == 'E': dx = 1
        elif self.current_direction == 'W': dx = -1

        new_pos = (self.current_pos[0] + dx, self.current_pos[1] + dy)

        if (0 <= new_pos[1] < len(self.maze) and
            0 <= new_pos[0] < len(self.maze[0]) and
            self.maze[new_pos[1]][new_pos[0]] == 0):
            print(f"Encoder Tracker: Moved from {self.current_pos} -> {new_pos}")
            self.current_pos = new_pos
        else:
            print(f"Encoder Tracker: WARN: Advance to {new_pos} blocked by wall or boundary.")

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
            'status': 'OPERATIONAL' if self.running else 'IDLE',
            'confidence': 0.9,
            'current_position': self.current_pos,
            'current_direction': self.current_direction,
            'scene_type': 'encoder',
            'message': f"Dist in cell: {self.distance_since_last_cell:.2f}m"
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