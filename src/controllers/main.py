#!/usr/bin/env python3

import time
import logging
import cv2
import numpy as np
import math
import socket
import threading
import base64
from typing import List, Tuple, Optional
from flask import Flask, jsonify, render_template, request, Response

# Import our clean modules
from object_detection import ObjectDetector
from pathfinder import Pathfinder
from box import BoxHandler
from pid import PIDController
from camera_line_follower import CameraLineFollower, CameraLineFollowingMixin
from encoder_position_tracker import EncoderPositionTracker
from pi_motor_controller import PiMotorController
from audio_feedback import AudioFeedback

# ================================
# FEATURE CONFIGURATION
# ================================
# Enable/disable features for easy testing and debugging
FEATURES = {
    'OBJECT_DETECTION_ENABLED': False,
    'OBSTACLE_AVOIDANCE_ENABLED': False,
    'VISION_SYSTEM_ENABLED': False, # Disabled to use encoders
    'CAMERA_LINE_FOLLOWING_ENABLED': True, 
    'POSITION_CORRECTION_ENABLED': False, # N/A for encoder-based tracking
    'PERFORMANCE_LOGGING_ENABLED': False,    # Disabled to reduce log spam
    'DEBUG_VISUALIZATION_ENABLED': True,
    'SMOOTH_CORNERING_ENABLED': True,
    'ADAPTIVE_SPEED_ENABLED': True,
}

# ================================
# ROBOT CONFIGURATION
# ================================
CELL_SIZE_M = 0.11
BASE_SPEED = 40
TURN_SPEED = 60
CORNER_SPEED = 70

# Hardware-specific trims to account for motor differences.
# Values are multipliers (1.0 = no change, 0.9 = 10% slower).
MOTOR_TRIMS = {
    'fl': 1.0,
    'fr': 1.0,
    'bl': 1.0,
    'br': 0.75  # Back-right motor is a bit faster
}

# Maze and Mission Configuration
MAZE_GRID = [
    [0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1], # Row 0
    [0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1], # Row 1
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Row 2
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 3
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 4
    [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0], # Row 5
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 6
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Row 7
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 8
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0], # Row 9
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 10
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 11
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Row 12
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0], # Row 13
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0]  # Row 14
]
START_CELL = (0, 12) # Start position (col, row)
END_CELL = (20, 8)   # End position (col, row)

# START_DIRECTION must be a cardinal direction: 'N', 'S', 'E', or 'W'.
# This tells the robot its initial orientation on the map grid.
#  - 'N': Faces towards smaller row numbers (Up on the map)
#  - 'S': Faces towards larger row numbers (Down on the map)
#  - 'E': Faces towards larger column numbers (Right on the map)
#  - 'W': Faces towards smaller column numbers (Left on the map)
START_DIRECTION = 'E'

# Time in seconds it takes for the robot to cross one 12cm cell at BASE_SPEED.
# This is now primarily used as a timeout.
CELL_CROSSING_TIME_S = 1.5  # Increased to act as a safeguard timeout

# Corner turning configuration
CORNER_TURN_MODES = {
    'SMOOTH': 'smooth',
    'SIDEWAYS': 'sideways',
    'PIVOT': 'pivot',
    'FRONT_TURN': 'front_turn'
}
CORNER_DETECTION_THRESHOLD = 0.35
CORNER_TURN_DURATION = 30
SHARP_CORNER_THRESHOLD = 0.6

# Camera configuration
WEBCAM_INDEX = 1
CAMERA_WIDTH, CAMERA_HEIGHT = 320, 240
CAMERA_FPS = 30

class RobotController(CameraLineFollowingMixin):
    """Main robot controller integrating visual localization and direct motor control."""

    def __init__(self):
        self.running = True
        
        # Initialize motor controller directly with error handling
        try:
            self.motor_controller = PiMotorController(trims=MOTOR_TRIMS)
        except Exception as e:
            print(f"Failed to initialize motor controller: {e}")
            print("Robot will run in simulation mode without motor control.")
            self.motor_controller = None

        # Initialize audio feedback system
        self.audio_feedback = AudioFeedback()

        self.object_detector = None
        self.frame = None
        self.processed_frame = None
        self.frame_lock = threading.Lock()

        self.detections = {}
        self.last_intersection_time = 0
        self.motor_speeds = {'fl': 0, 'fr': 0, 'bl': 0, 'br': 0}
        self.camera_line_result = {}

        # Position tracking using Encoders
        self.position_tracker = EncoderPositionTracker(
            maze=MAZE_GRID,
            start_pos=START_CELL,
            motor_controller=self.motor_controller,
            start_direction=START_DIRECTION,
            cell_size_m=CELL_SIZE_M,
            debug=FEATURES['DEBUG_VISUALIZATION_ENABLED']
        )

        # Pathfinder setup
        self.pathfinder = Pathfinder(grid=MAZE_GRID, cell_size_m=CELL_SIZE_M)

        self.path = []
        self.current_target_index = 0
        self.turn_to_execute = None # Stores the next turn ('left' or 'right')
        self.turn_start_time = 0
        self.last_turn_complete_time = 0 # Cooldown timer for turns
        self.is_straight_corridor = False
        self.last_cell_update_time = 0

        self.line_pid = PIDController(kp=0.5, ki=0.01, kd=0.1, output_limits=(-100, 100))
        self.state = "idle"

        # Vision setup is now minimal, only for line following
        if FEATURES['CAMERA_LINE_FOLLOWING_ENABLED']:
            self.init_camera_line_following(
                camera_index=WEBCAM_INDEX,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fps=CAMERA_FPS,
                debug=FEATURES['DEBUG_VISUALIZATION_ENABLED']
            )

    def _set_motor_speeds(self, fl, fr, bl, br):
        """A wrapper to set motor speeds and store them for the UI."""
        self.motor_speeds = {'fl': int(fl), 'fr': int(fr), 'bl': int(bl), 'br': int(br)}
        if self.motor_controller:
            self.motor_controller.send_motor_speeds(int(fl), int(fr), int(bl), int(br))

    def _setup_vision(self):
        """Initialize vision systems if enabled."""
        # This is now handled by the CameraLineFollowingMixin's init method
        pass

    def run(self):
        """Main control loop."""
        # Initialize the camera for line following if enabled.
        if FEATURES['CAMERA_LINE_FOLLOWING_ENABLED']:
            if not self.camera_line_follower.initialize_camera():
                print("CRITICAL: Camera for line following failed to initialize. Aborting mission.")
                self.running = False
                return

        # The PiMotorController is initialized in the constructor. No connect() needed.
        while self.running:
            self._run_state_machine()
            time.sleep(0.01)
        
        self.stop()
        
    def _start_mission(self):
        """Start the defined mission."""
        if self.state != "idle":
            print("Cannot start mission, robot is not idle.")
            return

        print("Starting mission...")
        self.audio_feedback.speak("Starting mission")
        
        # Reset timers and state for the new mission
        self.last_cell_update_time = time.time()

        # Start position tracker only when mission begins
        if not self.position_tracker.running:
            self.position_tracker.start()
            
        # Pose is now managed by the encoder localizer
        self.state = "planning"

    def _process_vision(self):
        """Process the latest camera frame for events."""
        # This function is now OBSOLETE as vision processing is tied
        # directly to the line following logic.
        pass

    def _plan_path_to_target(self):
        """Plan the path to the target cell."""
        current_cell = self.position_tracker.get_current_cell()
        print(f"Planning path from {current_cell} to {END_CELL}...")
        
        path_nodes = self.pathfinder.find_path(current_cell, END_CELL)
        
        if path_nodes:
            self.path = path_nodes
            self.current_target_index = 0
            self.state = "path_following"
            
            # Check if the path is a straight line to determine the following strategy.
            self.is_straight_corridor = self._is_path_straight(self.path)
            
            path_message = f"Path planned with {len(self.path)} waypoints."
            print(path_message)
            self.audio_feedback.speak(path_message)
            
        else:
            print(f"Failed to plan path from {current_cell} to {END_CELL}")
            self.audio_feedback.speak("Path planning failed")
            self.state = "error"

    def _follow_path(self):
        """Follow the planned path using encoder-based cell transitions and camera-based line centering."""
        if not self.path or self.current_target_index >= len(self.path):
            self.state = "mission_complete"
            self._stop_motors()
            return

        current_cell = self.position_tracker.get_current_cell()

        # The primary goal is to reach the final destination.
        if current_cell == self.path[-1]:
            print(f"Reached final destination: {current_cell}")
            self.state = "mission_complete"
            self._stop_motors()
            return

        # --- Waypoint Arrival Logic (Encoder-based) ---
        # Check if the robot has arrived at the next waypoint in its path.
        if current_cell == self.path[self.current_target_index]:
            print(f"Reached waypoint {self.current_target_index}: {current_cell}")
            self.last_cell_update_time = time.time()
            self.current_target_index += 1
            if self.current_target_index >= len(self.path):
                self.state = "mission_complete"
                self._stop_motors()
                return

        # --- Turn Determination Logic ---
        # Based on the *next* waypoint, determine if a turn is needed *now*.
        current_dir = self.position_tracker.current_direction
        next_waypoint = self.path[self.current_target_index]
        required_turn = self._get_required_turn(current_cell, current_dir, next_waypoint)
        
        print(f"Path Follow: Cell={current_cell}, Target={next_waypoint}, Required Turn={required_turn}")

        # If a turn is required, change state to 'turning'.
        # This is triggered by the cell change from the encoder tracker.
        TURN_COOLDOWN_S = 2.0
        can_turn = (time.time() - self.last_turn_complete_time) > TURN_COOLDOWN_S

        if required_turn in ['left', 'right'] and can_turn:
            turn_message = f"Waypoint reached. Turning {required_turn} towards {next_waypoint}."
            print(turn_message)
            self.audio_feedback.speak(turn_message)
            
            # Stop briefly before turning to make it more precise
            self._stop_motors()
            time.sleep(0.2) 

            self.turn_to_execute = required_turn
            self.state = 'turning'
            self.turn_start_time = time.time()
            return # Exit to let the 'turning' state take over
        
        # --- Default Action: Line Following ---
        # If no turn is needed, or if on cooldown, continue following the line forward.
        frame = self.camera_line_follower.get_camera_frame()
        if frame is None:
            # If camera fails, maybe we should stop or rely on blind forward movement
            print("WARN: No camera frame for line following. Moving forward blindly for a moment.")
            if self.motor_controller:
                self.motor_controller.send_motor_speeds(BASE_SPEED, BASE_SPEED, BASE_SPEED, BASE_SPEED)
            return

        self.camera_line_result = self.camera_line_follower.detect_line(frame)
        fl, fr, bl, br = self.camera_line_follower.get_motor_speeds(self.camera_line_result, base_speed=BASE_SPEED)
        self._set_motor_speeds(fl, fr, bl, br)

    def _execute_arcing_turn(self):
        """
        Executes a smart, arcing turn, continuing until the line is re-centered and straight.
        """
        # Define turn parameters
        TURN_TIMEOUT_S = 2.5          # Max duration for a turn to prevent getting stuck
        MIN_TURN_DURATION_S = 0.5     # Increased slightly to ensure it commits to the turn
        LINE_CENTERED_THRESHOLD = 0.15 # How close to center the line must be
        STRAIGHT_LINE_ASPECT_RATIO_THRESHOLD = 0.5 # A re-acquired line should be tall and thin

        # Get the latest camera frame and analyze it for the line
        frame = self.camera_line_follower.get_camera_frame()
        if frame is None:
            self._stop_motors()
            return

        self.camera_line_result = self.camera_line_follower.detect_line(frame)
        line_offset = self.camera_line_result.get('line_offset', 1.0) # Default to a large offset
        aspect_ratio = self.camera_line_result.get('aspect_ratio', 1.0) # Default to a wide shape

        is_line_centered = abs(line_offset) < LINE_CENTERED_THRESHOLD
        is_line_straight = aspect_ratio < STRAIGHT_LINE_ASPECT_RATIO_THRESHOLD
        
        time_in_turn = time.time() - self.turn_start_time
        
        # --- DEBUG LOG ---
        print(f"Turning: Offset={line_offset:.2f}, AR={aspect_ratio:.2f}, Centered={is_line_centered}, Straight={is_line_straight}, Time={time_in_turn:.2f}s")
        
        # Check for completion conditions
        # The robot must have been turning for a minimum duration AND see a centered, straight line.
        turn_complete = is_line_centered and is_line_straight and (time_in_turn > MIN_TURN_DURATION_S)
        turn_timed_out = time_in_turn > TURN_TIMEOUT_S

        if turn_complete or turn_timed_out:
            if turn_complete:
                print("Turn complete: Line re-acquired and is straight.")
                self.audio_feedback.speak("Turn complete.")
            else: # Timed out
                print(f"WARN: Turn timed out after {TURN_TIMEOUT_S}s. Proceeding anyway.")
                self.audio_feedback.speak("Turn timed out.")
                
            self._stop_motors()
            self.position_tracker.update_direction_after_turn(self.turn_to_execute)
            self.turn_to_execute = None
            self.state = 'path_following'
            self.last_turn_complete_time = time.time() # Start cooldown
            time.sleep(0.25) # Short pause to stabilize
        else:
            # Condition not met, continue turning
            self._perform_arcing_turn(self.turn_to_execute)

    def _run_state_machine(self):
        """Run the robot's state machine."""
        # Update position from encoders at the start of each cycle.
        self.position_tracker.update_position()

        if self.state == "idle":
            self._stop_motors()
            self.position_tracker.set_moving(False)
        elif self.state == "planning":
            self._plan_path_to_target()
        elif self.state == "path_following":
            self.position_tracker.set_moving(True)
            self._follow_path()
        elif self.state == "turning":
            self.position_tracker.set_moving(True)
            self._execute_arcing_turn()
        elif self.state == "mission_complete":
            self.audio_feedback.speak("Mission complete.")
            self._stop_motors()
            self.position_tracker.set_moving(False)
        elif self.state == "error":
            self.audio_feedback.speak("Error state reached.")
            self._stop_motors()
            self.position_tracker.set_moving(False)
            self.running = False
            
    def _follow_line_with_sensor(self):
        """Follow the line using hardware sensors via ESP32."""
        # This method is no longer used, but kept for potential fallback.
        print("ESP32 line sensor is disabled in features.")
        self._stop_motors()
        return

    def _execute_corner_turn(self, corner_direction: str, line_error: int):
        """Execute different types of corner turns based on the robot's location."""
        
        # Determine if the robot is in a special zone (pickup/dropoff area)
        # We assume these are at the beginning and end of the planned path.
        is_special_zone = False
        if self.path:
            # Check if near the start of the path (e.g., first waypoint)
            if self.current_target_index <= 0:
                is_special_zone = True
            
            # Check if near the end of the path (e.g., last two waypoints)
            if self.current_target_index >= len(self.path) - 2:
                is_special_zone = True

        if is_special_zone:
            # Use a precise turn (e.g., pivot) in special zones for maneuvering
            print("Executing PIVOT turn for precision maneuver.")
            return self._pivot_corner_turn(corner_direction, line_error)
        else:
            # Use a smooth turn for general navigation
            return self._smooth_corner_turn(corner_direction, line_error)

    def _smooth_corner_turn(self, corner_direction: str, line_error: int):
        """Normal wheel-like smooth cornering (current behavior)."""
        turn_correction = self.line_pid.update(line_error)
        
        # Gradual turn like normal wheels
        left_speed = int(CORNER_SPEED - turn_correction)
        right_speed = int(CORNER_SPEED + turn_correction)
        
        self._set_motor_speeds(left_speed, right_speed, left_speed, right_speed)
        return True

    def _sideways_corner_turn(self, corner_direction: str, line_error: int):
        """Strafe sideways through corners using omni-wheel capabilities."""
        # Determine strafe direction based on corner direction
        if corner_direction == "left":
            # Strafe left while maintaining forward motion
            fl_speed = CORNER_SPEED // 2      # Front left: reduced forward
            fr_speed = CORNER_SPEED           # Front right: full forward  
            bl_speed = CORNER_SPEED           # Back left: full forward
            br_speed = CORNER_SPEED // 2      # Back right: reduced forward
        else:  # right turn
            # Strafe right while maintaining forward motion
            fl_speed = CORNER_SPEED           # Front left: full forward
            fr_speed = CORNER_SPEED // 2      # Front right: reduced forward
            bl_speed = CORNER_SPEED // 2      # Back left: reduced forward
            br_speed = CORNER_SPEED           # Back right: full forward
        
        self._set_motor_speeds(fl_speed, fr_speed, bl_speed, br_speed)
        return True

    def _pivot_corner_turn(self, corner_direction: str, line_error: int):
        """Turn in place like a tank - pure rotation."""
        if corner_direction == "left":
            # Rotate counter-clockwise (left turn)
            fl_speed = -TURN_SPEED    # Front left: reverse
            fr_speed = TURN_SPEED     # Front right: forward
            bl_speed = -TURN_SPEED    # Back left: reverse  
            br_speed = TURN_SPEED     # Back right: forward
        else:  # right turn
            # Rotate clockwise (right turn)
            fl_speed = TURN_SPEED     # Front left: forward
            fr_speed = -TURN_SPEED    # Front right: reverse
            bl_speed = TURN_SPEED     # Back left: forward
            br_speed = -TURN_SPEED    # Back right: reverse
        
        self._set_motor_speeds(fl_speed, fr_speed, bl_speed, br_speed)
        return True

    def _front_turn_corner(self, corner_direction: str, line_error: int):
        """Turn primarily using front wheels like a front-wheel-drive car."""
        base_speed = CORNER_SPEED
        
        if corner_direction == "left":
            # Front wheels turn left, back wheels follow
            fl_speed = base_speed // 2        # Front left: slower
            fr_speed = base_speed             # Front right: normal
            bl_speed = base_speed * 3 // 4    # Back left: moderate
            br_speed = base_speed             # Back right: normal
        else:  # right turn
            #
            fl_speed = base_speed             # Front left: normal
            fr_speed = base_speed // 2        # Front right: slower
            bl_speed = base_speed             # Back left: normal
            br_speed = base_speed * 3 // 4    # Back right: moderate
        
        self._set_motor_speeds(fl_speed, fr_speed, bl_speed, br_speed)
        return True

    def _recover_line(self):
        """Basic line recovery: stop for now."""
        self._stop_motors()

    def _stop_motors(self):
        """Stop all motors."""
        self._set_motor_speeds(0, 0, 0, 0)
    
    def stop(self):
        """Stop the robot and clean up resources."""
        print("Stopping robot...")
        self.running = False
        self._stop_motors()
        
        # Stop the position tracker
        self.position_tracker.stop()

        # Stop camera line follower resources
        if FEATURES['CAMERA_LINE_FOLLOWING_ENABLED'] and hasattr(self, 'camera_line_follower'):
            self.camera_line_follower.release_camera()

        if self.motor_controller:
            self.motor_controller.stop()

    def _get_required_turn(self, current_pos: Tuple[int, int], current_dir: str, target_pos: Tuple[int, int]) -> str:
        """Determines the turn required to move from the current pose to the target position."""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]

        if dx == 0 and dy == 0:
            return 'stop'

        # Determine target direction
        target_dir = ''
        if dx > 0: target_dir = 'E'
        elif dx < 0: target_dir = 'W'
        elif dy > 0: target_dir = 'S'
        elif dy < 0: target_dir = 'N'

        if current_dir == target_dir:
            return 'forward'

        # Defines clockwise turns
        turn_logic = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}
        if turn_logic[current_dir] == target_dir:
            return 'right'
        else:
            # If it's not forward and not right, it must be left (or reverse)
            # This simple logic assumes no reverse moves are in the path plan.
            return 'left'

    def _perform_arcing_turn(self, direction: str):
        """
        Commands motor speeds for a smooth, car-like arcing turn.
        This uses omni-wheel kinematics for combined forward and rotational motion.
        """
        vx = BASE_SPEED * 0.8  # Forward speed during the turn
        omega = TURN_SPEED     # Rotational speed for the turn

        # Based on the kinematic model in camera_line_follower, a left
        # turn requires a negative omega to make the right wheels spin faster.
        if direction == 'left':
            turn_omega = -omega
        else: # 'right'
            turn_omega = omega

        # vy (strafing) is zero for a pure arcing turn.
        vy = 0
        
        # Kinematic model from camera_line_follower.py
        fl = int(vx - vy + turn_omega)
        fr = int(vx + vy - turn_omega)
        bl = int(vx + vy + turn_omega)
        br = int(vx - vy - turn_omega)

        self._set_motor_speeds(fl, fr, bl, br)

    def _is_path_straight(self, path: List[Tuple[int, int]]) -> bool:
        """Checks if a path is a straight line (either horizontal or vertical)."""
        if len(path) < 2:
            return True
        
        is_straight_x = all(p[0] == path[0][0] for p in path)
        is_straight_y = all(p[1] == path[0][1] for p in path)
        
        return is_straight_x or is_straight_y


def print_feature_status():
    """Print current feature configuration for debugging."""
    print("=" * 50)
    print("ROBOT FEATURE CONFIGURATION")
    print("=" * 50)
    for feature, enabled in FEATURES.items():
        status = "ENABLED" if enabled else "DISABLED"
        print(f"{feature:<30} : {status}")
    print("=" * 50)
    print()


def main():
    """Main entry point for the robot controller."""
    print_feature_status()
    
    # --- Direction Validation ---
    # Ensure start direction is a valid cardinal direction.
    valid_directions = ['N', 'S', 'E', 'W']
    system_start_direction = START_DIRECTION.upper()
    
    if system_start_direction not in valid_directions:
        print(f"ERROR: Invalid START_DIRECTION '{START_DIRECTION}'. Must be one of {valid_directions}.")
        print("Defaulting to 'N'.")
        system_start_direction = 'N'
    
    robot = RobotController()
    
    app = Flask(__name__, template_folder='../../templates', static_folder='../../static')
    
    @app.route('/')
    def index():
        return render_template('navigation.html')

    @app.route('/api/robot_data')
    def robot_data():
        """Provide robot data to the web UI."""
        x, y, heading_rad = robot.position_tracker.get_pose()
        heading_deg = math.degrees(heading_rad)
        
        # Get localizer status from the active tracker
        position_tracker_status = robot.position_tracker.get_status()

        encoder_counts = robot.motor_controller.get_encoder_counts() if robot.motor_controller else {}

        data = {
            'state': robot.state,
            'x': x,
            'y': y,
            'heading': heading_deg,
            'position_tracker': {
                'status': position_tracker_status.get('status', 'N/A'),
                'confidence': position_tracker_status.get('confidence', 0),
                'position': position_tracker_status.get('current_position', (0,0)),
                'direction': position_tracker_status.get('current_direction', 'N/A'),
                'message': position_tracker_status.get('message', ''),
            },
            'line_follower': {
                'line_offset': robot.camera_line_result.get('line_offset', 0),
                'is_at_intersection': robot.camera_line_result.get('is_at_intersection', False),
            },
            'motors': robot.motor_speeds,
            'encoders': encoder_counts,
            'path': robot.path,
            'current_target_index': robot.current_target_index,
        }
        return jsonify(data)

    @app.route('/video_feed')
    def video_feed():
        """Video streaming route. Returns a placeholder if camera is disabled."""
        if not FEATURES['CAMERA_LINE_FOLLOWING_ENABLED']:
            return Response(status=204) # No content

        def generate_frames():
            while robot.running:
                frame = None
                # Get frame from the line follower camera
                if hasattr(robot, 'camera_line_follower'):
                    frame = robot.camera_line_follower.get_camera_frame()

                if frame is None:
                    # If no frame, send a placeholder or just wait
                    time.sleep(0.1)
                    continue
                
                # Add camera line following debug overlay if available
                # In the new structure, detect_line is called inside _follow_path,
                # but the result isn't stored on the robot object.
                # For debugging, we might want to call it here or store the result.
                # For now, we just show the raw frame.
                
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.05) # Limit frame rate

        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @app.route('/start_mission')
    def start_mission():
        """Manually start the mission and pathfinding."""
        robot._start_mission()
        robot._run_state_machine()  # Run one cycle to trigger planning
        return jsonify({
            'status': 'Mission started',
            'robot_state': robot.state,
            'path_length': len(robot.path)
        })

    @app.route('/stop_robot')
    def stop_robot():
        """Stop the robot's movement and all processes."""
        robot.stop()
        return jsonify({'status': 'Robot stopped'})

    @app.route('/grid_feed')
    def grid_feed():
        """Streams the grid map visualization."""
        def generate():
            while robot.running:
                robot_cell = robot.position_tracker.get_current_cell()
                path = robot.path
                
                grid_img = generate_grid_image(
                    pathfinder=robot.pathfinder,
                    robot_cell=robot_cell,
                    path=path,
                    start_cell=START_CELL,
                    end_cell=END_CELL
                )
                
                _, buffer = cv2.imencode('.jpg', grid_img)
                frame = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
                time.sleep(0.1)
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def generate_grid_image(pathfinder, robot_cell, path, start_cell, end_cell):
        """Generates the grid image for the web UI."""
        grid = np.array(pathfinder.get_grid())
        cell_size = 20
        height, width = grid.shape
        grid_img = np.zeros((height * cell_size, width * cell_size, 3), dtype=np.uint8)

        for r in range(height):
            for c in range(width):
                color = (255, 255, 255) if grid[r, c] == 1 else (0, 0, 0)
                cv2.rectangle(grid_img, (c * cell_size, r * cell_size), 
                              ((c + 1) * cell_size, (r + 1) * cell_size), color, -1)
        
        if path:
            for i in range(len(path) - 1):
                p1_x = path[i][0] * cell_size + cell_size // 2
                p1_y = path[i][1] * cell_size + cell_size // 2
                p2_x = path[i+1][0] * cell_size + cell_size // 2
                p2_y = path[i+1][1] * cell_size + cell_size // 2
                # Use blue for path to distinguish from red end cell
                cv2.line(grid_img, (p1_x, p1_y), (p2_x, p2_y), (255, 0, 0), 2)

        start_color = (0, 255, 0)
        start_x, start_y = start_cell[0], start_cell[1]
        cv2.rectangle(grid_img, (start_x * cell_size, start_y * cell_size),
                      ((start_x + 1) * cell_size, (start_y + 1) * cell_size), start_color, -1)
        
        end_color = (0, 0, 255)
        end_x, end_y = end_cell[0], end_cell[1]
        cv2.rectangle(grid_img, (end_x * cell_size, end_y * cell_size),
                      ((end_x + 1) * cell_size, (end_y + 1) * cell_size), end_color, -1)

        if robot_cell:
            robot_x, robot_y = robot_cell[0], robot_cell[1]
            cv2.circle(grid_img, 
                       (robot_x * cell_size + cell_size // 2, robot_y * cell_size + cell_size // 2), 
                       cell_size // 3, (203, 102, 255), -1) # Use pink for robot
        
        return grid_img
    
    # Start the robot controller in a separate thread
    robot_thread = threading.Thread(target=robot.run, daemon=True)
    robot_thread.start()
    
    # Run Flask app
    print("Starting Flask web server...")
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting program.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logging.exception("Error details:")
