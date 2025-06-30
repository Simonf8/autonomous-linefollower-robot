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
from camera_obstacle_avoidance import CameraObstacleAvoidance

# ================================
# FEATURE CONFIGURATION
# ================================
# Enable/disable features for easy testing and debugging
FEATURES = {
    'OBJECT_DETECTION_ENABLED': False,
    'OBSTACLE_AVOIDANCE_ENABLED': True,  # Enabled for line blocking detection
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
CELL_SIZE_M = 0.085
BASE_SPEED = 40
TURN_SPEED = 20     # Reduced from 25 to 20 for even slower turning
CORNER_SPEED = 20   # Reduced from 25 to 20 for even slower cornering

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
START_CELL = (3, 12) # Start position (col, row)
END_CELL = (0, 0)   # End position (col, row)

# START_DIRECTION must be a cardinal direction: 'N', 'S', 'E', or 'W'.
# This tells the robot its initial orientation on the map grid.
#  - 'N': Faces towards smaller row numbers (Up on the map)
#  - 'S': Faces towards larger row numbers (Down on the map)
#  - 'E': Faces towards larger column numbers (Right on the map)
#  - 'W': Faces towards smaller column numbers (Left on the map)
START_DIRECTION = 'W'

# Time in seconds it takes for the robot to cross one 12cm cell at BASE_SPEED.

CELL_CROSSING_TIME_S = 1.5  # Increased to act as a safeguard timeout

# Corner turning configuration
CORNER_TURN_MODES = {
    'SMOOTH': 'smooth',
    'SIDEWAYS': 'sideways',
    'PIVOT': 'pivot',
    'FRONT_TURN': 'front_turn'
}
CORNER_DETECTION_THRESHOLD = 0.35
CORNER_TURN_DURATION = 60    # Increased from 30 to 60 for longer turn timeout
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

        # Initialize obstacle avoidance for line blocking detection
        if FEATURES['OBSTACLE_AVOIDANCE_ENABLED']:
            self.obstacle_avoidance = CameraObstacleAvoidance(debug=FEATURES['DEBUG_VISUALIZATION_ENABLED'])
        else:
            self.obstacle_avoidance = None

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
        self.debug = FEATURES['DEBUG_VISUALIZATION_ENABLED']

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
            self.audio_feedback.speak("Final destination reached!")
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
                print("All waypoints completed - mission complete!")
                self.audio_feedback.speak("All waypoints completed!")
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
        
       
        frame = self.camera_line_follower.get_camera_frame()
        if frame is None:
            # If camera fails, maybe we should stop or rely on blind forward movement
            
            if self.motor_controller:
                self.motor_controller.send_motor_speeds(BASE_SPEED, BASE_SPEED, BASE_SPEED, BASE_SPEED)
            return

        # PERFORMANCE: Skip every other frame for obstacle detection to improve speed
        if not hasattr(self, '_frame_skip_counter'):
            self._frame_skip_counter = 0
        self._frame_skip_counter += 1
        
        self.camera_line_result = self.camera_line_follower.detect_line(frame)
        
        # Feed camera results to position tracker for hybrid tracking
        self.position_tracker.set_camera_line_result(self.camera_line_result)
        
        # Check for obstacles blocking the line (skip every other frame for performance)
        if FEATURES['OBSTACLE_AVOIDANCE_ENABLED'] and self.obstacle_avoidance and self._frame_skip_counter % 2 == 0:
            line_center_x = self.camera_line_result.get('line_center_x', frame.shape[1] // 2)
            obstacle_result = self.obstacle_avoidance.detect_line_blocking_obstacle(frame, line_center_x)
            
            if obstacle_result['is_blocking']:
                action = obstacle_result['recommended_action']
                
                self.audio_feedback.speak("Obstacle blocking line, turning around")
                
                # MARK OBSTACLE IN GRID for future A* planning
                current_cell = self.position_tracker.get_current_cell()
                # Mark the cell ahead as an obstacle (where robot was heading)
                current_dir = self.position_tracker.current_direction
                if current_dir == 'N':
                    obstacle_cell = (current_cell[0], current_cell[1] - 1)
                elif current_dir == 'S':
                    obstacle_cell = (current_cell[0], current_cell[1] + 1)
                elif current_dir == 'E':
                    obstacle_cell = (current_cell[0] + 1, current_cell[1])
                elif current_dir == 'W':
                    obstacle_cell = (current_cell[0] - 1, current_cell[1])
                else:
                    obstacle_cell = current_cell
                
                # Update pathfinder grid with obstacle
                self.pathfinder.update_obstacle(obstacle_cell[0], obstacle_cell[1], True)
                
                
                # IMMEDIATE 180-degree turn - no gradual turning or other actions
                # This ensures the robot immediately starts turning when obstacle is detected
                self._stop_motors()  # Brief stop before turning
                time.sleep(0.1)      # Very brief pause
                
                self.state = 'obstacle_turn_around'
                self.turn_start_time = time.time()
                return
        
        # Normal line following if no obstacles
        # Adjust speed based on proximity to destination
        current_base_speed = BASE_SPEED
        if self.path and len(self.path) > 0:
            remaining_waypoints = len(self.path) - self.current_target_index
            final_destination = self.path[-1]
            distance_to_destination = abs(current_cell[0] - final_destination[0]) + abs(current_cell[1] - final_destination[1])
            
            # Slow down when approaching destination
            if remaining_waypoints <= 1 or distance_to_destination <= 1:
                current_base_speed = int(BASE_SPEED * 0.6)  # 60% speed for final approach
            elif remaining_waypoints <= 2 or distance_to_destination <= 2:
                current_base_speed = int(BASE_SPEED * 0.8)  # 80% speed when close
        
        fl, fr, bl, br = self.camera_line_follower.get_motor_speeds(self.camera_line_result, base_speed=current_base_speed)
        self._set_motor_speeds(fl, fr, bl, br)

    def _execute_arcing_turn(self):
        """
        Executes a turn - uses pivot turn for precision near destination, arcing turn otherwise.
        """
        # Check if we're near the destination for precision turning
        current_cell = self.position_tracker.get_current_cell()
        is_near_destination = False
        
        if self.path and len(self.path) > 0:
            # Check if we're at the last few waypoints
            remaining_waypoints = len(self.path) - self.current_target_index
            is_near_destination = remaining_waypoints <= 2
            
            # Also check if we're close to the final destination
            final_destination = self.path[-1]
            distance_to_destination = abs(current_cell[0] - final_destination[0]) + abs(current_cell[1] - final_destination[1])
            is_near_destination = is_near_destination or distance_to_destination <= 2

        if is_near_destination:
            self._execute_precision_pivot_turn()
        else:
            
            self._execute_smooth_arcing_turn()

    def _execute_precision_pivot_turn(self):
        """
        Execute a precise pivot turn in place - stops, turns until line found and centered.
        """
        TURN_TIMEOUT_S = 3.5          # Moderate timeout for precision turns
        MIN_TURN_DURATION_S = 0.8     # Ensure we turn away from current line
        LINE_CENTERED_THRESHOLD = 0.2 # More lenient for precision turns
        
        time_in_turn = time.time() - self.turn_start_time
        
        # Get the latest camera frame and analyze it for the line
        frame = self.camera_line_follower.get_camera_frame()
        if frame is None:
            self._stop_motors()
            return

        self.camera_line_result = self.camera_line_follower.detect_line(frame)
        
        # Feed camera results to position tracker for hybrid tracking
        self.position_tracker.set_camera_line_result(self.camera_line_result)
        
        line_detected = self.camera_line_result.get('line_detected', False)
        line_offset = self.camera_line_result.get('line_offset', 1.0)
        
        is_line_centered = abs(line_offset) < LINE_CENTERED_THRESHOLD
        
        # --- DEBUG LOG ---
        
        # Check for completion conditions
        # Must turn for minimum duration AND find a reasonably centered line
        turn_complete = line_detected and is_line_centered and (time_in_turn > MIN_TURN_DURATION_S)
        turn_timed_out = time_in_turn > TURN_TIMEOUT_S

        if turn_complete or turn_timed_out:
            if turn_complete:
        
                self.audio_feedback.speak("Turn complete.")
            else:
                
                self.audio_feedback.speak("Turn timed out.")
                
            self._stop_motors()
            self.position_tracker.update_direction_after_turn(self.turn_to_execute)
            self.turn_to_execute = None
            self.state = 'path_following'
            self.last_turn_complete_time = time.time()
            time.sleep(0.5)  # Longer pause for precision
        else:
            # Continue pivot turning (rotate in place)
            self._perform_pivot_turn(self.turn_to_execute)

    def _execute_smooth_arcing_turn(self):
        """
        Execute the original arcing turn for general navigation.
        """
        # Define turn parameters
        TURN_TIMEOUT_S = 3.5          # Balanced timeout for reliable turns
        MIN_TURN_DURATION_S = 0.6     # Moderate turn commitment time
        LINE_CENTERED_THRESHOLD = 0.15 # How close to center the line must be
        STRAIGHT_LINE_ASPECT_RATIO_THRESHOLD = 0.5 # A re-acquired line should be tall and thin

        # Get the latest camera frame and analyze it for the line
        frame = self.camera_line_follower.get_camera_frame()
        if frame is None:
            self._stop_motors()
            return

        self.camera_line_result = self.camera_line_follower.detect_line(frame)
        
        # Feed camera results to position tracker for hybrid tracking
        self.position_tracker.set_camera_line_result(self.camera_line_result)
        
        line_offset = self.camera_line_result.get('line_offset', 1.0) # Default to a large offset
        aspect_ratio = self.camera_line_result.get('aspect_ratio', 1.0) # Default to a wide shape

        is_line_centered = abs(line_offset) < LINE_CENTERED_THRESHOLD
        is_line_straight = aspect_ratio < STRAIGHT_LINE_ASPECT_RATIO_THRESHOLD
        
        time_in_turn = time.time() - self.turn_start_time
        
        # --- DEBUG LOG ---
        
        
        # Check for completion conditions
        # The robot must have been turning for a minimum duration AND see a centered, straight line.
        turn_complete = is_line_centered and is_line_straight and (time_in_turn > MIN_TURN_DURATION_S)
        turn_timed_out = time_in_turn > TURN_TIMEOUT_S

        if turn_complete or turn_timed_out:
            if turn_complete:
               
                self.audio_feedback.speak("Turn complete.")
            else: # Timed out

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
        elif self.state == "obstacle_turn_around":
            self.position_tracker.set_moving(True)
            self._execute_180_turn()
        elif self.state == "mission_complete":
            self.audio_feedback.speak("Mission complete.")
            self._stop_motors()
            self.position_tracker.set_moving(False)
        elif self.state == "error":
            self.audio_feedback.speak("Error state reached.")
            self._stop_motors()
            self.position_tracker.set_moving(False)
            self.running = False
            


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
        vx = BASE_SPEED * 0.6  # Reduced from 0.8 to 0.6 for slower forward speed during turns
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

    def _perform_pivot_turn(self, direction: str):
        """
        Commands motor speeds for a precise pivot turn in place.
        This rotates the robot without forward motion for maximum precision.
        """
        # Use the same pivot logic as the existing _pivot_corner_turn method
        if direction == "left":
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

    def _execute_180_turn(self):
        """
        Execute a 180-degree turn in place when an obstacle blocks the line.
        Keep turning until the line is found, then replan path using A*.
        """
        MAX_TURN_DURATION = 12.0   # Increased from 8.0 to 12.0 for longer timeout
        MIN_TURN_DURATION = 2.0    # Increased from 1.2 to 2.0 for more gradual turning
        
        time_in_turn = time.time() - self.turn_start_time
        
        # Get current camera frame to check for line
        frame = self.camera_line_follower.get_camera_frame()
        line_found = False
        
        if frame is not None and time_in_turn > MIN_TURN_DURATION:
            # Check if we can see the line again
            line_result = self.camera_line_follower.detect_line(frame)
            line_found = line_result.get('line_detected', False)
            
            if line_found:
                line_offset = line_result.get('line_offset', 1.0)
                # Only consider line found if it's reasonably centered
                if abs(line_offset) < 0.3:  # Line is reasonably centered
                    line_found = True
                else:
                    line_found = False
        
        if time_in_turn < MAX_TURN_DURATION and not line_found:
            # Continue turning - pivot turn (rotate in place)
            fl_speed = TURN_SPEED     # Front left: forward
            fr_speed = -TURN_SPEED    # Front right: reverse
            bl_speed = TURN_SPEED     # Back left: forward
            br_speed = -TURN_SPEED    # Back right: reverse
            
            self._set_motor_speeds(fl_speed, fr_speed, bl_speed, br_speed)
            
            if self.debug and int(time_in_turn) != int(time_in_turn - 0.1):  # Print every second
                print(f"Turning until line found: {time_in_turn:.1f}s / {MAX_TURN_DURATION}s, Line found: {line_found}")
        else:
            # Turn complete - either line found or timed out
            if line_found:
                print("180-degree turn complete. Line reacquired!")
                self.audio_feedback.speak("Line found, replanning path")
            else:
                print("Turn timed out. Proceeding with path following.")
                self.audio_feedback.speak("Turn complete, resuming")
            
            self._stop_motors()
            
            # Update direction in position tracker (180 degrees)
            current_dir = self.position_tracker.current_direction
            opposite_directions = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
            new_direction = opposite_directions.get(current_dir, current_dir)
            self.position_tracker.current_direction = new_direction
            
            # IMPORTANT: Trigger path replanning using A* after obstacle avoidance
            print("Replanning path after obstacle avoidance...")
            self.state = 'planning'  # This will trigger A* replanning
            time.sleep(0.5)  # Brief pause to stabilize

    def _is_path_straight(self, path: List[Tuple[int, int]]) -> bool:
        """Checks if a path is a straight line (either horizontal or vertical)."""
        if len(path) < 2:
            return True
        
        is_straight_x = all(p[0] == path[0][0] for p in path)
        is_straight_y = all(p[1] == path[0][1] for p in path)
        
        return is_straight_x or is_straight_y

def main():
    """Main entry point for the robot controller."""
    
    # --- Direction Validation ---
    # Ensure start direction is a valid cardinal direction.
    valid_directions = ['N', 'S', 'E', 'W']
    system_start_direction = START_DIRECTION.upper()
    
    if system_start_direction not in valid_directions:
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

        # Get audio feedback status
        audio_status = robot.audio_feedback.get_status()

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
                'intersection_count': robot.position_tracker.intersection_count,
                'arm_filtering': robot.camera_line_follower.get_arm_filtering_status() if hasattr(robot, 'camera_line_follower') else {'enabled': False},
                'adaptive_threshold': robot.camera_line_follower.get_adaptive_threshold_status() if hasattr(robot, 'camera_line_follower') else {'adaptive_enabled': False},
            },
            'obstacle_avoidance': {
                'enabled': FEATURES['OBSTACLE_AVOIDANCE_ENABLED'],
                'status': 'active' if robot.obstacle_avoidance else 'disabled',
            },
            'motors': robot.motor_speeds,
            'encoders': encoder_counts,
            'audio_feedback': {
                'enabled': audio_status.get('enabled', False),
                'provider': audio_status.get('preferred_provider', 'none'),
                'available_providers': audio_status.get('available_providers', []),
                'queue_size': audio_status.get('queue_size', 0),
            },
            'path': robot.path,
            'current_target_index': robot.current_target_index,
        }
        return jsonify(data)

    @app.route('/video_feed')
    def video_feed():
        """Video streaming route. Returns raw camera feed."""
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
                
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.05) # Limit frame rate

        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/debug_feed')
    def debug_feed():
        """Debug video streaming route showing processed frames with overlays."""
        if not FEATURES['CAMERA_LINE_FOLLOWING_ENABLED']:
            return Response(status=204) # No content

        def generate_debug_frames():
            while robot.running:
                frame = None
                debug_frame = None
                
                # Get frame from the line follower camera
                if hasattr(robot, 'camera_line_follower'):
                    frame = robot.camera_line_follower.get_camera_frame()

                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Process the frame to get debug visualization
                if hasattr(robot, 'camera_line_follower'):
                    # Call detect_line to get the processed frame with debug overlay
                    result = robot.camera_line_follower.detect_line(frame)
                    debug_frame = result.get('processed_frame', frame)
                
                if debug_frame is None:
                    debug_frame = frame
                
                _, buffer = cv2.imencode('.jpg', debug_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.05) # Limit frame rate

        return Response(generate_debug_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/binary_feed')
    def binary_feed():
        """Binary/grayscale processed feed showing just the line detection mask."""
        if not FEATURES['CAMERA_LINE_FOLLOWING_ENABLED']:
            return Response(status=204) # No content

        def generate_binary_frames():
            while robot.running:
                frame = None
                
                # Get frame from the line follower camera
                if hasattr(robot, 'camera_line_follower'):
                    frame = robot.camera_line_follower.get_camera_frame()

                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Get the binary processed ROI
                if hasattr(robot, 'camera_line_follower'):
                    height, width = frame.shape[:2]
                    roi_start_y = int(height * robot.camera_line_follower.ROI_START_RATIO)
                    roi_end_y = int(height * robot.camera_line_follower.ARM_EXCLUSION_RATIO)
                    roi = frame[roi_start_y:roi_end_y, :]
                    
                    # Get the preprocessed binary ROI
                    binary_roi = robot.camera_line_follower._preprocess_roi(roi)
                    
                    # Create a full-size frame showing just the binary ROI
                    binary_frame = np.zeros((height, width), dtype=np.uint8)
                    binary_frame[roi_start_y:roi_end_y, :] = binary_roi
                    
                    # Convert to BGR for consistent encoding
                    binary_frame_bgr = cv2.cvtColor(binary_frame, cv2.COLOR_GRAY2BGR)
                else:
                    binary_frame_bgr = frame
                
                _, buffer = cv2.imencode('.jpg', binary_frame_bgr)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.05) # Limit frame rate

        return Response(generate_binary_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
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

    @app.route('/set_tts_provider', methods=['POST'])
    def set_tts_provider():
        """Change the TTS provider for audio feedback."""
        data = request.get_json()
        provider = data.get('provider', '')
        
        if robot.audio_feedback.set_provider(provider):
            return jsonify({
                'status': 'success',
                'message': f'TTS provider changed to {provider}',
                'current_provider': robot.audio_feedback.preferred_provider
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Provider {provider} not available',
                'available_providers': list(robot.audio_feedback.available_providers.keys())
            }), 400

    @app.route('/test_tts', methods=['POST'])
    def test_tts():
        """Test the current TTS provider with a sample message."""
        data = request.get_json()
        message = data.get('message', 'TTS test successful')
        provider = data.get('provider', None)
        
        robot.audio_feedback.speak(message, force_provider=provider)
        return jsonify({
            'status': 'success',
            'message': f'TTS test queued with provider: {provider or robot.audio_feedback.preferred_provider}'
        })

    @app.route('/arm_filter_status')
    def arm_filter_status():
        """Get current arm filtering status."""
        if hasattr(robot, 'camera_line_follower'):
            status = robot.camera_line_follower.get_arm_filtering_status()
            return jsonify({
                'status': 'success',
                'arm_filtering': status
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Camera line follower not available'
            }), 400

    @app.route('/set_arm_filter', methods=['POST'])
    def set_arm_filter():
        """Enable or disable purple arm filtering."""
        data = request.get_json()
        enabled = data.get('enabled', True)
        
        if hasattr(robot, 'camera_line_follower'):
            robot.camera_line_follower.set_arm_filtering(enabled)
            return jsonify({
                'status': 'success',
                'message': f'Arm filtering {"enabled" if enabled else "disabled"}',
                'arm_filtering': robot.camera_line_follower.get_arm_filtering_status()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Camera line follower not available'
            }), 400

    @app.route('/configure_arm_colors', methods=['POST'])
    def configure_arm_colors():
        """Configure purple color ranges for arm detection."""
        data = request.get_json()
        color_ranges = data.get('color_ranges', [])
        
        if not hasattr(robot, 'camera_line_follower'):
            return jsonify({
                'status': 'error',
                'message': 'Camera line follower not available'
            }), 400
        
        try:
            # Convert color ranges from list format to numpy arrays
            parsed_ranges = []
            for range_data in color_ranges:
                lower = np.array(range_data['lower'])
                upper = np.array(range_data['upper'])
                parsed_ranges.append((lower, upper))
            
            robot.camera_line_follower.configure_arm_colors(parsed_ranges)
            return jsonify({
                'status': 'success',
                'message': f'Configured {len(parsed_ranges)} color ranges',
                'arm_filtering': robot.camera_line_follower.get_arm_filtering_status()
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Failed to configure colors: {str(e)}'
            }), 400

    @app.route('/test_obstacle_detection')
    def test_obstacle_detection():
        """Test obstacle detection with current camera frame."""
        if not FEATURES['OBSTACLE_AVOIDANCE_ENABLED'] or not robot.obstacle_avoidance:
            return jsonify({
                'status': 'error',
                'message': 'Obstacle avoidance not enabled'
            }), 400
        
        if not hasattr(robot, 'camera_line_follower'):
            return jsonify({
                'status': 'error',
                'message': 'Camera line follower not available'
            }), 400
        
        frame = robot.camera_line_follower.get_camera_frame()
        if frame is None:
            return jsonify({
                'status': 'error',
                'message': 'No camera frame available'
            }), 400
        
        # Get line center from current detection
        line_center_x = robot.camera_line_result.get('line_center_x', frame.shape[1] // 2)
        
        # Test obstacle detection
        obstacle_result = robot.obstacle_avoidance.detect_line_blocking_obstacle(frame, line_center_x)
        
        return jsonify({
            'status': 'success',
            'obstacle_detection': {
                'is_blocking': obstacle_result['is_blocking'],
                'obstacle_detected': obstacle_result['obstacle_detected'],
                'distance': obstacle_result['distance'],
                'recommended_action': obstacle_result['recommended_action'],
                'line_center_x': line_center_x,
                'frame_size': f"{frame.shape[1]}x{frame.shape[0]}"
            }
        })

    @app.route('/adaptive_threshold_status')
    def adaptive_threshold_status():
        """Get current adaptive thresholding status."""
        if not hasattr(robot, 'camera_line_follower'):
            return jsonify({
                'status': 'error',
                'message': 'Camera line follower not available'
            }), 400
        
        status = robot.camera_line_follower.get_adaptive_threshold_status()
        return jsonify({
            'status': 'success',
            'adaptive_threshold': status
        })

    @app.route('/set_adaptive_threshold', methods=['POST'])
    def set_adaptive_threshold():
        """Configure adaptive threshold parameters."""
        if not hasattr(robot, 'camera_line_follower'):
            return jsonify({
                'status': 'error',
                'message': 'Camera line follower not available'
            }), 400
        
        data = request.get_json()
        
        # Extract parameters
        block_size = data.get('block_size')
        c_constant = data.get('c_constant')
        method_str = data.get('method', 'gaussian')
        condition = data.get('condition', 'normal')
        
        # Convert method string to OpenCV constant
        method = None
        if method_str == 'gaussian':
            method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        elif method_str == 'mean':
            method = cv2.ADAPTIVE_THRESH_MEAN_C
        
        # Set parameters
        success = robot.camera_line_follower.set_adaptive_threshold_params(
            block_size=block_size,
            c_constant=c_constant,
            method=method,
            condition=condition
        )
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Updated {condition} adaptive threshold parameters',
                'adaptive_threshold': robot.camera_line_follower.get_adaptive_threshold_status()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Invalid parameters provided'
            }), 400

    @app.route('/set_brightness_thresholds', methods=['POST'])
    def set_brightness_thresholds():
        """Set brightness thresholds for automatic adaptation."""
        if not hasattr(robot, 'camera_line_follower'):
            return jsonify({
                'status': 'error',
                'message': 'Camera line follower not available'
            }), 400
        
        data = request.get_json()
        bright_threshold = data.get('bright_threshold')
        dim_threshold = data.get('dim_threshold')
        
        success = robot.camera_line_follower.set_brightness_thresholds(
            bright_threshold=bright_threshold,
            dim_threshold=dim_threshold
        )
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Updated brightness thresholds',
                'adaptive_threshold': robot.camera_line_follower.get_adaptive_threshold_status()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Invalid threshold values'
            }), 400

    @app.route('/toggle_threshold_methods', methods=['POST'])
    def toggle_threshold_methods():
        """Enable/disable different thresholding methods."""
        if not hasattr(robot, 'camera_line_follower'):
            return jsonify({
                'status': 'error',
                'message': 'Camera line follower not available'
            }), 400
        
        data = request.get_json()
        
        robot.camera_line_follower.enable_threshold_methods(
            adaptive=data.get('adaptive'),
            simple=data.get('simple'),
            hsv=data.get('hsv'),
            auto_adapt=data.get('auto_adapt')
        )
        
        return jsonify({
            'status': 'success',
            'message': 'Updated thresholding methods',
            'adaptive_threshold': robot.camera_line_follower.get_adaptive_threshold_status()
        })

    @app.route('/set_simple_threshold', methods=['POST'])
    def set_simple_threshold():
        """Set simple threshold value."""
        if not hasattr(robot, 'camera_line_follower'):
            return jsonify({
                'status': 'error',
                'message': 'Camera line follower not available'
            }), 400
        
        data = request.get_json()
        threshold_value = data.get('threshold_value')
        
        if threshold_value is not None:
            robot.camera_line_follower.set_simple_threshold(threshold_value)
            return jsonify({
                'status': 'success',
                'message': f'Updated simple threshold to {threshold_value}',
                'adaptive_threshold': robot.camera_line_follower.get_adaptive_threshold_status()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'No threshold value provided'
            }), 400

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
            center_x = robot_x * cell_size + cell_size // 2
            center_y = robot_y * cell_size + cell_size // 2
            
            # Get robot's current direction from position tracker
            robot_direction = robot.position_tracker.current_direction
            
            # Map direction to angle in radians (0 = right, π/2 = down, π = left, 3π/2 = up)
            direction_angles = {
                'E': 0,           # East (right)
                'S': np.pi/2,     # South (down)
                'W': np.pi,       # West (left)
                'N': 3*np.pi/2    # North (up)
            }
            
            angle = direction_angles.get(robot_direction, 0)
            
            # Draw robot as an arrow pointing in the facing direction
            arrow_length = int(cell_size * 0.6)  # Make arrow longer and more prominent
            arrow_end_x = int(center_x + arrow_length * np.cos(angle))
            arrow_end_y = int(center_y + arrow_length * np.sin(angle))
            
            # Draw the main body of the robot as a circle
            cv2.circle(grid_img, (center_x, center_y), cell_size // 4, (203, 102, 255), -1)
            
            # Draw a white border around the robot for better visibility
            cv2.circle(grid_img, (center_x, center_y), cell_size // 4, (255, 255, 255), 2)
            
            # Draw the arrow showing direction with a thicker, more visible line
            cv2.arrowedLine(grid_img, (center_x, center_y), (arrow_end_x, arrow_end_y), 
                           (255, 255, 0), 4, tipLength=0.4)  # Changed to bright cyan (BGR format)
            
            # Add a direction indicator text
            direction_text = f"{robot_direction}"
            text_x = center_x - 8
            text_y = center_y + cell_size // 2 + 15
            cv2.putText(grid_img, direction_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)  # Match arrow color
        
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
