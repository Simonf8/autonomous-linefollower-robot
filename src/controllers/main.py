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
    'BOX_MISSION_ENABLED': True,
    'OBJECT_DETECTION_ENABLED': True,
    'OBSTACLE_AVOIDANCE_ENABLED': False,  # Enabled for line blocking detection
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
CELL_SIZE_M = 0.061
BASE_SPEED = 75  # Further increased to reduce stopping and correction loops
TURN_SPEED = 60     # Increased for smoother turning
CORNER_SPEED = 55   # Increased for less stopping

# If corners are still too fast, you can further reduce these values:
# TURN_SPEED = 10   # Ultra-slow turning
# CORNER_SPEED = 8  # Ultra-slow cornering

# Hardware-specific trims to account for motor differences.
# Values are multipliers (1.0 = no change, 0.9 = 10% slower).
MOTOR_TRIMS =  {
    'left': 1.0,   
    'right': 0.98, 
}

# Box Mission Configuration
PICKUP_LOCATIONS = [(20, 14), (18, 14), (16, 14), (14, 14)]
DROPOFF_LOCATIONS = [(0, 0), (2, 0), (4, 0), (6, 0)]

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
if not FEATURES['BOX_MISSION_ENABLED']:
    END_CELL = (20, 14)   # End position for non-box missions
else:
    END_CELL = None

# START_DIRECTION must be a cardinal direction: 'N', 'S', 'E', or 'W'.
# This tells the robot its initial orientation on the map grid.
#  - 'N': Faces towards smaller row numbers (Up on the map)
#  - 'S': Faces towards larger row numbers (Down on the map)
#  - 'E': Faces towards larger column numbers (Right on the map)
#  - 'W': Faces towards smaller column numbers (Left on the map)
START_DIRECTION = 'E'

# Time in seconds it takes for the robot to cross one 12cm cell at BASE_SPEED.

CELL_CROSSING_TIME_S = 1.5  # Increased to act as a safeguard timeout

# Corner turning configuration
CORNER_TURN_MODES = {
    'SMOOTH': 'smooth',
    'SIDEWAYS': 'sideways',
    'PIVOT': 'pivot',
    'FRONT_TURN': 'front_turn'
}
CORNER_DETECTION_THRESHOLD = 0.05
CORNER_TURN_DURATION = 1    # Increased from 30 to 60 for longer turn timeout
SHARP_CORNER_THRESHOLD = 0.5

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

        if FEATURES['BOX_MISSION_ENABLED']:
            self.box_handler = BoxHandler(
                pickup_locations=PICKUP_LOCATIONS,
                dropoff_locations=DROPOFF_LOCATIONS
            )
        else:
            self.box_handler = None

        if FEATURES['OBJECT_DETECTION_ENABLED']:
            self.object_detector = ObjectDetector()
        else:
            self.object_detector = None
        
        self.frame = None
        self.processed_frame = None
        self.frame_lock = threading.Lock()

        self.detections = {}
        self.box_detection_result = None
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

        # Pathfinder setup with turn penalty for preferring straight lines
        # Turn penalty: 0 = shortest path, 3 = moderate preference for straight, 5+ = strong preference
        self.pathfinder = Pathfinder(grid=MAZE_GRID, cell_size_m=CELL_SIZE_M, turn_penalty=4.0)

        self.path = []
        self.current_target_index = 0
        self.turn_to_execute = None # Stores the next turn ('left' or 'right')
        self.turn_start_time = 0
        self.wait_start_time = 0
        self.action_start_time = 0 # For timed actions like approaching/reversing from box
        self.last_turn_complete_time = 0 # Cooldown timer for turns
        self.corner_cell_to_highlight = None # The cell where a turn is planned
        self.total_corners_in_path = 0 # How many turns are in the planned path
        self.corners_passed = 0 # How many corners we have passed so far
        self.is_straight_corridor = False
        self.last_cell_update_time = 0
        self.box_lost_counter = 0
        self.path_planned_for_dropoff = False  # Track if we've already planned path to dropoff

        self.line_pid = PIDController(kp=0.15, ki=0.001, kd=0.02, output_limits=(-30, 30))
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
            # Convert 4-wheel omni speeds to 2-wheel differential drive
            # Average left side (fl + bl) and right side (fr + br)
            left_speed = int((fl + bl) / 2)
            right_speed = int((fr + br) / 2)
            # MOTOR FIX: Back to normal - left motor = left_speed, right motor = right_speed
            self.motor_controller.send_motor_speeds(left_speed, right_speed)

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

        # Clear line memory buffer and reset PID controller for fresh start
        if FEATURES['CAMERA_LINE_FOLLOWING_ENABLED'] and hasattr(self, 'camera_line_follower'):
            self.camera_line_follower.clear_line_memory_buffer()
            self.camera_line_follower.reset_pid_controller()
            print("Line memory buffer cleared and adaptive PID reset for new mission")

        # Start position tracker only when mission begins
        if not self.position_tracker.running:
            self.position_tracker.start()
            
        # Pose is now managed by the encoder localizer
        if FEATURES['BOX_MISSION_ENABLED']:
            self.state = "going_to_pickup"
        else:
            self.state = "planning"

    def _process_vision(self):
        """Process the latest camera frame for events."""
        # This function is now OBSOLETE as vision processing is tied
        # directly to the line following logic.
        pass

    def _plan_path_to_target(self, target_cell):
        """Plan the path to the specified target cell."""
        current_cell = self.position_tracker.get_current_cell()
        print(f"Planning path from {current_cell} to {target_cell}...")
        
        # Use the new pathfinding that prefers straight lines
        path_nodes = self.pathfinder.find_path(current_cell, target_cell, prefer_straight=True)
        
        if path_nodes:
            self.path = path_nodes
            self.current_target_index = 0
            
            # Calculate and store the number of corners in the planned path
            self.total_corners_in_path = self._calculate_corners_in_path(self.path)
            self.corners_passed = 0

            self.state = "path_following"
            
            # Check if the path is a straight line to determine the following strategy.
            self.is_straight_corridor = self._is_path_straight(self.path)
            
            # Get path analysis for debugging
            turn_count = self.pathfinder.count_turns_in_path(self.path)
            segments = self.pathfinder.get_path_segments(self.path)
            
            # Log path segments
            print(f"Path analysis: {turn_count} turns, {len(segments)} segments")
            for i, (length, direction) in enumerate(segments):
                print(f"  Segment {i+1}: {length} cells {direction}")
            
            path_message = f"Path planned with {len(self.path)} waypoints and {self.total_corners_in_path} corners."
            print(path_message)
            self.audio_feedback.speak(path_message)
            
        else:
            print(f"Failed to plan path from {current_cell} to {target_cell}")
            self.audio_feedback.speak("Path planning failed")
            self.state = "error"

    def _follow_path(self):
        """Follow the planned path using encoder-based cell transitions and camera-based line centering."""
        if not self.path or self.current_target_index >= len(self.path):
            # This case should ideally not be hit if logic is correct,
            # as arrival at the destination cell should change state first.
            self._stop_motors()
            # Determine what to do next based on mission
            if FEATURES['BOX_MISSION_ENABLED']:
                if self.box_handler.has_package:
                    self.state = "at_dropoff"
                else:
                    self.state = "at_pickup"
            else:
                self.state = "mission_complete"
            return

        current_cell = self.position_tracker.get_current_cell()
        target_cell = self.path[-1]
        
        # Update camera line follower with upcoming turn sequence
        if hasattr(self, 'camera_line_follower') and len(self.path) > self.current_target_index:
            # Calculate turn sequence for remaining path
            turn_sequence = self._calculate_turn_sequence_for_path(
                self.path[self.current_target_index:],
                self.position_tracker.current_direction
            )
            if turn_sequence:
                self.camera_line_follower.set_path_to_destination(turn_sequence)

        # --- Waypoint Arrival Check ---
        target_waypoint = self.path[self.current_target_index]
        waypoint_reached = (current_cell == target_waypoint)

        # If we didn't get an exact match, try a more lenient check for corners where precision is critical.
        if not waypoint_reached:
            # Check if the target waypoint is a corner in the path
            is_target_a_corner = False
            if 0 < self.current_target_index < len(self.path) - 1:
                prev_wp = self.path[self.current_target_index - 1]
                next_wp = self.path[self.current_target_index + 1]
                # A corner is where the direction of movement changes.
                if (target_waypoint[0] - prev_wp[0] != next_wp[0] - target_waypoint[0]) or \
                   (target_waypoint[1] - prev_wp[1] != next_wp[1] - target_waypoint[1]):
                    is_target_a_corner = True
            
            if is_target_a_corner:
                # Check Manhattan distance for proximity
                dist = abs(current_cell[0] - target_waypoint[0]) + abs(current_cell[1] - target_waypoint[1])
                if dist <= 1:  # If we are at or adjacent to the corner
                    # Check if we have arrived at or passed the corner to avoid turning too early.
                    direction = self.position_tracker.current_direction
                    dx = current_cell[0] - target_waypoint[0]
                    dy = current_cell[1] - target_waypoint[1]
                    
                    passed_or_at_corner = False
                    if direction == 'E' and dx >= 0: passed_or_at_corner = True
                    if direction == 'W' and dx <= 0: passed_or_at_corner = True
                    if direction == 'S' and dy >= 0: passed_or_at_corner = True
                    if direction == 'N' and dy <= 0: passed_or_at_corner = True
                    
                    if passed_or_at_corner:
                        waypoint_reached = True
                        print(f"LENIENT MATCH: Close to corner {target_waypoint} (at {current_cell}), triggering turn.")
        
        if waypoint_reached:
            print(f"Reached waypoint {self.current_target_index} by ENCODER: {current_cell}")
            self.last_cell_update_time = time.time()
        
            # Check if this waypoint is a corner that requires a turn.
            if self.current_target_index < len(self.path) - 1:
                next_waypoint = self.path[self.current_target_index + 1]
                current_direction = self.position_tracker.current_direction
                required_turn = self._get_required_turn(current_cell, current_direction, next_waypoint)
                
                if required_turn != 'forward':
                    print(f"ENCODER: Path requires a '{required_turn}' turn. Initiating turn.")
                    # Use the actual required turn direction - no more hardcoded left turns!
                    self.audio_feedback.speak(f"Turning {required_turn}.")
                    
                    self._stop_motors()
                    time.sleep(0.05)  # Reduced pause to minimize stopping
                    self.turn_to_execute = required_turn
                    self.state = 'turning'
                    self.turn_start_time = time.time()
                    self.current_target_index += 1  # We've handled this waypoint's turn
                    return

            # If no turn needed, or it's the last waypoint in a segment
            self.current_target_index += 1
            if self.current_target_index >= len(self.path):
                # This handles arrival at the destination for this path segment
                # The final state transition is handled by the `current_cell == target_cell` check below
                pass

        # Remove early transition - let robot follow path all the way to the box
        # The box will be visible after the turn, no need to search for it early

        # Check for arrival at the final destination of the current path segment
        if current_cell == target_cell:
            print(f"Reached target destination: {current_cell}")
            self.audio_feedback.speak("Target reached.")
            
            if FEATURES['BOX_MISSION_ENABLED']:
                if self.box_handler.has_package:
                    self.state = "at_dropoff"
                else:
                    self.state = "at_pickup"
            else:
                self.state = "mission_complete" # Fallback for non-box mission
                self._stop_motors()
                return
       
        frame = self.camera_line_follower.get_camera_frame()
        if frame is None:
            # If camera fails, maybe we should stop or rely on blind forward movement
            
            if self.motor_controller:
                self.motor_controller.send_motor_speeds(BASE_SPEED, BASE_SPEED)
            return

        # PERFORMANCE: Skip every other frame for obstacle detection to improve speed
        if not hasattr(self, '_frame_skip_counter'):
            self._frame_skip_counter = 0
        self._frame_skip_counter += 1
        
        # Prepare robot state and encoder data for line detection with memory buffer
        current_cell = self.position_tracker.get_current_cell()
        robot_state = {
            'position': current_cell,
            'direction': self.position_tracker.current_direction,
            'motor_speeds': {'left': self.motor_speeds.get('fl', 0), 'right': self.motor_speeds.get('fr', 0)},
            'timestamp': time.time()
        }
        
        encoder_counts = self.motor_controller.get_encoder_counts() if self.motor_controller else {}
        
        # Use the new look-ahead detection system
        self.camera_line_result = self.camera_line_follower.detect_line_with_lookahead(frame)
        
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
        
        # Check if look-ahead system detects an intersection is imminent
        if self.camera_line_result.get('intersection_now', False):
            # The look-ahead system has detected we're at an intersection
            # This can be used for verification or emergency handling
            print(f"LOOK-AHEAD: Intersection NOW detected - type: {self.camera_line_result.get('upcoming_intersection_type', 'unknown')}")
            
        # Check if look-ahead system sees an intersection ahead
        if self.camera_line_result.get('intersection_ahead', False):
            countdown = self.camera_line_result.get('intersection_countdown', 0)
            if self.debug and countdown % 5 == 0:
                print(f"LOOK-AHEAD: Intersection ahead in {countdown} frames")

        # Normal line following if no obstacles or intersections
        # Adjust speed based on proximity to destination
        current_base_speed = BASE_SPEED
        if self.path and len(self.path) > 0:
            remaining_waypoints = len(self.path) - self.current_target_index
            final_destination = self.path[-1]
            distance_to_destination = abs(current_cell[0] - final_destination[0]) + abs(current_cell[1] - final_destination[1])
            
            # ELECTROMAGNET CONTROL: Turn on electromagnet when 3 cells before pickup location
            if FEATURES['BOX_MISSION_ENABLED'] and self.box_handler and not self.box_handler.has_package:
                # Check if we're heading to a pickup location
                if target_cell in self.box_handler.pickup_locations:
                    if distance_to_destination <= 3 and self.motor_controller:
                        if not self.motor_controller.get_electromagnet_status():
                            print(f"Approaching pickup zone ({distance_to_destination} cells away). Activating electromagnet.")
                            self.motor_controller.electromagnet_on()
            
            # Slow down when approaching destination (less aggressive)
            if remaining_waypoints <= 1 or distance_to_destination <= 1:
                current_base_speed = int(BASE_SPEED * 0.8)  # 80% speed for final approach (less slowdown)
            elif remaining_waypoints <= 2 or distance_to_destination <= 2:
                current_base_speed = int(BASE_SPEED * 0.9)  # 90% speed when close (minimal slowdown)
        
        # Use the new look-ahead motor control system
        fl, fr, bl, br = self.camera_line_follower.get_motor_speeds_lookahead(self.camera_line_result, base_speed=current_base_speed)
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
        Execute a precise, time-based pivot turn in place. This is more robust against vision failure.
        """
        # A 90-degree pivot turn takes a predictable amount of time.
        # This value may need tuning based on the robot's hardware and surface.
        PIVOT_DURATION_S = 1.4 # Reduced from 1.8
        TURN_TIMEOUT_S = 2.5          # Safety timeout
        
        time_in_turn = time.time() - self.turn_start_time
        
        # Turn completion is now based on time, not on re-acquiring the line with vision.
        turn_complete = time_in_turn >= PIVOT_DURATION_S
        turn_timed_out = time_in_turn > TURN_TIMEOUT_S

        if turn_complete or turn_timed_out:
            if turn_timed_out:
                self.audio_feedback.speak("Turn timed out.")
            else:
                self.audio_feedback.speak("Turn complete.")
                
            self._stop_motors()
            self.position_tracker.update_direction_after_turn(self.turn_to_execute)
            self.turn_to_execute = None
            self.corner_cell_to_highlight = None # Clear the highlight after turn completes
            self.state = 'path_following'
            self.last_turn_complete_time = time.time()
            time.sleep(0.1)  # Minimal pause for precision
        else:
            # Continue pivot turning (rotate in place)
            self._perform_pivot_turn(self.turn_to_execute)

    def _execute_smooth_arcing_turn(self):
        """
        Execute a more reliable, time-based arcing turn for general navigation.
        This removes dependency on vision for turn completion, making it more robust.
        """
        # Define a longer duration for slower, more controlled turns
        # This provides more consistent turning behavior.
        ARCING_TURN_DURATION_S = 1.6  # Reduced from 2.0
        TURN_TIMEOUT_S = 3.0  # Increased timeout to match longer turn duration

        time_in_turn = time.time() - self.turn_start_time

        # If the plan is to go forward, no turn is needed. Just exit the turning state.
        if self.turn_to_execute == 'forward':
            print("Turn type is 'forward'. Skipping physical turn.")
            self.audio_feedback.speak("Proceeding straight.")
            self._stop_motors()
            # No change in direction needed
            self.turn_to_execute = None
            self.state = 'path_following'
            self.last_turn_complete_time = time.time()
            self.corner_cell_to_highlight = None # Clear highlight
            time.sleep(0.25)
            return

        # The turn is complete once the fixed duration has elapsed or it times out.
        if time_in_turn >= ARCING_TURN_DURATION_S or time_in_turn > TURN_TIMEOUT_S:
            if time_in_turn > TURN_TIMEOUT_S:
                self.audio_feedback.speak("Turn timed out.")
            else:
                self.audio_feedback.speak("Turn complete.")

            if self.turn_to_execute in ['left', 'right']:
                self.corners_passed += 1

            self._stop_motors()
            self.position_tracker.update_direction_after_turn(self.turn_to_execute)
            self.turn_to_execute = None
            self.corner_cell_to_highlight = None # Clear the highlight after turn completes
            self.state = 'path_following'
            self.last_turn_complete_time = time.time()  # Start cooldown
            time.sleep(0.1)  # Minimal pause to stabilize and re-acquire line
        else:
            # Condition not met, continue the arcing turn.
            # We no longer need to check for line centering during the turn itself.
            self._perform_arcing_turn(self.turn_to_execute)

    def _run_state_machine(self):
        """Run the robot's state machine."""
        # Update position from encoders at the start of each cycle.
        self.position_tracker.update_position()

        if self.state == "idle":
            self._stop_motors()
            self.position_tracker.set_moving(False)
        elif self.state == "planning":
            self._plan_path_to_target(END_CELL)
        elif self.state == "path_following":
            self.position_tracker.set_moving(True)
            self._follow_path()
        elif self.state == "turning":
            self.position_tracker.set_moving(True)
            self._execute_arcing_turn()
        elif self.state == "obstacle_turn_around":
            self.position_tracker.set_moving(True)
            self._execute_180_turn()
        elif self.state == "destination_alignment":
            self.position_tracker.set_moving(True)
            self._execute_destination_alignment()
        elif self.state == "going_to_pickup":
            target_info = self.box_handler.get_current_target()
            if target_info:
                target_cell, mission_type = target_info
                if mission_type == "PICKUP":
                    self._plan_path_to_target(target_cell)
                else: # Should not happen in this state
                    self.state = "error"
            else:
                self.state = "mission_complete"
        elif self.state == "at_pickup":
            print("At pickup location. Box will be ahead - approaching directly.")
            # No need to stop and search - the box will be there after the turn!
            self.audio_feedback.speak("Approaching for pickup.")
            self.state = 'approaching_box'
            self.action_start_time = time.time()

        elif self.state == "locating_box":
            self._handle_locating_box()

        elif self.state == "approaching_box":
            self._handle_approaching_box()

        elif self.state == "grabbing_box":
            self._handle_grabbing_box()
        
        elif self.state == "reversing_from_box":
            self._handle_reversing_from_box()

        elif self.state == "turning_after_pickup":
            self._handle_turning_after_pickup()

        elif self.state == "going_to_dropoff":
            if not self.path_planned_for_dropoff:
                target_info = self.box_handler.get_current_target()
                if target_info:
                    target_cell, mission_type = target_info
                    if mission_type == "DROPOFF":
                        print(f"Planning path to dropoff at {target_cell}")
                        # Force update position tracker after the left turn
                        current_cell = self.position_tracker.get_current_cell()
                        print(f"Robot is now at {current_cell} facing {self.position_tracker.current_direction}")
                        self._plan_path_to_target(target_cell)
                        self.path_planned_for_dropoff = True
                        # Note: _plan_path_to_target() will change state to "path_following" if successful
                    else: # Should not happen
                        print(f"ERROR: Expected DROPOFF mission but got {mission_type}")
                        self.state = "error"
                else:
                    print("ERROR: No target info available for dropoff")
                    self.state = "mission_complete" # Should not happen
            else:
                # Path already planned, should be in path_following state by now
                # If we're still here, there might be an issue - force state change
                print("WARNING: Path planned but still in going_to_dropoff state. Forcing path_following.")
                if self.path and len(self.path) > 0:
                    self.state = "path_following"
                else:
                    print("ERROR: No valid path found for dropoff")
                    self.state = "error"

        elif self.state == "at_dropoff":
            print("At dropoff location. Disengaging electromagnet.")
            self.audio_feedback.speak("Delivering box.")
            if self.motor_controller:
                self.motor_controller.electromagnet_off()
            
            current_cell = self.position_tracker.get_current_cell()
            self.box_handler.deliver_package(current_cell)
            time.sleep(1.0) # Wait for box to release
            
            if self.box_handler.is_mission_complete():
                self.state = "mission_complete"
            else:
                self.state = "going_to_pickup"

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
        slow_turn_speed = CORNER_SPEED  # Use even slower corner speed for pivot turns
        
        if corner_direction == "left":
            # Rotate counter-clockwise (left turn)
            fl_speed = -slow_turn_speed    # Front left: reverse
            fr_speed = slow_turn_speed     # Front right: forward
            bl_speed = -slow_turn_speed    # Back left: reverse  
            br_speed = slow_turn_speed     # Back right: forward
        else:  # right turn
            # Rotate clockwise (right turn)
            fl_speed = slow_turn_speed     # Front left: forward
            fr_speed = -slow_turn_speed    # Front right: reverse
            bl_speed = slow_turn_speed     # Back left: forward
            br_speed = -slow_turn_speed    # Back right: reverse
        
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
        vx = BASE_SPEED * 0.3  # Much slower forward speed during turns for better control
        omega = TURN_SPEED * 0.8     # Reduce rotational speed for smoother turns

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
        # Use slower corner speed for controlled precision turning
        slow_turn_speed = CORNER_SPEED
        
        if direction == "left":
            # Rotate counter-clockwise (left turn)
            fl_speed = -slow_turn_speed    # Front left: reverse
            fr_speed = slow_turn_speed     # Front right: forward
            bl_speed = -slow_turn_speed    # Back left: reverse  
            br_speed = slow_turn_speed     # Back right: forward
        else:  # right turn
            # Rotate clockwise (right turn)
            fl_speed = slow_turn_speed     # Front left: forward
            fr_speed = -slow_turn_speed    # Front right: reverse
            bl_speed = slow_turn_speed     # Back left: forward
            br_speed = -slow_turn_speed    # Back right: reverse
        
        self._set_motor_speeds(fl_speed, fr_speed, bl_speed, br_speed)

    def _perform_slow_pivot_turn(self, direction: str):
        """
        Commands motor speeds for a very slow, controlled pivot turn after pickup.
        This prevents over-spinning and provides more precise control.
        """
        # Use much slower speed for post-pickup turn
        extra_slow_turn_speed = 35  # Much slower than CORNER_SPEED (55)
        
        if direction == "left":
            # Rotate counter-clockwise (left turn)
            fl_speed = -extra_slow_turn_speed    # Front left: reverse
            fr_speed = extra_slow_turn_speed     # Front right: forward
            bl_speed = -extra_slow_turn_speed    # Back left: reverse  
            br_speed = extra_slow_turn_speed     # Back right: forward
        else:  # right turn
            # Rotate clockwise (right turn)
            fl_speed = extra_slow_turn_speed     # Front left: forward
            fr_speed = -extra_slow_turn_speed    # Front right: reverse
            bl_speed = extra_slow_turn_speed     # Back left: forward
            br_speed = -extra_slow_turn_speed    # Back right: reverse
        
        self._set_motor_speeds(fl_speed, fr_speed, bl_speed, br_speed)

    def _execute_destination_alignment(self):
        """
        Turn at destination until line is found and centered.
        """
        MAX_TURN_DURATION = 8.0   # Reduced timeout for destination alignment
        MIN_TURN_DURATION = 0.5    # Shorter minimum for destination
        
        time_in_turn = time.time() - self.turn_start_time
        
        # Get current camera frame to check for line
        frame = self.camera_line_follower.get_camera_frame()
        line_found_and_centered = False
        
        if frame is not None and time_in_turn > MIN_TURN_DURATION:
            # Prepare robot state for line detection
            current_cell = self.position_tracker.get_current_cell()
            robot_state = {
                'position': current_cell,
                'direction': self.position_tracker.current_direction,
                'motor_speeds': {'left': self.motor_speeds.get('fl', 0), 'right': self.motor_speeds.get('fr', 0)},
                'timestamp': time.time()
            }
            
            encoder_counts = self.motor_controller.get_encoder_counts() if self.motor_controller else {}
            
            # Detect line with full processing
            line_result = self.camera_line_follower.detect_line(frame, robot_state, encoder_counts)
            line_found = line_result.get('line_detected', False)
            
            if line_found:
                line_offset = line_result.get('line_offset', 1.0)
                # Line must be well-centered for destination alignment
                if abs(line_offset) < 0.15:  # Stricter centering requirement
                    line_found_and_centered = True
                    if self.debug:
                        print(f"DESTINATION: Line found and centered! Offset: {line_offset:.3f}")
        
        if time_in_turn < MAX_TURN_DURATION and not line_found_and_centered:
            # Continue turning slowly - pivot turn (rotate in place)
            slow_turn_speed = CORNER_SPEED  # Use slower speed for precise alignment
            fl_speed = slow_turn_speed     # Front left: forward
            fr_speed = -slow_turn_speed    # Front right: reverse
            bl_speed = slow_turn_speed     # Back left: forward
            br_speed = -slow_turn_speed    # Back right: reverse
            
            self._set_motor_speeds(fl_speed, fr_speed, bl_speed, br_speed)
            
            if self.debug and int(time_in_turn * 2) != int((time_in_turn - 0.1) * 2):  # Print every 0.5 seconds
                print(f"DESTINATION ALIGNMENT: Turning to find line: {time_in_turn:.1f}s / {MAX_TURN_DURATION}s")
        else:
            # Alignment complete - either line found or timed out
            if line_found_and_centered:
                print("Destination alignment complete! Line centered.")
                self.audio_feedback.speak("Mission complete! Line aligned.")
            else:
                print("Destination alignment timed out.")
                self.audio_feedback.speak("Mission complete! Alignment timeout.")
            
            self._stop_motors()
            self.state = "mission_complete"
            time.sleep(0.5)  # Brief pause to stabilize

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

    def _calculate_corners_in_path(self, path: List[Tuple[int, int]]) -> int:
        """Calculates the number of turns ('left' or 'right') in a given path."""
        if len(path) < 2:
            return 0

        corners = 0
        # To count corners accurately, we must track the robot's orientation through the path.
        # We start with the robot's initial direction.
        current_direction = self.position_tracker.start_direction
        
        for i in range(len(path) - 1):
            current_pos = path[i]
            next_pos = path[i+1]
            
            required_turn = self._get_required_turn(current_pos, current_direction, next_pos)
            
            if required_turn in ['left', 'right']:
                corners += 1
                
                # Update the simulated direction for the next path segment calculation.
                turn_map_right = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}
                turn_map_left = {'N': 'W', 'W': 'S', 'S': 'E', 'E': 'N'}
                if required_turn == 'left':
                    current_direction = turn_map_left[current_direction]
                else: # right
                    current_direction = turn_map_right[current_direction]
        
        return corners
    
    def _calculate_turn_sequence_for_path(self, path: List[Tuple[int, int]], start_direction: str) -> List[str]:
        """Calculate the sequence of turns needed for a given path."""
        if len(path) < 2:
            return []
        
        turn_sequence = []
        current_direction = start_direction
        
        for i in range(len(path) - 1):
            current_pos = path[i]
            next_pos = path[i+1]
            
            required_turn = self._get_required_turn(current_pos, current_direction, next_pos)
            
            if required_turn in ['left', 'right']:
                turn_sequence.append(required_turn)
                # Update direction after turn
                turn_map_right = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}
                turn_map_left = {'N': 'W', 'W': 'S', 'S': 'E', 'E': 'N'}
                if required_turn == 'left':
                    current_direction = turn_map_left[current_direction]
                else:
                    current_direction = turn_map_right[current_direction]
            elif required_turn == 'forward':
                # No turn needed, but we could add 'straight' to the sequence if needed
                pass
        
        return turn_sequence

    def _handle_locating_box(self):
        """Scan for a box, center it, and then approach it."""
        LOCATING_TIMEOUT_S = 15.0
        
        if time.time() - self.turn_start_time > LOCATING_TIMEOUT_S:
            print("Could not visually locate box within timeout. Trusting position and attempting blind approach.")
            self.audio_feedback.speak("Box not found. Trying blind pickup.")
            self._stop_motors()
            time.sleep(0.5)
            self.state = 'approaching_box' # Fallback to blind/timed approach
            self.action_start_time = time.time()
            return

        frame = self.camera_line_follower.get_camera_frame()
        if frame is None:
            return

        if self.object_detector:
            self.box_detection_result = self.object_detector.detect_objects(frame)
            
            if self.box_detection_result and self.box_detection_result['package_detected']:
                closest_package = self.object_detector.get_closest_package(self.box_detection_result)
                if closest_package:
                    # Box found, now center it
                    frame_center_x = frame.shape[1] / 2
                    box_center_x = closest_package['center'][0]
                    
                    centering_tolerance = 20 # pixels
                    
                    if abs(box_center_x - frame_center_x) > centering_tolerance:
                        # Box is not centered, pivot to center it
                        if box_center_x < frame_center_x:
                            self._perform_pivot_turn('left')
                        else:
                            self._perform_pivot_turn('right')
                    else:
                        # Box is centered, proceed to approach
                        print("Box centered. Approaching.")
                        self.audio_feedback.speak("Approaching box.")
                        self._stop_motors()
                        self.state = 'approaching_box'
                        self.action_start_time = time.time()
                return

        # If no box is found, keep pivoting slowly to scan
        self._perform_pivot_turn('left')

    def _handle_approaching_box(self):
        """
        Use the look-ahead system's integrated box detection and approach control.
        The camera system handles all the complex logic for us.
        """
        # The look-ahead system handles everything for box approach
        result = self.camera_line_result
        
        # Check if we're ready for pickup (box is in position)
        if result.get('ready_for_pickup', False):
            print("Look-ahead system signals: Box in pickup position!")
            self.audio_feedback.speak("Box reached. Grabbing.")
            self._stop_motors()
            self.state = 'grabbing_box'
            return
            
        # Check box detection status
        if result.get('box_detected', False):
            box_info = result.get('box_info', {})
            if box_info:
                print(f"Box detected - Color: {box_info.get('dominant_color', 'unknown')}, "
                      f"Position: {box_info.get('bottom_y_ratio', 0):.2f}")
            
            # The look-ahead system is already controlling motors for approach
            # We don't need to set motor speeds here - they're already set
            
        else:
            # No box detected - might need to search or continue approach
            print("No box in view - continuing approach...")
            
        # Safety timeout
        APPROACH_TIMEOUT_S = 5.0
        if time.time() - self.action_start_time > APPROACH_TIMEOUT_S:
            print("Box approach timed out. Attempting grab anyway.")
            self.audio_feedback.speak("Approach timeout. Grabbing.")
            self._stop_motors()
            self.state = 'grabbing_box'

    def _handle_grabbing_box(self):
        """Activate the electromagnet to pick up the box."""
        print("GRABBING BOX STATE: Entered state.")
        self.audio_feedback.speak("Grabbing box.")
        if self.motor_controller:
            print("GRABBING BOX STATE: Motor controller found. Activating electromagnet.")
            self.motor_controller.electromagnet_on()
        else:
            print("GRABBING BOX STATE: MOTOR CONTROLLER NOT FOUND. Electromagnet not activated.")
        
        current_cell = self.position_tracker.get_current_cell()
        print(f"GRABBING BOX: Collecting package at {current_cell}")
        self.box_handler.collect_package(current_cell)
        time.sleep(0.8) # Shorter wait for electromagnet to engage
        
        # Reset the box approach state in camera line follower
        if hasattr(self, 'camera_line_follower'):
            self.camera_line_follower.reset_box_approach()
        
        print("GRABBING BOX: Transitioning to reverse state")
        self.state = "reversing_from_box"
        self.action_start_time = time.time()

    def _handle_reversing_from_box(self):
        """Move backward for a fixed duration after picking up the box."""
        REVERSE_DURATION_S = 1.5  # Longer reverse time to get clear of box area
        REVERSE_SPEED = -35  # Faster reverse to be more decisive

        elapsed_time = time.time() - self.action_start_time
        print(f"REVERSING FROM BOX: {elapsed_time:.1f}s / {REVERSE_DURATION_S}s")
        
        if elapsed_time < REVERSE_DURATION_S:
            self._set_motor_speeds(REVERSE_SPEED, REVERSE_SPEED, REVERSE_SPEED, REVERSE_SPEED)
        else:
            self._stop_motors()
            print("Finished reversing from box. Starting turn.")
            time.sleep(0.1) # Minimal pause
            self.state = 'turning_after_pickup'
            self.action_start_time = time.time()

    def _handle_turning_after_pickup(self):
        """Perform a 180-degree turn after pickup to go back to dropoff."""
        # After picking up box, robot needs to turn around and go back to dropoff area
        TURN_DURATION_S = 1.2  # 180-degree turn takes more time than 90-degree

        elapsed_time = time.time() - self.action_start_time
        print(f"POST-PICKUP 180-TURN: {elapsed_time:.1f}s / {TURN_DURATION_S}s")
        
        if elapsed_time < TURN_DURATION_S:
            # 180-degree turn to face back towards dropoff area
            self._perform_slow_pivot_turn('right') 
        else:
            self._stop_motors()
            print("Completed 180-degree turn after pickup.")
            self.audio_feedback.speak("Turn complete. Going to dropoff.")
            
            # Update direction in position tracker (180 degrees)
            current_dir = self.position_tracker.current_direction
            opposite_directions = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
            new_direction = opposite_directions.get(current_dir, current_dir)
            self.position_tracker.current_direction = new_direction
            print(f"DIRECTION UPDATE: Changed from {current_dir} to {new_direction} after 180-turn")

            print("TRANSITIONING TO DROPOFF STATE")
            self.state = 'going_to_dropoff'
            self.path_planned_for_dropoff = False  # Reset flag for new dropoff planning
            time.sleep(0.1) # Minimal pause to stabilize

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

        box_states_serializable = {}
        if robot.box_handler:
            for box_id, box_info in robot.box_handler.box_states.items():
                info_copy = box_info.copy()
                info_copy['state'] = info_copy['state'].value
                box_states_serializable[box_id] = info_copy

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
                'using_prediction': robot.camera_line_result.get('using_prediction', False),
                'buffer_status': robot.camera_line_result.get('buffer_status', {}),
                'pid_metrics': robot.camera_line_follower.get_pid_performance_metrics() if hasattr(robot, 'camera_line_follower') else {},
                # Look-ahead system status
                'intersection_ahead': robot.camera_line_result.get('intersection_ahead', False),
                'intersection_countdown': robot.camera_line_result.get('intersection_countdown', 0),
                'intersection_now': robot.camera_line_result.get('intersection_now', False),
                'box_detected': robot.camera_line_result.get('box_detected', False),
                'box_in_position': robot.camera_line_result.get('box_in_position', False),
                'box_approach_active': robot.camera_line_result.get('box_approach_active', False),
            },
            'obstacle_avoidance': {
                'enabled': FEATURES['OBSTACLE_AVOIDANCE_ENABLED'],
            },
            'motors': robot.motor_speeds,
            'electromagnet_on': robot.motor_controller.get_electromagnet_status() if robot.motor_controller else False,
            'encoders': encoder_counts,
            'audio_feedback': {
                'enabled': audio_status.get('enabled', False),
                'provider': audio_status.get('preferred_provider', 'none'),
                'available_providers': audio_status.get('available_providers', []),
                'queue_size': audio_status.get('queue_size', 0),
            },
            'path_info': {
                'path': robot.path,
                'current_target_index': robot.current_target_index,
                'turn_to_execute': robot.turn_to_execute,
                'total_corners': robot.total_corners_in_path,
                'corners_passed': robot.corners_passed
            },
            'box_mission': {
                'enabled': FEATURES['BOX_MISSION_ENABLED'],
                'boxes': box_states_serializable,
                'has_package': robot.box_handler.has_package if robot.box_handler else False
            }
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

    @app.route('/tune_pid', methods=['POST'])
    def tune_pid():
        """Adjust adaptive PID controller tuning parameters."""
        if not FEATURES['CAMERA_LINE_FOLLOWING_ENABLED'] or not hasattr(robot, 'camera_line_follower'):
            return jsonify({
                'status': 'error',
                'message': 'Camera line follower not available'
            }), 400
        
        data = request.get_json()
        
        try:
            kp = data.get('kp')
            ki = data.get('ki')
            kd = data.get('kd')
            
            # Validate parameters
            if kp is not None and (kp < 0 or kp > 5.0):
                raise ValueError("kp must be between 0 and 5.0")
            if ki is not None and (ki < 0 or ki > 1.0):
                raise ValueError("ki must be between 0 and 1.0")
            if kd is not None and (kd < 0 or kd > 2.0):
                raise ValueError("kd must be between 0 and 2.0")
            
            # Update PID tuning
            robot.camera_line_follower.pid.set_tuning(kp=kp, ki=ki, kd=kd)
            
            # Get current status
            pid_status = robot.camera_line_follower.get_pid_status()
            
            return jsonify({
                'status': 'success',
                'message': 'PID tuning updated successfully',
                'current_tuning': pid_status['base_gains']
            })
            
        except ValueError as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 400
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Failed to update PID tuning: {str(e)}'
            }), 500

    @app.route('/get_pid_status')
    def get_pid_status():
        """Get current adaptive PID controller status and performance metrics."""
        if not FEATURES['CAMERA_LINE_FOLLOWING_ENABLED'] or not hasattr(robot, 'camera_line_follower'):
            return jsonify({
                'status': 'error',
                'message': 'Camera line follower not available'
            }), 400
        
        try:
            pid_status = robot.camera_line_follower.get_pid_status()
            pid_metrics = robot.camera_line_follower.get_pid_performance_metrics()
            
            return jsonify({
                'status': 'success',
                'pid_status': pid_status,
                'performance_metrics': pid_metrics
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Failed to get PID status: {str(e)}'
            }), 500

    @app.route('/grid_feed')
    def grid_feed():
        """Streams the grid map visualization."""
        def generate():
            while robot.running:
                robot_cell = robot.position_tracker.get_current_cell()
                path = robot.path
                corner_cell = robot.corner_cell_to_highlight # Get the corner cell to highlight
                boxes = robot.box_handler.box_states if robot.box_handler else {}
                has_package = robot.box_handler.has_package if robot.box_handler else False
                
                grid_img = generate_grid_image(
                    pathfinder=robot.pathfinder,
                    robot_cell=robot_cell,
                    path=path,
                    start_cell=START_CELL,
                    end_cell=END_CELL,
                    corner_cell=corner_cell,
                    boxes=boxes,
                    has_package=has_package
                )
                
                _, buffer = cv2.imencode('.jpg', grid_img)
                frame = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
                time.sleep(0.1)
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def generate_grid_image(pathfinder, robot_cell, path, start_cell, end_cell, corner_cell=None, boxes=None, has_package=False):
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
        
        # Draw boxes based on their state
        if boxes:
            for box_id, box_info in boxes.items():
                if box_info['state'].value == 'available':
                    loc = box_info['pickup_location']
                    color = (0, 255, 255) # Yellow for available boxes
                elif box_info['state'].value == 'delivered':
                    loc = box_info['dropoff_location']
                    color = (0, 0, 255) # Red for delivered boxes
                else: # Collected, on the robot
                    continue

                if loc:
                    cx, cy = loc
                    # Draw a circle for the box
                    center_x = cx * cell_size + cell_size // 2
                    center_y = cy * cell_size + cell_size // 2
                    cv2.circle(grid_img, (center_x, center_y), cell_size // 3, color, -1)
                    cv2.circle(grid_img, (center_x, center_y), cell_size // 3, (0,0,0), 1)


        # Highlight the corner cell if one is detected
        if corner_cell:
            cx, cy = corner_cell
            # Don't draw over start/end cells
            if (cx, cy) != start_cell and (cx, cy) != end_cell:
                # Use a different shade of green to distinguish from the start cell
                cv2.rectangle(grid_img, (cx * cell_size, cy * cell_size),
                              ((cx + 1) * cell_size, (cy + 1) * cell_size), (0, 180, 0), -1)

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
        
        if end_cell:
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
            robot_color = (0, 255, 255) if has_package else (203, 102, 255)
            cv2.circle(grid_img, (center_x, center_y), cell_size // 4, robot_color, -1)
            
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
