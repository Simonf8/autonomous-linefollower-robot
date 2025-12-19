#!/usr/bin/env python3

import time
import cv2
import numpy as np
import threading
import math
from typing import List, Tuple, Optional

# Import configuration
from config import *

# Import our modular components
from object_detection import ObjectDetector
from pathfinder import Pathfinder
from box import BoxHandler
from pid import PIDController
from camera_line_follower import CameraLineFollower, CameraLineFollowingMixin
from encoder_position_tracker import EncoderPositionTracker
from pi_motor_controller import PiMotorController
from audio_feedback import AudioFeedback
from camera_obstacle_avoidance import CameraObstacleAvoidance

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
        self.pathfinder = Pathfinder(grid=MAZE_GRID, cell_size_m=CELL_SIZE_M, turn_penalty=4.0)

        self.path = []
        self.current_target_index = 0
        self.turn_to_execute = None
        self.turn_start_time = 0
        self.wait_start_time = 0
        self.action_start_time = 0
        self.last_turn_complete_time = 0
        self.corner_cell_to_highlight = None
        self.total_corners_in_path = 0
        self.corners_passed = 0
        self.is_straight_corridor = False
        self.last_cell_update_time = 0
        self.box_lost_counter = 0
        self.path_planned_for_dropoff = False

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
            left_speed = int((fl + bl) / 2)
            right_speed = int((fr + br) / 2)
            self.motor_controller.send_motor_speeds(left_speed, right_speed)

    def _setup_vision(self):
        """Initialize vision systems if enabled."""
        pass

    def run(self):
        """Main control loop."""
        if FEATURES['CAMERA_LINE_FOLLOWING_ENABLED']:
            if not self.camera_line_follower.initialize_camera():
                print("CRITICAL: Camera failed to initialize.")
                self.running = False
                return

        while self.running:
            self._run_state_machine()
            time.sleep(0.01)
        
        self.stop()
        
    def _start_mission(self):
        """Start the defined mission."""
        if self.state != "idle":
            return

        print("Starting mission...")
        self.audio_feedback.speak("Starting mission")
        self.last_cell_update_time = time.time()

        if FEATURES['CAMERA_LINE_FOLLOWING_ENABLED'] and hasattr(self, 'camera_line_follower'):
            self.camera_line_follower.clear_line_memory_buffer()
            self.camera_line_follower.reset_pid_controller()

        if not self.position_tracker.running:
            self.position_tracker.start()
            
        if FEATURES['BOX_MISSION_ENABLED']:
            self.state = "going_to_pickup"
        else:
            self.state = "planning"

    def _plan_path_to_target(self, target_cell):
        """Plan the path to the specified target cell."""
        current_cell = self.position_tracker.get_current_cell()
        path_nodes = self.pathfinder.find_path(current_cell, target_cell, prefer_straight=True)
        
        if path_nodes:
            self.path = path_nodes
            self.current_target_index = 0
            self.total_corners_in_path = self._calculate_corners_in_path(self.path)
            self.corners_passed = 0
            self.state = "path_following"
            self.is_straight_corridor = self._is_path_straight(self.path)
            self.audio_feedback.speak(f"Path planned with {self.total_corners_in_path} corners.")
        else:
            self.audio_feedback.speak("Path planning failed")
            self.state = "error"

    def _follow_path(self):
        """Follow the planned path using encoder-based cell transitions and camera alignment."""
        if not self.path or self.current_target_index >= len(self.path):
            self._stop_motors()
            if FEATURES['BOX_MISSION_ENABLED']:
                self.state = "at_dropoff" if self.box_handler.has_package else "at_pickup"
            else:
                self.state = "mission_complete"
            return

        current_cell = self.position_tracker.get_current_cell()
        target_cell = self.path[-1]
        
        # Replanning if off track
        if self.path and self.current_target_index < len(self.path):
            expected = self.path[self.current_target_index]
            if abs(current_cell[0] - expected[0]) + abs(current_cell[1] - expected[1]) > 2:
                self.audio_feedback.speak("Replanning route.")
                self._plan_path_to_target(target_cell)
                return
        
        # Update camera turn sequence
        if hasattr(self, 'camera_line_follower') and len(self.path) > self.current_target_index:
            turn_sequence = self._calculate_turn_sequence_for_path(
                self.path[self.current_target_index:],
                self.position_tracker.current_direction
            )
            if turn_sequence:
                self.camera_line_follower.set_path_to_destination(turn_sequence)

        waypoint_reached = (current_cell == self.path[self.current_target_index])
        
        if waypoint_reached:
            if self.current_target_index < len(self.path) - 1:
                next_wp = self.path[self.current_target_index + 1]
                required_turn = self._get_required_turn(current_cell, self.position_tracker.current_direction, next_wp)
                
                if required_turn != 'forward':
                    self.audio_feedback.speak(f"Turning {required_turn}.")
                    self._stop_motors()
                    self.turn_to_execute = required_turn
                    self.state = 'turning'
                    self.turn_start_time = time.time()
                    self.current_target_index += 1
                    return

            self.current_target_index += 1

        # Check for destination arrival
        if current_cell == target_cell:
            self.audio_feedback.speak("Target reached.")
            if FEATURES['BOX_MISSION_ENABLED']:
                self.state = "at_dropoff" if self.box_handler.has_package else "at_pickup"
            else:
                self.state = "mission_complete"
            return
       
        frame = self.camera_line_follower.get_camera_frame()
        if frame is None:
            if self.motor_controller:
                self.motor_controller.send_motor_speeds(BASE_SPEED, BASE_SPEED)
            return

        self.camera_line_result = self.camera_line_follower.detect_line_with_lookahead(frame)
        self.position_tracker.set_camera_line_result(self.camera_line_result)
        
        # Obstacle avoidance logic here if needed...
        
        # Motor control
        current_base_speed = BASE_SPEED
        # (Speed adjustment logic...)
        
        # Electromagnet activation
        if FEATURES['BOX_MISSION_ENABLED'] and self.box_handler and not self.box_handler.has_package:
            if target_cell in self.box_handler.pickup_locations:
                dist = abs(current_cell[0] - target_cell[0]) + abs(current_cell[1] - target_cell[1])
                if dist <= 3 and self.motor_controller:
                    self.motor_controller.electromagnet_on()

        fl, fr, bl, br = self.camera_line_follower.get_motor_speeds_lookahead(self.camera_line_result, base_speed=current_base_speed)
        
        if self.camera_line_result.get('ready_for_pickup', False):
            if self.box_handler and any(abs(current_cell[0]-loc[0]) + abs(current_cell[1]-loc[1]) <= 1 for loc in self.box_handler.pickup_locations):
                self.audio_feedback.speak("Box reached.")
                self._stop_motors()
                self.state = 'grabbing_box'
                return
        
        self._set_motor_speeds(fl, fr, bl, br)

    def _execute_arcing_turn(self):
        current_cell = self.position_tracker.get_current_cell()
        dist = 99
        if self.path:
            final = self.path[-1]
            dist = abs(current_cell[0] - final[0]) + abs(current_cell[1] - final[1])
        
        if dist <= 2:
            self._execute_precision_pivot_turn()
        else:
            self._execute_smooth_arcing_turn()

    def _execute_precision_pivot_turn(self):
        PIVOT_DURATION_S = 1.6
        time_in_turn = time.time() - self.turn_start_time
        if time_in_turn >= PIVOT_DURATION_S:
            self._stop_motors()
            self.position_tracker.update_direction_after_turn(self.turn_to_execute)
            self.turn_to_execute = None
            self.state = 'path_following'
        else:
            self._perform_pivot_turn(self.turn_to_execute)

    def _execute_smooth_arcing_turn(self):
        ARCING_TURN_DURATION_S = 1.8
        time_in_turn = time.time() - self.turn_start_time
        if time_in_turn >= ARCING_TURN_DURATION_S:
            self.corners_passed += 1
            self._stop_motors()
            self.position_tracker.update_direction_after_turn(self.turn_to_execute)
            self.turn_to_execute = None
            self.state = 'path_following'
        else:
            self._perform_arcing_turn(self.turn_to_execute)

    def _run_state_machine(self):
        self.position_tracker.update_position()
        if self.state == "idle":
            self._stop_motors()
            self.position_tracker.set_moving(False)
        elif self.state == "planning":
            self._plan_path_to_target(END_CELL if 'END_CELL' in globals() else (0,0)) # Fallback
        elif self.state == "path_following":
            self.position_tracker.set_moving(True)
            self._follow_path()
        elif self.state == "turning":
            self.position_tracker.set_moving(True)
            self._execute_arcing_turn()
        elif self.state == "going_to_pickup":
            target_info = self.box_handler.get_current_target()
            if target_info:
                self._plan_path_to_target(target_info[0])
            else:
                self.state = "mission_complete"
        elif self.state == "at_pickup":
            self.state = 'approaching_box'
            self.action_start_time = time.time()
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
                    self._plan_path_to_target(target_info[0])
                    self.path_planned_for_dropoff = True
        elif self.state == "at_dropoff":
            self.audio_feedback.speak("Delivering box.")
            if self.motor_controller:
                self.motor_controller.electromagnet_off()
            self.box_handler.deliver_package(self.position_tracker.get_current_cell())
            time.sleep(1.0)
            self.state = "mission_complete" if self.box_handler.is_mission_complete() else "going_to_pickup"
        elif self.state == "mission_complete":
            self.audio_feedback.speak("Mission complete.")
            self._stop_motors()
            self.position_tracker.set_moving(False)

    def _stop_motors(self):
        self._set_motor_speeds(0, 0, 0, 0)
    
    def stop(self):
        self.running = False
        self._stop_motors()
        self.position_tracker.stop()
        if FEATURES['CAMERA_LINE_FOLLOWING_ENABLED'] and hasattr(self, 'camera_line_follower'):
            self.camera_line_follower.release_camera()
        if self.motor_controller:
            self.motor_controller.stop()

    def _get_required_turn(self, current_pos, current_dir, target_pos) -> str:
        dx, dy = target_pos[0] - current_pos[0], target_pos[1] - current_pos[1]
        target_dir = ''
        if dx > 0: target_dir = 'E'
        elif dx < 0: target_dir = 'W'
        elif dy > 0: target_dir = 'S'
        elif dy < 0: target_dir = 'N'
        if current_dir == target_dir: return 'forward'
        turn_logic = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}
        return 'right' if turn_logic[current_dir] == target_dir else 'left'

    def _perform_arcing_turn(self, direction):
        vx = BASE_SPEED * 0.3
        omega = TURN_SPEED * 0.8
        turn_omega = -omega if direction == 'left' else omega
        fl = int(vx + turn_omega)
        fr = int(vx - turn_omega)
        self._set_motor_speeds(fl, fr, fl, fr)

    def _perform_pivot_turn(self, direction):
        speed = CORNER_SPEED
        if direction == "left":
            self._set_motor_speeds(-speed, speed, -speed, speed)
        else:
            self._set_motor_speeds(speed, -speed, speed, -speed)

    def _handle_approaching_box(self):
        result = self.camera_line_result
        if result.get('ready_for_pickup', False):
            self.state = 'grabbing_box'
            return
        if result.get('box_detected', False):
            self._set_motor_speeds(30, 30, 30, 30)
        else:
            self._set_motor_speeds(20, 20, 20, 20)
        if time.time() - self.action_start_time > 10.0:
            self.state = 'grabbing_box'

    def _handle_grabbing_box(self):
        if self.motor_controller:
            self.motor_controller.electromagnet_on()
        self.box_handler.collect_package(self.position_tracker.get_current_cell())
        self._set_motor_speeds(40, 40, 40, 40)
        time.sleep(1.0)
        self._stop_motors()
        time.sleep(1.0)
        self.state = "reversing_from_box"
        self.action_start_time = time.time()

    def _handle_reversing_from_box(self):
        if time.time() - self.action_start_time < 2.0:
            self._set_motor_speeds(-30, -30, -30, -30)
        else:
            self._stop_motors()
            self.state = 'turning_after_pickup'
            self.action_start_time = time.time()

    def _handle_turning_after_pickup(self):
        if time.time() - self.action_start_time < 2.0:
            self._perform_pivot_turn('right')
        else:
            self._stop_motors()
            self.position_tracker.update_direction_after_turn('right')
            self.position_tracker.update_direction_after_turn('right') # 180 deg
            self.state = 'going_to_dropoff'
            self.path_planned_for_dropoff = False

    def _is_path_straight(self, path):
        if len(path) < 2: return True
        return all(p[0] == path[0][0] for p in path) or all(p[1] == path[0][1] for p in path)

    def _calculate_corners_in_path(self, path):
        if len(path) < 2: return 0
        corners = 0
        cur_dir = self.position_tracker.start_direction
        for i in range(len(path)-1):
            turn = self._get_required_turn(path[i], cur_dir, path[i+1])
            if turn in ['left', 'right']:
                corners += 1
                cur_dir = self._simulate_direction(cur_dir, turn)
        return corners

    def _simulate_direction(self, d, t):
        m = {'left': {'N': 'W', 'W': 'S', 'S': 'E', 'E': 'N'}, 'right': {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}}
        return m[t][d]

    def _calculate_turn_sequence_for_path(self, path, direction):
        seq = []
        cur_dir = direction
        for i in range(len(path)-1):
            turn = self._get_required_turn(path[i], cur_dir, path[i+1])
            if turn in ['left', 'right']:
                seq.append(turn)
                cur_dir = self._simulate_direction(cur_dir, turn)
        return seq
