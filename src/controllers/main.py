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
from visual_localizer import PreciseMazeLocalizer
from pi_motor_controller import PiMotorController

# ================================
# FEATURE CONFIGURATION
# ================================
# Enable/disable features for easy testing and debugging
FEATURES = {
    'OBJECT_DETECTION_ENABLED': False,
    'OBSTACLE_AVOIDANCE_ENABLED': False,
    'VISION_SYSTEM_ENABLED': True,
    'CAMERA_LINE_FOLLOWING_ENABLED': True, 
    'POSITION_CORRECTION_ENABLED': True,
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
    'br': 0.75  # Back-right motor is a bit faster, slow it down by 5%
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
START_CELL = (20, 5) # Start position (col, row)
END_CELL = (0, 0)   # End position (col, row)
START_DIRECTION = 'F' # Use 'F'(North), 'B'(South), 'L'(West), 'R'(East)

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
        
        # Initialize motor controller directly
        self.motor_controller = PiMotorController(trims=MOTOR_TRIMS)

        self.object_detector = None
        self.frame = None
        self.processed_frame = None
        self.frame_lock = threading.Lock()

        self.detections = {}
        self.last_intersection_time = 0

        # Position tracking using Visual Localizer
        self.position_tracker = PreciseMazeLocalizer(
            maze=MAZE_GRID,
            start_pos=START_CELL,
            camera_width=CAMERA_WIDTH,
            camera_height=CAMERA_HEIGHT,
            camera_fps=CAMERA_FPS,
            start_direction='N' # Default, will be updated from main
        )

        # Pathfinder setup
        self.pathfinder = Pathfinder(grid=MAZE_GRID, cell_size_m=CELL_SIZE_M)

        self.path = []
        self.current_target_index = 0
        self.turn_to_execute = None # Stores the next turn ('left' or 'right')
        self.turn_start_time = 0
        self.last_turn_complete_time = 0 # Cooldown timer for turns

        self.line_pid = PIDController(kp=0.5, ki=0.01, kd=0.1, output_limits=(-100, 100))
        self.state = "idle"

        if FEATURES['VISION_SYSTEM_ENABLED']:
            self._setup_vision()
            
        if FEATURES['CAMERA_LINE_FOLLOWING_ENABLED']:
            self.init_camera_line_following()

    def _setup_vision(self):
        """Initialize vision systems if enabled."""
        if not FEATURES['VISION_SYSTEM_ENABLED']:
            print("Vision system is disabled.")
            return

        print("Initializing vision system...")
        if FEATURES['OBJECT_DETECTION_ENABLED']:
            self.object_detector = ObjectDetector()

    def run(self):
        """Main control loop."""
        # Initialize the camera.
        if FEATURES['VISION_SYSTEM_ENABLED']:
            if isinstance(self.position_tracker, PreciseMazeLocalizer):
                if not self.position_tracker.initialize_camera():
                    print("CRITICAL: Camera failed to initialize. Aborting mission.")
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
        # Start localizer only when mission begins
        if isinstance(self.position_tracker, PreciseMazeLocalizer) and not self.position_tracker.running:
            self.position_tracker.start_localization()
            
        # Pose is now managed by the visual localizer
        self.state = "planning"

    def _process_vision(self):
        """Process the latest camera frame for events."""
        if not FEATURES['VISION_SYSTEM_ENABLED'] or self.frame is None:
            return

        with self.frame_lock:
            frame_copy = self.frame.copy()

        processed_frame = frame_copy
        
        # Visual localization is handled by its own thread, no need to call it here.
        # The intersection logic below can be used to augment or validate.
        
        # Store the processed frame for the video feed
        with self.frame_lock:
            self.processed_frame = processed_frame

    def _plan_path_to_target(self):
        """Plan the path to the target cell."""
        current_cell = self.position_tracker.get_current_cell()
        print(f"Planning path from {current_cell} to {END_CELL}...")
        
        path_nodes = self.pathfinder.find_path(current_cell, END_CELL)
        
        if path_nodes:
            self.path = path_nodes
            self.current_target_index = 0
            self.state = "path_following"
            print(f"Path planned: {len(self.path)} waypoints")
        else:
            print(f"Failed to plan path from {current_cell} to {END_CELL}")
            self.state = "error"

    def _follow_path(self):
        """Follow the planned path using smart, vision-based navigation."""
        if not self.path or self.current_target_index >= len(self.path):
            self.state = "mission_complete"
            self._stop_motors()
            return

        # Update waypoint if reached
        current_cell = self.position_tracker.get_current_cell()
        if current_cell == self.path[self.current_target_index]:
            print(f"Reached waypoint {self.current_target_index}: {current_cell}")
            self.current_target_index += 1
            if self.current_target_index >= len(self.path):
                self.state = "mission_complete"
                self._stop_motors()
                return

        # Determine required action to reach the next waypoint
        current_dir = self.position_tracker.current_direction
        next_waypoint = self.path[self.current_target_index]
        required_turn = self._get_required_turn(current_cell, current_dir, next_waypoint)

        # Get the latest camera frame and analyze it for the line and intersections
        frame = self.position_tracker.get_camera_frame()
        if frame is None:
            self._stop_motors()
            return

        vision_result = self.camera_line_follower.detect_line(frame)
        is_at_intersection = vision_result.get('is_at_intersection', False)
        
        # --- DEBUG LOG ---
        solidity = vision_result.get('solidity', 1.0)
        aspect_ratio = vision_result.get('aspect_ratio', 0.0)
        print(f"Path Follow: Cell={current_cell}, Target={next_waypoint}, Turn={required_turn}, Intersection={is_at_intersection} (S: {solidity:.2f}, AR: {aspect_ratio:.2f})")

        # Decide whether to turn or go forward
        TURN_COOLDOWN_S = 2.0 # Minimum time between turns
        can_turn = (time.time() - self.last_turn_complete_time) > TURN_COOLDOWN_S

        if required_turn in ['left', 'right'] and is_at_intersection and can_turn:
            print(f"Intersection detected! Stopping to prepare for {required_turn} turn.")
            self._stop_motors() # Stop immediately
            self.turn_to_execute = required_turn
            self.state = 'turning'
            self.turn_start_time = time.time()
        else:
            # Default action: follow the line forward using the vision result
            fl, fr, bl, br = self.camera_line_follower.get_motor_speeds(vision_result, base_speed=BASE_SPEED)
            self.motor_controller.send_motor_speeds(fl, fr, bl, br)

    def _execute_intersection_turn(self):
        """Executes a timed pivot turn at an intersection after a brief pause."""
        PRE_TURN_PAUSE_S = 0.2 # Brief pause to stabilize before turning
        TURN_DURATION_S = 0.8  # Time in seconds for a 90-degree turn. Needs tuning.
        
        if time.time() - self.turn_start_time < PRE_TURN_PAUSE_S:
            # Still in the pre-turn pause phase. Motors are already stopped.
            return
            
        turn_elapsed = (time.time() - self.turn_start_time) - PRE_TURN_PAUSE_S
        
        if turn_elapsed < TURN_DURATION_S:
            # Still turning
            self._pivot_corner_turn(self.turn_to_execute, 0)
        else:
            # Turn finished
            print("Turn complete.")
            self._stop_motors()
            self.position_tracker.update_direction_after_turn(self.turn_to_execute)
            self.turn_to_execute = None
            self.state = 'path_following'
            self.last_turn_complete_time = time.time() # Start cooldown
            time.sleep(0.5) # Pause to stabilize before next move

    def _run_state_machine(self):
        """Run the robot's state machine."""
        if self.state == "idle":
            self._stop_motors()
        elif self.state == "planning":
            self._plan_path_to_target()
        elif self.state == "path_following":
            self._follow_path()
        elif self.state == "turning":
            self._execute_intersection_turn()
        elif self.state == "mission_complete":
            self._stop_motors()
        elif self.state == "error":
            self._stop_motors()
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
        
        self.motor_controller.send_motor_speeds(left_speed, right_speed, left_speed, right_speed)
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
        
        self.motor_controller.send_motor_speeds(fl_speed, fr_speed, bl_speed, br_speed)
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
        
        self.motor_controller.send_motor_speeds(fl_speed, fr_speed, bl_speed, br_speed)
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
        
        self.motor_controller.send_motor_speeds(fl_speed, fr_speed, bl_speed, br_speed)
        return True

    def _recover_line(self):
        """Basic line recovery: stop for now."""
        self._stop_motors()

    def _stop_motors(self):
        """Stop all motors."""
        self.motor_controller.send_motor_speeds(0, 0, 0, 0)
    
    def stop(self):
        """Stop the robot and clean up resources."""
        print("Stopping robot...")
        self.running = False
        self._stop_motors()
        if isinstance(self.position_tracker, PreciseMazeLocalizer):
            self.position_tracker.stop_localization()
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
    
    # --- Direction Mapping ---
    # Translate intuitive directions to the system's cardinal directions
    direction_map = {
        'F': 'N',  # Forward -> North
        'B': 'S',  # Backward -> South
        'L': 'W',  # Left -> West
        'R': 'E'   # Right -> East
    }
    # Also allow cardinal directions to be used directly
    direction_map.update({ 'N': 'N', 'S': 'S', 'E': 'E', 'W': 'W' })

    # Get the system-compatible direction
    try:
        system_start_direction = direction_map[START_DIRECTION.upper()]
    except KeyError:
        print(f"ERROR: Invalid START_DIRECTION '{START_DIRECTION}'. Using 'N' as default.")
        system_start_direction = 'N'
    
    robot = RobotController()
    # Manually set the start direction for the localizer
    robot.position_tracker.current_direction = system_start_direction
    
    app = Flask(__name__, template_folder='../../templates', static_folder='../../static')
    
    @app.route('/')
    def index():
        return render_template('navigation.html')

    @app.route('/api/robot_data')
    def robot_data():
        """Provide robot data to the web UI."""
        x, y, heading_rad = robot.position_tracker.get_pose()
        heading_deg = math.degrees(heading_rad)
        
        # Get visual localizer status
        localizer_status = robot.position_tracker.get_status()

        data = {
            'state': robot.state,
            'x': x,
            'y': y,
            'heading': heading_deg,
            'line_position': -1,
            'line_error': 0,
            'line_sensors': [0,0,0],
            'visual_localizer': {
                'status': localizer_status.get('status', 'N/A'),
                'confidence': localizer_status.get('confidence', 0),
                'position': localizer_status.get('current_position', (0,0)),
                'direction': localizer_status.get('current_direction', 'N/A'),
                'scene_type': localizer_status.get('scene_type', 'unknown'),
            },
            'motors': {'fl': 0, 'fr': 0, 'bl': 0, 'br': 0},
            'path': robot.path,
            'current_target_index': robot.current_target_index,
            'camera_image': None 
        }
        return jsonify(data)

    @app.route('/video_feed')
    def video_feed():
        """Video streaming route. Returns a placeholder since camera is disabled."""
        if not FEATURES['VISION_SYSTEM_ENABLED']:
            return Response(status=204) # No content

        def generate_frames():
            while robot.running:
                frame = None
                if isinstance(robot.position_tracker, PreciseMazeLocalizer):
                    frame = robot.position_tracker.get_camera_frame()

                if frame is None:
                    # If no frame, send a placeholder or just wait
                    time.sleep(0.1)
                    continue
                
                # Add camera line following debug overlay if available
                if (hasattr(robot, 'camera_line_result') and 
                    robot.camera_line_result.get('processed_frame') is not None):
                    frame = robot.camera_line_result['processed_frame']
                
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
    
    # Start the camera capture thread if vision is enabled
    if FEATURES['VISION_SYSTEM_ENABLED']:
        print("Vision system enabled. Camera is managed by PreciseMazeLocalizer.")

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
