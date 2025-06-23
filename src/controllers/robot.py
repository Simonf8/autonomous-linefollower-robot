import time
import math
import threading

from hardware import ESP32Bridge
from perception import Perception
from navigation import Navigator

class Robot:
    """The main robot class, orchestrating all components."""

    def __init__(self, config):
        self.config = config
        self.running = True
        self.state = "idle"
        
        # Components
        self.esp32 = ESP32Bridge(config['ESP32_IP'])
        self.perception = Perception(config['FEATURES'])
        self.navigator = Navigator(
            cell_size_m=config['CELL_SIZE_M'],
            start_cell=config['START_CELL'],
            end_cell=config['END_CELL'],
            camera_offset=config['CAMERA_FORWARD_OFFSET_M']
        )
        
        # Robot State - now using a precise pose
        start_x = (config['START_CELL'][0] + 0.5) * config['CELL_SIZE_M']
        start_y = (config['START_CELL'][1] + 0.5) * config['CELL_SIZE_M']
        self.pose = (start_x, start_y, config['START_HEADING']) # (x, y, heading) in meters and radians
        
        # Vision
        self.frame = None
        self.processed_frame = None
        self.frame_lock = threading.Lock()
        
        # Recovery
        self.recovery_state = "idle"
        self.recovery_start_time = 0
        self.last_known_line_offset = 0.0
        self.last_recovery_maneuver = ""
        
    def start(self):
        """Start all robot components and threads."""
        if not self.esp32.start():
            print("WARNING: ESP32 not connected. Running in simulation mode.")
        
        # In a real scenario, vision and other threads would be started here.
        # For this structure, the main loop is in the Flask thread.
        self.start_mission()

    def run_main_loop(self):
        """The main control loop to be called periodically."""
        while self.running:
            self._run_state_machine()
            time.sleep(0.01)
        self.stop()
    
    def start_mission(self):
        if self.state != "idle":
            print("Cannot start mission, robot is not idle.")
            return

        print("Starting mission...")
        self.current_cell = self.config['START_CELL']
        self.estimated_heading = self.config['START_HEADING']
        self.state = "planning"

    def _run_state_machine(self):
        # --- State Transitions and Actions ---
        if self.state == "idle":
            self._stop_motors()

        elif self.state == "planning":
            current_cell = (
                int(self.pose[0] / self.config['CELL_SIZE_M']),
                int(self.pose[1] / self.config['CELL_SIZE_M'])
            )
            if self.navigator.plan_path(current_cell):
                self.state = "path_following"
            else:
                self.state = "error"

        elif self.state == "path_following":
            self._handle_path_following()

        elif self.state == "replanning":
            print("State: Replanning due to obstacle.")
            self._stop_motors()
            self.state = "planning"

        elif self.state == "recovering_line":
            self._execute_line_recovery()

        elif self.state == "mission_complete" or self.state == "error":
            self._stop_motors()
            if self.state == "error":
                self.running = False

    def _handle_path_following(self):
        if self.navigator.is_mission_complete():
            print("Mission complete!")
            self.state = "mission_complete"
            self._stop_motors()
            return
        
        # Process vision frame for events and localization
        if self.frame is not None:
            # 1. Update pose from visual odometry
            current_cell = (
                int(self.pose[0] / self.config['CELL_SIZE_M']),
                int(self.pose[1] / self.config['CELL_SIZE_M'])
            )
            new_pose = self.perception.estimate_pose_from_grid(
                self.frame, 
                current_cell, 
                self.config['CELL_SIZE_M']
            )
            if new_pose:
                self.pose = new_pose

            # 2. Process for other events (obstacles, intersections)
            current_cell = (
                int(self.pose[0] / self.config['CELL_SIZE_M']),
                int(self.pose[1] / self.config['CELL_SIZE_M'])
            )
            processed_frame, event = self.perception.process_frame(self.frame, self.state, current_cell, self.pose[2])
            if event:
                if event['type'] == 'obstacle':
                    obstacle_cell_x = current_cell[0] + int(math.cos(self.pose[2]))
                    obstacle_cell_y = current_cell[1] + int(math.sin(self.pose[2]))
                    self.navigator.update_obstacle((obstacle_cell_x, obstacle_cell_y))
                    self.state = "replanning"
                    return
                elif event['type'] == 'intersection':
                    new_heading = self.navigator.advance_waypoint(current_cell)
                    if new_heading is not None:
                        # With odometry, we would blend this, but for now we just update heading
                        self.pose = (self.pose[0], self.pose[1], new_heading)

        # Follow line or smoothed path
        self._follow_path_controller()

    def _follow_path_controller(self):
        """Decides which controller to use for path following."""
        # Use Pure Pursuit if a smoothed path is available
        if self.navigator.smoothed_path is not None:
            turn_command, alpha = self.navigator.pure_pursuit_controller(self.pose)
            
            # Use adaptive speed, but also slow down for sharp turns (large alpha)
            current_speed = self.config['BASE_SPEED']
            if self.config['FEATURES']['ADAPTIVE_SPEED_ENABLED']:
                # Slow down based on the angle to the lookahead point
                speed_reduction_factor = (abs(alpha) / (math.pi / 2)) # Normalize angle error
                current_speed -= (self.config['BASE_SPEED'] - self.config['CORNER_SPEED']) * speed_reduction_factor
                current_speed = max(self.config['CORNER_SPEED'], current_speed)

            # --- Mecanum wheel kinematics ---
            # Combine forward speed and turn command to get wheel speeds
            # This is a simplified kinematic model.
            fl = int(current_speed - turn_command)
            fr = int(current_speed + turn_command)
            bl = int(current_speed - turn_command)
            br = int(current_speed + turn_command)
            
            self.esp32.send_motor_speeds(fl, fr, bl, br)

        else:
            # Fallback to camera-based line following if no smoothed path
            if not self.perception.latest_line_result or not self.perception.latest_line_result['line_detected']:
                print("Camera: Line lost! Initiating recovery.")
                self.state = "recovering_line"
                self.recovery_state = "start"
                self.last_known_line_offset = self.perception.latest_line_result.get('line_offset', 0.0) if self.perception.latest_line_result else 0.0
                self._stop_motors()
                return
            
            # Adaptive speed calculation for line following
            current_speed = self.config['LINE_FOLLOW_SPEED']
            if self.config['FEATURES']['ADAPTIVE_SPEED_ENABLED']:
                offset = self.perception.latest_line_result.get('line_offset', 0.0)
                speed_reduction = abs(offset) * (self.config['BASE_SPEED'] - self.config['CORNER_SPEED']) * 2.0
                current_speed = self.config['BASE_SPEED'] - speed_reduction
                current_speed = max(self.config['CORNER_SPEED'], min(current_speed, self.config['MAX_SPEED']))
                
            # Get motor speeds from perception component for line following
            fl, fr, bl, br = self.perception.get_line_following_speeds(int(current_speed))
            self.esp32.send_motor_speeds(fl, fr, bl, br)

    def _execute_line_recovery(self):
        # This logic remains largely the same but uses the components
        # Check if line is re-acquired
        if self.frame is not None:
            _, event = self.perception.process_frame(self.frame, self.state, self.current_cell, self.estimated_heading)
            if self.perception.latest_line_result['line_detected'] and self.perception.latest_line_result['confidence'] > 0.3:
                print("Camera: Line re-acquired! Resuming path following.")
                self.state = "path_following"
                self.recovery_state = "idle"
                return

        # Recovery maneuvers
        if self.recovery_state == "start":
            self.recovery_start_time = time.time()
            self.recovery_state = "initial_swing_left" if self.last_known_line_offset <= 0 else "initial_swing_right"
        
        # Timeout and logic...
        # (This is kept brief for the refactoring step, can be expanded later)
        duration = time.time() - self.recovery_start_time
        if duration > 8.0:
            self.state = "error"
            return
            
        turn_speed = self.config['TURN_SPEED']
        if "left" in self.recovery_state:
             self.esp32.send_motor_speeds(-turn_speed, turn_speed, -turn_speed, turn_speed)
        else:
             self.esp32.send_motor_speeds(turn_speed, -turn_speed, turn_speed, -turn_speed)

    def _stop_motors(self):
        self.esp32.send_motor_speeds(0, 0, 0, 0)

    def stop(self):
        print("Stopping robot...")
        self.running = False
        self._stop_motors()
        self.esp32.stop() 