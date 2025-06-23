import time
import numpy as np
import threading
import math

from .hardware import ESP32Bridge
from .perception import Perception
from .navigation import Navigator
from .kinematics import MecanumKinematics
from .state_estimator import StateEstimator

class Robot:
    """The main class representing the robot and its control systems."""

    def __init__(self, config: dict):
        self.config = config
        self.state = 'IDLE'
        self.running = True
        self.frame = None
        self.frame_lock = threading.Lock()
        self.loop_rate = 50 # Hz
        self.dt = 1.0 / self.loop_rate

        # Core components
        self.esp32 = ESP32Bridge(ip=config['ESP32_IP'])
        self.perception = Perception(config)
        self.navigator = Navigator(config)
        self.kinematics = MecanumKinematics(
            wheel_radius_m=config['WHEEL_RADIUS_M'],
            robot_width_m=config['ROBOT_WIDTH_M'],
            robot_length_m=config['ROBOT_LENGTH_M']
        )

        # State, Pose, and Estimation
        initial_pose = (
            config['START_CELL'][0] * config['CELL_SIZE_M'],
            config['START_CELL'][1] * config['CELL_SIZE_M'],
            config['START_HEADING']
        )
        self.estimator = StateEstimator(self.dt, initial_pose)
        self.pose = self.estimator.pose
        self.last_control_input = np.zeros(3) # [vx, vy, v_theta]
        
        # Calculate the maximum wheel speed in rad/s from RPM for scaling motor commands
        max_rpm = self.config.get('MOTOR_MAX_RPM', 200)
        self.max_wheel_rad_per_s = (max_rpm / 60.0) * 2 * math.pi
        
    def start_mission(self):
        """Starts the autonomous mission."""
        print("Attempting to connect to ESP32...")
        self.esp32.connect()
        if not self.esp32.connected:
            print("CRITICAL: ESP32 connection failed. Mission aborted.")
            self.state = 'ERROR'
            return

        print("Planning initial path...")
        start_world = (self.pose[0], self.pose[1])
        end_world = (
            self.config['END_CELL'][0] * self.config['CELL_SIZE_M'],
            self.config['END_CELL'][1] * self.config['CELL_SIZE_M']
        )
        
        # For now, we assume a static, known grid.
        # In a real scenario, this grid would come from a mapping phase.
        grid = self.perception.get_occupancy_grid()

        if self.navigator.find_path(start_world, end_world, grid):
            self.state = 'FOLLOW_PATH'
            print("Mission started: Following path.")
        else:
            print("ERROR: Could not find a path to the destination.")
            self.state = 'ERROR'

    def run_main_loop(self):
        """The main execution loop for the robot."""
        while self.running:
            start_time = time.time()
            
            # Perception, State Estimation, and Control
            self._update_state_and_control()
            
            # Maintain loop rate
            time_elapsed = time.time() - start_time
            sleep_time = max(0, self.dt - time_elapsed)
            time.sleep(sleep_time)

    def _update_state_and_control(self):
        """
        The core logic block for a single tick of the robot's operation.
        It performs prediction, measurement, update, and control.
        """
        # 1. Predict new state based on last control input
        self.estimator.predict(self.last_control_input)
        
        # 2. Get new measurement from vision
        vision_pose = None
        if self.frame is not None:
            with self.frame_lock:
                frame_copy = self.frame.copy()
            # In the future, the grid could also be updated here
            vision_pose, _ = self.perception.estimate_pose_from_grid(frame_copy)

        # 3. Update state estimate with the new measurement
        if vision_pose:
            self.estimator.update(vision_pose)

        # 4. Update the robot's official pose from the estimator
        self.pose = self.estimator.pose
        
        # 5. Execute controller based on current state
        if self.state == 'FOLLOW_PATH':
            self._follow_path_controller()
            if self.navigator.is_mission_complete(self.pose):
                print("Mission Complete!")
                self.stop_robot()
                self.state = 'IDLE'
        
        elif self.state == 'IDLE' or self.state == 'ERROR':
            # Ensure motors are stopped and reset control input
            self.stop_robot()
            self.last_control_input = np.zeros(3)

    def _follow_path_controller(self):
        """
        Uses the navigator and kinematics to calculate and send motor commands.
        """
        if not self.esp32.connected:
            return
            
        # Get target velocity from the pure pursuit controller
        vx, vy, v_theta = self.navigator.pure_pursuit_controller(self.pose)
        
        # Store this control input for the next prediction cycle
        self.last_control_input = np.array([vx, vy, v_theta])

        # Get required wheel angular velocities from inverse kinematics
        wheel_rad_velocities = self.kinematics.get_wheel_speeds(vx, vy, v_theta)
        
        # Scale wheel velocities to the motor command range [-255, 255]
        motor_commands = self._scale_wheel_speeds_to_motor_commands(wheel_rad_velocities)
        fl, fr, bl, br = motor_commands

        self.esp32.send_motor_commands(fl, fr, bl, br)

    def _scale_wheel_speeds_to_motor_commands(self, wheel_rads: np.ndarray) -> tuple:
        """
        Scales wheel angular velocities (rad/s) to integer motor commands.
        """
        scaled_speeds = (wheel_rads / self.max_wheel_rad_per_s) * 255
        # Clamp the values to the -255 to 255 range and convert to int
        fl = int(np.clip(scaled_speeds[0], -255, 255))
        fr = int(np.clip(scaled_speeds[1], -255, 255))
        bl = int(np.clip(scaled_speeds[2], -255, 255))
        br = int(np.clip(scaled_speeds[3], -255, 255))
        return (fl, fr, bl, br)

    def stop_robot(self):
        """Stops the robot's movement."""
        print("Stopping robot.")
        if self.esp32.connected:
            self.esp32.stop()

    def shutdown(self):
        """Gracefully shuts down the robot and its components."""
        self.running = False
        self.stop_robot()
        print("Robot has been shut down.") 