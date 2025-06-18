#!/usr/bin/env python3

import time
import logging
import cv2
import numpy as np
import math
from typing import List, Tuple, Optional

# Import our clean modules
from object_detection import ObjectDetector, PathShapeDetector
from pathfinder import Pathfinder
from box_handler import BoxHandler
from position_tracker import OmniWheelOdometry, PositionTracker
from pid_controller import LineFollowPID

# ================================
# FEATURE CONFIGURATION
# ================================
# Enable/disable features for easy testing and debugging
FEATURES = {
    'OBJECT_DETECTION_ENABLED': True,       # Enable YOLO object detection
    'PATH_SHAPE_DETECTION_ENABLED': True,   # Enable path shape analysis
    'OBSTACLE_AVOIDANCE_ENABLED': True,     # Enable obstacle avoidance behavior
    'VISION_SYSTEM_ENABLED': True,          # Enable camera and vision processing
    'POSITION_CORRECTION_ENABLED': True,    # Enable waypoint position corrections
    'PERFORMANCE_LOGGING_ENABLED': False,   # Enable detailed performance logging
    'DEBUG_VISUALIZATION_ENABLED': False,   # Enable debug visualization windows
    'SMOOTH_CORNERING_ENABLED': True,       # Enable smooth cornering like normal wheels
    'ADAPTIVE_SPEED_ENABLED': True,         # Enable speed adaptation based on conditions
}

# ================================
# ROBOT CONFIGURATION
# ================================
ESP32_IP = "192.168.128.245"
CELL_WIDTH_M = 0.025
BASE_SPEED = 60
TURN_SPEED = 80
CORNER_SPEED = 45  # Slower speed for smooth cornering

# Robot physical constants
PULSES_PER_REV = 960
WHEEL_DIAMETER_M = 0.025
ROBOT_WIDTH_M = 0.225
ROBOT_LENGTH_M = 0.075

# Mission configuration
START_CELL = (11, 2)
PICKUP_CELLS = [(20, 14), (18, 14), (16, 14), (14, 14)]
DROPOFF_CELLS = [(0, 0), (2, 0), (4, 0), (6, 0)]
START_POSITION = ((START_CELL[0] + 0.5) * CELL_WIDTH_M, (START_CELL[1] + 0.5) * CELL_WIDTH_M)
START_HEADING = math.pi  # Facing left

# Line following configuration
LINE_FOLLOW_SPEED = 60

# Vision configuration
IMG_PATH_SRC_PTS = np.float32([[200, 300], [440, 300], [580, 480], [60, 480]])
IMG_PATH_DST_PTS = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])

class ESP32Bridge:
    """ESP32 communication bridge for motors, encoders, and line sensors."""
    
    def __init__(self, ip: str):
        self.ip = ip
        self.connected = False
        # Latest sensor data from ESP32
        self.latest_encoder_data = [0, 0, 0, 0]
        self.latest_line_position = -1  # -1 means no line detected
        self.latest_line_error = 0
        self.latest_sensor_values = [0, 0, 0, 0, 0]
        # Placeholder for actual socket implementation
        
    def start(self):
        """Start communication with ESP32."""
        self.connected = True
        
    def send_motor_speeds(self, fl: int, fr: int, bl: int, br: int):
        """Send motor speeds to ESP32."""
        if self.connected:
            # Placeholder for actual motor command sending
            pass
    
    def send_line_follow_command(self, base_speed: int = 60):
        """Send line following command to ESP32."""
        if self.connected:
            # Placeholder - would send "LINE_FOLLOW,{base_speed}"
            pass
    
    def send_calibrate_command(self):
        """Send calibration command to ESP32."""
        if self.connected:
            # Placeholder - would send "CALIBRATE"
            pass
            
    def get_encoder_ticks(self) -> List[int]:
        """Get encoder ticks from ESP32."""
        return self.latest_encoder_data
    
    def get_line_sensor_data(self) -> Tuple[int, int, List[int]]:
        """
        Get line sensor data from ESP32.
        
        Returns:
            Tuple of (line_position, line_error, sensor_values)
            line_position: 0-4000 (0=leftmost, 2000=center, 4000=rightmost, -1=no line)
            line_error: -2000 to +2000 (error from center)
            sensor_values: List of 5 calibrated sensor readings (0-1000 each)
        """
        return (self.latest_line_position, self.latest_line_error, self.latest_sensor_values)
    
    def is_line_detected(self) -> bool:
        """Check if line is currently detected."""
        return self.latest_line_position != -1
    
    def update_sensor_data(self, data_string: str):
        """
        Update sensor data from received ESP32 data.
        Expected format: "encoder_fl,encoder_fr,encoder_bl,encoder_br,line_pos,line_error,s0,s1,s2,s3,s4"
        """
        try:
            parts = data_string.strip().split(',')
            if len(parts) >= 11:
                # Update encoder data
                self.latest_encoder_data = [int(parts[i]) for i in range(4)]
                
                # Update line sensor data
                self.latest_line_position = int(parts[4])
                self.latest_line_error = int(parts[5])
                self.latest_sensor_values = [int(parts[i]) for i in range(6, 11)]
        except (ValueError, IndexError):
            # Invalid data format - keep previous values
            pass
        
    def stop(self):
        """Stop ESP32 communication."""
        self.connected = False



class RobotController:
    """Main robot controller using modular components."""
    
    def __init__(self):
        """Initialize robot controller with all modules."""
        # Hardware interfaces
        self.esp32_bridge = ESP32Bridge(ESP32_IP)
        
        # Position tracking
        initial_pose = (START_POSITION[0], START_POSITION[1], START_HEADING)
        self.odometry = OmniWheelOdometry(
            initial_pose=initial_pose,
            pulses_per_rev=PULSES_PER_REV,
            wheel_diameter=WHEEL_DIAMETER_M,
            robot_width=ROBOT_WIDTH_M,
            robot_length=ROBOT_LENGTH_M
        )
        self.position_tracker = PositionTracker(self.odometry, CELL_WIDTH_M)
        
        # Navigation
        maze_grid = self._create_maze_grid()
        self.pathfinder = Pathfinder(maze_grid, CELL_WIDTH_M)
        self.current_path = None
        self.current_waypoint_idx = 0
        
        # Box handling
        self.box_handler = BoxHandler(PICKUP_CELLS, DROPOFF_CELLS)
        
        # Vision system
        self.camera = None
        self.object_detector = None
        self.path_detector = None
        self._setup_vision()
        
        # Control
        self.line_follower = LineFollowPID()
        self.state = "STARTING"
        self.latest_frame = None
    
    def _create_maze_grid(self) -> List[List[int]]:
        """Create the maze grid layout."""
        return self.pathfinder.create_maze_grid()
    
    def _setup_vision(self):
        """Initialize camera and vision systems."""
        try:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.object_detector = ObjectDetector('yolo11n.pt', confidence_threshold=0.5)
            self.path_detector = PathShapeDetector(IMG_PATH_SRC_PTS, IMG_PATH_DST_PTS)
            
            # Vision system initialized
        except Exception:
            self.camera = None
    
    def run(self):
        """Main control loop."""
        self.esp32_bridge.start()
        self.box_handler.start_mission()
        
        try:
            while True:
                # Update position tracking
                encoder_ticks = self.esp32_bridge.get_encoder_ticks()
                self.position_tracker.update(encoder_ticks)
                
                # Update ESP32 sensor data (this would normally come from network)
                # For now, we'll simulate receiving data
                # self.esp32_bridge.update_sensor_data(received_data_string)
                
                # Process vision
                self._process_vision()
                
                # State machine
                self._run_state_machine()
                
                time.sleep(0.05)  # 20Hz control loop
                
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def _process_vision(self):
        """Process camera frame for object detection and path analysis."""
        if self.camera is None:
            return
            
        ret, frame = self.camera.read()
        if ret:
            self.latest_frame = frame.copy()
            
            # Object detection
            if self.object_detector:
                detections = self.object_detector.detect_objects(frame)
                self._handle_detections(detections)
            
            # Path shape detection
            if self.path_detector:
                path_shape = self.path_detector.detect_path_shape(frame)
                self._handle_path_shape(path_shape)
    
    def _handle_detections(self, detections: dict):
        """Handle object detection results."""
        if detections['obstacle_detected']:
            if self.state not in ["AVOIDING_OBSTACLE"]:
                self.state = "AVOIDING_OBSTACLE"
    
    def _handle_path_shape(self, path_shape: str):
        """Handle path shape detection results."""
        # Path shape detection
    
    def _run_state_machine(self):
        """Main state machine logic."""
        if self.state == "STARTING":
            self._start_mission()
        elif self.state == "PLANNING_PATH":
            self._plan_path_to_target()
        elif self.state == "FOLLOWING_PATH":
            self._follow_path()
        elif self.state == "AT_PICKUP":
            self._handle_pickup()
        elif self.state == "AT_DROPOFF":
            self._handle_dropoff()
        elif self.state == "AVOIDING_OBSTACLE":
            self._handle_obstacle_avoidance()
        elif self.state == "MISSION_COMPLETE":
            self._handle_mission_complete()
    
    def _start_mission(self):
        """Initialize mission."""
        self.state = "PLANNING_PATH"
    
    def _plan_path_to_target(self):
        """Plan path to next target."""
        target_info = self.box_handler.get_current_target()
        
        if target_info is None:
            self.state = "MISSION_COMPLETE"
            return
        
        target_cell, mission_type = target_info
        current_cell = self.position_tracker.get_current_cell()
        
        self.current_path = self.pathfinder.find_path(current_cell, target_cell)
        
        if self.current_path:
            self.current_waypoint_idx = 0
            self.state = "FOLLOWING_PATH"
        else:
            self.state = "MISSION_COMPLETE"
    
    def _follow_path(self):
        """Follow the planned path using line following."""
        if not self.current_path or self.current_waypoint_idx >= len(self.current_path):
            # Reached end of path
            target_info = self.box_handler.get_current_target()
            if target_info:
                _, mission_type = target_info
                if mission_type == "PICKUP":
                    self.state = "AT_PICKUP"
                else:
                    self.state = "AT_DROPOFF"
            else:
                self.state = "MISSION_COMPLETE"
            return
        
        # Check if we're at current waypoint
        current_waypoint = self.current_path[self.current_waypoint_idx]
        if self.position_tracker.is_at_cell(current_waypoint[0], current_waypoint[1]):
            # Correct odometry at waypoint
            self.position_tracker.correct_at_waypoint(current_waypoint)
            
            # Move to next waypoint
            self.current_waypoint_idx += 1
            
            # Check for intersection turn if needed
            if self._is_intersection():
                self._handle_intersection()
                return
        
        # Follow line using PID control
        if self.esp32_bridge.is_line_detected():
            line_position, line_error, sensor_values = self.esp32_bridge.get_line_sensor_data()
            # Convert ESP32 line position (0-4000) to our expected range (-1.0 to 1.0)
            normalized_position = (line_position - 2000) / 2000.0
            is_corner = self._detect_corner(sensor_values)
            
            vx, vy, omega = self.line_follower.calculate_control(normalized_position, is_corner, LINE_FOLLOW_SPEED)
            self._move_omni(vx, vy, omega)
        else:
            # Line lost - stop and search
            self._stop_motors()
    
    def _detect_corner(self, sensor_values: List[int]) -> bool:
        """
        Detect if robot is approaching/on a corner using ESP32 sensor data.
        
        Args:
            sensor_values: List of 5 calibrated sensor readings (0-1000 each)
        """
        # Corner detection based on sensor pattern
        # If center sensors (indices 1,2,3) have low values, might be a corner
        center_sensors = sensor_values[1:4]  # Left-center, center, right-center
        return any(value < 300 for value in center_sensors)  # Low values indicate line
    
    def _is_intersection(self) -> bool:
        """Check if robot is at an intersection using ESP32 sensor data."""
        if not self.esp32_bridge.is_line_detected():
            return False
        
        _, _, sensor_values = self.esp32_bridge.get_line_sensor_data()
        # Intersection: multiple sensors detect line (low values)
        line_detections = sum(1 for value in sensor_values if value < 300)
        return line_detections >= 3  # 3 or more sensors detect line = intersection
    
    def _handle_intersection(self):
        """Handle intersection navigation."""
        if self.current_waypoint_idx < len(self.current_path):
            # Determine turn direction based on next waypoint
            turn_direction = self._get_turn_direction()
            self._execute_turn(turn_direction)
        
        self.line_follower.reset_controllers()
    
    def _get_turn_direction(self) -> str:
        """Calculate turn direction for next waypoint."""
        if self.current_waypoint_idx >= len(self.current_path):
            return "STRAIGHT"
        
        current_pos = self.position_tracker.odometry.get_position()
        current_heading = self.position_tracker.odometry.get_heading()
        
        next_waypoint = self.current_path[self.current_waypoint_idx]
        target_x = (next_waypoint[0] + 0.5) * CELL_WIDTH_M
        target_y = (next_waypoint[1] + 0.5) * CELL_WIDTH_M
        
        angle_to_target = math.atan2(target_y - current_pos[1], target_x - current_pos[0])
        heading_error = angle_to_target - current_heading
        
        # Normalize angle
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        if abs(heading_error) < math.pi / 4:
            return "STRAIGHT"
        elif heading_error > 0:
            return "LEFT"
        else:
            return "RIGHT"
    
    def _execute_turn(self, direction: str):
        """Execute turn at intersection."""
        # Executing turn
        
        if direction == "STRAIGHT":
            return
        
        # Simple turn execution
        omega = TURN_SPEED if direction == "LEFT" else -TURN_SPEED
        self._move_omni(BASE_SPEED * 0.3, 0, omega)
        
        # Wait for turn completion (simplified)
        time.sleep(0.5)
        self._stop_motors()
    
    def _handle_pickup(self):
        """Handle package pickup."""
        current_cell = self.position_tracker.get_current_cell()
        
        if self.box_handler.collect_package(current_cell):
            self.state = "PLANNING_PATH"  # Plan path to dropoff
        else:
            self.state = "MISSION_COMPLETE"
    
    def _handle_dropoff(self):
        """Handle package dropoff."""
        current_cell = self.position_tracker.get_current_cell()
        
        if self.box_handler.deliver_package(current_cell):
            if self.box_handler.is_mission_complete():
                self.state = "MISSION_COMPLETE"
            else:
                self.state = "PLANNING_PATH"  # Plan path to next pickup
        else:
            self.state = "MISSION_COMPLETE"
    
    def _handle_obstacle_avoidance(self):
        """Handle obstacle avoidance."""
        # Executing obstacle avoidance
        
        # Simple avoidance: stop, turn around, replan
        self._stop_motors()
        time.sleep(1.0)
        
        # Add obstacle to map
        current_cell = self.position_tracker.get_current_cell()
        # Estimate obstacle position
        robot_pos = self.position_tracker.odometry.get_position()
        robot_heading = self.position_tracker.odometry.get_heading()
        obstacle_x = robot_pos[0] + 0.2 * math.cos(robot_heading)
        obstacle_y = robot_pos[1] + 0.2 * math.sin(robot_heading)
        obstacle_cell = self.pathfinder.world_to_cell(obstacle_x, obstacle_y)
        
        self.pathfinder.update_obstacle(obstacle_cell[0], obstacle_cell[1], True)
        
        # Replan path
        self.state = "PLANNING_PATH"
    
    def _handle_mission_complete(self):
        """Handle mission completion."""
        self._stop_motors()
        self.box_handler.print_mission_summary()
    
    def _move_omni(self, vx: float, vy: float, omega: float):
        """Move robot using omni-wheel kinematics."""
        R = ROBOT_WIDTH_M / 2
        
        # Inverse kinematics
        v_fl = (vx - vy - R * omega)
        v_fr = (vx + vy + R * omega)
        v_bl = (vx + vy - R * omega)
        v_br = (vx - vy + R * omega)
        
        # Scale to motor range
        speeds = [v_fl, v_fr, v_bl, v_br]
        max_speed = max(abs(s) for s in speeds)
        if max_speed > 100:
            scale = 100 / max_speed
            speeds = [s * scale for s in speeds]
        
        self.esp32_bridge.send_motor_speeds(*speeds)
    
    def _stop_motors(self):
        """Stop all motors."""
        self.esp32_bridge.send_motor_speeds(0, 0, 0, 0)
    
    def stop(self):
        """Stop robot and cleanup resources."""
        self._stop_motors()
        self.esp32_bridge.stop()
        
        if self.camera:
            self.camera.release()
        
        cv2.destroyAllWindows()

def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    controller = RobotController()
    
    try:
        controller.run()
    except Exception:
        pass
    finally:
        controller.stop()

if __name__ == "__main__":
    main() 