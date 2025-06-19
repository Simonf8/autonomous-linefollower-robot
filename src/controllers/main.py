#!/usr/bin/env python3

import time
import logging
import cv2
import numpy as np
import math
import socket
from typing import List, Tuple, Optional

# Import our clean modules
from object_detection import ObjectDetector, PathShapeDetector
from pathfinder import Pathfinder
from box import BoxHandler
from position_tracker import OmniWheelOdometry, PositionTracker
from pid import LineFollowPID

# ================================
# FEATURE CONFIGURATION
# ================================
# Enable/disable features for easy testing and debugging
FEATURES = {
    'OBJECT_DETECTION_ENABLED': True,       # Enable YOLO object detection
    'PATH_SHAPE_DETECTION_ENABLED': True,   # Enable path shape analysis
    'OBSTACLE_AVOIDANCE_ENABLED': False,     # Enable obstacle avoidance behavior
    'VISION_SYSTEM_ENABLED': False,          # Enable camera and vision processing
    'POSITION_CORRECTION_ENABLED': False,    # Enable waypoint position corrections
    'PERFORMANCE_LOGGING_ENABLED': True,    # Enable detailed performance logging
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
TURN_SPEED = 40
CORNER_SPEED = 25  # Slower speed for smooth cornering

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
    
    def __init__(self, ip: str, port: int = 1234):
        self.ip = ip
        self.port = port
        self.socket = None
        self.connected = False
        self.connection_attempts = 0
        
        # Latest sensor data from ESP32
        self.latest_encoder_data = [0, 0, 0, 0]
        self.latest_line_position = -1  # -1 means no line detected
        self.latest_line_error = 0
        self.latest_sensor_values = [0, 0, 0, 0, 0]
        
        # Command tracking
        self.last_command = None
        self.last_send_time = 0.0
        
    def start(self):
        """Start communication with ESP32."""
        return self.connect()
        
    def connect(self):
        """Establish connection to ESP32."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        try:
            import socket
            self.socket = socket.create_connection((self.ip, self.port), timeout=3)
            self.socket.settimeout(0.5)
            self.connected = True
            self.connection_attempts = 0
            print(f"Connected to ESP32 at {self.ip}:{self.port}")
            return True
        except Exception as e:
            self.connected = False
            self.connection_attempts += 1
            if self.connection_attempts % 10 == 1:  # Log every 10 attempts
                print(f"Failed to connect to ESP32 (attempt {self.connection_attempts}): {e}")
            self.socket = None
            return False
        
    def send_motor_speeds(self, fl: int, fr: int, bl: int, br: int):
        """Send motor speeds to ESP32."""
        if not self.connected and not self.connect():
            return False
            
        try:
            command = f"{fl},{fr},{bl},{br}"
            return self._send_command(command)
        except Exception as e:
            print(f"Error sending motor speeds: {e}")
            self.connected = False
            return False
    
    def send_line_follow_command(self, base_speed: int = 60):
        """Send line following command to ESP32."""
        if not self.connected and not self.connect():
            return False
            
        try:
            command = f"LINE_FOLLOW,{base_speed}"
            return self._send_command(command)
        except Exception as e:
            print(f"Error sending line follow command: {e}")
            self.connected = False
            return False
    
    def send_calibrate_command(self):
        """Send calibration command to ESP32."""
        if not self.connected and not self.connect():
            return False
            
        try:
            return self._send_command("CALIBRATE")
        except Exception as e:
            print(f"Error sending calibrate command: {e}")
            self.connected = False
            return False
    
    def _send_command(self, command: str):
        """Internal method to send command to ESP32."""
        if not self.socket:
            return False
            
        try:
            full_command = f"{command}\n"
            current_time = time.time()
            
            # Send commands to ESP32
            self.socket.sendall(full_command.encode())
            self.last_command = full_command
            self.last_send_time = current_time
            
            if FEATURES['PERFORMANCE_LOGGING_ENABLED']:
                print(f"Sent to ESP32: {command}")
            
            return True
        except Exception as e:
            print(f"Socket error: {e}")
            self.connected = False
            self.socket = None
            return False
            
    def get_encoder_ticks(self) -> List[int]:
        """Get encoder ticks from ESP32."""
        self._receive_data()  # Try to get fresh data
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
        self._receive_data()  # Try to get fresh data
        return (self.latest_line_position, self.latest_line_error, self.latest_sensor_values)
    
    def is_line_detected(self) -> bool:
        """Check if line is currently detected."""
        self._receive_data()  # Try to get fresh data
        return self.latest_line_position != -1
    
    def _receive_data(self):
        """Try to receive data from ESP32 (non-blocking)."""
        if not self.socket or not self.connected:
            return
            
        try:
            # Non-blocking receive
            data = self.socket.recv(1024)
            if data:
                data_string = data.decode().strip()
                for line in data_string.split('\n'):
                    if line:
                        self.update_sensor_data(line)
        except Exception:
            # No data available or connection error - that's okay for non-blocking
            pass
    
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
                
                # Show sensor readings in your format [Left Center Right]
                if len(self.latest_sensor_values) >= 3:
                    left = 1 if self.latest_sensor_values[0] < 500 else 0
                    center = 1 if self.latest_sensor_values[2] < 500 else 0  
                    right = 1 if self.latest_sensor_values[4] < 500 else 0
                    print(f"[{left} {center} {right}]")
        except (ValueError, IndexError):
            # Invalid data format - keep previous values
            if FEATURES['PERFORMANCE_LOGGING_ENABLED']:
                print(f"Invalid data format: {data_string}")
        
    def stop(self):
        """Stop ESP32 communication."""
        if self.socket:
            try:
                self._send_command("STOP")
                time.sleep(0.1)
                self.socket.close()
            except:
                pass
            self.socket = None
        self.connected = False



class RobotController:
    """Main robot controller using modular components."""
    
    def __init__(self):
        """Initialize robot controller with all modules."""
        # Hardware interfaces
        self.esp32_bridge = ESP32Bridge(ESP32_IP, 1234)
        
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
        self.pathfinder = Pathfinder([], CELL_WIDTH_M)  # Initialize with empty grid first
        maze_grid = self.pathfinder.create_maze_grid()  # Get the maze grid
        self.pathfinder = Pathfinder(maze_grid, CELL_WIDTH_M)  # Reinitialize with actual grid
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
    

    
    def _setup_vision(self):
        """Initialize camera and vision systems based on feature flags."""
        if not FEATURES['VISION_SYSTEM_ENABLED']:
            self.camera = None
            self.object_detector = None
            self.path_detector = None
            return
            
        try:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Initialize object detector if enabled
            if FEATURES['OBJECT_DETECTION_ENABLED']:
                self.object_detector = ObjectDetector('yolo11n.pt', confidence_threshold=0.5)
            else:
                self.object_detector = None
            
            # Initialize path shape detector if enabled
            if FEATURES['PATH_SHAPE_DETECTION_ENABLED']:
                self.path_detector = PathShapeDetector(IMG_PATH_SRC_PTS, IMG_PATH_DST_PTS)
            else:
                self.path_detector = None
            
        except Exception as e:
            if FEATURES['PERFORMANCE_LOGGING_ENABLED']:
                print(f"Vision system initialization failed: {e}")
            self.camera = None
            self.object_detector = None
            self.path_detector = None
    
    def run(self):
        """Main control loop."""
        self.esp32_bridge.start()
        self.box_handler.start_mission(silent=True)
        
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
        if not FEATURES['VISION_SYSTEM_ENABLED'] or self.camera is None:
            return
            
        ret, frame = self.camera.read()
        if ret:
            self.latest_frame = frame.copy()
            
            # Object detection
            if FEATURES['OBJECT_DETECTION_ENABLED'] and self.object_detector:
                detections = self.object_detector.detect_objects(frame)
                self._handle_detections(detections)
            
            # Path shape detection
            if FEATURES['PATH_SHAPE_DETECTION_ENABLED'] and self.path_detector:
                path_shape = self.path_detector.detect_path_shape(frame)
                self._handle_path_shape(path_shape)
            
            # Debug visualization
            if FEATURES['DEBUG_VISUALIZATION_ENABLED']:
                cv2.imshow('Robot Vision', frame)
                cv2.waitKey(1)
    
    def _handle_detections(self, detections: dict):
        """Handle object detection results."""
        if FEATURES['OBSTACLE_AVOIDANCE_ENABLED'] and detections['obstacle_detected']:
            if self.state not in ["AVOIDING_OBSTACLE"]:
                self.state = "AVOIDING_OBSTACLE"
                if FEATURES['PERFORMANCE_LOGGING_ENABLED']:
                    print("Obstacle detected - switching to avoidance mode")
    
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
        print("Starting robot mission...")
        
        # Calibrate sensors if needed
        if not self.esp32_bridge.connected:
            print("Connecting to ESP32...")
            if not self.esp32_bridge.connect():
                print("Warning: ESP32 not connected - running in simulation mode")
        else:
            print("ESP32 connected - starting real mission")
        
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
            # Correct odometry at waypoint if enabled
            if FEATURES['POSITION_CORRECTION_ENABLED']:
                self.position_tracker.correct_at_waypoint(current_waypoint)
            
            # Move to next waypoint
            self.current_waypoint_idx += 1
            
            # Check for intersection turn if needed
            if self._is_intersection():
                self._handle_intersection()
                return
        
        # Follow line using enhanced PID control with smooth cornering
        if self.esp32_bridge.is_line_detected():
            line_position, _, _ = self.esp32_bridge.get_line_sensor_data()
            # Convert ESP32 line position (0-4000) to our expected range (-1.0 to 1.0)
            normalized_position = (line_position - 2000) / 2000.0
            
            # Use the new, simplified PID controller
            vx, vy, omega = self.line_follower.calculate_control(normalized_position, base_speed=LINE_FOLLOW_SPEED)
            
            self._move_omni(vx, vy, omega)
            
            # Performance logging
            if FEATURES['PERFORMANCE_LOGGING_ENABLED']:
                stats = self.position_tracker.get_tracking_statistics()
                if stats['status']['is_strafing']:
                    print(f"Strafing: pos={normalized_position:.2f}, strafe_eff={stats['performance']['strafe_efficiency']:.2f}")
        else:
            # Line lost - stop and search
            self._stop_motors()
    
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
        self.box_handler.print_mission_summary(silent=True)
    
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
        
        # Convert to integers for the ESP32
        int_speeds = [int(s) for s in speeds]
        
        self.esp32_bridge.send_motor_speeds(*int_speeds)
    
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
    """Main entry point."""
    # Print feature status
    print_feature_status()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    controller = RobotController()
    
    try:
        print("Robot controller initialized successfully")
        print("Starting main control loop...")
        controller.run()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        controller.stop()
        print("Robot controller stopped")

if __name__ == "__main__":
    main() 