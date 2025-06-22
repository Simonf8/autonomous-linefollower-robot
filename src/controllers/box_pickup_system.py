import time
import math
import logging
from typing import List, Tuple, Dict, Optional, Callable
from enum import Enum
from dataclasses import dataclass


class MissionPhase(Enum):
    """Mission execution phases."""
    IDLE = "idle"
    INITIALIZATION = "initialization"
    PICKUP = "pickup"
    TRANSPORT = "transport"
    DELIVERY = "delivery"
    COMPLETE = "complete"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class RobotConfig:
    """Robot hardware configuration parameters."""
    # Servo pins
    arm_servo_pin: int = 9
    delivery_servo_pin: int = 11
    electromagnet_pin: int = 10
    
    # Servo angles (degrees)
    arm_neutral: float = 90.0
    arm_pickup: float = 90.0
    arm_secured: float = 270.0  # After 180° rotation
    delivery_retracted: float = 0.0
    delivery_extended: float = 180.0
    
    # Movement parameters
    pickup_speed: int = 30
    sideways_speed: int = 40
    delivery_speed: int = 25
    
    # Distances (meters)
    backup_distance: float = 0.15
    sideways_distance: float = 0.22  # 2 grid cells
    delivery_approach: float = 0.10
    
    # Timing (seconds)
    arm_rotation_time: float = 2.0
    electromagnet_hold_time: float = 0.5
    delivery_push_time: float = 1.5
    settle_time: float = 0.5


@dataclass
class MissionStatus:
    """Current mission status and progress."""
    phase: MissionPhase
    boxes_collected: int
    boxes_delivered: int
    current_pickup_index: int
    current_delivery_index: int
    electromagnet_active: bool
    error_message: Optional[str] = None


class BoxPickupSystem:
    """
    Advanced box pickup and delivery system for omni-wheel robots.
    
    Features:
    - Electromagnetic arm with 180° rotation capability
    - Precision servo-controlled delivery mechanism
    - Strategic sideways movement between collection points
    - Backwards delivery approach for optimal box placement
    - Comprehensive error handling and status monitoring
    """
    
    # Mission locations (grid coordinates)
    PICKUP_LOCATIONS: List[Tuple[int, int]] = [
        (20, 14), (18, 14), (16, 14), (14, 14)
    ]
    
    DELIVERY_LOCATIONS: List[Tuple[int, int]] = [
        (0, 0), (2, 0), (4, 0), (6, 0)
    ]
    
    def __init__(self, esp32_bridge, position_tracker, config: Optional[RobotConfig] = None):
        """
        Initialize the box pickup system.
        
        Args:
            esp32_bridge: Hardware communication interface
            position_tracker: Robot position tracking system
            config: Hardware configuration parameters
        """
        self.esp32 = esp32_bridge
        self.position_tracker = position_tracker
        self.config = config or RobotConfig()
        
        self.status = MissionStatus(
            phase=MissionPhase.IDLE,
            boxes_collected=0,
            boxes_delivered=0,
            current_pickup_index=0,
            current_delivery_index=0,
            electromagnet_active=False
        )
        
        # Navigation callback for integration with main pathfinding
        self.navigation_callback: Optional[Callable] = None
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info("Box pickup system initialized")
    
    def _setup_logging(self) -> None:
        """Configure logging for the pickup system."""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def set_navigation_callback(self, callback: Callable) -> None:
        """Set callback for main navigation system integration."""
        self.navigation_callback = callback
        self.logger.info("Navigation callback registered")
    
    # Public API Methods
    
    def start_mission(self) -> bool:
        """
        Start the complete box pickup and delivery mission.
        
        Returns:
            bool: True if mission started successfully, False otherwise
        """
        self.logger.info("Starting box pickup mission")
        
        if not self._initialize_hardware():
            return False
        
        self.status.phase = MissionPhase.PICKUP
        self._reset_counters()
        
        return True
    
    def execute_pickup_phase(self) -> bool:
        """
        Execute the complete pickup sequence for all boxes.
        
        Returns:
            bool: True if all boxes collected successfully
        """
        if self.status.phase != MissionPhase.PICKUP:
            self.logger.error("Not in pickup phase")
            return False
        
        self.logger.info(f"Starting pickup sequence for {len(self.PICKUP_LOCATIONS)} boxes")
        
        for i, location in enumerate(self.PICKUP_LOCATIONS):
            self.status.current_pickup_index = i
            self.logger.info(f"Collecting box {i+1}/{len(self.PICKUP_LOCATIONS)} at {location}")
            
            if not self._execute_single_pickup(location, i):
                self._handle_error(f"Failed to pickup box at {location}")
                return False
            
            self.status.boxes_collected += 1
            self.logger.info(f"Box {i+1} collected successfully")
            
            # Move to next pickup location (except for last box)
            if i < len(self.PICKUP_LOCATIONS) - 1:
                if not self._move_to_next_pickup():
                    self._handle_error("Failed to move to next pickup location")
                    return False
        
        self.logger.info("All boxes collected successfully")
        self.status.phase = MissionPhase.TRANSPORT
        return True
    
    def execute_delivery_phase(self) -> bool:
        """
        Execute the complete delivery sequence for all boxes.
        
        Returns:
            bool: True if all boxes delivered successfully
        """
        if self.status.phase != MissionPhase.TRANSPORT:
            self.logger.error("Not in transport phase")
            return False
        
        self.logger.info("Starting delivery sequence")
        self.status.phase = MissionPhase.DELIVERY
        
        if not self._navigate_to_delivery_area():
            self._handle_error("Failed to navigate to delivery area")
            return False
        
        for i, location in enumerate(self.DELIVERY_LOCATIONS):
            self.status.current_delivery_index = i
            self.logger.info(f"Delivering box {i+1}/{len(self.DELIVERY_LOCATIONS)} to {location}")
            
            if not self._execute_single_delivery(location, i):
                self._handle_error(f"Failed to deliver box at {location}")
                return False
            
            self.status.boxes_delivered += 1
            self.logger.info(f"Box {i+1} delivered successfully")
            
            # Move to next delivery location (except for last box)
            if i < len(self.DELIVERY_LOCATIONS) - 1:
                if not self._move_to_next_delivery():
                    self._handle_error("Failed to move to next delivery location")
                    return False
        
        self.logger.info("All boxes delivered successfully")
        self.status.phase = MissionPhase.COMPLETE
        self._cleanup_mission()
        return True
    
    def get_status(self) -> MissionStatus:
        """Get current mission status."""
        return self.status
    
    def is_mission_complete(self) -> bool:
        """Check if mission is complete."""
        return (self.status.phase == MissionPhase.COMPLETE and 
                self.status.boxes_delivered == len(self.DELIVERY_LOCATIONS))
    
    def emergency_stop(self) -> None:
        """Execute emergency stop procedure."""
        self.logger.warning("EMERGENCY STOP ACTIVATED")
        self.status.phase = MissionPhase.EMERGENCY_STOP
        
        # Stop all movement
        self._stop_all_motors()
        
        # Safely disable electromagnet
        self._set_electromagnet(False)
        
        # Return servos to safe positions
        self._set_arm_angle(self.config.arm_neutral)
        self._set_delivery_servo(self.config.delivery_retracted)
    
    # Private Implementation Methods
    
    def _initialize_hardware(self) -> bool:
        """Initialize and test all hardware components."""
        self.logger.info("Initializing hardware components")
        self.status.phase = MissionPhase.INITIALIZATION
        
        try:
            # Initialize arm servo
            self._set_arm_angle(self.config.arm_neutral)
            time.sleep(1.0)
            
            # Initialize electromagnet (off)
            self._set_electromagnet(False)
            
            # Initialize delivery servo
            self._set_delivery_servo(self.config.delivery_retracted)
            time.sleep(0.5)
            
            self.logger.info("Hardware initialization complete")
            return True
            
        except Exception as e:
            self._handle_error(f"Hardware initialization failed: {e}")
            return False
    
    def _execute_single_pickup(self, location: Tuple[int, int], index: int) -> bool:
        """Execute pickup procedure for a single box."""
        try:
            # Navigate to pickup location
            if not self._navigate_to_pickup(location):
                return False
            
            # Lower arm for pickup
            self._set_arm_angle(self.config.arm_pickup)
            time.sleep(self.config.settle_time)
            
            # Activate electromagnet
            self._set_electromagnet(True)
            time.sleep(self.config.electromagnet_hold_time)
            
            # Rotate arm 180° to secure box
            self._rotate_arm_to_secured_position()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pickup failed at {location}: {e}")
            return False
    
    def _execute_single_delivery(self, location: Tuple[int, int], index: int) -> bool:
        """Execute delivery procedure for a single box."""
        try:
            # Position for backwards delivery
            if not self._position_for_delivery(location):
                return False
            
            # Back up to delivery position
            self._move_backward(self.config.delivery_approach)
            
            # Release electromagnet
            self._set_electromagnet(False)
            time.sleep(0.5)
            
            # Push box out with servo
            self._push_box_with_servo()
            
            # Clear the delivery area
            self._move_forward(0.15)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Delivery failed at {location}: {e}")
            return False
    
    def _rotate_arm_to_secured_position(self) -> None:
        """Smoothly rotate arm 180° to secure box inside robot."""
        self.logger.debug("Rotating arm to secured position")
        
        start_angle = self.config.arm_pickup
        end_angle = self.config.arm_secured
        steps = 20
        
        angle_increment = (end_angle - start_angle) / steps
        time_increment = self.config.arm_rotation_time / steps
        
        for i in range(steps + 1):
            current_angle = start_angle + (angle_increment * i)
            self._set_arm_angle(current_angle)
            time.sleep(time_increment)
        
        self.logger.debug(f"Arm secured at {end_angle}°")
    
    def _push_box_with_servo(self) -> None:
        """Use delivery servo to push box out of robot."""
        self.logger.debug("Pushing box with delivery servo")
        
        # Extend servo to push
        self._set_delivery_servo(self.config.delivery_extended)
        time.sleep(self.config.delivery_push_time)
        
        # Retract servo
        self._set_delivery_servo(self.config.delivery_retracted)
        time.sleep(0.5)
    
    # Navigation Methods
    
    def _navigate_to_pickup(self, location: Tuple[int, int]) -> bool:
        """Navigate to specific pickup location."""
        target_x, target_y = self._grid_to_meters(location)
        self.logger.debug(f"Navigating to pickup at ({target_x:.2f}, {target_y:.2f})m")
        
        if self.navigation_callback:
            return self.navigation_callback(target_x, target_y)
        else:
            # Basic navigation fallback
            return self._basic_navigate_to(target_x, target_y)
    
    def _navigate_to_delivery_area(self) -> bool:
        """Navigate from pickup area to delivery area."""
        self.logger.info("Navigating to delivery area")
        
        # Back up from final pickup position
        self._move_backward(self.config.backup_distance * 2)
        time.sleep(self.config.settle_time)
        
        # Turn toward delivery area
        self._turn_right_90()
        time.sleep(self.config.settle_time)
        
        # Use navigation callback if available
        if self.navigation_callback:
            delivery_x, delivery_y = self._grid_to_meters(self.DELIVERY_LOCATIONS[0])
            return self.navigation_callback(delivery_x, delivery_y)
        
        return True
    
    def _position_for_delivery(self, location: Tuple[int, int]) -> bool:
        """Position robot for backwards delivery."""
        target_x, target_y = self._grid_to_meters(location)
        self.logger.debug(f"Positioning for delivery at ({target_x:.2f}, {target_y:.2f})m")
        
        # Approach delivery location
        if self.navigation_callback:
            if not self.navigation_callback(target_x, target_y):
                return False
        
        # Turn around for backwards delivery
        self._turn_around_180()
        time.sleep(self.config.settle_time)
        
        return True
    
    # Movement Methods
    
    def _move_forward(self, distance: float) -> None:
        """Move forward by specified distance."""
        move_time = self._calculate_move_time(distance, self.config.pickup_speed)
        self._send_motor_command(self.config.pickup_speed, self.config.pickup_speed,
                               self.config.pickup_speed, self.config.pickup_speed)
        time.sleep(move_time)
        self._stop_all_motors()
    
    def _move_backward(self, distance: float) -> None:
        """Move backward by specified distance."""
        move_time = self._calculate_move_time(distance, self.config.pickup_speed)
        speed = -self.config.pickup_speed
        self._send_motor_command(speed, speed, speed, speed)
        time.sleep(move_time)
        self._stop_all_motors()
    
    def _move_sideways_right(self, distance: float) -> None:
        """Move sideways to the right by specified distance."""
        move_time = self._calculate_move_time(distance, self.config.sideways_speed)
        speed = self.config.sideways_speed
        # Omni-wheel sideways: FL/BR forward, FR/BL backward
        self._send_motor_command(speed, -speed, -speed, speed)
        time.sleep(move_time)
        self._stop_all_motors()
    
    def _turn_right_90(self) -> None:
        """Turn right 90 degrees in place."""
        self._send_motor_command(40, -40, 40, -40)
        time.sleep(1.5)  # Calibrate for actual robot
        self._stop_all_motors()
    
    def _turn_around_180(self) -> None:
        """Turn around 180 degrees in place."""
        self._send_motor_command(40, -40, 40, -40)
        time.sleep(3.0)  # Calibrate for actual robot
        self._stop_all_motors()
    
    def _move_to_next_pickup(self) -> bool:
        """Move to next pickup location."""
        self.logger.debug("Moving to next pickup location")
        
        # Back up slightly
        self._move_backward(self.config.backup_distance)
        time.sleep(self.config.settle_time)
        
        # Move sideways to next position
        self._move_sideways_right(self.config.sideways_distance)
        time.sleep(self.config.settle_time)
        
        return True
    
    def _move_to_next_delivery(self) -> bool:
        """Move to next delivery location."""
        self.logger.debug("Moving to next delivery location")
        
        # Move sideways to next delivery position
        self._move_sideways_right(self.config.sideways_distance)
        time.sleep(self.config.settle_time)
        
        return True
    
    # Hardware Control Methods
    
    def _set_arm_angle(self, angle: float) -> None:
        """Set arm servo to specified angle."""
        command = f"SERVO,{self.config.arm_servo_pin},{int(angle)}"
        if self.esp32.connected:
            self.esp32._send_command(command)
        self.logger.debug(f"Arm servo set to {angle}°")
    
    def _set_electromagnet(self, active: bool) -> None:
        """Control electromagnet on/off state."""
        self.status.electromagnet_active = active
        state = 1 if active else 0
        command = f"DIGITAL,{self.config.electromagnet_pin},{state}"
        if self.esp32.connected:
            self.esp32._send_command(command)
        self.logger.debug(f"Electromagnet {'ON' if active else 'OFF'}")
    
    def _set_delivery_servo(self, angle: float) -> None:
        """Set delivery servo to specified angle."""
        command = f"SERVO,{self.config.delivery_servo_pin},{int(angle)}"
        if self.esp32.connected:
            self.esp32._send_command(command)
        self.logger.debug(f"Delivery servo set to {angle}°")
    
    def _send_motor_command(self, fl: int, fr: int, bl: int, br: int) -> None:
        """Send motor speed command to all wheels."""
        if self.esp32.connected:
            self.esp32.send_motor_speeds(fl, fr, bl, br)
    
    def _stop_all_motors(self) -> None:
        """Stop all motors."""
        self._send_motor_command(0, 0, 0, 0)
    
    # Utility Methods
    
    def _grid_to_meters(self, grid_coord: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid coordinates to meters."""
        x, y = grid_coord
        return ((x + 0.5) * 0.11, (y + 0.5) * 0.11)
    
    def _calculate_move_time(self, distance: float, speed: int) -> float:
        """Calculate time needed to move specified distance at given speed."""
        return distance / (speed * 0.01)  # Convert speed to m/s
    
    def _basic_navigate_to(self, x: float, y: float) -> bool:
        """Basic navigation fallback when no callback is available."""
        current_x, current_y, _ = self.position_tracker.get_pose()
        distance = math.sqrt((x - current_x)**2 + (y - current_y)**2)
        
        if distance > 0.05:  # 5cm threshold
            if x > current_x:
                self._move_forward(distance)
            else:
                self._move_backward(distance)
        
        return True
    
    def _reset_counters(self) -> None:
        """Reset mission counters."""
        self.status.boxes_collected = 0
        self.status.boxes_delivered = 0
        self.status.current_pickup_index = 0
        self.status.current_delivery_index = 0
    
    def _handle_error(self, message: str) -> None:
        """Handle mission errors."""
        self.logger.error(message)
        self.status.phase = MissionPhase.ERROR
        self.status.error_message = message
        self._stop_all_motors()
        self._set_electromagnet(False)
    
    def _cleanup_mission(self) -> None:
        """Clean up after mission completion."""
        self.logger.info("Cleaning up after mission")
        
        # Turn off electromagnet
        self._set_electromagnet(False)
        
        # Return servos to neutral positions
        self._set_arm_angle(self.config.arm_neutral)
        self._set_delivery_servo(self.config.delivery_retracted)
        
        self.logger.info("Mission cleanup complete")


# Integration and Factory Functions

def create_box_pickup_system(esp32_bridge, position_tracker, 
                           custom_config: Optional[RobotConfig] = None) -> BoxPickupSystem:
    """
    Factory function to create a properly configured box pickup system.
    
    Args:
        esp32_bridge: Hardware communication interface
        position_tracker: Robot position tracking system
        custom_config: Optional custom configuration
        
    Returns:
        BoxPickupSystem: Configured pickup system instance
    """
    return BoxPickupSystem(esp32_bridge, position_tracker, custom_config)


def integrate_with_main_controller(robot_controller, box_system: BoxPickupSystem) -> Callable:
    """
    Create integration function for main robot controller.
    
    Args:
        robot_controller: Main robot controller instance
        box_system: Box pickup system instance
        
    Returns:
        Callable: Function to execute complete box mission
    """
    def execute_box_mission() -> bool:
        """Execute complete box pickup and delivery mission."""
        logger = logging.getLogger(__name__)
        logger.info("Starting integrated box mission")
        
        try:
            # Start mission
            if not box_system.start_mission():
                logger.error("Failed to start mission")
                return False
            
            # Execute pickup phase
            robot_controller.state = "box_pickup"
            if not box_system.execute_pickup_phase():
                logger.error("Pickup phase failed")
                return False
            
            # Execute delivery phase
            robot_controller.state = "box_delivery"
            if not box_system.execute_delivery_phase():
                logger.error("Delivery phase failed")
                return False
            
            # Mission complete
            robot_controller.state = "mission_complete"
            logger.info("Box mission completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Mission failed with exception: {e}")
            robot_controller.state = "error"
            box_system.emergency_stop()
            return False
    
    return execute_box_mission


# Example Usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    