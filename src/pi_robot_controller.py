#!/usr/bin/env python3
"""
Raspberry Pi Direct Robot Controller
===================================

Controls motors and line sensor directly from Raspberry Pi GPIO pins
instead of using ESP32 communication.
"""

import time
import threading
import RPi.GPIO as GPIO
import logging
from typing import List, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LineSensor:
    """3-channel infrared line sensor using Pi GPIO pins"""
    
    def __init__(self, left_pin: int = 21, center_pin: int = 20, right_pin: int = 26):
        """
        Initialize line sensor with GPIO pins.
        
        Args:
            left_pin: GPIO pin for left sensor
            center_pin: GPIO pin for center sensor  
            right_pin: GPIO pin for right sensor
        """
        self.left_pin = left_pin
        self.center_pin = center_pin
        self.right_pin = right_pin
        
        # Setup GPIO pins as inputs with pull-up resistors
        GPIO.setup(self.left_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.center_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.right_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # Sensor weights for position calculation
        self.weights = [0, 1000, 2000]  # Left, center, right positions
        
        logger.info(f"Line sensor initialized - Left: {left_pin}, Center: {center_pin}, Right: {right_pin}")
    
    def read_sensors(self) -> List[int]:
        """
        Read digital values from all sensors.
        
        Returns:
            List of sensor values [left, center, right] where 0=line detected, 1=no line
        """
        left = GPIO.input(self.left_pin)
        center = GPIO.input(self.center_pin)
        right = GPIO.input(self.right_pin)
        
        return [left, center, right]
    
    def is_line_detected(self) -> bool:
        """Check if any sensor detects the line"""
        sensors = self.read_sensors()
        return any(sensor == 0 for sensor in sensors)
    
    def get_line_position(self) -> int:
        """
        Calculate line position using weighted average.
        
        Returns:
            Position value: 0-2000 (0=left, 1000=center, 2000=right) or -1 if no line
        """
        sensors = self.read_sensors()
        
        # Calculate weighted average
        total_weight = 0
        weighted_sum = 0
        line_detected = False
        
        for i, sensor_value in enumerate(sensors):
            if sensor_value == 0:  # Line detected (active low)
                line_detected = True
                weight = 1000 - sensor_value * 1000  # Invert for calculation
                total_weight += weight
                weighted_sum += self.weights[i] * weight
        
        if not line_detected:
            return -1
        
        if total_weight == 0:
            return 1000  # Default to center
        
        position = weighted_sum / total_weight
        return max(0, min(2000, int(position)))
    
    def get_line_error(self) -> int:
        """
        Get line position error for PID control.
        
        Returns:
            Error from center position (-1000 to +1000)
        """
        position = self.get_line_position()
        if position == -1:
            return 0  # No line detected
        return position - 1000  # Center is at 1000
    
    def print_debug(self):
        """Print sensor readings for debugging"""
        sensors = self.read_sensors()
        position = self.get_line_position()
        error = self.get_line_error()
        
        print(f"Sensors: [L:{sensors[0]} C:{sensors[1]} R:{sensors[2]}] Position: {position} Error: {error}")


class MotorController:
    """Control 4 DC motors using 2 pins per motor"""
    
    def __init__(self):
        """Initialize motor controller with GPIO pins"""
        
        # Motor pin definitions - 2 pins per motor (adjust these based on your wiring)
        self.motors = {
            'fl': {'pin1': 17, 'pin2': 27},    # Front Left
            'fr': {'pin1': 22, 'pin2': 24},     # Front Right  
            'bl': {'pin1': 23, 'pin2': 25},      # Back Left
            'br': {'pin1': 9, 'pin2': 11}      # Back Right
        }
        
        # Setup GPIO pins and create PWM instances
        for motor, pins in self.motors.items():
            GPIO.setup(pins['pin1'], GPIO.OUT)
            GPIO.setup(pins['pin2'], GPIO.OUT)
            
            # Create PWM instances for both pins
            self.motors[motor]['pwm1'] = GPIO.PWM(pins['pin1'], 50)  # 1kHz frequency
            self.motors[motor]['pwm2'] = GPIO.PWM(pins['pin2'], 50)  # 1kHz frequency
            self.motors[motor]['pwm1'].start(0)  # Start with 0% duty cycle
            self.motors[motor]['pwm2'].start(0)  # Start with 0% duty cycle
        
        logger.info("Motor controller initialized")
    
    def set_motor_speed(self, motor: str, speed: int):
        """
        Set speed and direction for a specific motor using 2-pin PWM control.
        
        Args:
            motor: Motor name ('fl', 'fr', 'bl', 'br')
            speed: Speed value (-100 to +100, negative = reverse)
        """
        if motor not in self.motors:
            logger.error(f"Invalid motor: {motor}")
            return
        
        # Clamp speed to valid range
        speed = max(-100, min(100, speed))
        
        motor_data = self.motors[motor]
        
        if speed > 0:
            # Forward: pin1 PWM, pin2 off
            motor_data['pwm1'].ChangeDutyCycle(abs(speed))
            motor_data['pwm2'].ChangeDutyCycle(0)
        elif speed < 0:
            # Reverse: pin1 off, pin2 PWM
            motor_data['pwm1'].ChangeDutyCycle(0)
            motor_data['pwm2'].ChangeDutyCycle(abs(speed))
        else:
            # Stop: both pins off
            motor_data['pwm1'].ChangeDutyCycle(0)
            motor_data['pwm2'].ChangeDutyCycle(0)
    
    def set_all_speeds(self, fl: int, fr: int, bl: int, br: int):
        """Set speeds for all motors simultaneously"""
        self.set_motor_speed('fl', fl)
        self.set_motor_speed('fr', fr)
        self.set_motor_speed('bl', bl)
        self.set_motor_speed('br', br)
    
    def stop_all(self):
        """Stop all motors"""
        self.set_all_speeds(0, 0, 0, 0)
    
    def move_forward(self, speed: int = 50):
        """Move robot forward"""
        self.set_all_speeds(speed, speed, speed, speed)
    
    def move_backward(self, speed: int = 50):
        """Move robot backward"""
        self.set_all_speeds(-speed, -speed, -speed, -speed)
    
    def turn_left(self, speed: int = 50):
        """Turn robot left"""
        self.set_all_speeds(-speed, speed, -speed, speed)
    
    def turn_right(self, speed: int = 50):
        """Turn robot right"""
        self.set_all_speeds(speed, -speed, speed, -speed)
    
    def strafe_left(self, speed: int = 50):
        """Strafe robot left (omniwheel movement)"""
        self.set_all_speeds(-speed, speed, speed, -speed)
    
    def strafe_right(self, speed: int = 50):
        """Strafe robot right (omniwheel movement)"""
        self.set_all_speeds(speed, -speed, -speed, speed)


class PIDController:
    """Simple PID controller for line following"""
    
    def __init__(self, kp: float = 0.5, ki: float = 0.0, kd: float = 0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.previous_error = 0
        self.integral = 0
        self.last_time = time.time()
    
    def update(self, error: float) -> float:
        """
        Update PID controller with new error value.
        
        Args:
            error: Current error value
            
        Returns:
            PID output value
        """
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0:
            dt = 0.01  # Minimum time step
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term
        self.integral += error * dt
        integral = self.ki * self.integral
        
        # Derivative term
        derivative = self.kd * (error - self.previous_error) / dt
        
        # Calculate output
        output = proportional + integral + derivative
        
        # Store for next iteration
        self.previous_error = error
        self.last_time = current_time
        
        return output


class PiRobotController:
    """Main robot controller using Pi GPIO directly"""
    
    def __init__(self):
        """Initialize the robot controller"""
        
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Initialize components
        self.line_sensor = LineSensor()
        self.motor_controller = MotorController()
        self.pid = PIDController(kp=0.3, ki=0.0, kd=0.15)
        
        # Control parameters
        self.base_speed = 50
        self.max_turn_speed = 30
        self.running = False
        
        # Threading
        self.control_thread = None
        
        logger.info("Pi Robot Controller initialized")
    
    def start_line_following(self, base_speed: int = 50):
        """Start line following mode"""
        self.base_speed = base_speed
        self.running = True
        
        if self.control_thread is None or not self.control_thread.is_alive():
            self.control_thread = threading.Thread(target=self._line_follow_loop)
            self.control_thread.daemon = True
            self.control_thread.start()
            
        logger.info(f"Line following started with base speed: {base_speed}")
    
    def stop_line_following(self):
        """Stop line following mode"""
        self.running = False
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=1.0)
        self.motor_controller.stop_all()
        logger.info("Line following stopped")
    
    def _line_follow_loop(self):
        """Main line following control loop"""
        logger.info("Line following loop started")
        
        while self.running:
            try:
                # Get line sensor data
                line_error = self.line_sensor.get_line_error()
                
                if line_error == 0 and self.line_sensor.get_line_position() == -1:
                    # No line detected - search for line
                    self._search_for_line()
                else:
                    # Line detected - follow it
                    self._follow_line(line_error)
                
                time.sleep(0.05)  # 20Hz control loop
                
            except Exception as e:
                logger.error(f"Error in line following loop: {e}")
                self.motor_controller.stop_all()
                break
        
        logger.info("Line following loop ended")
    
    def _follow_line(self, line_error: int):
        """Follow the detected line using PID control"""
        
        # Calculate PID output
        pid_output = self.pid.update(line_error)
        
        # Limit PID output
        pid_output = max(-self.max_turn_speed, min(self.max_turn_speed, pid_output))
        
        # Calculate motor speeds
        left_speed = self.base_speed - pid_output
        right_speed = self.base_speed + pid_output
        
        # Apply speeds (assuming differential drive mapping for omniwheel)
        fl_speed = int(left_speed)
        fr_speed = int(right_speed)
        bl_speed = int(left_speed)
        br_speed = int(right_speed)
        
        self.motor_controller.set_all_speeds(fl_speed, fr_speed, bl_speed, br_speed)
    
    def _search_for_line(self):
        """Search for line when lost"""
        # Simple search pattern - turn slowly
        self.motor_controller.turn_left(20)
        time.sleep(0.1)
    
    def manual_control(self, fl: int, fr: int, bl: int, br: int):
        """Manual motor control"""
        if self.running:
            logger.warning("Cannot use manual control while line following is active")
            return
        
        self.motor_controller.set_all_speeds(fl, fr, bl, br)
    
    def get_sensor_data(self) -> dict:
        """Get current sensor data"""
        sensors = self.line_sensor.read_sensors()
        position = self.line_sensor.get_line_position()
        error = self.line_sensor.get_line_error()
        
        return {
            'sensors': sensors,
            'position': position,
            'error': error,
            'line_detected': self.line_sensor.is_line_detected()
        }
    
    def cleanup(self):
        """Clean up GPIO resources"""
        self.stop_line_following()
        GPIO.cleanup()
        logger.info("GPIO cleanup completed")


# Example usage and testing
def main():
    """Test the robot controller"""
    robot = PiRobotController()
    
    try:
        print("Starting robot test...")
        
        # Test sensor readings
        print("\nTesting sensors for 5 seconds...")
        start_time = time.time()
        while time.time() - start_time < 5:
            robot.line_sensor.print_debug()
            time.sleep(0.5)
        
        # Test manual motor control
        print("\nTesting motors...")
        robot.manual_control(30, 30, 30, 30)  # Forward
        time.sleep(1)
        robot.manual_control(0, 0, 0, 0)      # Stop
        time.sleep(1)
        
        # Test line following
        print("\nStarting line following test...")
        robot.start_line_following(40)
        time.sleep(10)  # Run for 10 seconds
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test error: {e}")
    finally:
        robot.cleanup()
        print("Test completed")


if __name__ == "__main__":
    main() 