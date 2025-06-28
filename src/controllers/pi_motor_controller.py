from gpiozero import Motor
import time
import atexit
import signal
import sys

class PiMotorController:
    """
    Controller for a 4-wheel omni-drive robot using the Raspberry Pi's GPIO.
    Uses the gpiozero library, which is compatible with the Raspberry Pi 5.
    """
    def __init__(self, trims: dict = None):
        # GPIO pin numbers (BCM mode) for the motor driver.
        self.motor_pins = {
            'fl': {'p1': 20, 'p2': 21},  # Front Left
            'fr': {'p1': 16, 'p2': 26},  # Front Right
            'bl': {'p1': 17, 'p2': 27},  # Back Left
            'br': {'p1': 23, 'p2': 22},   # Back Right
        }
        
        # Apply motor trims if provided
        self.trims = trims if trims else {'fl': 1.0, 'fr': 1.0, 'bl': 1.0, 'br': 1.0}
        
        self.motors = {}
        self._setup_motors()
        
        # Register cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_motors(self):
        """Initialize motor objects using gpiozero with error handling."""
        try:
            for wheel, pins in self.motor_pins.items():
                # The gpiozero Motor class handles the two-pin setup automatically.
                self.motors[wheel] = Motor(forward=pins['p1'], backward=pins['p2'])
            print("Pi Motor Controller initialized using gpiozero.")
        except Exception as e:
            if "GPIO busy" in str(e):
                print(f"GPIO Error: {e}")
                print("Attempting to clean up GPIO resources and retry...")
                self._force_gpio_cleanup()
                # Retry once after cleanup
                try:
                    for wheel, pins in self.motor_pins.items():
                        self.motors[wheel] = Motor(forward=pins['p1'], backward=pins['p2'])
                    print("Pi Motor Controller initialized after GPIO cleanup.")
                except Exception as retry_error:
                    print(f"Failed to initialize motors after cleanup: {retry_error}")
                    raise
            else:
                print(f"Motor initialization error: {e}")
                raise

    def _force_gpio_cleanup(self):
        """Force cleanup of GPIO resources."""
        try:
            # Import gpiozero's Device class to access cleanup methods
            from gpiozero import Device
            Device.pin_factory.reset()
            print("GPIO resources reset.")
        except Exception as e:
            print(f"GPIO cleanup warning: {e}")

    def _signal_handler(self, signum, frame):
        """Handle system signals for clean shutdown."""
        print(f"Received signal {signum}, cleaning up GPIO...")
        self._cleanup()
        sys.exit(0)

    def _cleanup(self):
        """Clean up GPIO resources."""
        try:
            print("Cleaning up motor controller resources...")
            for motor in self.motors.values():
                motor.stop()
                motor.close()
            self.motors.clear()
            
            # Reset the pin factory to release all pins
            from gpiozero import Device
            Device.pin_factory.reset()
            print("GPIO cleanup completed.")
        except Exception as e:
            print(f"Warning during GPIO cleanup: {e}")

    def send_motor_speeds(self, fl: int, fr: int, bl: int, br: int):
        """
        Set speeds for all four motors.
        Speed is from -100 (reverse) to 100 (forward).
        """
        if not self.motors:
            print("Warning: Motors not initialized, cannot set speeds")
            return
            
        self._set_motor_speed('fl', fl)
        self._set_motor_speed('fr', fr)
        self._set_motor_speed('bl', bl)
        self._set_motor_speed('br', br)

    def _set_motor_speed(self, wheel: str, speed: int):
        """Set speed for a single motor."""
        if wheel not in self.motors:
            print(f"Warning: Motor {wheel} not available")
            return
            
        speed = max(-100, min(100, speed))
        
        # Apply the trim for this specific motor
        trimmed_speed = speed * self.trims.get(wheel, 1.0)
        
        # gpiozero uses a speed range of 0 to 1.
        motor_speed = abs(trimmed_speed) / 100.0
        motor = self.motors[wheel]
        
        try:
            if trimmed_speed > 0:
                motor.forward(speed=motor_speed)
            elif trimmed_speed < 0:
                motor.backward(speed=motor_speed)
            else:
                motor.stop()
        except Exception as e:
            print(f"Error controlling motor {wheel}: {e}")

    def stop(self):
        """Stop all motors."""
        print("Stopping Pi Motor Controller...")
        for motor in self.motors.values():
            try:
                motor.stop()
            except Exception as e:
                print(f"Error stopping motor: {e}")
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self._cleanup() 