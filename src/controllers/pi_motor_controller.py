from gpiozero import Motor, RotaryEncoder
import time
import atexit
import signal
import sys

class PiMotorController:
    """
    Controller for a 4-wheel omni-drive robot using the Raspberry Pi's GPIO.
    Uses the gpiozero library, which is compatible with the Raspberry Pi 5.
    """
    # Encoders produce ~960 steps per wheel revolution.
    STEPS_PER_REVOLUTION = 960

    def __init__(self, trims: dict = None):
        # GPIO pin numbers (BCM mode) for the motor driver.
        self.motor_pins = {
            'fl': {'p1': 20, 'p2': 21},  # Front Left
            'fr': {'p1': 16, 'p2': 26},  # Front Right
            'bl': {'p1': 17, 'p2': 27},  # Back Left
            'br': {'p1': 23, 'p2': 22},   # Back Right
        }
        
        # GPIO pin numbers for the wheel encoders (A and B phases)
        self.encoder_pins = {
            'fl': {'A': 13, 'B': 19},    # Front Left
            'fr': {'A': 24, 'B': 2},  # Front Right
            'bl': {'A': 6, 'B': 5},  # Back Left
            'br': {'A': 3, 'B': 4},  # Back Right
        }
        
        # Apply motor trims if provided. These are used to calibrate for motor speed
        # differences. If a motor is too fast (e.g., the back-right), lower its
        # trim value below 1.0.
        self.trims = trims if trims else {'fl': 1.0, 'fr': 1.0, 'bl': 1.0, 'br': 1.0}
        
        self.motors = {}
        self.encoders = {}
        self._setup_motors_and_encoders()
        
        # Register cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_motors_and_encoders(self):
        """Initialize motor and encoder objects using gpiozero with error handling."""
        try:
            # Initialize motors
            for wheel, pins in self.motor_pins.items():
                # The gpiozero Motor class handles the two-pin setup automatically.
                self.motors[wheel] = Motor(forward=pins['p1'], backward=pins['p2'])

            # Initialize encoders
            for wheel, pins in self.encoder_pins.items():
                # max_steps=0 allows for unlimited counting in both directions
                self.encoders[wheel] = RotaryEncoder(a=pins['A'], b=pins['B'], max_steps=0)

            print("Pi Motor Controller and Encoders initialized using gpiozero.")
        except Exception as e:
            if "GPIO busy" in str(e):
                print(f"GPIO Error: {e}")
                print("Attempting to clean up GPIO resources and retry...")
                self._force_gpio_cleanup()
                # Retry once after cleanup
                try:
                    for wheel, pins in self.motor_pins.items():
                        self.motors[wheel] = Motor(forward=pins['p1'], backward=pins['p2'])
                    for wheel, pins in self.encoder_pins.items():
                        self.encoders[wheel] = RotaryEncoder(a=pins['A'], b=pins['B'], max_steps=0)
                    print("Pi Motor Controller and Encoders initialized after GPIO cleanup.")
                except Exception as retry_error:
                    print(f"Failed to initialize motors/encoders after cleanup: {retry_error}")
                    raise
            else:
                print(f"Motor/Encoder initialization error: {e}")
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

            print("Cleaning up encoder resources...")
            for encoder in self.encoders.values():
                encoder.close()
            self.encoders.clear()
            
            # The .close() calls on individual devices are sufficient.
            # The 'reset()' method is not available on the LGPIOFactory.
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

    def get_encoder_counts(self):
        """Returns the current raw step count for each encoder."""
        if not self.encoders:
            return {'fl': 0, 'fr': 0, 'bl': 0, 'br': 0}
        
        return {
            wheel: encoder.steps
            for wheel, encoder in self.encoders.items()
        }

    def reset_encoders(self):
        """Resets all encoder counts to zero."""
        if not self.encoders:
            print("Warning: Encoders not initialized, cannot reset.")
            return

        for encoder in self.encoders.values():
            encoder.steps = 0
        print("All encoder counts have been reset.")

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