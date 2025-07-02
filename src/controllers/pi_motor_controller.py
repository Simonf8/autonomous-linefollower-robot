from gpiozero import Motor, RotaryEncoder
import time
import atexit
import signal
import sys
from gpiozero import OutputDevice

class PiMotorController:
    """
    Controller for a 2-wheel differential drive robot using the Raspberry Pi's GPIO.
    Uses the gpiozero library, which is compatible with the Raspberry Pi 5.
    """
    # Encoders produce ~960 steps per wheel revolution.
    STEPS_PER_REVOLUTION = 960

    def __init__(self, trims: dict = None):
        # GPIO pin numbers (BCM mode) for the motor driver.
        self.motor_pins = {
            'left': {'p1': 13, 'p2': 19},   # Left Motor
            'right': {'p1': 21, 'p2': 20}, # Right Motor
        }
        
        # GPIO pin numbers for the wheel encoders (A and B phases)
        self.encoder_pins = {
            'left': {'A': 15, 'B': 14},     # Left Encoder
            'right': {'A': 3, 'B': 2},     # Right Encoder
        }
        
        # GPIO pin for the electromagnet
        self.electromagnet_pin = 25
        
        # Apply motor trims if provided. These are used to calibrate for motor speed
        # differences. If a motor is too fast, lower its trim value below 1.0.
        self.trims = trims if trims else {'left': 1.0, 'right': 1.0}
        
        self.motors = {}
        self.encoders = {}
        self._setup_motors_and_encoders()
        
        # Setup electromagnet
        try:
            self.electromagnet = OutputDevice(self.electromagnet_pin, active_high=True, initial_value=False)
            print(f"Electromagnet initialized on GPIO {self.electromagnet_pin}.")
        except Exception as e:
            print(f"Failed to initialize electromagnet: {e}")
            self.electromagnet = None
        
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
            
            if self.electromagnet:
                print("Cleaning up electromagnet resource...")
                self.electromagnet.off()
                self.electromagnet.close()
            
            # The .close() calls on individual devices are sufficient.
            # The 'reset()' method is not available on the LGPIOFactory.
            print("GPIO cleanup completed.")
        except Exception as e:
            print(f"Warning during GPIO cleanup: {e}")

    def send_motor_speeds(self, left: int, right: int):
        """
        Set speeds for both motors.
        Speed is from -100 (reverse) to 100 (forward).
        
        Args:
            left: Speed for left motor (-100 to 100)
            right: Speed for right motor (-100 to 100)
        """
        if not self.motors:
            print("Warning: Motors not initialized, cannot set speeds")
            return
            
        self._set_motor_speed('left', left)
        self._set_motor_speed('right', right)

    def move_forward(self, speed: int = 50):
        """Move robot forward at specified speed."""
        self.send_motor_speeds(speed, speed)

    def move_backward(self, speed: int = 50):
        """Move robot backward at specified speed."""
        self.send_motor_speeds(-speed, -speed)

    def turn_left(self, speed: int = 50):
        """Turn robot left by spinning wheels in opposite directions."""
        self.send_motor_speeds(-speed, speed)

    def turn_right(self, speed: int = 50):
        """Turn robot right by spinning wheels in opposite directions."""
        self.send_motor_speeds(speed, -speed)

    def pivot_left(self, speed: int = 50):
        """Pivot left by stopping left wheel and moving right wheel forward."""
        self.send_motor_speeds(0, speed)

    def pivot_right(self, speed: int = 50):
        """Pivot right by stopping right wheel and moving left wheel forward."""
        self.send_motor_speeds(speed, 0)

    def get_encoder_counts(self):
        """Returns the current raw step count for each encoder."""
        if not self.encoders:
            return {'left': 0, 'right': 0}
        
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

    def electromagnet_on(self):
        """Activates the electromagnet."""
        if self.electromagnet:
            self.electromagnet.on()
            print("Electromagnet ON")
        else:
            print("Warning: Electromagnet not initialized.")

    def electromagnet_off(self):
        """Deactivates the electromagnet."""
        if self.electromagnet:
            self.electromagnet.off()
            print("Electromagnet OFF")
        else:
            print("Warning: Electromagnet not initialized.")

    def get_electromagnet_status(self) -> bool:
        """Returns the current status of the electromagnet (on/off)."""
        if self.electromagnet:
            return self.electromagnet.is_active
        return False

    def move_forward(self, speed: int = 50):
        """Move robot forward at specified speed."""
        self.send_motor_speeds(speed, speed) 