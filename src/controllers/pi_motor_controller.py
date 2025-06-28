from gpiozero import Motor
import time

class PiMotorController:
    """
    Controller for a 4-wheel omni-drive robot using the Raspberry Pi's GPIO.
    Uses the gpiozero library, which is compatible with the Raspberry Pi 5.
    """
    def __init__(self):
        # GPIO pin numbers (BCM mode) for the motor driver.
        self.motor_pins = {
            'fl': {'p1': 20, 'p2': 21},  # Front Left
            'fr': {'p1': 16, 'p2': 26},  # Front Right
            'bl': {'p1': 17, 'p2': 27},  # Back Left
            'br': {'p1': 23, 'p2': 22},   # Back Right
        }
        
        self.motors = {}
        self._setup_motors()

    def _setup_motors(self):
        """Initialize motor objects using gpiozero."""
        for wheel, pins in self.motor_pins.items():
            # The gpiozero Motor class handles the two-pin setup automatically.
            self.motors[wheel] = Motor(forward=pins['p1'], backward=pins['p2'])
        print("Pi Motor Controller initialized using gpiozero.")

    def send_motor_speeds(self, fl: int, fr: int, bl: int, br: int):
        """
        Set speeds for all four motors.
        Speed is from -100 (reverse) to 100 (forward).
        """
        self._set_motor_speed('fl', fl)
        self._set_motor_speed('fr', fr)
        self._set_motor_speed('bl', bl)
        self._set_motor_speed('br', br)

    def _set_motor_speed(self, wheel: str, speed: int):
        """Set speed for a single motor."""
        speed = max(-100, min(100, speed))
        
        # gpiozero uses a speed range of 0 to 1.
        motor_speed = abs(speed) / 100.0
        motor = self.motors[wheel]
        
        if speed > 0:
            motor.forward(speed=motor_speed)
        elif speed < 0:
            motor.backward(speed=motor_speed)
        else:
            motor.stop()

    def stop(self):
        """Stop all motors."""
        print("Stopping Pi Motor Controller...")
        for motor in self.motors.values():
            motor.stop()
        # gpiozero handles cleanup automatically on exit. 