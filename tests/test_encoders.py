import time
import atexit
import signal
import sys
from gpiozero import Motor, RotaryEncoder

# --- CONFIGURATION ---
# GPIO pin numbers (BCM mode) for the motor driver.
MOTOR_PINS = {
    'fl': {'p1': 20, 'p2': 21},  # Front Left
    'fr': {'p1': 16, 'p2': 26},  # Front Right
    'bl': {'p1': 17, 'p2': 27},  # Back Left
    'br': {'p1': 23, 'p2': 22},   # Back Right
}

# GPIO pin numbers for the wheel encoders (A and B phases)
ENCODER_PINS = {
    'fl': {'A': 5, 'B': 6},    # Front Left
    'fr': {'A': 13, 'B': 19},  # Front Right
    'bl': {'A': 12, 'B': 18},  # Back Left
    'br': {'A': 24, 'B': 25},  # Back Right
}

# --- GLOBALS ---
MOTORS = {}
ENCODERS = {}

# --- SETUP AND CLEANUP ---
def setup_gpio():
    """Initialize all motor and encoder objects."""
    print("Setting up GPIO for motors and encoders...")
    try:
        for wheel, pins in MOTOR_PINS.items():
            MOTORS[wheel] = Motor(forward=pins['p1'], backward=pins['p2'])
        
        for wheel, pins in ENCODER_PINS.items():
            ENCODERS[wheel] = RotaryEncoder(a=pins['A'], b=pins['B'], max_steps=0)
        
        print("GPIO setup complete.")
    except Exception as e:
        print(f"FATAL: Failed to set up GPIO devices: {e}")
        print("Please check your connections and ensure gpiod is running.")
        sys.exit(1)

def cleanup_gpio():
    """Stop all motors and close GPIO resources."""
    print("\nCleaning up GPIO resources...")
    if MOTORS:
        for motor in MOTORS.values():
            motor.stop()
            motor.close()
    
    if ENCODERS:
        for encoder in ENCODERS.values():
            encoder.close()
    
    from gpiozero import Device
    Device.pin_factory.reset()
    print("Cleanup complete.")

def signal_handler(signum, frame):
    """Handle exit signals cleanly."""
    cleanup_gpio()
    sys.exit(0)

# Register cleanup functions to run on exit
atexit.register(cleanup_gpio)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# --- TEST LOGIC ---
def test_single_wheel(wheel_name: str, speed: float = 0.3, duration: float = 2.0):
    """
    Tests a single motor and its corresponding encoder.
    
    Args:
        wheel_name: The name of the wheel to test (e.g., 'fl').
        speed: The speed to run the motor at (0.0 to 1.0).
        duration: The time in seconds to run the motor for.
    """
    if wheel_name not in MOTORS or wheel_name not in ENCODERS:
        print(f"ERROR: Invalid wheel name '{wheel_name}'.")
        return

    print(f"\n--- Testing Wheel: {wheel_name.upper()} ---")
    motor = MOTORS[wheel_name]
    encoder = ENCODERS[wheel_name]

    # --- Test Forward ---
    encoder.steps = 0 # Reset count
    print(f"Running FORWARD at {int(speed*100)}% speed for {duration}s...")
    motor.forward(speed)
    time.sleep(duration)
    motor.stop()
    time.sleep(0.2) # Settle time
    
    forward_count = encoder.steps
    print(f"Forward Count: {forward_count} steps")
    if forward_count <= 0:
        print(f"  WARNING: Expected positive count, got {forward_count}. Check wiring.")
    else:
        print(f"  SUCCESS: Got a positive count.")

    # --- Test Backward ---
    encoder.steps = 0 # Reset count
    print(f"Running BACKWARD at {int(speed*100)}% speed for {duration}s...")
    motor.backward(speed)
    time.sleep(duration)
    motor.stop()
    time.sleep(0.2) # Settle time
    
    backward_count = encoder.steps
    print(f"Backward Count: {backward_count} steps")
    if backward_count >= 0:
        print(f"  WARNING: Expected negative count, got {backward_count}. Check wiring.")
    else:
        print(f"  SUCCESS: Got a negative count.")
    
    print("-" * (19 + len(wheel_name)))


def main():
    """Main function to run the individual wheel tests."""
    setup_gpio()
    
    print("\nStarting individual wheel and encoder test.")
    print("The robot should be elevated to allow wheels to spin freely.")
    
    wheels_to_test = ['fl', 'fr', 'bl', 'br']
    
    for wheel in wheels_to_test:
        test_single_wheel(wheel)
        time.sleep(1) # Pause between testing each wheel
        
    print("\nAll tests complete.")

if __name__ == "__main__":
    main() 