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
    'fl': {'A': 13, 'B': 19},    # Front Left
    'fr': {'A': 24, 'B': 2},  # Front Right
    'bl': {'A': 6, 'B': 5},  # Back Left
    'br': {'A': 3, 'B': 4},  # Back Right
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
    
    # The .close() calls on the devices are sufficient for cleanup.
    # The 'reset()' method is not available on the LGPIOFactory used by newer Pis.
    print("Cleanup complete.")

def signal_handler(signum, frame):
    """Handle exit signals cleanly."""
    cleanup_gpio()
    sys.exit(0)

# Register cleanup functions to run on exit
atexit.register(cleanup_gpio)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# --- AUTOMATIC TEST LOGIC ---
def test_single_wheel_auto(wheel_name: str, speed: float = 0.3, duration: float = 2.0):
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

def run_automatic_test():
    """Main function to run the individual wheel tests."""
    print("\nStarting individual wheel and encoder test.")
    print("The robot should be elevated to allow wheels to spin freely.")
    
    wheels_to_test = ['fl', 'fr', 'bl', 'br']
    
    for wheel in wheels_to_test:
        test_single_wheel_auto(wheel)
        time.sleep(1) # Pause between testing each wheel
        
    print("\nAll tests complete.")


# --- MANUAL TEST LOGIC ---
def print_manual_help():
    """Prints the help text for manual mode."""
    print("\n--- Manual Wheel & Encoder Test ---")
    print("Commands:")
    print("  <wheel>      - Select a wheel to control. E.g., 'fl', 'fr', 'bl', 'br'")
    print("  f <speed>    - Run selected wheel FORWARD at speed (0-100). E.g., 'f 50'")
    print("  b <speed>    - Run selected wheel BACKWARD at speed (0-100). E.g., 'b 30'")
    print("  s            - STOP the selected wheel's motor.")
    print("  r            - RESET the selected wheel's encoder count to 0.")
    print("  c            - Check and print ALL current encoder counts.")
    print("  help         - Show this help message.")
    print("  exit or q    - Quit the program.")
    print("------------------------------------")

def run_manual_test():
    """Runs an interactive loop for manually testing wheels."""
    selected_wheel = None
    print_manual_help()

    while True:
        if selected_wheel:
            prompt = f"Wheel({selected_wheel.upper()}) | Encoder: {ENCODERS[selected_wheel].steps} > "
        else:
            prompt = "No wheel selected > "
            
        try:
            cmd_input = input(prompt).strip().lower()
            parts = cmd_input.split()
            if not parts:
                continue

            command = parts[0]

            if command in ['fl', 'fr', 'bl', 'br']:
                selected_wheel = command
                print(f"Selected wheel: {selected_wheel.upper()}")
                continue
            
            if command in ['exit', 'q']:
                print("Exiting manual test.")
                break

            if command == 'help':
                print_manual_help()
                continue
                
            if command == 'c':
                 print("--- Current Encoder Counts ---")
                 for w, enc in ENCODERS.items():
                     print(f"  {w.upper()}: {enc.steps}")
                 print("------------------------------")
                 continue

            if not selected_wheel:
                print("No wheel selected. Please select a wheel first (e.g., 'fl').")
                continue
            
            motor = MOTORS[selected_wheel]
            encoder = ENCODERS[selected_wheel]

            if command in ['f', 'b']:
                speed = 0.3 # Default speed
                if len(parts) > 1 and parts[1].isdigit():
                    speed = min(100, int(parts[1])) / 100.0
                
                if command == 'f':
                    print(f"Running {selected_wheel.upper()} forward at {speed*100:.0f}%...")
                    motor.forward(speed)
                else:
                    print(f"Running {selected_wheel.upper()} backward at {speed*100:.0f}%...")
                    motor.backward(speed)

            elif command == 's':
                print(f"Stopping {selected_wheel.upper()}...")
                motor.stop()

            elif command == 'r':
                print(f"Resetting {selected_wheel.upper()} encoder count.")
                encoder.steps = 0
            
            else:
                print(f"Unknown command: '{command}'. Type 'help' for options.")

        except KeyboardInterrupt:
            print("\nExiting manual test.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


# --- MAIN DISPATCHER ---
def main():
    """Parses command line args and runs the appropriate test mode."""
    setup_gpio()
    
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'manual':
        run_manual_test()
    else:
        run_automatic_test()

if __name__ == "__main__":
    main() 