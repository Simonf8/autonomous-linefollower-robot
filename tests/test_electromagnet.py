import time
import sys
import os

# Add the project root to the Python path to allow for `src` imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.controllers.pi_motor_controller import PiMotorController

def print_help():
    """Prints the available commands."""
    print("\n--- Electromagnet Test ---")
    print("Commands:")
    print("  on      - Turn the electromagnet ON")
    print("  off     - Turn the electromagnet OFF")
    print("  status  - Get the electromagnet status")
    print("  exit, q - Exit the test")
    print("  help    - Show this help message")
    print("--------------------------\n")

def main():
    """Main interactive loop for testing the electromagnet."""
    print("Initializing motor controller for electromagnet test...")
    
    try:
        with PiMotorController() as controller:
            print("\nElectromagnet test ready.")
            print_help()

            while True:
                try:
                    cmd = input("Electromagnet > ").strip().lower()

                    if cmd in ['q', 'exit', 'quit']:
                        print("Exiting test.")
                        break
                    elif cmd == 'on':
                        controller.electromagnet_on()
                    elif cmd == 'off':
                        controller.electromagnet_off()
                    elif cmd == 'status':
                        is_on = controller.get_electromagnet_status()
                        print(f"Electromagnet status: {'ON' if is_on else 'OFF'}")
                    elif cmd == 'help':
                        print_help()
                    else:
                        if cmd:
                            print(f"Unknown command: '{cmd}'")
                        print_help()

                except KeyboardInterrupt:
                    print("\nCaught interrupt, exiting.")
                    break
                except Exception as e:
                    print(f"\nAn error occurred during command execution: {e}")

    except Exception as e:
        print(f"FATAL: Failed to initialize PiMotorController: {e}")
        print("Please ensure you are running on a Raspberry Pi with correct permissions.")
        sys.exit(1)

    print("Test finished. Resources have been cleaned up.")
    sys.exit(0)

if __name__ == "__main__":
    main() 