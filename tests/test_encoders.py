import time
import atexit
import signal
import sys
from gpiozero import RotaryEncoder

# --- CONFIGURATION ---
# GPIO pin numbers for the wheel encoders (A and B phases)
ENCODER_PINS = {
    'fl': {'A': 15, 'B': 14},    # Front Left
    'fr': {'A': 2, 'B': 3},     # Front Right
    # Add more encoders here if needed:
    # 'bl': {'A': 4, 'B': 5},     # Back Left
    # 'br': {'A': 6, 'B': 7},     # Back Right
}

# --- GLOBALS ---
ENCODERS = {}

# --- SETUP AND CLEANUP ---
def setup_encoders():
    """Initialize all encoder objects."""
    print("Setting up encoders...")
    try:
        for wheel, pins in ENCODER_PINS.items():
            ENCODERS[wheel] = RotaryEncoder(a=pins['A'], b=pins['B'], max_steps=0)
            print(f"  {wheel.upper()}: pins A={pins['A']}, B={pins['B']}")
        
        print("Encoder setup complete.")
    except Exception as e:
        print(f"FATAL: Failed to set up encoders: {e}")
        print("Please check your connections and pin numbers.")
        sys.exit(1)

def cleanup_encoders():
    """Close encoder resources."""
    print("\nCleaning up encoders...")
    if ENCODERS:
        for encoder in ENCODERS.values():
            encoder.close()
    print("Cleanup complete.")

def signal_handler(signum, frame):
    """Handle exit signals cleanly."""
    cleanup_encoders()
    sys.exit(0)

# Register cleanup functions
atexit.register(cleanup_encoders)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# --- MANUAL TEST ---
def print_help():
    """Print help commands."""
    print("\n--- Simple Encoder Test ---")
    print("Commands:")
    print("  show         - Show current encoder counts")
    print("  reset        - Reset all encoder counts to 0")
    print("  reset <name> - Reset specific encoder (e.g., 'reset fl')")
    print("  watch        - Continuously watch encoder counts (Ctrl+C to stop)")
    print("  help         - Show this help")
    print("  exit or q    - Quit")
    print("----------------------------")
    print("To test: Manually spin your wheels and watch the counts change!")

def show_counts():
    """Display current encoder counts."""
    print("\n--- Current Encoder Counts ---")
    for name, encoder in ENCODERS.items():
        print(f"  {name.upper()}: {encoder.steps:6d} steps")
    print("------------------------------")

def reset_encoders(specific_encoder=None):
    """Reset encoder counts."""
    if specific_encoder:
        if specific_encoder in ENCODERS:
            ENCODERS[specific_encoder].steps = 0
            print(f"Reset {specific_encoder.upper()} encoder to 0")
        else:
            print(f"Unknown encoder: {specific_encoder}")
    else:
        for encoder in ENCODERS.values():
            encoder.steps = 0
        print("Reset all encoders to 0")

def watch_encoders():
    """Continuously display encoder counts."""
    print("\nWatching encoders... (Press Ctrl+C to stop)")
    print("Spin your wheels manually to see counts change!")
    
    try:
        while True:
            # Clear screen (works on most terminals)
            print("\033[2J\033[H", end="")
            
            print("=== LIVE ENCODER COUNTS ===")
            for name, encoder in ENCODERS.items():
                print(f"{name.upper():3s}: {encoder.steps:8d} steps")
            print("===========================")
            print("Press Ctrl+C to stop watching")
            
            time.sleep(0.1)  # Update 10 times per second
            
    except KeyboardInterrupt:
        print("\nStopped watching encoders.")

def main():
    """Main interactive loop."""
    setup_encoders()
    
    print("Simple Encoder Test Ready!")
    print("Manually spin your wheels to test the encoders.")
    print("Type 'help' for commands.")
    
    while True:
        try:
            cmd_input = input("\nEncoder Test > ").strip().lower()
            parts = cmd_input.split()
            
            if not parts:
                continue
                
            command = parts[0]
            
            if command in ['exit', 'q', 'quit']:
                print("Goodbye!")
                break
                
            elif command == 'help':
                print_help()
                
            elif command == 'show':
                show_counts()
                
            elif command == 'reset':
                if len(parts) > 1:
                    reset_encoders(parts[1])
                else:
                    reset_encoders()
                    
            elif command == 'watch':
                watch_encoders()
                
            else:
                print(f"Unknown command: '{command}'. Type 'help' for options.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()