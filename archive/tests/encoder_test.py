#!/usr/bin/env python3

import time
import sys
import os

# Add the project root and the controllers directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
controllers_path = os.path.join(project_root, 'src', 'controllers')
sys.path.insert(0, project_root)
sys.path.insert(0, controllers_path)

# Now we can import the ESP32Bridge from the main controller file
from main import ESP32Bridge

# --- CONFIGURATION ---
# The IP address of your ESP32. Make sure this matches the one in main.py
ESP32_IP = "192.168.2.36"

def run_encoder_test():
    """
    Connects to the ESP32 and continuously prints encoder ticks.
    """
    print("--- ENCODER TEST UTILITY ---")
    print("This script will connect to the ESP32 and display live encoder data.")
    print("Manually turn each wheel to see if the corresponding tick values change.")
    print(f"Attempting to connect to ESP32 at {ESP32_IP}...")

    # Initialize the communication bridge
    esp32_bridge = ESP32Bridge(ESP32_IP)

    if not esp32_bridge.start():
        print("\nERROR: Could not connect to the ESP32.")
        print("Please check the following:")
        print("1. The ESP32_IP address is correct.")
        print("2. The robot is powered on and connected to the same WiFi network.")
        print("3. The ESP32 server program is running.")
        return

    print("\nSuccessfully connected to ESP32.")
    print("Press CTRL+C to stop the test.\n")

    last_ticks = [0, 0, 0, 0]

    try:
        while True:
            # Get the latest raw encoder ticks from the bridge
            current_ticks = esp32_bridge.get_encoder_ticks()

            # Calculate the change since the last reading
            delta_ticks = [current - last for current, last in zip(current_ticks, last_ticks)]
            
            # Update the last known ticks
            last_ticks = current_ticks

            # Format the output for readability
            output = (
                f"TOTAL TICKS: [FL: {current_ticks[0]:>8}, FR: {current_ticks[1]:>8}, "
                f"BL: {current_ticks[2]:>8}, BR: {current_ticks[3]:>8}] | "
                f"CHANGE: [FL: {delta_ticks[0]:>4}, FR: {delta_ticks[1]:>4}, "
                f"BL: {delta_ticks[2]:>4}, BR: {delta_ticks[3]:>4}]"
            )
            
            # Print to the same line to avoid spamming the console
            print(f"\r{output}", end="")

            # Wait a short moment before the next poll
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nEncoder test stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("Closing connection to ESP32.")
        esp32_bridge.stop()

if __name__ == '__main__':
    run_encoder_test() 