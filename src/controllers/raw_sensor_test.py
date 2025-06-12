#!/usr/bin/env python3
"""
Raw Sensor Diagnostic - ESP32
Shows RAW sensor values to determine correct logic
Upload this to ESP32 to see what your sensors actually output
"""

import time
from machine import Pin

# Line Sensor Pins
SENSOR_PINS = [14, 27, 16, 17, 25]
SENSOR_NAMES = ["Left2", "Left1", "Center", "Right1", "Right2"]

def main():
    print("RAW Sensor Diagnostic")
    print("=" * 40)
    
    # Initialize sensors
    sensors = []
    for i, pin in enumerate(SENSOR_PINS):
        try:
            gpio_pin = Pin(pin, Pin.IN)
            sensors.append(gpio_pin)
            print(f"✓ {SENSOR_NAMES[i]} (pin {pin}) OK")
        except Exception as e:
            print(f"✗ Pin {pin} failed: {e}")
            sensors.append(None)
    
    print("\nStarting raw sensor readings...")
    print("Test instructions:")
    print("1. Place robot on WHITE surface - note the values")
    print("2. Place robot over BLACK line - note the values")
    print("3. Move individual sensors over line - see which change")
    print("\nPress Ctrl+C to stop")
    print("-" * 40)
    
    try:
        while True:
            # Read raw values
            values = []
            for sensor in sensors:
                if sensor:
                    values.append(sensor.value())
                else:
                    values.append("ERR")
            
            # Display values
            print(f"RAW: {values[0]}  {values[1]}  {values[2]}  {values[3]}  {values[4]} ", end="")
            print(f"| {SENSOR_NAMES[0]:>5} {SENSOR_NAMES[1]:>5} {SENSOR_NAMES[2]:>6} {SENSOR_NAMES[3]:>6} {SENSOR_NAMES[4]:>6}")
            
            time.sleep(0.3)
            
    except KeyboardInterrupt:
        print("\nDiagnostic complete!")
        print("\nHow to interpret results:")
        print("- If sensors show 1 on WHITE and 0 on BLACK → sensors are NORMAL")
        print("- If sensors show 0 on WHITE and 1 on BLACK → sensors are INVERTED")
        print("- If no change → check wiring or sensor power")

if __name__ == "__main__":
    main() 