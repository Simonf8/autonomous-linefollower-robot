#!/usr/bin/env python3
"""
ESP32 Line Sensor Test Script
Simple test to verify line sensors are working correctly
Upload this to your ESP32 to test the sensors independently
"""

import time
from machine import Pin

# Line Sensor Configuration (same as main code)
LINE_SENSOR_PINS = [14, 27, 16, 17, 26]  # GPIO pins for line sensors
SENSOR_NAMES = ["Left2", "Left1", "Center", "Right1", "Right2"]

class LineSensorTester:
    def __init__(self, sensor_pins, sensor_names):
        self.sensors = []
        self.sensor_names = sensor_names
        
        print("Initializing line sensors...")
        for i, pin in enumerate(sensor_pins):
            try:
                gpio_pin = Pin(pin, Pin.IN)
                self.sensors.append(gpio_pin)
                print(f"✓ {sensor_names[i]} sensor (pin {pin}) initialized")
            except Exception as e:
                print(f"✗ Failed to initialize sensor on pin {pin}: {e}")
                self.sensors.append(None)
        
        print("-" * 50)
    
    def read_sensors(self):
        """Read all sensor values (converted for line following)"""
        values = []
        for sensor in self.sensors:
            if sensor is not None:
                try:
                    # Convert: 0 = line detected, 1 = no line → 1 = line detected, 0 = no line
                    raw_value = sensor.value()
                    line_detected = 1 - raw_value  # 0 becomes 1, 1 becomes 0
                    values.append(line_detected)
                except:
                    values.append(0)
            else:
                values.append(0)
        return values
    
    def calculate_position(self, values):
        """Calculate line position (-1.0 to 1.0)"""
        # Weighted sum method
        weighted_sum = 0
        total = sum(values)
        
        if total == 0:
            return None  # No line detected
        
        # Calculate weighted position
        for i, value in enumerate(values):
            # Position weights: -1.0, -0.5, 0.0, 0.5, 1.0
            position_weight = (i - 2) * 0.5
            weighted_sum += value * position_weight
        
        position = weighted_sum / total
        return position
    
    def display_readings(self, values, position):
        """Display sensor readings in a nice format"""
        # Clear screen (works on most terminals)
        print("\033[2J\033[H", end="")
        
        print("ESP32 Line Sensor Test")
        print("=" * 50)
        print()
        
        # Display sensor status
        print("Sensor Status (● = Line Detected, ○ = No Line):")
        for i, (name, value) in enumerate(zip(self.sensor_names, values)):
            status = "●" if value else "○"
            print(f"  {name:>8}: {status} (line: {value})")
        
        print()
        
        # Display line position
        if position is not None:
            print(f"Line Position: {position:+.2f}")
            
            # Visual position indicator
            pos_display = ""
            for i in range(21):  # -10 to +10
                pos_val = (i - 10) / 10.0
                if abs(pos_val - position) < 0.1:
                    pos_display += "█"
                elif abs(pos_val - position) < 0.2:
                    pos_display += "▓"
                else:
                    pos_display += "░"
            
            print(f"Position:     {pos_display}")
            print("              ←---------CENTER---------→")
            
            # Suggested robot action
            if abs(position) < 0.1:
                action = "FORWARD"
            elif position > 0.5:
                action = "TURN RIGHT"
            elif position < -0.5:
                action = "TURN LEFT"
            elif position > 0.2:
                action = "SLIGHT RIGHT"
            elif position < -0.2:
                action = "SLIGHT LEFT"
            else:
                action = "FORWARD"
                
            print(f"Suggested Action: {action}")
        else:
            print("Line Position: NO LINE DETECTED")
            print("Suggested Action: SEARCH FOR LINE")
        
        print()
        print("Press Ctrl+C to stop")
        print("-" * 50)
    
    def run_test(self):
        """Run continuous sensor test"""
        print("Starting line sensor test...")
        print("Place the robot over a line to see sensor responses")
        print()
        
        try:
            while True:
                # Read sensors
                values = self.read_sensors()
                
                # Calculate position
                position = self.calculate_position(values)
                
                # Display results
                self.display_readings(values, position)
                
                # Wait before next reading
                time.sleep(0.2)
                
        except KeyboardInterrupt:
            print("\nTest stopped by user")
        except Exception as e:
            print(f"Error during test: {e}")

def main():
    print("ESP32 Line Sensor Tester")
    print("=" * 30)
    
    # Create tester
    tester = LineSensorTester(LINE_SENSOR_PINS, SENSOR_NAMES)
    
    # Run test
    tester.run_test()
    
    print("Test complete!")

if __name__ == "__main__":
    main() 