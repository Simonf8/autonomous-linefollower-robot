#!/usr/bin/env python3

import time
import sys
import os

# Add the project root and the controllers directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
controllers_path = os.path.join(project_root, 'src', 'controllers')
sys.path.insert(0, project_root)
sys.path.insert(0, controllers_path)

from main import ESP32Bridge

# Configuration
ESP32_IP = "192.168.2.38" # Update this to match your ESP32's IP

def test_camera_navigation():
    """Test camera-based navigation without encoders"""
    print("🤖 Testing Camera-Based Navigation (No Encoders)")
    print("=" * 50)
    
    # Create ESP32 connection
    esp32 = ESP32Bridge(ESP32_IP)
    
    print(f"🔌 Connecting to ESP32 at {ESP32_IP}...")
    
    if esp32.start():
        print("✅ Connected successfully!")
        
        try:
            # Test 1: Line sensor readings
            print("\n📊 Test 1: Line Sensor Readings")
            print("Place the robot on a line and observe sensor readings...")
            
            for i in range(10):
                line_pos, line_err, sensor_vals = esp32.get_line_sensor_data()
                line_detected = esp32.is_line_detected()
                
                print(f"   Reading {i+1}:")
                print(f"     Line Position: {line_pos} ({'detected' if line_detected else 'not detected'})")
                print(f"     Line Error: {line_err}")
                print(f"     Sensor Values: {sensor_vals}")
                time.sleep(1)
            
            # Test 2: Basic movement test
            print("\n🔧 Test 2: Basic Movement Test")
            
            print("   Moving forward for 2 seconds...")
            esp32.send_motor_speeds(40, 40, 40, 40)
            time.sleep(2)
            
            print("   Stopping...")
            esp32.send_motor_speeds(0, 0, 0, 0)
            time.sleep(1)
            
            # Test 3: Line following simulation
            print("\n🛣️  Test 3: Line Following Test (10 seconds)")
            print("   Place robot on line. Robot will attempt to follow...")
            
            start_time = time.time()
            while time.time() - start_time < 10:
                line_pos, line_err, sensor_vals = esp32.get_line_sensor_data()
                
                if line_pos == -1:
                    # No line detected, stop
                    esp32.send_motor_speeds(0, 0, 0, 0)
                    print("     No line detected, stopping...")
                else:
                    # Simple line following logic
                    base_speed = 30
                    turn_factor = 0.3
                    turn_correction = line_err * turn_factor
                    
                    left_speed = int(base_speed - turn_correction / 10)
                    right_speed = int(base_speed + turn_correction / 10)
                    
                    # Ensure speeds are within bounds
                    left_speed = max(-100, min(100, left_speed))
                    right_speed = max(-100, min(100, right_speed))
                    
                    esp32.send_motor_speeds(left_speed, right_speed, left_speed, right_speed)
                    
                    if time.time() - start_time > 8:  # Print status in last 2 seconds
                        print(f"     Following line: pos={line_pos}, err={line_err}, speeds=({left_speed},{right_speed})")
                
                time.sleep(0.1)
            
            # Final stop
            print("   Test complete. Stopping robot...")
            esp32.send_motor_speeds(0, 0, 0, 0)
            
            print("\n✅ Camera-based navigation test completed successfully!")
            print("The robot is now configured for camera-only navigation.")
            print("Encoder position tracking has been disabled.")
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Test stopped by user.")
            esp32.send_motor_speeds(0, 0, 0, 0)
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            esp32.send_motor_speeds(0, 0, 0, 0)
        finally:
            print("Closing connection to ESP32.")
            esp32.stop()
    else:
        print("\n❌ ERROR: Could not connect to the ESP32.")
        print("Please check the following:")
        print("1. The ESP32_IP address is correct.")
        print("2. The robot is powered on and connected to the same WiFi network.")
        print("3. The ESP32 server program is running.")

if __name__ == "__main__":
    test_camera_navigation() 