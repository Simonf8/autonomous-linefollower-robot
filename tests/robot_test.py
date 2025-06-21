#!/usr/bin/env python3

import time
from main import ESP32Bridge

# Configuration
ESP32_IP = "192.168.2.36"

def test_robot():
    """Test ESP32 robot control from Pi"""
    print("🤖 Testing ESP32 Robot Control from Pi")
    print("=" * 50)
    
    # Create ESP32 connection
    esp32 = ESP32Bridge(ESP32_IP)
    
    print(f"🔌 Connecting to ESP32 at {ESP32_IP}...")
    
    if esp32.start():
        print("✅ Connected successfully!")
        
        try:
            # Test 1: Basic motor test
            print("\n🔧 Test 1: Basic Motor Control")
            
            print("   Moving forward for 3 seconds...")
            esp32.send_motor_speeds(60, 60, 60, 60)
            time.sleep(3)
            
            print("   Stopping...")
            esp32.send_motor_speeds(0, 0, 0, 0)
            time.sleep(1)
            
            print("   Turning right for 2 seconds...")
            esp32.send_motor_speeds(50, -50, 50, -50)
            time.sleep(2)
            
            print("   Stopping...")
            esp32.send_motor_speeds(0, 0, 0, 0)
            time.sleep(1)
            
            print("   Strafing left for 2 seconds...")
            esp32.send_motor_speeds(-60, 60, 60, -60)
            time.sleep(2)
            
            print("   Final stop...")
            esp32.send_motor_speeds(0, 0, 0, 0)
            
            # Test 2: Sensor readings
            print("\n📊 Test 2: Sensor Readings")
            for i in range(5):
                encoders = esp32.get_encoder_ticks()
                line_pos, line_err, sensor_vals = esp32.get_line_sensor_data()
                
                print(f"   Reading {i+1}:")
                print(f"     Encoders: FL={encoders[0]}, FR={encoders[1]}, BL={encoders[2]}, BR={encoders[3]}")
                print(f"     Line: pos={line_pos}, err={line_err}")
                print(f"     Sensors: {sensor_vals}")
                time.sleep(1)
            
            # Test 3: Line following (optional)
            response = input("\n🔍 Test line following? (y/n): ").lower().strip()
            if response == 'y':
                print("\n🔧 Calibrating sensors...")
                print("   Move robot over white and black areas for 10 seconds!")
                esp32.send_calibrate_command()
                time.sleep(10)
                
                print("🚀 Starting line following for 15 seconds...")
                esp32.send_line_follow_command(50)
                
                # Monitor line following
                start_time = time.time()
                while time.time() - start_time < 15:
                    line_pos, line_err, sensor_vals = esp32.get_line_sensor_data()
                    status = "LINE FOUND" if line_pos != -1 else "NO LINE"
                    print(f"   {status}: pos={line_pos}, err={line_err}")
                    time.sleep(1)
                
                # Stop line following
                esp32.send_motor_speeds(0, 0, 0, 0)
                print("✅ Line following test completed")
            
        except KeyboardInterrupt:
            print("\n⚠️  Test interrupted by user")
            esp32.send_motor_speeds(0, 0, 0, 0)
        
        finally:
            esp32.stop()
            print("🔌 Disconnected from ESP32")
            
    else:
        print("❌ Failed to connect to ESP32!")
        print("Check:")
        print("  - ESP32 is powered on")
        print("  - ESP32 is connected to WiFi")
        print(f"  - ESP32 IP is {ESP32_IP}")
        print("  - Port 1234 is not blocked")

if __name__ == '__main__':
    test_robot() 