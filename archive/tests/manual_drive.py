#!/usr/bin/env python3

import sys
import time
import select
import termios
import tty
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'controllers'))
from main import ESP32Bridge

# Configuration - UPDATE THIS IP WHEN YOU FIND YOUR ESP32
ESP32_IP = "192.168.83.245"  # Change this to your ESP32's actual IP
BASE_SPEED = 60
TURN_SPEED = 50

class ManualRobotDriver:
    def __init__(self, esp32_ip):
        self.esp32_ip = esp32_ip
        self.esp32 = ESP32Bridge(esp32_ip)
        self.running = False
        
    def connect(self):
        """Connect to ESP32"""
        print(f"🔌 Connecting to ESP32 at {self.esp32_ip}...")
        if self.esp32.start():
            print("✅ Connected!")
            return True
        else:
            print("❌ Connection failed!")
            return False
    
    def print_controls(self):
        """Print control instructions"""
        print("\n" + "="*50)
        print("🎮 MANUAL ROBOT CONTROL")
        print("="*50)
        print("MOVEMENT:")
        print("  W/w - Forward")
        print("  S/s - Backward")
        print("  A/a - Turn Left")
        print("  D/d - Turn Right")
        print("  Q/q - Strafe Left")
        print("  E/e - Strafe Right")
        print("  X/x - STOP")
        print("\nAUTO:")
        print("  L/l - Line Following")
        print("  C/c - Calibrate Sensors")
        print("\nINFO:")
        print("  R/r - Show Readings")
        print("  H/h - Help")
        print("  ESC - Exit")
        print("="*50)
    
    def get_key(self):
        """Get single keypress without Enter"""
        if select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return None
    
    def drive_forward(self):
        self.esp32.send_motor_speeds(BASE_SPEED, BASE_SPEED, BASE_SPEED, BASE_SPEED)
        print("🔼 Forward")
    
    def drive_backward(self):
        self.esp32.send_motor_speeds(-BASE_SPEED, -BASE_SPEED, -BASE_SPEED, -BASE_SPEED)
        print("🔽 Backward")
    
    def turn_left(self):
        self.esp32.send_motor_speeds(-TURN_SPEED, TURN_SPEED, -TURN_SPEED, TURN_SPEED)
        print("↺ Turn Left")
    
    def turn_right(self):
        self.esp32.send_motor_speeds(TURN_SPEED, -TURN_SPEED, TURN_SPEED, -TURN_SPEED)
        print("↻ Turn Right")
    
    def strafe_left(self):
        self.esp32.send_motor_speeds(-BASE_SPEED, BASE_SPEED, BASE_SPEED, -BASE_SPEED)
        print("⬅ Strafe Left")
    
    def strafe_right(self):
        self.esp32.send_motor_speeds(BASE_SPEED, -BASE_SPEED, -BASE_SPEED, BASE_SPEED)
        print("➡ Strafe Right")
    
    def stop(self):
        self.esp32.send_motor_speeds(0, 0, 0, 0)
        print("⏹ STOP")
    
    def line_follow(self):
        print("🔍 Line Following Mode")
        self.esp32.send_line_follow_command(50)
    
    def calibrate(self):
        print("🔧 Calibrating sensors...")
        print("   Move robot over white and black areas!")
        self.esp32.send_calibrate_command()
    
    def show_readings(self):
        encoders = self.esp32.get_encoder_ticks()
        line_pos, line_err, sensor_vals = self.esp32.get_line_sensor_data()
        
        print("\n📊 SENSOR READINGS:")
        print(f"   Encoders: FL={encoders[0]}, FR={encoders[1]}, BL={encoders[2]}, BR={encoders[3]}")
        print(f"   Line: pos={line_pos}, err={line_err}")
        print(f"   Sensors: {sensor_vals}")
        print()
    
    def run(self):
        """Main control loop"""
        if not self.connect():
            print("Cannot start - ESP32 not connected!")
            return
        
        # Set terminal to raw mode for single key input
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            tty.setcbreak(sys.stdin.fileno())
            self.print_controls()
            self.running = True
            
            print("\n🚀 Robot ready! Press keys to control...")
            
            while self.running:
                key = self.get_key()
                
                if key:
                    key_lower = key.lower()
                    
                    # Movement commands
                    if key_lower == 'w':
                        self.drive_forward()
                    elif key_lower == 's':
                        self.drive_backward()
                    elif key_lower == 'a':
                        self.turn_left()
                    elif key_lower == 'd':
                        self.turn_right()
                    elif key_lower == 'q':
                        self.strafe_left()
                    elif key_lower == 'e':
                        self.strafe_right()
                    elif key_lower == 'x':
                        self.stop()
                    
                    # Auto commands
                    elif key_lower == 'l':
                        self.line_follow()
                    elif key_lower == 'c':
                        self.calibrate()
                    
                    # Info commands
                    elif key_lower == 'r':
                        self.show_readings()
                    elif key_lower == 'h':
                        self.print_controls()
                    
                    # Exit
                    elif key == '\x1b':  # ESC
                        print("👋 Exiting...")
                        self.running = False
                
                time.sleep(0.05)  # Small delay
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by Ctrl+C")
        
        finally:
            # Restore terminal and stop robot
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            self.stop()
            time.sleep(0.5)
            self.esp32.stop()
            print("🔌 Disconnected")

def main():
    """Main entry point"""
    print("🤖 Manual Robot Driver")
    
    # You can change the IP here or pass it as argument
    if len(sys.argv) > 1:
        esp32_ip = sys.argv[1]
    else:
        esp32_ip = ESP32_IP
    
    print(f"Using ESP32 IP: {esp32_ip}")
    
    driver = ManualRobotDriver(esp32_ip)
    driver.run()

if __name__ == '__main__':
    main() 