#!/usr/bin/env python3
"""
Individual Wheel Test - Test each motor separately
"""

import time

# Try gpiozero first (better for Pi 5), then fall back to RPi.GPIO
try:
    from gpiozero import OutputDevice
    print("Using gpiozero library (Pi 5 compatible)")
    
    # Motor pins using gpiozero
    fl_pin1 = OutputDevice(22)  # Front Left motor pin 1
    fl_pin2 = OutputDevice(23)  # Front Left motor pin 2
    fr_pin1 = OutputDevice(27)  # Front Right motor pin 1  
    fr_pin2 = OutputDevice(17)  # Front Right motor pin 2
    bl_pin1 = OutputDevice(21)  # Back Left motor pin 1
    bl_pin2 = OutputDevice(20)  # Back Left motor pin 2
    br_pin1 = OutputDevice(26)   # Back Right motor pin 1
    br_pin2 = OutputDevice(19)  # Back Right motor pin 2
    
    def stop_all():
        fl_pin1.off()
        fl_pin2.off()
        fr_pin1.off()
        fr_pin2.off()
        bl_pin1.off()
        bl_pin2.off()
        br_pin1.off()
        br_pin2.off()
    
    def test_front_left_forward():
        print("  FL FORWARD")
        fl_pin1.on()
        fl_pin2.off()
    
    def test_front_left_backward():
        print("  FL BACKWARD")
        fl_pin1.off()
        fl_pin2.on()
    
    def test_front_right_forward():
        print("  FR FORWARD")
        fr_pin1.on()
        fr_pin2.off()
    
    def test_front_right_backward():
        print("  FR BACKWARD")
        fr_pin1.off()
        fr_pin2.on()
    
    def test_back_left_forward():
        print("  BL FORWARD")
        bl_pin1.on()
        bl_pin2.off()
    
    def test_back_left_backward():
        print("  BL BACKWARD")
        bl_pin1.off()
        bl_pin2.on()
    
    def test_back_right_forward():
        print("  BR FORWARD")
        br_pin1.on()
        br_pin2.off()
    
    def test_back_right_backward():
        print("  BR BACKWARD")
        br_pin1.off()
        br_pin2.on()

except ImportError:
    print("gpiozero not available, running in simulation")
    
    def stop_all():
        pass
    
    def test_front_left_forward():
        print("  FL FORWARD")
    
    def test_front_left_backward():
        print("  FL BACKWARD")
    
    def test_front_right_forward():
        print("  FR FORWARD")
    
    def test_front_right_backward():
        print("  FR BACKWARD")
    
    def test_back_left_forward():
        print("  BL FORWARD")
    
    def test_back_left_backward():
        print("  BL BACKWARD")
    
    def test_back_right_forward():
        print("  BR FORWARD")
    
    def test_back_right_backward():
        print("  BR BACKWARD")

def main():
    try:
        print("Testing each wheel individually...")
        
        # Test Front Left
        print("Front Left Forward")
        test_front_left_forward()
        time.sleep(2)
        stop_all()
        time.sleep(1)
        
        print("Front Left Backward")
        test_front_left_backward()
        time.sleep(2)
        stop_all()
        time.sleep(1)
        
        # Test Front Right
        print("Front Right Forward")
        test_front_right_forward()
        time.sleep(2)
        stop_all()
        time.sleep(1)
        
        print("Front Right Backward")
        test_front_right_backward()
        time.sleep(2)
        stop_all()
        time.sleep(1)
        
        # Test Back Left
        print("Back Left Forward")
        test_back_left_forward()
        time.sleep(2)
        stop_all()
        time.sleep(1)
        
        print("Back Left Backward")
        test_back_left_backward()
        time.sleep(2)
        stop_all()
        time.sleep(1)
        
        # Test Back Right
        print("Back Right Forward")
        test_back_right_forward()
        time.sleep(2)
        stop_all()
        time.sleep(1)
        
        print("Back Right Backward")
        test_back_right_backward()
        time.sleep(2)
        stop_all()
        
        print("Done! All wheels tested individually.")
        
    except KeyboardInterrupt:
        print("Stopped")
    finally:
        stop_all()

if __name__ == "__main__":
    main() 