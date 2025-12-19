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
    fl_pin1 = OutputDevice(5)  # Front Left motor pin 1
    fl_pin2 = OutputDevice(6)  # Front Left motor pin 2
    fr_pin1 = OutputDevice(19)  # Front Right motor pin 1  
    fr_pin2 = OutputDevice(13)  # Front Right motor pin 2
    

    
    
    def stop_all():
        fl_pin1.off()
        fl_pin2.off()
        fr_pin1.off()
        fr_pin2.off()
    
    

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
        
        print("Done! All wheels tested individually.")
        
    except KeyboardInterrupt:
        print("Stopped")
    finally:
        stop_all()

if __name__ == "__main__":
    main() 