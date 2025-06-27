#!/usr/bin/env python3
"""
Pi 5 Motor Test - Uses gpiozero for better Pi 5 compatibility
"""

import time

# Try gpiozero first (better for Pi 5), then fall back to RPi.GPIO
try:
    from gpiozero import OutputDevice
    print("Using gpiozero library (Pi 5 compatible)")
    
    # Motor pins using gpiozero
    fl_pin1 = OutputDevice(17)  # Front Left motor pin 1
    fl_pin2 = OutputDevice(27)  # Front Left motor pin 2
    fr_pin1 = OutputDevice(22)  # Front Right motor pin 1  
    fr_pin2 = OutputDevice(24)  # Front Right motor pin 2
    bl_pin1 = OutputDevice(23)  # Back Left motor pin 1
    bl_pin2 = OutputDevice(25)  # Back Left motor pin 2
    br_pin1 = OutputDevice(9)   # Back Right motor pin 1
    br_pin2 = OutputDevice(11)  # Back Right motor pin 2
    
    def stop_all():
        print("  STOP: All motors off")
        fl_pin1.off()
        fl_pin2.off()
        fr_pin1.off()
        fr_pin2.off()
        bl_pin1.off()
        bl_pin2.off()
        br_pin1.off()
        br_pin2.off()
    
    def forward():
        print("  FORWARD: FL+, FR+, BL+, BR+")
        fl_pin1.on()
        fl_pin2.off()
        fr_pin1.on()
        fr_pin2.off()
        bl_pin1.on()
        bl_pin2.off()
        br_pin1.on()
        br_pin2.off()
    
    def backward():
        print("  BACKWARD: FL-, FR-, BL-, BR-")
        fl_pin1.off()
        fl_pin2.on()
        fr_pin1.off()
        fr_pin2.on()
        bl_pin1.off()
        bl_pin2.on()
        br_pin1.off()
        br_pin2.on()
    
    def turn_left():
        print("  LEFT: FL-, FR+, BL-, BR+")
        fl_pin1.off()
        fl_pin2.on()
        fr_pin1.on()
        fr_pin2.off()
        bl_pin1.off()
        bl_pin2.on()
        br_pin1.on()
        br_pin2.off()
    
    def turn_right():
        print("  RIGHT: FL+, FR-, BL+, BR-")
        fl_pin1.on()
        fl_pin2.off()
        fr_pin1.off()
        fr_pin2.on()
        bl_pin1.on()
        bl_pin2.off()
        br_pin1.off()
        br_pin2.on()
    
    def cleanup():
        stop_all()

except ImportError:
    print("gpiozero not available, falling back to simulation")
    
    def stop_all():
        print("  STOP: All motors off")
    
    def forward():
        print("  FORWARD: FL+, FR+, BL+, BR+")
    
    def backward():
        print("  BACKWARD: FL-, FR-, BL-, BR-")
    
    def turn_left():
        print("  LEFT: FL-, FR+, BL-, BR+")
    
    def turn_right():
        print("  RIGHT: FL+, FR-, BL+, BR-")
    
    def cleanup():
        pass

def main():
    try:
        print("Testing motors...")
        
        print("Forward")
        forward()
        time.sleep(2)
        
        stop_all()
        time.sleep(1)
        
        print("Backward")
        backward()
        time.sleep(2)
        
        stop_all()
        time.sleep(1)
        
        print("Left")
        turn_left()
        time.sleep(2)
        
        stop_all()
        time.sleep(1)
        
        print("Right")
        turn_right()
        time.sleep(2)
        
        stop_all()
        print("Done!")
        
    except KeyboardInterrupt:
        print("Stopped")
    finally:
        cleanup()

if __name__ == "__main__":
    main() 