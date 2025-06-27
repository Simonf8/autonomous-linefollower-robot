#!/usr/bin/env python3
"""
Standalone Motor Test - No imports from other files
"""

import time

# Try to import GPIO, if it fails use simulation mode
try:
    import RPi.GPIO as GPIO
    SIMULATION_MODE = False
    print("GPIO library loaded - hardware mode")
except (ImportError, RuntimeError) as e:
    print(f"GPIO error: {e}")
    print("Running in simulation mode - no actual GPIO control")
    SIMULATION_MODE = True
    class MockGPIO:
        BCM = "BCM"
        OUT = "OUT"
        HIGH = "HIGH"
        LOW = "LOW"
        @staticmethod
        def setmode(mode): pass
        @staticmethod
        def setwarnings(enable): pass
        @staticmethod
        def setup(pin, mode): pass
        @staticmethod
        def output(pin, state): pass
        @staticmethod
        def cleanup(): pass
    GPIO = MockGPIO()

# Motor pins
FL_PIN1 = 17  # Front Left motor pin 1
FL_PIN2 = 27  # Front Left motor pin 2
FR_PIN1 = 22  # Front Right motor pin 1  
FR_PIN2 = 24  # Front Right motor pin 2
BL_PIN1 = 23  # Back Left motor pin 1
BL_PIN2 = 25  # Back Left motor pin 2
BR_PIN1 = 9   # Back Right motor pin 1
BR_PIN2 = 11  # Back Right motor pin 2

def setup_motors():
    global SIMULATION_MODE, GPIO
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup all pins
        pins = [FL_PIN1, FL_PIN2, FR_PIN1, FR_PIN2, BL_PIN1, BL_PIN2, BR_PIN1, BR_PIN2]
        for pin in pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        print("GPIO setup successful")
    except RuntimeError as e:
        print(f"GPIO setup failed: {e}")
        print("Switching to simulation mode")
        SIMULATION_MODE = True
        # Replace GPIO with mock
        class MockGPIO:
            BCM = "BCM"
            OUT = "OUT"
            HIGH = "HIGH"
            LOW = "LOW"
            @staticmethod
            def setmode(mode): pass
            @staticmethod
            def setwarnings(enable): pass
            @staticmethod
            def setup(pin, mode): pass
            @staticmethod
            def output(pin, state): pass
            @staticmethod
            def cleanup(): pass
        GPIO = MockGPIO()

def stop_all():
    if SIMULATION_MODE:
        print("  STOP: All motors off")
    pins = [FL_PIN1, FL_PIN2, FR_PIN1, FR_PIN2, BL_PIN1, BL_PIN2, BR_PIN1, BR_PIN2]
    for pin in pins:
        GPIO.output(pin, GPIO.LOW)

def forward():
    if SIMULATION_MODE:
        print("  FORWARD: FL+, FR+, BL+, BR+")
    # All motors forward
    GPIO.output(FL_PIN1, GPIO.HIGH)
    GPIO.output(FL_PIN2, GPIO.LOW)
    GPIO.output(FR_PIN1, GPIO.HIGH)
    GPIO.output(FR_PIN2, GPIO.LOW)
    GPIO.output(BL_PIN1, GPIO.HIGH)
    GPIO.output(BL_PIN2, GPIO.LOW)
    GPIO.output(BR_PIN1, GPIO.HIGH)
    GPIO.output(BR_PIN2, GPIO.LOW)

def backward():
    if SIMULATION_MODE:
        print("  BACKWARD: FL-, FR-, BL-, BR-")
    # All motors backward
    GPIO.output(FL_PIN1, GPIO.LOW)
    GPIO.output(FL_PIN2, GPIO.HIGH)
    GPIO.output(FR_PIN1, GPIO.LOW)
    GPIO.output(FR_PIN2, GPIO.HIGH)
    GPIO.output(BL_PIN1, GPIO.LOW)
    GPIO.output(BL_PIN2, GPIO.HIGH)
    GPIO.output(BR_PIN1, GPIO.LOW)
    GPIO.output(BR_PIN2, GPIO.HIGH)

def turn_left():
    if SIMULATION_MODE:
        print("  LEFT: FL-, FR+, BL-, BR+")
    # Left motors backward, right motors forward
    GPIO.output(FL_PIN1, GPIO.LOW)
    GPIO.output(FL_PIN2, GPIO.HIGH)
    GPIO.output(FR_PIN1, GPIO.HIGH)
    GPIO.output(FR_PIN2, GPIO.LOW)
    GPIO.output(BL_PIN1, GPIO.LOW)
    GPIO.output(BL_PIN2, GPIO.HIGH)
    GPIO.output(BR_PIN1, GPIO.HIGH)
    GPIO.output(BR_PIN2, GPIO.LOW)

def turn_right():
    if SIMULATION_MODE:
        print("  RIGHT: FL+, FR-, BL+, BR-")
    # Left motors forward, right motors backward
    GPIO.output(FL_PIN1, GPIO.HIGH)
    GPIO.output(FL_PIN2, GPIO.LOW)
    GPIO.output(FR_PIN1, GPIO.LOW)
    GPIO.output(FR_PIN2, GPIO.HIGH)
    GPIO.output(BL_PIN1, GPIO.HIGH)
    GPIO.output(BL_PIN2, GPIO.LOW)
    GPIO.output(BR_PIN1, GPIO.LOW)
    GPIO.output(BR_PIN2, GPIO.HIGH)

def main():
    setup_motors()
    
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
        stop_all()
        GPIO.cleanup()

if __name__ == "__main__":
    main() 