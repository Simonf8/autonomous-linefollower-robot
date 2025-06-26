from machine import Pin, PWM
import time

# Initialize servo on pin 2 (you can change this)
servo = PWM(Pin(2))
servo.freq(50)  # 50Hz frequency for servo control

def set_servo_angle(angle):
    """
    Set servo to specific angle (0-90 degrees only)
    """
    # Limit angle to 0-90 degrees
    angle = max(0, min(90, angle))
    
    # Convert angle to duty cycle for 90-degree range
    # Servo typically needs 1-1.5ms pulse width for 90 degrees
    # At 50Hz: 1ms = ~40 duty, 1.5ms = ~77 duty
    min_duty = 0   # ~1ms pulse width (0 degrees)
    max_duty = 77   # ~1.5ms pulse width (90 degrees)
    
    duty = int(min_duty + (max_duty - min_duty) * angle / 90)
    servo.duty(duty)
    print(f"Servo moved to {angle} degrees")

def sweep_servo():
    """
    Sweep servo from 0 to 90 degrees and back
    """
    # Sweep from 0 to 90
    for angle in range(0, 91, 5):
        set_servo_angle(angle)
        time.sleep(0.1)
    
    # Sweep from 90 to 0
    for angle in range(90, -1, -5):
        set_servo_angle(angle)
        time.sleep(0.1)

# Main program
try:
    print("Starting servo control (0-90 degrees)...")
    
    # Set servo to center position (45 degrees)
    set_servo_angle(0)
    time.sleep(1)
    
    # Mode 1: Continuous sweeping
    print("Mode 1: Continuous sweeping (0-90°)")
    for cycle in range(3):
        sweep_servo()
        time.sleep(0.5)
    
    time.sleep(2)
    
    # Mode 2: Step positions
    print("Mode 2: Moving to specific positions")
    positions = [0, 15, 30, 45, 60, 75, 90]
    for pos in positions:
        print(f"Moving to {pos} degrees")
        set_servo_angle(pos)
        time.sleep(1)
    
    time.sleep(1)
    
    # Mode 3: Smooth movement
    print("Mode 3: Smooth movement demonstration")
    for angle in range(0, 91, 1):
        set_servo_angle(angle)
        time.sleep(0.05)
    
    for angle in range(90, -1, -1):
        set_servo_angle(angle)
        time.sleep(0.05)
    
    # Return to center
    set_servo_angle(45)
    
except KeyboardInterrupt:
    print("Stopping servo...")
    servo.deinit()  # Clean up PWM
    print("Servo stopped.")

