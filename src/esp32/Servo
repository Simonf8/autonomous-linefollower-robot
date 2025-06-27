import machine
import time

# Pin definitions (change as needed for your board)
SERVO_PIN = 34         # PWM-capable pin for servo
MAGNET_PIN = 35        # Output pin for magnet control
SWITCH1_PIN = 32       # Input pin for micro-switch 1
SWITCH2_PIN = 33       # Input pin for micro-switch 2

# Setup pins
servo = machine.PWM(machine.Pin(SERVO_PIN), freq=50)
magnet = machine.Pin(MAGNET_PIN, machine.Pin.OUT)
switch1 = machine.Pin(SWITCH1_PIN, machine.Pin.IN, machine.Pin.PULL_UP)
switch2 = machine.Pin(SWITCH2_PIN, machine.Pin.IN, machine.Pin.PULL_UP)

# Helper to read switches (returns True if pressed)
def read_switches():
    s1 = switch1.value() == 0  # Pressed if LOW
    s2 = switch2.value() == 0
    return s1, s2

# Servo angle to duty conversion (for 0-180 degrees)
def set_servo_angle(angle):
    # For most servos: 0 deg = 0.5ms, 180 deg = 2.5ms pulse width
    min_us = 500
    max_us = 2500
    us = min_us + (max_us - min_us) * angle // 180
    duty = int(us * 1023 // 20000)  # 20ms period (50Hz), 10-bit resolution
    servo.duty(duty)

# Wait until either switch is pressed
def wait_for_switch():
    while True:
        s1, s2 = read_switches()
        if s1 or s2:
            return
        time.sleep_ms(10)

# Wait until both switches are released
def wait_for_release():
    while True:
        s1, s2 = read_switches()
        if not s1 and not s2:
            return
        time.sleep_ms(10)

def main():
    # Initial positions
    set_servo_angle(0)
    magnet.value(0)  # Magnet off

    while True:
        wait_for_switch()
        print("Switch activated! Waiting 5 seconds before action...")
        time.sleep(5)
        print("Activating magnet and moving servo to 130 degrees.")
        magnet.value(1)
        set_servo_angle(130)
        time.sleep(5)
        print("Deactivating magnet and returning servo to 0 degrees.")
        magnet.value(0)
        set_servo_angle(0)
        wait_for_release()  # Wait for switches to be released before next cycle

if __name__ == "__main__":
    main() 
