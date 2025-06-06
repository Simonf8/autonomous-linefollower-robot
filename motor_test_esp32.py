import time
from machine import Pin, PWM

# -----------------------------------------------------------------------------
# --- SIMPLE MOTOR TEST FOR ESP32 ---
# -----------------------------------------------------------------------------

# Motor pins for 2-wheel drive
LEFT_MOTOR_1 = 5    # Left motor pin 1
LEFT_MOTOR_2 = 23   # Left motor pin 2

RIGHT_MOTOR_1 = 18  # Right motor pin 1
RIGHT_MOTOR_2 = 19  # Right motor pin 2

PWM_FREQ = 1000
TEST_SPEED = 30  # 30% speed for safe testing

# -----------------------------------------------------------------------------
# --- MOTOR CONTROLLER ---
# -----------------------------------------------------------------------------
class TestMotors:
    def __init__(self):
        # Initialize PWM pins
        self.left_1 = PWM(Pin(LEFT_MOTOR_1), freq=PWM_FREQ)
        self.left_2 = PWM(Pin(LEFT_MOTOR_2), freq=PWM_FREQ)
        
        self.right_1 = PWM(Pin(RIGHT_MOTOR_1), freq=PWM_FREQ)
        self.right_2 = PWM(Pin(RIGHT_MOTOR_2), freq=PWM_FREQ)
        
        self.stop()
        print("🔧 Motor test controller initialized")
    
    def _set_left_motor(self, speed, forward=True):
        duty = int(speed * 1023 / 100)
        if forward:
            self.left_1.duty(duty)
            self.left_2.duty(0)
        else:
            self.left_1.duty(0)
            self.left_2.duty(duty)
    
    def _set_right_motor(self, speed, forward=True):
        duty = int(speed * 1023 / 100)
        if forward:
            self.right_1.duty(duty)
            self.right_2.duty(0)
        else:
            self.right_1.duty(0)
            self.right_2.duty(duty)
    
    def forward(self, speed=TEST_SPEED):
        print(f"🔼 FORWARD at {speed}%")
        self._set_left_motor(speed, True)
        self._set_right_motor(speed, True)
    
    def backward(self, speed=TEST_SPEED):
        print(f"🔽 BACKWARD at {speed}%")
        self._set_left_motor(speed, False)
        self._set_right_motor(speed, False)
    
    def left(self, speed=TEST_SPEED):
        print(f"⬅️ LEFT at {speed}%")
        self._set_left_motor(speed, False)  # Left motor backward
        self._set_right_motor(speed, True)  # Right motor forward
    
    def right(self, speed=TEST_SPEED):
        print(f"➡️ RIGHT at {speed}%")
        self._set_left_motor(speed, True)   # Left motor forward
        self._set_right_motor(speed, False) # Right motor backward
    
    def stop(self):
        print("🛑 STOP")
        self.left_1.duty(0)
        self.left_2.duty(0)
        self.right_1.duty(0)
        self.right_2.duty(0)
    
    # Test individual motors
    def test_left_motor_forward(self, speed=TEST_SPEED):
        print(f"🔧 LEFT MOTOR FORWARD at {speed}%")
        self.stop()
        self._set_left_motor(speed, True)
    
    def test_left_motor_backward(self, speed=TEST_SPEED):
        print(f"🔧 LEFT MOTOR BACKWARD at {speed}%")
        self.stop()
        self._set_left_motor(speed, False)
    
    def test_right_motor_forward(self, speed=TEST_SPEED):
        print(f"🔧 RIGHT MOTOR FORWARD at {speed}%")
        self.stop()
        self._set_right_motor(speed, True)
    
    def test_right_motor_backward(self, speed=TEST_SPEED):
        print(f"🔧 RIGHT MOTOR BACKWARD at {speed}%")
        self.stop()
        self._set_right_motor(speed, False)

# -----------------------------------------------------------------------------
# --- MAIN TEST PROGRAM ---
# -----------------------------------------------------------------------------
def main():
    print("🤖 ESP32 Motor Test Script")
    print("=" * 40)
    print("This will test all motor directions")
    print("Watch your robot and note which way it moves!")
    print("=" * 40)
    
    motors = TestMotors()
    
    print("\n🧪 Starting Motor Tests...")
    print("Each test runs for 2 seconds")
    
    # Test sequence
    tests = [
        ("FORWARD", motors.forward),
        ("BACKWARD", motors.backward), 
        ("LEFT", motors.left),
        ("RIGHT", motors.right),
        ("LEFT MOTOR ONLY - FORWARD", motors.test_left_motor_forward),
        ("LEFT MOTOR ONLY - BACKWARD", motors.test_left_motor_backward),
        ("RIGHT MOTOR ONLY - FORWARD", motors.test_right_motor_forward),
        ("RIGHT MOTOR ONLY - BACKWARD", motors.test_right_motor_backward),
    ]
    
    for test_name, test_func in tests:
        print(f"\n⏰ Testing: {test_name}")
        print("   (Watch which direction the robot moves!)")
        
        # Start the movement
        test_func()
        
        # Run for 2 seconds
        time.sleep(2)
        
        # Stop
        motors.stop()
        print("   ✅ Test complete")
        
        # Wait before next test
        time.sleep(1)
    
    print("\n🎉 All tests complete!")
    print("\n📝 Analysis:")
    print("   - If FORWARD went backward, swap ALL motor wires")
    print("   - If LEFT went right, swap left/right motor definitions")
    print("   - If individual motors spun wrong way, swap that motor's wires")
    
    motors.stop()

if __name__ == "__main__":
    main() 