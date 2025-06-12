#!/usr/bin/env python3
"""
ESP32 Motor Direction Test
Upload this to ESP32 to test if motors are wired correctly
"""

import time
from machine import Pin, PWM

# Motor Pins (same as main code)
LEFT_MOTOR_1 = 18
LEFT_MOTOR_2 = 19
RIGHT_MOTOR_1 = 12
RIGHT_MOTOR_2 = 13

class MotorTest:
    def __init__(self):
        self.left_1 = PWM(Pin(LEFT_MOTOR_1), freq=100)
        self.left_2 = PWM(Pin(LEFT_MOTOR_2), freq=100)
        self.right_1 = PWM(Pin(RIGHT_MOTOR_1), freq=100)
        self.right_2 = PWM(Pin(RIGHT_MOTOR_2), freq=100)
        self.stop()
        print("Motor test initialized")
    
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
    
    def stop(self):
        self.left_1.duty(0)
        self.left_2.duty(0)
        self.right_1.duty(0)
        self.right_2.duty(0)
    
    def test_individual_motors(self):
        speed = 40
        
        print("Testing LEFT motor FORWARD for 2 seconds")
        print("Robot should move forward and slightly to the RIGHT")
        self._set_left_motor(speed, True)
        time.sleep(2)
        self.stop()
        time.sleep(1)
        
        print("Testing RIGHT motor FORWARD for 2 seconds") 
        print("Robot should move forward and slightly to the LEFT")
        self._set_right_motor(speed, True)
        time.sleep(2)
        self.stop()
        time.sleep(1)
        
        print("Testing turn LEFT (left backward, right forward)")
        print("Robot should spin LEFT (counterclockwise)")
        self._set_left_motor(speed, False)
        self._set_right_motor(speed, True)
        time.sleep(2)
        self.stop()
        time.sleep(1)
        
        print("Testing turn RIGHT (left forward, right backward)")
        print("Robot should spin RIGHT (clockwise)")
        self._set_left_motor(speed, True)
        self._set_right_motor(speed, False)
        time.sleep(2)
        self.stop()
        
        print("Motor test complete!")

def main():
    print("ESP32 Motor Direction Test")
    print("=" * 30)
    print("Watch the robot carefully and verify:")
    print("1. Left motor forward → robot veers RIGHT")
    print("2. Right motor forward → robot veers LEFT") 
    print("3. Turn LEFT command → robot spins LEFT")
    print("4. Turn RIGHT command → robot spins RIGHT")
    print()
    print("Starting test in 3 seconds...")
    time.sleep(3)
    
    motor_test = MotorTest()
    motor_test.test_individual_motors()

if __name__ == "__main__":
    main() 