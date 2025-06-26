from machine import Pin
import time


motor1_pin1 = Pin(1, Pin.OUT)
motor1_pin2 = Pin(2, Pin.OUT)
motor2_pin1 = Pin(41, Pin.OUT)
motor2_pin2 = Pin(42, Pin.OUT)


motor3_pin1 = Pin(39, Pin.OUT)
motor3_pin2 = Pin(40, Pin.OUT)
motor4_pin1 = Pin(47, Pin.OUT)
motor4_pin2 = Pin(48, Pin.OUT)

def stop_all():
    motor1_pin1.off()
    motor1_pin2.off()
    motor2_pin1.off()
    motor2_pin2.off()
    motor3_pin1.off()
    motor3_pin2.off()
    motor4_pin1.off()
    motor4_pin2.off()

def motor1_forward():
    motor1_pin1.on()
    motor1_pin2.off()

def motor1_backward():
    motor1_pin1.off()
    motor1_pin2.on()

def motor2_forward():
    motor2_pin1.on()
    motor2_pin2.off()

def motor2_backward():
    motor2_pin1.off()
    motor2_pin2.on()

def motor3_forward():
    motor3_pin1.on()
    motor3_pin2.off()

def motor3_backward():
    motor3_pin1.off()
    motor3_pin2.on()

def motor4_forward():
    motor4_pin1.on()
    motor4_pin2.off()

def motor4_backward():
    motor4_pin1.off()
    motor4_pin2.on()


try:
    print("Testing 4 motors...")
    stop_all()
    time.sleep(1)
    
    # Test each motor forward
    print("Motor 1 forward")
    motor1_forward()
    time.sleep(4)
    stop_all()
    time.sleep(0.5)
    
    print("Motor 2 forward")
    motor2_forward()
    time.sleep(4)
    stop_all()
    time.sleep(0.2)
    
    print("Motor 3 forward")
    motor3_forward()
    time.sleep(4)
    stop_all()
    time.sleep(0.5)
    
    print("Motor 4 forward")
    motor4_forward()
    time.sleep(4)
    stop_all()
    time.sleep(0.5)
    
    
    print("Motor 1 backward")
    motor1_backward()
    time.sleep(4)
    stop_all()
    time.sleep(0.5)
    
    print("Motor 2 backward")
    motor2_backward()
    time.sleep(4)
    stop_all()
    time.sleep(0.5)
    
    print("Motor 3 backward")
    motor3_backward()
    time.sleep(4)
    stop_all()
    time.sleep(0.5)
    
    print("Motor 4 backward")
    motor4_backward()
    time.sleep(4)
    stop_all()
    
    print("All motors tested!")
    
except KeyboardInterrupt:
    stop_all()
