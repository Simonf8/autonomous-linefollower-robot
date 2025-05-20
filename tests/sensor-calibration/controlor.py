# Webots Controller - Line Following HIL
# This controller reads three line sensors from the simulated robot, sends their 
# binary status via serial to an ESP32, and receives the robot state (movement command)
# back from the ESP32 to adjust the wheel speeds.

from controller import Robot
import serial

# Initialize Webots Robot and devices
robot = Robot()
time_step = int(robot.getBasicTimeStep())

# Get device handles (adjust names if different in your robot model)
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))   # Use velocity control mode
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Line sensors (assumed to be e.g. infrared ground sensors returning reflectance)
left_sensor = robot.getDevice('line_sensor_left')
center_sensor = robot.getDevice('line_sensor_center')
right_sensor = robot.getDevice('line_sensor_right')
left_sensor.enable(time_step)
center_sensor.enable(time_step)
right_sensor.enable(time_step)

# Serial port setup: adjust COM port and baud rate to match the ESP32 settings
try:
    ser = serial.Serial(port='COM5', baudrate=115200, timeout=1)  # e.g., COM5 on Windows
except Exception as e:
    print(f"Failed to open serial port: {e}")
    ser = None

current_state = 'S'  # start assuming a 'Stopped/Search' state until ESP32 sends a command

# Define a threshold to decide line detection vs no line (depends on sensor characteristics)
LINE_THRESHOLD = 500  # Example threshold value; may need tuning based on simulation

# Main control loop
while robot.step(time_step) != -1:
    # Read and preprocess sensor values to binary line/no-line
    left_val = left_sensor.getValue()
    center_val = center_sensor.getValue()
    right_val = right_sensor.getValue()
    # Determine if each sensor is on the line (True) or off the line (False)
    line_left = True if left_val < LINE_THRESHOLD else False
    line_center = True if center_val < LINE_THRESHOLD else False
    line_right = True if right_val < LINE_THRESHOLD else False

    # Construct the 3-character sensor message ('1' = line detected, '0' = no line)
    # Note: We use '1' for line present to match the MicroPython logic.
    message = ''
    message += '1' if line_left else '0'
    message += '1' if line_center else '0'
    message += '1' if line_right else '0'
    # Send the sensor message over serial (add newline as terminator)
    if ser:
        try:
            ser.write((message + '\n').encode('utf-8'))
        except Exception as e:
            print(f"Serial write error: {e}")

    # Check if a response (state) was received from the ESP32
    if ser and ser.in_waiting:  # bytes waiting in serial buffer:contentReference[oaicite:5]{index=5}
        try:
            line = ser.readline().decode('utf-8').strip()  # read a line and strip newline
        except Exception as e:
            print(f"Serial read error: {e}")
            line = ''
        if line != '':
            current_state = line  # update current state to the latest message

    # Set wheel speeds based on the current state
    if current_state == 'F':      # Forward
        left_speed = 5.0
        right_speed = 5.0
    elif current_state == 'L':    # Turn Left (sharp turn left)
        left_speed = 0.0
        right_speed = 5.0
    elif current_state == 'R':    # Turn Right (sharp turn right)
        left_speed = 5.0
        right_speed = 0.0
    elif current_state == 'S':    # Search/Stopped (spin in place to find line)
        left_speed = 3.0
        right_speed = -3.0
    else:
        # Unknown state - stop as a safety fallback
        left_speed = 0.0
        right_speed = 0.0

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)
