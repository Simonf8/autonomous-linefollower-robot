"""
ESP32 Line Following Robot Controller - MicroPython Version
Supports 4 wheels with encoders and dual motor drivers
Compatible with Raspberry Pi vision system

Hardware Requirements:
- ESP32 DevKit
- 2x Motor Drivers (L298N or similar)
- 4x DC Motors with encoders
- WiFi connection

Author: AI Assistant
Date: 2024
"""

import network
import socket
import time
import json
from machine import Pin, PWM, Timer
import _thread
import gc

# -----------------------------------------------------------------------------
# --- CONFIGURATION ---
# -----------------------------------------------------------------------------

# WiFi Configuration
WIFI_SSID = "YOUR_WIFI_SSID"        # Replace with your WiFi SSID
WIFI_PASSWORD = "YOUR_WIFI_PASSWORD" # Replace with your WiFi password
SERVER_PORT = 1234

# Motor Driver Pins (L298N Configuration)
# Motor Driver 1 (Left Side)
LEFT_MOTOR_PWM_A = 25     # ENA pin
LEFT_MOTOR_IN1 = 26       # IN1 pin
LEFT_MOTOR_IN2 = 27       # IN2 pin
LEFT_MOTOR_PWM_B = 14     # ENB pin  
LEFT_MOTOR_IN3 = 12       # IN3 pin
LEFT_MOTOR_IN4 = 13       # IN4 pin

# Motor Driver 2 (Right Side)
RIGHT_MOTOR_PWM_A = 32    # ENA pin
RIGHT_MOTOR_IN1 = 33      # IN1 pin
RIGHT_MOTOR_IN2 = 25      # IN2 pin  
RIGHT_MOTOR_PWM_B = 26    # ENB pin
RIGHT_MOTOR_IN3 = 27      # IN3 pin
RIGHT_MOTOR_IN4 = 14      # IN4 pin

# Encoder Pins
LEFT_ENCODER_A_PIN = 18   # Left front motor encoder A
LEFT_ENCODER_B_PIN = 19   # Left front motor encoder B
LEFT_ENCODER_A2_PIN = 21  # Left rear motor encoder A
LEFT_ENCODER_B2_PIN = 22  # Left rear motor encoder B

RIGHT_ENCODER_A_PIN = 16  # Right front motor encoder A  
RIGHT_ENCODER_B_PIN = 17  # Right front motor encoder B
RIGHT_ENCODER_A2_PIN = 4  # Right rear motor encoder A
RIGHT_ENCODER_B2_PIN = 2  # Right rear motor encoder B

# Status LED
STATUS_LED_PIN = 2

# Motor Speed Settings
SPEED_FAST = 1023      # Maximum speed (10-bit PWM)
SPEED_NORMAL = 700     # Normal cruising speed  
SPEED_SLOW = 400       # Slow speed for precise movements
SPEED_TURN = 600       # Speed for turning
SPEED_STOP = 0         # Stop

# PWM Configuration
PWM_FREQUENCY = 1000   # 1kHz

# Encoder Configuration
ENCODER_PPR = 20       # Pulses per revolution (adjust for your encoders)
WHEEL_DIAMETER = 6.5   # Wheel diameter in cm
WHEEL_CIRCUMFERENCE = 3.14159 * WHEEL_DIAMETER

# Command timeout
COMMAND_TIMEOUT = 2000  # 2 seconds in milliseconds

# -----------------------------------------------------------------------------
# --- GLOBAL VARIABLES ---
# -----------------------------------------------------------------------------

# WiFi and server
wlan = None
server_socket = None
client_socket = None

# Motor control pins
left_motor_pwm_a = None
left_motor_in1 = None
left_motor_in2 = None
left_motor_pwm_b = None
left_motor_in3 = None
left_motor_in4 = None

right_motor_pwm_a = None
right_motor_in1 = None
right_motor_in2 = None
right_motor_pwm_b = None
right_motor_in3 = None
right_motor_in4 = None

# Encoder pins
left_encoder_a = None
left_encoder_b = None
left_encoder_a2 = None
left_encoder_b2 = None
right_encoder_a = None
right_encoder_b = None
right_encoder_a2 = None
right_encoder_b2 = None

# Status LED
status_led = None

# Encoder counters
left_encoder_count = 0
left_encoder_count2 = 0
right_encoder_count = 0
right_encoder_count2 = 0

# Motor states
left_motor_speed = 0
right_motor_speed = 0
left_motor_direction = "STOP"
right_motor_direction = "STOP"

# Command tracking
last_command = ""
last_command_time = 0
last_heartbeat = 0

# Status tracking
robot_status = "Initializing"
is_connected = False

# -----------------------------------------------------------------------------
# --- ENCODER INTERRUPT HANDLERS ---
# -----------------------------------------------------------------------------

def left_encoder_a_handler(pin):
    global left_encoder_count
    left_encoder_count += 1

def left_encoder_a2_handler(pin):
    global left_encoder_count2
    left_encoder_count2 += 1

def right_encoder_a_handler(pin):
    global right_encoder_count
    right_encoder_count += 1

def right_encoder_a2_handler(pin):
    global right_encoder_count2
    right_encoder_count2 += 1

# -----------------------------------------------------------------------------
# --- SETUP FUNCTIONS ---
# -----------------------------------------------------------------------------

def setup_pins():
    """Initialize all GPIO pins and PWM"""
    global left_motor_pwm_a, left_motor_in1, left_motor_in2, left_motor_pwm_b
    global left_motor_in3, left_motor_in4, right_motor_pwm_a, right_motor_in1
    global right_motor_in2, right_motor_pwm_b, right_motor_in3, right_motor_in4
    global status_led
    
    print("🔧 Setting up GPIO pins...")
    
    # Left motor driver pins
    left_motor_pwm_a = PWM(Pin(LEFT_MOTOR_PWM_A), freq=PWM_FREQUENCY)
    left_motor_in1 = Pin(LEFT_MOTOR_IN1, Pin.OUT)
    left_motor_in2 = Pin(LEFT_MOTOR_IN2, Pin.OUT)
    left_motor_pwm_b = PWM(Pin(LEFT_MOTOR_PWM_B), freq=PWM_FREQUENCY)
    left_motor_in3 = Pin(LEFT_MOTOR_IN3, Pin.OUT)
    left_motor_in4 = Pin(LEFT_MOTOR_IN4, Pin.OUT)
    
    # Right motor driver pins
    right_motor_pwm_a = PWM(Pin(RIGHT_MOTOR_PWM_A), freq=PWM_FREQUENCY)
    right_motor_in1 = Pin(RIGHT_MOTOR_IN1, Pin.OUT)
    right_motor_in2 = Pin(RIGHT_MOTOR_IN2, Pin.OUT)
    right_motor_pwm_b = PWM(Pin(RIGHT_MOTOR_PWM_B), freq=PWM_FREQUENCY)
    right_motor_in3 = Pin(RIGHT_MOTOR_IN3, Pin.OUT)
    right_motor_in4 = Pin(RIGHT_MOTOR_IN4, Pin.OUT)
    
    # Status LED
    status_led = Pin(STATUS_LED_PIN, Pin.OUT)
    
    # Initialize motors to stop
    stop_all_motors()
    print("✅ GPIO pins initialized")

def setup_encoders():
    """Initialize encoder pins with interrupts"""
    global left_encoder_a, left_encoder_b, left_encoder_a2, left_encoder_b2
    global right_encoder_a, right_encoder_b, right_encoder_a2, right_encoder_b2
    
    print("🔄 Setting up encoders...")
    
    # Left encoders
    left_encoder_a = Pin(LEFT_ENCODER_A_PIN, Pin.IN, Pin.PULL_UP)
    left_encoder_b = Pin(LEFT_ENCODER_B_PIN, Pin.IN, Pin.PULL_UP)
    left_encoder_a2 = Pin(LEFT_ENCODER_A2_PIN, Pin.IN, Pin.PULL_UP)
    left_encoder_b2 = Pin(LEFT_ENCODER_B2_PIN, Pin.IN, Pin.PULL_UP)
    
    # Right encoders
    right_encoder_a = Pin(RIGHT_ENCODER_A_PIN, Pin.IN, Pin.PULL_UP)
    right_encoder_b = Pin(RIGHT_ENCODER_B_PIN, Pin.IN, Pin.PULL_UP)
    right_encoder_a2 = Pin(RIGHT_ENCODER_A2_PIN, Pin.IN, Pin.PULL_UP)
    right_encoder_b2 = Pin(RIGHT_ENCODER_B2_PIN, Pin.IN, Pin.PULL_UP)
    
    # Setup interrupts
    left_encoder_a.irq(trigger=Pin.IRQ_RISING, handler=left_encoder_a_handler)
    left_encoder_a2.irq(trigger=Pin.IRQ_RISING, handler=left_encoder_a2_handler)
    right_encoder_a.irq(trigger=Pin.IRQ_RISING, handler=right_encoder_a_handler)
    right_encoder_a2.irq(trigger=Pin.IRQ_RISING, handler=right_encoder_a2_handler)
    
    print("✅ Encoders initialized")

def setup_wifi():
    """Connect to WiFi network"""
    global wlan, is_connected
    
    print("🌐 Connecting to WiFi...")
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    
    if not wlan.isconnected():
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)
        attempts = 0
        while not wlan.isconnected() and attempts < 20:
            print(".", end="")
            time.sleep(0.5)
            attempts += 1
    
    if wlan.isconnected():
        print(f"\n✅ WiFi Connected!")
        print(f"📍 IP Address: {wlan.ifconfig()[0]}")
        is_connected = True
        return True
    else:
        print(f"\n❌ WiFi Connection Failed!")
        return False

def setup_server():
    """Setup TCP server"""
    global server_socket
    
    print("📡 Setting up server...")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('', SERVER_PORT))
    server_socket.listen(1)
    
    print(f"✅ Server listening on port {SERVER_PORT}")
    print(f"🌐 Connect to: {wlan.ifconfig()[0]}:{SERVER_PORT}")

# -----------------------------------------------------------------------------
# --- MOTOR CONTROL FUNCTIONS ---
# -----------------------------------------------------------------------------

def set_left_motors(direction, speed):
    """Control left side motors"""
    global left_motor_speed, left_motor_direction
    
    left_motor_speed = speed
    left_motor_direction = direction
    
    # Left front motor (Motor A)
    if direction == "FORWARD":
        left_motor_in1.value(1)
        left_motor_in2.value(0)
    elif direction == "BACKWARD":
        left_motor_in1.value(0)
        left_motor_in2.value(1)
    else:  # STOP
        left_motor_in1.value(0)
        left_motor_in2.value(0)
    
    left_motor_pwm_a.duty(speed)
    
    # Left rear motor (Motor B)
    if direction == "FORWARD":
        left_motor_in3.value(1)
        left_motor_in4.value(0)
    elif direction == "BACKWARD":
        left_motor_in3.value(0)
        left_motor_in4.value(1)
    else:  # STOP
        left_motor_in3.value(0)
        left_motor_in4.value(0)
    
    left_motor_pwm_b.duty(speed)

def set_right_motors(direction, speed):
    """Control right side motors"""
    global right_motor_speed, right_motor_direction
    
    right_motor_speed = speed
    right_motor_direction = direction
    
    # Right front motor (Motor A)
    if direction == "FORWARD":
        right_motor_in1.value(1)
        right_motor_in2.value(0)
    elif direction == "BACKWARD":
        right_motor_in1.value(0)
        right_motor_in2.value(1)
    else:  # STOP
        right_motor_in1.value(0)
        right_motor_in2.value(0)
    
    right_motor_pwm_a.duty(speed)
    
    # Right rear motor (Motor B)
    if direction == "FORWARD":
        right_motor_in3.value(1)
        right_motor_in4.value(0)
    elif direction == "BACKWARD":
        right_motor_in3.value(0)
        right_motor_in4.value(1)
    else:  # STOP
        right_motor_in3.value(0)
        right_motor_in4.value(0)
    
    right_motor_pwm_b.duty(speed)

def move_forward(speed):
    """Move robot forward"""
    set_left_motors("FORWARD", speed)
    set_right_motors("FORWARD", speed)
    print("⬆️ Moving Forward")

def turn_left(speed):
    """Turn robot left using differential steering"""
    left_speed = int(speed * 0.3)  # Slow left side
    right_speed = speed            # Normal right side
    
    set_left_motors("FORWARD", left_speed)
    set_right_motors("FORWARD", right_speed)
    print("⬅️ Turning Left")

def turn_right(speed):
    """Turn robot right using differential steering"""
    left_speed = speed             # Normal left side
    right_speed = int(speed * 0.3) # Slow right side
    
    set_left_motors("FORWARD", left_speed)
    set_right_motors("FORWARD", right_speed)
    print("➡️ Turning Right")

def stop_all_motors():
    """Stop all motors"""
    set_left_motors("STOP", 0)
    set_right_motors("STOP", 0)
    print("⏹️ All Motors Stopped")

# -----------------------------------------------------------------------------
# --- COMMAND PROCESSING ---
# -----------------------------------------------------------------------------

def parse_speed_command(speed_cmd):
    """Parse speed command and return PWM value"""
    speed_map = {
        'F': SPEED_FAST,
        'N': SPEED_NORMAL,
        'S': SPEED_SLOW,
        'T': SPEED_TURN,
        'H': SPEED_STOP
    }
    return speed_map.get(speed_cmd, SPEED_STOP)

def execute_movement(speed, direction):
    """Execute movement command"""
    print(f"🚗 Executing: Speed={speed}, Direction={direction}")
    
    if direction == "FORWARD":
        move_forward(speed)
    elif direction == "LEFT":
        turn_left(speed)
    elif direction == "RIGHT":
        turn_right(speed)
    elif speed == SPEED_STOP:
        stop_all_motors()
    else:
        print(f"⚠️ Unknown direction: {direction}")
        stop_all_motors()

def process_command(command):
    """Process incoming command"""
    global last_command, last_command_time, last_heartbeat
    
    print(f"📨 Received: {command}")
    
    # Parse command format: "SPEED:DIRECTION"
    try:
        parts = command.split(':')
        if len(parts) != 2:
            print("❌ Invalid command format")
            return
        
        speed_cmd = parts[0].strip()
        direction_cmd = parts[1].strip()
        
        # Process speed command
        target_speed = parse_speed_command(speed_cmd)
        
        # Execute movement
        execute_movement(target_speed, direction_cmd)
        
        last_command = command
        last_command_time = time.ticks_ms()
        last_heartbeat = time.ticks_ms()
        
    except Exception as e:
        print(f"❌ Error processing command: {e}")
        stop_all_motors()

# -----------------------------------------------------------------------------
# --- CLIENT HANDLER ---
# -----------------------------------------------------------------------------

def handle_client():
    """Handle client connections in separate thread"""
    global client_socket, last_heartbeat
    
    while True:
        try:
            if server_socket:
                print("👂 Waiting for client connection...")
                client_socket, addr = server_socket.accept()
                print(f"📱 Client connected from {addr}")
                last_heartbeat = time.ticks_ms()
                
                while True:
                    try:
                        data = client_socket.recv(1024)
                        if not data:
                            break
                        
                        command = data.decode('utf-8').strip()
                        if command:
                            process_command(command)
                            
                    except OSError:
                        break
                    except Exception as e:
                        print(f"❌ Client error: {e}")
                        break
                
                print("📱 Client disconnected")
                client_socket.close()
                client_socket = None
                
        except Exception as e:
            print(f"❌ Server error: {e}")
            time.sleep(1)

# -----------------------------------------------------------------------------
# --- MONITORING FUNCTIONS ---
# -----------------------------------------------------------------------------

def check_command_timeout():
    """Check for command timeout and stop motors if needed"""
    global last_command_time
    
    if last_command_time > 0:
        if time.ticks_diff(time.ticks_ms(), last_command_time) > COMMAND_TIMEOUT:
            print("⚠️ Command timeout - stopping motors")
            stop_all_motors()
            last_command_time = 0

def update_status_led():
    """Update status LED based on connection state"""
    current_time = time.ticks_ms()
    
    if is_connected and client_socket:
        # Solid on when connected and receiving commands
        if time.ticks_diff(current_time, last_heartbeat) < 3000:
            status_led.value(1)
        else:
            # Slow blink when connected but no recent commands
            status_led.value((current_time // 1000) % 2)
    elif is_connected:
        # Fast blink when WiFi connected but no client
        status_led.value((current_time // 250) % 2)
    else:
        # Off when no WiFi
        status_led.value(0)

def print_status():
    """Print robot status for debugging"""
    print(f"📊 Status - Left: {left_motor_speed} ({left_encoder_count + left_encoder_count2}/2), "
          f"Right: {right_motor_speed} ({right_encoder_count + right_encoder_count2}/2)")
    print(f"📡 WiFi: {is_connected}, Client: {client_socket is not None}")
    print(f"💾 Free memory: {gc.mem_free()} bytes")

# -----------------------------------------------------------------------------
# --- MAIN PROGRAM ---
# -----------------------------------------------------------------------------

def main():
    """Main program"""
    global robot_status
    
    print("🤖 ESP32 Line Following Robot - MicroPython")
    print("=" * 50)
    
    # Initialize hardware
    robot_status = "Setting up pins"
    setup_pins()
    
    robot_status = "Setting up encoders"
    setup_encoders()
    
    robot_status = "Connecting WiFi"
    if not setup_wifi():
        print("❌ Failed to connect to WiFi. Check credentials.")
        return
    
    robot_status = "Starting server"
    setup_server()
    
    # Flash LED to indicate ready
    for i in range(5):
        status_led.value(1)
        time.sleep(0.1)
        status_led.value(0)
        time.sleep(0.1)
    
    robot_status = "Ready"
    print("✅ Robot Ready!")
    
    # Start client handler in separate thread
    _thread.start_new_thread(handle_client, ())
    
    # Main monitoring loop
    last_status_print = 0
    status_counter = 0
    
    while True:
        try:
            current_time = time.ticks_ms()
            
            # Check command timeout
            check_command_timeout()
            
            # Update status LED
            update_status_led()
            
            # Print status every 5 seconds
            if time.ticks_diff(current_time, last_status_print) > 5000:
                if status_counter % 2 == 0:  # Every 10 seconds
                    print_status()
                last_status_print = current_time
                status_counter += 1
            
            # Garbage collection
            if status_counter % 10 == 0:
                gc.collect()
            
            time.sleep(0.1)
            
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested")
            break
        except Exception as e:
            print(f"❌ Main loop error: {e}")
            time.sleep(1)
    
    # Cleanup
    print("🧹 Cleaning up...")
    stop_all_motors()
    if client_socket:
        client_socket.close()
    if server_socket:
        server_socket.close()
    status_led.value(0)
    print("✅ Cleanup complete")

# -----------------------------------------------------------------------------
# --- BOOT SEQUENCE ---
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        # Emergency stop
        try:
            stop_all_motors()
        except:
            pass

"""
SETUP INSTRUCTIONS:

1. Install MicroPython on ESP32:
   - Download firmware from micropython.org
   - Flash using esptool.py: esptool.py --chip esp32 write_flash -z 0x1000 firmware.bin

2. Upload this file to ESP32:
   - Use Thonny, uPyCraft, or ampy
   - Save as main.py on the ESP32

3. Update WiFi credentials:
   - Change WIFI_SSID and WIFI_PASSWORD

4. Wire hardware according to pin definitions in code

5. Update ESP32_IP in Raspberry Pi code to match ESP32's IP

WIRING GUIDE:

Motor Driver 1 (Left Side):
- ENA -> GPIO 25, IN1 -> GPIO 26, IN2 -> GPIO 27
- ENB -> GPIO 14, IN3 -> GPIO 12, IN4 -> GPIO 13

Motor Driver 2 (Right Side):  
- ENA -> GPIO 32, IN1 -> GPIO 33, IN2 -> GPIO 25
- ENB -> GPIO 26, IN3 -> GPIO 27, IN4 -> GPIO 14

Encoders:
- Left Front: A->GPIO18, B->GPIO19
- Left Rear: A->GPIO21, B->GPIO22  
- Right Front: A->GPIO16, B->GPIO17
- Right Rear: A->GPIO4, B->GPIO2

Power:
- Motor drivers: 12V battery + 5V logic
- ESP32: 5V VIN or 3.3V supply
- Common ground for all components

Status LED: GPIO2 (built-in LED on most ESP32 boards)
""" 