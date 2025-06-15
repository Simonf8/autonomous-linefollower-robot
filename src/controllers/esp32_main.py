import time
import network
import socket
from machine import Pin, PWM

# WiFi Configuration
WIFI_SSID = ""
WIFI_PASSWORD = ""
SERVER_PORT = 1234

# Motor Pins
LEFT_MOTOR_1 = 18
LEFT_MOTOR_2 = 19
RIGHT_MOTOR_1 = 12
RIGHT_MOTOR_2 = 13

# Encoder Pins (QUADRATURE - 2 pins per encoder)
# Please verify these pins with your hardware setup
LEFT_ENCODER_A = 23
LEFT_ENCODER_B = 5
RIGHT_ENCODER_A = 26
RIGHT_ENCODER_B = 25

# Sensor Pins
SENSOR_PINS = [14, 27, 16, 17, 33]  # L2, L1, C, R1, R2

# Global tick counters for encoders
left_ticks = 0
right_ticks = 0

# Global Pin objects for reading in ISRs, required for speed.
left_b_pin = None
right_b_pin = None

# Interrupt handlers for encoders
def left_encoder_handler(pin):
    """Handle left encoder interrupt for quadrature decoding."""
    global left_ticks
    # Read pin B to determine direction.
    # The logic might need to be inverted (+= 1 vs -= 1) depending on wiring.
    if left_b_pin.value() == 0:
        left_ticks -= 1 # Backward
    else:
        left_ticks += 1 # Forward

def right_encoder_handler(pin):
    """Handle right encoder interrupt for quadrature decoding."""
    global right_ticks
    # Read pin B to determine direction.
    if right_b_pin.value() == 0:
        right_ticks -= 1 # Backward
    else:
        right_ticks += 1 # Forward

class Encoders:
    """Manages reading from quadrature wheel encoders using interrupts."""
    
    def __init__(self, left_pin_a_num, left_pin_b_num, right_pin_a_num, right_pin_b_num):
        """Initialize encoder pins and set up interrupt handlers."""
        global left_b_pin, right_b_pin

        # Initialize pins
        left_a_pin = Pin(left_pin_a_num, Pin.IN, Pin.PULL_UP)
        left_b_pin = Pin(left_pin_b_num, Pin.IN, Pin.PULL_UP)
        right_a_pin = Pin(right_pin_a_num, Pin.IN, Pin.PULL_UP)
        right_b_pin = Pin(right_pin_b_num, Pin.IN, Pin.PULL_UP)
        
        # Attach interrupts to detect rising edges on A pins
        left_a_pin.irq(trigger=Pin.IRQ_RISING, handler=left_encoder_handler)
        right_a_pin.irq(trigger=Pin.IRQ_RISING, handler=right_encoder_handler)
        
        print("Quadrature Encoders ready")
        
    def get_ticks(self):
        """Get the current tick counts for both encoders."""
        global left_ticks, right_ticks
        return left_ticks, right_ticks

class Motors:
    """Simple motor control"""
    
    def __init__(self):
        self.left_1 = PWM(Pin(LEFT_MOTOR_1), freq=50)
        self.left_2 = PWM(Pin(LEFT_MOTOR_2), freq=50)
        self.right_1 = PWM(Pin(RIGHT_MOTOR_1), freq=50)
        self.right_2 = PWM(Pin(RIGHT_MOTOR_2), freq=50)
        self.stop()
        print("Motors ready")
    
    def set_speeds(self, left_speed, right_speed):
        """Set motor speeds (-100 to 100)"""
        # Clamp speeds
        left_speed = max(-100, min(100, left_speed))
        right_speed = max(-100, min(100, right_speed))
        
        # Set left motor
        if left_speed > 0:
            duty = int(left_speed * 1023 / 100)
            self.left_1.duty(duty)
            self.left_2.duty(0)
        elif left_speed < 0:
            duty = int(abs(left_speed) * 1023 / 100)
            self.left_1.duty(0)
            self.left_2.duty(duty)
        else:
            self.left_1.duty(0)
            self.left_2.duty(0)
            
        # Set right motor
        if right_speed > 0:
            duty = int(right_speed * 1023 / 100)
            self.right_1.duty(duty)
            self.right_2.duty(0)
        elif right_speed < 0:
            duty = int(abs(right_speed) * 1023 / 100)
            self.right_1.duty(0)
            self.right_2.duty(duty)
        else:
            self.right_1.duty(0)
            self.right_2.duty(0)
    
    def stop(self):
        """Stop all motors"""
        self.left_1.duty(0)
        self.left_2.duty(0)
        self.right_1.duty(0)
        self.right_2.duty(0)

class Sensors:
    """Simple sensor reading"""
    
    def __init__(self, pins):
        self.sensors = []
        for pin in pins:
            self.sensors.append(Pin(pin, Pin.IN))
        print("Sensors ready")
    
    def read(self):
        """Read all sensors and return values"""
        values = []
        for sensor in self.sensors:
            # Convert: 0 = line detected, 1 = no line
            raw = sensor.value()
            line_detected = 1 - raw
            values.append(line_detected)
        return values

def connect_wifi():
    """Connect to WiFi"""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    
    if wlan.isconnected():
        return wlan.ifconfig()[0]
    
    print(f"Connecting to WiFi...")
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    
    timeout = 0
    while not wlan.isconnected() and timeout < 20:
        time.sleep(1)
        timeout += 1
    
    if wlan.isconnected():
        ip = wlan.ifconfig()[0]
        print(f"WiFi connected: {ip}")
        return ip
    else:
        print("WiFi failed")
        return None

def run_server(motors, sensors, encoders):
    """Simple server - send sensor data, receive motor commands"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('', SERVER_PORT))
    server.listen(1)
    server.settimeout(0.1)
        
    print(f"Server ready on port {SERVER_PORT}")
    
    client = None
    last_sensor_time = time.ticks_ms()
    
    while True:
        current_time = time.ticks_ms()
        
        try:
            # Accept connections
            if client is None:
                try:
                    client, addr = server.accept()
                    client.settimeout(0.1)
                    print(f"Pi connected: {addr[0]}")
                except OSError:
                    pass
            
            if client:
                # Send sensor and encoder data every 50ms
                if time.ticks_diff(current_time, last_sensor_time) >= 50:
                    sensor_values = sensors.read()
                    lt, rt = encoders.get_ticks()
                    
                    # Data format: "s1,s2,s3,s4,s5,left_ticks,right_ticks\n"
                    data_str = f"{','.join(map(str, sensor_values))},{lt},{rt}\n"
                    
                    try:
                        client.send(data_str.encode())
                        last_sensor_time = current_time
                    except:
                        print("Send failed")
                        client.close()
                        client = None
                        motors.stop()
                
                # Receive motor commands
                try:
                    data = client.recv(64)  # Increased buffer size
                    if data:
                        # Handle multiple commands in buffer
                        commands = data.decode().strip().split('\n')
                        
                        for command in commands:
                            command = command.strip()
                            if not command:  # Skip empty commands
                                continue
                                
                            if ',' in command:
                                # Motor speed command: "left,right"
                                try:
                                    parts = command.split(',')
                                    if len(parts) == 2:
                                        left = int(float(parts[0]))  # Handle float strings
                                        right = int(float(parts[1]))
                                        
                                        # Validate reasonable speed range
                                        if -100 <= left <= 100 and -100 <= right <= 100:
                                            motors.set_speeds(left, right)
                                        else:
                                            print(f"Speed out of range: {left},{right}")
                                    else:
                                        print(f"Invalid format: {command}")
                                except ValueError as e:
                                    print(f"Parse error: {command} - {e}")
                                except Exception as e:
                                    print(f"Command error: {command} - {e}")
                            elif command == "STOP":
                                motors.stop()
                            else:
                                print(f"Unknown command: {command}")
                    else:
                        print("Pi disconnected")
                        client.close()
                        client = None
                        motors.stop()
                
                except OSError:
                    pass  # No data available
                except:
                    print("Receive failed")
                    client.close()
                    client = None
                    motors.stop()
            
        except Exception as e:
            print(f"Server error: {e}")
        
        # Only stop motors when there's an actual error, not every loop
        time.sleep(0.01)

def main():
    print("ESP32 Line Sensor Interface")
    
    # Connect WiFi
    if not connect_wifi():
        print("No WiFi - stopping")
        return
    
    # Initialize hardware
    motors = Motors()
    sensors = Sensors(SENSOR_PINS)
    encoders = Encoders(LEFT_ENCODER_A, LEFT_ENCODER_B, RIGHT_ENCODER_A, RIGHT_ENCODER_B)
    
    try:
        run_server(motors, sensors, encoders)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        motors.stop()

if __name__ == "__main__":
    main() 