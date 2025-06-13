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

# Sensor Pins
SENSOR_PINS = [14, 27, 16, 17, 25]  # L2, L1, C, R1, R2

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

def run_server(motors, sensors):
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
                # Send sensor data every 50ms
                if time.ticks_diff(current_time, last_sensor_time) >= 50:
                    sensor_data = sensors.read()
                    data_str = f"{sensor_data[0]},{sensor_data[1]},{sensor_data[2]},{sensor_data[3]},{sensor_data[4]}\n"
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
                    data = client.recv(32)
                    if data:
                        command = data.decode().strip()
                        if ',' in command:
                            # Motor speed command: "left,right"
                            try:
                                left, right = command.split(',')
                                motors.set_speeds(int(left), int(right))
                            except:
                                print(f"Bad command: {command}")
                        elif command == "STOP":
                            motors.stop()
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
            motors.stop()
        
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
    
    try:
        run_server(motors, sensors)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        motors.stop()

if __name__ == "__main__":
    main() 