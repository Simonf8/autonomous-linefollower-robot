import time
import network
import socket
from machine import Pin, ADC, PWM

# Network Configuration
WIFI_SSID = ""  # Fill in your WiFi name
WIFI_PASSWORD = ""  # Fill in your WiFi password
SERVER_PORT = 1234

# Motor Configuration
FORWARD_SPEED = 55
TURN_SPEED = 50
EMERGENCY_SPEED = 85
STOP_SPEED = 0

# Motor Pins
LEFT_MOTOR_1 = 18
LEFT_MOTOR_2 = 19
RIGHT_MOTOR_1 = 12
RIGHT_MOTOR_2 = 13

# Line Sensor Pins
LINE_SENSORS = [32, 33, 34, 35, 36]  # ADC capable pins

class LineSensor:
    def __init__(self, sensor_pins):
        self.sensors = [ADC(Pin(pin)) for pin in sensor_pins]
        for sensor in self.sensors:
            sensor.atten(ADC.ATTN_11DB)
            sensor.width(ADC.WIDTH_12BIT)
        
        self.threshold = 2000
        self.last_position = 0
    
    def read_raw(self):
        return [sensor.read() for sensor in self.sensors]
    
    def read_digital(self):
        raw_values = self.read_raw()
        return [1 if value > self.threshold else 0 for value in raw_values]
    
    def get_line_position(self):
        values = self.read_digital()
        weighted_sum = 0
        total = sum(values)
        
        if total == 0:
            return self.last_position, False
        
        for i, value in enumerate(values):
            weighted_sum += value * (i - (len(values) - 1) / 2)
        
        position = weighted_sum / total
        position = position / ((len(values) - 1) / 2)
        
        self.last_position = position
        return position, True
    
    def calibrate(self, samples=100):
        print("Starting calibration...")
        min_values = [4095] * len(self.sensors)
        max_values = [0] * len(self.sensors)
        
        for _ in range(samples):
            current_values = self.read_raw()
            for i, value in enumerate(current_values):
                min_values[i] = min(min_values[i], value)
                max_values[i] = max(max_values[i], value)
            time.sleep_ms(10)
        
        self.threshold = sum([(max_val + min_val) / 2 
                            for min_val, max_val 
                            in zip(min_values, max_values)]) / len(self.sensors)
        print(f"Calibration complete. Threshold: {self.threshold}")

class Motors:
    def __init__(self):
        self.left_1 = PWM(Pin(LEFT_MOTOR_1), freq=100)
        self.left_2 = PWM(Pin(LEFT_MOTOR_2), freq=100)
        self.right_1 = PWM(Pin(RIGHT_MOTOR_1), freq=100)
        self.right_2 = PWM(Pin(RIGHT_MOTOR_2), freq=100)
        self.stop()
        print("Motors initialized")
    
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
    
    def forward(self, speed=FORWARD_SPEED):
        print("Forward")
        self._set_left_motor(speed, True)
        self._set_right_motor(speed, True)
    
    def backward(self, speed=FORWARD_SPEED):
        print("Backward")
        self._set_left_motor(speed, False)
        self._set_right_motor(speed, False)
    
    def left(self, speed=TURN_SPEED):
        print("Left turn")
        self._set_left_motor(speed, False)
        self._set_right_motor(speed, True)
    
    def right(self, speed=TURN_SPEED):
        print("Right turn")
        self._set_left_motor(speed, True)
        self._set_right_motor(speed, False)
    
    def turn_around(self, speed=TURN_SPEED):
        print("Turning around")
        # First back up a bit
        self.backward(speed)
        time.sleep(0.5)
        # Then do a 180-degree turn
        self._set_left_motor(speed, True)
        self._set_right_motor(speed, False)
        time.sleep(2.0)  # Adjust this time based on your robot's turning speed
        self.stop()
    
    def emergency_left(self, speed=EMERGENCY_SPEED):
        print("EMERGENCY LEFT")
        self._set_left_motor(speed, False)
        self._set_right_motor(speed, True)
    
    def emergency_right(self, speed=EMERGENCY_SPEED):
        print("EMERGENCY RIGHT")
        self._set_left_motor(speed, True)
        self._set_right_motor(speed, False)
    
    def stop(self):
        print("Stop")
        self.left_1.duty(0)
        self.left_2.duty(0)
        self.right_1.duty(0)
        self.right_2.duty(0)

def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    
    if wlan.isconnected():
        ip = wlan.ifconfig()[0]
        print(f"Already connected: {ip}")
        return ip
    
    print(f"Connecting to {WIFI_SSID}...")
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    
    timeout = 0
    while not wlan.isconnected() and timeout < 30:
        if timeout % 5 == 0:
            print(f"Connecting... {timeout}s")
        time.sleep(1)
        timeout += 1
    
    if wlan.isconnected():
        ip = wlan.ifconfig()[0]
        print(f"WiFi connected: {ip}")
        return ip
    else:
        print(f"WiFi connection failed after {timeout}s")
        return None

def process_command(command, motors):
    try:
        command = command.strip().upper()
        print(f"Received command: '{command}'")
        
        if command == 'FORWARD':
            motors.forward()
        elif command == 'LEFT':
            motors.left()
        elif command == 'RIGHT':
            motors.right()
        elif command == 'TURN_AROUND':
            motors.turn_around()
        elif command == 'EMERGENCY_LEFT':
            motors.emergency_left()
        elif command == 'EMERGENCY_RIGHT':
            motors.emergency_right()
        elif command == 'STOP':
            motors.stop()
        else:
            print(f"Unknown command: '{command}'")
            return False
        return True
        
    except Exception as e:
        print(f"Command processing error: {e}")
        motors.stop()
        return False

def run_server(motors, line_sensor):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server.bind(('', SERVER_PORT))
        server.listen(1)
        server.settimeout(1.0)
        print(f"Server listening on port {SERVER_PORT}")
        
    except Exception as e:
        print(f"Server bind error: {e}")
        return
    
    client = None
    last_command_time = time.time()
    COMMAND_TIMEOUT = 2.0
    
    while True:
        try:
            if client is None:
                try:
                    client, addr = server.accept()
                    client.settimeout(0.1)
                    print(f"Client connected from: {addr}")
                    last_command_time = time.time()
                except OSError:
                    pass
            
            if client:
                try:
                    data = client.recv(64)
                    if data:
                        command = data.decode('utf-8').strip()
                        if command:
                            success = process_command(command, motors)
                            if success:
                                last_command_time = time.time()
                                
                                # Send line sensor data back to Pi
                                position, detected = line_sensor.get_line_position()
                                sensor_data = f"{position},{1 if detected else 0}\n"
                                client.send(sensor_data.encode('utf-8'))
                    else:
                        print("Client disconnected")
                        client.close()
                        client = None
                        motors.stop()
                        
                except OSError:
                    current_time = time.time()
                    if current_time - last_command_time > COMMAND_TIMEOUT:
                        print("Command timeout - stopping")
                        motors.stop()
                        last_command_time = current_time
                    
                except Exception as e:
                    print(f"Client communication error: {e}")
                    if client:
                        client.close()
                        client = None
                    motors.stop()
        
        except KeyboardInterrupt:
            print("Keyboard interrupt - stopping server")
            break
            
        except Exception as e:
            print(f"Server error: {e}")
            time.sleep(1)
    
    print("Cleaning up...")
    motors.stop()
    if client:
        client.close()
    server.close()

def main():
    print("ESP32 Line Follower Robot")
    print("=" * 30)
    
    # Connect WiFi
    ip = connect_wifi()
    if not ip:
        print("Cannot continue without WiFi connection")
        return
    
    # Initialize components
    motors = Motors()
    line_sensor = LineSensor(LINE_SENSORS)
    
    # Calibrate line sensors
    print("Calibrating line sensors...")
    line_sensor.calibrate()
    
    # Test motors briefly
    print("Testing motors...")
    motors.forward(30)
    time.sleep(0.5)
    motors.stop()
    time.sleep(0.5)
    
    print(f"Robot ready!")
    print(f"Connect to: {ip}:{SERVER_PORT}")
    print("Press Ctrl+C to stop")
    print("-" * 30)
    
    try:
        run_server(motors, line_sensor)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        motors.stop()
        print("ESP32 stopped safely")

if __name__ == "__main__":
    main() 