import time
import network
import socket
from machine import Pin, ADC, PWM

# Network Configuration
WIFI_SSID = "CJ"  # Fill in your WiFi name
WIFI_PASSWORD = "4533simon"  # Fill in your WiFi password
SERVER_PORT = 1234

# Motor Configuration
FORWARD_SPEED = 50  # Reduced for stability
TURN_SPEED = 50     # Reduced for stability  
EMERGENCY_SPEED = 50
STOP_SPEED = 0

# Motor Pins
LEFT_MOTOR_1 = 18
LEFT_MOTOR_2 = 19
RIGHT_MOTOR_1 = 12
RIGHT_MOTOR_2 = 13

# Line Sensor Pins
LINE_SENSORS = [14, 27, 16, 17, 25]  # ADC capable pins

class LineSensor:
    def __init__(self, sensor_pins):
        self.sensors = []
        self.sensor_pins = sensor_pins
        
        # Initialize digital GPIO pins for line sensors
        for pin in sensor_pins:
            try:
                gpio_pin = Pin(pin, Pin.IN)
                self.sensors.append(gpio_pin)
                print(f"Digital sensor on pin {pin} initialized")
            except Exception as e:
                print(f"Failed to initialize GPIO on pin {pin}: {e}")
                self.sensors.append(None)
        
        self.last_position = 0
    
    def read_raw(self):
        # For digital sensors, raw and digital are the same
        return self.read_digital()
    
    def read_digital(self):
        values = []
        for sensor in self.sensors:
            if sensor is not None:
                try:
                    # Use raw values directly (0 = line detected, 1 = no line)
                    raw_value = sensor.value()
                    line_detected = 1 - raw_value  # Convert: 0 becomes 1 (line), 1 becomes 0 (no line)
                    values.append(line_detected)
                except:
                    values.append(0)  # Default value if read fails
            else:
                values.append(0)  # Default value for failed sensors
        
        # Apply reflection filtering for more stable readings
        return self._filter_reflections(values)
    
    def _filter_reflections(self, sensor_values):
        """Filter out reflection-caused false readings using temporal consistency"""
        # Initialize sensor history for reflection detection
        if not hasattr(self, 'sensor_history'):
            self.sensor_history = []
        
        self.sensor_history.append(sensor_values.copy())
        if len(self.sensor_history) > 5:  # Keep last 5 readings
            self.sensor_history.pop(0)
        
        # If we have enough history, apply filtering
        if len(self.sensor_history) >= 3:
            filtered_values = []
            for i in range(len(sensor_values)):
                # Get last 3 readings for this sensor
                recent_readings = [reading[i] for reading in self.sensor_history[-3:]]
                
                # Use majority vote to filter reflections
                if sum(recent_readings) >= 2:  # 2 out of 3 readings show line
                    filtered_values.append(1)
                else:
                    filtered_values.append(0)
                    
            return filtered_values
        else:
            # Not enough history yet, return current values
            return sensor_values
    
    def get_line_position(self):
        values = self.read_digital()
        
        # Special case: If middle 3 sensors are all 0 (black), robot is perfectly centered
        if len(values) == 5:
            if values[1] == 0 and values[2] == 0 and values[3] == 0:
                print(f"PERFECT CENTER: All 3 middle sensors on line: {values}")
                self.last_position = 0.0
                return 0.0, True
        
        # Check if any sensors detect the line
        total = sum(values)
        if total == 0:
            print(f"NO LINE: All sensors read 0: {values}")
            return self.last_position, False
        
        # Calculate weighted position for other cases
        weighted_sum = 0
        for i, value in enumerate(values):
            weighted_sum += value * (i - (len(values) - 1) / 2)
        
        position = weighted_sum / total
        position = position / ((len(values) - 1) / 2)
        
        print(f"LINE POS: sensors={values}, pos={position:.2f}")
        self.last_position = position
        return position, True
    
    def calibrate(self, samples=100):
        print("Digital line sensors don't need calibration")
        print("Sensors will read 1 for line detected, 0 for no line")
        
        # Test read to make sure sensors are working
        print("Testing sensor readings:")
        values = self.read_digital()
        for i, value in enumerate(values):
            print(f"  Sensor {i} (pin {self.sensor_pins[i]}): {value}")
        print("Calibration complete")

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
        print(f"MOTOR: Forward at speed {speed}")
        self._set_left_motor(speed, True)
        self._set_right_motor(speed, True)
    
    def backward(self, speed=FORWARD_SPEED):
        print("Backward")
        self._set_left_motor(speed, False)
        self._set_right_motor(speed, False)
    
    def left(self, speed=35):  # Slower sharp turns
        print(f"MOTOR: Left turn at speed {speed}")
        self._set_left_motor(speed, False)
        self._set_right_motor(speed, True)
    
    def right(self, speed=35):  # Slower sharp turns
        print(f"MOTOR: Right turn at speed {speed}")
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
    
    def slight_left(self, speed=40):  # More responsive differential speed
        left_speed = int(speed * 0.7)   # 70% speed on inside wheel (more effective)
        right_speed = speed             # 100% speed on outside wheel
        print(f"MOTOR: Slight left - L:{left_speed} R:{right_speed}")
        self._set_left_motor(left_speed, True)
        self._set_right_motor(right_speed, True)
    
    def slight_right(self, speed=40):  # More responsive differential speed
        left_speed = speed              # 100% speed on outside wheel  
        right_speed = int(speed * 0.7)  # 70% speed on inside wheel (more effective)
        print(f"MOTOR: Slight right - L:{left_speed} R:{right_speed}")
        self._set_left_motor(left_speed, True)
        self._set_right_motor(right_speed, True)
    
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
        elif command == 'SLIGHT_LEFT':
            motors.slight_left()
        elif command == 'SLIGHT_RIGHT':
            motors.slight_right()
        elif command == 'BACKWARD':
            motors.backward()
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