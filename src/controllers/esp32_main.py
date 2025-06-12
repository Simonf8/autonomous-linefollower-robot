import time
import network
import socket
from machine import Pin, ADC, PWM

# Network Configuration
WIFI_SSID = "CJ"  # Fill in your WiFi name
WIFI_PASSWORD = "4533simon"  # Fill in your WiFi password
SERVER_PORT = 1234

# Motor Configuration - Faster settings
FORWARD_SPEED = 60  # Faster forward movement
TURN_SPEED = 70     # Much faster turning for quicker search
EMERGENCY_SPEED = 80
STOP_SPEED = 0

# Motor Pins
LEFT_MOTOR_1 = 18
LEFT_MOTOR_2 = 19
RIGHT_MOTOR_1 = 12
RIGHT_MOTOR_2 = 13

# Line Sensor Pins - Waveshare ITR20001/T
LINE_SENSORS = [14, 27, 16, 17, 26]  # Correct pins: Left2, Left1, Center, Right1, Right2

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
        
        # Check if any sensors detect the line (value = 1 means line detected)
        total = sum(values)
        if total == 0:
            print(f"NO LINE: All sensors read 0 (white surface): {values}")
            return self.last_position, 1  # Return 1 for "no line detected"
        
        # Calculate weighted position with correct indexing
        # Sensors: [0, 1, 2, 3, 4] where 2 is center
        # Position: [-2, -1, 0, 1, 2] relative to center
        weighted_sum = 0
        for i, value in enumerate(values):
            sensor_position = i - 2  # Convert index to position relative to center
            weighted_sum += value * sensor_position
        
        position = weighted_sum / total
        
        # Position is already in correct range (-2 to +2)
        # Normalize to -1.0 to +1.0 range
        position = position / 2.0
        
        print(f"LINE POS: sensors={values}, weighted_sum={weighted_sum}, total={total}, pos={position:.2f}")
        self.last_position = position
        return position, 0  # Return 0 for "line detected"
    
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
    
    def left(self, speed=TURN_SPEED):  # Use faster turn speed
        print(f"MOTOR: Left turn at speed {speed} - LEFT BACKWARD, RIGHT FORWARD")
        self._set_left_motor(speed, False)
        self._set_right_motor(speed, True)
    
    def right(self, speed=TURN_SPEED):  # Use faster turn speed
        print(f"MOTOR: Right turn at speed {speed} - LEFT FORWARD, RIGHT BACKWARD")
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
    
    def slight_left(self, speed=FORWARD_SPEED):  # Use forward speed for slight turns
        left_speed = int(speed * 0.7)   # 70% speed on inside wheel
        right_speed = speed             # 100% speed on outside wheel
        print(f"MOTOR: Slight left - L:{left_speed} R:{right_speed}")
        self._set_left_motor(left_speed, True)
        self._set_right_motor(right_speed, True)
    
    def slight_right(self, speed=FORWARD_SPEED):  # Use forward speed for slight turns
        left_speed = speed              # 100% speed on outside wheel  
        right_speed = int(speed * 0.7)  # 70% speed on inside wheel
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
        print(f"ESP32 RECEIVED COMMAND: '{command}'")
        
        if command == 'FORWARD':
            print("EXECUTING: Forward")
            motors.forward()
        elif command == 'LEFT':
            print("EXECUTING: Left turn")
            motors.left()
        elif command == 'RIGHT':
            print("EXECUTING: Right turn")
            motors.right()
        elif command == 'SLIGHT_LEFT':
            print("EXECUTING: Slight left")
            motors.slight_left()
        elif command == 'SLIGHT_RIGHT':
            print("EXECUTING: Slight right")
            motors.slight_right()
        elif command == 'BACKWARD':
            print("EXECUTING: Backward")
            motors.backward()
        elif command == 'TURN_AROUND':
            print("EXECUTING: Turn around")
            motors.turn_around()
        elif command == 'EMERGENCY_LEFT':
            print("EXECUTING: Emergency left")
            motors.emergency_left()
        elif command == 'EMERGENCY_RIGHT':
            print("EXECUTING: Emergency right")
            motors.emergency_right()
        elif command == 'STOP':
            print("EXECUTING: Stop")
            motors.stop()
        else:
            print(f"UNKNOWN COMMAND: '{command}'")
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