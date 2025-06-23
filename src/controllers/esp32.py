import time
import network
import socket
import math
from machine import Pin, PWM, ADC

# WiFi Configuration
WIFI_SSID = "CJ"
WIFI_PASSWORD = "4533simon"
SERVER_PORT = 1234

class TrackerSensor:
    """3-channel analog infrared line tracker sensor using working pins"""
    
    def __init__(self, left_pin=11, center_pin=10, right_pin=3, threshold=60000):
        """
        Initialize tracker sensor with proven working configuration.
        """
        # Initialize ADC pins for analog reading using your working pins
        self.left_sensor = ADC(Pin(left_pin), atten=ADC.ATTN_11DB)
        self.center_sensor = ADC(Pin(center_pin), atten=ADC.ATTN_11DB)
        self.right_sensor = ADC(Pin(right_pin), atten=ADC.ATTN_11DB)
        
        # Working threshold from your testing
        self.threshold = threshold
        
        # Using 3 physical sensors
        self.num_sensors = 3
        
        # Calibration data
        self.min_values = [0] * self.num_sensors
        self.max_values = [65535] * self.num_sensors  # 16-bit ADC max
        self.calibrated = True  # Assuming pre-calibrated with threshold
        
        # Sensor weights for weighted average (0, 1000, 2000 for 3 sensors)
        self.weights = [i * 1000 for i in range(self.num_sensors)]
        
        print(f"Line sensor initialized - Pins: Left={left_pin}, Center={center_pin}, Right={right_pin}")
        print(f"Threshold: {threshold} - Line detected when below this value")
    
    def read_raw(self):
        """Read raw ADC values from the 3 physical sensors."""
        left_raw = self.left_sensor.read_u16()
        center_raw = self.center_sensor.read_u16()
        right_raw = self.right_sensor.read_u16()
        
        return [left_raw, center_raw, right_raw]
    

    
    def calibrate(self, samples=100):
        """
        Calibrate sensor by finding min/max values.
        Robot should be moving over the line during calibration.
        """
        # Initialize min/max arrays
        self.min_values = [65535] * self.num_sensors
        self.max_values = [0] * self.num_sensors
        
        for i in range(samples):
            sensor_values = self.read_raw()  # Use filtered readings for calibration
            
            for j in range(self.num_sensors):
                if sensor_values[j] < self.min_values[j]:
                    self.min_values[j] = sensor_values[j]
                if sensor_values[j] > self.max_values[j]:
                    self.max_values[j] = sensor_values[j]
            
            time.sleep_ms(20)  # 50Hz sampling rate
        
        self.calibrated = True
    
    def read_calibrated(self):
        """
        Read normalized sensor values using your working threshold method.
        1000 = far from line (white surface)
        0 = on line (black surface)
        """
        raw_values = self.read_raw()
        calibrated_values = []
        
        for raw in raw_values:
            # Use your proven threshold method
            if raw < self.threshold:
                calibrated_values.append(0)    # Line detected
            else:
                calibrated_values.append(1000) # No line detected
        
        return calibrated_values
    
    def read_line_position(self):
        """
        Calculate line position using weighted average.
        Returns: 0-2000 (0=left, 1000=center, 2000=right)
        """
        sensor_values = self.read_calibrated()
        
        # Invert readings for weighted average (0=line, 1000=no line)
        # We want to give more weight to sensors that see the line (lower values)
        # So we use (1000 - value)
        avg_num = 0
        avg_den = 0
        line_detected = False

        for i in range(self.num_sensors):
            value = sensor_values[i]
            if value < 1000: # If a line is detected by this sensor
                line_detected = True
            
            # Weighted average calculation
            avg_num += self.weights[i] * (1000 - value)
            avg_den += (1000 - value)
            
        if not line_detected:
            return -1 # No line detected

        if avg_den == 0:
            return 1000  # Should not happen if line is detected, but as a fallback
        
        position = avg_num / avg_den
        return max(0, min(2000, int(position)))
    
    def is_line_detected(self):
        """Check if any sensor detects the line using your working method"""
        # Read 3 physical sensors directly using your proven approach
        left = 1 if self.left_sensor.read_u16() < self.threshold else 0
        center = 1 if self.center_sensor.read_u16() < self.threshold else 0
        right = 1 if self.right_sensor.read_u16() < self.threshold else 0
        
        # Line detected if any sensor detects line
        return (left == 1) or (center == 1) or (right == 1)
    
    def print_sensor_debug(self):
        """Print sensor readings like your original code"""
        left = 1 if self.left_sensor.read_u16() < self.threshold else 0
        center = 1 if self.center_sensor.read_u16() < self.threshold else 0
        right = 1 if self.right_sensor.read_u16() < self.threshold else 0
        
        print(f"[{left} {center} {right}]")
    

    
    def get_line_error(self):
        """
        Get line position error for PID control.
        Returns: error from center position (-1000 to +1000)
        """
        position = self.read_line_position()
        if position == -1:
            return 0 # No line, no error
        return position - 1000  # Center is at 1000

class OmniwheelRobot:
    def __init__(self):
        # Motor pin setup
        self.motors = {
            'fl': {'p1': PWM(Pin(5), freq=50), 'p2': PWM(Pin(4), freq=50)},
            'fr': {'p1': PWM(Pin(6), freq=50), 'p2': PWM(Pin(7), freq=50)},
            'bl': {'p1': PWM(Pin(16), freq=50), 'p2': PWM(Pin(15), freq=50)},
            'br': {'p1': PWM(Pin(17), freq=50), 'p2': PWM(Pin(18), freq=50)}
        }
        
        # Encoder pin setup
        self.encoders = {
            'fl': {'a': Pin(38, Pin.IN, Pin.PULL_UP), 'b': Pin(39, Pin.IN, Pin.PULL_UP)},
            'fr': {'a': Pin(2, Pin.IN, Pin.PULL_UP), 'b': Pin(42, Pin.IN, Pin.PULL_UP)},
            'bl': {'a': Pin(41, Pin.IN, Pin.PULL_UP), 'b': Pin(40, Pin.IN, Pin.PULL_UP)},
            'br': {'a': Pin(0, Pin.IN, Pin.PULL_UP), 'b': Pin(45, Pin.IN, Pin.PULL_UP)}
        }
        
        # Initialize tracker sensor with your working configuration
        self.tracker = TrackerSensor(left_pin=10, center_pin=11, right_pin=3, threshold=60000)
        
        # Global-like storage for ticks and encoder B-pins for ISRs
        self.ticks = {'fl': 0, 'fr': 0, 'bl': 0, 'br': 0}
        

        
        self.setup_encoders()
        self.stop()

    def setup_encoders(self):
        """Setup encoder interrupt handlers"""
        for wheel, pins in self.encoders.items():
            pins['a'].irq(trigger=Pin.IRQ_RISING, handler=self.create_isr(wheel, pins))

    def create_isr(self, wheel, pins):
        """Creates a unique ISR for each wheel to avoid scope issues."""
        def isr(pin):
            if pins['b'].value():
                self.ticks[wheel] += 1  # Forward
            else:
                self.ticks[wheel] -= 1  # Reverse
        return isr
    
    def get_ticks(self):
        """Get the current raw tick counts for all encoders."""
        return self.ticks['fl'], self.ticks['fr'], self.ticks['bl'], self.ticks['br']
    
    def get_tracker_data(self):
        """Get tracker sensor data"""
        position = self.tracker.read_line_position()
        
        if position == -1:
            return -1, 0, [1000, 1000, 1000] # No line detected
            
        error = self.tracker.get_line_error()
        sensor_values = self.tracker.read_calibrated()
        
        return position, error, sensor_values
    

    
    def get_sensor_status(self):
        """
        Get current sensor status for line detection.
        Returns True if line is detected, False otherwise.
        """
        return self.tracker.is_line_detected()
    
    def calibrate_tracker(self):
        """Calibrate the tracker sensor"""
        self.tracker.calibrate(100)
    
    def set_motor_speed(self, wheel, speed):
        """Set speed for a single motor. Speed is from -100 to 100."""
        speed = max(-100, min(100, speed))
        duty = int(abs(speed) * 10.23) # Scale to 0-1023

        motor = self.motors[wheel]
        if speed > 0:
            motor['p1'].duty(duty)
            motor['p2'].duty(0)
        elif speed < 0:
            motor['p1'].duty(0)
            motor['p2'].duty(duty)
        else:
            motor['p1'].duty(0)
            motor['p2'].duty(0)

    def set_all_speeds(self, fl, fr, bl, br):
        """Set speeds for all four motors."""
        self.set_motor_speed('fl', fl)
        self.set_motor_speed('fr', fr)
        self.set_motor_speed('bl', bl)
        self.set_motor_speed('br', br)

    def line_follow(self, base_speed=40):
        """Follows the line using a simple proportional controller."""
        error = self.tracker.get_line_error()
        
        # This Kp value is a starting point and may need tuning.
        Kp = 0.05
        turn = Kp * error
        
        # For an omni-wheel robot, turning involves all four wheels.
        # This logic should make it turn in place based on the line error.
        fl_speed = base_speed + turn
        fr_speed = base_speed - turn
        bl_speed = base_speed + turn
        br_speed = base_speed - turn
        
        self.set_all_speeds(fl_speed, fr_speed, bl_speed, br_speed)

    def stop(self):
        """Stop all motors."""
        self.set_all_speeds(0, 0, 0, 0)

def connect_wifi():
    """Connect to WiFi"""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if wlan.isconnected():
        ip = wlan.ifconfig()[0]
        print(f"WiFi already connected. IP: {ip}")
        return ip
    
    print(f"Connecting to WiFi SSID: {WIFI_SSID}...")
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    
    timeout = 15
    while not wlan.isconnected() and timeout > 0:
        print(".")
        time.sleep(1)
        timeout -= 1
    
    if wlan.isconnected():
        ip = wlan.ifconfig()[0]
        print(f"WiFi connected. IP: {ip}")
        return ip
    else:
        return None

def run_server(robot):
    # Setup server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('', SERVER_PORT))
    server_socket.listen(1)
    
    print(f"Server listening on port {SERVER_PORT}")
    
    while True:
        conn = None
        try:
            conn, addr = server_socket.accept()
            conn.settimeout(1.0)
            print(f"Connection from {addr}")

            while True:
                try:
                    # Receive commands from client
                    data = conn.recv(1024)
                    if not data:
                        break # Connection closed by client
                    
                    command = data.decode().strip()
                    
                    # Basic command parsing
                    if command == "CALIBRATE":
                        robot.calibrate_tracker()
                    elif command.startswith("LINE_FOLLOW"):
                        parts = command.split(',')
                        speed = int(parts[1]) if len(parts) > 1 else 40
                        robot.line_follow(speed)
                    else:
                        # Motor speed command by default
                        parts = command.split(',')
                        if len(parts) == 4:
                            speeds = [int(p) for p in parts]
                            robot.set_all_speeds(*speeds)

                    # Send back sensor data
                    ticks = robot.get_ticks()
                    line_pos, line_err, sensor_vals = robot.get_tracker_data()
                    
                    # Format: "ENCODERS,fl,fr,bl,br"
                    encoder_data = f"ENCODERS,{ticks[0]},{ticks[1]},{ticks[2]},{ticks[3]}\n"
                    conn.send(encoder_data.encode())
                    
                    # Format: "LINE,pos,err,s1,s2,s3"
                    line_data = f"LINE,{line_pos},{line_err},{','.join(map(str, sensor_vals))}\n"
                    conn.send(line_data.encode())

                except socket.timeout:
                    # No command received, continue loop
                    pass
                except Exception as e:
                    print(f"Error during communication: {e}")
                    break
        
        except Exception as e:
            print(f"Server error: {e}")
            
        finally:
            if conn:
                conn.close()
                print("Connection closed")
            # Brief pause before accepting new connection
            time.sleep(1)

def main():
    if not connect_wifi():
        return
    
    robot = OmniwheelRobot()
    
    try:
        run_server(robot)
    except KeyboardInterrupt:
        pass
    finally:
        robot.stop()

if __name__ == "__main__":
    main() 
