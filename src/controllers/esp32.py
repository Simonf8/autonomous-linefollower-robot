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
    
    def __init__(self, left_pin=5, center_pin=2, right_pin=4, threshold=65000):
        """
        Initialize tracker sensor with proven working configuration.
        """
        # Initialize ADC pins for analog reading using your working pins
        self.left_sensor = ADC(Pin(left_pin), atten=ADC.ATTN_11DB)
        self.center_sensor = ADC(Pin(center_pin), atten=ADC.ATTN_11DB)
        self.right_sensor = ADC(Pin(right_pin), atten=ADC.ATTN_11DB)
        
        # Working threshold from your testing
        self.threshold = threshold
        
        # For compatibility, we'll create 5 virtual sensors from 3 physical
        self.num_sensors = 5
        
        # Calibration data (using your threshold as baseline)
        self.min_values = [0] * self.num_sensors
        self.max_values = [65535] * self.num_sensors  # 16-bit ADC max
        self.calibrated = True  # Already calibrated with your threshold
        
        # Sensor weights for weighted average (0, 1000, 2000, 3000, 4000)
        self.weights = [i * 1000 for i in range(self.num_sensors)]
        
        print(f"Line sensor initialized - Pins: Left={left_pin}, Center={center_pin}, Right={right_pin}")
        print(f"Threshold: {threshold} - Line detected when below this value")
    
    def read_raw(self):
        """Read raw ADC values from all sensors using your working code"""
        # Read 3 physical sensors using your proven method
        left_raw = self.left_sensor.read_u16()
        center_raw = self.center_sensor.read_u16()
        right_raw = self.right_sensor.read_u16()
        
        # Create 5 virtual sensors from 3 physical sensors for compatibility
        # Left sensor, Left-Center interpolation, Center, Right-Center interpolation, Right
        sensor_values = [
            left_raw,                                    # Leftmost (sensor 0)
            (left_raw + center_raw) // 2,               # Left-Center (sensor 1) 
            center_raw,                                  # Center (sensor 2)
            (center_raw + right_raw) // 2,              # Right-Center (sensor 3)
            right_raw                                    # Rightmost (sensor 4)
        ]
        
        return sensor_values
    

    
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
        
        for i in range(self.num_sensors):
            raw = raw_values[i]
            
            # Use your proven threshold method
            # If below threshold = line detected (return low value)
            # If above threshold = no line (return high value)
            if raw < self.threshold:
                calibrated_values.append(0)    # Line detected
            else:
                calibrated_values.append(1000) # No line detected
        
        return calibrated_values
    
    def read_line_position(self):
        """
        Calculate line position using weighted average.
        Returns: 0-4000 (0=leftmost, 2000=center, 4000=rightmost)
        """
        sensor_values = self.read_calibrated()
        
        # Calculate weighted average
        numerator = 0
        denominator = 0
        
        for i in range(self.num_sensors):
            numerator += self.weights[i] * sensor_values[i]
            denominator += sensor_values[i]
        
        if denominator == 0:
            return 2000  # Default to center if no line detected
        
        position = numerator // denominator
        return max(0, min(4000, position))
    
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
        Returns: error from center position (-2000 to +2000)
        """
        position = self.read_line_position()
        return position - 2000  # Center is at 2000

class OmniwheelRobot:
    def __init__(self):
        # Motor pin setup
        self.motors = {
            'fl': {'p1': PWM(Pin(6), freq=500), 'p2': PWM(Pin(19), freq=500)},
            'fr': {'p1': PWM(Pin(7), freq=500), 'p2': PWM(Pin(42), freq=500)},
            'bl': {'p1': PWM(Pin(18), freq=500), 'p2': PWM(Pin(17), freq=500)},
            'br': {'p1': PWM(Pin(15), freq=500), 'p2': PWM(Pin(16), freq=500)}
        }
        
        # Encoder pin setup
        self.encoders = {
            'fl': {'a': Pin(3, Pin.IN, Pin.PULL_UP), 'b': Pin(46, Pin.IN, Pin.PULL_UP)},
            'fr': {'a': Pin(9, Pin.IN, Pin.PULL_UP), 'b': Pin(10, Pin.IN, Pin.PULL_UP)},
            'bl': {'a': Pin(11, Pin.IN, Pin.PULL_UP), 'b': Pin(12, Pin.IN, Pin.PULL_UP)},
            'br': {'a': Pin(13, Pin.IN, Pin.PULL_UP), 'b': Pin(14, Pin.IN, Pin.PULL_UP)}
        }
        
        # Initialize tracker sensor with your working configuration
        self.tracker = TrackerSensor(left_pin=5, center_pin=2, right_pin=4, threshold=65000)
        
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
        if not self.tracker.is_line_detected():
            return None, None, None
        
        position = self.tracker.read_line_position()
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
        return wlan.ifconfig()[0]
    
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
    """Runs a server to send encoder data and receive motor commands."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('', SERVER_PORT))
    server.listen(1)
    server.settimeout(0.2)
    
    client = None
    last_send_time = time.ticks_ms()
    
    while True:
        try:
            if client is None:
                try:
                    client, addr = server.accept()
                    client.settimeout(0.2)
                    print(f"Connected: {addr[0]}")
                except OSError:
                    continue

            # Send encoder and tracker data periodically
            if time.ticks_diff(time.ticks_ms(), last_send_time) >= 50: # 20Hz
                fl, fr, bl, br = robot.get_ticks()
                
                # Get tracker data
                position, error, sensor_values = robot.get_tracker_data()
                
                # Format: "encoder_fl,encoder_fr,encoder_bl,encoder_br,line_pos,line_error,s0,s1,s2,s3,s4"
                if position is not None:
                    data_str = f"{fl},{fr},{bl},{br},{position},{error},{sensor_values[0]},{sensor_values[1]},{sensor_values[2]},{sensor_values[3]},{sensor_values[4]}\n"
                else:
                    # No line detected
                    data_str = f"{fl},{fr},{bl},{br},-1,0,0,0,0,0,0\n"
                
                try:
                    client.sendall(data_str.encode())
                    last_send_time = time.ticks_ms()
                except OSError:
                    client.close()
                    client = None
                    robot.stop()
                    continue

            # Receive and process motor commands
            try:
                data = client.recv(128)
                if data:
                    command = data.decode().strip()
                    
                    # Handle different command types
                    if command == "CALIBRATE":
                        robot.calibrate_tracker()
                        client.sendall(b"CALIBRATION_COMPLETE\n")
                    elif command == "STOP":
                        robot.stop()
                    elif command.startswith("LINE_FOLLOW"):
                        # Extract base speed if provided
                        parts = command.split(',')
                        base_speed = int(parts[1]) if len(parts) > 1 else 60
                        robot.line_follow(base_speed)
                    elif ',' in command:
                        # Regular motor command: "fl,fr,bl,br"
                        parts = command.split(',')
                        if len(parts) == 4:
                            speeds = [int(float(p)) for p in parts]
                            robot.set_all_speeds(speeds[0], speeds[1], speeds[2], speeds[3])
                else:
                    # No data means client disconnected
                    client.close()
                    client = None
                    robot.stop()
            except OSError:
                pass # No data received, which is normal

        except Exception as e:
            if client:
                client.close()
                client = None
            robot.stop()
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
