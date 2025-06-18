import time
import network
import socket
import math
from machine import Pin, PWM, ADC

# WiFi Configuration
WIFI_SSID = "YOUR_SSID"
WIFI_PASSWORD = "YOUR_PASSWORD"
SERVER_PORT = 1234

class TrackerSensor:
    """5-channel analog infrared line tracker sensor with median filtering"""
    
    def __init__(self, sensor_pins=[19, 20, 21], median_window_size=5):
        """
        Initialize tracker sensor with median filtering.
        """
        # Initialize ADC pins for analog reading
        self.adcs = []
        for pin in sensor_pins:
            adc = ADC(Pin(pin))
            adc.atten(ADC.ATTN_11DB)  # 0-3.3V range
            self.adcs.append(adc)
        
        # For 5 sensors, we'll read 3 physical + interpolate 2 virtual sensors
        self.num_sensors = 5
        self.median_window_size = median_window_size
        
        # Median filter buffers for each sensor
        self.raw_buffers = [[] for _ in range(self.num_sensors)]
        
        # Calibration data
        self.min_values = [0] * self.num_sensors
        self.max_values = [4095] * self.num_sensors  # 12-bit ADC max
        self.calibrated = False
        
        # Sensor weights for weighted average (0, 1000, 2000, 3000, 4000)
        self.weights = [i * 1000 for i in range(self.num_sensors)]
        
        # Tracker sensor initialized
    
    def read_raw(self):
        """Read raw ADC values from all sensors"""
        if len(self.adcs) == 3:
            # Read 3 physical sensors
            raw = [adc.read() for adc in self.adcs]
            
            # Interpolate to create 5 virtual sensors
            # Left sensor, Left-Center interpolation, Center, Right-Center interpolation, Right
            sensor_values = [
                raw[0],                           # Leftmost (sensor 0)
                (raw[0] + raw[1]) // 2,          # Left-Center (sensor 1) 
                raw[1],                           # Center (sensor 2)
                (raw[1] + raw[2]) // 2,          # Right-Center (sensor 3)
                raw[2]                            # Rightmost (sensor 4)
            ]
        else:
            # If we have exactly 5 sensors, read them directly
            sensor_values = [adc.read() for adc in self.adcs[:5]]
            
        return sensor_values
    
    def _median_filter(self, values):
        """Simple median filter implementation for MicroPython"""
        sorted_values = sorted(values)
        n = len(sorted_values)
        if n % 2 == 1:
            return sorted_values[n // 2]
        else:
            return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) // 2
    
    def _apply_median_filter(self, sensor_idx, raw_value):
        """
        Apply median filtering to reduce noise spikes while preserving edges.
        
        Args:
            sensor_idx: Index of the sensor (0-4)
            raw_value: Raw sensor reading
            
        Returns:
            Median-filtered value
        """
        # Add new value to buffer
        self.raw_buffers[sensor_idx].append(raw_value)
        
        # Maintain buffer size
        if len(self.raw_buffers[sensor_idx]) > self.median_window_size:
            self.raw_buffers[sensor_idx].pop(0)  # Remove oldest value
        
        # Return median of buffer values
        if len(self.raw_buffers[sensor_idx]) >= 3:  # Need at least 3 values for meaningful median
            return self._median_filter(self.raw_buffers[sensor_idx])
        else:
            # Not enough data for median, return raw value
            return raw_value
    
    def read_raw_filtered(self):
        """Read raw ADC values with median filtering applied"""
        if len(self.adcs) == 3:
            # Read 3 physical sensors
            raw = [adc.read() for adc in self.adcs]
            
            # Apply median filtering to physical sensors
            filtered_raw = [self._apply_median_filter(i, raw[i]) for i in range(3)]
            
            # Interpolate to create 5 virtual sensors using filtered data
            sensor_values = [
                filtered_raw[0],                                    # Leftmost (sensor 0)
                (filtered_raw[0] + filtered_raw[1]) // 2,          # Left-Center (sensor 1) 
                filtered_raw[1],                                    # Center (sensor 2)
                (filtered_raw[1] + filtered_raw[2]) // 2,          # Right-Center (sensor 3)
                filtered_raw[2]                                     # Rightmost (sensor 4)
            ]
            
            # Apply median filtering to interpolated sensors
            for i in range(3, 5):  # Only filter the interpolated sensors
                sensor_values[i] = self._apply_median_filter(i, sensor_values[i])
                
        else:
            # If we have exactly 5 sensors, read and filter them directly
            raw_values = [adc.read() for adc in self.adcs[:5]]
            sensor_values = [self._apply_median_filter(i, raw_values[i]) for i in range(5)]
            
        return sensor_values
    
    def calibrate(self, samples=100):
        """
        Calibrate sensor by finding min/max values.
        Robot should be moving over the line during calibration.
        """
        # Initialize min/max arrays
        self.min_values = [4095] * self.num_sensors
        self.max_values = [0] * self.num_sensors
        
        for i in range(samples):
            sensor_values = self.read_raw_filtered()  # Use filtered readings for calibration
            
            for j in range(self.num_sensors):
                if sensor_values[j] < self.min_values[j]:
                    self.min_values[j] = sensor_values[j]
                if sensor_values[j] > self.max_values[j]:
                    self.max_values[j] = sensor_values[j]
            
            time.sleep_ms(20)  # 50Hz sampling rate
        
        self.calibrated = True
    
    def read_calibrated(self):
        """
        Read normalized sensor values (0-1000 range).
        1000 = far from line (white surface)
        0 = on line (black surface)
        """
        if not self.calibrated:
            # Use default calibration values
            self.min_values = [500] * self.num_sensors
            self.max_values = [3500] * self.num_sensors
        
        raw_values = self.read_raw_filtered()  # Use filtered readings
        calibrated_values = []
        
        for i in range(self.num_sensors):
            raw = raw_values[i]
            min_val = self.min_values[i]
            max_val = self.max_values[i]
            
            # Normalize to 0-1000 range
            if max_val > min_val:
                normalized = (raw - min_val) * 1000 // (max_val - min_val)
                normalized = max(0, min(1000, normalized))
            else:
                normalized = 500  # Default middle value
            
            calibrated_values.append(normalized)
        
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
        """Check if any sensor detects the line"""
        sensor_values = self.read_calibrated()
        # If any sensor value is below threshold, line is detected
        return any(value < 300 for value in sensor_values)
    
    def reset_filters(self):
        """Reset median filter buffers (useful after line loss or significant changes)"""
        for buffer in self.raw_buffers:
            buffer.clear()
    
    def get_filter_status(self):
        """Get the current state of median filter buffers"""
        return [len(buffer) for buffer in self.raw_buffers]
    
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
            'fl': {'p1': PWM(Pin(5), freq=50), 'p2': PWM(Pin(4), freq=50)},
            'fr': {'p1': PWM(Pin(6), freq=50), 'p2': PWM(Pin(7), freq=50)},
            'bl': {'p1': PWM(Pin(17), freq=50), 'p2': PWM(Pin(16), freq=50)},
            'br': {'p1': PWM(Pin(18), freq=50), 'p2': PWM(Pin(8), freq=50)}
        }
        
        # Encoder pin setup
        self.encoders = {
            'fl': {'a': Pin(3, Pin.IN, Pin.PULL_UP), 'b': Pin(46, Pin.IN, Pin.PULL_UP)},
            'fr': {'a': Pin(9, Pin.IN, Pin.PULL_UP), 'b': Pin(10, Pin.IN, Pin.PULL_UP)},
            'bl': {'a': Pin(11, Pin.IN, Pin.PULL_UP), 'b': Pin(12, Pin.IN, Pin.PULL_UP)},
            'br': {'a': Pin(13, Pin.IN, Pin.PULL_UP), 'b': Pin(14, Pin.IN, Pin.PULL_UP)}
        }
        
        # Initialize tracker sensor with median filtering
        self.tracker = TrackerSensor([19, 20, 21], median_window_size=5)
        
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
