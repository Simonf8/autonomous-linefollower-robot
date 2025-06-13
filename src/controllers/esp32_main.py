import time
import network
import socket
from machine import Pin, ADC, PWM

# Network Configuration
WIFI_SSID = "CJ"  # Fill in your WiFi name
WIFI_PASSWORD = "4533simon"  # Fill in your WiFi password
SERVER_PORT = 1234

# IMPROVED: Communication and timing settings
HEARTBEAT_INTERVAL = 1000  # Send heartbeat every 1000ms
COMMAND_TIMEOUT = 3000     # Command timeout in milliseconds
MESSAGE_QUEUE_SIZE = 10    # Maximum queued messages
SENSOR_READ_INTERVAL = 50  # Read sensors every 50ms (20Hz)

# Motor Configuration - Faster settings
FORWARD_SPEED = 60  # Faster forward movement
TURN_SPEED = 70     # Much faster turning for quicker search
EMERGENCY_SPEED = 80
STOP_SPEED = 0
TURN_AROUND_DURATION = 2.5  # seconds to spin for ~180° (tune as needed)

# Motor Pins
LEFT_MOTOR_1 = 18
LEFT_MOTOR_2 = 19
RIGHT_MOTOR_1 = 12
RIGHT_MOTOR_2 = 13

# Line Sensor Pins - Waveshare ITR20001/T
LINE_SENSORS = [14, 27, 16, 17, 25]  # Correct pins: Left2, Left1, Center, Right1, Right2

# IMPROVED: Advanced reflection filtering parameters
REFLECTION_FILTER_WINDOW = 7      # Number of readings to analyze
REFLECTION_STABILITY_THRESHOLD = 0.6  # Minimum stability for valid reading
REFLECTION_PATTERN_THRESHOLD = 0.7    # Pattern consistency threshold
REFLECTION_VARIANCE_LIMIT = 0.3       # Maximum allowed variance in readings
REFLECTION_DEBOUNCE_TIME = 100        # Milliseconds to debounce rapid changes

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
        
        # IMPROVED: Unified sensor fusion system
        self.last_position = 0
        self.sensor_history = []
        self.position_history = []
        self.confidence_history = []
        self.last_read_time = time.ticks_ms()
        
        # IMPROVED: Advanced reflection filtering system
        self.raw_sensor_history = []  # Store raw sensor readings
        self.filtered_sensor_history = []  # Store filtered readings
        self.sensor_variance_history = []  # Track variance for each sensor
        self.last_stable_reading = [0, 0, 0, 0, 0]  # Last known stable reading
        self.sensor_debounce_timers = [0, 0, 0, 0, 0]  # Debounce timers for each sensor
        self.reflection_confidence = [1.0, 1.0, 1.0, 1.0, 1.0]  # Confidence per sensor
        
        # Sensor fusion parameters
        self.fusion_window = 5  # Number of readings to consider
        self.confidence_threshold = 0.7  # Minimum confidence for position
        self.position_change_limit = 0.5  # Maximum position change per reading
    
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
        """ADVANCED: Multi-layer reflection filtering using statistical analysis and pattern recognition"""
        current_time = time.ticks_ms()
        
        # Store raw readings for analysis
        self.raw_sensor_history.append(sensor_values.copy())
        if len(self.raw_sensor_history) > REFLECTION_FILTER_WINDOW:
            self.raw_sensor_history.pop(0)
        
        # If we don't have enough history, use basic filtering
        if len(self.raw_sensor_history) < 3:
            return sensor_values
        
        filtered_values = []
        
        for sensor_idx in range(len(sensor_values)):
            current_value = sensor_values[sensor_idx]
            
            # LAYER 1: Statistical variance analysis
            variance_filtered = self._apply_variance_filter(sensor_idx, current_value)
            
            # LAYER 2: Temporal consistency check
            temporal_filtered = self._apply_temporal_filter(sensor_idx, variance_filtered, current_time)
            
            # LAYER 3: Pattern recognition filter
            pattern_filtered = self._apply_pattern_filter(sensor_idx, temporal_filtered)
            
            # LAYER 4: Neighbor consistency check
            neighbor_filtered = self._apply_neighbor_filter(sensor_idx, pattern_filtered, sensor_values)
            
            # LAYER 5: Confidence-based final decision
            final_value = self._apply_confidence_filter(sensor_idx, neighbor_filtered, current_time)
            
            filtered_values.append(final_value)
        
        # Store filtered result
        self.filtered_sensor_history.append(filtered_values.copy())
        if len(self.filtered_sensor_history) > REFLECTION_FILTER_WINDOW:
            self.filtered_sensor_history.pop(0)
        
        return filtered_values
    
    def _apply_variance_filter(self, sensor_idx, current_value):
        """Filter based on statistical variance - reflections cause high variance"""
        if len(self.raw_sensor_history) < 5:
            return current_value
        
        # Get recent readings for this sensor
        recent_readings = [reading[sensor_idx] for reading in self.raw_sensor_history[-5:]]
        
        # Calculate variance
        mean_val = sum(recent_readings) / len(recent_readings)
        variance = sum((x - mean_val) ** 2 for x in recent_readings) / len(recent_readings)
        
        # High variance indicates potential reflection
        if variance > REFLECTION_VARIANCE_LIMIT:
            # Use the most common value instead of current reading
            if recent_readings.count(0) > recent_readings.count(1):
                stable_value = 0
            else:
                stable_value = 1
            
            # Reduce confidence for this sensor
            self.reflection_confidence[sensor_idx] *= 0.8
            return stable_value
        else:
            # Low variance - reading is stable
            self.reflection_confidence[sensor_idx] = min(1.0, self.reflection_confidence[sensor_idx] * 1.1)
            return current_value
    
    def _apply_temporal_filter(self, sensor_idx, current_value, current_time):
        """Filter based on temporal consistency - reflections cause rapid changes"""
        # Check if sensor value changed recently
        if len(self.filtered_sensor_history) > 0:
            last_value = self.filtered_sensor_history[-1][sensor_idx]
            
            if current_value != last_value:
                # Value changed - check debounce timer
                if time.ticks_diff(current_time, self.sensor_debounce_timers[sensor_idx]) < REFLECTION_DEBOUNCE_TIME:
                    # Too soon since last change - likely reflection
                    return last_value
                else:
                    # Enough time passed - accept change
                    self.sensor_debounce_timers[sensor_idx] = current_time
                    return current_value
            else:
                # Value unchanged - stable
                return current_value
        else:
            return current_value
    
    def _apply_pattern_filter(self, sensor_idx, current_value):
        """Filter based on pattern recognition - reflections break logical patterns"""
        if len(self.filtered_sensor_history) < 3:
            return current_value
        
        # Get recent pattern for this sensor
        recent_pattern = [reading[sensor_idx] for reading in self.filtered_sensor_history[-3:]]
        recent_pattern.append(current_value)
        
        # Check for reflection patterns (rapid oscillation)
        if len(recent_pattern) >= 4:
            # Pattern like [0,1,0,1] or [1,0,1,0] indicates reflection
            if (recent_pattern == [0,1,0,1] or recent_pattern == [1,0,1,0] or
                recent_pattern == [0,1,0,0] or recent_pattern == [1,0,1,1]):
                # Likely reflection - use most stable value
                stable_value = max(set(recent_pattern[:-1]), key=recent_pattern[:-1].count)
                return stable_value
        
        return current_value
    
    def _apply_neighbor_filter(self, sensor_idx, current_value, all_sensor_values):
        """Filter based on neighbor sensor consistency - reflections usually affect single sensors"""
        # Check if current sensor reading is inconsistent with neighbors
        if sensor_idx == 0:  # Leftmost sensor
            neighbors = [all_sensor_values[1]] if len(all_sensor_values) > 1 else []
        elif sensor_idx == len(all_sensor_values) - 1:  # Rightmost sensor
            neighbors = [all_sensor_values[sensor_idx - 1]]
        else:  # Middle sensors
            neighbors = [all_sensor_values[sensor_idx - 1], all_sensor_values[sensor_idx + 1]]
        
        if neighbors:
            # If current sensor disagrees with all neighbors, it might be reflection
            neighbor_agreement = sum(1 for n in neighbors if n == current_value) / len(neighbors)
            
            if neighbor_agreement < 0.5 and len(self.filtered_sensor_history) > 0:
                # Disagrees with neighbors - use previous stable value
                return self.filtered_sensor_history[-1][sensor_idx]
        
        return current_value
    
    def _apply_confidence_filter(self, sensor_idx, current_value, current_time):
        """Final confidence-based decision with adaptive thresholds"""
        sensor_confidence = self.reflection_confidence[sensor_idx]
        
        # If confidence is low, be more conservative
        if sensor_confidence < REFLECTION_STABILITY_THRESHOLD:
            # Use last known stable reading for this sensor
            if len(self.filtered_sensor_history) > 0:
                stable_value = self.last_stable_reading[sensor_idx]
                
                # Gradually recover confidence if reading is consistent
                if current_value == stable_value:
                    self.reflection_confidence[sensor_idx] = min(1.0, sensor_confidence * 1.05)
                
                return stable_value
        
        # High confidence - accept current reading and update stable reading
        self.last_stable_reading[sensor_idx] = current_value
        return current_value
    
    def get_line_position(self):
        """IMPROVED: Unified sensor fusion with confidence and temporal filtering"""
        current_time = time.ticks_ms()
        
        # Non-blocking sensor read with timing control
        if time.ticks_diff(current_time, self.last_read_time) < SENSOR_READ_INTERVAL:
            # Return cached position if reading too frequently
            return self.last_position, 0 if self.last_position is not None else 1
        
        self.last_read_time = current_time
        values = self.read_digital()
        
        # Check if any sensors detect the line (value = 1 means line detected)
        total = sum(values)
        if total == 0:
            print(f"NO LINE: All sensors read 0 (white surface): {values}")
            return self.last_position, 1  # Return 1 for "no line detected"
        
        # IMPROVED: Calculate confidence including reflection filtering quality
        confidence = self._calculate_confidence(values, total)
        
        # IMPROVED: Better center detection and balancing
        # Sensors: [Left2, Left1, Center, Right1, Right2] = indices [0, 1, 2, 3, 4]
        # Positions: [-2, -1, 0, 1, 2] relative to center
        
        # Special case: Only center sensor detects line = perfect center
        if values == [0, 0, 1, 0, 0]:
            position = 0.0
            confidence = 1.0  # Perfect confidence for center detection
        else:
            # Calculate weighted position with improved logic
            weighted_sum = 0
            for i, value in enumerate(values):
                sensor_position = i - 2  # Convert index to position relative to center
                weighted_sum += value * sensor_position
            
            position = weighted_sum / total
            
            # IMPROVED: Better handling of edge cases and multiple sensor combinations
            if values[2] == 1:  # Center sensor is active
                if total == 2:
                    # Center + one adjacent sensor
                    if values[1] == 1:  # Center + Left1
                        position = -0.3  # Slightly left of center
                    elif values[3] == 1:  # Center + Right1
                        position = 0.3   # Slightly right of center
                elif total == 3:
                    # Center + two sensors - use weighted but reduce extreme values
                    position = position * 0.8  # Reduce sensitivity when center is involved
            
            # Normalize to -1.0 to +1.0 range (divide by 2 since max position is ±2)
            position = position / 2.0
        
        # IMPROVED: Sensor fusion with temporal consistency
        fused_position = self._apply_sensor_fusion(position, confidence)
        
        # IMPROVED: Clamp position to valid range
        fused_position = max(-1.0, min(1.0, fused_position))
        
        print(f"LINE POS: sensors={values}, pos={fused_position:.3f}, conf={confidence:.2f}")
        self.last_position = fused_position
        return fused_position, 0  # Return 0 for "line detected"
    
    def _calculate_confidence(self, values, total):
        """IMPROVED: Calculate confidence including reflection filtering quality"""
        if total == 0:
            return 0.0
        
        # Base confidence from number of active sensors
        base_confidence = min(total / 3.0, 1.0)  # Optimal is 2-3 sensors
        
        # Bonus for center sensor being active
        center_bonus = 0.2 if values[2] == 1 else 0.0
        
        # Penalty for edge sensors only
        edge_penalty = 0.0
        if values[0] == 1 or values[4] == 1:  # Outer sensors active
            if values[1] == 0 and values[2] == 0 and values[3] == 0:  # Only outer sensors
                edge_penalty = 0.3
        
        # Pattern consistency bonus
        pattern_bonus = 0.0
        if total >= 2:
            # Check for continuous sensor activation (good pattern)
            continuous = True
            first_active = -1
            last_active = -1
            for i, val in enumerate(values):
                if val == 1:
                    if first_active == -1:
                        first_active = i
                    last_active = i
            
            # Check if all sensors between first and last are active
            for i in range(first_active, last_active + 1):
                if values[i] == 0:
                    continuous = False
                    break
            
            if continuous:
                pattern_bonus = 0.2
        
        # IMPROVED: Reflection filtering confidence factor
        reflection_confidence_avg = sum(self.reflection_confidence) / len(self.reflection_confidence)
        reflection_bonus = (reflection_confidence_avg - 0.5) * 0.3  # Can be negative if low confidence
        
        # IMPROVED: Sensor stability factor
        stability_bonus = 0.0
        if len(self.filtered_sensor_history) >= 3:
            # Check how stable the readings have been
            recent_readings = self.filtered_sensor_history[-3:]
            stability_count = 0
            for sensor_idx in range(len(values)):
                sensor_readings = [reading[sensor_idx] for reading in recent_readings]
                if len(set(sensor_readings)) == 1:  # All same value
                    stability_count += 1
            
            stability_ratio = stability_count / len(values)
            stability_bonus = stability_ratio * 0.2
        
        confidence = base_confidence + center_bonus + pattern_bonus + reflection_bonus + stability_bonus - edge_penalty
        return max(0.0, min(1.0, confidence))
    
    def _apply_sensor_fusion(self, new_position, confidence):
        """Apply temporal sensor fusion for smoother, more reliable positioning"""
        # Add to history
        self.position_history.append(new_position)
        self.confidence_history.append(confidence)
        
        # Maintain history size
        if len(self.position_history) > self.fusion_window:
            self.position_history.pop(0)
            self.confidence_history.pop(0)
        
        # If we don't have enough history, use current reading with light smoothing
        if len(self.position_history) < 2:
            return new_position
        
        # Weighted average based on confidence and recency
        total_weight = 0
        weighted_sum = 0
        
        for i, (pos, conf) in enumerate(zip(self.position_history, self.confidence_history)):
            # Recent readings get higher weight
            recency_weight = (i + 1) / len(self.position_history)
            # Confidence weight
            conf_weight = conf
            # Combined weight
            weight = recency_weight * conf_weight
            
            weighted_sum += pos * weight
            total_weight += weight
        
        if total_weight > 0:
            fused_position = weighted_sum / total_weight
        else:
            fused_position = new_position
        
        # Limit sudden position changes for stability
        if self.last_position is not None:
            max_change = self.position_change_limit
            position_diff = fused_position - self.last_position
            if abs(position_diff) > max_change:
                fused_position = self.last_position + (max_change if position_diff > 0 else -max_change)
        
        return fused_position
    
    def get_reflection_diagnostics(self):
        """Get diagnostic information about reflection filtering performance"""
        if len(self.raw_sensor_history) < 2:
            return None
        
        diagnostics = {
            'sensor_confidence': self.reflection_confidence.copy(),
            'avg_confidence': sum(self.reflection_confidence) / len(self.reflection_confidence),
            'min_confidence': min(self.reflection_confidence),
            'max_confidence': max(self.reflection_confidence),
            'stability_score': 0.0,
            'reflection_events': 0,
            'filter_effectiveness': 0.0
        }
        
        # Calculate stability score
        if len(self.filtered_sensor_history) >= 3:
            stable_sensors = 0
            for sensor_idx in range(len(self.reflection_confidence)):
                recent_readings = [reading[sensor_idx] for reading in self.filtered_sensor_history[-3:]]
                if len(set(recent_readings)) <= 1:  # Stable (0 or 1 unique values)
                    stable_sensors += 1
            diagnostics['stability_score'] = stable_sensors / len(self.reflection_confidence)
        
        # Count potential reflection events (high variance periods)
        if len(self.raw_sensor_history) >= 5:
            reflection_events = 0
            for sensor_idx in range(len(self.reflection_confidence)):
                recent_raw = [reading[sensor_idx] for reading in self.raw_sensor_history[-5:]]
                mean_val = sum(recent_raw) / len(recent_raw)
                variance = sum((x - mean_val) ** 2 for x in recent_raw) / len(recent_raw)
                if variance > REFLECTION_VARIANCE_LIMIT:
                    reflection_events += 1
            diagnostics['reflection_events'] = reflection_events
        
        # Calculate filter effectiveness (how much filtering changed the readings)
        if len(self.raw_sensor_history) > 0 and len(self.filtered_sensor_history) > 0:
            raw_latest = self.raw_sensor_history[-1]
            filtered_latest = self.filtered_sensor_history[-1]
            changes = sum(1 for i in range(len(raw_latest)) if raw_latest[i] != filtered_latest[i])
            diagnostics['filter_effectiveness'] = changes / len(raw_latest)
        
        return diagnostics
    
    def print_reflection_status(self):
        """Print current reflection filtering status for debugging"""
        diag = self.get_reflection_diagnostics()
        if diag:
            print(f"REFLECTION STATUS:")
            print(f"  Avg Confidence: {diag['avg_confidence']:.2f}")
            print(f"  Stability: {diag['stability_score']:.2f}")
            print(f"  Active Reflections: {diag['reflection_events']}")
            print(f"  Filter Activity: {diag['filter_effectiveness']:.2f}")
            print(f"  Per-Sensor Conf: {[f'{c:.2f}' for c in diag['sensor_confidence']]}")
    
    def calibrate(self, samples=100):
        """IMPROVED: Calibration with reflection filtering diagnostics"""
        print("Digital line sensors don't need calibration")
        print("Sensors will read 1 for line detected, 0 for no line")
        
        # Test read to make sure sensors are working
        print("Testing sensor readings:")
        values = self.read_digital()
        for i, value in enumerate(values):
            print(f"  Sensor {i} (pin {self.sensor_pins[i]}): {value}")
        
        # Initialize reflection filtering with some baseline readings
        print("Initializing reflection filtering...")
        for _ in range(10):
            test_values = self.read_digital()
            time.sleep_ms(50)  # Small delay between readings
        
        print("Calibration complete")
        print("Reflection filtering initialized and ready")

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
        # Then do a 180-degree turn (duration tuned by constant)
        self._set_left_motor(speed, True)
        self._set_right_motor(speed, False)
        time.sleep(TURN_AROUND_DURATION)
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
    """IMPROVED: Non-blocking server with message queue and heartbeat"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server.bind(('', SERVER_PORT))
        server.listen(1)
        server.settimeout(0.1)  # Non-blocking with short timeout
        print(f"Server listening on port {SERVER_PORT}")
        
    except Exception as e:
        print(f"Server bind error: {e}")
        return
    
    # IMPROVED: Message queue and heartbeat system
    client = None
    message_queue = []
    last_command_time = time.ticks_ms()
    last_heartbeat_time = time.ticks_ms()
    last_sensor_send_time = time.ticks_ms()
    last_reflection_report_time = time.ticks_ms()  # For periodic reflection diagnostics
    connection_stable = False
    
    while True:
        current_time = time.ticks_ms()
        
        try:
            # Accept new connections (non-blocking)
            if client is None:
                try:
                    client, addr = server.accept()
                    client.settimeout(0.05)  # Very short timeout for non-blocking
                    print(f"Client connected from: {addr}")
                    last_command_time = current_time
                    last_heartbeat_time = current_time
                    connection_stable = True
                    message_queue.clear()  # Clear any old messages
                except OSError:
                    pass  # No connection available
            
            if client:
                # IMPROVED: Non-blocking message processing
                try:
                    data = client.recv(64)
                    if data:
                        command = data.decode('utf-8').strip()
                        if command:
                            # Add to message queue for processing
                            if len(message_queue) < MESSAGE_QUEUE_SIZE:
                                message_queue.append({
                                    'command': command,
                                    'timestamp': current_time
                                })
                            last_command_time = current_time
                            connection_stable = True
                    else:
                        print("Client disconnected")
                        client.close()
                        client = None
                        connection_stable = False
                        motors.stop()
                        
                except OSError:
                    # No data available - this is normal for non-blocking
                    pass
                except Exception as e:
                    print(f"Client communication error: {e}")
                    if client:
                        client.close()
                        client = None
                    connection_stable = False
                    motors.stop()
                
                # IMPROVED: Process message queue
                if message_queue:
                    message = message_queue.pop(0)  # Process oldest message first
                    success = process_command(message['command'], motors)
                    if success:
                        last_command_time = current_time
                
                # IMPROVED: Send sensor data and heartbeat
                if connection_stable:
                    try:
                        # Send sensor data at regular intervals
                        if time.ticks_diff(current_time, last_sensor_send_time) >= SENSOR_READ_INTERVAL:
                            position, detected = line_sensor.get_line_position()
                            sensor_data = f"{position:.3f},{detected}\n"
                            client.send(sensor_data.encode('utf-8'))
                            last_sensor_send_time = current_time
                        
                        # Send heartbeat
                        if time.ticks_diff(current_time, last_heartbeat_time) >= HEARTBEAT_INTERVAL:
                            heartbeat = "HEARTBEAT\n"
                            client.send(heartbeat.encode('utf-8'))
                            last_heartbeat_time = current_time
                            
                    except Exception as e:
                        print(f"Failed to send data: {e}")
                        connection_stable = False
                
                # IMPROVED: Command timeout handling
                if time.ticks_diff(current_time, last_command_time) > COMMAND_TIMEOUT:
                    if connection_stable:
                        print("Command timeout - stopping motors")
                        motors.stop()
                        connection_stable = False
            
            # Periodic reflection diagnostics
            if time.ticks_diff(current_time, last_reflection_report_time) >= 5000:  # Every 5 seconds
                reflection_status = line_sensor.get_reflection_diagnostics()
                if reflection_status:
                    print("REFLECTION DIAGNOSTICS:")
                    print(f"  Avg Confidence: {reflection_status['avg_confidence']:.2f}")
                    print(f"  Stability: {reflection_status['stability_score']:.2f}")
                    print(f"  Active Reflections: {reflection_status['reflection_events']}")
                    print(f"  Filter Activity: {reflection_status['filter_effectiveness']:.2f}")
                    print(f"  Per-Sensor Conf: {[f'{c:.2f}' for c in reflection_status['sensor_confidence']]}")
                last_reflection_report_time = current_time
        
        except KeyboardInterrupt:
            print("Keyboard interrupt - stopping server")
            break
            
        except Exception as e:
            print(f"Server error: {e}")
            # Don't sleep on error - just continue to maintain responsiveness
    
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