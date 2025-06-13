#!/usr/bin/env python3

import socket
import time
import logging

class SimpleESP32:
    """Simple ESP32 communication - no overcomplicated features"""
    
    def __init__(self, ip_address, port=1234):
        self.ip_address = ip_address
        self.port = port
        self.socket = None
        self.connected = False
        
        # Simple sensor data
        self.sensors = [0, 0, 0, 0, 0]  # [L2, L1, C, R1, R2]
        self.line_position = 0.0
        self.line_detected = False
    
    def connect(self):
        """Connect to ESP32"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(2.0)
            self.socket.connect((self.ip_address, self.port))
            self.connected = True
            logging.info(f"Connected to ESP32 at {self.ip_address}")
            return True
        except Exception as e:
            logging.error(f"Failed to connect: {e}")
            return False
    
    def send_command(self, command):
        """Send command to ESP32"""
        if not self.connected:
            return False
        try:
            self.socket.send(command.encode())
            return True
        except:
            self.connected = False
            return False
    
    def read_sensors(self):
        """Read sensor data from ESP32"""
        if not self.connected:
            return False
        
        try:
            self.socket.settimeout(0.1)
            data = self.socket.recv(64).decode().strip()
            
            if ',' in data:
                parts = data.split(',')
                if len(parts) >= 5:
                    self.sensors = [int(float(p)) for p in parts[:5]]
                    self.calculate_position()
                    return True
        except:
            pass
        return False
    
    def calculate_position(self):
        """Calculate line position from sensors"""
        # Simple weighted average
        weights = [-2, -1, 0, 1, 2]
        total_weight = 0
        weighted_sum = 0
        
        for i, sensor in enumerate(self.sensors):
            if sensor > 0:
                weighted_sum += weights[i] * sensor
                total_weight += sensor
        
        if total_weight > 0:
            self.line_position = weighted_sum / total_weight
            self.line_detected = True
        else:
            self.line_detected = False
        
        # Normalize to -1.0 to 1.0
        self.line_position = max(-1.0, min(1.0, self.line_position / 2.0))

class SimplePID:
    """Dead simple PID controller"""
    
    def __init__(self):
        self.kp = 1.5
        self.ki = 0.0
        self.kd = 0.1
        self.last_error = 0.0
        self.integral = 0.0
    
    def update(self, error):
        """Update PID with error"""
        # Proportional
        proportional = self.kp * error
        
        # Integral (with limits)
        self.integral += error
        self.integral = max(-5.0, min(5.0, self.integral))
        integral = self.ki * self.integral
        
        # Derivative
        derivative = self.kd * (error - self.last_error)
        self.last_error = error
        
        # PID output
        output = proportional + integral + derivative
        return max(-1.0, min(1.0, output))

class SimpleLineFollower:
    """Simple state machine line follower"""
    
    def __init__(self, esp32_ip):
        self.esp32 = SimpleESP32(esp32_ip)
        self.pid = SimplePID()
        
        # Simple states
        self.state = "SEARCHING"  # SEARCHING, FOLLOWING, LOST
        self.lost_time = 0
        self.search_direction = 1  # 1 = right, -1 = left
        
        logging.basicConfig(level=logging.INFO)
    
    def run(self):
        """Main control loop"""
        if not self.esp32.connect():
            print("Failed to connect to ESP32")
            return
        
        print("Simple line follower started!")
        
        try:
            while True:
                # Read sensors
                if self.esp32.read_sensors():
                    self.update_state()
                    command = self.get_command()
                    self.esp32.send_command(command)
                    print(f"Sensors: {self.esp32.sensors}, Pos: {self.esp32.line_position:.2f}, State: {self.state}, Cmd: {command}")
                
                time.sleep(0.05)  # 20Hz
                
        except KeyboardInterrupt:
            print("Stopping...")
            self.esp32.send_command("STOP")
    
    def update_state(self):
        """Update state machine"""
        if self.esp32.line_detected:
            if self.state == "LOST":
                print("Line found again!")
            self.state = "FOLLOWING"
            self.lost_time = 0
            self.pid.integral = 0  # Reset integral when line found
        else:
            if self.state == "FOLLOWING":
                print("Line lost!")
                self.lost_time = time.time()
            self.state = "LOST"
    
    def get_command(self):
        """Get motor command based on state"""
        if self.state == "FOLLOWING":
            # Use PID to follow line
            pid_output = self.pid.update(self.esp32.line_position)
            
            # Convert PID output to motor commands
            if abs(pid_output) < 0.1:
                return "FORWARD"
            elif pid_output > 0.3:
                return "LEFT"
            elif pid_output > 0.1:
                return "SLIGHT_LEFT"
            elif pid_output < -0.3:
                return "RIGHT"
            elif pid_output < -0.1:
                return "SLIGHT_RIGHT"
            else:
                return "FORWARD"
        
        elif self.state == "LOST":
            # Simple search pattern
            if time.time() - self.lost_time < 1.0:
                # Try slight movements first
                return "SLIGHT_RIGHT" if self.search_direction > 0 else "SLIGHT_LEFT"
            elif time.time() - self.lost_time < 3.0:
                # Try stronger movements
                return "RIGHT" if self.search_direction > 0 else "LEFT"
            else:
                # Switch search direction every 2 seconds
                if int(time.time() - self.lost_time) % 4 < 2:
                    return "RIGHT"
                else:
                    return "LEFT"
        
        return "STOP"

if __name__ == "__main__":
    # Simple line follower - no bloat!
    robot = SimpleLineFollower("192.168.2.21")
    robot.run() 