import time
import socket

class ESP32Bridge:
    """ESP32 communication bridge for motors only (camera-based navigation)."""
    
    def __init__(self, ip: str, port: int = 1234):
        self.ip = ip
        self.port = port
        self.socket = None
        self.connected = False
        self.connection_attempts = 0
        
        # Command tracking
        self.last_command = None
        self.last_send_time = 0.0
        
    def start(self):
        """Start communication with ESP32."""
        return self.connect()
        
    def connect(self):
        """Establish connection to ESP32."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        if self.connection_attempts > 0:
            time.sleep(1)

        try:
            self.socket = socket.create_connection((self.ip, self.port), timeout=3)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            if hasattr(socket, "TCP_KEEPIDLE"):
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 1)
            if hasattr(socket, "TCP_KEEPINTVL"):
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 3)
            if hasattr(socket, "TCP_KEEPCNT"):
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)

            self.socket.settimeout(0.5)
            self.connected = True
            self.connection_attempts = 0
            print(f"Connected to ESP32 at {self.ip}:{self.port}")
            return True
        except Exception as e:
            self.connected = False
            self.connection_attempts += 1
            if self.connection_attempts % 10 == 1:
                print(f"Failed to connect to ESP32 (attempt {self.connection_attempts}): {e}")
            self.socket = None
            return False
        
    def send_motor_speeds(self, fl: int, fr: int, bl: int, br: int):
        """Send motor speeds to ESP32."""
        if not self.connected and not self.connect():
            return False
            
        try:
            command = f"{fl},{fr},{bl},{br}"
            return self._send_command(command)
        except Exception as e:
            print(f"Error sending motor speeds: {e}")
            self.connected = False
            return False
    
    def send_stop_command(self):
        """Send stop command to ESP32."""
        if not self.connected and not self.connect():
            return False
            
        try:
            return self._send_command("STOP")
        except Exception as e:
            print(f"Error sending stop command: {e}")
            self.connected = False
            return False
    
    def _send_command(self, command: str):
        """Internal method to send command to ESP32."""
        if not self.socket:
            return False
            
        try:
            full_command = f"{command}\n"
            current_time = time.time()
            
            self.socket.sendall(full_command.encode())
            self.last_command = full_command
            self.last_send_time = current_time
            
            return True
        except Exception as e:
            print(f"Socket error: {e}")
            self.connected = False
            self.socket = None
            return False
            
    def _receive_data(self):
        """Try to receive acknowledgment from ESP32 (non-blocking)."""
        if not self.socket or not self.connected:
            return
            
        try:
            data = self.socket.recv(1024)
            if data:
                response = data.decode().strip()
                # Optionally log response
            elif len(data) == 0:
                print("ESP32 closed the connection.")
                self.connected = False
                self.socket.close()
                self.socket = None
        except (socket.timeout, BlockingIOError):
            pass
        except Exception as e:
            print(f"Socket receive error: {e}")
            self.connected = False
            self.socket = None
            
    def stop(self):
        """Close connection to ESP32."""
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None 