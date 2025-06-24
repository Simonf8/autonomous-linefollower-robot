import time
import socket
import logging

class ESP32Bridge:
    """
    Manages communication with the ESP32 microcontroller over TCP socket.
    This class is responsible for sending motor commands and checking the connection status.
    """
    def __init__(self, ip_with_port: str):
        """
        Initializes the ESP32 bridge.
        :param ip_with_port: The IP address with port (e.g., "192.168.2.38:1234")
        """
        if ':' in ip_with_port:
            self.ip, self.port = ip_with_port.split(':')
            self.port = int(self.port)
        else:
            self.ip = ip_with_port
            self.port = 1234  # Default port
        
        self.socket = None
        self.connected = False
        self.last_command_time = 0
        self.command_interval = 0.05  # 50ms between commands

    def connect(self):
        """
        Attempts to establish a TCP connection with the ESP32.
        """
        try:
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
                self.socket = None
            
            self.socket = socket.create_connection((self.ip, self.port), timeout=2)
            self.socket.settimeout(0.5)
            self.connected = True
            print(f"✓ ESP32 connected successfully at {self.ip}:{self.port}")
            return True
        except Exception as e:
            self.connected = False
            logging.error(f"Failed to connect to ESP32 at {self.ip}:{self.port}: {e}")
            self.socket = None
            return False

    def send_motor_commands(self, fl: int, fr: int, bl: int, br: int):
        """
        Sends motor commands for each of the four wheels to the ESP32.
        :param fl: Front-left wheel speed (-255 to 255)
        :param fr: Front-right wheel speed (-255 to 255)
        :param bl: Back-left wheel speed (-255 to 255)
        :param br: Back-right wheel speed (-255 to 255)
        """
        if not self.connected and not self.connect():
            return

        # Throttle commands to avoid flooding the ESP32
        current_time = time.time()
        if current_time - self.last_command_time < self.command_interval:
            return
            
        # Scale from -255/+255 to -100/+100 (ESP32 expects -100 to +100)
        fl = int(max(-100, min(100, fl * 100 / 255)))
        fr = int(max(-100, min(100, fr * 100 / 255)))
        bl = int(max(-100, min(100, bl * 100 / 255)))
        br = int(max(-100, min(100, br * 100 / 255)))

        # Send command as comma-separated values
        command = f"{fl},{fr},{bl},{br}"
        self._send_command(command)
        self.last_command_time = current_time

    def stop(self):
        """Stops all motors by sending STOP command."""
        if self.connected:
            self._send_command("STOP")

    def _send_command(self, command: str):
        """
        Internal method to send a command string to the ESP32.
        """
        if not self.connected and not self.connect():
            return False
        
        try:
            # Add newline to command and send
            full_command = f"{command}\n"
            self.socket.sendall(full_command.encode())
            return True
        except Exception as e:
            logging.warning(f"Failed to send command to ESP32: {e}")
            self.connected = False
            self.socket = None
            return False 