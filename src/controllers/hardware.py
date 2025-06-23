import time
import requests
import logging

class ESP32Bridge:
    """
    Manages communication with the ESP32 microcontroller over HTTP.
    This class is responsible for sending motor commands and checking the connection status.
    """
    def __init__(self, ip: str):
        """
        Initializes the ESP32 bridge.
        :param ip: The IP address of the ESP32.
        """
        self.ip = ip
        self.base_url = f"http://{ip}"
        self.session = requests.Session()
        self.connected = False
        self.last_command_time = 0
        self.command_interval = 0.05  # 50ms between commands

    def connect(self):
        """
        Attempts to establish a connection with the ESP32.
        It sends a status request and checks for a specific response.
        """
        try:
            response = self.session.get(f"{self.base_url}/status", timeout=2)
            if response.status_code == 200 and "ESP32 Ready" in response.text:
                self.connected = True
                print(f"✓ ESP32 connected successfully at {self.ip}")
            else:
                self.connected = False
                logging.warning(f"ESP32 at {self.ip} responded but is not ready. Response: {response.text}")
        except requests.RequestException as e:
            self.connected = False
            logging.error(f"Failed to connect to ESP32 at {self.ip}: {e}")

    def send_motor_commands(self, fl: int, fr: int, bl: int, br: int):
        """
        Sends motor commands for each of the four wheels to the ESP32.
        :param fl: Front-left wheel speed (-255 to 255)
        :param fr: Front-right wheel speed (-255 to 255)
        :param bl: Back-left wheel speed (-255 to 255)
        :param br: Back-right wheel speed (-255 to 255)
        """
        if not self.connected:
            return

        # Throttle commands to avoid flooding the ESP32
        current_time = time.time()
        if current_time - self.last_command_time < self.command_interval:
            return
            
        # Clamp values to be safe
        fl = int(max(-255, min(255, fl)))
        fr = int(max(-255, min(255, fr)))
        bl = int(max(-255, min(255, bl)))
        br = int(max(-255, min(255, br)))

        command_url = f"{self.base_url}/motor?fl={fl}&fr={fr}&bl={bl}&br={br}"
        self._send_command(command_url)
        self.last_command_time = current_time

    def stop(self):
        """Stops all motors by sending zero speeds."""
        if self.connected:
            self.send_motor_commands(0, 0, 0, 0)

    def _send_command(self, url: str):
        """
        Internal method to send a command URL to the ESP32.
        """
        try:
            # Use a short timeout for sending commands as we don't need a response
            self.session.get(url, timeout=0.5)
        except requests.RequestException as e:
            logging.warning(f"Failed to send command to ESP32: {e}")
            # If a command fails, we might have lost connection.
            self.connected = False 