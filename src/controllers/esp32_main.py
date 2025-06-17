import time
import network
import socket
import math
from machine import Pin, PWM

# WiFi Configuration
WIFI_SSID = "YOUR_SSID"
WIFI_PASSWORD = "YOUR_PASSWORD"
SERVER_PORT = 1234

class OmniwheelRobot:
    def __init__(self):
        # Motor pin setup
        self.motors = {
            'fl': {'p1': PWM(Pin(5), freq=1000), 'p2': PWM(Pin(4), freq=1000)},
            'fr': {'p1': PWM(Pin(6), freq=1000), 'p2': PWM(Pin(7), freq=1000)},
            'bl': {'p1': PWM(Pin(17), freq=1000), 'p2': PWM(Pin(16), freq=1000)},
            'br': {'p1': PWM(Pin(18), freq=1000), 'p2': PWM(Pin(8), freq=1000)}
        }
        
        # Encoder pin setup
        self.encoders = {
            'fl': {'a': Pin(3, Pin.IN, Pin.PULL_UP), 'b': Pin(46, Pin.IN, Pin.PULL_UP)},
            'fr': {'a': Pin(9, Pin.IN, Pin.PULL_UP), 'b': Pin(10, Pin.IN, Pin.PULL_UP)},
            'bl': {'a': Pin(11, Pin.IN, Pin.PULL_UP), 'b': Pin(12, Pin.IN, Pin.PULL_UP)},
            'br': {'a': Pin(13, Pin.IN, Pin.PULL_UP), 'b': Pin(14, Pin.IN, Pin.PULL_UP)}
        }
        
        # Global-like storage for ticks and encoder B-pins for ISRs
        self.ticks = {'fl': 0, 'fr': 0, 'bl': 0, 'br': 0}
        
        self.setup_encoders()
        self.stop()
        print("Omniwheel Robot Initialized")

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
        print(f"WiFi connected successfully. IP: {ip}")
        return ip
    else:
        print("WiFi connection failed.")
        return None

def run_server(robot):
    """Runs a server to send encoder data and receive motor commands."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('', SERVER_PORT))
    server.listen(1)
    server.settimeout(0.2)
    print(f"Server listening on port {SERVER_PORT}")
    
    client = None
    last_send_time = time.ticks_ms()
    
    while True:
        try:
            if client is None:
                print("Waiting for connection from Raspberry Pi...")
                try:
                    client, addr = server.accept()
                    client.settimeout(0.2)
                    print(f"Raspberry Pi connected from: {addr[0]}")
                except OSError:
                    continue

            # Send encoder data periodically
            if time.ticks_diff(time.ticks_ms(), last_send_time) >= 50: # 20Hz
                fl, fr, bl, br = robot.get_ticks()
                data_str = f"{fl},{fr},{bl},{br}\n"
                try:
                    client.sendall(data_str.encode())
                    last_send_time = time.ticks_ms()
                except OSError:
                    print("Send failed. Client disconnected.")
                    client.close()
                    client = None
                    robot.stop()
                    continue

            # Receive and process motor commands
            try:
                data = client.recv(128)
                if data:
                    command = data.decode().strip()
                    if ',' in command:
                        parts = command.split(',')
                        if len(parts) == 4:
                            speeds = [int(float(p)) for p in parts]
                            robot.set_all_speeds(speeds[0], speeds[1], speeds[2], speeds[3])
                else:
                    # No data means client disconnected
                    print("Client connection lost.")
                    client.close()
                    client = None
                    robot.stop()
            except OSError:
                pass # No data received, which is normal

        except Exception as e:
            print(f"An error occurred: {e}")
            if client:
                client.close()
                client = None
            robot.stop()
            time.sleep(1)

def main():
    print("ESP32 Omni-Wheel Motor & Encoder Controller")
    if not connect_wifi():
        print("Could not connect to WiFi. Halting.")
        return
    
    robot = OmniwheelRobot()
    
    try:
        run_server(robot)
    except KeyboardInterrupt:
        print("Program stopped by user.")
    finally:
        print("Cleaning up...")
        robot.stop()

if __name__ == "__main__":
    main() 