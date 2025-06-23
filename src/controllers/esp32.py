import time
import network
import socket
from machine import Pin, PWM

# WiFi Configuration
WIFI_SSID = "CJ"
WIFI_PASSWORD = "4533simon"
SERVER_PORT = 1234

class OmniwheelRobot:
    def __init__(self):
        # Motor pin setup
        self.motors = {
            'fl': {'p1': PWM(Pin(12), freq=50), 'p2': PWM(Pin(13), freq=50)},
            'fr': {'p1': PWM(Pin(14), freq=50), 'p2': PWM(Pin(27), freq=50)},
            'bl': {'p1': PWM(Pin(25), freq=50), 'p2': PWM(Pin(26), freq=50)},
            'br': {'p1': PWM(Pin(33), freq=50), 'p2': PWM(Pin(18), freq=50)}
        }
        
        print("Motors initialized - Camera-only navigation mode")
        self.stop()

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
                    
                    # Motor speed command
                    parts = command.split(',')
                    if len(parts) == 4:
                        speeds = [int(p) for p in parts]
                        robot.set_all_speeds(*speeds)
                    elif command == "STOP":
                        robot.stop()

                    # Send back acknowledgment (no sensor data)
                    response = "OK\n"
                    conn.send(response.encode())

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

def main():
    # Connect to WiFi
    ip = connect_wifi()
    if not ip:
        print("Failed to connect to WiFi. Exiting.")
        return
    
    # Create robot and start server
    print("Initializing robot...")
    robot = OmniwheelRobot()
    
    print("Starting server...")
    run_server(robot)

if __name__ == "__main__":
    main() 
