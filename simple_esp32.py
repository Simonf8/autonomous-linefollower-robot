import time
import network
import socket
from machine import Pin, PWM

# -----------------------------------------------------------------------------
# --- SIMPLE WIFI CONFIGURATION ---
# -----------------------------------------------------------------------------
WIFI_SSID = "CJ"           # Change this
WIFI_PASSWORD = "4533simon"   # Change this
SERVER_PORT = 1234

# -----------------------------------------------------------------------------
# --- SIMPLE MOTOR CONFIGURATION ---
# -----------------------------------------------------------------------------
# Motor speeds (0-100) - simplified for slow, controlled movement
FORWARD_SPEED = 25    # Slow forward speed
TURN_SPEED = 20       # Even slower for turns
STOP_SPEED = 0        # Stop

# Valid commands: 'FORWARD', 'LEFT', 'RIGHT', 'STOP'

# Motor pins for 2-wheel drive (adjust for your hardware)
LEFT_MOTOR_1 = 5    # Left motor pin 1
LEFT_MOTOR_2 = 23   # Left motor pin 2

RIGHT_MOTOR_1 = 18  # Right motor pin 1
RIGHT_MOTOR_2 = 19  # Right motor pin 2

PWM_FREQ = 1000

# -----------------------------------------------------------------------------
# --- SIMPLE MOTOR CONTROLLER ---
# -----------------------------------------------------------------------------
class SimpleMotors:
    def __init__(self):
        # Initialize PWM pins for 2 motors only
        self.left_1 = PWM(Pin(LEFT_MOTOR_1), freq=PWM_FREQ)
        self.left_2 = PWM(Pin(LEFT_MOTOR_2), freq=PWM_FREQ)
        
        self.right_1 = PWM(Pin(RIGHT_MOTOR_1), freq=PWM_FREQ)
        self.right_2 = PWM(Pin(RIGHT_MOTOR_2), freq=PWM_FREQ)
        
        self.stop()
        print("2-wheel drive motors initialized")
    
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
    
    def forward(self, speed):
        print(f"Forward at {speed}%")
        self._set_left_motor(speed, True)
        self._set_right_motor(speed, True)
    
    def left(self, speed):
        print(f"Left at {speed}%")
        self._set_left_motor(speed, False)  # Left motor backward
        self._set_right_motor(speed, True)  # Right motor forward
    
    def right(self, speed):
        print(f"Right at {speed}%")
        self._set_left_motor(speed, True)   # Left motor forward
        self._set_right_motor(speed, False) # Right motor backward
    
    def stop(self):
        print("Stop")
        self.left_1.duty(0)
        self.left_2.duty(0)
        self.right_1.duty(0)
        self.right_2.duty(0)

# -----------------------------------------------------------------------------
# --- SIMPLE WIFI ---
# -----------------------------------------------------------------------------
def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    
    if wlan.isconnected():
        print(f"Already connected: {wlan.ifconfig()[0]}")
        return wlan.ifconfig()[0]
    
    print(f"Connecting to {WIFI_SSID}...")
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    
    # Wait for connection
    timeout = 0
    while not wlan.isconnected() and timeout < 20:
        print(".", end="")
        time.sleep(1)
        timeout += 1
    
    if wlan.isconnected():
        ip = wlan.ifconfig()[0]
        print(f"\n✅ WiFi connected: {ip}")
        return ip
    else:
        print("\n❌ WiFi connection failed")
        return None

# -----------------------------------------------------------------------------
# --- SIMPLE COMMAND PROCESSOR ---
# -----------------------------------------------------------------------------
def process_command(command, motors):
    """Process simple commands: 'FORWARD', 'LEFT', 'RIGHT', 'STOP'"""
    try:
        command = command.strip().upper()
        
        print(f"Command: {command}")
        
        # Execute movement based on command
        if command == 'FORWARD':
            motors.forward(FORWARD_SPEED)
        elif command == 'LEFT':
            motors.left(TURN_SPEED)
        elif command == 'RIGHT':
            motors.right(TURN_SPEED)
        elif command == 'STOP':
            motors.stop()
        else:
            print(f"Unknown command: {command}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Command error: {e}")
        return False

# -----------------------------------------------------------------------------
# --- SIMPLE TCP SERVER ---
# -----------------------------------------------------------------------------
def run_server(motors):
    # Create server socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('', SERVER_PORT))
    server.listen(1)
    server.settimeout(1.0)
    
    print(f"🔗 Server listening on port {SERVER_PORT}")
    
    client = None
    
    while True:
        try:
            # Accept connections
            if client is None:
                try:
                    client, addr = server.accept()
                    client.settimeout(0.1)
                    print(f"✅ Client connected: {addr}")
                except OSError:
                    pass  # No connection yet
            
            # Handle client data
            if client:
                try:
                    data = client.recv(64)
                    if data:
                        command = data.decode('utf-8').strip()
                        if command:
                            process_command(command, motors)
                    else:
                        # Client disconnected
                        print("Client disconnected")
                        client.close()
                        client = None
                        motors.stop()
                        
                except OSError:
                    pass  # No data
                except Exception as e:
                    print(f"Client error: {e}")
                    if client:
                        client.close()
                        client = None
                    motors.stop()
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Server error: {e}")
            time.sleep(1)
    
    # Cleanup
    motors.stop()
    if client:
        client.close()
    server.close()

# -----------------------------------------------------------------------------
# --- MAIN ---
# -----------------------------------------------------------------------------
def main():
    print("🤖 Simple ESP32 Line Follower")
    print("=" * 30)
    
    # Connect WiFi
    ip = connect_wifi()
    if not ip:
        print("Cannot continue without WiFi")
        return
    
    # Initialize motors
    motors = SimpleMotors()
    
    print(f"🚀 Ready! Connect Pi to: {ip}:{SERVER_PORT}")
    print("Press Ctrl+C to stop")
    
    try:
        run_server(motors)
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        motors.stop()
        print("✅ Done")

if __name__ == "__main__":
    main() 