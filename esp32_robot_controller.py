import time
import network
import socket
from machine import Pin, PWM

# -----------------------------------------------------------------------------
# --- ENHANCED WIFI CONFIGURATION ---
# -----------------------------------------------------------------------------
WIFI_SSID = ""           # Change this
WIFI_PASSWORD = ""   # Change this
SERVER_PORT = 1234

# -----------------------------------------------------------------------------
# --- ENHANCED MOTOR CONFIGURATION ---
# -----------------------------------------------------------------------------
# Motor speeds (0-100) - enhanced for emergency avoidance
FORWARD_SPEED = 55        # Normal forward speed
TURN_SPEED = 50          # Normal turn speed
EMERGENCY_SPEED = 85     # EMERGENCY AVOIDANCE SPEED - Much higher!
STOP_SPEED = 0           # Stop

# Valid commands: 'FORWARD', 'LEFT', 'RIGHT', 'STOP', 'EMERGENCY_LEFT', 'EMERGENCY_RIGHT'

# Motor pins for 2-wheel drive
LEFT_MOTOR_1 = 18    # Left motor pin 1
LEFT_MOTOR_2 = 19    # Left motor pin 2
RIGHT_MOTOR_1 = 12   # Right motor pin 1
RIGHT_MOTOR_2 = 13   # Right motor pin 2

PWM_FREQ = 100

# -----------------------------------------------------------------------------
# --- ENHANCED MOTOR CONTROLLER WITH EMERGENCY AVOIDANCE ---
# -----------------------------------------------------------------------------
class EnhancedMotors:
    def __init__(self):
        # Initialize PWM pins for 2 motors
        self.left_1 = PWM(Pin(LEFT_MOTOR_1), freq=PWM_FREQ)
        self.left_2 = PWM(Pin(LEFT_MOTOR_2), freq=PWM_FREQ)
        
        self.right_1 = PWM(Pin(RIGHT_MOTOR_1), freq=PWM_FREQ)
        self.right_2 = PWM(Pin(RIGHT_MOTOR_2), freq=PWM_FREQ)
        
        self.stop()
        print("Enhanced 2-wheel drive motors initialized with emergency avoidance")
    
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
        print(f"🡅 Forward at {speed}%")
        self._set_left_motor(speed, True)
        self._set_right_motor(speed, True)
    
    def left(self, speed=TURN_SPEED):
        print(f"🡄 Left turn at {speed}%")
        self._set_left_motor(speed, False)  # Left motor backward
        self._set_right_motor(speed, True)   # Right motor forward
    
    def right(self, speed=TURN_SPEED):
        print(f"🡆 Right turn at {speed}%")
        self._set_left_motor(speed, True)    # Left motor forward
        self._set_right_motor(speed, False)  # Right motor backward
    
    def emergency_left(self, speed=EMERGENCY_SPEED):
        print(f"🚨 EMERGENCY LEFT at {speed}% - MAXIMUM AVOIDANCE!")
        # More aggressive left turn - stronger differential
        self._set_left_motor(speed, False)   # Left motor backward at high speed
        self._set_right_motor(speed, True)   # Right motor forward at high speed
        
    def emergency_right(self, speed=EMERGENCY_SPEED):
        print(f"🚨 EMERGENCY RIGHT at {speed}% - MAXIMUM AVOIDANCE!")
        # More aggressive right turn - stronger differential  
        self._set_left_motor(speed, True)    # Left motor forward at high speed
        self._set_right_motor(speed, False)  # Right motor backward at high speed
    
    def stop(self):
        print("🛑 Stop")
        self.left_1.duty(0)
        self.left_2.duty(0)
        self.right_1.duty(0)
        self.right_2.duty(0)

# -----------------------------------------------------------------------------
# --- ENHANCED WIFI CONNECTION ---
# -----------------------------------------------------------------------------
def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    
    if wlan.isconnected():
        ip = wlan.ifconfig()[0]
        print(f"Already connected: {ip}")
        return ip
    
    print(f"Connecting to {WIFI_SSID}...")
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    
    # Wait for connection with better feedback
    timeout = 0
    while not wlan.isconnected() and timeout < 30:
        if timeout % 5 == 0:
            print(f"Connecting... {timeout}s")
        else:
            print(".", end="")
        time.sleep(1)
        timeout += 1
    
    if wlan.isconnected():
        ip = wlan.ifconfig()[0]
        print(f"\n✅ WiFi connected: {ip}")
        print(f"   Network config: {wlan.ifconfig()}")
        return ip
    else:
        print(f"\n❌ WiFi connection failed after {timeout}s")
        return None

# -----------------------------------------------------------------------------
# --- ENHANCED COMMAND PROCESSOR WITH EMERGENCY AVOIDANCE ---
# -----------------------------------------------------------------------------
def process_command(command, motors):
    """Process enhanced commands including emergency avoidance"""
    try:
        command = command.strip().upper()
        
        print(f"📡 Received command: '{command}'")
        
        # Execute movement based on command
        if command == 'FORWARD':
            motors.forward()
            
        elif command == 'LEFT':
            motors.left()
            
        elif command == 'RIGHT':
            motors.right()
            
        elif command == 'EMERGENCY_LEFT':
            motors.emergency_left()
            
        elif command == 'EMERGENCY_RIGHT':
            motors.emergency_right()
            
        elif command == 'STOP':
            motors.stop()
            
        else:
            print(f"❌ Unknown command: '{command}'")
            print("Valid commands: FORWARD, LEFT, RIGHT, EMERGENCY_LEFT, EMERGENCY_RIGHT, STOP")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Command processing error: {e}")
        motors.stop()  # Safety stop on error
        return False

# -----------------------------------------------------------------------------
# --- ENHANCED TCP SERVER ---
# -----------------------------------------------------------------------------
def run_server(motors):
    # Create server socket with better error handling
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server.bind(('', SERVER_PORT))
        server.listen(1)
        server.settimeout(1.0)
        
        print(f"🔗 Enhanced server listening on port {SERVER_PORT}")
        print("🚨 Emergency avoidance commands enabled!")
        
    except Exception as e:
        print(f"❌ Server bind error: {e}")
        return
    
    client = None
    last_command_time = time.time()
    COMMAND_TIMEOUT = 2.0  # Stop if no command for 2 seconds
    
    while True:
        try:
            # Accept connections
            if client is None:
                try:
                    client, addr = server.accept()
                    client.settimeout(0.1)
                    print(f"✅ Client connected from: {addr}")
                    last_command_time = time.time()
                except OSError:
                    pass  # No connection yet
            
            # Handle client data
            if client:
                try:
                    data = client.recv(64)
                    if data:
                        command = data.decode('utf-8').strip()
                        if command:
                            success = process_command(command, motors)
                            if success:
                                last_command_time = time.time()
                    else:
                        # Client disconnected
                        print("📡 Client disconnected")
                        client.close()
                        client = None
                        motors.stop()
                        
                except OSError:
                    # No data available - check timeout
                    current_time = time.time()
                    if current_time - last_command_time > COMMAND_TIMEOUT:
                        print("⏰ Command timeout - stopping for safety")
                        motors.stop()
                        last_command_time = current_time
                    pass
                    
                except Exception as e:
                    print(f"❌ Client communication error: {e}")
                    if client:
                        client.close()
                        client = None
                    motors.stop()
        
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt - stopping server")
            break
            
        except Exception as e:
            print(f"❌ Server error: {e}")
            time.sleep(1)
    
    # Cleanup
    print("🧹 Cleaning up...")
    motors.stop()
    if client:
        client.close()
    server.close()

# -----------------------------------------------------------------------------
# --- MAIN PROGRAM ---
# -----------------------------------------------------------------------------
def main():
    print("🤖 Enhanced ESP32 Line Follower with Emergency Avoidance")
    print("=" * 55)
    
    # Connect WiFi
    ip = connect_wifi()
    if not ip:
        print("❌ Cannot continue without WiFi connection")
        return
    
    # Initialize enhanced motors
    motors = EnhancedMotors()
    
    # Test motors briefly
    print("🔧 Testing motors...")
    motors.forward(30)
    time.sleep(0.5)
    motors.stop()
    time.sleep(0.5)
    
    print(f"🚀 Enhanced robot ready!")
    print(f"📡 Connect Python robot to: {ip}:{SERVER_PORT}")
    print("🚨 Emergency avoidance enabled - EMERGENCY_LEFT/EMERGENCY_RIGHT supported")
    print("🛑 Press Ctrl+C to stop")
    print("-" * 55)
    
    try:
        run_server(motors)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        motors.stop()
        print("✅ ESP32 robot controller stopped safely")

if __name__ == "__main__":
    main() 