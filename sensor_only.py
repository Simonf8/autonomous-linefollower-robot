#!/usr/bin/env python3

import time
import socket

ESP32_IP = "192.168.128.245"
ESP32_PORT = 1234

def main():
    """Simple sensor reading script - just show [Left Center Right] format"""
    print("Connecting to ESP32 for sensor readings only...")
    
    while True:
        try:
            # Try to connect to ESP32
            sock = socket.create_connection((ESP32_IP, ESP32_PORT), timeout=3)
            sock.settimeout(0.5)
            print(f"Connected to ESP32 at {ESP32_IP}:{ESP32_PORT}")
            
            while True:
                try:
                    # Receive data from ESP32
                    data = sock.recv(1024)
                    if data:
                        data_string = data.decode().strip()
                        for line in data_string.split('\n'):
                            if line:
                                try:
                                    parts = line.strip().split(',')
                                    if len(parts) >= 11:
                                        # Extract sensor values (s0, s1, s2, s3, s4)
                                        sensor_values = [int(parts[i]) for i in range(6, 11)]
                                        
                                        # Convert to your format [Left Center Right]
                                        # Use threshold of 500 for line detection
                                        left = 1 if sensor_values[0] < 500 else 0
                                        center = 1 if sensor_values[2] < 500 else 0  
                                        right = 1 if sensor_values[4] < 500 else 0
                                        print(f"[{left} {center} {right}]")
                                except (ValueError, IndexError):
                                    # Skip invalid data
                                    pass
                except socket.timeout:
                    # No data - continue
                    pass
                except Exception as e:
                    print(f"Receive error: {e}")
                    break
                    
        except Exception as e:
            print(f"Connection failed: {e}")
            print("Retrying in 2 seconds...")
            time.sleep(2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user") 