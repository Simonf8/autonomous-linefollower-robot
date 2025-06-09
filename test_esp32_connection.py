#!/usr/bin/env python3

import socket
import time
import sys

ESP32_IP = '192.168.2.21'
ESP32_PORT = 1234

def test_connection():
    print(f"Testing connection to ESP32 at {ESP32_IP}:{ESP32_PORT}")
    
    for i in range(10):
        try:
            start_time = time.time()
            sock = socket.create_connection((ESP32_IP, ESP32_PORT), timeout=2.0)
            connect_time = (time.time() - start_time) * 1000
            
            # Try sending a command
            start_time = time.time()
            sock.sendall(b"FORWARD\n")
            send_time = (time.time() - start_time) * 1000
            
            sock.close()
            print(f"✅ Test {i+1}: Connect={connect_time:.1f}ms, Send={send_time:.1f}ms")
            
        except socket.timeout:
            print(f"❌ Test {i+1}: TIMEOUT")
        except Exception as e:
            print(f"💥 Test {i+1}: ERROR - {e}")
        
        time.sleep(0.5)

if __name__ == "__main__":
    test_connection() 