#!/usr/bin/env python3

import requests
import time
import json

def monitor_robot():
    print("🤖 Robot Monitor Started")
    print("Watching for stopping patterns...")
    
    last_command = None
    command_count = {}
    last_status = None
    
    while True:
        try:
            response = requests.get('http://localhost:5000/api/status', timeout=1)
            data = response.json()
            
            current_command = data.get('command', 'UNKNOWN')
            current_status = data.get('status', 'UNKNOWN')
            confidence = data.get('confidence', 0)
            line_detected = data.get('line_detected', False)
            esp_connected = data.get('esp_connected', False)
            
            # Track command changes
            if current_command != last_command:
                print(f"🔄 Command: {last_command} -> {current_command}")
                last_command = current_command
                command_count = {}
            
            # Count repeated commands
            command_count[current_command] = command_count.get(current_command, 0) + 1
            
            # Detect potential stopping (same command for too long)
            if command_count[current_command] > 50:  # 25 seconds at 2Hz
                print(f"⚠️  POSSIBLE STOPPING: {current_command} sent {command_count[current_command]} times")
                command_count[current_command] = 0  # Reset to avoid spam
            
            # Status changes
            if current_status != last_status:
                print(f"📊 Status: {current_status}")
                last_status = current_status
            
            # Connection issues
            if not esp_connected:
                print("❌ ESP32 DISCONNECTED!")
            
            # Line detection issues
            if not line_detected and confidence == 0:
                print(f"👁️  LINE LOST - searching...")
            
        except requests.exceptions.RequestException as e:
            print(f"🌐 Monitor connection error: {e}")
        except Exception as e:
            print(f"💥 Monitor error: {e}")
        
        time.sleep(0.5)

if __name__ == "__main__":
    monitor_robot() 