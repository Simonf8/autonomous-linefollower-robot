#!/usr/bin/env python3

import time
import sys
import os
import cv2
import numpy as np

# Add the project root and the controllers directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
controllers_path = os.path.join(project_root, 'src', 'controllers')
sys.path.insert(0, project_root)
sys.path.insert(0, controllers_path)

from main import ESP32Bridge
from camera_line_follower import CameraLineFollower

# Configuration
ESP32_IP = "192.168.83.245"  # Update this to match your ESP32's IP
WEBCAM_INDEX = 0  # Usually 0 for built-in camera, 1 for external USB webcam

def test_camera_line_following():
    """Test complete camera-based line following system"""
    print("📹 Testing Complete Camera-Based Line Following")
    print("=" * 60)
    
    # Create components
    esp32 = ESP32Bridge(ESP32_IP)
    camera_line_follower = CameraLineFollower(debug=True)
    
    print(f"🔌 Connecting to ESP32 at {ESP32_IP}...")
    print(f"📷 Connecting to USB webcam...")
    
    # Connect to ESP32
    esp32_connected = esp32.start()
    if esp32_connected:
        print("ESP32 connected successfully!")
    else:
        print("ESP32 not connected - motor commands will be simulated")
    
    # Connect to camera - try multiple indices
    cap = None
    for cam_index in [WEBCAM_INDEX, 0, 1, 2]:
        print(f"Trying camera index {cam_index}...")
        cap = cv2.VideoCapture(cam_index)
        if cap.isOpened():
            print(f"Camera connected successfully at index {cam_index}!")
            break
        cap.release()
        cap = None
    
    if cap is None:
        print("ERROR: Could not connect to any camera")
       
        if esp32_connected:
            esp32.stop()
        return
    
    try:
        
        
        # Test line detection for 10 seconds
        start_time = time.time()
        detection_count = 0
        total_frames = 0
        
        while time.time() - start_time < 10:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read camera frame")
                continue
            
            total_frames += 1
            
            # Resize frame to match robot's camera processing
            resized_frame = cv2.resize(frame, (416, 320))
            
            # Detect line
            line_result = camera_line_follower.detect_line(resized_frame)
            
            if line_result['line_detected']:
                detection_count += 1
                offset = line_result['line_offset']
                confidence = line_result['confidence']
                intersection = line_result.get('intersection_detected', False)
                
                print(f"Line detected: offset={offset:+.3f}, confidence={confidence:.3f}, intersection={intersection}")
            else:
                print(f"No line detected")
            
            # Show debug frame if available
            if line_result.get('processed_frame') is not None:
                cv2.imshow('Camera Line Detection', line_result['processed_frame'])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            time.sleep(0.2)  # 5 FPS for testing
        
        detection_rate = (detection_count / total_frames) * 100 if total_frames > 0 else 0
        print(f"   Detection rate: {detection_rate:.1f}% ({detection_count}/{total_frames} frames)")
        
        print(f"\n🚗 Test 2: Line Following with Motor Commands")
        print("Robot will follow line for 15 seconds...")
        
        # Test line following with motor commands
        start_time = time.time()
        follow_commands = 0
        
        while time.time() - start_time < 15:
            ret, frame = cap.read()
            if not ret:
                continue
            
            resized_frame = cv2.resize(frame, (416, 320))
            line_result = camera_line_follower.detect_line(resized_frame)
            
            if line_result['line_detected']:
                # Get motor speeds
                fl, fr, bl, br = camera_line_follower.get_motor_speeds(line_result, base_speed=30)
                
                # Send to ESP32 if connected
                if esp32_connected:
                    esp32.send_motor_speeds(fl, fr, bl, br)
                
                follow_commands += 1
                offset = line_result['line_offset']
                confidence = line_result['confidence']
                
                print(f"   Following: offset={offset:+.3f}, conf={confidence:.2f}, motors=({fl:+3d},{fr:+3d},{bl:+3d},{br:+3d})")
            else:
                # Stop motors if no line
                if esp32_connected:
                    esp32.send_motor_speeds(0, 0, 0, 0)
                print(f"   No line - stopping motors")
            
            # Show debug frame
            if line_result.get('processed_frame') is not None:
                cv2.imshow('Camera Line Following', line_result['processed_frame'])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            time.sleep(0.1)  # 10 FPS for following
        
        # Stop motors
        if esp32_connected:
            esp32.send_motor_speeds(0, 0, 0, 0)
            print("   Motors stopped")
        
        print(f"   Executed {follow_commands} line following commands")
        
       
        print("Show camera intersections/T-junctions to test detection...")
        
        

        # Test intersection detection
        start_time = time.time()
        intersection_count = 0
        
        while time.time() - start_time < 8:
            ret, frame = cap.read()
            if not ret:
                continue
            
            resized_frame = cv2.resize(frame, (416, 320))
            line_result = camera_line_follower.detect_line(resized_frame)
            
            if line_result.get('intersection_detected', False):
                intersection_count += 1
                print(f"INTERSECTION DETECTED!")
                time.sleep(1)  # Brief pause to avoid multiple triggers
            
            # Show debug frame
            if line_result.get('processed_frame') is not None:
                cv2.imshow('Intersection Detection', line_result['processed_frame'])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            time.sleep(0.1)
        
        print(f"   Detected {intersection_count} intersections")
        
        print("\nCamera-based line following test completed!")
        print("Summary:")
        print(f"  - Line detection rate: {detection_rate:.1f}%")
        print(f"  - Line following commands: {follow_commands}")
        print(f"  - Intersection detections: {intersection_count}")
        print("  - System ready for autonomous navigation!")
        
    except KeyboardInterrupt:
        print("\n\nTest stopped by user")
    except Exception as e:
        print(f"\nTest error: {e}")
    finally:
        # Cleanup
        if esp32_connected:
            esp32.send_motor_speeds(0, 0, 0, 0)
            esp32.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Cleanup completed")

def test_camera_only():
    """Test just camera connection and line detection without ESP32"""
   
    
    camera_line_follower = CameraLineFollower(debug=True)
    
    # Connect to camera - try multiple indices
    cap = None
    for cam_index in [WEBCAM_INDEX, 0, 1, 2]:
        print(f"Trying camera index {cam_index}...")
        cap = cv2.VideoCapture(cam_index)
        if cap.isOpened():
            print(f"Camera connected at index {cam_index}!")
            break
        cap.release()
        cap = None
    
    if cap is None:
        
        return
    
   
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            resized_frame = cv2.resize(frame, (416, 320))
            line_result = camera_line_follower.detect_line(resized_frame)
            
            # Show results
            if line_result['line_detected']:
                offset = line_result['line_offset']
                confidence = line_result['confidence']
                intersection = line_result.get('intersection_detected', False)
                
                print(f"Line: offset={offset:+.3f}, confidence={confidence:.3f}, intersection={intersection}")
            
            # Display debug frame
            if line_result.get('processed_frame') is not None:
                cv2.imshow('Camera Line Detection', line_result['processed_frame'])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--camera-only":
        test_camera_only()
    else:
        test_camera_line_following() 