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
from camera_obstacle_avoidance import CameraObstacleAvoidance

# Configuration
ESP32_IP = "192.168.83.245"  # Update this to match your ESP32's IP
PHONE_IP = "192.168.83.169"  # Update this to match your phone's IP

def test_camera_obstacle_avoidance():
    """Test complete camera-based obstacle avoidance and corner detection system"""
    print("📹 Testing Camera-Based Obstacle Avoidance & Corner Detection")
    print("=" * 60)
    
    # Create components
    esp32 = ESP32Bridge(ESP32_IP)
    camera_system = CameraObstacleAvoidance(debug=True)
    
    print(f"🔌 Connecting to ESP32 at {ESP32_IP}...")
    print(f"📷 Connecting to camera at {PHONE_IP}...")
    
    # Connect to ESP32
    esp32_connected = esp32.start()
    if esp32_connected:
        print("✅ ESP32 connected successfully!")
    else:
        print("⚠️  ESP32 not connected - motor commands will be simulated")
    
    # Connect to camera
    cap = cv2.VideoCapture(f"http://{PHONE_IP}:8080/video")
    camera_connected = cap.isOpened()
    
    if camera_connected:
        print("✅ Camera connected successfully!")
    else:
        print("❌ ERROR: Could not connect to camera")
        print("Please check:")
        print("1. Phone camera app is running")
        print("2. Phone IP address is correct")
        print("3. Phone and Pi are on same network")
        if esp32_connected:
            esp32.stop()
        return
    
    try:
        print("\n🧪 Test 1: Obstacle Detection")
        print("Point camera at obstacles to test detection...")
        
        # Test obstacle detection for 10 seconds
        start_time = time.time()
        obstacle_count = 0
        total_frames = 0
        
        while time.time() - start_time < 10:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to read camera frame")
                continue
            
            total_frames += 1
            
            # Resize frame to match robot's camera processing
            resized_frame = cv2.resize(frame, (416, 320))
            
            # Detect obstacles and corners
            detection_result = camera_system.detect_obstacles_and_corners(resized_frame)
            
            if detection_result['obstacle_detected']:
                obstacle_count += 1
                direction = detection_result['obstacle_direction']
                distance = detection_result['obstacle_distance']
                action = detection_result['avoidance_action']
                
                print(f"   🚨 OBSTACLE: {direction}, distance={distance:.3f}, action={action}")
            elif detection_result['corner_detected']:
                corner_dir = detection_result['corner_direction']
                angle = detection_result['corner_angle']
                action = detection_result['avoidance_action']
                
                print(f"   🔄 CORNER: {corner_dir}, angle={angle:.1f}°, action={action}")
            else:
                print(f"   ✅ Path clear - continue forward")
            
            # Show debug frame if available
            if detection_result.get('processed_frame') is not None:
                cv2.imshow('Obstacle Avoidance & Corner Detection', detection_result['processed_frame'])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            time.sleep(0.2)  # 5 FPS for testing
        
        detection_rate = (obstacle_count / total_frames) * 100 if total_frames > 0 else 0
        print(f"   Obstacle detection rate: {detection_rate:.1f}% ({obstacle_count}/{total_frames} frames)")
        
        print(f"\n🚗 Test 2: Autonomous Navigation with Obstacle Avoidance")
        print("Robot will navigate autonomously for 15 seconds, avoiding obstacles...")
        
        # Test autonomous navigation with obstacle avoidance
        start_time = time.time()
        avoidance_actions = 0
        forward_actions = 0
        
        while time.time() - start_time < 15:
            ret, frame = cap.read()
            if not ret:
                continue
            
            resized_frame = cv2.resize(frame, (416, 320))
            detection_result = camera_system.detect_obstacles_and_corners(resized_frame)
            
            # Get motor speeds based on detection
            fl, fr, bl, br = camera_system.get_motor_speeds(detection_result, base_speed=35)
            
            # Send to ESP32 if connected
            if esp32_connected:
                esp32.send_motor_speeds(fl, fr, bl, br)
            
            action = detection_result['avoidance_action']
            
            if action != 'continue_forward':
                avoidance_actions += 1
                print(f"   🚨 Action: {action}, motors=({fl:+3d},{fr:+3d},{bl:+3d},{br:+3d})")
            else:
                forward_actions += 1
                if forward_actions % 20 == 0:  # Print every 2 seconds at 10 FPS
                    print(f"   ➡️  Moving forward, motors=({fl:+3d},{fr:+3d},{bl:+3d},{br:+3d})")
            
            # Show debug frame
            if detection_result.get('processed_frame') is not None:
                cv2.imshow('Autonomous Navigation', detection_result['processed_frame'])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            time.sleep(0.1)  # 10 FPS for navigation
        
        # Stop motors
        if esp32_connected:
            esp32.send_motor_speeds(0, 0, 0, 0)
            print("   Motors stopped")
        
        print(f"   Avoidance actions: {avoidance_actions}, Forward actions: {forward_actions}")
        
        print(f"\n🎯 Test 3: Corner Detection")
        print("Show camera corners/intersections to test detection...")
        
        # Test corner detection
        start_time = time.time()
        corner_count = 0
        
        while time.time() - start_time < 8:
            ret, frame = cap.read()
            if not ret:
                continue
            
            resized_frame = cv2.resize(frame, (416, 320))
            detection_result = camera_system.detect_obstacles_and_corners(resized_frame)
            
            if detection_result['corner_detected']:
                corner_count += 1
                direction = detection_result['corner_direction']
                angle = detection_result['corner_angle']
                print(f"   🎯 CORNER DETECTED: {direction}, angle={angle:.1f}°")
                time.sleep(1)  # Brief pause to avoid multiple triggers
            
            # Show debug frame
            if detection_result.get('processed_frame') is not None:
                cv2.imshow('Corner Detection', detection_result['processed_frame'])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            time.sleep(0.1)
        
        print(f"   Detected {corner_count} corners")
        
        print("\n✅ Camera-based obstacle avoidance test completed!")
        print("Summary:")
        print(f"  - Obstacle detection rate: {detection_rate:.1f}%")
        print(f"  - Avoidance actions executed: {avoidance_actions}")
        print(f"  - Corner detections: {corner_count}")
        print("  - System ready for autonomous obstacle avoidance!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Test stopped by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
    finally:
        # Cleanup
        if esp32_connected:
            esp32.send_motor_speeds(0, 0, 0, 0)
            esp32.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Cleanup completed")

def test_camera_only():
    """Test just camera connection and obstacle/corner detection without ESP32"""
    print("📷 Testing Camera-Only Obstacle & Corner Detection")
    print("=" * 50)
    
    camera_system = CameraObstacleAvoidance(debug=True)
    cap = cv2.VideoCapture(f"http://{PHONE_IP}:8080/video")
    
    if not cap.isOpened():
        print("❌ ERROR: Could not connect to camera")
        return
    
    print("✅ Camera connected! Press 'q' to quit")
    print("Point camera at obstacles and corners to see detection in action")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            resized_frame = cv2.resize(frame, (416, 320))
            detection_result = camera_system.detect_obstacles_and_corners(resized_frame)
            
            # Show results
            if detection_result['obstacle_detected']:
                direction = detection_result['obstacle_direction']
                distance = detection_result['obstacle_distance']
                action = detection_result['avoidance_action']
                print(f"OBSTACLE: {direction}, dist={distance:.3f}, action={action}")
            
            if detection_result['corner_detected']:
                direction = detection_result['corner_direction']
                angle = detection_result['corner_angle']
                print(f"CORNER: {direction}, angle={angle:.1f}°")
            
            # Display debug frame
            if detection_result.get('processed_frame') is not None:
                cv2.imshow('Camera Obstacle & Corner Detection', detection_result['processed_frame'])
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
        test_camera_obstacle_avoidance() 