#!/usr/bin/env python3
"""
Hybrid Line Following + YOLOv11 Object Detection
==============================================

Combines your fast line detection with YOLOv11 object recognition
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time

class HybridRobotVision:
    def __init__(self):
        # Your current line detection parameters
        self.BLACK_THRESHOLD = 80
        self.BLUR_SIZE = 5
        self.MIN_CONTOUR_AREA = 50
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # YOLOv11 for object detection
        self.yolo = YOLO('yolo11n.pt')
        self.last_yolo_detection = time.time()
        self.yolo_interval = 0.2  # Run YOLO every 200ms
        self.current_objects = []
        
        # Danger zones for different objects
        self.danger_distances = {
            'person': 100,     # Stay far from people
            'car': 150,        # Very dangerous
            'bicycle': 80,     # Give space
            'dog': 60,         # Unpredictable
            'stop sign': 200   # Must stop
        }
    
    def detect_line_fast(self, frame):
        """Your current fast line detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.BLUR_SIZE, self.BLUR_SIZE), 0)
        _, binary = cv2.threshold(blurred, self.BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.morph_kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.morph_kernel)
        
        # Find line center
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= self.MIN_CONTOUR_AREA:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy), binary
        
        return None, binary
    
    def detect_objects_yolo(self, frame):
        """Run YOLOv11 object detection (periodically)"""
        current_time = time.time()
        
        # Only run YOLO every few frames to save processing time
        if current_time - self.last_yolo_detection > self.yolo_interval:
            results = self.yolo(frame, verbose=False)
            self.current_objects = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_name = self.yolo.names[int(box.cls[0].cpu().numpy())]
                        
                        if confidence > 0.6:  # High confidence only
                            self.current_objects.append({
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'class': class_name,
                                'confidence': float(confidence),
                                'center': (int((x1+x2)/2), int((y1+y2)/2))
                            })
            
            self.last_yolo_detection = current_time
        
        return self.current_objects
    
    def assess_danger(self, objects, line_center, frame_shape):
        """Assess danger level based on detected objects"""
        if not line_center:
            return "NO_LINE", []
        
        height, width = frame_shape[:2]
        line_x, line_y = line_center
        
        dangers = []
        
        for obj in objects:
            obj_x, obj_y = obj['center']
            distance = np.sqrt((obj_x - line_x)**2 + (obj_y - line_y)**2)
            
            # Check if object is in robot's path
            path_width = 100  # Robot path width
            if abs(obj_x - line_x) < path_width:
                obj_class = obj['class']
                
                # Determine danger level
                if obj_class in self.danger_distances:
                    danger_dist = self.danger_distances[obj_class]
                    
                    if distance < danger_dist:
                        if obj_class == 'stop sign':
                            danger_level = "STOP"
                        elif distance < danger_dist * 0.5:
                            danger_level = "HIGH"
                        else:
                            danger_level = "MEDIUM"
                        
                        dangers.append({
                            'object': obj,
                            'distance': distance,
                            'level': danger_level
                        })
        
        # Return highest danger level
        if any(d['level'] == 'STOP' for d in dangers):
            return "STOP", dangers
        elif any(d['level'] == 'HIGH' for d in dangers):
            return "HIGH_DANGER", dangers
        elif any(d['level'] == 'MEDIUM' for d in dangers):
            return "MEDIUM_DANGER", dangers
        else:
            return "SAFE", dangers
    
    def process_frame(self, frame):
        """Main processing function"""
        # Fast line detection (every frame)
        line_center, line_binary = self.detect_line_fast(frame)
        
        # Periodic object detection
        objects = self.detect_objects_yolo(frame)
        
        # Assess situation
        danger_level, dangers = self.assess_danger(objects, line_center, frame.shape)
        
        return {
            'line_center': line_center,
            'line_binary': line_binary,
            'objects': objects,
            'danger_level': danger_level,
            'dangers': dangers
        }
    
    def visualize_results(self, frame, results):
        """Draw all detection results"""
        # Draw line center
        if results['line_center']:
            cv2.circle(frame, results['line_center'], 8, (0, 255, 0), -1)
        
        # Draw objects
        for obj in results['objects']:
            x1, y1, x2, y2 = obj['bbox']
            
            # Color based on object type
            if obj['class'] == 'person':
                color = (255, 0, 0)  # Red for people
            elif obj['class'] in ['car', 'truck', 'bus']:
                color = (0, 0, 255)  # Blue for vehicles
            else:
                color = (0, 255, 255)  # Yellow for others
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{obj['class']}: {obj['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw danger level
        danger_colors = {
            'SAFE': (0, 255, 0),
            'MEDIUM_DANGER': (0, 255, 255),
            'HIGH_DANGER': (0, 165, 255),
            'STOP': (0, 0, 255),
            'NO_LINE': (128, 128, 128)
        }
        
        color = danger_colors.get(results['danger_level'], (255, 255, 255))
        cv2.putText(frame, f"STATUS: {results['danger_level']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return frame

# Example usage
def main():
    robot_vision = HybridRobotVision()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        results = robot_vision.process_frame(frame)
        
        # Visualize
        frame = robot_vision.visualize_results(frame, results)
        
        # Display
        cv2.imshow('Hybrid Robot Vision', frame)
        
        # Robot control logic would go here
        if results['danger_level'] == 'STOP':
            print("STOPPING - Stop sign detected")
        elif results['danger_level'] == 'HIGH_DANGER':
            print("SLOWING DOWN - Danger detected")
        elif results['line_center']:
            print(f"Following line - Center: {results['line_center']}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
