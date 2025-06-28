#!/usr/bin/env python3
"""
Simple Line Following Vision
==============================================

A simple, fast line detection implementation for a robot.
"""
import os
os.environ['DISPLAY'] = ':0'

import cv2
import numpy as np
import time
from picamera2 import Picamera2
from flask import Flask, render_template, Response

app = Flask(__name__)

# Global variable for the camera
picam2 = None
robot_vision = None

class LineFollowerVision:
    def __init__(self):
       
        self.BLACK_THRESHOLD = 80
        self.BLUR_SIZE = 5
        self.MIN_CONTOUR_AREA = 50
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
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
    
    def process_frame(self, frame):
        """Main processing function"""
        # Fast line detection (every frame)
        line_center, line_binary = self.detect_line_fast(frame)
        
        return {
            'line_center': line_center,
            'line_binary': line_binary
        }
    
    def visualize_results(self, frame, results):
        """Draw all detection results"""
        # Draw line center
        if results['line_center']:
            cv2.circle(frame, results['line_center'], 8, (0, 255, 0), -1)
        
        # Display status based on line detection
        status = "Line Detected" if results['line_center'] else "No Line Detected"
        color = (0, 255, 0) if results['line_center'] else (0, 0, 255)
        
        cv2.putText(frame, f"STATUS: {status}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return frame

def setup_camera():
    global picam2
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()
    time.sleep(1) # wait for camera to warm up

def generate_frames():
    """Video streaming generator function."""
    global picam2, robot_vision
    if picam2 is None:
        setup_camera()
    if robot_vision is None:
        robot_vision = LineFollowerVision()
        
    while True:
        frame = picam2.capture_array()
        
        # Process frame
        results = robot_vision.process_frame(frame)
        
        # Visualize
        frame = robot_vision.visualize_results(frame, results)

        # Robot control logic would go here
        if results['line_center']:
            print(f"Following line - Center: {results['line_center']}")
        else:
            print("No line detected.")

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)
