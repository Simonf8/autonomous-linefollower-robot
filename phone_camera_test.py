from flask import Flask, Response, render_template_string, jsonify
import cv2
import numpy as np
import time
import threading
import logging
from collections import deque

# Simple obstacle detection - no need for YOLO!
print("Using simple obstacle detection (non-black objects on line path)")

app = Flask(__name__)

# CHANGE THIS TO YOUR PHONE'S IP ADDRESS
PHONE_IP = "192.168.83.169"

# Configuration
CAMERA_WIDTH, CAMERA_HEIGHT = 416, 320
DETECTION_MODE = "both"  # "line", "obstacle", or "both"

# Line detection parameters
BLACK_THRESHOLD = 80
BLUR_SIZE = 5
MIN_CONTOUR_AREA = 50

# Obstacle detection parameters (non-black objects on line path)
OBSTACLE_DETECTION_ENABLED = True
OBSTACLE_COLOR_THRESHOLD = 100  # Anything brighter than this is considered obstacle
OBSTACLE_MIN_AREA = 200  # Minimum area to consider as obstacle
LINE_PATH_WIDTH = 100  # Width of the line path to check for obstacles (pixels)

# Global state
detection_state = {
    'mode': DETECTION_MODE,
    'line_detected': False,
    'line_offset': 0.0,
    'confidence': 0.0,
    'fps': 0.0,
    'obstacles': [],
    'obstacle_detected': False,
    'status': 'Initializing'
}

# Thread-safe frame buffer
frame_buffer = {
    'frame': None,
    'processed_frame': None,
    'lock': threading.Lock()
}

# No need for YOLO - using simple color-based obstacle detection

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PhoneCameraTest")

class ImageProcessor:
    def __init__(self):
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
    def detect_line(self, frame):
        """Detect line in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)
        
        # Binary threshold
        _, binary = cv2.threshold(blurred, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.morph_kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.morph_kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, binary
        
        # Find the largest contour (assuming it's the line)
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < MIN_CONTOUR_AREA:
            return None, binary
        
        # Calculate center of the line
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Calculate confidence based on contour area
            height, width = binary.shape
            confidence = min(1.0, cv2.contourArea(largest_contour) / (width * height * 0.1))
            
            return {
                'center': (cx, cy),
                'contour': largest_contour,
                'confidence': confidence
            }, binary
        
        return None, binary
    
    def detect_obstacles(self, frame, line_info=None):
        """Detect obstacles that are blocking the black line"""
        if line_info is None:
            return []
        
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get the original line binary mask for comparison
        blurred = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)
        _, original_line_mask = cv2.threshold(blurred, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        
        # Define the line path area to check for obstacles
        line_x = line_info['center'][0]
        path_left = max(0, line_x - LINE_PATH_WIDTH // 2)
        path_right = min(width, line_x + LINE_PATH_WIDTH // 2)
        
        # Extract the line path region from original line mask
        line_path_mask = original_line_mask[:, path_left:path_right]
        
        # Find where the line SHOULD be continuous
        line_contours, _ = cv2.findContours(line_path_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not line_contours:
            return []
        
        # Get the main line contour in the path area
        main_line_contour = max(line_contours, key=cv2.contourArea)
        
        # Create a mask of where we expect the line to be
        expected_line_mask = np.zeros_like(line_path_mask)
        cv2.drawContours(expected_line_mask, [main_line_contour], -1, 255, -1)
        
        # Now check the actual grayscale image in the same area
        actual_path_roi = gray[:, path_left:path_right]
        
        # Find areas where the line should be black but isn't (blocked by obstacles)
        # These are areas where:
        # 1. We expect line (expected_line_mask is white)
        # 2. But the actual image is bright (not black)
        
        obstacles = []
        
        # Divide the path into horizontal strips to check for line continuity
        strip_height = 20  # Check every 20 pixels vertically
        
        for y in range(0, height - strip_height, strip_height):
            y_end = min(y + strip_height, height)
            
            # Extract strip from expected line mask
            expected_strip = expected_line_mask[y:y_end, :]
            actual_strip = actual_path_roi[y:y_end, :]
            
            # If there should be line in this strip
            if np.any(expected_strip > 0):
                # Check if the actual strip has bright pixels where line should be
                line_pixels_mask = expected_strip > 0
                actual_line_pixels = actual_strip[line_pixels_mask]
                
                # If the average brightness where line should be is too high, there's an obstacle
                if len(actual_line_pixels) > 0:
                    avg_brightness = np.mean(actual_line_pixels)
                    
                    # If it's significantly brighter than expected black line
                    if avg_brightness > OBSTACLE_COLOR_THRESHOLD:
                        # Find the bright contours in this strip
                        _, bright_mask = cv2.threshold(actual_strip, OBSTACLE_COLOR_THRESHOLD, 255, cv2.THRESH_BINARY)
                        
                        # Only consider areas that overlap with expected line
                        blocking_mask = cv2.bitwise_and(bright_mask, expected_strip)
                        
                        if np.sum(blocking_mask) > OBSTACLE_MIN_AREA // 4:  # Smaller threshold for strips
                            # Find contours of the blocking object
                            block_contours, _ = cv2.findContours(blocking_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            for block_contour in block_contours:
                                if cv2.contourArea(block_contour) < 50:  # Very small threshold for strip
                                    continue
                                
                                # Get bounding box
                                bx, by, bw, bh = cv2.boundingRect(block_contour)
                                
                                # Convert coordinates back to full frame
                                bx += path_left
                                by += y
                                
                                # Calculate center
                                center_x = bx + bw // 2
                                center_y = by + bh // 2
                                
                                # Calculate how much of the line is blocked
                                blocking_ratio = cv2.contourArea(block_contour) / np.sum(expected_strip > 0) if np.sum(expected_strip > 0) > 0 else 0
                                
                                # Only consider it an obstacle if it blocks a significant portion
                                if blocking_ratio > 0.3:  # Blocks at least 30% of expected line in this strip
                                    obstacles.append({
                                        'bbox': (bx, by, bx + bw, by + bh),
                                        'center': (center_x, center_y),
                                        'area': cv2.contourArea(block_contour),
                                        'confidence': min(1.0, blocking_ratio * 2),  # Higher blocking = higher confidence
                                        'distance_from_line': abs(center_x - line_x),
                                        'blocking_ratio': blocking_ratio,
                                        'type': 'line_blocker'
                                    })
        
        # Remove duplicate obstacles (merge nearby ones)
        filtered_obstacles = []
        for obstacle in obstacles:
            # Check if this obstacle is too close to an existing one
            is_duplicate = False
            for existing in filtered_obstacles:
                dist = np.sqrt((obstacle['center'][0] - existing['center'][0])**2 + 
                             (obstacle['center'][1] - existing['center'][1])**2)
                if dist < 30:  # Within 30 pixels
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if obstacle['confidence'] > existing['confidence']:
                        filtered_obstacles.remove(existing)
                        filtered_obstacles.append(obstacle)
                    break
            
            if not is_duplicate:
                filtered_obstacles.append(obstacle)
        
        return filtered_obstacles

# Initialize image processor
processor = ImageProcessor()

def process_frame(frame):
    """Process frame for line detection and/or object detection"""
    processed_frame = frame.copy()
    
    # Line detection
    line_info = None
    if detection_state['mode'] in ['line', 'both']:
        line_info, binary = processor.detect_line(frame)
        
        if line_info:
            detection_state['line_detected'] = True
            detection_state['confidence'] = line_info['confidence']
            
            # Calculate offset from center
            height, width = frame.shape[:2]
            center_x = width // 2
            line_x = line_info['center'][0]
            offset = (line_x - center_x) / center_x
            detection_state['line_offset'] = offset
            
            # Draw line detection
            cv2.drawContours(processed_frame, [line_info['contour']], -1, (0, 255, 0), 2)
            cv2.circle(processed_frame, line_info['center'], 5, (255, 0, 0), -1)
            cv2.line(processed_frame, (line_x, 0), (line_x, height), (255, 0, 255), 2)
            cv2.line(processed_frame, (center_x, 0), (center_x, height), (0, 255, 255), 1)
            
            detection_state['status'] = f"Line detected (offset: {offset:.2f})"
        else:
            detection_state['line_detected'] = False
            detection_state['confidence'] = 0.0
            detection_state['line_offset'] = 0.0
            detection_state['status'] = "No line detected"
    
    # Obstacle detection (objects blocking the black line)
    obstacles = []
    if detection_state['mode'] in ['obstacle', 'both'] and line_info:
        obstacles = processor.detect_obstacles(frame, line_info)
        detection_state['obstacles'] = obstacles
        detection_state['obstacle_detected'] = len(obstacles) > 0
        
        # Draw line path area
        if line_info:
            line_x = line_info['center'][0]
            path_left = max(0, line_x - LINE_PATH_WIDTH // 2)
            path_right = min(frame.shape[1], line_x + LINE_PATH_WIDTH // 2)
            cv2.rectangle(processed_frame, (path_left, 0), (path_right, frame.shape[0]), (255, 255, 0), 1)
        
        # Draw obstacle detection
        for obstacle in obstacles:
            x1, y1, x2, y2 = obstacle['bbox']
            
            # Color based on how much line is blocked (red = high blocking, orange = partial blocking)
            blocking_ratio = obstacle.get('blocking_ratio', 0)
            if blocking_ratio > 0.7:  # Blocks more than 70% of line
                color = (0, 0, 255)  # Red - critical
                thickness = 3
                status_text = "CRITICAL"
            elif blocking_ratio > 0.5:  # Blocks more than 50% of line
                color = (0, 100, 255)  # Orange-red - danger
                thickness = 3
                status_text = "DANGER"
            else:
                color = (0, 165, 255)  # Orange - warning
                thickness = 2
                status_text = "BLOCKING"
            
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(processed_frame, obstacle['center'], 5, color, -1)
            
            label = f"{status_text}: {blocking_ratio*100:.0f}% blocked"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(processed_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(processed_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if obstacles:
            # Find the most blocking obstacle
            most_blocking = max(obstacles, key=lambda x: x.get('blocking_ratio', 0))
            blocking_percentage = most_blocking.get('blocking_ratio', 0) * 100
            
            if blocking_percentage > 70:
                detection_state['status'] = f"CRITICAL: Line {blocking_percentage:.0f}% blocked!"
            elif blocking_percentage > 50:
                detection_state['status'] = f"DANGER: Line {blocking_percentage:.0f}% blocked!"
            else:
                detection_state['status'] = f"WARNING: Line {blocking_percentage:.0f}% blocked"
        elif detection_state['mode'] == 'obstacle':
            detection_state['status'] = "Line clear - no blocking obstacles"
    
    # Draw status information
    status_info = [
        f"Mode: {detection_state['mode'].upper()}",
        f"Status: {detection_state['status']}",
        f"FPS: {detection_state['fps']:.1f}"
    ]
    
    if detection_state['line_detected']:
        status_info.append(f"Line Confidence: {detection_state['confidence']:.2f}")
        status_info.append(f"Line Offset: {detection_state['line_offset']:.2f}")
    
    for i, text in enumerate(status_info):
        cv2.putText(processed_frame, text, (10, 30 + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return processed_frame

@app.route('/')
def index():
    return render_template_string('''
    <html>
    <head>
        <title>Phone Camera - Object Detection & Line Following</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                text-align: center; 
                margin: 20px; 
                background: #1a1a1a; 
                color: #fff; 
            }
            .container { max-width: 1000px; margin: 0 auto; }
            img { max-width: 90%; height: auto; border: 2px solid #333; margin: 20px 0; }
            h1 { color: #4CAF50; }
            .controls {
                margin: 20px 0;
                display: flex;
                justify-content: center;
                gap: 10px;
                flex-wrap: wrap;
            }
            button {
                padding: 10px 20px;
                font-size: 16px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                background: #4CAF50;
                color: white;
            }
            button:hover { background: #45a049; }
            button.active { background: #2196F3; }
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .stat-card {
                background: #2a2a2a;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }
            .stat-value {
                font-size: 1.5em;
                font-weight: bold;
                color: #4CAF50;
            }
            .stat-label {
                color: #888;
                margin-top: 5px;
                font-size: 0.9em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Phone Camera - Detection System</h1>
            
                         <div class="controls">
                 <button onclick="setMode('line')" id="btn-line">Line Following</button>
                 <button onclick="setMode('obstacle')" id="btn-obstacle">Obstacle Detection</button>
                 <button onclick="setMode('both')" id="btn-both">Both</button>
             </div>
            
            <img src="/video" alt="Camera Stream" id="video-feed">
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value" id="status">Initializing</div>
                    <div class="stat-label">Status</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="mode">LINE</div>
                    <div class="stat-label">Mode</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="fps">0.0</div>
                    <div class="stat-label">FPS</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="confidence">0%</div>
                    <div class="stat-label">Line Confidence</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="offset">0.00</div>
                    <div class="stat-label">Line Offset</div>
                </div>
                                 <div class="stat-card">
                     <div class="stat-value" id="obstacles">0</div>
                     <div class="stat-label">Obstacles Detected</div>
                 </div>
            </div>
            
            <p>Make sure your phone's IP Webcam app is running at {{ phone_ip }}:8080</p>
        </div>
        
        <script>
            function setMode(mode) {
                fetch('/set_mode/' + mode, {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        updateButtonStates(mode);
                    });
            }
            
                         function updateButtonStates(activeMode) {
                 ['line', 'obstacle', 'both'].forEach(mode => {
                     const btn = document.getElementById('btn-' + mode);
                     btn.className = mode === activeMode ? 'active' : '';
                 });
             }
            
            function updateStatus() {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('status').textContent = data.status;
                        document.getElementById('mode').textContent = data.mode.toUpperCase();
                        document.getElementById('fps').textContent = data.fps.toFixed(1);
                        document.getElementById('confidence').textContent = 
                            Math.round(data.confidence * 100) + '%';
                                                 document.getElementById('offset').textContent = data.offset.toFixed(2);
                         document.getElementById('obstacles').textContent = data.obstacles.length;
                        
                        updateButtonStates(data.mode);
                    })
                    .catch(error => console.log('Status update error:', error));
            }
            
            setInterval(updateStatus, 500);
            updateStatus();
        </script>
    </body>
    </html>
    ''', phone_ip=PHONE_IP)

@app.route('/video')
def video():
    def get_frames():
        # Connect to phone camera
        cap = cv2.VideoCapture(f"http://{PHONE_IP}:8080/video")
        
        if not cap.isOpened():
            print(f"Error: Could not connect to camera at {PHONE_IP}:8080")
            return
        
        print(f"Successfully connected to camera at {PHONE_IP}:8080")
        
        # FPS tracking
        fps_counter = 0
        fps_timer = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if ret:
                    # Resize frame for processing (already smaller for speed)
                    frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
                    
                    # Process frame
                    processed_frame = process_frame(frame)
                    
                    # Update frame buffer (non-blocking)
                    if frame_buffer['lock'].acquire(blocking=False):
                        try:
                            frame_buffer['frame'] = frame
                            frame_buffer['processed_frame'] = processed_frame
                        finally:
                            frame_buffer['lock'].release()
                    
                    # Update FPS
                    fps_counter += 1
                    if time.time() - fps_timer >= 1.0:
                        detection_state['fps'] = fps_counter
                        fps_counter = 0
                        fps_timer = time.time()
                    
                    # Convert frame to jpg with lower quality for speed
                    _, jpg = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
                else:
                    print("Failed to read frame from camera")
                    break
                    
        except Exception as e:
            print(f"Camera processing error: {e}")
        finally:
            cap.release()
    
    return Response(get_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode/<mode>', methods=['POST'])
def set_mode(mode):
    if mode in ['line', 'obstacle', 'both']:
        detection_state['mode'] = mode
        logger.info(f"Detection mode changed to: {mode}")
        return jsonify({'success': True, 'mode': mode})
    return jsonify({'success': False, 'error': 'Invalid mode'})

@app.route('/api/status')
def api_status():
    return jsonify({
        'status': detection_state['status'],
        'mode': detection_state['mode'],
        'line_detected': detection_state['line_detected'],
        'obstacle_detected': detection_state['obstacle_detected'],
        'offset': detection_state['line_offset'],
        'confidence': detection_state['confidence'],
        'fps': detection_state['fps'],
        'obstacles': detection_state['obstacles']
    })

if __name__ == '__main__':

    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\nShutting down camera detection system...") 