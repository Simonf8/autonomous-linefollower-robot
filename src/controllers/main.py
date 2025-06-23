#!/usr/bin/env python3

import time
import logging
import cv2
import numpy as np
import math
import io
import threading
from flask import Flask, jsonify, render_template, Response
from PIL import Image

# Import the new Robot class
from robot import Robot

# ================================
# CONFIGURATION
# ================================
CONFIG = {
    'FEATURES': {
        'OBJECT_DETECTION_ENABLED': False,
        'PATH_SHAPE_DETECTION_ENABLED': False,
        'OBSTACLE_AVOIDANCE_ENABLED': True,
        'VISION_SYSTEM_ENABLED': True,
        'INTERSECTION_CORRECTION_ENABLED': True,
        'DEBUG_VISUALIZATION_ENABLED': True,
        'ADAPTIVE_SPEED_ENABLED': True,
    },
    'ESP32_IP': "192.168.2.38",
    'CELL_SIZE_M': 0.11,
    'MAX_SPEED': 80,
    'BASE_SPEED': 75,
    'TURN_SPEED': 50,
    'CORNER_SPEED': 45,
    'LINE_FOLLOW_SPEED': 50,
    'ROBOT_WIDTH_M': 0.225,
    'ROBOT_LENGTH_M': 0.075,
    'CAMERA_FORWARD_OFFSET_M': 0.05,
    'START_CELL': (14, 14),
    'END_CELL': (2, 0),
    'START_HEADING': 0.0,
    'PHONE_IP': "192.168.2.6",
    'WEBCAM_INDEX': 0,
    'CAMERA_WIDTH': 416,
    'CAMERA_HEIGHT': 320,
}

def print_feature_status():
    """Print current feature configuration for debugging."""
    print("=" * 50)
    print("ROBOT FEATURE CONFIGURATION")
    print("=" * 50)
    for feature, enabled in CONFIG['FEATURES'].items():
        status = "ENABLED" if enabled else "DISABLED"
        print(f"{feature:<30} : {status}")
    print("=" * 50)
    print()

def main():
    """Main entry point for the robot controller."""
    print_feature_status()
    
    robot = Robot(CONFIG)
    app = Flask(__name__, template_folder='../../templates', static_folder='../../static')
    
    @app.route('/')
    def index():
        return render_template('navigation.html')

    @app.route('/api/robot_data')
    def robot_data():
        """Provide robot data to the web UI from the robot object."""
        line_result = robot.perception.latest_line_result
        movement_mode = 'UNKNOWN'
        
        if line_result:
            if line_result['line_detected']:
                offset = line_result['line_offset']
                abs_offset = abs(offset)
                is_intersection = line_result.get('intersection_detected', False)
                if is_intersection or abs_offset > 0.4: movement_mode = "TURNING"
                elif abs_offset > 0.15: movement_mode = "STRAFE+FWD" 
                elif abs_offset > 0.03: movement_mode = "SIDEWAYS"
                else: movement_mode = "STRAIGHT"
            else:
                movement_mode = "STOPPED"

        data = {
            'esp32_connected': robot.esp32.connected,
            'state': robot.state,
            'x': robot.pose[0],
            'y': robot.pose[1],
            'heading': math.degrees(robot.pose[2]),
            'path': robot.navigator.path,
            'smoothed_path': robot.navigator.smoothed_path.tolist() if robot.navigator.smoothed_path is not None else [],
            'current_target_index': robot.navigator.current_target_index,
            'movement_mode': movement_mode
        }
        return jsonify(data)

    @app.route('/camera_debug_feed')
    def camera_debug_feed():
        if not CONFIG['FEATURES']['VISION_SYSTEM_ENABLED']:
            return Response(status=204)

        def generate_debug_frames():
            while robot.running:
                time.sleep(1.0 / 15) # 15 FPS
                with robot.frame_lock:
                    if robot.perception.debug_frame is None:
                        continue
                    frame = robot.perception.debug_frame.copy()
                
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        return Response(generate_debug_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @app.route('/start_mission')
    def start_mission():
        robot.start_mission()
        return jsonify({'status': 'Mission started', 'robot_state': robot.state})

    @app.route('/grid_feed')
    def grid_feed():
        def generate():
            while robot.running:
                robot_cell = (
                    int(robot.pose[0] / CONFIG['CELL_SIZE_M']),
                    int(robot.pose[1] / CONFIG['CELL_SIZE_M'])
                )
                grid_array = generate_grid_image(
                    robot.navigator.pathfinder, 
                    robot_cell, 
                    robot.navigator.path, 
                    CONFIG['START_CELL'], 
                    CONFIG['END_CELL'],
                    robot.navigator.smoothed_path
                )
                img_io = io.BytesIO()
                Image.fromarray(cv2.cvtColor(grid_array, cv2.COLOR_BGR2RGB)).save(img_io, 'PNG')
                img_io.seek(0)
                yield (b'--frame\r\n'
                       b'Content-Type: image/png\r\n\r\n' + img_io.read() + b'\r\n')
                time.sleep(0.1)
        
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def generate_grid_image(pathfinder, robot_cell, path, start_cell, end_cell, smoothed_path):
        grid = np.array(pathfinder.get_grid())
        cell_size = 20
        h, w = grid.shape
        img = np.zeros((h * cell_size, w * cell_size, 3), dtype=np.uint8)

        for r in range(h):
            for c in range(w):
                color = (255, 255, 255) if grid[r, c] == 1 else (0, 0, 0)
                cv2.rectangle(img, (c*cell_size, r*cell_size), ((c+1)*cell_size, (r+1)*cell_size), color, -1)
        
        if path:
            for i in range(len(path) - 1):
                p1 = (path[i][0]*cell_size + cell_size//2, path[i][1]*cell_size + cell_size//2)
                p2 = (path[i+1][0]*cell_size + cell_size//2, path[i+1][1]*cell_size + cell_size//2)
                cv2.line(img, p1, p2, (128, 0, 128), 2)
        
        if smoothed_path is not None:
            path_pixels = (smoothed_path / CONFIG['CELL_SIZE_M'] * cell_size).astype(int)
            cv2.polylines(img, [path_pixels], isClosed=False, color=(255, 100, 0), thickness=1)

        cv2.rectangle(img, (start_cell[0]*cell_size, start_cell[1]*cell_size), ((start_cell[0]+1)*cell_size, (start_cell[1]+1)*cell_size), (0,255,0), -1)
        cv2.rectangle(img, (end_cell[0]*cell_size, end_cell[1]*cell_size), ((end_cell[0]+1)*cell_size, (end_cell[1]+1)*cell_size), (0,0,255), -1)
        if robot_cell:
            cv2.circle(img, (robot_cell[0]*cell_size + cell_size//2, robot_cell[1]*cell_size + cell_size//2), cell_size//3, (255,165,0), -1)
        
        return img
    
    def camera_capture_thread(robot_instance):
        cap = None
        sources = [
            f"http://{CONFIG['PHONE_IP']}:8080/video",
            CONFIG['WEBCAM_INDEX']
        ]
        for source in sources:
            try:
                cap = cv2.VideoCapture(source)
                if cap.isOpened():
                    print(f"✓ Camera connected successfully at {source}")
                    break
            except Exception:
                continue
        
        if cap is None or not cap.isOpened():
            print("ERROR: No camera source available. Vision system disabled.")
            robot_instance.config['FEATURES']['VISION_SYSTEM_ENABLED'] = False
            return

        while robot_instance.running:
            ret, frame = cap.read()
            if ret and frame is not None:
                resized_frame = cv2.resize(frame, (CONFIG['CAMERA_WIDTH'], CONFIG['CAMERA_HEIGHT']))
                with robot_instance.frame_lock:
                    robot_instance.frame = resized_frame
            else:
                time.sleep(0.5)
        cap.release()

    robot_thread = threading.Thread(target=robot.run_main_loop, daemon=True)
    robot_thread.start()
    
    if CONFIG['FEATURES']['VISION_SYSTEM_ENABLED']:
        camera_thread = threading.Thread(target=camera_capture_thread, args=(robot,), daemon=True)
        camera_thread.start()
    
    print("Starting Flask web server...")
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting program.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logging.exception("Error details:") 