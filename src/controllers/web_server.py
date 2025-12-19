#!/usr/bin/env python3

from flask import Flask, jsonify, render_template, request, Response
import cv2
import math
from config import *

def create_app(robot):
    app = Flask(__name__, template_folder='../../templates', static_folder='../../static')
    
    @app.route('/')
    def index():
        return render_template('navigation.html')

    @app.route('/api/robot_data')
    def robot_data():
        """Provide robot data to the web UI."""
        x, y, heading_rad = robot.position_tracker.get_pose()
        heading_deg = math.degrees(heading_rad)
        
        status = robot.position_tracker.get_status()
        encoder_counts = robot.motor_controller.get_encoder_counts() if robot.motor_controller else {}
        audio_status = robot.audio_feedback.get_status()

        box_states = {}
        if robot.box_handler:
            for b_id, b_info in robot.box_handler.box_states.items():
                ic = b_info.copy()
                ic['state'] = ic['state'].value
                box_states[b_id] = ic

        data = {
            'state': robot.state,
            'x': x,
            'y': y,
            'heading': heading_deg,
            'position_tracker': {
                'status': status.get('status', 'N/A'),
                'confidence': status.get('confidence', 0),
                'position': status.get('current_position', (0,0)),
                'direction': status.get('current_direction', 'N/A'),
                'message': status.get('message', ''),
            },
            'motors': robot.motor_speeds,
            'electromagnet_on': robot.motor_controller.get_electromagnet_status() if robot.motor_controller else False,
            'encoders': encoder_counts,
            'audio_feedback': audio_status,
            'path_info': {
                'path': robot.path,
                'current_target_index': robot.current_target_index,
                'total_corners': robot.total_corners_in_path,
                'corners_passed': robot.corners_passed
            },
            'box_mission': box_states
        }
        return jsonify(data)

    @app.route('/video_feed')
    def video_feed():
        def generate():
            while True:
                frame = robot.camera_line_follower.get_camera_frame()
                if frame is not None:
                    # Draw position if available
                    curr_pos = robot.position_tracker.get_current_cell()
                    cv2.putText(frame, f"Pos: {curr_pos} {robot.position_tracker.current_direction}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/api/control', methods=['POST'])
    def control():
        action = request.json.get('action')
        if action == 'start':
            robot._start_mission()
        elif action == 'stop':
            robot.state = "idle"
            robot._stop_motors()
        return jsonify({'status': 'ok'})

    return app
