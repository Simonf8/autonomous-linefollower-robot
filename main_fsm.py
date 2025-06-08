#!/usr/bin/env python3
"""
Integrated Line Following + FSM Obstacle Avoidance + ESP32 Communication + Flask Dashboard
"""
import cv2
import numpy as np
import socket
import time
import threading
from collections import deque
from flask import Flask, Response, jsonify, render_template_string

from avoidance_fsm import ObstacleAvoidanceFSM
from avoidance_fsm_config import (
    FSM_TURN_AWAY_DURATION_S,
    FSM_PASS_OBSTACLE_DURATION_S,
    FSM_RETURN_TO_LINE_DURATION_S
)
from main import (
    CAMERA_FPS,
    ESP32_IP,
    ESP32_PORT,
    BLACK_THRESHOLD,
    BLUR_SIZE,
    MIN_CONTOUR_AREA,
    ZONE_BOTTOM_HEIGHT,
    ZONE_TOP_HEIGHT,
    OBJECT_WIDTH_THRESHOLD,
    OBJECT_HEIGHT_THRESHOLD,
    COMMANDS,
    KP, KI, KD,
    MAX_INTEGRAL,
    STEERING_DEADZONE,
    opposite_direction
)

# ESP32 connection
class ESP32Connection:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = None
        self.connect()
    def connect(self):
        try:
            self.sock = socket.create_connection((self.ip, self.port), timeout=2)
        except Exception:
            self.sock = None
    def send_command(self, cmd):
        if not self.sock:
            self.connect()
        if not self.sock:
            return False
        try:
            self.sock.sendall((cmd + "\n").encode())
            return True
        except Exception:
            self.sock = None
            return False
    def close(self):
        if self.sock:
            try: self.sock.close()
            except: pass
            self.sock = None

# Simple PID controller
class PID:
    def __init__(self, kp, ki, kd, max_i=1.0):
        self.kp=kp; self.ki=ki; self.kd=kd; self.max_i=max_i
        self.i=0; self.prev=0; self.last_time=time.time()
    def calculate(self, error):
        now=time.time(); dt=now - self.last_time; self.last_time=now
        self.i += error * dt; self.i = np.clip(self.i, -self.max_i, self.max_i)
        d = (error - self.prev)/dt if dt>0 else 0
        self.prev = error
        return np.clip(self.kp*error + self.ki*self.i + self.kd*d, -1.0, 1.0)

# Frame globals
latest_frame = None
frame_lock = threading.Lock()
last_command = COMMANDS['STOP']

# Initialize modules
fsm = ObstacleAvoidanceFSM()
pid = PID(KP, KI, KD, max_i=1.0)
esp = ESP32Connection(ESP32_IP, ESP32_PORT)

# Detection functions
height_roi = None

def detect_line(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)
    _, binary = cv2.threshold(blur, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    h, w = binary.shape
    y0 = int(h * (1 - ZONE_BOTTOM_HEIGHT))
    roi = binary[y0:h, :]
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0
    cnt = max(contours, key=cv2.contourArea)
    M = cv2.moments(cnt)
    if M['m00']==0:
        return None, 0.0
    cx = int(M['m10']/M['m00'])
    conf = min(cv2.contourArea(cnt)/(w*(h*ZONE_BOTTOM_HEIGHT)*0.1), 1.0)
    return cx, conf

def detect_obstacle(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)
    _, binary = cv2.threshold(blur, BLACK_THRESHOLD+40, 255, cv2.THRESH_BINARY)
    h, w = binary.shape
    y1 = int(h * (1 - ZONE_BOTTOM_HEIGHT - ZONE_TOP_HEIGHT))
    y2 = int(h * (1 - ZONE_BOTTOM_HEIGHT))
    roi = binary[y1:y2, :]
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < MIN_CONTOUR_AREA:
        return None
    x, y, wi, hi = cv2.boundingRect(cnt)
    w_ratio = wi / w
    h_ratio = hi / (y2 - y1)
    if w_ratio < OBJECT_WIDTH_THRESHOLD or h_ratio < OBJECT_HEIGHT_THRESHOLD:
        return None
    cx = x + wi/2
    pos = (cx - w/2)/(w/2)
    return pos

# Flask app
app = Flask(__name__)

INDEX_HTML = """<!doctype html><html><body>
<h1>Robot Dashboard</h1>
<img src='/video_feed' width='320'/>
<pre id='status'></pre>
<script>
 function update(){fetch('/api/status').then(r=>r.json()).then(d=>{document.getElementById('status').textContent=JSON.stringify(d,null,2);});}
 setInterval(update, 500);
 update();
</script>
</body></html>"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

def gen_frames():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            ret, buf = cv2.imencode('.jpg', latest_frame)
        if not ret:
            continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    return jsonify({
        'fsm_state': fsm.state,
        'last_command': last_command
    })

# Main loop
def main():
    global latest_frame, last_command
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Camera error')
        return
    # Start Flask in thread
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False), daemon=True).start()
    print('Dashboard at http://localhost:5000')
    # Continuous processing
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        cx, conf = detect_line(frame)
        obs_pos = detect_obstacle(frame)
        # FSM start
        if obs_pos is not None and fsm.state == 'idle':
            fsm.start(obs_pos)
        # FSM update
        cmd = None
        if fsm.state != 'idle':
            cmd = fsm.update(obs_pos is not None, cx is not None, (cx - frame.shape[1]/2)/(frame.shape[1]/2) if cx else 0.0)
        # Normal line following
        if cmd is None:
            if cx is not None and conf > 0.2:
                offset = (cx - frame.shape[1]/2)/(frame.shape[1]/2)
                steer = pid.calculate(-offset)
                if abs(steer) < STEERING_DEADZONE:
                    cmd = COMMANDS['FORWARD']
                else:
                    cmd = COMMANDS['LEFT'] if steer > 0 else COMMANDS['RIGHT']
            else:
                cmd = COMMANDS['STOP']
        # Send to ESP32
        esp.send_command(cmd)
        last_command = cmd
        # Update frame for web
        with frame_lock:
            latest_frame = frame.copy()
        time.sleep(1.0/CAMERA_FPS)

if __name__ == '__main__':
    main() 