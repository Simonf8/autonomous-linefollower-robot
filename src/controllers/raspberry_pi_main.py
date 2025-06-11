#!/usr/bin/env python3

import cv2
import numpy as np
import socket
import time
import logging
from ultralytics import YOLO
from flask import Flask, render_template_string, Response
from flask_socketio import SocketIO
import threading
import base64
import os
import pygame
import requests
import json
from pathlib import Path

# HTML template for web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Line Follower Robot Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="/static/dashboard.css">
</head>
<body>
    <div class="container">
        <div class="video-feed">
            <h2>Camera Feed</h2>
            <img id="camera-feed" src="" alt="Robot Camera Feed">
            <canvas id="path-canvas" class="path-canvas" width="600" height="400"></canvas>
        </div>
        
        <div class="status-panel">
            <h2>Robot Status</h2>
            
            <div id="line-status" class="status-item">
                <div class="status-indicator inactive"></div>
                Line Status: Not Detected
            </div>
            
            <div class="status-item">
                <h3>Position</h3>
                <div class="position-grid">
                    <div class="position-item">
                        <div class="position-label">X</div>
                        <div class="position-value" id="position-x">0.00</div>
                    </div>
                    <div class="position-item">
                        <div class="position-label">Y</div>
                        <div class="position-value" id="position-y">0.00</div>
                    </div>
                    <div class="position-item">
                        <div class="position-label">Angle</div>
                        <div class="position-value" id="position-theta">0.00</div>
                    </div>
                </div>
            </div>
            
            <div id="command" class="status-item">
                Command: None
            </div>
            
            <div class="status-item">
                Objects Detected: <span id="objects-count">0</span>
            </div>
            
            <div class="status-item">
                Uptime: <span id="uptime">00:00:00</span>
            </div>
            
            <div class="status-item">
                <h3>Performance Metrics</h3>
                <canvas id="metrics-chart" width="300" height="200"></canvas>
            </div>
            
            <div class="controls-info">
                <h3>Manual Controls</h3>
                <p>W - Forward</p>
                <p>A - Left</p>
                <p>D - Right</p>
                <p>S - Stop</p>
                <p>Q - Turn Around</p>
            </div>
        </div>
    </div>
    <script src="/static/dashboard.js"></script>
</body>
</html>
'''

class ESP32Interface:
    """Handles communication with ESP32"""
    
    VALID_COMMANDS = [
        'FORWARD', 'LEFT', 'RIGHT', 'STOP',
        'EMERGENCY_LEFT', 'EMERGENCY_RIGHT'
    ]
    
    def __init__(self, ip_address, port=1234):
        self.ip_address = ip_address
        self.port = port
        self.socket = None
        self.connected = False
        self.logger = logging.getLogger(__name__)
        self.line_position = 0
        self.line_detected = False
    
    def connect(self):
        """Establish connection to ESP32"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.ip_address, self.port))
            self.connected = True
            self.logger.info(f"Connected to ESP32 at {self.ip_address}:{self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to ESP32: {e}")
            self.connected = False
            return False
    
    def reconnect(self):
        """Attempt to reconnect to ESP32"""
        self.logger.info("Attempting to reconnect to ESP32...")
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.socket = None
        return self.connect()
    
    def send_command(self, command):
        """Send command to ESP32 and receive line sensor data"""
        if not command in self.VALID_COMMANDS:
            self.logger.error(f"Invalid command: {command}")
            return False
        
        if not self.connected:
            if not self.reconnect():
                return False
        
        try:
            # Send command
            self.socket.send(command.encode('utf-8'))
            
            # Receive line sensor data
            data = self.socket.recv(64).decode('utf-8').strip()
            if data:
                try:
                    position, detected = map(float, data.split(','))
                    self.line_position = position
                    self.line_detected = bool(detected)
                except:
                    self.logger.error("Failed to parse line sensor data")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to communicate with ESP32: {e}")
            self.connected = False
            return False
    
    def close(self):
        """Close connection to ESP32"""
        if self.socket:
            try:
                self.send_command('STOP')
                self.socket.close()
            except:
                pass
        self.socket = None
        self.connected = False

class VisualOdometry:
    """Handles visual odometry for position tracking"""
    
    def __init__(self, camera_matrix=None):
        if camera_matrix is None:
            self.camera_matrix = np.array([
                [500, 0, 320],
                [0, 500, 240],
                [0, 0, 1]
            ])
        else:
            self.camera_matrix = camera_matrix
            
        # Initialize feature detector
        self.orb = cv2.ORB_create(
            nfeatures=1000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            patchSize=31,
            fastThreshold=20
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # State variables
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None
        self.position = np.zeros(3)  # [x, y, theta]
        self.tracking_quality = 1.0
    
    def process_frame(self, frame):
        """Process a new frame and update position estimate"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features
        kp, des = self.orb.detectAndCompute(gray, None)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            return self.position, 1.0
        
        # Match features
        if des is not None and self.prev_des is not None:
            matches = self.bf.match(self.prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)
            
            if len(matches) > 8:
                # Get matched keypoints
                pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp[m.trainIdx].pt for m in matches])
                
                # Calculate Essential matrix
                E, mask = cv2.findEssentialMat(
                    pts1, pts2, self.camera_matrix,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=1.0
                )
                
                if E is not None:
                    # Recover pose
                    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix)
                    
                    # Update position
                    scale = 1.0  # This should be calibrated
                    self.position[0] += scale * t[0]
                    self.position[1] += scale * t[1]
                    self.position[2] = np.arctan2(R[1, 0], R[0, 0])
                    
                    # Update tracking quality
                    self.tracking_quality = min(len(matches) / 500.0, 1.0)
        
        # Update previous frame data
        self.prev_frame = gray
        self.prev_kp = kp
        self.prev_des = des
        
        return self.position, self.tracking_quality
    
    def draw_debug(self, frame):
        """Draw debug visualization"""
        if self.prev_kp is not None:
            cv2.drawKeypoints(
                frame,
                self.prev_kp,
                frame,
                color=(0, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
        
        # Draw position and heading
        height, width = frame.shape[:2]
        center = (width // 2, height // 2)
        
        # Draw robot position
        cv2.circle(frame, center, 5, (0, 255, 0), -1)
        
        # Draw heading arrow
        arrow_length = 50
        end_point = (
            int(center[0] + arrow_length * np.cos(self.position[2])),
            int(center[1] + arrow_length * np.sin(self.position[2]))
        )
        cv2.arrowedLine(frame, center, end_point, (0, 255, 0), 2)

class VoiceSystem:
    """High-quality voice system using Bark TTS"""
    
    def __init__(self):
        # Initialize pygame mixer for audio playback
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        self.cache_dir = Path("voice_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Import Bark
        from bark import SAMPLE_RATE, generate_audio, preload_models
        from scipy.io.wavfile import write as write_wav
        
        self.SAMPLE_RATE = SAMPLE_RATE
        self.generate_audio = generate_audio
        self.write_wav = write_wav
        
        # Preload models for faster generation
        print("Loading Bark TTS models...")
        preload_models()
        print("Bark TTS ready!")
        
        # Voice presets with different speakers and emotions
        self.voices = {
            'obstacle': {
                'text': "Obstacle detected. Initiating turn around maneuver.",
                'voice_preset': "v2/en_speaker_6"  # Confident male voice
            },
            'line_lost': {
                'text': "Line signal lost. Searching for path.",
                'voice_preset': "v2/en_speaker_3"  # Concerned voice
            },
            'line_found': {
                'text': "Line detected. Resuming navigation.",
                'voice_preset': "v2/en_speaker_9"  # Excited voice
            },
            'startup': {
                'text': "Line follower robot system activated. Ready for operation.",
                'voice_preset': "v2/en_speaker_6"  # Dramatic voice
            },
            'turn_complete': {
                'text': "Turn around maneuver complete. Searching for line.",
                'voice_preset': "v2/en_speaker_5"  # Satisfied voice
            },
            'object_detected': {
                'text': "Object detected in path. Executing avoidance protocol.",
                'voice_preset': "v2/en_speaker_8"  # Alert voice
            }
        }
        
        # Cache for storing generated audio files
        self.audio_cache = {}
        self.is_playing = False
        
    def _get_cached_audio(self, text, voice_preset):
        """Get cached audio file or generate new one"""
        cache_key = f"{text}_{voice_preset}"
        cache_file = self.cache_dir / f"{abs(hash(cache_key))}.wav"
        
        if cache_file.exists():
            return str(cache_file)
            
        # Generate audio using Bark
        try:
            print(f"Generating voice: {text[:50]}...")
            
            # Generate audio with Bark
            audio_array = self.generate_audio(text, history_prompt=voice_preset)
            
            # Save to file
            self.write_wav(str(cache_file), self.SAMPLE_RATE, audio_array)
            print(f"Voice generated and cached: {cache_file.name}")
            return str(cache_file)
                
        except Exception as e:
            logging.error(f"Failed to generate audio with Bark: {e}")
            return None
    
    def play_sound(self, event):
        """Play a sound for a specific event"""
        if self.is_playing:
            return  # Don't interrupt current playback
            
        if event in self.voices:
            voice_data = self.voices[event]
            audio_file = self._get_cached_audio(voice_data['text'], voice_data['voice_preset'])
            
            if audio_file:
                try:
                    self.is_playing = True
                    pygame.mixer.music.load(audio_file)
                    pygame.mixer.music.play()
                    
                    # Start a thread to reset playing status when done
                    def reset_playing():
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                        self.is_playing = False
                    
                    threading.Thread(target=reset_playing, daemon=True).start()
                    
                except Exception as e:
                    logging.error(f"Failed to play audio: {e}")
                    self.is_playing = False
    
    def add_custom_voice(self, event_name, text, voice_preset="v2/en_speaker_6"):
        """Add a custom voice line"""
        self.voices[event_name] = {
            'text': text,
            'voice_preset': voice_preset
        }

class Robot:
    """Main robot control class"""
    
    def __init__(self, esp32_ip, esp32_port=1234):
        # Initialize Flask app and SocketIO
        self.app = Flask(__name__, static_folder='../static')
        self.socketio = SocketIO(self.app)
        self.app.route('/')(self.index)
        
        # Add manual command handler
        @self.socketio.on('manual_command')
        def handle_manual_command(command):
            if command in self.esp32.VALID_COMMANDS:
                self.send_command(command)
                logging.info(f"Manual command executed: {command}")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize components
        self.esp32 = ESP32Interface(esp32_ip, esp32_port)
        self.vo = VisualOdometry()
        self.yolo = YOLO("yolov8n.pt")
        self.voice = VoiceSystem()
        
        # State variables
        self.position = np.zeros(3)
        self.tracking_quality = 1.0
        self.detected_objects = []
        self.last_command = None
        self.last_command_time = time.time()
        self.line_status = False  # Track if line was previously detected
        
        # Path tracking
        self.path_history = []
        
        # Start web server thread
        self.web_thread = threading.Thread(target=self._run_web_server)
        self.web_thread.daemon = True
        self.web_thread.start()
        
        # Play startup sound
        self.voice.play_sound('startup')
    
    def index(self):
        """Serve the web interface"""
        return render_template_string(HTML_TEMPLATE)
    
    def _run_web_server(self):
        """Run the web server in a separate thread"""
        self.socketio.run(self.app, host='0.0.0.0', port=5000)
    
    def _encode_frame(self, frame):
        """Convert OpenCV frame to base64 for web display"""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    
    def _emit_update(self, frame):
        """Emit updates to web clients"""
        try:
            frame_data = self._encode_frame(frame)
            self.socketio.emit('robot_update', {
                'frame': frame_data,
                'data': {
                    'position': self.position.tolist(),
                    'line_position': self.esp32.line_position,
                    'line_detected': self.esp32.line_detected,
                    'detected_objects': self.detected_objects,
                    'last_command': self.last_command,
                    'path_history': self.path_history
                }
            })
        except Exception as e:
            logging.error(f"Failed to emit update: {e}")
    
    def process_frame(self, frame):
        """Process a single camera frame"""
        # 1. Update position using visual odometry
        self.position, self.tracking_quality = self.vo.process_frame(frame)
        
        if self.tracking_quality > 0.5:
            self.path_history.append((self.position[0], self.position[1]))
            if len(self.path_history) > 100:
                self.path_history.pop(0)
        
        # 2. Detect objects using YOLO
        results = self.yolo(frame)
        self.detected_objects = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = box.cls[0]
                
                if conf > 0.5:
                    self.detected_objects.append({
                        'class': int(cls),
                        'confidence': float(conf),
                        'box': (int(x1), int(y1), int(x2), int(y2))
                    })
        
        # 3. Draw visualizations
        self.draw_visualizations(frame)
        
        # 4. Emit update to web clients
        self._emit_update(frame)
        
        return frame
    
    def draw_visualizations(self, frame):
        """Draw debug visualizations"""
        # Draw visual odometry debug info
        self.vo.draw_debug(frame)
        
        # Draw path history
        if len(self.path_history) > 1:
            points = np.array(self.path_history, dtype=np.int32)
            cv2.polylines(frame, [points], False, (0, 255, 255), 2)
        
        # Draw detected objects
        for obj in self.detected_objects:
            x1, y1, x2, y2 = obj['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Class {obj['class']}: {obj['confidence']:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        # Draw line following status
        cv2.putText(
            frame,
            f"Line: {'Detected' if self.esp32.line_detected else 'Lost'} "
            f"Position: {self.esp32.line_position:.2f}",
            (10, frame.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
        
        # Draw current command
        if self.last_command:
            cv2.putText(
                frame,
                f"Command: {self.last_command}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
    
    def calculate_control(self):
        """Calculate control command based on sensor data"""
        # Check for obstacles
        for obj in self.detected_objects:
            x1, y1, x2, y2 = obj['box']
            # If obstacle is in the path
            if (x2 - x1) > 100 and y2 > 240:  # Large and close object
                self.voice.play_sound('object_detected')
                return "TURN_AROUND"
        
        # Line following with voice feedback
        if self.esp32.line_detected:
            if not self.line_status:  # Line was just found
                self.voice.play_sound('line_found')
                self.line_status = True
                
            if abs(self.esp32.line_position) < 0.1:
                return "FORWARD"
            elif self.esp32.line_position > 0:
                return "RIGHT"
            else:
                return "LEFT"
        else:
            if self.line_status:  # Line was just lost
                self.voice.play_sound('line_lost')
                self.line_status = False
            return "STOP"
    
    def send_command(self, command):
        """Send command to ESP32 with rate limiting"""
        current_time = time.time()
        
        # Only send new commands if different from last command
        # or if more than 100ms has passed
        if (command != self.last_command or 
            current_time - self.last_command_time > 0.1):
            
            if self.esp32.send_command(command):
                # Add voice feedback for turn around completion
                if command == "TURN_AROUND" and self.last_command != "TURN_AROUND":
                    # Play turn complete sound after a delay
                    def delayed_voice():
                        time.sleep(3)  # Wait for turn to complete
                        self.voice.play_sound('turn_complete')
                    threading.Thread(target=delayed_voice, daemon=True).start()
                
                self.last_command = command
                self.last_command_time = current_time
                logging.debug(f"Sent command: {command}")
            else:
                logging.error("Failed to send command to ESP32")
    
    def run(self):
        """Main control loop"""
        try:
            logging.info("Starting robot control loop")
            while True:
                # 1. Get camera frame
                ret, frame = self.cap.read()
                if not ret:
                    logging.error("Failed to get camera frame")
                    continue
                
                # 2. Process frame
                processed_frame = self.process_frame(frame)
                
                # 3. Calculate and send control command
                command = self.calculate_control()
                self.send_command(command)
                
                # 4. Display frame
                cv2.imshow('Robot View', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            logging.info("Shutting down...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        logging.info("Cleaning up...")
        self.send_command("STOP")
        self.cap.release()
        cv2.destroyAllWindows()
        self.esp32.close()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create and run robot
    # Replace with your ESP32's IP address
    robot = Robot(esp32_ip="192.168.2.21")
    robot.run() 