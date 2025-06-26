from flask import Flask, render_template_string, jsonify, Response, request
import cv2
import numpy as np
import threading
import time
from typing import Tuple, Optional, Dict, List
from collections import defaultdict
import json

app = Flask(__name__)

class PreciseMazeLocalizer:
    def __init__(self, start_pos: Tuple[int, int], start_direction: str):
        # Your exact maze
        self.maze = [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0], # Row 0
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0], # Row 1
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Row 2
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 3
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 4
            [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0], # Row 5
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 6
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Row 7
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 8
            [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0], # Row 9
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 10
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0], # Row 11
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Row 12
            [0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1], # Row 13
            [0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1]  # Row 14
        ]
        
        self.current_pos = start_pos
        self.current_direction = start_direction
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Warning: Could not open camera")
            self.cap = None
        
        # Movement detection
        self.prev_frame = None
        self.movement_threshold = 50
        self.stationary_frames = 0
        self.max_stationary_frames = 10
        
        # Corner distance detection
        self.close_corner_threshold = 0.7
        self.far_corner_threshold = 0.3
        
        # Create precise signatures for your maze
        self.corner_signatures = self.create_precise_signatures()
        
        # Confidence tracking
        self.position_confidence = 1.0
        self.min_confidence = 0.6
        
        # Status tracking
        self.last_status = {
            'status': 'initialized',
            'position': self.current_pos,
            'direction': self.current_direction,
            'confidence': self.position_confidence,
            'scene_type': 'unknown',
            'is_moving': False,
            'message': 'System initialized'
        }
        
        # Threading
        self.running = False
        self.localization_thread = None
        
    def create_precise_signatures(self) -> Dict:
        """Create precise corner signatures for each valid position in YOUR maze"""
        signatures = {}
        
        for y in range(len(self.maze)):
            for x in range(len(self.maze[0])):
                if self.maze[y][x] == 0:  # Valid path
                    for direction in ['N', 'S', 'E', 'W']:
                        sig = self.analyze_maze_position(x, y, direction)
                        signatures[(x, y, direction)] = sig
        
        return signatures
    
    def analyze_maze_position(self, x: int, y: int, direction: str) -> Dict:
        """Analyze what should be visible from position (x,y) facing direction"""
        signature = {
            'scene_type': 'unknown',
            'corner_ahead_left': False,
            'corner_ahead_right': False,
            'wall_ahead': False,
            'opening_left': False,
            'opening_right': False,
            'intersection_type': 'none',
            'unique_features': []
        }
        
        # Get surrounding positions based on direction
        if direction == 'N':
            ahead = (x, y-1)
            left = (x-1, y)
            right = (x+1, y)
        elif direction == 'S':
            ahead = (x, y+1)
            left = (x+1, y)
            right = (x-1, y)
        elif direction == 'E':
            ahead = (x+1, y)
            left = (x, y-1)
            right = (x, y+1)
        elif direction == 'W':
            ahead = (x-1, y)
            left = (x, y+1)
            right = (x, y-1)
        
        # Check each position
        wall_ahead = self.is_wall_at(ahead)
        opening_left = not self.is_wall_at(left)
        opening_right = not self.is_wall_at(right)
        
        signature['wall_ahead'] = wall_ahead
        signature['opening_left'] = opening_left
        signature['opening_right'] = opening_right
        
        # Determine scene type
        if not wall_ahead and opening_left and opening_right:
            signature['scene_type'] = 'intersection'
            signature['intersection_type'] = '4way'
        elif not wall_ahead and (opening_left or opening_right):
            signature['scene_type'] = 'T_junction'
            signature['intersection_type'] = 'T'
        elif wall_ahead and (opening_left or opening_right):
            signature['scene_type'] = 'L_turn'
            signature['intersection_type'] = 'L'
        elif wall_ahead and not opening_left and not opening_right:
            signature['scene_type'] = 'dead_end'
        else:
            signature['scene_type'] = 'corridor'
            
        return signature
    
    def is_wall_at(self, pos: Tuple[int, int]) -> bool:
        """Check if position is a wall or out of bounds"""
        x, y = pos
        if x < 0 or x >= len(self.maze[0]) or y < 0 or y >= len(self.maze):
            return True
        return self.maze[y][x] == 1
    
    def detect_movement(self, current_frame: np.ndarray) -> bool:
        """Detect if robot is moving using frame difference"""
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return False
        
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        diff = cv2.absdiff(self.prev_frame, current_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Count changed pixels
        movement_pixels = cv2.countNonZero(thresh)
        is_moving = movement_pixels > self.movement_threshold
        
        if not is_moving:
            self.stationary_frames += 1
        else:
            self.stationary_frames = 0
            
        self.prev_frame = current_gray
        return is_moving
    
    def detect_scene_with_precision(self) -> Optional[Dict]:
        """Precisely detect current scene with movement and distance awareness"""
        if not self.cap or not self.cap.isOpened():
            return {
                'status': 'error',
                'confidence': 0.0,
                'message': 'Camera not available'
            }
        
        ret, frame = self.cap.read()
        if not ret:
            return {
                'status': 'error',
                'confidence': 0.0,
                'message': 'Failed to read camera frame'
            }
        
        # Check if robot is moving
        is_moving = self.detect_movement(frame)
        
        # If stationary for too long, don't update position
        if self.stationary_frames > self.max_stationary_frames:
            return {
                'status': 'stationary',
                'confidence': 0.0,
                'is_moving': False,
                'message': 'Robot stationary - no position update'
            }
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhanced corner detection
        enhanced = cv2.equalizeHist(gray)
        corners = cv2.cornerHarris(enhanced, blockSize=3, ksize=5, k=0.04)
        corners = cv2.dilate(corners, None)
        
        # Get corner coordinates
        corner_threshold = 0.01 * corners.max() if corners.max() > 0 else 0
        corner_coords = np.where(corners > corner_threshold)
        corner_points = list(zip(corner_coords[1], corner_coords[0]))
        
        # Analyze corner distances
        distance_analysis = self.analyze_corner_distance(corner_points, frame.shape[0])
        
        # Determine scene type based on corner patterns
        scene = self.classify_scene_precisely(corner_points, frame.shape, distance_analysis)
        scene['is_moving'] = is_moving
        scene['distance_analysis'] = distance_analysis
        
        return scene
    
    def analyze_corner_distance(self, corner_points: List[Tuple[int, int]], frame_height: int) -> Dict:
        """Analyze if corners are close (immediate) or far (ahead)"""
        if not corner_points:
            return {'close_corners': 0, 'far_corners': 0, 'corner_distance': 'none'}
        
        close_threshold = frame_height * self.close_corner_threshold
        far_threshold = frame_height * self.far_corner_threshold
        
        close_corners = sum(1 for _, y in corner_points if y > close_threshold)
        far_corners = sum(1 for _, y in corner_points if y < far_threshold)
        
        if close_corners > far_corners:
            distance = 'close'
        elif far_corners > close_corners:
            distance = 'far'
        else:
            distance = 'medium'
            
        return {
            'close_corners': close_corners,
            'far_corners': far_corners,
            'corner_distance': distance
        }
    
    def classify_scene_precisely(self, corner_points: List[Tuple[int, int]], 
                                frame_shape: Tuple[int, int], 
                                distance_analysis: Dict) -> Dict:
        """Precisely classify the scene based on corner distribution"""
        height, width = frame_shape[:2]
        
        scene = {
            'scene_type': 'unknown',
            'corner_ahead_left': False,
            'corner_ahead_right': False,
            'wall_ahead': False,
            'opening_left': False,
            'opening_right': False,
            'intersection_type': 'none',
            'confidence': 0.0
        }
        
        if not corner_points:
            scene['scene_type'] = 'corridor'
            scene['confidence'] = 0.8
            return scene
        
        # Divide frame into precise regions
        left_third = width // 3
        right_third = 2 * width // 3
        top_half = height // 2
        
        # Count corners in each region
        corners_left = [(x, y) for x, y in corner_points if x < left_third]
        corners_right = [(x, y) for x, y in corner_points if x > right_third]
        corners_center = [(x, y) for x, y in corner_points if left_third <= x <= right_third]
        corners_top = [(x, y) for x, y in corner_points if y < top_half]
        
        # Adjust confidence based on corner distance
        if distance_analysis['corner_distance'] == 'far':
            scene['confidence'] = 0.4
        else:
            scene['confidence'] = 0.9
        
        # Classification logic
        if len(corners_center) > 4 and len(corners_top) > 3:
            scene['scene_type'] = 'dead_end'
            scene['wall_ahead'] = True
        elif len(corner_points) < 3:
            scene['scene_type'] = 'intersection'
            scene['intersection_type'] = '4way'
        elif len(corners_left) > 2 and len(corners_center) > 2:
            scene['scene_type'] = 'L_turn'
            scene['corner_ahead_left'] = True
            scene['opening_right'] = True
        elif len(corners_right) > 2 and len(corners_center) > 2:
            scene['scene_type'] = 'L_turn'
            scene['corner_ahead_right'] = True
            scene['opening_left'] = True
        elif len(corners_center) > 2 and (len(corners_left) > 1 or len(corners_right) > 1):
            scene['scene_type'] = 'T_junction'
            scene['intersection_type'] = 'T'
            scene['opening_left'] = len(corners_left) < 2
            scene['opening_right'] = len(corners_right) < 2
        else:
            scene['scene_type'] = 'corridor'
            
        return scene
    
    def localize_with_confidence(self) -> Dict:
        """Localize position with confidence tracking"""
        observed_scene = self.detect_scene_with_precision()
        if not observed_scene:
            return {
                'status': 'error',
                'confidence': 0.0,
                'message': 'Failed to analyze scene'
            }
        
        # Handle special cases
        if observed_scene.get('status') == 'stationary':
            return observed_scene
        
        if observed_scene.get('status') == 'error':
            return observed_scene
        
        # If corners are too far ahead, reduce confidence
        if observed_scene.get('distance_analysis', {}).get('corner_distance') == 'far':
            return {
                'status': 'corners_too_far',
                'confidence': 0.2,
                'is_moving': observed_scene.get('is_moving', False),
                'scene_type': observed_scene.get('scene_type', 'unknown'),
                'message': 'Corners detected too far ahead - position uncertain'
            }
        
        # Match against known signatures
        best_matches = []
        for (x, y, direction), expected in self.corner_signatures.items():
            match_score = self.compare_scenes_precisely(observed_scene, expected)
            if match_score > 0.7:
                best_matches.append(((x, y, direction), match_score))
        
        if best_matches:
            best_matches.sort(key=lambda x: x[1], reverse=True)
            best_pos = best_matches[0][0]
            confidence = best_matches[0][1]
            
            # Update position only if confidence is high enough
            if confidence > self.min_confidence:
                self.current_pos = (best_pos[0], best_pos[1])
                self.current_direction = best_pos[2]
                self.position_confidence = confidence
                
                return {
                    'status': 'localized',
                    'position': self.current_pos,
                    'direction': self.current_direction,
                    'confidence': confidence,
                    'scene_type': observed_scene['scene_type'],
                    'is_moving': observed_scene.get('is_moving', False),
                    'message': f'Localized at ({self.current_pos[0]}, {self.current_pos[1]})'
                }
        
        return {
            'status': 'uncertain',
            'confidence': 0.0,
            'scene_type': observed_scene.get('scene_type', 'unknown'),
            'is_moving': observed_scene.get('is_moving', False),
            'message': 'Cannot confidently determine position'
        }
    
    def compare_scenes_precisely(self, observed: Dict, expected: Dict) -> float:
        """Precisely compare observed scene with expected signature"""
        score = 0.0
        total_weight = 0.0
        
        # Scene type match (high weight)
        if observed.get('scene_type') == expected.get('scene_type'):
            score += 3.0
        total_weight += 3.0
        
        # Specific feature matches
        features = ['corner_ahead_left', 'corner_ahead_right', 'wall_ahead', 
                   'opening_left', 'opening_right']
        
        for feature in features:
            if observed.get(feature) == expected.get(feature):
                score += 1.0
            total_weight += 1.0
        
        # Intersection type match
        if observed.get('intersection_type') == expected.get('intersection_type'):
            score += 2.0
        total_weight += 2.0
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def start_localization(self):
        """Start the localization thread"""
        if not self.running:
            self.running = True
            self.localization_thread = threading.Thread(target=self._localization_loop)
            self.localization_thread.daemon = True
            self.localization_thread.start()
    
    def stop_localization(self):
        """Stop the localization thread"""
        self.running = False
        if self.localization_thread:
            self.localization_thread.join()
    
    def _localization_loop(self):
        """Main localization loop running in background thread"""
        while self.running:
            try:
                result = self.localize_with_confidence()
                self.last_status = result
                time.sleep(0.5)  # Update every 500ms
            except Exception as e:
                self.last_status = {
                    'status': 'error',
                    'confidence': 0.0,
                    'message': f'Localization error: {str(e)}'
                }
                time.sleep(1.0)  # Wait longer on error
    
    def get_status(self) -> Dict:
        """Get current localization status"""
        status = self.last_status.copy()
        status.update({
            'current_position': self.current_pos,
            'current_direction': self.current_direction,
            'position_confidence': self.position_confidence,
            'stationary_frames': self.stationary_frames,
            'is_stationary': self.stationary_frames > self.max_stationary_frames
        })
        return status
    
    def get_camera_frame(self):
        """Get current camera frame for video feed"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

# Global localizer instance
localizer = PreciseMazeLocalizer(start_pos=(0, 2), start_direction='E')

# HTML template (embedded)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Maze Robot Localization Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!-- Include the complete CSS and HTML from the previous artifact -->
</head>
<body>
    <!-- Include the complete body from the previous artifact -->
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the main dashboard page"""
    # For simplicity, return a basic page that loads the dashboard
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Maze Robot Localization</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-align: center;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: rgba(255,255,255,0.1);
                padding: 30px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            .btn {
                padding: 12px 24px;
                margin: 10px;
                border: none;
                border-radius: 8px;
                background: #3498db;
                color: white;
                cursor: pointer;
                font-size: 16px;
                text-decoration: none;
                display: inline-block;
            }
            .btn:hover { background: #2980b9; }
            .status {
                background: rgba(255,255,255,0.2);
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Maze Robot Localization Dashboard</h1>
            
            <div class="status">
                <h3>Current Status</h3>
                <p id="status-text">Initializing...</p>
                <p id="position-text">Position: (0, 2)</p>
                <p id="confidence-text">Confidence: 100%</p>
            </div>
            
            <div>
                <button class="btn" onclick="startLocalization()">Start Localization</button>
                <button class="btn" onclick="stopLocalization()">Stop Localization</button>
                <a href="/dashboard" class="btn">Full Dashboard</a>
            </div>
            
            <div id="log" style="background: rgba(0,0,0,0.3); padding: 15px; border-radius: 8px; margin: 20px 0; text-align: left; font-family: monospace; max-height: 200px; overflow-y: auto;"></div>
        </div>
        
        <script>
            let isRunning = false;
            
            function addLog(message) {
                const log = document.getElementById('log');
                const time = new Date().toLocaleTimeString();
                log.innerHTML += `${time}: ${message}<br>`;
                log.scrollTop = log.scrollHeight;
            }
            
            function updateStatus() {
                fetch('/status')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('status-text').textContent = `Status: ${data.status || 'Unknown'}`;
                        if (data.current_position) {
                            document.getElementById('position-text').textContent = 
                                `Position: (${data.current_position[0]}, ${data.current_position[1]}) facing ${data.current_direction || 'Unknown'}`;
                        }
                        document.getElementById('confidence-text').textContent = 
                            `Confidence: ${((data.confidence || 0) * 100).toFixed(1)}%`;
                        
                        if (data.message) {
                            addLog(data.message);
                        }
                    })
                    .catch(error => {
                        addLog(`Error: ${error.message}`);
                    });
            }
            
            function startLocalization() {
                fetch('/start', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        addLog(data.message);
                        isRunning = true;
                        if (isRunning) {
                            setInterval(() => {
                                if (isRunning) updateStatus();
                            }, 500);
                        }
                    });
            }
            
            function stopLocalization() {
                fetch('/stop', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        addLog(data.message);
                        isRunning = false;
                    });
            }
            
            // Initial status update
            addLog('Dashboard loaded');
            updateStatus();
        </script>
    </body>
    </html>
    '''

@app.route('/dashboard')
def dashboard():
    """Serve the full interactive dashboard"""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Maze Robot Localization Dashboard</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }

        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }

        .dashboard {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 30px;
            align-items: start;
        }

        .maze-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        }

        .maze {
            display: grid;
            grid-template-columns: repeat(21, 1fr);
            gap: 2px;
            max-width: 800px;
            margin: 0 auto;
            background: #333;
            padding: 10px;
            border-radius: 10px;
        }

        .cell {
            width: 30px;
            height: 30px;
            border-radius: 3px;
            position: relative;
            transition: all 0.3s ease;
        }

        .cell.wall {
            background: #2c3e50;
            box-shadow: inset 2px 2px 4px rgba(0, 0, 0, 0.3);
        }

        .cell.path {
            background: #ecf0f1;
            box-shadow: inset 1px 1px 2px rgba(0, 0, 0, 0.1);
        }

        .cell.robot {
            background: #e74c3c !important;
            box-shadow: 0 0 20px rgba(231, 76, 60, 0.8);
            animation: pulse 2s infinite;
        }

        .cell.target {
            background: #27ae60 !important;
            box-shadow: 0 0 15px rgba(39, 174, 96, 0.6);
        }

        .cell.visited {
            background: #f39c12 !important;
            opacity: 0.7;
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }

        .robot-direction {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 16px;
            font-weight: bold;
            color: white;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
        }

        .status-panel {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 25px;
            color: #333;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        }

        .status-item {
            margin: 15px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #3498db;
            transition: all 0.3s ease;
        }

        .status-item:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        .status-item.error {
            border-left-color: #e74c3c;
            background: #fdf2f2;
        }

        .status-item.success {
            border-left-color: #27ae60;
            background: #f0f9f4;
        }

        .status-item.warning {
            border-left-color: #f39c12;
            background: #fefbf3;
        }

        .confidence-bar {
            width: 100%;
            height: 20px;
            background: #ddd;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }

        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #e74c3c 0%, #f39c12 50%, #27ae60 100%);
            border-radius: 10px;
            transition: width 0.5s ease;
        }

        .controls {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin: 20px 0;
        }

        .btn {
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .btn-primary {
            background: linear-gradient(45deg, #3498db, #2980b9);
            color: white;
        }

        .btn-danger {
            background: linear-gradient(45deg, #e74c3c, #c0392b);
            color: white;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }

        .camera-feed {
            width: 100%;
            max-width: 350px;
            border-radius: 10px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            margin: 15px 0;
        }

        .coordinates {
            font-family: 'Courier New', monospace;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            padding: 10px;
            background: #34495e;
            color: white;
            border-radius: 8px;
            margin: 10px 0;
        }

        .log-container {
            max-height: 200px;
            overflow-y: auto;
            background: #2c3e50;
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            margin: 15px 0;
        }

        .log-entry {
            margin: 5px 0;
            padding: 5px;
            border-radius: 4px;
        }

        .log-entry.info { background: rgba(52, 152, 219, 0.2); }
        .log-entry.warning { background: rgba(243, 156, 18, 0.2); }
        .log-entry.error { background: rgba(231, 76, 60, 0.2); }
        .log-entry.success { background: rgba(39, 174, 96, 0.2); }

        .legend {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(255, 255, 255, 0.1);
            padding: 8px 15px;
            border-radius: 20px;
            backdrop-filter: blur(5px);
        }

        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }

        @media (max-width: 1200px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
            
            .maze {
                grid-template-columns: repeat(21, 1fr);
                gap: 1px;
            }
            
            .cell {
                width: 20px;
                height: 20px;
            }
        }

        @media (max-width: 768px) {
            .cell {
                width: 15px;
                height: 15px;
            }
        }

        /* Ensure maze is visible */
        .maze-container h3 {
            color: #333;
            text-align: center;
            margin-bottom: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Maze Robot Localization Dashboard</h1>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #2c3e50;"></div>
                <span>Wall</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #ecf0f1;"></div>
                <span>Path</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #e74c3c;"></div>
                <span>Robot</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #27ae60;"></div>
                <span>Target</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #f39c12;"></div>
                <span>Visited</span>
            </div>
        </div>

        <div class="dashboard">
            <div class="maze-container">
                <h3>Maze Layout</h3>
                <div id="maze" class="maze"></div>
                <p style="text-align: center; color: #666; margin-top: 10px;">
                    Click any white cell to set target
                </p>
            </div>

            <div class="status-panel">
                <h3>🎯 Robot Status</h3>
                
                <div class="coordinates">
                    <div>Position: <span id="position">(0, 2)</span></div>
                    <div>Direction: <span id="direction">E</span></div>
                </div>

                <div class="status-item" id="status-item">
                    <strong>Status:</strong> <span id="status">Initializing...</span>
                </div>

                <div class="status-item">
                    <strong>Confidence:</strong>
                    <div class="confidence-bar">
                        <div class="confidence-fill" id="confidence-fill" style="width: 0%"></div>
                    </div>
                    <span id="confidence-text">0%</span>
                </div>

                <div class="status-item">
                    <strong>Scene Type:</strong> <span id="scene-type">Unknown</span>
                </div>

                <div class="status-item">
                    <strong>Movement:</strong> <span id="movement">Stationary</span>
                </div>

                <div class="controls">
                    <button class="btn btn-primary" onclick="startLocalization()">Start</button>
                    <button class="btn btn-danger" onclick="stopLocalization()">Stop</button>
                    <button class="btn btn-primary" onclick="setTarget()">Set Target</button>
                    <button class="btn btn-primary" onclick="resetPath()">Reset Path</button>
                </div>

                <img id="camera-feed" class="camera-feed" src="/video_feed" style="display: none;" alt="Camera Feed">
                <button class="btn btn-primary" onclick="toggleCamera()">Toggle Camera</button>

                <div class="log-container" id="log-container">
                    <div class="log-entry info">System initialized</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Maze configuration - your exact maze
        const maze = [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
            [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        ];

        let robotPos = [0, 2];
        let robotDirection = 'E';
        let targetPos = [20, 2];
        let visitedPath = [];
        let isRunning = false;
        let cameraVisible = false;

        // Direction symbols
        const directionSymbols = {
            'N': '↑',
            'S': '↓',
            'E': '→',
            'W': '←'
        };

        function createMaze() {
            const mazeElement = document.getElementById('maze');
            mazeElement.innerHTML = '';

            for (let y = 0; y < maze.length; y++) {
                for (let x = 0; x < maze[y].length; x++) {
                    const cell = document.createElement('div');
                    cell.className = 'cell';
                    cell.id = `cell-${x}-${y}`;
                    
                    if (maze[y][x] === 1) {
                        cell.classList.add('wall');
                    } else {
                        cell.classList.add('path');
                    }

                    cell.onclick = () => setTarget(x, y);
                    mazeElement.appendChild(cell);
                }
            }
            updateMazeDisplay();
        }

        function updateMazeDisplay() {
            // Clear all special classes
            document.querySelectorAll('.cell').forEach(cell => {
                cell.classList.remove('robot', 'target', 'visited');
                cell.innerHTML = '';
            });

            // Mark visited path
            visitedPath.forEach(([x, y]) => {
                const cell = document.getElementById(`cell-${x}-${y}`);
                if (cell) cell.classList.add('visited');
            });

            // Mark target
            const targetCell = document.getElementById(`cell-${targetPos[0]}-${targetPos[1]}`);
            if (targetCell) targetCell.classList.add('target');

            // Mark robot position
            const robotCell = document.getElementById(`cell-${robotPos[0]}-${robotPos[1]}`);
            if (robotCell) {
                robotCell.classList.add('robot');
                robotCell.innerHTML = `<span class="robot-direction">${directionSymbols[robotDirection]}</span>`;
            }
        }

        function updateStatus(data) {
            document.getElementById('position').textContent = `(${robotPos[0]}, ${robotPos[1]})`;
            document.getElementById('direction').textContent = robotDirection;
            
            if (data) {
                const statusElement = document.getElementById('status');
                const statusItem = document.getElementById('status-item');
                
                statusElement.textContent = data.status || 'Unknown';
                
                // Update status item class based on status
                statusItem.className = 'status-item';
                if (data.status === 'localized') {
                    statusItem.classList.add('success');
                } else if (data.status === 'uncertain' || data.status === 'corners_too_far') {
                    statusItem.classList.add('warning');
                } else if (data.status === 'error') {
                    statusItem.classList.add('error');
                }

                // Update confidence
                const confidence = (data.confidence || 0) * 100;
                document.getElementById('confidence-fill').style.width = `${confidence}%`;
                document.getElementById('confidence-text').textContent = `${confidence.toFixed(1)}%`;

                // Update other fields
                document.getElementById('scene-type').textContent = data.scene_type || 'Unknown';
                document.getElementById('movement').textContent = data.is_moving ? 'Moving' : 'Stationary';

                // Update robot position if provided
                if (data.current_position) {
                    robotPos = data.current_position;
                    robotDirection = data.current_direction || robotDirection;
                    
                    // Add to visited path if not already there
                    if (!visitedPath.some(([x, y]) => x === robotPos[0] && y === robotPos[1])) {
                        visitedPath.push([...robotPos]);
                    }
                }

                updateMazeDisplay();
                addLogEntry(data.message || `Status: ${data.status}`, getLogType(data.status));
            }
        }

        function getLogType(status) {
            if (status === 'localized') return 'success';
            if (status === 'uncertain' || status === 'corners_too_far' || status === 'stationary') return 'warning';
            if (status === 'error') return 'error';
            return 'info';
        }

        function addLogEntry(message, type = 'info') {
            const logContainer = document.getElementById('log-container');
            const logEntry = document.createElement('div');
            logEntry.className = `log-entry ${type}`;
            logEntry.textContent = `${new Date().toLocaleTimeString()}: ${message}`;
            logContainer.appendChild(logEntry);
            logContainer.scrollTop = logContainer.scrollHeight;
        }

        function startLocalization() {
            if (!isRunning) {
                fetch('/start', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        addLogEntry(data.message, 'info');
                        isRunning = true;
                        pollStatus();
                    })
                    .catch(error => {
                        addLogEntry(`Error starting: ${error.message}`, 'error');
                    });
            }
        }

        function stopLocalization() {
            isRunning = false;
            fetch('/stop', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    addLogEntry(data.message, 'warning');
                })
                .catch(error => {
                    addLogEntry(`Error stopping: ${error.message}`, 'error');
                });
        }

        function setTarget(x = null, y = null) {
            if (x !== null && y !== null) {
                if (maze[y] && maze[y][x] === 0) {
                    targetPos = [x, y];
                    updateMazeDisplay();
                    addLogEntry(`Target set to (${x}, ${y})`, 'success');
                } else {
                    addLogEntry(`Cannot set target on wall at (${x}, ${y})`, 'error');
                }
            } else {
                const x = prompt('Enter target X coordinate (0-20):');
                const y = prompt('Enter target Y coordinate (0-14):');
                if (x !== null && y !== null) {
                    setTarget(parseInt(x), parseInt(y));
                }
            }
        }

        function resetPath() {
            visitedPath = [];
            updateMazeDisplay();
            addLogEntry('Path history cleared', 'info');
        }

        function toggleCamera() {
            const camera = document.getElementById('camera-feed');
            cameraVisible = !cameraVisible;
            camera.style.display = cameraVisible ? 'block' : 'none';
        }

        async function pollStatus() {
            if (!isRunning) return;

            try {
                const response = await fetch('/status');
                const data = await response.json();
                updateStatus(data);
            } catch (error) {
                addLogEntry(`Error fetching status: ${error.message}`, 'error');
            }

            if (isRunning) {
                setTimeout(pollStatus, 500); // Poll every 500ms
            }
        }

        // Initialize the maze on page load
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Initializing dashboard...');
            createMaze();
            addLogEntry('Dashboard initialized', 'success');
            
            // Initial status update
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    console.log('Initial status:', data);
                    updateStatus(data);
                })
                .catch(error => {
                    console.error('Error fetching initial status:', error);
                    addLogEntry(`Error: ${error.message}`, 'error');
                });
        });
    </script>
</body>
</html>
    '''

@app.route('/status')
def get_status():
    """Get current localization status as JSON"""
    try:
        status = localizer.get_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'confidence': 0.0
        })

@app.route('/start', methods=['POST'])
def start_localization():
    """Start the localization process"""
    try:
        localizer.start_localization()
        return jsonify({
            'success': True,
            'message': 'Localization started'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to start localization: {str(e)}'
        })

@app.route('/stop', methods=['POST'])
def stop_localization():
    """Stop the localization process"""
    try:
        localizer.stop_localization()
        return jsonify({
            'success': True,
            'message': 'Localization stopped'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to stop localization: {str(e)}'
        })

@app.route('/test_maze')
def test_maze():
    """Simple test page to check maze rendering"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Maze Test</title>
        <style>
            .maze {
                display: grid;
                grid-template-columns: repeat(21, 30px);
                gap: 2px;
                margin: 20px;
                background: #333;
                padding: 10px;
                border-radius: 10px;
                width: fit-content;
            }
            .cell {
                width: 30px;
                height: 30px;
                border-radius: 3px;
            }
            .cell.wall { background: #2c3e50; }
            .cell.path { background: #ecf0f1; }
            .cell.robot { background: #e74c3c; }
        </style>
    </head>
    <body>
        <h1>Maze Test</h1>
        <div id="maze" class="maze"></div>
        
        <script>
            const maze = [
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
                [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
                [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0],
                [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
                [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
                [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
                [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            ];
            
            function createMaze() {
                const mazeElement = document.getElementById('maze');
                mazeElement.innerHTML = '';

                for (let y = 0; y < maze.length; y++) {
                    for (let x = 0; x < maze[y].length; x++) {
                        const cell = document.createElement('div');
                        cell.className = 'cell';
                        
                        if (maze[y][x] === 1) {
                            cell.classList.add('wall');
                        } else {
                            cell.classList.add('path');
                        }
                        
                        // Mark robot position
                        if (x === 0 && y === 2) {
                            cell.classList.add('robot');
                        }

                        mazeElement.appendChild(cell);
                    }
                }
            }
            
            createMaze();
        </script>
    </body>
    </html>
    '''

@app.route('/debug')
def debug_info():
    """Debug information"""
    return jsonify({
        'maze_size': f"{len(localizer.maze)} x {len(localizer.maze[0])}",
        'current_position': localizer.current_pos,
        'current_direction': localizer.current_direction,
        'camera_available': localizer.cap is not None and localizer.cap.isOpened(),
        'signatures_count': len(localizer.corner_signatures),
        'sample_maze_row': localizer.maze[2]  # Show row 2 as example
    })
def set_target():
    """Set a new target position"""
    try:
        data = request.get_json()
        x, y = data.get('x'), data.get('y')
        
        if x is None or y is None:
            return jsonify({
                'success': False,
                'message': 'Missing x or y coordinates'
            })
        
        # Validate target position
        if (0 <= y < len(localizer.maze) and 
            0 <= x < len(localizer.maze[0]) and 
            localizer.maze[y][x] == 0):
            
            return jsonify({
                'success': True,
                'message': f'Target set to ({x}, {y})'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Invalid target position ({x}, {y}) - must be on a valid path'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error setting target: {str(e)}'
        })

def generate_camera_frames():
    """Generate camera frames for video streaming"""
    while True:
        frame = localizer.get_camera_frame()
        if frame is not None:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # Send placeholder frame if camera not available
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, 'Camera Not Available', (150, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.1)  # 10 FPS

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_camera_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting Maze Localization Server...")
    print("Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    finally:
        localizer.stop_localization()
        if localizer.cap:
            localizer.cap.release()