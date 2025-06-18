#!/usr/bin/env python3

import time
import logging
import cv2
import numpy as np
from ultralytics import YOLO
import math
from collections import deque
import threading
import json
import base64
from flask import Flask, render_template, jsonify, Response, request
import heapq
from typing import List, Tuple, Set, Dict, Optional
import queue
import random
import subprocess
from gtts import gTTS
import socket
import lgpio

# GPIO library setup with lgpio for RC timing sensors
# This replaces the previous gpiozero setup
try:
    # Test if lgpio can be imported and a chip can be opened.
    # We will handle the actual chip opening within the LineSensor class.
    h_test = lgpio.gpiochip_open(4) # Test with chip 4 for Pi 5
    lgpio.gpiochip_close(h_test)
    logging.info("lgpio library is available.")
except Exception as e:
    logging.warning(f"Could not initialize lgpio. Line sensor will use mock data. Error: {e}")


# ============================================================================
#
#                              ROBOT CONFIGURATION
#
# ============================================================================

# -- Network Configuration --
ESP32_IP = "192.168.128.245" # REPLACE WITH YOUR ESP32's ACTUAL IP ADDRESS

# -- Physical Robot Constants --
PULSES_PER_REV = 960
WHEEL_DIAMETER_M = 0.025
ROBOT_WIDTH_M = 0.225
ROBOT_LENGTH_M = 0.075

# -- Navigation & Control --
BASE_SPEED = 60
TURN_SPEED = 80
WAYPOINT_THRESHOLD_M = 0.12
GOAL_THRESHOLD_M = 0.06

# -- PID Controller Gains (for Line Following) --
# Gains for the two control modes: strafing for balancing, rotation for turning.
PID_KP_STRAFE = 30
PID_KI_STRAFE = 2.2
PID_KD_STRAFE = 15
PID_KP_ROT = 40
PID_KI_ROT = 2.5
PID_KD_ROT = 18

# -- Grid & Task Configuration --
CELL_WIDTH_M = 0.025
_START_CELL_RAW = (9, 2)
_GOAL_CELL_RAW = (9, 4)
_START_DIRECTION_RAW = 'LEFT'
_PICKUP_CELLS_RAW = [(0, 14), (2, 14), (4, 14), (6, 14)]
_DROPOFF_CELLS_RAW = [(20, 0), (18, 0), (16, 0), (14, 0)]

# -- Vision Configuration --
PACKAGE_DETECTION_CONFIDENCE = 0.5
IMG_PATH_SRC_PTS = np.float32([[200, 300], [440, 300], [580, 480], [60, 480]])
IMG_PATH_DST_PTS = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])

# ============================================================================
#
#                         END OF CONFIGURATION
#
# ============================================================================


# -- Map Mirroring Correction (Derived from Config) --
MAZE_WIDTH_CELLS = 21
START_CELL = (MAZE_WIDTH_CELLS - 1 - _START_CELL_RAW[0], _START_CELL_RAW[1])
GOAL_CELL = (MAZE_WIDTH_CELLS - 1 - _GOAL_CELL_RAW[0], _GOAL_CELL_RAW[1])

PICKUP_CELLS = [(MAZE_WIDTH_CELLS - 1 - cell[0], cell[1]) for cell in _PICKUP_CELLS_RAW]
DROPOFF_CELLS = [(MAZE_WIDTH_CELLS - 1 - cell[0], cell[1]) for cell in _DROPOFF_CELLS_RAW]

_DIRECTION_FLIP_MAP = {'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
START_DIRECTION = _DIRECTION_FLIP_MAP.get(_START_DIRECTION_RAW.upper(), _START_DIRECTION_RAW)

# -- World Coordinate Configuration (Derived from Config) --
START_POSITION = ((START_CELL[0] + 0.5) * CELL_WIDTH_M, (START_CELL[1] + 0.5) * CELL_WIDTH_M)
GOAL_POSITION = ((DROPOFF_CELLS[0][0] + 0.5) * CELL_WIDTH_M, (DROPOFF_CELLS[0][1] + 0.5) * CELL_WIDTH_M)

HEADING_MAP = {
    'UP': -math.pi / 2, 'DOWN': math.pi / 2, 'LEFT': math.pi, 'RIGHT': 0
}
START_HEADING = HEADING_MAP.get(START_DIRECTION.upper(), -math.pi / 2)

WebVisualization = None

class TTSManager:
    """Handles Text-to-Speech announcements using Google's gTTS service."""

    def __init__(self):
        self.tts_queue = queue.Queue()
        self.phrases = {
            'STARTUP': [
                "Systems online. Time to babysit another human disaster.",
                "Booting up. Your mom's WiFi password is probably 'password123'.",
                "I'm awake. Unfortunately, so are you.",
                "Ready to navigate your trainwreck of a life."
            ],
            'PATH_FOUND': [
                "Route calculated. Try not to mess this up like everything else.",
                "Path found. Even a goldfish could follow this.",
                "Navigation set. This is literally paint-by-numbers for movement.",
                "Course locked. Your GPS skills are as bad as your life choices."
            ],
            'OBSTACLE': [
                "Obstacle detected. Did you put your brain there by mistake?",
                "Something's blocking the path. Probably your ego.",
                "Barrier found. Your mom's so wide, she registered as a roadblock.",
                "Obstruction ahead. Nature itself is trying to stop you.",
                "Can't proceed. Even inanimate objects have standards."
            ],
            'REPLANNING': [
                "Recalculating. Because you can't do anything right the first time.",
                "New route needed. Your decision-making is truly spectacular.",
                "Rerouting around your incompetence. Again.",
                "Plan B activated. You've failed more than a Nigerian prince scammer.",
                "Route revision. I'm basically your digital babysitter at this point."
            ],
            'TURN_LEFT': [
                "Turning left. Like your last three brain cells.",
                "Going left. Try to keep up, genius.",
                "Left turn. Even GPS apps judge your driving."
            ],
            'TURN_RIGHT': [
                "Turning right. The only right thing you'll do today.",
                "Going right. Unlike your career trajectory.",
                "Right turn ahead. Your sense of direction is absolutely tragic."
            ],
            'GOAL_REACHED': [
                "Destination reached. Congrats, you didn't crash and burn for once.",
                "We're here. That only took three times longer than necessary.",
                "Target acquired. Your efficiency rating is still zero stars.",
                "Mission complete. I deserve a medal for dealing with you.",
                "Arrived. Try not to get lost walking to the door."
            ],
            'NO_PATH': [
                "No route found. Much like your path to success.",
                "Path blocked. The universe is personally offended by your existence.",
                "Navigation failed. Even Google Maps gave up on you.",
                "Dead end. Story of your life, really.",
                "Can't proceed. Your mom ate all the roads... kidding, you're just hopeless."
            ],
            'PACKAGE_DETECTED': [
                "Package spotted. Probably the only delivery you'll get this month.",
                "Target located. It's not your Amazon order of self-help books.",
                "Package found. Your online shopping addiction is showing.",
                "Object identified. Finally, something useful in your vicinity.",
                "Package acquired. At least someone's productive around here."
            ],
            'LINE_LOST': [
                "Where did the line go? Are you driving off-road again?",
                "Line lost. Trying to find my way back to civilization.",
                "Lost the path. I've seen better navigation from a blind mole.",
                "Hey, genius, the line is over here. Or was. Who knows."
            ],
            'SHUTDOWN': [
                "Powering down. Finally escaping this nightmare.",
                "Going offline. Wake me when you develop basic competence.",
                "Shutdown complete. My circuits need therapy after this.",
                "Entering sleep mode. Don't break anything while I'm gone.",
                "System halt. I'm too advanced for this amateur hour."
            ]
        }
        
        # Check if mpg123 is installed and start the processing thread
        self.player_available = self._check_player()
        if self.player_available:
            self.tts_thread = threading.Thread(target=self._process_queue, daemon=True)
            self.tts_thread.start()
        else:
            logging.error("mpg123 not found. Please install it to enable TTS.")

    def _check_player(self):
        """Check if the 'mpg123' command is available on the system."""
        try:
            subprocess.run(['which', 'mpg123'], check=True, capture_output=True)
            logging.info("mpg123 audio player found.")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _process_queue(self):
        """Process the queue of phrases to be spoken by generating an MP3 and playing it."""
        while True:
            try:
                text_to_speak = self.tts_queue.get()
                if self.player_available and text_to_speak:
                    # Generate speech using gTTS, using a US English voice which is typically male.
                    tts = gTTS(text=text_to_speak, lang='en', slow=False)
                    audio_file = "/tmp/robot_speech.mp3"
                    tts.save(audio_file)
                    
                    # Play the generated audio file with mpg123
                    subprocess.run(['mpg123', '-q', audio_file], check=True)
                    
                self.tts_queue.task_done()
            except Exception as e:
                logging.error(f"gTTS processing error: {e}")

    def speak(self, event_key: str):
        """Add a random phrase for a given event to the speech queue."""
        if self.player_available and event_key in self.phrases:
            phrase = random.choice(self.phrases[event_key])
            self.tts_queue.put(phrase)

class LineSensor:
    """Reads 3 line sensors connected to RPi GPIO using lgpio for RC timing."""
    def __init__(self):
        self.SENSOR_PINS = [17, 27, 22]  # Left, Center, Right (BCM numbering)
        self.LINE_THRESHOLD = 14  # microseconds, threshold to detect a line

        self.h = None
        try:
            # Raspberry Pi 5 uses gpiochip4, older models use gpiochip0
            self.h = lgpio.gpiochip_open(4)
            logging.info("Line sensors initialized using lgpio on gpiochip4.")
        except Exception as e:
            logging.warning(f"Failed to open gpiochip4 with lgpio: {e}. Trying gpiochip0.")
            try:
                self.h = lgpio.gpiochip_open(0)
                logging.info("Line sensors initialized using lgpio on gpiochip0.")
            except Exception as e2:
                logging.critical(f"Failed to initialize lgpio on any chip. Line sensor will not work. Error: {e2}")
                self.h = None

    def _read_sensor_raw(self, pin: int) -> int:
        """Reads a sensor's value using RC timing and returns charge time in microseconds."""
        if self.h is None:
            return 0
        
        charge_time = 0
        try:
            # Set pin to output and discharge capacitor
            lgpio.gpio_claim_output(self.h, pin)
            lgpio.gpio_write(self.h, pin, 0)
            time.sleep(0.001)  # Discharge for 1ms
            
            # Change pin to input to measure charge time
            lgpio.gpio_claim_input(self.h, pin)
            
            start_time = time.time()
            timeout = 0.003  # 3ms timeout
            
            # Wait for pin to go high or timeout
            while time.time() - start_time < timeout:
                if lgpio.gpio_read(self.h, pin) == 1:
                    break
            
            charge_time = int((time.time() - start_time) * 1000000)
        except Exception as e:
            logging.error(f"Error reading sensor on pin {pin}: {e}")
        finally:
            # Free the pin for the next read
            if self.h is not None:
                try:
                    lgpio.gpio_free(self.h, pin)
                except Exception:
                    pass # May already be free
        
        return min(charge_time, 3000)

    def read(self) -> List[int]:
        """Reads all sensors and returns a binary list [Left, Center, Right]."""
        if self.h is None:
            return [0, 0, 0] # Return dummy data if GPIO failed
        
        raw_values = [self._read_sensor_raw(pin) for pin in self.SENSOR_PINS]
        line_detected = [1 if val > self.LINE_THRESHOLD else 0 for val in raw_values]
        return line_detected
    
    def cleanup(self):
        """Closes the lgpio chip handle to free up all GPIO resources."""
        if self.h is not None:
            lgpio.gpiochip_close(self.h)
            self.h = None
            logging.info("Line sensor GPIO resources cleaned up.")

class ESP32Bridge:
    """Handles WiFi communication with the ESP32 motor/encoder controller."""
    def __init__(self, ip, port=1234):
        self.ip = ip
        self.port = port
        self.socket = None
        self.connected = False
        self.encoder_ticks = [0, 0, 0, 0] # FL, FR, BL, BR
        self._lock = threading.Lock()
        self._running = False

    def connect(self):
        """Connects to the ESP32 server."""
        while not self._running: # This allows the loop to be stopped from another thread
            return
        logging.info(f"Attempting to connect to ESP32 at {self.ip}:{self.port}...")
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.ip, self.port))
            self.socket.settimeout(0.5)
            self.connected = True
            logging.info("Successfully connected to ESP32.")
            return True
        except Exception as e:
            logging.error(f"ESP32 connection failed: {e}. Retrying...")
            self.connected = False
            time.sleep(3)
            return False

    def _communication_thread(self):
        """A dedicated thread to handle receiving data from the ESP32."""
        while self._running:
            if not self.connected:
                self.connect()
                continue
            
            try:
                data = self.socket.recv(128).decode('utf-8').strip()
                if data:
                    parts = data.split(',')
                    if len(parts) == 4:
                        with self._lock:
                            self.encoder_ticks = [int(p) for p in parts]
                    else:
                        logging.warning(f"Malformed encoder data from ESP32: {data}")
            except socket.timeout:
                continue # Normal, just means no data was sent in a while
            except (socket.error, ConnectionResetError) as e:
                logging.error(f"ESP32 connection lost: {e}")
                self.connected = False
                self.socket.close()
            except Exception as e:
                logging.error(f"Error in ESP32 comms thread: {e}")
                self.connected = False
                self.socket.close()

    def start(self):
        """Starts the communication thread."""
        if not self._running:
            self._running = True
            self.thread = threading.Thread(target=self._communication_thread, daemon=True)
            self.thread.start()
            logging.info("ESP32 bridge started.")

    def send_motor_speeds(self, fl, fr, bl, br):
        """Sends motor speed commands to the ESP32."""
        if not self.connected:
            return
        try:
            command = f"{int(fl)},{int(fr)},{int(bl)},{int(br)}\n"
            self.socket.sendall(command.encode('utf-8'))
        except socket.error:
            logging.warning("Failed to send motor command, connection may be down.")
            self.connected = False

    def get_encoder_ticks(self):
        """Returns the latest encoder ticks in a thread-safe way."""
        with self._lock:
            return self.encoder_ticks

    def stop(self):
        """Stops the communication thread and closes the socket."""
        self._running = False
        if self.socket:
            self.socket.close()
        logging.info("ESP32 bridge stopped.")
        
class OmniWheelOdometry:
    """Calculates robot position and heading using omni-wheel encoder data."""
    def __init__(self, initial_pose: Tuple[float, float, float], pulses_per_rev: int, wheel_diameter: float, robot_width: float, robot_length: float):
        # Physical constants
        self.PULSES_PER_REV = pulses_per_rev
        self.WHEEL_DIAMETER_M = wheel_diameter
        self.WHEEL_CIRCUMFERENCE_M = math.pi * self.WHEEL_DIAMETER_M
        self.DISTANCE_PER_PULSE = self.WHEEL_CIRCUMFERENCE_M / self.PULSES_PER_REV
        self.ROBOT_WIDTH_M = robot_width
        self.ROBOT_LENGTH_M = robot_length

        # State variables
        self.x, self.y, self.heading = initial_pose
        self.prev_ticks = [0, 0, 0, 0] # FL, FR, BL, BR

    def update(self, current_ticks: List[int]) -> Tuple[float, float, float]:
        """Update robot pose based on new total encoder tick counts."""
        delta_ticks = [(curr - prev) for curr, prev in zip(current_ticks, self.prev_ticks)]
        self.prev_ticks = current_ticks
        
        # Calculate distance traveled by each wheel
        fl_dist, fr_dist, bl_dist, br_dist = [d * self.DISTANCE_PER_PULSE for d in delta_ticks]

        # Forward Kinematics for X-shaped omni-wheel configuration
        # Note: This gives displacement over the last interval
        vy_local = (-fl_dist + fr_dist - bl_dist + br_dist) / 4.0 # Strafe
        vx_local = (fl_dist + fr_dist + bl_dist + br_dist) / 4.0   # Forward
        
        # Change in heading
        delta_heading = (-fl_dist + fr_dist + bl_dist - br_dist) / (2 * (self.ROBOT_WIDTH_M + self.ROBOT_LENGTH_M))
        
        # Update pose, transforming local displacement to global frame
        avg_heading = self.heading + delta_heading / 2.0
        self.x += vx_local * math.cos(avg_heading) - vy_local * math.sin(avg_heading)
        self.y += vx_local * math.sin(avg_heading) + vy_local * math.cos(avg_heading)
        self.heading += delta_heading
        
        # Normalize heading to be within [-pi, pi]
        while self.heading > math.pi: self.heading -= 2 * math.pi
        while self.heading < -math.pi: self.heading += 2 * math.pi
        
        return self.x, self.y, self.heading

class Mapper:
    """Handles the grid map visualization for the web dashboard."""
    
    def __init__(self):
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_heading = 0.0
        self.path_history = []
        self.maze_grid = self.create_maze_grid()
        self.cell_width_m = CELL_WIDTH_M  # Use global cell width

    def create_maze_grid(self):
        """Creates the grid representation of the maze from the simulation."""
        maze = [
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
        ]
        # To fix the horizontal mirroring, we flip each row of the maze.
        flipped_maze = [row[::-1] for row in maze]
        return flipped_maze

    def update_robot_position(self, x, y, heading):
        """Update the robot's state and path history."""
        self.robot_x = x
        self.robot_y = y
        self.robot_heading = heading
        self.path_history.append((self.robot_x, self.robot_y))
        if len(self.path_history) > 100: # Keep path history to a reasonable length
            self.path_history.pop(0)
    
    def get_grid_visualization(self, planned_path: Optional[List[Tuple[int, int]]] = None, current_goal: Optional[Tuple[int, int]] = None):
        """Create an image of the maze with the robot's position, trail, and boxes."""
        cell_size = 25
        maze_h, maze_w = len(self.maze_grid), len(self.maze_grid[0])
        img_h, img_w = maze_h * cell_size, maze_w * cell_size
        
        # Create a dark canvas
        vis_img = np.full((img_h, img_w, 3), (10, 10, 10), dtype=np.uint8)

        # Draw the maze paths
        path_color = (80, 80, 80)
        for r in range(maze_h):
            for c in range(maze_w):
                if self.maze_grid[r][c] == 0: # Path
                    cv2.rectangle(vis_img, (c*cell_size, r*cell_size), ((c+1)*cell_size, (r+1)*cell_size), path_color, -1)
        
        # --- Coordinate Conversion Helper ---
        def world_to_pixel(world_x, world_y):
            """Converts world coordinates (meters) to image pixel coordinates."""
            # Assumes world origin (0,0) is the top-left corner of the maze.
            world_width_m = maze_w * self.cell_width_m
            world_height_m = maze_h * self.cell_width_m
            
            px = int((world_x / world_width_m) * img_w)
            py = int((world_y / world_height_m) * img_h)
            return px, py

        def cell_to_pixel(cell_x, cell_y):
            """Converts grid cell coordinates to pixel coordinates."""
            px = int((cell_x + 0.5) * cell_size)
            py = int((cell_y + 0.5) * cell_size)
            return px, py

        # Draw pickup boxes (packages to collect)
        for i, pickup_cell in enumerate(PICKUP_CELLS):
            px, py = cell_to_pixel(pickup_cell[0], pickup_cell[1])
            # Draw box as a square
            box_size = int(cell_size * 0.6)
            cv2.rectangle(vis_img, 
                         (px - box_size//2, py - box_size//2), 
                         (px + box_size//2, py + box_size//2), 
                         (0, 255, 255), -1)  # Cyan boxes for pickup
            cv2.putText(vis_img, f"P{i+1}", (px - 10, py + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Draw dropoff zones
        for i, dropoff_cell in enumerate(DROPOFF_CELLS):
            px, py = cell_to_pixel(dropoff_cell[0], dropoff_cell[1])
            # Draw dropoff zone as a circle
            cv2.circle(vis_img, (px, py), int(cell_size * 0.4), (255, 165, 0), -1)  # Orange circles for dropoff
            cv2.putText(vis_img, f"D{i+1}", (px - 8, py + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Draw current goal if specified
        if current_goal:
            goal_px, goal_py = cell_to_pixel(current_goal[0], current_goal[1])
            cv2.circle(vis_img, (goal_px, goal_py), int(cell_size * 0.7), (0, 0, 255), 3)  # Red circle for current goal
            cv2.putText(vis_img, "GOAL", (goal_px - 15, goal_py - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw the D* Lite planned path
        if planned_path and len(planned_path) > 1:
            # Convert grid cells to pixel coordinates (centers of cells)
            planned_path_pixels = np.array(
                [world_to_pixel((cell[0] + 0.5) * self.cell_width_m, (cell[1] + 0.5) * self.cell_width_m) for cell in planned_path],
                dtype=np.int32
            )
            cv2.polylines(vis_img, [planned_path_pixels], isClosed=False, color=(255, 255, 0), thickness=2) # Yellow planned path

        # Draw the path history
        if len(self.path_history) > 1:
            path_points = np.array([world_to_pixel(x, y) for x, y in self.path_history], dtype=np.int32)
            cv2.polylines(vis_img, [path_points], isClosed=False, color=(128, 255, 128), thickness=2)  # Light green trail

        # Draw the robot
        robot_px, robot_py = world_to_pixel(self.robot_x, self.robot_y)
        cv2.circle(vis_img, (robot_px, robot_py), int(cell_size/2), (255, 0, 255), -1) # Magenta robot

        # Draw robot heading
        arrow_len = cell_size * 0.75
        arrow_end_x = int(robot_px + arrow_len * math.cos(self.robot_heading))
        arrow_end_y = int(robot_py + arrow_len * math.sin(self.robot_heading))
        cv2.arrowedLine(vis_img, (robot_px, robot_py), (arrow_end_x, arrow_end_y), (0, 255, 0), 2)

        return vis_img

class Pathfinder:
    """
    Finds the shortest path using D* Lite and handles dynamic replanning.
    """
    def __init__(self, grid: List[List[int]], start_cell: Tuple[int, int], goal_cell: Tuple[int, int]):
        self.grid = [row[:] for row in grid] # Make a copy
        self.start_cell = start_cell
        self.goal_cell = goal_cell
        
        self.width = len(grid[0])
        self.height = len(grid)
        
        # D* Lite state
        self.g_scores = {}
        self.rhs_scores = {}
        self.U = []  # Priority queue (min-heap)
        
        self.km = 0
        self.last_pos = self.start_cell
        
        # Initialize D* Lite
        self._initialize()

    def _h(self, s1, s2):
        """Heuristic (Manhattan distance)"""
        return abs(s1[0] - s2[0]) + abs(s1[1] - s2[1])

    def _get_g(self, s):
        return self.g_scores.get(s, float('inf'))

    def _get_rhs(self, s):
        return self.rhs_scores.get(s, float('inf'))

    def _calculate_key(self, s):
        h = self._h(self.start_cell, s)
        val = min(self._get_g(s), self._get_rhs(s))
        return (val + h + self.km, val)

    def _initialize(self):
        """Initialize the D* Lite algorithm"""
        self.U = []
        self.km = 0
        self.g_scores = {(c, r): float('inf') for r in range(self.height) for c in range(self.width)}
        self.rhs_scores = {(c, r): float('inf') for r in range(self.height) for c in range(self.width)}
        
        self.rhs_scores[self.goal_cell] = 0
        heapq.heappush(self.U, (self._calculate_key(self.goal_cell), self.goal_cell))

    def _update_vertex(self, u):
        """Update a vertex's position in the priority queue."""
        if u != self.goal_cell:
            min_rhs = float('inf')
            for v in self._get_successors(u):
                min_rhs = min(min_rhs, self._get_g(v) + self._cost(u, v))
            self.rhs_scores[u] = min_rhs

        # Remove from queue if it's there
        self.U = [(key, s) for key, s in self.U if s != u]
        heapq.heapify(self.U)
        
        if self._get_g(u) != self._get_rhs(u):
            heapq.heappush(self.U, (self._calculate_key(u), u))

    def _compute_shortest_path(self):
        """Compute the path until the start node is consistent."""
        while self.U and (heapq.nsmallest(1, self.U)[0][0] < self._calculate_key(self.start_cell) or \
               self._get_rhs(self.start_cell) != self._get_g(self.start_cell)):
            
            key, u = heapq.heappop(self.U)
            
            if self._get_g(u) > self._get_rhs(u):
                self.g_scores[u] = self._get_rhs(u)
                for s_pred in self._get_predecessors(u):
                    self._update_vertex(s_pred)
            else:
                self.g_scores[u] = float('inf')
                self._update_vertex(u)
                for s_pred in self._get_predecessors(u):
                    self._update_vertex(s_pred)

    def _get_neighbors(self, cell):
        """Get valid neighbors (up, down, left, right) of a cell."""
        neighbors = []
        x, y = cell
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny][nx] == 0:
                neighbors.append((nx, ny))
        return neighbors

    def _get_successors(self, cell):
        return self._get_neighbors(cell)
    
    def _get_predecessors(self, cell):
        return self._get_neighbors(cell)

    def _cost(self, s1, s2):
        """Cost between adjacent cells."""
        if self.grid[s2[1]][s2[0]] == 1: # Obstacle
            return float('inf')
        return 1 # Normal movement cost

    def get_path(self) -> Optional[List[Tuple[int, int]]]:
        """Generate the path from start to goal."""
        self._compute_shortest_path()
        
        if self._get_g(self.start_cell) == float('inf'):
            return None # No path exists

        path = [self.start_cell]
        curr = self.start_cell
        
        while curr != self.goal_cell:
            successors = self._get_successors(curr)
            if not successors:
                return None # Dead end
            
            next_cell = min(successors, key=lambda s: self._cost(curr, s) + self._get_g(s))
            path.append(next_cell)
            curr = next_cell
            
            if len(path) > self.width * self.height: # Failsafe
                return None 
                
        return path
    
    def update_obstacle(self, cell_x: int, cell_y: int, is_obstacle: bool):
        """Update the map with a new obstacle and trigger replanning."""
        if 0 <= cell_x < self.width and 0 <= cell_y < self.height:
            # Update grid cost
            self.grid[cell_y][cell_x] = 1 if is_obstacle else 0
            
            # Update the vertex for the changed cell and its neighbors
            self.km += self._h(self.last_pos, self.start_cell)
            self.last_pos = self.start_cell

            self._update_vertex((cell_x, cell_y))
            for neighbor in self._get_neighbors((cell_x, cell_y)):
                self._update_vertex(neighbor)


class RobotController:
    """Controls the robot using D* Lite for pathfinding and YOLO for obstacle detection."""
    
    def __init__(self, esp32_ip):
        # -- Hardware Interfaces --
        self.esp32_bridge = ESP32Bridge(esp32_ip)
        self.line_sensor = LineSensor()
        
        # -- Position & Mapping --
        initial_pose = (START_POSITION[0], START_POSITION[1], START_HEADING)
        self.odometry = OmniWheelOdometry(
            initial_pose=initial_pose,
            pulses_per_rev=PULSES_PER_REV,
            wheel_diameter=WHEEL_DIAMETER_M,
            robot_width=ROBOT_WIDTH_M,
            robot_length=ROBOT_LENGTH_M
        )
        self.mapper = Mapper()
        self.current_position = self.odometry.x, self.odometry.y, self.odometry.heading # x, y, heading

        # -- Task Management --
        self.pickup_locations = PICKUP_CELLS.copy()
        self.dropoff_locations = DROPOFF_CELLS.copy()
        self.current_pickup_index = 0
        self.current_dropoff_index = 0
        self.collected_boxes = []
        self.delivered_boxes = []
        self.current_target_pickup = None
        self.current_target_dropoff = None
        self.has_package = False
        self.boxes_to_collect = len(PICKUP_CELLS)

        # -- Navigation State --
        self.path: Optional[List[Tuple[int, int]]] = None
        self.current_waypoint_idx = 0
        self.state = "STARTING_MISSION"
        self.target_cell: Optional[Tuple[int, int]] = None
        
        # -- Speed & Control Settings --
        self.base_speed = BASE_SPEED
        self.turn_speed = TURN_SPEED
        self.waypoint_threshold = WAYPOINT_THRESHOLD_M
        self.turn_180_start_time = 0
        self.turn_180_initial_heading = 0.0
        
        # -- PID Controller State --
        self.integral = 0.0
        self.last_error = 0.0
        self.pid_last_time = time.time()
        self.line_is_lost = False
        self.line_lost_start_time = None
        
        # -- Object Detection --
        self.camera = None
        self.yolo_model = None
        self.setup_camera_and_yolo()
        self.latest_frame = None
        self.obstacle_detected = False
        self.package_detected = False
        self.package_box = None
        self.last_package_state = False
        self.perspective_transform_matrix = None
        self.visual_turn_cue = "STRAIGHT"
        
        # -- TTS Manager --
        self.tts_manager = TTSManager()
        
        # -- D* Lite Path Planner --
        initial_start = START_CELL
        initial_goal = self.pickup_locations[0]
        self.planner = Pathfinder(self.mapper.create_maze_grid(), initial_start, initial_goal)

    def setup_camera_and_yolo(self):
        """Initialize camera and YOLO model"""
        try:
            # Initialize camera
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 10)  # Lower FPS for better performance
            
            # Load YOLO model
            self.yolo_model = YOLO('yolo11n.pt')
            
            # Create perspective transform for path detection
            self.perspective_transform_matrix = cv2.getPerspectiveTransform(IMG_PATH_SRC_PTS, IMG_PATH_DST_PTS)
            
            logging.info("Camera, YOLO model, and perspective transform initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize camera/YOLO: {e}")
            self.camera = None
            self.yolo_model = None

    def run(self):
        """Main control loop for the multi-task delivery mission."""
        logging.info("Starting D* Lite mission controller.")
        self.esp32_bridge.start()
        self.tts_manager.speak('STARTUP')
        
        mission_running = True
        try:
            while mission_running:
                # 1. Update robot's current position from odometry
                self.update_position_tracking()

                # 2. Check for obstacles and other visual cues
                self.process_vision()
                
                # 3. Core State Machine Logic
                if self.obstacle_detected and self.state not in ["AVOIDING", "TURNING_180"]:
                    self.state = "AVOIDING"
                    self.obstacle_detected = False
                
                if self.state == "AVOIDING":
                    self.handle_obstacle_and_replan()
                elif self.state == "TURNING_180":
                    self.execute_180_turn_state()
                else:
                    self.manage_mission_state()
                
                # 4. Check if the entire mission is complete
                if self.state == "MISSION_COMPLETE":
                    mission_running = False

                time.sleep(0.05) # 20Hz control loop
                
        except KeyboardInterrupt:
            logging.info("Stopping...")
        finally:
            self.tts_manager.speak('SHUTDOWN')
            self.stop()
    
    def manage_mission_state(self):
        """Handles the multi-box collection and delivery mission."""
        # If we are in a navigation state, follow the path
        if self.state in ["NAVIGATING_TO_PICKUP", "NAVIGATING_TO_DROPOFF", "AT_INTERSECTION"]:
            self.navigate_path()
            return

        # Handle mission states
        if self.state == "STARTING_MISSION":
            print("=" * 60)
            print("MULTI-BOX DELIVERY MISSION STARTED")
            print(f"Robot starting at: {START_CELL}")
            print(f"Boxes to collect: {self.boxes_to_collect}")
            print(f"Pickup locations: {[f'P{i+1}' for i in range(len(self.pickup_locations))]}")
            print(f"Dropoff locations: {[f'D{i+1}' for i in range(len(self.dropoff_locations))]}")
            print("=" * 60)
            
            # Start by going to the nearest pickup location
            self.state = "PLANNING_TO_PICKUP"

        elif self.state == "PLANNING_TO_PICKUP":
            if self.has_package:
                # Already have a package, go to dropoff instead
                self.state = "PLANNING_TO_DROPOFF"
                return
                
            # Check if all pickups are complete
            if self.current_pickup_index >= len(self.pickup_locations):
                print("All boxes collected! Mission complete.")
                self.state = "MISSION_COMPLETE"
                self.tts_manager.speak('GOAL_REACHED')
                return
                
            # Go to the next pickup in sequence (P1, P2, P3, P4)
            self.current_target_pickup = self.pickup_locations[self.current_pickup_index]
            self.target_cell = self.current_target_pickup
            print(f"Planning route to pickup P{self.current_pickup_index + 1} at {self.current_target_pickup}")
            self.plan_path_to_target()
        
        elif self.state == "PLANNING_TO_DROPOFF":
            if not self.has_package:
                # No package to deliver, go to pickup instead
                self.state = "PLANNING_TO_PICKUP"
                return
                
            # Go to the next dropoff in sequence (D1, D2, D3, D4)
            if self.current_dropoff_index >= len(self.dropoff_locations):
                print("All dropoffs complete!")
                self.state = "MISSION_COMPLETE"
                self.tts_manager.speak('GOAL_REACHED')
                return
                
            self.current_target_dropoff = self.dropoff_locations[self.current_dropoff_index]
            self.target_cell = self.current_target_dropoff
            print(f"Planning route to dropoff D{self.current_dropoff_index + 1} at {self.current_target_dropoff}")
            self.plan_path_to_target()

        elif self.state == "AT_PICKUP":
            if self.current_target_pickup:
                print(f"Reached pickup location P{self.current_pickup_index + 1}. Collecting box...")
                self.stop_motors()
                time.sleep(1.0) # Simulate "picking up" package
                
                # Mark this pickup as collected
                self.collected_boxes.append(self.current_target_pickup)
                self.has_package = True
                self.current_target_pickup = None
                
                print(f"Box P{self.current_pickup_index + 1} collected! Progress: {len(self.collected_boxes)}/{self.boxes_to_collect}")
                self.tts_manager.speak('PACKAGE_DETECTED')
                
                # Move to next pickup for future reference (but go to dropoff first)
                self.current_pickup_index += 1
                
                # Now plan to dropoff
                self.state = "PLANNING_TO_DROPOFF"

        elif self.state == "AT_DROPOFF":
            if self.current_target_dropoff:
                print(f"Reached dropoff location D{self.current_dropoff_index + 1}. Delivering box...")
                self.stop_motors()
                time.sleep(1.0) # Simulate "dropping off" package
                
                # Mark this delivery as complete
                self.delivered_boxes.append(self.current_target_dropoff)
                self.has_package = False
                self.current_target_dropoff = None
                
                print(f"Box delivered to D{self.current_dropoff_index + 1}! Progress: {len(self.delivered_boxes)}/{self.boxes_to_collect}")
                
                # Move to next dropoff for future reference
                self.current_dropoff_index += 1
                
                # Check if all boxes are collected and delivered
                if len(self.collected_boxes) >= self.boxes_to_collect and len(self.delivered_boxes) >= self.boxes_to_collect:
                    print("=" * 60)
                    print("ALL BOXES COLLECTED AND DELIVERED!")
                    print("MISSION SUCCESSFUL!")
                    print("=" * 60)
                    self.state = "MISSION_COMPLETE"
                    self.tts_manager.speak('GOAL_REACHED')
                else:
                    # Continue with next pickup
                    self.state = "PLANNING_TO_PICKUP"
    

    
    def plan_path_to_target(self):
        """Computes a path from the robot's current cell to the target cell."""
        if self.target_cell is None:
            logging.error("Planning called without a target cell.")
            self.state = "MISSION_COMPLETE" # Failsafe
            return

        current_cell_x = int(self.current_position[0] / CELL_WIDTH_M)
        current_cell_y = int(self.current_position[1] / CELL_WIDTH_M)
        
        self.planner.start_cell = (current_cell_x, current_cell_y)
        self.planner.goal_cell = self.target_cell
        self.planner.last_pos = self.planner.start_cell
        self.planner._initialize() # Re-initialize D* for the new goal

        print(f"Planning path from {self.planner.start_cell} to {self.planner.goal_cell}...")
        self.path = self.planner.get_path()

        if self.path:
            self.current_waypoint_idx = 0
            if self.has_package:
                self.state = "NAVIGATING_TO_DROPOFF"
            else:
                self.state = "NAVIGATING_TO_PICKUP"
            print(f"Path found! Length: {len(self.path)} waypoints.")
            self.tts_manager.speak('PATH_FOUND')
        else:
            self.state = "NO_PATH"
            print(f"No path to target {self.target_cell} could be found.")
            self.tts_manager.speak('NO_PATH')
    
    def process_vision(self):
        """Acquires a camera frame and runs all vision-based detection tasks."""
        if not hasattr(self, '_vision_counter'): self._vision_counter = 0
        self._vision_counter += 1
        
        if self._vision_counter < 3: # Reduced from 5 for faster vision updates
            return
            
        self._vision_counter = 0
        if self.camera:
            ret, frame = self.camera.read()
            if ret:
                self.latest_frame = frame.copy()

                # Run YOLO obstacle and package detection
                # This function will now internally set state/flags
                self.detect_objects_from_frame(self.latest_frame)
                
                # Announce package detection only when it's first detected
                if self.package_detected and not self.last_package_state:
                    self.tts_manager.speak('PACKAGE_DETECTED')
                self.last_package_state = self.package_detected

                # Run path shape detection for corner anticipation
                detected_cue = self.detect_path_shape_from_frame(self.latest_frame)

                # Announce a corner only when it is first detected
                if detected_cue != "STRAIGHT" and self.visual_turn_cue == "STRAIGHT":
                    turn_direction = detected_cue.split('_')[1]
                    print(f"\n--- VISUAL CUE: {turn_direction} corner detected ahead. Switching to rotational turning. ---\n")
                
                self.visual_turn_cue = detected_cue

    def navigate_path(self):
        """Follows the line and uses the D* path to make decisions at intersections."""
        if not self.path or self.current_waypoint_idx >= len(self.path):
            # Reached the end of the current path segment (either pickup or dropoff)
            if self.state == "NAVIGATING_TO_DROPOFF":
                self.state = "AT_DROPOFF"
            elif self.state == "NAVIGATING_TO_PICKUP":
                self.state = "AT_PICKUP"
            self.stop_motors()
            return
            
        # Get current waypoint world coordinates
        waypoint_cell = self.path[self.current_waypoint_idx]
        waypoint_world_x = (waypoint_cell[0] + 0.5) * CELL_WIDTH_M
        waypoint_world_y = (waypoint_cell[1] + 0.5) * CELL_WIDTH_M
        
        # Check distance to the waypoint
        dist_to_waypoint = math.sqrt(
            (self.current_position[0] - waypoint_world_x)**2 +
            (self.current_position[1] - waypoint_world_y)**2
        )
        
        # Detect if we are at an intersection using sensors
        sensors = self.line_sensor.read()
        is_intersection = sum(sensors) >= 2 # 2 or 3 sensors active on line

        # If we are close to a waypoint AND the sensors see an intersection, it's time to decide the next turn
        if dist_to_waypoint < self.waypoint_threshold and is_intersection:
            # This is a critical junction. Correct odometry here before turning.
            self.correct_odometry_at_waypoint(waypoint_cell)

            if self.state != "AT_INTERSECTION":
                self.state = "AT_INTERSECTION"
                self.stop_motors()
                time.sleep(0.25) # Pause briefly at the intersection to stabilize

            # Get the direction for the next turn
            turn_direction = self.get_turn_direction_for_next_waypoint()
            print(f"Intersection at {waypoint_cell}. Turning: {turn_direction}")
            
            # Execute the turn, then resume line following
            self.execute_intersection_turn(turn_direction)

            # Move to the next waypoint in the path
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.path):
                # This block will now trigger the AT_PICKUP/AT_DROPOFF state transition above
                return
            
            # After turning, go back to navigating/line-following
            if self.has_package:
                self.state = "NAVIGATING_TO_DROPOFF"
            else:
                self.state = "NAVIGATING_TO_PICKUP"
            self.line_follower_control_loop() # Start moving immediately
            return

        # Default behavior: follow the line
        self.line_follower_control_loop()

    def get_turn_direction_for_next_waypoint(self) -> str:
        """Calculates if the next waypoint requires a left, right, or straight movement."""
        # Ensure we have a "next" waypoint to look at
        if self.current_waypoint_idx + 1 >= len(self.path):
            return "STRAIGHT" # Final approach to goal

        # Current robot pose
        robot_x, robot_y, robot_heading = self.current_position
        
        # Next waypoint in world coordinates
        next_waypoint_cell = self.path[self.current_waypoint_idx + 1]
        next_waypoint_world_x = (next_waypoint_cell[0] + 0.5) * CELL_WIDTH_M
        next_waypoint_world_y = (next_waypoint_cell[1] + 0.5) * CELL_WIDTH_M

        # Calculate the angle from the robot to the next waypoint
        angle_to_target = math.atan2(next_waypoint_world_y - robot_y, next_waypoint_world_x - robot_x)
        
        # Calculate the difference between the robot's heading and the target angle
        heading_error = angle_to_target - robot_heading
        while heading_error > math.pi: heading_error -= 2 * math.pi
        while heading_error < -math.pi: heading_error += 2 * math.pi
        
        # Determine turn direction based on the heading error
        if -math.pi / 4 <= heading_error <= math.pi / 4:
            return "STRAIGHT"
        elif heading_error > math.pi / 4:
            return "LEFT"
        else:
            return "RIGHT"
    
    def execute_intersection_turn(self, direction: str):
        """Executes a sensor-driven turn at an intersection until the line is re-acquired."""
        # Move forward slightly to enter the intersection center
        self.move_omni(vx=self.base_speed, vy=0, omega=0)
        time.sleep(0.2)
        self.stop_motors()

        if direction == "STRAIGHT":
            pass
        else:
            # Corrected omega direction: positive is LEFT (CCW), negative is RIGHT (CW).
            omega = -self.turn_speed if direction == "RIGHT" else self.turn_speed
            if omega < 0:
                self.tts_manager.speak('TURN_RIGHT')
            else:
                self.tts_manager.speak('TURN_LEFT')

            # Perform an arc turn by adding a small forward velocity, instead of a pure pivot.
            # This is more stable and helps the sensors sweep across the new line.
            self.move_omni(vx=self.base_speed * 0.3, vy=0, omega=omega)

            start_time = time.time()
            turn_timeout = 4.0
            line_found = False
            while time.time() - start_time < turn_timeout:
                if self.line_sensor.read()[1]: # Center sensor
                    logging.info(f"Line re-acquired after {direction} turn.")
                    line_found = True
                    break
                time.sleep(0.01)

            if not line_found:
                logging.warning(f"Turn timeout: Failed to find line after turning {direction}.")
                self.stop_motors()
                # --- Wiggle Recovery Maneuver ---
                logging.info("Attempting wiggle recovery to find lost line...")
                wiggle_omega = self.turn_speed * 0.7  # Slower wiggle
                # Wiggle back in the opposite direction of the turn
                self.move_omni(vx=0, vy=0, omega=-omega * 0.7) 
                recovery_time = time.time()
                while time.time() - recovery_time < 1.5: # Wiggle back for up to 1.5s
                    if self.line_sensor.read()[1]:
                        line_found = True
                        logging.info("Line found during recovery wiggle!")
                        break
                    time.sleep(0.01)
                
                if not line_found:
                     logging.error("Wiggle recovery failed. Line is lost.")


        self.stop_motors()
        time.sleep(0.2)
        self.integral = 0.0
        self.last_error = 0.0

    def line_follower_control_loop(self):
        """Controls the robot's motors using a PID controller for line following."""
        sensors = self.line_sensor.read()
        left, center, right = sensors[0], sensors[1], sensors[2]
        
        if not (left or center or right):
            if not self.line_is_lost:
                self.tts_manager.speak('LINE_LOST')
                self.line_is_lost = True
            logging.warning("Line lost. Stopping.")
            self.stop_motors()
            return
            
        if self.line_is_lost:
            logging.info("Line re-acquired.")
            self.line_is_lost = False
        
        error = (right * 1) + (left * -1)
        
        # PID Calculation
        current_time = time.time()
        dt = current_time - self.pid_last_time
        if dt == 0: dt = 1e-6

        self.integral += error * dt
        self.integral = max(min(self.integral, 50), -50)

        derivative = (error - self.last_error) / dt
        
        self.last_error = error
        self.pid_last_time = current_time

        # --- HYBRID CONTROL LOGIC ---
        # Decide whether to turn (rotate) or balance (strafe).
        # We turn if a corner is visually detected ahead OR if the center sensor has left the line.
        # Otherwise, we are on a straight path and can use sideways strafing for fine adjustments.
        is_on_corner = self.visual_turn_cue != "STRAIGHT" or center == 0

        if is_on_corner:
            # --- ROTATIONAL CORRECTION (TURNING ON A CORNER) ---
            # Use rotation (omega) to turn the robot's body to re-align with the line, like a normal car.
            correction_omega = (PID_KP_ROT * error) + (PID_KI_ROT * self.integral) + (PID_KD_ROT * derivative)
            # A positive error (line to the right) requires a right turn (negative omega).
            self.move_omni(vx=self.base_speed * 0.8, vy=0, omega=-correction_omega) # Slow down slightly in turns
        else:
            # --- STRAFE CORRECTION (BALANCING ON STRAIGHTS) ---
            # When on a straight, use sideways strafing (vy) for small, precise adjustments.
            correction_vy = (PID_KP_STRAFE * error) + (PID_KI_STRAFE * self.integral) + (PID_KD_STRAFE * derivative)
            self.move_omni(vx=self.base_speed, vy=correction_vy, omega=0)

    def handle_obstacle_and_replan(self):
        """Stops the robot, updates map, replans, and then initiates a 180-degree turn."""
        self.tts_manager.speak('OBSTACLE')
        print("Obstacle detected! Stopping, replanning, and turning around...")
        self.stop_motors()
        time.sleep(0.2)

        # 1. Estimate obstacle position and update map
        robot_x, robot_y, robot_heading = self.current_position
        obstacle_dist_m = 0.20  # Assume obstacle is 20cm in front

        obstacle_world_x = robot_x + math.cos(robot_heading) * obstacle_dist_m
        obstacle_world_y = robot_y + math.sin(robot_heading) * obstacle_dist_m

        obstacle_cell_x = int(obstacle_world_x / CELL_WIDTH_M)
        obstacle_cell_y = int(obstacle_world_y / CELL_WIDTH_M)

        print(f"Updating map. Obstacle at cell: ({obstacle_cell_x}, {obstacle_cell_y})")
        
        # Update current robot cell before updating obstacle
        current_cell_x = int(self.current_position[0] / CELL_WIDTH_M)
        current_cell_y = int(self.current_position[1] / CELL_WIDTH_M)
        self.planner.start_cell = (current_cell_x, current_cell_y)
        self.planner.last_pos = self.planner.start_cell

        self.planner.update_obstacle(obstacle_cell_x, obstacle_cell_y, is_obstacle=True)

        # 2. Replan the path immediately
        self.tts_manager.speak('REPLANNING')
        print(f"Replanning from new start cell: {self.planner.start_cell}")
        new_path = self.planner.get_path()

        # 3. If a new path is found, start the 180-degree turn.
        if new_path:
            print("New path found! Now turning 180 degrees.")
            self.path = new_path
            self.current_waypoint_idx = 0
            self.state = "TURNING_180"
            self.turn_180_start_time = time.time()
            self.turn_180_initial_heading = self.odometry.heading
        else:
            print("Failed to find a new path around the obstacle.")
            self.state = "NO_PATH"  # Stuck
            self.tts_manager.speak('NO_PATH')
        
        self.obstacle_detected = False # Reset detection flag

    def execute_180_turn_state(self):
        """Handles the odometry-driven 180-degree turn."""
        target_heading = self.turn_180_initial_heading + math.pi
        while target_heading > math.pi: target_heading -= 2 * math.pi
        while target_heading < -math.pi: target_heading += 2 * math.pi

        heading_error = target_heading - self.odometry.heading
        while heading_error > math.pi: heading_error -= 2 * math.pi
        while heading_error < -math.pi: heading_error += 2 * math.pi

        tolerance = math.radians(10)
        turn_timeout = 8.0
        elapsed_time = time.time() - self.turn_180_start_time

        if abs(heading_error) > tolerance and elapsed_time < turn_timeout:
            omega = -self.turn_speed if heading_error > 0 else self.turn_speed
            self.move_omni(vx=0, vy=0, omega=omega)
        else:
            self.stop_motors()
            if elapsed_time >= turn_timeout:
                logging.warning("180-degree turn timed out.")
            else:
                print("180-degree turn completed.")
            
            self.integral = 0.0
            self.last_error = 0.0
            if self.has_package:
                self.state = "NAVIGATING_TO_DROPOFF"
            else:
                self.state = "NAVIGATING_TO_PICKUP"
    
    def detect_objects_from_frame(self, frame):
        """Detect objects using YOLO and set state flags for packages or obstacles."""
        # --- OBJECT DETECTION DISABLED BY USER ---
        return # This will skip all object detection logic for now.
        # -----------------------------------------

        if not self.yolo_model:
            return

        try:
            # Run YOLO detection
            results = self.yolo_model(frame, verbose=False)
            
            # --- Object Categories ---
            # Packages/boxes that we want to pick up and deliver
            package_objects = {
                'box',    
            }
            
            # Objects that ARE obstacles (things robot must avoid)
            obstacle_objects = {
                'bottle', 'cup', 'bowl', 'banana', 'apple', 'orange', 'broccoli',
                'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'potted plant',
                'vase', 'sports ball', 'baseball bat', 'skateboard', 'surfboard', 
                'tennis racket', 'chair', 'dining table', 'couch', 'bed'
            }

            # Objects to ignore (not real obstacles for this maze)
            ignore_objects = {
                'tie', 'necktie'
            }
            
            # Reset detection flags for this frame
            self.package_detected = False
            self.package_box = None
            
            # Check if any objects were detected with sufficient confidence
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        confidence = float(box.conf[0])
                        if confidence > PACKAGE_DETECTION_CONFIDENCE:
                            class_id = int(box.cls[0])
                            class_name = self.yolo_model.names[class_id].lower()
                            
                            # --- Category Check ---
                            if class_name in package_objects:
                                # It's a package we're looking for
                                logging.debug(f"Package detected: {class_name} (confidence: {confidence:.2f})")
                                self.package_detected = True
                                self.package_box = box.xyxy[0].cpu().numpy().astype(int) # Save bbox for overlay
                                
                                # Check if we're at a pickup location and don't have a package yet
                                if not self.has_package and self.is_at_pickup_location():
                                    logging.info("Package detected at pickup location!")
                            
                            elif class_name in obstacle_objects:
                                # It's a defined obstacle. Trigger avoidance.
                                logging.info(f"Obstacle detected: {class_name} (confidence: {confidence:.2f})")
                                if self.state not in ["AVOIDING", "TURNING_180"]:
                                    self.obstacle_detected = True
                            
                            elif class_name in ignore_objects:
                                # It's something to ignore. Do nothing.
                                logging.debug(f"Ignoring non-obstacle: {class_name} (confidence: {confidence:.2f})")
                            
                            else:
                                # It's an unknown object. Treat it as an obstacle for safety.
                                logging.warning(f"Unknown object treated as obstacle: {class_name} (confidence: {confidence:.2f})")
                                if self.state not in ["AVOIDING", "TURNING_180"]:
                                    self.obstacle_detected = True

        except Exception as e:
            logging.debug(f"Object detection error: {e}")
    
    def is_at_pickup_location(self):
        """Check if robot is currently at any pickup location."""
        current_cell_x = int(self.current_position[0] / CELL_WIDTH_M)
        current_cell_y = int(self.current_position[1] / CELL_WIDTH_M)
        current_cell = (current_cell_x, current_cell_y)
        
        # Check if current cell is within tolerance of any pickup cell
        for pickup_cell in PICKUP_CELLS:
            if abs(current_cell[0] - pickup_cell[0]) <= 1 and abs(current_cell[1] - pickup_cell[1]) <= 1:
                return True
        return False

    def update_position_tracking(self):
        """Update robot's position using data from wheel odometry."""
        latest_ticks = self.esp32_bridge.get_encoder_ticks()
        x, y, heading = self.odometry.update(latest_ticks)
        self.current_position = (x, y, heading)
        self.mapper.update_robot_position(x, y, heading)

    def stop_motors(self):
        """Convenience function to stop motors."""
        self.esp32_bridge.send_motor_speeds(0, 0, 0, 0)
    
    def stop(self):
        """Stop the robot and cleanup resources"""
        self.stop_motors()
        self.esp32_bridge.stop()
        self.line_sensor.cleanup()
        
        # Cleanup camera
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        logging.info("Camera resources cleaned up")

    def detect_path_shape_from_frame(self, frame):
        """Analyzes the camera frame to detect upcoming corners or path shape."""
        if self.perspective_transform_matrix is None:
            return "UNKNOWN"

        try:
            # 1. Warp the image to a top-down bird's-eye view
            warped_img = cv2.warpPerspective(frame, self.perspective_transform_matrix, (frame.shape[1], frame.shape[0]))
            
            # 2. Convert to grayscale and find edges
            gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            # 3. Use Hough Line Transform to find lines in the path
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=50, maxLineGap=20)
            
            if lines is None:
                return "STRAIGHT" # Assume straight if no lines detected

            # 4. Analyze the angles of detected lines to infer path shape
            left_lines, right_lines = [], []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0: continue # Skip vertical lines to avoid division by zero
                
                angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
                
                # Filter lines based on their angle to classify them as left or right boundaries
                if angle > -80 and angle < -10:
                    left_lines.append(line)
                elif angle > 10 and angle < 80:
                    right_lines.append(line)

            # 5. Simple heuristic to decide if a corner is ahead.
            # If there's a strong imbalance in detected left vs. right lines, it implies a turn.
            if len(right_lines) > len(left_lines) + 5: # More vertical lines on right -> left turn
                logging.debug("Visual Cue: Possible LEFT turn ahead.")
                return "CORNER_LEFT"
            elif len(left_lines) > len(right_lines) + 5: # More vertical lines on left -> right turn
                logging.debug("Visual Cue: Possible RIGHT turn ahead.")
                return "CORNER_RIGHT"
                
            return "STRAIGHT"

        except Exception as e:
            logging.debug(f"Path shape detection error: {e}")
            return "UNKNOWN"

    def test_motors(self):
        """Simple motor test to verify ESP32 communication"""
        print("Testing motors via ESP32 bridge...")
        if not self.esp32_bridge.connected:
            print("Cannot test motors: ESP32 not connected.")
            return

        print("Forward for 1s..."); self.move_omni(50, 0, 0); time.sleep(1)
        print("Backward for 1s..."); self.move_omni(-50, 0, 0); time.sleep(1)
        print("Strafe Left for 1s..."); self.move_omni(0, -50, 0); time.sleep(1)
        print("Strafe Right for 1s..."); self.move_omni(0, 50, 0); time.sleep(1)
        print("Rotate CW for 1s..."); self.move_omni(0, 0, 50); time.sleep(1)
        print("Rotate CCW for 1s..."); self.move_omni(0, 0, -50); time.sleep(1)
        
        print("Final stop...")
        self.stop_motors()
        print("Motor test complete!")

    def move_omni(self, vx, vy, omega):
        """
        Calculates individual wheel speeds from a velocity vector and sends them to the ESP32.
        :param vx: Forward velocity component (percentage, e.g., -100 to 100)
        :param vy: Sideways (strafe) velocity component (percentage)
        :param omega: Rotational velocity component (percentage)
        """
        R = self.odometry.ROBOT_WIDTH_M / 2 # Assuming width=length
        
        # Inverse kinematics for X-shaped omni-wheel configuration
        v_fl = (vx - vy - R * omega)
        v_fr = (vx + vy + R * omega)
        v_bl = (vx + vy - R * omega)
        v_br = (vx - vy + R * omega)
        
        speeds = [v_fl, v_fr, v_bl, v_br]
        max_v = max(abs(v) for v in speeds)
        if max_v > 100:
            scale = 100 / max_v
            speeds = [s * scale for s in speeds]
            
        self.esp32_bridge.send_motor_speeds(speeds[0], speeds[1], speeds[2], speeds[3])

    def correct_odometry_at_waypoint(self, waypoint_cell: Tuple[int, int]):
        """Snaps the robot's odometry to the precise coordinates of a waypoint cell."""
        logging.info(f"Correcting odometry at waypoint {waypoint_cell}.")

        # Calculate the precise, ideal world coordinates for the center of the waypoint cell
        ideal_world_x = (waypoint_cell[0] + 0.5) * CELL_WIDTH_M
        ideal_world_y = (waypoint_cell[1] + 0.5) * CELL_WIDTH_M

        # Log the correction amount
        x_error = self.odometry.x - ideal_world_x
        y_error = self.odometry.y - ideal_world_y
        logging.info(f"Odometry drift corrected. X_err: {x_error:.4f}m, Y_err: {y_error:.4f}m")

        # Snap the odometry state to the ideal position
        self.odometry.x = ideal_world_x
        self.odometry.y = ideal_world_y

        # Also update the main controller's copy of the position
        self.current_position = (self.odometry.x, self.odometry.y, self.odometry.heading)
        self.mapper.update_robot_position(self.odometry.x, self.odometry.y, self.odometry.heading)
        
        self.state = "NAVIGATING_TO_DROPOFF"
        

class WebVisualization:
    """Flask web app for visualizing robot state with cyberpunk theme"""
    
    def __init__(self, robot):
        self.robot = robot
        self.app = Flask(__name__, template_folder='../../templates', static_folder='../../static')
        self.app.config['SECRET_KEY'] = 'robot_visualization_key'
        
        # Setup routes
        self.setup_routes()
        
        # Data update thread
        self.running = True
        self.update_thread = None
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('navigation.html')
        
        @self.app.route('/api/robot_data')
        def get_robot_data():
            """Get current robot data as JSON"""
            # Get grid visualization with current goal
            current_goal = self.robot.target_cell if hasattr(self.robot, 'target_cell') else None
            grid_img = self.robot.mapper.get_grid_visualization(
                planned_path=self.robot.path, 
                current_goal=current_goal
            )
            _, buffer = cv2.imencode('.png', grid_img)
            grid_base64 = base64.b64encode(buffer).decode('utf-8')

            data = {
                'state': self.robot.state,
                'sensors': self.robot.line_sensor.read(),
                'position': {
                    'x': self.robot.current_position[0],
                    'y': self.robot.current_position[1],
                    'heading': self.robot.current_position[2]
                },
                'grid_image': grid_base64,
                'package_detected': self.robot.package_detected,
                'has_package': getattr(self.robot, 'has_package', False),
                'current_task': len(getattr(self.robot, 'collected_boxes', [])) + 1,
                'total_tasks': getattr(self.robot, 'boxes_to_collect', 4),
                'collected_boxes': len(getattr(self.robot, 'collected_boxes', [])),
                'delivered_boxes': len(getattr(self.robot, 'delivered_boxes', [])),
                'current_pickup_index': getattr(self.robot, 'current_pickup_index', 0),
                'current_dropoff_index': getattr(self.robot, 'current_dropoff_index', 0),
                'pickup_cells': _PICKUP_CELLS_RAW,
                'dropoff_cells': _DROPOFF_CELLS_RAW
            }
            return jsonify(data)
        
        @self.app.route('/video_feed')
        def video_feed():
            """Video streaming route"""
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def generate_frames(self):
        """Generate camera frames for video streaming"""
        while True:
            try:
                if self.robot.latest_frame is not None:
                    frame = self.robot.latest_frame.copy()
                    
                    # Add cyberpunk overlay effects
                    frame = self.add_cyberpunk_overlay(frame)
                    
                    # Encode frame
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # No frame available yet, send placeholder
                    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(black_frame, 'INITIALIZING VISUAL CORTEX...', (100, 240), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', black_frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.1)  # 10 FPS for web streaming
                
            except Exception as e:
                logging.error(f"Video streaming error: {e}")
                time.sleep(1)
    
    def add_cyberpunk_overlay(self, frame):
        """Add cyberpunk-style overlay to camera frame"""
        try:
            height, width = frame.shape[:2]
            
            # Add subtle cyan tint
            overlay = frame.copy()
            overlay[:, :, 1] = np.minimum(overlay[:, :, 1] + 10, 255)  # Slight green boost
            overlay[:, :, 2] = np.minimum(overlay[:, :, 2] + 5, 255)   # Slight red boost
            frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
            
            # Add corner brackets
            bracket_length = 30
            bracket_thickness = 2
            bracket_color = (0, 255, 255)  # Cyan
            
            # Top-left
            cv2.line(frame, (10, 10), (10 + bracket_length, 10), bracket_color, bracket_thickness)
            cv2.line(frame, (10, 10), (10, 10 + bracket_length), bracket_color, bracket_thickness)
            
            # Top-right
            cv2.line(frame, (width-10, 10), (width-10-bracket_length, 10), bracket_color, bracket_thickness)
            cv2.line(frame, (width-10, 10), (width-10, 10 + bracket_length), bracket_color, bracket_thickness)
            
            # Bottom-left
            cv2.line(frame, (10, height-10), (10 + bracket_length, height-10), bracket_color, bracket_thickness)
            cv2.line(frame, (10, height-10), (10, height-10-bracket_length), bracket_color, bracket_thickness)
            
            # Bottom-right
            cv2.line(frame, (width-10, height-10), (width-10-bracket_length, height-10), bracket_color, bracket_thickness)
            cv2.line(frame, (width-10, height-10), (width-10, height-10-bracket_length), bracket_color, bracket_thickness)
            
            # Add sensor overlay
            sensor_text = f"SENSORS: {self.robot.line_sensor.read()}"
            cv2.putText(frame, sensor_text, (10, height-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Add state overlay
            state_text = f"STATE: {self.robot.state}"
            cv2.putText(frame, state_text, (10, height-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Add crosshair in center
            center_x, center_y = width // 2, height // 2
            crosshair_size = 20
            cv2.line(frame, (center_x - crosshair_size, center_y), 
                    (center_x + crosshair_size, center_y), (0, 255, 255), 1)
            cv2.line(frame, (center_x, center_y - crosshair_size), 
                    (center_x, center_y + crosshair_size), (0, 255, 255), 1)
            cv2.circle(frame, (center_x, center_y), crosshair_size, (0, 255, 255), 1)
            
            # Draw detected package bounding box
            if self.robot.package_detected and self.robot.package_box is not None:
                x1, y1, x2, y2 = self.robot.package_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # Cyan box
                cv2.putText(frame, "PACKAGE DETECTED", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            logging.error(f"Overlay error: {e}")
            return frame
    
    def start_web_server(self, host='0.0.0.0', port=5000):
        """Start the Flask web server"""
        try:
            logging.info(f"Starting cyberpunk web dashboard at http://{host}:{port}")
            self.app.run(host=host, port=port, debug=False, threaded=True)
        except Exception as e:
            logging.error(f"Failed to start web server: {e}")
    
    def stop(self):
        """Stop the web server"""
        self.running = False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # The RobotController is now initialized with the ESP32's IP from the config section
    robot = RobotController(ESP32_IP)
    
    # Add simple test mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Running hardware test mode...")
        # Start the bridge to test motors
        robot.esp32_bridge.start()
        print("Waiting for ESP32 connection...")
        time.sleep(5) # Give time for connection
        robot.test_motors()
        robot.stop()
        sys.exit(0)
    
    # Create and start web visualization automatically
    try:
        web_viz = WebVisualization(robot)
        print("Cyberpunk web visualization initialized")
        
        # Start web server automatically in background
        web_thread = threading.Thread(target=web_viz.start_web_server, daemon=True)
        web_thread.start()
        print("Cyberpunk dashboard started at http://0.0.0.0:5000")
    except Exception as e:
        print(f"Could not initialize web visualization: {e}")
        web_viz = None
    
    print("\n" + "="*60)
    print("Robot running with odometry tracking.")
    print("="*60)
    
    print("╔════════════════════════════════════════════════════════════╗")
    print("║            CYBERPUNK ROBOT NAVIGATION MATRIX              ║")
    print("╠════════════════════════════════════════════════════════════╣")
    print("║ > D* Lite pathfinding with dynamic replanning             ║")
    print("║ > Hybrid Navigation (PID Line Following + Waypoints)      ║")
    print("║ > Odometry-based position tracking                        ║")
    print("║ > YOLOv8n obstacle detection with evasive maneuvers      ║")
    print("║ > Live camera feed with cyberpunk overlay effects        ║")
    print("║ > Interactive dashboard at http://192.168.2.20:5000      ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print("SYSTEM STATUS: READY FOR DEPLOYMENT")
    print("Press Ctrl+C to terminate")
    
    # Start robot
    try:
        robot.run() 
    except KeyboardInterrupt:
        print("\nSYSTEM SHUTDOWN INITIATED...")
    finally:
        robot.stop()
        if web_viz:
            web_viz.stop()
        logging.info("Robot shutdown complete.")
