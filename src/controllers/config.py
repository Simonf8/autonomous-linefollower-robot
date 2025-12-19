#!/usr/bin/env python3

# ================================
# FEATURE CONFIGURATION
# ================================
FEATURES = {
    'BOX_MISSION_ENABLED': True,
    'OBJECT_DETECTION_ENABLED': False,
    'OBSTACLE_AVOIDANCE_ENABLED': False,
    'VISION_SYSTEM_ENABLED': False,
    'CAMERA_LINE_FOLLOWING_ENABLED': True, 
    'POSITION_CORRECTION_ENABLED': False,
    'PERFORMANCE_LOGGING_ENABLED': False,
    'DEBUG_VISUALIZATION_ENABLED': True,
    'SMOOTH_CORNERING_ENABLED': True,
    'ADAPTIVE_SPEED_ENABLED': True,
}

# ================================
# ROBOT CONFIGURATION
# ================================
CELL_SIZE_M = 0.064
BASE_SPEED = 50
TURN_SPEED = 40
CORNER_SPEED = 35

MOTOR_TRIMS = {
    'left': 1.0,   
    'right': 0.98, 
}

# Mission Configuration
PICKUP_LOCATIONS = [(20, 14), (18, 14), (16, 14), (14, 14)]
DROPOFF_LOCATIONS = [(0, 0), (2, 0), (4, 0), (6, 0)]

# Grid Map (0 = Empty, 1 = Wall/Obstacle)
MAZE_GRID = [
    [0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0]
]

START_CELL = (13, 2)
START_DIRECTION = 'E'

# Navigation parameters
CELL_CROSSING_TIME_S = 1.5
CORNER_DETECTION_THRESHOLD = 0.05
CORNER_TURN_DURATION = 1
SHARP_CORNER_THRESHOLD = 0.5

# Camera Settings
WEBCAM_INDEX = 1
CAMERA_WIDTH, CAMERA_HEIGHT = 320, 240
CAMERA_FPS = 30
