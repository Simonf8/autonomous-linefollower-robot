# Finite-State Machine (FSM) Obstacle Avoidance Configuration
# Adjust these durations (in seconds) to tune the robot's avoidance behavior

# Phase 1: How long to turn away from the obstacle
FSM_TURN_AWAY_DURATION_S = 1.6  # seconds

# Phase 2: How long to drive forward past the obstacle
FSM_PASS_OBSTACLE_DURATION_S = 3.0  # seconds

# Phase 3: How long to turn back to the line
FSM_RETURN_TO_LINE_DURATION_S = 2.4  # seconds

# Note: To convert these durations into frames, multiply by CAMERA_FPS defined in main.py 