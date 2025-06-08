#!/usr/bin/env python3
"""
Finite-State Machine (FSM) Obstacle Avoidance Module
Uses durations (in seconds) from avoidance_fsm_config.py to drive a 3-phase avoidance behavior:
  1) Turn away from obstacle
  2) Pass obstacle (drive forward)
  3) Return to line (turn back)

Usage:
  fsm = ObstacleAvoidanceFSM()
  # On new obstacle detected:
  fsm.start(obstacle_position)
  # Every frame:
  cmd = fsm.update(obstacle_detected, line_detected, line_offset)
  # cmd will be one of COMMANDS values or None if idle
"""

from avoidance_fsm_config import (
    FSM_TURN_AWAY_DURATION_S,
    FSM_PASS_OBSTACLE_DURATION_S,
    FSM_RETURN_TO_LINE_DURATION_S
)
from main import CAMERA_FPS, COMMANDS, opposite_direction


def _to_frames(seconds):
    """Convert seconds to frame count based on CAMERA_FPS"""
    return max(1, int(seconds * CAMERA_FPS))


class ObstacleAvoidanceFSM:
    """FSM for 3-phase obstacle avoidance"""
    def __init__(self):
        # Calculate frame thresholds for each phase
        self.turn_frames = _to_frames(FSM_TURN_AWAY_DURATION_S)
        self.pass_frames = _to_frames(FSM_PASS_OBSTACLE_DURATION_S)
        self.return_frames = _to_frames(FSM_RETURN_TO_LINE_DURATION_S)
        # Internal state
        self.state = 'idle'       # 'idle', 'turn_away', 'pass_obstacle', 'return_to_line'
        self.counter = 0
        self.side = None          # 'left' or 'right' for avoidance direction

    def reset(self):
        """Return to idle state"""
        self.state = 'idle'
        self.counter = 0
        self.side = None

    def start(self, obstacle_pos):
        """
        Start a new avoidance sequence.
        obstacle_pos: relative position (-1.0 left .. +1.0 right)
        """
        # Determine avoidance side opposite the obstacle
        self.side = 'right' if obstacle_pos < 0 else 'left'
        self.state = 'turn_away'
        self.counter = 0

    def update(self, obstacle_detected, line_detected, line_offset):
        """
        Advance FSM one frame and return the next command.
        obstacle_detected: bool, whether an obstacle is still seen
        line_detected: bool, whether line is detected
        line_offset: float, current line offset (not used in FSM logic)
        Returns a COMMANDS[...] string or None if idle.
        """
        if self.state == 'idle':
            return None

        self.counter += 1

        # Phase 1: Turn away
        if self.state == 'turn_away':
            if self.counter <= self.turn_frames:
                # Gentle avoidance turn
                return COMMANDS['AVOID_LEFT'] if self.side == 'left' else COMMANDS['AVOID_RIGHT']
            # Move to next phase
            self.state = 'pass_obstacle'
            self.counter = 0

        # Phase 2: Pass obstacle (move forward)
        if self.state == 'pass_obstacle':
            if self.counter <= self.pass_frames:
                return COMMANDS['FORWARD']
            # Move to return phase
            self.state = 'return_to_line'
            self.counter = 0

        # Phase 3: Return to line (turn back opposite direction)
        if self.state == 'return_to_line':
            # If line is reacquired early, finish FSM
            if line_detected:
                self.reset()
                return None
            if self.counter <= self.return_frames:
                # Turn back toward line
                opp = opposite_direction(self.side)
                return COMMANDS['AVOID_LEFT'] if opp == 'left' else COMMANDS['AVOID_RIGHT']
            # Done returning
            self.reset()
            return None

        # Fallback: should not reach here
        self.reset()
        return None


# Example usage (if run directly)
if __name__ == '__main__':
    # Quick test stub
    fsm = ObstacleAvoidanceFSM()
    # Simulate obstacle at center (pos=0)
    fsm.start(0.0)
    for frame in range(fsm.turn_frames + fsm.pass_frames + fsm.return_frames + 10):
        cmd = fsm.update(obstacle_detected=True, line_detected=(frame % 20 == 0), line_offset=0.0)
        print(f"Frame {frame:03d}: state={fsm.state}, cmd={cmd}") 