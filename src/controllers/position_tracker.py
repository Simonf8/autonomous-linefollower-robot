#!/usr/bin/env python3

import math
import time
import logging
import threading
from typing import Tuple, List, Optional

class OmniWheelOdometry:
    """Calculates robot position and heading using omni-wheel encoder data."""
    
    def __init__(self, initial_pose: Tuple[float, float, float], 
                 pulses_per_rev: int, wheel_diameter: float, 
                 robot_width: float, robot_length: float):
        """
        Initialize odometry system.
        
        Args:
            initial_pose: (x, y, heading) starting position
            pulses_per_rev: Encoder pulses per wheel revolution
            wheel_diameter: Wheel diameter in meters
            robot_width: Robot width in meters
            robot_length: Robot length in meters
        """
        # Physical constants
        self.PULSES_PER_REV = pulses_per_rev
        self.WHEEL_DIAMETER_M = wheel_diameter
        self.WHEEL_CIRCUMFERENCE_M = math.pi * self.WHEEL_DIAMETER_M
        self.DISTANCE_PER_PULSE = self.WHEEL_CIRCUMFERENCE_M / self.PULSES_PER_REV
        self.ROBOT_WIDTH_M = robot_width
        self.ROBOT_LENGTH_M = robot_length
        
        # State variables
        self.x, self.y, self.heading = initial_pose
        self.prev_ticks = [0, 0, 0, 0]  # FL, FR, BL, BR
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Velocity tracking
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.angular_velocity = 0.0
        self.last_update_time = time.time()
        
        # Odometry initialized
    
    def update(self, current_ticks: List[int]) -> Tuple[float, float, float]:
        """
        Update robot pose based on new encoder tick counts.
        
        Args:
            current_ticks: Current total encoder ticks [FL, FR, BL, BR]
            
        Returns:
            Updated pose (x, y, heading)
        """
        with self._lock:
            current_time = time.time()
            dt = current_time - self.last_update_time
            
            # Calculate tick differences
            delta_ticks = [(curr - prev) for curr, prev in zip(current_ticks, self.prev_ticks)]
            self.prev_ticks = current_ticks.copy()
            
            # Convert to distances
            fl_dist, fr_dist, bl_dist, br_dist = [d * self.DISTANCE_PER_PULSE for d in delta_ticks]
            
            # Forward kinematics for X-shaped omni-wheel configuration
            # Local velocity components
            vx_local = (fl_dist + fr_dist + bl_dist + br_dist) / 4.0   # Forward/backward
            vy_local = (-fl_dist + fr_dist - bl_dist + br_dist) / 4.0  # Left/right strafe
            
            # Change in heading (rotation)
            delta_heading = (-fl_dist + fr_dist + bl_dist - br_dist) / (2 * (self.ROBOT_WIDTH_M + self.ROBOT_LENGTH_M))
            
            # Update velocities for monitoring
            if dt > 0:
                self.velocity_x = vx_local / dt
                self.velocity_y = vy_local / dt
                self.angular_velocity = delta_heading / dt
            
            # Update pose using average heading for better accuracy
            avg_heading = self.heading + delta_heading / 2.0
            
            # Transform local movement to global coordinates
            self.x += vx_local * math.cos(avg_heading) - vy_local * math.sin(avg_heading)
            self.y += vx_local * math.sin(avg_heading) + vy_local * math.cos(avg_heading)
            self.heading += delta_heading
            
            # Normalize heading to [-π, π]
            self.heading = self._normalize_angle(self.heading)
            
            self.last_update_time = current_time
            
            return self.x, self.y, self.heading
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to range [-π, π]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def get_pose(self) -> Tuple[float, float, float]:
        """Get current robot pose in thread-safe manner."""
        with self._lock:
            return (self.x, self.y, self.heading)
    
    def get_position(self) -> Tuple[float, float]:
        """Get current robot position (x, y)."""
        with self._lock:
            return (self.x, self.y)
    
    def get_heading(self) -> float:
        """Get current robot heading in radians."""
        with self._lock:
            return self.heading
    
    def get_heading_degrees(self) -> float:
        """Get current robot heading in degrees."""
        with self._lock:
            return math.degrees(self.heading)
    
    def get_velocities(self) -> Tuple[float, float, float]:
        """Get current velocities (vx, vy, omega)."""
        with self._lock:
            return (self.velocity_x, self.velocity_y, self.angular_velocity)
    
    def set_pose(self, x: float, y: float, heading: float):
        """
        Manually set robot pose (for corrections).
        
        Args:
            x: X coordinate in meters
            y: Y coordinate in meters  
            heading: Heading in radians
        """
        with self._lock:
            self.x = x
            self.y = y
            self.heading = self._normalize_angle(heading)
    
    def reset_pose(self, new_pose: Tuple[float, float, float]):
        """Reset odometry to new pose."""
        self.set_pose(*new_pose)
        with self._lock:
            self.prev_ticks = [0, 0, 0, 0]
            self.velocity_x = 0.0
            self.velocity_y = 0.0
            self.angular_velocity = 0.0
    
    def get_distance_traveled(self) -> float:
        """Calculate total distance traveled since last reset."""
        # This is an approximation - for exact calculation, we'd need to track the path
        return math.sqrt(self.x**2 + self.y**2)
    
    def correct_position(self, true_x: float, true_y: float):
        """
        Correct position while maintaining heading.
        
        Args:
            true_x: Corrected X position
            true_y: Corrected Y position
        """
        with self._lock:
            old_x, old_y = self.x, self.y
            self.x = true_x
            self.y = true_y
            
            error_x = abs(old_x - true_x)
            error_y = abs(old_y - true_y)
            error_distance = math.sqrt(error_x**2 + error_y**2)
            
            # Position corrected

class PositionTracker:
    """High-level position tracking with path history and waypoint management."""
    
    def __init__(self, odometry: OmniWheelOdometry, cell_width_m: float = 0.025):
        """
        Initialize position tracker.
        
        Args:
            odometry: OmniWheelOdometry instance
            cell_width_m: Grid cell width in meters
        """
        self.odometry = odometry
        self.cell_width_m = cell_width_m
        
        # Path tracking
        self.path_history = []
        self.max_history_length = 1000
        
        # Waypoint management
        self.current_waypoint = None
        self.waypoint_threshold = 0.12  # meters
        
        # Statistics
        self.total_distance = 0.0
        self.last_position = self.odometry.get_position()
        
    def update(self, encoder_ticks: List[int]):
        """Update position tracking with new encoder data."""
        # Update odometry
        x, y, heading = self.odometry.update(encoder_ticks)
        
        # Update path history
        self._update_path_history(x, y)
        
        # Update distance tracking
        self._update_distance_tracking(x, y)
        
        return (x, y, heading)
    
    def _update_path_history(self, x: float, y: float):
        """Update path history with current position."""
        self.path_history.append((x, y, time.time()))
        
        # Limit history length
        if len(self.path_history) > self.max_history_length:
            self.path_history.pop(0)
    
    def _update_distance_tracking(self, x: float, y: float):
        """Update total distance traveled."""
        current_pos = (x, y)
        distance_delta = math.sqrt((x - self.last_position[0])**2 + (y - self.last_position[1])**2)
        self.total_distance += distance_delta
        self.last_position = current_pos
    
    def get_current_cell(self) -> Tuple[int, int]:
        """Get current grid cell coordinates."""
        x, y = self.odometry.get_position()
        cell_x = int(x / self.cell_width_m)
        cell_y = int(y / self.cell_width_m)
        return (cell_x, cell_y)
    
    def get_distance_to_point(self, target_x: float, target_y: float) -> float:
        """Calculate distance to a target point."""
        x, y = self.odometry.get_position()
        return math.sqrt((target_x - x)**2 + (target_y - y)**2)
    
    def get_distance_to_cell(self, cell_x: int, cell_y: int) -> float:
        """Calculate distance to center of target cell."""
        target_x = (cell_x + 0.5) * self.cell_width_m
        target_y = (cell_y + 0.5) * self.cell_width_m
        return self.get_distance_to_point(target_x, target_y)
    
    def is_at_waypoint(self, waypoint_x: float, waypoint_y: float, threshold: float = None) -> bool:
        """Check if robot is at a specific waypoint."""
        if threshold is None:
            threshold = self.waypoint_threshold
        return self.get_distance_to_point(waypoint_x, waypoint_y) <= threshold
    
    def is_at_cell(self, cell_x: int, cell_y: int, threshold: float = None) -> bool:
        """Check if robot is at a specific cell."""
        if threshold is None:
            threshold = self.waypoint_threshold
        return self.get_distance_to_cell(cell_x, cell_y) <= threshold
    
    def correct_at_waypoint(self, waypoint_cell: Tuple[int, int]):
        """Correct odometry to snap to exact waypoint coordinates."""
        ideal_x = (waypoint_cell[0] + 0.5) * self.cell_width_m
        ideal_y = (waypoint_cell[1] + 0.5) * self.cell_width_m
        
        # Get current position for error calculation
        current_x, current_y = self.odometry.get_position()
        error_x = current_x - ideal_x
        error_y = current_y - ideal_y
        error_distance = math.sqrt(error_x**2 + error_y**2)
        
        # Correct position
        self.odometry.correct_position(ideal_x, ideal_y)
        
        # Odometry corrected at waypoint
    
    def get_path_history(self, max_points: int = None) -> List[Tuple[float, float]]:
        """Get recent path history."""
        if max_points is None:
            return [(x, y) for x, y, t in self.path_history]
        else:
            return [(x, y) for x, y, t in self.path_history[-max_points:]]
    
    def get_tracking_statistics(self) -> dict:
        """Get position tracking statistics."""
        x, y, heading = self.odometry.get_pose()
        vx, vy, omega = self.odometry.get_velocities()
        
        return {
            'current_position': (x, y),
            'current_heading_deg': math.degrees(heading),
            'current_cell': self.get_current_cell(),
            'velocities': (vx, vy, omega),
            'total_distance': self.total_distance,
            'path_points': len(self.path_history)
        }
    
    def clear_path_history(self):
        """Clear path history."""
        self.path_history.clear()
    
    def set_waypoint_threshold(self, threshold: float):
        """Set waypoint detection threshold."""
        self.waypoint_threshold = threshold 