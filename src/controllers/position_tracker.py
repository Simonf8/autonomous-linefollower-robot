#!/usr/bin/env python3

import math
import time
import logging
import threading
from typing import Tuple, List, Optional, Dict
from collections import deque
from . import OmniWheelOdometry

class OmniWheelOdometry:
    """Enhanced odometry system for omni-wheel robots with improved accuracy and slip compensation."""
    
    def __init__(self, initial_pose: Tuple[float, float, float], 
                 pulses_per_rev: int, wheel_diameter: float, 
                 robot_width: float, robot_length: float,
                 slip_factors: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)):
        """
        Initialize enhanced odometry system.
        
        Args:
            initial_pose: (x, y, heading) starting position
            pulses_per_rev: Encoder pulses per wheel revolution
            wheel_diameter: Wheel diameter in meters
            robot_width: Robot width in meters (between left and right wheels)
            robot_length: Robot length in meters (between front and rear wheels)
            slip_factors: Compensation factors for wheel slip (FL, FR, BL, BR)
        """
        # Physical constants
        self.PULSES_PER_REV = pulses_per_rev
        self.WHEEL_DIAMETER_M = wheel_diameter
        self.WHEEL_CIRCUMFERENCE_M = math.pi * self.WHEEL_DIAMETER_M
        self.DISTANCE_PER_PULSE = self.WHEEL_CIRCUMFERENCE_M / self.PULSES_PER_REV
        self.ROBOT_WIDTH_M = robot_width
        self.ROBOT_LENGTH_M = robot_length
        
        # Slip compensation factors
        self.slip_factors = list(slip_factors)
        
        # State variables
        self.x, self.y, self.heading = initial_pose
        self.prev_ticks = [0, 0, 0, 0]  # FL, FR, BL, BR
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Enhanced velocity tracking with filtering
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.angular_velocity = 0.0
        self.last_update_time = time.time()
        
        # Velocity history for filtering (reduces encoder noise)
        self.velocity_history_size = 5
        self.vx_history = deque(maxlen=self.velocity_history_size)
        self.vy_history = deque(maxlen=self.velocity_history_size)
        self.omega_history = deque(maxlen=self.velocity_history_size)
        
        # Motion statistics for omni-wheel analysis
        self.total_distance = 0.0
        self.total_strafe_distance = 0.0
        self.total_rotation = 0.0
        
        # Position uncertainty tracking
        self.position_uncertainty = 0.0
        self.heading_uncertainty = 0.0
    
    def update(self, current_ticks: List[int], dt: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Update robot pose with enhanced accuracy and slip compensation.
        
        Args:
            current_ticks: Current total encoder ticks [FL, FR, BL, BR]
            dt: Time delta (auto-calculated if None)
            
        Returns:
            Updated pose (x, y, heading)
        """
        with self._lock:
            current_time = time.time()
            if dt is None:
                dt = current_time - self.last_update_time
            
            # Prevent division by zero and handle very small time steps
            if dt <= 0.001:
                dt = 0.001
            
            # Calculate tick differences with slip compensation
            delta_ticks = [(curr - prev) * slip for curr, prev, slip in 
                          zip(current_ticks, self.prev_ticks, self.slip_factors)]
            self.prev_ticks = current_ticks.copy()
            
            # Convert to distances
            wheel_distances = [d * self.DISTANCE_PER_PULSE for d in delta_ticks]
            fl_dist, fr_dist, bl_dist, br_dist = wheel_distances
            
            # Enhanced forward kinematics for X-shaped omni-wheel configuration
            # The distance from the center to any wheel
            L = self.ROBOT_WIDTH_M / 2.0
            W = self.ROBOT_LENGTH_M / 2.0
            
            # Local velocity components (robot frame)
            vx_local = ( fl_dist + fr_dist + bl_dist + br_dist) / 4.0
            vy_local = (-fl_dist + fr_dist - bl_dist + br_dist) / 4.0
            
            # Rotational velocity (corrected calculation)
            delta_heading = (-fl_dist + fr_dist + bl_dist - br_dist) / (4.0 * (L + W))
            
            # Apply velocity filtering to reduce encoder noise
            self.vx_history.append(vx_local / dt)
            self.vy_history.append(vy_local / dt)
            self.omega_history.append(delta_heading / dt)
            
            # Filtered velocities
            self.velocity_x = sum(self.vx_history) / len(self.vx_history)
            self.velocity_y = sum(self.vy_history) / len(self.vy_history)
            self.angular_velocity = sum(self.omega_history) / len(self.omega_history)
            
            # Update pose using improved integration (Runge-Kutta-like approach)
            mid_heading = self.heading + delta_heading / 2.0
            
            # Transform local movement to global coordinates
            cos_mid = math.cos(mid_heading)
            sin_mid = math.sin(mid_heading)
            
            delta_x = vx_local * cos_mid - vy_local * sin_mid
            delta_y = vx_local * sin_mid + vy_local * cos_mid
            
            # Update position
            self.x += delta_x
            self.y += delta_y
            self.heading += delta_heading
            
            # Normalize heading to [-π, π]
            self.heading = self._normalize_angle(self.heading)
            
            # Update motion statistics
            distance_moved = math.sqrt(delta_x**2 + delta_y**2)
            strafe_component = abs(vy_local)
            
            self.total_distance += distance_moved
            self.total_strafe_distance += strafe_component
            self.total_rotation += abs(delta_heading)
            
            # Update uncertainty estimates
            self._update_uncertainty(distance_moved, abs(delta_heading), dt)
            
            self.last_update_time = current_time
            
            return self.x, self.y, self.heading
    
    def _update_uncertainty(self, distance_moved: float, rotation_moved: float, dt: float):
        """Update position and heading uncertainty estimates."""
        # Uncertainty grows with movement and time
        distance_factor = distance_moved * 0.01  # 1% distance error
        rotation_factor = rotation_moved * 0.02  # 2% rotation error
        time_factor = dt * 0.001  # Small drift over time
        
        self.position_uncertainty += distance_factor + time_factor
        self.heading_uncertainty += rotation_factor + time_factor
        
        # Cap maximum uncertainty
        self.position_uncertainty = min(self.position_uncertainty, 0.5)  # 50cm max
        self.heading_uncertainty = min(self.heading_uncertainty, 0.1)    # ~6 degrees max
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to range [-π, π]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def get_pose_with_uncertainty(self) -> Tuple[float, float, float, float, float]:
        """Get current pose with uncertainty estimates."""
        with self._lock:
            return (self.x, self.y, self.heading, 
                   self.position_uncertainty, self.heading_uncertainty)
    
    def get_motion_statistics(self) -> Dict[str, float]:
        """Get detailed motion statistics for omni-wheel analysis."""
        with self._lock:
            strafe_ratio = self.total_strafe_distance / max(self.total_distance, 0.001)
            return {
                'total_distance': self.total_distance,
                'total_strafe_distance': self.total_strafe_distance,
                'total_rotation': self.total_rotation,
                'strafe_ratio': strafe_ratio,
                'position_uncertainty': self.position_uncertainty,
                'heading_uncertainty': self.heading_uncertainty,
                'velocity_x': self.velocity_x,
                'velocity_y': self.velocity_y,
                'angular_velocity': self.angular_velocity
            }
    
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
    
    def get_velocity_magnitude(self) -> float:
        """Get current speed magnitude."""
        with self._lock:
            return math.sqrt(self.velocity_x**2 + self.velocity_y**2)
    
    def is_moving(self, threshold: float = 0.01) -> bool:
        """Check if robot is currently moving."""
        speed = self.get_velocity_magnitude()
        return speed > threshold
    
    def is_strafing(self, threshold: float = 0.5) -> bool:
        """Check if robot is primarily strafing vs moving forward."""
        with self._lock:
            if abs(self.velocity_x) < 0.01:  # Avoid division by zero
                return abs(self.velocity_y) > 0.01
            strafe_ratio = abs(self.velocity_y) / abs(self.velocity_x)
            return strafe_ratio > threshold
    
    def calibrate_slip_factors(self, expected_distances: List[float], 
                              actual_distances: List[float]):
        """
        Calibrate slip factors based on measured vs expected distances.
        
        Args:
            expected_distances: Expected distances for each wheel [FL, FR, BL, BR]
            actual_distances: Measured distances for each wheel
        """
        with self._lock:
            for i in range(4):
                if expected_distances[i] > 0:
                    self.slip_factors[i] = actual_distances[i] / expected_distances[i]
    
    def reset_uncertainty(self):
        """Reset uncertainty estimates (call after position correction)."""
        with self._lock:
            self.position_uncertainty = 0.0
            self.heading_uncertainty = 0.0
    
    def reset_pose(self, new_pose: Tuple[float, float, float]):
        """Reset odometry to new pose."""
        self.set_pose(*new_pose)
        with self._lock:
            self.prev_ticks = [0, 0, 0, 0]
            self.velocity_x = 0.0
            self.velocity_y = 0.0
            self.angular_velocity = 0.0
            # Clear velocity history
            self.vx_history.clear()
            self.vy_history.clear()
            self.omega_history.clear()
            # Reset motion statistics
            self.total_distance = 0.0
            self.total_strafe_distance = 0.0
            self.total_rotation = 0.0
            # Reset uncertainty
            self.position_uncertainty = 0.0
            self.heading_uncertainty = 0.0
    
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
            
            # Reset uncertainty after correction
            self.position_uncertainty = 0.0

class PositionTracker:
    """Enhanced position tracker with omni-wheel specific features and movement analysis."""
    
    def __init__(self, odometry: OmniWheelOdometry, cell_size_m: float = 0.025):
        """
        Initialize enhanced position tracker.
        
        Args:
            odometry: OmniWheelOdometry instance
            cell_size_m: Grid cell width in meters
        """
        self.odometry = odometry
        self.cell_size_m = cell_size_m
        
        # Enhanced path tracking
        self.path_history = deque(maxlen=1500)  # Store more history
        self.velocity_history = deque(maxlen=100)
        
        # Movement pattern analysis
        self.movement_patterns = {
            'forward_time': 0.0,
            'strafe_time': 0.0, 
            'rotation_time': 0.0,
            'stationary_time': 0.0
        }
        self.last_movement_analysis = time.time()
        
        # Waypoint management with tighter threshold for omni-wheels
        self.waypoint_threshold = 0.08  # meters - tighter for precise omni movement
        
        # Performance metrics
        self.efficiency_score = 1.0  # How direct the path is
        self.smoothness_score = 1.0  # How smooth the motion is
        
        # Statistics
        self.total_distance = 0.0
        self.last_position = self.odometry.get_position()
        
    def update(self, encoder_ticks: List[int]) -> Tuple[float, float, float]:
        """Update tracking with movement pattern analysis."""
        # Update odometry
        pose = self.odometry.update(encoder_ticks)
        x, y, heading = pose
        
        # Update path and velocity history
        current_time = time.time()
        self.path_history.append((x, y, heading, current_time))
        
        velocities = self.odometry.get_velocities()
        self.velocity_history.append((*velocities, current_time))
        
        # Update distance tracking
        self._update_distance_tracking(x, y)
        
        # Analyze movement patterns
        self._analyze_movement_patterns()
        
        # Update performance metrics
        self._update_performance_metrics()
        
        return pose
    
    def _analyze_movement_patterns(self):
        """Analyze robot movement patterns for omni-wheel diagnostics."""
        current_time = time.time()
        dt = current_time - self.last_movement_analysis
        
        if dt < 0.1:  # Update every 100ms
            return
        
        vx, vy, omega = self.odometry.get_velocities()
        
        # Classify current movement
        if self.odometry.is_moving(0.02):
            if abs(omega) > 0.1:  # Rotating
                self.movement_patterns['rotation_time'] += dt
            elif self.odometry.is_strafing(0.7):  # Mostly strafing
                self.movement_patterns['strafe_time'] += dt
            else:  # Mostly forward/backward
                self.movement_patterns['forward_time'] += dt
        else:
            self.movement_patterns['stationary_time'] += dt
        
        self.last_movement_analysis = current_time
    
    def _update_performance_metrics(self):
        """Update efficiency and smoothness scores."""
        if len(self.path_history) < 10:
            return
        
        # Calculate path efficiency (straight line vs actual path)
        recent_path = list(self.path_history)[-10:]
        start_pos = (recent_path[0][0], recent_path[0][1])
        end_pos = (recent_path[-1][0], recent_path[-1][1])
        
        straight_distance = math.sqrt((end_pos[0] - start_pos[0])**2 + 
                                     (end_pos[1] - start_pos[1])**2)
        
        actual_distance = 0.0
        for i in range(1, len(recent_path)):
            dx = recent_path[i][0] - recent_path[i-1][0]
            dy = recent_path[i][1] - recent_path[i-1][1]
            actual_distance += math.sqrt(dx**2 + dy**2)
        
        if actual_distance > 0.001:
            self.efficiency_score = straight_distance / actual_distance
        
        # Calculate smoothness (velocity consistency)
        if len(self.velocity_history) >= 5:
            recent_velocities = list(self.velocity_history)[-5:]
            velocity_variations = []
            
            for i in range(1, len(recent_velocities)):
                prev_vel = recent_velocities[i-1]
                curr_vel = recent_velocities[i]
                
                accel_x = abs(curr_vel[0] - prev_vel[0])
                accel_y = abs(curr_vel[1] - prev_vel[1])
                total_accel = math.sqrt(accel_x**2 + accel_y**2)
                velocity_variations.append(total_accel)
            
            if velocity_variations:
                avg_variation = sum(velocity_variations) / len(velocity_variations)
                self.smoothness_score = max(0.1, 1.0 - avg_variation * 10)
    
    def _update_distance_tracking(self, x: float, y: float):
        """Update total distance traveled."""
        current_pos = (x, y)
        distance_delta = math.sqrt((x - self.last_position[0])**2 + (y - self.last_position[1])**2)
        self.total_distance += distance_delta
        self.last_position = current_pos
    
    def get_current_cell(self) -> Tuple[int, int]:
        """Get current grid cell coordinates."""
        x, y = self.odometry.get_position()
        cell_x = int(x / self.cell_size_m)
        cell_y = int(y / self.cell_size_m)
        return (cell_x, cell_y)
    
    def get_distance_to_point(self, target_x: float, target_y: float) -> float:
        """Calculate distance to a target point."""
        x, y = self.odometry.get_position()
        return math.sqrt((target_x - x)**2 + (target_y - y)**2)
    
    def get_distance_to_cell(self, cell_x: int, cell_y: int) -> float:
        """Calculate distance to center of target cell."""
        target_x = (cell_x + 0.5) * self.cell_size_m
        target_y = (cell_y + 0.5) * self.cell_size_m
        return self.get_distance_to_point(target_x, target_y)
    
    def is_at_waypoint(self, waypoint_x: float, waypoint_y: float, threshold: float = None) -> bool:
        """Check if robot is at a specific waypoint."""
        if threshold is None:
            threshold = self.waypoint_threshold
        return self.get_distance_to_point(waypoint_x, waypoint_y) <= threshold
    
    def is_at_cell(self, cell_x: int, cell_y: int, tolerance_m: float = 0.03) -> bool:
        """Check if robot is at a specific cell within a tolerance."""
        target_x = (cell_x + 0.5) * self.cell_size_m
        target_y = (cell_y + 0.5) * self.cell_size_m
        
        current_x, current_y, _ = self.odometry.get_position()
        
        return self.get_distance_to_point(target_x, target_y) <= tolerance_m
    
    def correct_at_waypoint(self, waypoint_cell: Tuple[int, int]):
        """Correct odometry to snap to exact waypoint coordinates."""
        ideal_x = (waypoint_cell[0] + 0.5) * self.cell_size_m
        ideal_y = (waypoint_cell[1] + 0.5) * self.cell_size_m
        
        # Get current position for error calculation
        current_x, current_y = self.odometry.get_position()
        error_x = current_x - ideal_x
        error_y = current_y - ideal_y
        error_distance = math.sqrt(error_x**2 + error_y**2)
        
        # Correct position
        self.odometry.correct_position(ideal_x, ideal_y)
        
        # Odometry corrected at waypoint
    
    def get_enhanced_status(self) -> Dict:
        """Get comprehensive tracking status with omni-wheel specific data."""
        pose = self.odometry.get_pose_with_uncertainty()
        motion_stats = self.odometry.get_motion_statistics()
        
        return {
            'pose': {'x': pose[0], 'y': pose[1], 'heading': pose[2]},
            'uncertainty': {'position': pose[3], 'heading': pose[4]},
            'motion_stats': motion_stats,
            'movement_patterns': self.movement_patterns.copy(),
            'performance': {
                'efficiency_score': self.efficiency_score,
                'smoothness_score': self.smoothness_score
            },
            'is_moving': self.odometry.is_moving(),
            'is_strafing': self.odometry.is_strafing(),
            'current_cell': self.get_current_cell()
        }
    
    def get_strafe_efficiency(self) -> float:
        """Get how efficiently the robot uses omni-wheel strafing."""
        total_movement = self.movement_patterns['forward_time'] + self.movement_patterns['strafe_time']
        if total_movement > 0:
            return self.movement_patterns['strafe_time'] / total_movement
        return 0.0
    
    def correct_position_with_confidence(self, true_x: float, true_y: float, 
                                       confidence: float = 1.0):
        """Correct position with confidence weighting."""
        current_x, current_y = self.odometry.get_position()
        
        # Weighted correction based on confidence
        corrected_x = current_x + confidence * (true_x - current_x)
        corrected_y = current_y + confidence * (true_y - current_y)
        
        self.odometry.set_pose(corrected_x, corrected_y, self.odometry.get_heading())
        
        # Reduce uncertainty after correction
        if confidence > 0.8:
            self.odometry.reset_uncertainty()
    
    def get_path_history(self, max_points: int = None) -> List[Tuple[float, float]]:
        """Get recent path history."""
        if max_points is None:
            return [(x, y) for x, y, h, t in self.path_history]
        else:
            recent_points = list(self.path_history)[-max_points:] if max_points <= len(self.path_history) else list(self.path_history)
            return [(x, y) for x, y, h, t in recent_points]
    
    def get_tracking_statistics(self) -> dict:
        """Get comprehensive position tracking statistics with omni-wheel analysis."""
        pose_with_uncertainty = self.odometry.get_pose_with_uncertainty()
        motion_stats = self.odometry.get_motion_statistics()
        
        return {
            'current_position': (pose_with_uncertainty[0], pose_with_uncertainty[1]),
            'current_heading_deg': math.degrees(pose_with_uncertainty[2]),
            'current_cell': self.get_current_cell(),
            'velocities': (motion_stats['velocity_x'], motion_stats['velocity_y'], motion_stats['angular_velocity']),
            'uncertainty': {
                'position_m': pose_with_uncertainty[3],
                'heading_deg': math.degrees(pose_with_uncertainty[4])
            },
            'motion_analysis': {
                'total_distance': motion_stats['total_distance'],
                'strafe_distance': motion_stats['total_strafe_distance'], 
                'strafe_ratio': motion_stats['strafe_ratio'],
                'total_rotation_rad': motion_stats['total_rotation']
            },
            'movement_patterns': self.movement_patterns.copy(),
            'performance': {
                'efficiency_score': self.efficiency_score,
                'smoothness_score': self.smoothness_score,
                'strafe_efficiency': self.get_strafe_efficiency()
            },
            'status': {
                'is_moving': self.odometry.is_moving(),
                'is_strafing': self.odometry.is_strafing(),
                'speed_magnitude': self.odometry.get_velocity_magnitude()
            },
            'path_points': len(self.path_history)
        }
    
    def clear_path_history(self):
        """Clear path history."""
        self.path_history.clear()
    
    def set_waypoint_threshold(self, threshold: float):
        """Set waypoint detection threshold."""
        self.waypoint_threshold = threshold
    
    def get_pose(self) -> Tuple[float, float, float]:
        """Get current robot pose (x, y, heading) - delegates to odometry."""
        return self.odometry.get_pose() 