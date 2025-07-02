#!/usr/bin/env python3

import time
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque

class LineMemoryBuffer:
    """
    Line Memory Buffer with Lookahead capability.
    
    Stores recent line positions and robot states to provide predicted navigation
    when the camera temporarily loses the line (e.g., during corners, at endpoints).
    
    Features:
    - Buffers recent line detections with timestamps
    - Dead reckoning using encoder data and robot kinematics
    - Time-based prediction for short-term line loss recovery
    - Adaptive confidence based on buffer age and movement consistency
    """
    
    def __init__(self, buffer_size=20, max_prediction_time=2.0, debug=False):
        self.debug = debug
        self.buffer_size = buffer_size
        self.max_prediction_time = max_prediction_time  # Maximum time to predict without new line data
        
        # Line detection history buffer
        self.line_history = deque(maxlen=self.buffer_size)
        
        # Robot state tracking for dead reckoning
        self.last_robot_state = None
        self.last_encoder_counts = None
        self.prediction_start_time = None
        self.is_predicting = False
        
        # Physical robot parameters (should match main robot config)
        self.wheel_circumference_m = 0.204  # Match encoder_position_tracker.py
        self.ticks_per_revolution = 960
        self.meters_per_tick = self.wheel_circumference_m / self.ticks_per_revolution
        
        # Line prediction parameters
        self.prediction_confidence = 1.0
        self.confidence_decay_rate = 0.7  # How fast confidence decays during prediction
        self.min_confidence_threshold = 0.3  # Minimum confidence to provide predictions
        
        # Movement smoothing
        self.velocity_history = deque(maxlen=5)
        self.angular_velocity_history = deque(maxlen=5)
        
    def update_line_detection(self, line_result: Dict, robot_state: Dict, encoder_counts: Dict = None):
        """
        Update the buffer with new line detection and robot state.
        
        Args:
            line_result: Dictionary from camera line detection
            robot_state: Current robot state (position, direction, speeds)
            encoder_counts: Current encoder tick counts
        """
        current_time = time.time()
        
        # Check if we have a good line detection OR if we're severely off-line
        line_detected = line_result.get('line_detected', False)
        line_confidence = line_result.get('line_confidence', 0)
        line_offset = abs(line_result.get('line_offset', 0))
        
        # Consider line "effectively lost" if severely off-center, even if detected
        SEVERE_OFFSET_THRESHOLD = 0.4  # 40% offset considered "effectively lost"
        effectively_lost = not line_detected or line_confidence < 0.5 or line_offset > SEVERE_OFFSET_THRESHOLD
        
        if line_detected and line_confidence > 0.5 and not effectively_lost:
            # Good line detection - store it and reset prediction mode
            self._store_line_detection(line_result, robot_state, current_time)
            self._reset_prediction_mode()
            
            if self.debug and self.is_predicting:
                print(f"LINE BUFFER: Line reacquired after {current_time - self.prediction_start_time:.1f}s of prediction")
                
        else:
            # Line lost or severely off-center - enter or continue prediction mode
            if not self.is_predicting:
                self._enter_prediction_mode(current_time, robot_state, encoder_counts)
                if self.debug:
                    reason = "line lost" if not line_detected else f"severe offset ({line_offset:.2f})"
                    print(f"LINE BUFFER: Entering prediction mode due to {reason}")
            else:
                self._update_prediction(current_time, robot_state, encoder_counts)
    
    def _store_line_detection(self, line_result: Dict, robot_state: Dict, timestamp: float):
        """Store a good line detection in the history buffer."""
        detection_entry = {
            'timestamp': timestamp,
            'line_center_x': line_result.get('line_center_x', 0),
            'line_offset': line_result.get('line_offset', 0.0),
            'line_confidence': line_result.get('line_confidence', 0.0),
            'robot_position': robot_state.get('position', (0, 0)),
            'robot_direction': robot_state.get('direction', 'N'),
            'robot_speeds': robot_state.get('motor_speeds', {'left': 0, 'right': 0}),
            'is_at_intersection': line_result.get('is_at_intersection', False)
        }
        
        self.line_history.append(detection_entry)
        
        # Update velocity tracking for smoother predictions
        if len(self.line_history) >= 2:
            self._update_velocity_tracking()
        
        if self.debug and len(self.line_history) % 30 == 0:  # Only log every 30th detection to reduce spam
            print(f"LINE BUFFER: Stored detection - offset: {line_result.get('line_offset', 0):.3f}, conf: {line_result.get('line_confidence', 0):.2f}")
    
    def _update_velocity_tracking(self):
        """Update velocity estimates based on recent line detections."""
        if len(self.line_history) < 2:
            return
            
        recent = self.line_history[-1]
        previous = self.line_history[-2]
        
        dt = recent['timestamp'] - previous['timestamp']
        if dt <= 0:
            return
        
        # Calculate linear velocity from robot speeds (simplified)
        current_speeds = recent['robot_speeds']
        avg_speed = (current_speeds.get('left', 0) + current_speeds.get('right', 0)) / 2
        linear_velocity = avg_speed * 0.01  # Convert to approximate m/s
        
        # Calculate angular velocity from speed difference
        speed_diff = current_speeds.get('right', 0) - current_speeds.get('left', 0)
        angular_velocity = speed_diff * 0.01  # Simplified angular velocity
        
        self.velocity_history.append(linear_velocity)
        self.angular_velocity_history.append(angular_velocity)
    
    def _enter_prediction_mode(self, current_time: float, robot_state: Dict, encoder_counts: Dict):
        """Enter prediction mode when line is lost."""
        if len(self.line_history) == 0:
            if self.debug:
                print("LINE BUFFER: Cannot enter prediction mode - no line history")
            return
            
        self.is_predicting = True
        self.prediction_start_time = current_time
        self.prediction_confidence = 1.0
        self.last_robot_state = robot_state.copy()
        self.last_encoder_counts = encoder_counts.copy() if encoder_counts else None
        
        if self.debug:
            print(f"LINE BUFFER: Entered prediction mode at time {current_time:.1f}")
    
    def _update_prediction(self, current_time: float, robot_state: Dict, encoder_counts: Dict):
        """Update prediction based on dead reckoning and time-based extrapolation."""
        if not self.is_predicting or self.prediction_start_time is None:
            return
            
        prediction_duration = current_time - self.prediction_start_time
        
        # Check if we've exceeded maximum prediction time
        if prediction_duration > self.max_prediction_time:
            if self.debug:
                print(f"LINE BUFFER: Prediction timeout after {prediction_duration:.1f}s")
            self._reset_prediction_mode()
            return
        
        # Decay confidence over time
        self.prediction_confidence = max(
            self.min_confidence_threshold,
            1.0 * (self.confidence_decay_rate ** prediction_duration)
        )
        
        # Update position tracking if we have encoder data
        if encoder_counts and self.last_encoder_counts:
            self._update_dead_reckoning(encoder_counts)
    
    def _update_dead_reckoning(self, current_encoder_counts: Dict):
        """Update position estimate using encoder dead reckoning."""
        if not self.last_encoder_counts:
            return
        
        # Calculate movement since last update
        left_delta = current_encoder_counts.get('left', 0) - self.last_encoder_counts.get('left', 0)
        right_delta = current_encoder_counts.get('right', 0) - self.last_encoder_counts.get('right', 0)
        
        # Convert to distance (simplified for differential drive)
        distance_moved = ((left_delta + right_delta) / 2) * self.meters_per_tick
        
        if self.debug and distance_moved > 0.001:  # Only log significant movement
            print(f"LINE BUFFER: Dead reckoning - moved {distance_moved:.3f}m")
        
        # Update encoder counts for next iteration
        self.last_encoder_counts = current_encoder_counts.copy()
    
    def _reset_prediction_mode(self):
        """Reset prediction mode when line is reacquired."""
        self.is_predicting = False
        self.prediction_start_time = None
        self.prediction_confidence = 1.0
        self.last_robot_state = None
        self.last_encoder_counts = None
    
    def get_predicted_line_state(self, frame_width: int = 320) -> Optional[Dict]:
        """
        Get predicted line state when actual line detection fails.
        
        Args:
            frame_width: Camera frame width for offset calculation
            
        Returns:
            Dictionary with predicted line state or None if prediction not available
        """
        if not self.is_predicting or len(self.line_history) == 0:
            return None
        
        if self.prediction_confidence < self.min_confidence_threshold:
            return None
        
        # Get the most recent good line detection
        last_detection = self.line_history[-1]
        current_time = time.time()
        prediction_duration = current_time - self.prediction_start_time
        
        # Base prediction on last known line state
        original_offset = last_detection['line_offset']
        predicted_offset = original_offset
        predicted_center_x = last_detection['line_center_x']
        
        # Check if this is a severe offset situation requiring corrective guidance
        is_severe_offset = abs(original_offset) > 0.4
        
        # Apply movement-based correction if we have velocity data
        if len(self.velocity_history) > 0 and len(self.angular_velocity_history) > 0:
            avg_angular_vel = sum(self.angular_velocity_history) / len(self.angular_velocity_history)
            
            # Predict line offset change based on angular velocity
            # Positive angular velocity (right turn) should increase line offset (line appears more left)
            offset_change = avg_angular_vel * prediction_duration * 0.1  # Scale factor
            predicted_offset = max(-1.0, min(1.0, predicted_offset + offset_change))
        
        # For severe offsets, provide aggressive corrective guidance
        if is_severe_offset:
            # Guide the robot back toward center (0.0 offset)
            correction_strength = min(1.0, prediction_duration * 2.0)  # Increase over time
            corrective_offset = -original_offset * correction_strength * 0.8  # Pull toward center
            predicted_offset = max(-1.0, min(1.0, predicted_offset + corrective_offset))
            
            if self.debug:
                print(f"LINE BUFFER: Applying corrective guidance - original: {original_offset:.3f}, corrected: {predicted_offset:.3f}")
        
        # Update predicted center position
        predicted_center_x = int(frame_width // 2 + predicted_offset * (frame_width // 2))
        
        # Create predicted line result
        predicted_result = {
            'line_detected': True,
            'line_center_x': predicted_center_x,
            'line_offset': predicted_offset,
            'line_confidence': self.prediction_confidence,
            'status': 'predicted_from_memory',
            'is_at_intersection': False,  # Conservative assumption during prediction
            'prediction_duration': prediction_duration,
            'buffer_size': len(self.line_history),
            'last_real_detection_age': current_time - last_detection['timestamp']
        }
        
        if self.debug:
            print(f"LINE BUFFER: Providing prediction - offset: {predicted_offset:.3f}, conf: {self.prediction_confidence:.2f}, duration: {prediction_duration:.1f}s")
        
        return predicted_result
    
    def get_line_trend(self) -> Dict:
        """
        Analyze recent line detections to determine movement trends.
        
        Returns:
            Dictionary with trend analysis (direction, consistency, etc.)
        """
        if len(self.line_history) < 3:
            return {'trend': 'insufficient_data', 'consistency': 0.0}
        
        # Get recent offsets
        recent_offsets = [entry['line_offset'] for entry in list(self.line_history)[-5:]]
        
        # Calculate trend direction
        if len(recent_offsets) >= 2:
            trend_direction = 'left' if recent_offsets[-1] < recent_offsets[0] else 'right'
            if abs(recent_offsets[-1] - recent_offsets[0]) < 0.1:
                trend_direction = 'straight'
        else:
            trend_direction = 'unknown'
        
        # Calculate consistency (how stable the line tracking has been)
        offset_variance = np.var(recent_offsets) if len(recent_offsets) > 1 else 0
        consistency = max(0.0, 1.0 - offset_variance * 5)  # Scale variance to 0-1
        
        return {
            'trend': trend_direction,
            'consistency': consistency,
            'recent_offsets': recent_offsets,
            'average_offset': np.mean(recent_offsets),
            'offset_variance': offset_variance
        }
    
    def get_buffer_status(self) -> Dict:
        """Get current buffer status for debugging and monitoring."""
        return {
            'buffer_size': len(self.line_history),
            'max_buffer_size': self.buffer_size,
            'is_predicting': self.is_predicting,
            'prediction_confidence': self.prediction_confidence,
            'prediction_duration': time.time() - self.prediction_start_time if self.prediction_start_time else 0,
            'last_detection_age': time.time() - self.line_history[-1]['timestamp'] if self.line_history else float('inf'),
            'velocity_samples': len(self.velocity_history),
            'angular_velocity_samples': len(self.angular_velocity_history)
        }
    
    def clear_buffer(self):
        """Clear all buffered data - useful for mission restart."""
        self.line_history.clear()
        self.velocity_history.clear()
        self.angular_velocity_history.clear()
        self._reset_prediction_mode()
        
        if self.debug:
            print("LINE BUFFER: Buffer cleared") 