#!/usr/bin/env python3

import time
import math
from typing import Dict, Optional, Tuple
from collections import deque

class AdaptivePIDController:
    """
    Enhanced Adaptive PID Controller for Line Following.
    
    Features:
    - Dynamic gain scheduling based on error magnitude and robot speed
    - Velocity-based feedforward control for smoother operation
    - Anti-windup protection with intelligent integral reset
    - Adaptive deadband based on line confidence
    - Smooth gain transitions to prevent control jumps
    - Memory buffer integration for prediction-based tuning
    """
    
    def __init__(self, 
                 base_kp=1.2, base_ki=0.05, base_kd=0.4,
                 output_limits=(-70, 70),
                 sample_time=0.033,  # ~30 FPS
                 debug=False):
        
        # Base PID gains (optimized for line following)
        self.base_kp = base_kp
        self.base_ki = base_ki
        self.base_kd = base_kd
        
        # Current adaptive gains
        self.kp = base_kp
        self.ki = base_ki
        self.kd = base_kd
        
        # PID state variables
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        self.sample_time = sample_time
        
        # Output limits
        self.output_limits = output_limits
        self.last_output = 0.0
        
        # Adaptive tuning parameters
        self.error_history = deque(maxlen=10)
        self.output_history = deque(maxlen=5)
        self.derivative_history = deque(maxlen=3)
        
        # Speed-based gain scheduling
        self.speed_gain_map = {
            'slow': {'kp_mult': 1.4, 'ki_mult': 0.8, 'kd_mult': 1.2},    # High precision, low speed
            'medium': {'kp_mult': 1.0, 'ki_mult': 1.0, 'kd_mult': 1.0},  # Baseline gains
            'fast': {'kp_mult': 0.7, 'ki_mult': 1.3, 'kd_mult': 0.8}     # Smoother, higher speed
        }
        
        # Error magnitude gain scheduling - Enhanced for aggressive line centering
        self.error_gain_map = {
            'small':  {'kp_mult': 0.9, 'ki_mult': 1.2, 'kd_mult': 0.9},   # Fine corrections
            'medium': {'kp_mult': 1.5, 'ki_mult': 1.0, 'kd_mult': 1.2},  # Stronger response to drift
            'large':  {'kp_mult': 2.5, 'ki_mult': 0.5, 'kd_mult': 1.8},   # Aggressive correction
            'severe': {'kp_mult': 3.5, 'ki_mult': 0.3, 'kd_mult': 2.5}   # Maximum correction for severe offsets
        }
        
        # Anti-windup and integral management
        self.integral_limit = 20.0
        self.integral_decay_rate = 0.95
        self.windup_threshold = 0.8 * max(abs(output_limits[0]), abs(output_limits[1]))
        
        # Feedforward control
        self.feedforward_enabled = True
        self.velocity_gain = 0.3  # Feedforward gain for velocity
        
        # Adaptive deadband
        self.base_deadband = 0.03
        self.confidence_deadband_factor = 0.8  # How much confidence affects deadband
        
        # Smoothing and filtering
        self.gain_smoothing_alpha = 0.3  # Gain transition smoothing
        self.derivative_filter_alpha = 0.7  # Derivative noise filtering
        self.filtered_derivative = 0.0
        
        # Performance tracking
        self.debug = debug
        self.last_performance_metrics = {}
        
        # Prediction integration
        self.prediction_gain_factor = 0.8  # Reduce gains when using predictions
        
    def update(self, error: float, robot_state: Dict = None, line_confidence: float = 1.0, 
               using_prediction: bool = False) -> float:
        """
        Calculate PID output with adaptive tuning.
        
        Args:
            error: Line offset error (-1.0 to 1.0)
            robot_state: Current robot state for adaptive tuning
            line_confidence: Confidence in line detection (0.0 to 1.0)
            using_prediction: Whether we're using predicted line position
        """
        current_time = time.time()
        dt = current_time - self.last_time
        
        # Skip update if dt is too small (prevents division by zero and noise)
        if dt < self.sample_time * 0.5:
            return self.last_output
        
        # Store error in history for adaptive tuning
        self.error_history.append(abs(error))
        
        # Apply adaptive deadband based on line confidence
        adaptive_deadband = self.base_deadband * (2.0 - line_confidence * self.confidence_deadband_factor)
        if abs(error) < adaptive_deadband:
            error = 0.0
        
        # Calculate adaptive gains
        self._update_adaptive_gains(error, robot_state, line_confidence, using_prediction)
        
        # Calculate derivative with filtering
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.derivative_history.append(derivative)
        
        # Apply low-pass filter to derivative for noise reduction
        if len(self.derivative_history) > 1:
            self.filtered_derivative = (self.derivative_filter_alpha * self.filtered_derivative + 
                                      (1 - self.derivative_filter_alpha) * derivative)
        else:
            self.filtered_derivative = derivative
        
        # Calculate integral with anti-windup
        self.integral += error * dt
        
        # Anti-windup: Reset integral if output is saturated and error has same sign as integral
        if abs(self.last_output) > self.windup_threshold:
            if (self.integral > 0 and error > 0) or (self.integral < 0 and error < 0):
                self.integral *= self.integral_decay_rate
        
        # Limit integral to prevent excessive buildup
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        
        # Calculate PID components
        proportional = self.kp * error
        integral_component = self.ki * self.integral
        derivative_component = self.kd * self.filtered_derivative
        
        # Calculate feedforward component for smoother control
        feedforward = 0.0
        if self.feedforward_enabled and robot_state:
            # Simple velocity feedforward based on recent error trend
            if len(self.error_history) >= 2:
                error_velocity = (self.error_history[-1] - self.error_history[-2]) / dt
                feedforward = self.velocity_gain * error_velocity
        
        # Combine all components
        output = proportional + integral_component + derivative_component + feedforward
        
        # Apply output limits
        output = max(self.output_limits[0], min(self.output_limits[1], output))
        
        # Store for next iteration
        self.last_error = error
        self.last_time = current_time
        self.last_output = output
        
        # Store performance metrics
        if self.debug:
            self._update_performance_metrics(error, output, proportional, integral_component, 
                                           derivative_component, feedforward, line_confidence)
        
        return output
    
    def _update_adaptive_gains(self, error: float, robot_state: Dict, 
                             line_confidence: float, using_prediction: bool):
        """Update PID gains based on current conditions."""
        
        # Determine speed category for gain scheduling
        speed_category = self._get_speed_category(robot_state)
        error_category = self._get_error_category(error)
        
        # Get gain multipliers from scheduling maps
        speed_mults = self.speed_gain_map[speed_category]
        error_mults = self.error_gain_map[error_category]
        
        # Calculate new gains
        new_kp = self.base_kp * speed_mults['kp_mult'] * error_mults['kp_mult']
        new_ki = self.base_ki * speed_mults['ki_mult'] * error_mults['ki_mult']
        new_kd = self.base_kd * speed_mults['kd_mult'] * error_mults['kd_mult']
        
        # Apply confidence-based tuning
        confidence_factor = 0.5 + 0.5 * line_confidence  # Scale from 0.5 to 1.0
        new_kp *= confidence_factor
        new_ki *= confidence_factor
        new_kd *= confidence_factor
        
        # Reduce gains when using predictions (less aggressive control)
        if using_prediction:
            new_kp *= self.prediction_gain_factor
            new_ki *= self.prediction_gain_factor
            new_kd *= self.prediction_gain_factor
        
        # Apply smooth gain transitions to prevent control jumps
        alpha = self.gain_smoothing_alpha
        self.kp = alpha * new_kp + (1 - alpha) * self.kp
        self.ki = alpha * new_ki + (1 - alpha) * self.ki
        self.kd = alpha * new_kd + (1 - alpha) * self.kd
        
        if self.debug and time.time() % 2 < 0.1:  # Log every 2 seconds
            print(f"ADAPTIVE PID: kp={self.kp:.2f}, ki={self.ki:.3f}, kd={self.kd:.2f}, "
                  f"speed={speed_category}, error={error_category}, conf={line_confidence:.2f}")
    
    def _get_speed_category(self, robot_state: Dict) -> str:
        """Determine speed category for gain scheduling."""
        if not robot_state:
            return 'medium'
        
        # Get average motor speed as proxy for robot speed
        motor_speeds = robot_state.get('motor_speeds', {'left': 0, 'right': 0})
        avg_speed = abs(motor_speeds.get('left', 0) + motor_speeds.get('right', 0)) / 2
        
        if avg_speed < 20:
            return 'slow'
        elif avg_speed < 40:
            return 'medium'
        else:
            return 'fast'
    
    def _get_error_category(self, error: float) -> str:
        """Determine error magnitude category for gain scheduling."""
        abs_error = abs(error)
        
        if abs_error < 0.05:
            return 'small'
        elif abs_error < 0.15:
            return 'medium'
        elif abs_error < 0.35:
            return 'large'
        else:
            return 'severe'
    
    def _update_performance_metrics(self, error, output, p_term, i_term, d_term, ff_term, confidence):
        """Update performance tracking metrics."""
        self.last_performance_metrics = {
            'error': error,
            'output': output,
            'p_term': p_term,
            'i_term': i_term,
            'd_term': d_term,
            'ff_term': ff_term,
            'kp': self.kp,
            'ki': self.ki,
            'kd': self.kd,
            'integral': self.integral,
            'confidence': confidence,
            'filtered_derivative': self.filtered_derivative
        }
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics for debugging."""
        return self.last_performance_metrics.copy()
    
    def reset(self):
        """Reset the PID controller state."""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_output = 0.0
        self.filtered_derivative = 0.0
        self.last_time = time.time()
        self.error_history.clear()
        self.output_history.clear()
        self.derivative_history.clear()
        
        if self.debug:
            print("ADAPTIVE PID: Controller reset")
    
    def set_tuning(self, kp: float = None, ki: float = None, kd: float = None):
        """Update base PID tuning parameters."""
        if kp is not None:
            self.base_kp = kp
        if ki is not None:
            self.base_ki = ki
        if kd is not None:
            self.base_kd = kd
            
        if self.debug:
            print(f"ADAPTIVE PID: Base tuning updated - kp={self.base_kp}, ki={self.base_ki}, kd={self.base_kd}")
    
    def get_status(self) -> Dict:
        """Get current controller status."""
        return {
            'base_gains': {'kp': self.base_kp, 'ki': self.base_ki, 'kd': self.base_kd},
            'current_gains': {'kp': self.kp, 'ki': self.ki, 'kd': self.kd},
            'state': {'integral': self.integral, 'last_error': self.last_error, 'last_output': self.last_output},
            'history_sizes': {
                'error_history': len(self.error_history),
                'output_history': len(self.output_history),
                'derivative_history': len(self.derivative_history)
            }
        } 