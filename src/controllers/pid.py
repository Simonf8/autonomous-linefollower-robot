#!/usr/bin/env python3

import time
from typing import Tuple, Optional

class PIDController:
    """
    Generic PID controller implementation for precise control systems.
    """
    
    def __init__(self, kp: float, ki: float, kd: float, 
                 output_limits: Tuple[float, float] = (-100.0, 100.0),
                 integral_limits: Tuple[float, float] = (-50.0, 50.0)):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain  
            kd: Derivative gain
            output_limits: (min, max) output bounds
            integral_limits: (min, max) integral term bounds (prevents windup)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.output_limits = output_limits
        self.integral_limits = integral_limits
        
        # State variables
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = None
        
        # Statistics
        self.error_history = []
        self.max_history = 100
    
    def update(self, error: float, dt: Optional[float] = None) -> float:
        """
        Calculate PID output based on current error.
        
        Args:
            error: Current error value (setpoint - actual)
            dt: Time delta since last update (auto-calculated if None)
            
        Returns:
            PID control output
        """
        current_time = time.time()
        
        # Calculate time delta
        if dt is None:
            if self.last_time is None:
                dt = 0.02  # Default 20ms for first call
            else:
                dt = current_time - self.last_time
        
        # Prevent division by zero
        if dt <= 0:
            dt = 0.001
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term with windup protection
        self.integral += error * dt
        self.integral = max(self.integral_limits[0], 
                           min(self.integral_limits[1], self.integral))
        integral_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.last_error) / dt
        derivative_term = self.kd * derivative
        
        # Calculate total output
        output = proportional + integral_term + derivative_term
        
        # Apply output limits
        output = max(self.output_limits[0], 
                    min(self.output_limits[1], output))
        
        # Update state
        self.last_error = error
        self.last_time = current_time
        
        # Store error history for analysis
        self.error_history.append(error)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        return output
    
    def reset(self):
        """Reset PID controller state."""
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = None
        self.error_history.clear()
    
    def set_gains(self, kp: float, ki: float, kd: float):
        """Update PID gains."""
        self.kp = kp
        self.ki = ki
        self.kd = kd
    
    def get_components(self, error: float, dt: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Get individual P, I, D components without updating state.
        Useful for debugging and tuning.
        """
        if dt is None:
            dt = 0.02
        
        proportional = self.kp * error
        integral_term = self.ki * self.integral
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        derivative_term = self.kd * derivative
        
        return proportional, integral_term, derivative_term
    
    def get_stats(self) -> dict:
        """Get controller statistics."""
        if not self.error_history:
            return {
                'avg_error': 0.0,
                'max_error': 0.0,
                'min_error': 0.0,
                'current_integral': self.integral,
                'sample_count': 0
            }
        
        return {
            'avg_error': sum(self.error_history) / len(self.error_history),
            'max_error': max(self.error_history),
            'min_error': min(self.error_history),
            'current_integral': self.integral,
            'sample_count': len(self.error_history)
        }

class LineFollowPID:
    """
    PID controller for omni-wheel line following.
    Uses PID for rotation and direct proportional control for strafing.
    """
    
    def __init__(self):
        """Initialize line following PID controller."""
        self.pid = PIDController(
            kp=45.0,    # Proportional gain for rotation
            ki=2.5,     # Integral gain  
            kd=20.0,    # Derivative gain
            output_limits=(-100.0, 100.0),   # Rotation speed limits
            integral_limits=(-30.0, 30.0)    # Prevent integral windup
        )
        
        # Proportional gain for strafing
        self.strafe_gain = 35.0
    
    def calculate_control(self, line_position: float, base_speed: float = 60.0) -> Tuple[float, float, float]:
        """
        Calculate omni-wheel control outputs using a combined strategy.
        
        Args:
            line_position: Line position from -1.0 (left) to 1.0 (right), 0.0 = center
            base_speed: Forward speed
            
        Returns:
            Tuple of (vx, vy, omega) control values
        """
        # Error is how far we are from center (0.0)
        error = line_position
        
        # Rotational control using PID
        omega = -self.pid.update(error)
        
        # Strafing control using simple proportional gain
        vy = -self.strafe_gain * error
        
        # Reduce forward speed during sharp turns to improve stability
        # The reduction is proportional to the rotational speed
        speed_reduction_factor = 1.0 - min(1.0, abs(omega) / 100.0) * 0.7
        vx = base_speed * speed_reduction_factor
        
        return (vx, vy, omega)
    
    def reset_controllers(self):
        """Reset the PID controller."""
        self.pid.reset()
    
    def tune_pid(self, kp: float, ki: float, kd: float, strafe_gain: float):
        """Tune PID and strafe parameters."""
        self.pid.set_gains(kp, ki, kd)
        self.strafe_gain = strafe_gain
    
    def get_control_stats(self) -> dict:
        """Get statistics from the controller."""
        return {
            'pid_stats': self.pid.get_stats(),
            'strafe_gain': self.strafe_gain
        }

class AdaptivePID:
    """
    Advanced PID controller with adaptive gain scheduling.
    Automatically adjusts gains based on operating conditions.
    """
    
    def __init__(self, base_gains: Tuple[float, float, float]):
        """
        Initialize adaptive PID controller.
        
        Args:
            base_gains: (kp, ki, kd) base gain values
        """
        self.base_kp, self.base_ki, self.base_kd = base_gains
        self.controller = PIDController(self.base_kp, self.base_ki, self.base_kd)
        
        # Adaptation parameters
        self.error_threshold_high = 0.8
        self.error_threshold_low = 0.2
        self.speed_factor = 1.0
        
    def update_adaptive(self, error: float, speed_factor: float = 1.0) -> float:
        """
        Update PID with adaptive gain scheduling.
        
        Args:
            error: Current error value
            speed_factor: Speed-based gain adjustment (0.0 to 1.0+)
            
        Returns:
            PID control output
        """
        abs_error = abs(error)
        
        # Adaptive gain calculation
        if abs_error > self.error_threshold_high:
            # High error - increase proportional, reduce derivative
            kp_mult = 1.5
            kd_mult = 0.7
        elif abs_error < self.error_threshold_low:
            # Low error - reduce proportional, increase derivative
            kp_mult = 0.8
            kd_mult = 1.3
        else:
            # Normal error range
            kp_mult = 1.0
            kd_mult = 1.0
        
        # Speed-based adjustment
        speed_mult = 0.5 + 0.5 * speed_factor  # Range: 0.5 to 1.0+
        
        # Update gains
        new_kp = self.base_kp * kp_mult * speed_mult
        new_ki = self.base_ki * speed_mult
        new_kd = self.base_kd * kd_mult * speed_mult
        
        self.controller.set_gains(new_kp, new_ki, new_kd)
        
        return self.controller.update(error) 