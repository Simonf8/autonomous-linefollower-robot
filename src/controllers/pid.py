import time
from typing import Tuple, Optional, Dict, List
from collections import deque

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
            kp=55.0,    # Proportional gain for rotation
            ki=2.5,     # Integral gain  
            kd=25.0,    # Derivative gain
            output_limits=(-100.0, 100.0),   # Rotation speed limits
            integral_limits=(-30.0, 30.0)    # Prevent integral windup
        )
        
        # Proportional gain for strafing
        self.strafe_gain = 30.0
    
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
        vy = self.strafe_gain * error
        
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
    """
    
    def __init__(self, 
                 base_kp=1.2, base_ki=0.05, base_kd=0.4,
                 output_limits=(-70, 70),
                 sample_time=0.033,  # ~30 FPS
                 debug=False):
        
        self.base_kp = base_kp
        self.base_ki = base_ki
        self.base_kd = base_kd
        
        self.kp = base_kp
        self.ki = base_ki
        self.kd = base_kd
        
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        self.sample_time = sample_time
        
        self.output_limits = output_limits
        self.last_output = 0.0
        
        self.error_history = deque(maxlen=10)
        self.output_history = deque(maxlen=5)
        self.derivative_history = deque(maxlen=3)
        
        self.speed_gain_map = {
            'slow': {'kp_mult': 1.4, 'ki_mult': 0.8, 'kd_mult': 1.2},
            'medium': {'kp_mult': 1.0, 'ki_mult': 1.0, 'kd_mult': 1.0},
            'fast': {'kp_mult': 0.7, 'ki_mult': 1.3, 'kd_mult': 0.8}
        }
        
        self.error_gain_map = {
            'small':  {'kp_mult': 0.9, 'ki_mult': 1.2, 'kd_mult': 0.9},
            'medium': {'kp_mult': 1.5, 'ki_mult': 1.0, 'kd_mult': 1.2},
            'large':  {'kp_mult': 2.5, 'ki_mult': 0.5, 'kd_mult': 1.8},
            'severe': {'kp_mult': 3.5, 'ki_mult': 0.3, 'kd_mult': 2.5}
        }
        
        self.integral_limit = 20.0
        self.integral_decay_rate = 0.95
        self.windup_threshold = 0.8 * max(abs(output_limits[0]), abs(output_limits[1]))
        
        self.feedforward_enabled = True
        self.velocity_gain = 0.3
        
        self.base_deadband = 0.03
        self.confidence_deadband_factor = 0.8
        
        self.gain_smoothing_alpha = 0.3
        self.derivative_filter_alpha = 0.7
        self.filtered_derivative = 0.0
        
        self.debug = debug
        self.last_performance_metrics = {}
        self.prediction_gain_factor = 0.8
        
    def update(self, error: float, robot_state: Dict = None, line_confidence: float = 1.0, 
               using_prediction: bool = False) -> float:
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt < self.sample_time * 0.5:
            return self.last_output
        
        self.error_history.append(abs(error))
        
        adaptive_deadband = self.base_deadband * (2.0 - line_confidence * self.confidence_deadband_factor)
        if abs(error) < adaptive_deadband:
            error = 0.0
        
        self._update_adaptive_gains(error, robot_state, line_confidence, using_prediction)
        
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.derivative_history.append(derivative)
        
        self.filtered_derivative = (self.derivative_filter_alpha * self.filtered_derivative + 
                                  (1 - self.derivative_filter_alpha) * derivative)
        
        self.integral += error * dt
        
        if abs(self.last_output) > self.windup_threshold:
            if (self.integral > 0 and error > 0) or (self.integral < 0 and error < 0):
                self.integral *= self.integral_decay_rate
        
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        
        proportional = self.kp * error
        integral_component = self.ki * self.integral
        derivative_component = self.kd * self.filtered_derivative
        
        feedforward = 0.0
        if self.feedforward_enabled and robot_state:
            if len(self.error_history) >= 2:
                error_velocity = (self.error_history[-1] - self.error_history[-2]) / dt
                feedforward = self.velocity_gain * error_velocity
        
        output = proportional + integral_component + derivative_component + feedforward
        output = max(self.output_limits[0], min(self.output_limits[1], output))
        
        self.last_error = error
        self.last_time = current_time
        self.last_output = output
        
        if self.debug:
            self._update_performance_metrics(error, output, proportional, integral_component, 
                                           derivative_component, feedforward, line_confidence)
        
        return output
    
    def _update_adaptive_gains(self, error: float, robot_state: Dict, 
                             line_confidence: float, using_prediction: bool):
        speed_category = self._get_speed_category(robot_state)
        error_category = self._get_error_category(error)
        
        speed_mults = self.speed_gain_map[speed_category]
        error_mults = self.error_gain_map[error_category]
        
        new_kp = self.base_kp * speed_mults['kp_mult'] * error_mults['kp_mult']
        new_ki = self.base_ki * speed_mults['ki_mult'] * error_mults['ki_mult']
        new_kd = self.base_kd * speed_mults['kd_mult'] * error_mults['kd_mult']
        
        confidence_factor = 0.5 + 0.5 * line_confidence
        new_kp *= confidence_factor
        new_ki *= confidence_factor
        new_kd *= confidence_factor
        
        if using_prediction:
            new_kp *= self.prediction_gain_factor
            new_ki *= self.prediction_gain_factor
            new_kd *= self.prediction_gain_factor
        
        alpha = self.gain_smoothing_alpha
        self.kp = alpha * new_kp + (1 - alpha) * self.kp
        self.ki = alpha * new_ki + (1 - alpha) * self.ki
        self.kd = alpha * new_kd + (1 - alpha) * self.kd
    
    def _get_speed_category(self, robot_state: Dict) -> str:
        if not robot_state:
            return 'medium'
        motor_speeds = robot_state.get('motor_speeds', {'left': 0, 'right': 0})
        avg_speed = abs(motor_speeds.get('left', 0) + motor_speeds.get('right', 0)) / 2
        
        if avg_speed < 20: return 'slow'
        if avg_speed < 40: return 'medium'
        return 'fast'
    
    def _get_error_category(self, error: float) -> str:
        abs_error = abs(error)
        if abs_error < 0.05: return 'small'
        if abs_error < 0.15: return 'medium'
        if abs_error < 0.35: return 'large'
        return 'severe'
    
    def _update_performance_metrics(self, error, output, p_term, i_term, d_term, ff_term, confidence):
        self.last_performance_metrics = {
            'error': error, 'output': output, 'p_term': p_term, 'i_term': i_term,
            'd_term': d_term, 'ff_term': ff_term, 'kp': self.kp, 'ki': self.ki, 'kd': self.kd,
            'integral': self.integral, 'confidence': confidence
        }
    
    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.last_output = 0.0
        self.filtered_derivative = 0.0
        self.last_time = time.time()
        self.error_history.clear()
        self.output_history.clear()
        self.derivative_history.clear()
 