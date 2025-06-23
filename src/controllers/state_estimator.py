import numpy as np
from filterpy.kalman import KalmanFilter

class StateEstimator:
    """
    Estimates the robot's state (x, y, heading) using a Kalman filter.
    It fuses control inputs (from kinematics) with measurements (from vision).
    """

    def __init__(self, dt: float, initial_pose: tuple):
        """
        Initializes the Kalman Filter.
        :param dt: The time step (delta time) between updates.
        :param initial_pose: The initial pose (x, y, heading) of the robot.
        """
        self.dt = dt

        # Create the Kalman Filter
        # State: [x, y, heading]
        # Measurement: [x, y, heading]
        self.kf = KalmanFilter(dim_x=3, dim_z=3, dim_u=3)
        
        # Initial State
        self.kf.x = np.array([initial_pose]).T # [x, y, heading]

        # State Transition Matrix
        # Updated dynamically in predict step, but initialized here.
        self.kf.F = np.identity(3)

        # Measurement Function
        # We measure the state directly, so it's the identity matrix.
        self.kf.H = np.identity(3)

        # Measurement Noise Covariance
        # How much we trust the vision-based measurement. Lower values mean more trust.
        # Start with relatively high confidence in the measurement.
        self.kf.R = np.diag([0.05, 0.05, np.deg2rad(1.0)])**2

        # Process Noise Covariance
        # How much we trust our motion model (kinematics). Higher values mean less trust.
        # This accounts for wheel slippage, model inaccuracies, etc.
        self.kf.Q = np.diag([0.01, 0.01, np.deg2rad(0.5)])**2

        # Initial Estimate Uncertainty
        # Start with high uncertainty, the filter will converge.
        self.kf.P = np.identity(3) * 500.

    def predict(self, u: np.ndarray):
        """
        Predicts the next state based on the control input (robot velocity).
        :param u: The control input [vx, vy, v_theta].
        """
        vx, vy, v_theta = u[0], u[1], u[2]
        current_heading = self.kf.x[2, 0]

        # Build the control input model (B) dynamically
        # This matrix maps the control input u to the state space.
        # It's based on a simple kinematic model:
        # x_new = x_old + dt * (vx * cos(h) - vy * sin(h))
        # y_new = y_old + dt * (vx * sin(h) + vy * cos(h))
        # h_new = h_old + dt * v_theta
        B = np.array([
            [self.dt * np.cos(current_heading), -self.dt * np.sin(current_heading), 0],
            [self.dt * np.sin(current_heading),  self.dt * np.cos(current_heading), 0],
            [0, 0, self.dt]
        ])
        
        self.kf.predict(u=u, B=B)
        self._normalize_angle()

    def update(self, z: np.ndarray):
        """
        Updates the state estimate with a new measurement.
        :param z: The measurement [x, y, heading] from the vision system.
        """
        # The measurement vector z must be a column vector
        z = np.array([z]).T
        self.kf.update(z)
        self._normalize_angle()
        
    def _normalize_angle(self):
        """Normalizes the heading angle in the state vector to [-pi, pi]."""
        self.kf.x[2] = (self.kf.x[2] + np.pi) % (2 * np.pi) - np.pi
    
    @property
    def pose(self) -> np.ndarray:
        """Returns the current estimated pose as a 1D array [x, y, heading]."""
        return self.kf.x.flatten() 