import numpy as np

class MecanumKinematics:
    """
    Handles the forward and inverse kinematics for a four-wheel Mecanum drive robot.
    Assumes the standard Mecanum wheel configuration where rollers are angled at 45 degrees.
    """
    def __init__(self, wheel_radius_m: float, robot_width_m: float, robot_length_m: float):
        """
        Initializes the kinematics model.
        :param wheel_radius_m: The radius of the wheels in meters.
        :param robot_width_m: The distance from the center of the robot to the wheels along the y-axis.
        :param robot_length_m: The distance from the center of the robot to the wheels along the x-axis.
        """
        self.r = wheel_radius_m
        # The sum of the distances from the center to the wheels on x and y axes
        self.l_plus_w = (robot_width_m / 2) + (robot_length_m / 2)

        # The inverse kinematic matrix maps robot velocity (vx, vy, v_theta) to wheel angular velocities (rad/s)
        self.inverse_kinematic_matrix = (1 / self.r) * np.array([
            [1, -1, -self.l_plus_w],  # Front Left
            [1,  1,  self.l_plus_w],  # Front Right
            [1,  1, -self.l_plus_w],  # Back Left
            [1, -1,  self.l_plus_w]   # Back Right
        ])
        
        # The forward kinematic matrix maps wheel angular velocities to robot velocity
        self.forward_kinematic_matrix = (self.r / 4) * np.array([
            [1,  1,  1,  1],
            [-1, 1,  1, -1],
            [-1/self.l_plus_w, 1/self.l_plus_w, -1/self.l_plus_w, 1/self.l_plus_w]
        ])

    def get_wheel_speeds(self, vx: float, vy: float, v_theta: float) -> np.ndarray:
        """
        Calculates the required angular velocity for each wheel for a given robot velocity.
        :param vx: Desired forward velocity (m/s).
        :param vy: Desired sideways (strafe) velocity (m/s).
        :param v_theta: Desired angular velocity (rad/s).
        :return: A numpy array of wheel angular velocities [fl, fr, bl, br] in rad/s.
        """
        robot_velocity_vector = np.array([vx, vy, v_theta])
        wheel_angular_velocities = self.inverse_kinematic_matrix @ robot_velocity_vector
        return wheel_angular_velocities

    def get_robot_velocity(self, wheel_speeds: np.ndarray) -> np.ndarray:
        """
        Calculates the robot's velocity based on the measured speed of its wheels.
        :param wheel_speeds: A numpy array of wheel angular velocities [fl, fr, bl, br] in rad/s.
        :return: A numpy array of the robot's velocity [vx, vy, v_theta].
        """
        robot_velocity = self.forward_kinematic_matrix @ wheel_speeds
        return robot_velocity 