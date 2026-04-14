import numpy as np


def get_state(vehicle) -> dict:
    """Read the current vehicle state from the Pegasus State object.

    Returns:
        dict with keys:
            position        np.ndarray (3,)  ENU inertial frame [m]
            velocity        np.ndarray (3,)  ENU inertial frame [m/s]
            orientation     np.ndarray (4,)  quaternion [qx, qy, qz, qw]
            angular_velocity np.ndarray (3,) FLU body frame [rad/s]
    """
    s = vehicle._state
    return {
        "position":         s.position.copy(),
        "velocity":         s.linear_velocity.copy(),
        "orientation":      s.attitude.copy(),
        "angular_velocity": s.angular_velocity.copy(),
    }
