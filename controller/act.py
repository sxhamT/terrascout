import numpy as np


def compute_rotor_velocities(vehicle, force: float, torque: np.ndarray) -> np.ndarray:
    """Convert desired collective thrust and body-frame torque to per-rotor
    angular velocities using the Multirotor allocation matrix.

    Args:
        vehicle: Pegasus Multirotor instance (provides force_and_torques_to_velocities)
        force:   Collective thrust along body Z [N]
        torque:  Body-frame torque vector (3,) [Nm]

    Returns:
        np.ndarray of rotor angular velocities [rad/s] — returned value must be
        stored in backend.input_ref and served via input_reference().
    """
    return vehicle.force_and_torques_to_velocities(force, torque)
