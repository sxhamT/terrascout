import numpy as np


class PIDController:
    """Position + SO(3) attitude PID controller for a quadrotor.

    Outer loop  : position error → desired force  (Kp, Kd, Ki)
    Inner loop  : SO(3) attitude error → torque   (Kr, Kw)

    Follows the geometric controller of Mellinger & Kumar (ICRA 2011).
    Pure numpy — no external dependencies.
    """

    # Iris quadrotor physical constants
    MASS = 1.50   # kg
    GRAVITY = 9.81  # m/s²
    INTEGRAL_CLAMP = 5.0  # m  (anti-windup)

    def __init__(self):
        self.Kp = np.diag([10.0, 10.0, 10.0])
        self.Kd = np.diag([8.5,  8.5,  8.5])
        self.Ki = np.diag([1.5,  1.5,  1.5])
        self.Kr = np.diag([3.5,  3.5,  3.5])   # attitude error
        self.Kw = np.diag([0.5,  0.5,  0.5])   # angular rate error
        self._integral = np.zeros(3)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_to_rot(q: np.ndarray) -> np.ndarray:
        """Quaternion [qx, qy, qz, qw] → 3×3 rotation matrix."""
        qx, qy, qz, qw = q
        return np.array([
            [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),  1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),  1 - 2*(qx**2 + qy**2)],
        ])

    @staticmethod
    def _vee(S: np.ndarray) -> np.ndarray:
        """Vee map: skew-symmetric 3×3 → R³."""
        return np.array([-S[1, 2], S[0, 2], -S[0, 1]])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        orientation_quat: np.ndarray,
        angular_velocity: np.ndarray,
        target_position: np.ndarray,
        dt: float,
    ):
        """Compute collective thrust and body-frame torque.

        Args:
            position:         Current position ENU (3,) [m]
            velocity:         Current velocity ENU (3,) [m/s]
            orientation_quat: Current attitude [qx,qy,qz,qw]
            angular_velocity: Body-frame angular velocity (3,) [rad/s]
            target_position:  Desired position ENU (3,) [m]
            dt:               Time step [s]

        Returns:
            (force, torque): scalar thrust [N] and torque vector (3,) [Nm]
        """
        # --- Outer position loop ---
        ep = position - target_position           # position error
        ev = velocity                             # velocity error (v_ref = 0)

        self._integral = np.clip(
            self._integral + ep * dt,
            -self.INTEGRAL_CLAMP,
            self.INTEGRAL_CLAMP,
        )

        F_des = (
            -(self.Kp @ ep)
            - (self.Kd @ ev)
            - (self.Ki @ self._integral)
            + np.array([0.0, 0.0, self.MASS * self.GRAVITY])
        )

        # Project onto current body Z to get thrust
        R = self._quat_to_rot(orientation_quat)
        Z_B = R[:, 2]
        u_1 = float(F_des @ Z_B)

        # --- Inner SO(3) attitude loop ---
        norm_F = np.linalg.norm(F_des)
        Z_b_des = F_des / norm_F if norm_F > 1e-6 else np.array([0.0, 0.0, 1.0])

        # Desired body X axis — zero yaw reference
        X_c_des = np.array([1.0, 0.0, 0.0])
        cross = np.cross(Z_b_des, X_c_des)
        cross_norm = np.linalg.norm(cross)
        Y_b_des = cross / cross_norm if cross_norm > 1e-6 else np.array([0.0, 1.0, 0.0])
        X_b_des = np.cross(Y_b_des, Z_b_des)

        R_des = np.c_[X_b_des, Y_b_des, Z_b_des]

        # Rotation error (vee map of skew-symmetric part)
        e_R = 0.5 * self._vee((R_des.T @ R) - (R.T @ R_des))
        e_w = angular_velocity   # angular rate error (w_ref = 0)

        torque = -(self.Kr @ e_R) - (self.Kw @ e_w)

        return u_1, torque

    def reset(self):
        self._integral = np.zeros(3)
