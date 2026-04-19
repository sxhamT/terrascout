# controller/mpc_descent.py
# MPCDescentOptimizer — Week 10 deliverable.
#
# Discrete-time linear MPC for descent trajectory optimisation.
# State:  x = [px, py, pz, vx, vy, vz]  (6,)
# Input:  u = [ax, ay, az]               (3,) body accelerations [m/s²]
#
# Objective: min Σ_{k=0}^{N-1} (||xk - xref||_Q² + ||uk||_R²) + ||xN - xref||_P²
# Subject to: acceleration box constraints (bounds on u — velocity limits
#             enforced implicitly via a_max and the short horizon).
#
# Solver: scipy.optimize.minimize with SLSQP.
# Warm-started from previous solution.
# Falls back to returning zero input if solve time > 50 ms.
#
# Performance notes (to stay within 50 ms budget):
#   N=10 → 30 decision variables (vs 60 at N=20)
#   Velocity constraint loop removed — each constraint called _rollout N times,
#   making the old code O(N²) rollouts per SLSQP iteration.
#   _rollout vectorised via precomputed Sx/Su propagation matrices.

import numpy as np
from scipy.optimize import minimize
import time


class MPCDescentOptimizer:
    """
    Linear MPC for descent trajectory.

    Parameters
    ----------
    N : int
        Prediction horizon (number of steps).
    dt : float
        Timestep in seconds.
    Q : array (6,)
        State error weights [px, py, pz, vx, vy, vz].
    R : array (3,)
        Input effort weights [ax, ay, az].
    v_max_approach : float
        Max speed during general approach [m/s].
    v_max_descent : float
        Max vertical descent speed [m/s] during final 2m.
    a_max_lat : float
        Max lateral acceleration [m/s²].
    a_max_vert : float
        Max vertical acceleration [m/s²].
    """

    def __init__(
        self,
        N: int = 10,
        dt: float = 0.1,
        Q: np.ndarray = None,
        R: np.ndarray = None,
        v_max_approach: float = 2.0,
        v_max_descent: float  = 0.5,
        a_max_lat: float      = 3.0,
        a_max_vert: float     = 5.0,
        solve_timeout_ms: float = 50.0,
    ):
        self.N  = N
        self.dt = dt
        self.nx = 6
        self.nu = 3

        self.Q  = np.diag(Q  if Q  is not None else [10, 10, 10, 2, 2, 5])
        self.R  = np.diag(R  if R  is not None else [0.1, 0.1, 0.1])
        self.P  = self.Q * 2   # terminal cost (heavier than stage)

        self.v_max_approach = v_max_approach
        self.v_max_descent  = v_max_descent
        self.a_max_lat      = a_max_lat
        self.a_max_vert     = a_max_vert
        self.solve_timeout  = solve_timeout_ms / 1000.0

        # Discrete integrator dynamics: x_{k+1} = A x_k + B u_k
        # Position integrates velocity; velocity integrates acceleration.
        self.A = np.eye(6)
        self.A[0, 3] = dt
        self.A[1, 4] = dt
        self.A[2, 5] = dt

        self.B = np.zeros((6, 3))
        self.B[3, 0] = dt
        self.B[4, 1] = dt
        self.B[5, 2] = dt

        # Gravity compensation — subtract g from az before sending to plant
        self.g = 9.81

        # Warm-start: previous solution (N*nu,)
        self._u_prev = np.zeros(N * self.nu)

        # Precompute state propagation matrices for vectorised rollout.
        # X_flat = Sx @ x0 + Su @ u_flat  where X_flat is (N*nx,)
        self._Sx, self._Su = self._build_propagation_matrices()

        print(f"[MPC] Horizon N={N}, dt={dt}s, "
              f"v_max={v_max_approach}/{v_max_descent}m/s, "
              f"a_max={a_max_lat}/{a_max_vert}m/s²")

    def compute(self, x0: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """
        Solve MPC and return optimal first control input.

        Parameters
        ----------
        x0    : (6,) current state [px, py, pz, vx, vy, vz]
        x_ref : (6,) reference (landing zone centre + zero velocity)

        Returns
        -------
        u_opt : (3,) optimal [ax, ay, az] for current step [m/s²]
                Returns gravity-compensation-only (hover) on solver failure.
        """
        t_start = time.time()

        # Build bounds on decision variable (flattened input sequence)
        lb = np.tile([-self.a_max_lat, -self.a_max_lat, -self.a_max_vert], self.N)
        ub = np.tile([ self.a_max_lat,  self.a_max_lat,  self.a_max_vert], self.N)

        result = minimize(
            fun=self._cost,
            x0=self._u_prev,
            args=(x0, x_ref),
            method="SLSQP",
            bounds=list(zip(lb, ub)),
            options={
                "maxiter": 100,
                "ftol": 1e-3,
                "disp": False,
            },
        )

        elapsed = time.time() - t_start

        if result.success and elapsed < self.solve_timeout:
            u_seq = result.x.reshape(self.N, self.nu)
            self._u_prev = np.roll(result.x, -self.nu)
            self._u_prev[-self.nu:] = 0.0
            u_opt = u_seq[0]
        else:
            if not result.success:
                print(f"[MPC] Solver failed: {result.message} — using PID fallback")
            else:
                print(f"[MPC] Timeout ({elapsed*1000:.1f}ms) — using PID fallback")
            # Gravity compensation only (hover in place)
            u_opt = np.array([0.0, 0.0, 0.0])

        return u_opt

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_propagation_matrices(self):
        """
        Precompute Sx (N*nx, nx) and Su (N*nx, N*nu) so that the full
        state trajectory satisfies:

            X_flat = Sx @ x0 + Su @ u_flat

        where X_flat = [x_1; x_2; ...; x_N] stacked column-wise.
        Su is lower block-triangular (causal): Su[k, j] = A^(k-j) @ B.

        Built once at construction — zero per-call Python loops in rollout.
        """
        N, nx, nu = self.N, self.nx, self.nu

        # Powers of A: A_pow[k] = A^k
        A_pow = [np.eye(nx)]
        for _ in range(N):
            A_pow.append(self.A @ A_pow[-1])

        Sx = np.zeros((N * nx, nx))
        Su = np.zeros((N * nx, N * nu))

        for k in range(N):          # row block k  ↔  state x_{k+1}
            Sx[k*nx:(k+1)*nx, :] = A_pow[k + 1]
            for j in range(k + 1):  # col block j  ↔  input u_j
                Su[k*nx:(k+1)*nx, j*nu:(j+1)*nu] = A_pow[k - j] @ self.B

        return Sx, Su

    def _rollout(self, u_flat: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """
        Return (N+1, nx) state trajectory via precomputed propagation matrices.
        No Python loop — pure matrix multiply.
        """
        X_flat = self._Sx @ x0 + self._Su @ u_flat   # (N*nx,)
        xs = np.empty((self.N + 1, self.nx))
        xs[0] = x0
        xs[1:] = X_flat.reshape(self.N, self.nx)
        return xs

    def _cost(self, u_flat: np.ndarray, x0: np.ndarray, x_ref: np.ndarray) -> float:
        xs = self._rollout(u_flat, x0)
        u  = u_flat.reshape(self.N, self.nu)

        # Stage costs — vectorised over horizon
        dX = xs[:self.N] - x_ref                         # (N, nx)
        J  = float(np.einsum("ki,ij,kj->", dX, self.Q, dX) +
                   np.einsum("ki,ij,kj->", u,  self.R, u))
        # Terminal cost
        dx_N = xs[self.N] - x_ref
        J += float(dx_N @ self.P @ dx_N)
        return J

    def reset(self):
        self._u_prev = np.zeros(self.N * self.nu)
