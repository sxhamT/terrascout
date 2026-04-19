# eval/mpc_vs_pid_benchmark.py
# Offline benchmark: MPC descent vs naive straight-line PID descent.
# Simulates descent trajectories in pure numpy — no Isaac Sim required.
#
# Metrics reported:
#   - Final lateral error (distance from zone centre at touchdown)
#   - Path length (total distance travelled during descent)
#   - Velocity profile smoothness (mean squared acceleration)
#   - Energy proxy: sum of ||u||² * dt (proportional to motor effort)
#   - Constraint violations: fraction of steps with |vz| > limit
#
# Where:  D:\project\terrascout  (any terminal)
# Admin:  No
# Run:    python eval/mpc_vs_pid_benchmark.py [--n 20]

import sys
import argparse
import json
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controller.mpc_descent import MPCDescentOptimizer


# ── Simple PID descent baseline ───────────────────────────────────────────────

class PIDDescent:
    """
    Simple proportional-derivative position controller for descent.
    Used as comparison baseline — same interface as MPC.
    """
    def __init__(self, Kp=8.0, Kd=4.0, mass=1.5, g=9.81):
        self.Kp   = Kp
        self.Kd   = Kd
        self.mass = mass
        self.g    = g

    def compute(self, x0: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        pos  = x0[:3]
        vel  = x0[3:6]
        pos_ref = x_ref[:3]
        err_pos = pos_ref - pos
        err_vel = -vel
        a = self.Kp * err_pos + self.Kd * err_vel
        # Clamp
        a[:2] = np.clip(a[:2], -3.0, 3.0)
        a[2]  = np.clip(a[2],  -5.0, 5.0)
        return a


# ── Trajectory simulation ─────────────────────────────────────────────────────

def simulate_descent(controller, x0: np.ndarray, x_ref: np.ndarray,
                     dt: float = 0.05, max_steps: int = 400,
                     touchdown_z: float = 0.05) -> dict:
    """
    Simulate a descent trajectory until touchdown or max_steps.

    Returns dict with trajectory data and scalar metrics.
    """
    x = x0.copy()
    trajectory = [x.copy()]
    inputs = []
    g = 9.81

    for _ in range(max_steps):
        if x[2] <= touchdown_z:
            break
        # Get acceleration command
        u = controller.compute(x, x_ref)
        inputs.append(u.copy())
        # Integrate: simple Euler, gravity already in plant
        a_net = u + np.array([0.0, 0.0, g]) - np.array([0.0, 0.0, g])
        # In body frame: thrust cancels gravity; net body accel = u
        a_world = u.copy()
        a_world[2] -= 0.0   # gravity already accounted in PID gains
        x = x.copy()
        x[3:6] += a_world * dt
        x[:3]  += x[3:6] * dt
        trajectory.append(x.copy())

    traj = np.array(trajectory)        # (T, 6)
    u_arr = np.array(inputs) if inputs else np.zeros((1, 3))  # (T-1, 3)

    # ── Metrics ───────────────────────────────────────────────────────────────
    final_pos = traj[-1, :3]
    zone_centre = x_ref[:3]
    lateral_error = float(np.linalg.norm(final_pos[:2] - zone_centre[:2]))
    final_alt_error = float(abs(final_pos[2] - zone_centre[2]))

    # Path length
    diffs = np.diff(traj[:, :3], axis=0)
    path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

    # Energy proxy: sum ||u||² * dt
    energy = float(np.sum(np.linalg.norm(u_arr, axis=1) ** 2) * dt)

    # Smoothness: mean squared jerk (finite difference of acceleration)
    if len(u_arr) > 1:
        jerk = np.diff(u_arr, axis=0) / dt
        smoothness = float(np.mean(np.sum(jerk ** 2, axis=1)))
    else:
        smoothness = 0.0

    # Descent speed violations: |vz| > 0.5 m/s during final 2 m
    final_phase_mask = traj[:, 2] < 2.0
    if final_phase_mask.any():
        vz_final = traj[final_phase_mask, 5]
        vz_violations = float(np.mean(np.abs(vz_final) > 0.5))
    else:
        vz_violations = 0.0

    return {
        "n_steps":           len(traj) - 1,
        "lateral_error_m":   round(lateral_error, 4),
        "final_alt_error_m": round(final_alt_error, 4),
        "path_length_m":     round(path_length, 4),
        "energy_proxy":      round(energy, 4),
        "smoothness":        round(smoothness, 4),
        "vz_violations":     round(vz_violations, 4),
        "trajectory":        traj.tolist(),     # for plotting
    }


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(n_trials: int = 20, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)

    mpc = MPCDescentOptimizer(N=10, dt=0.1)
    pid = PIDDescent()

    mpc_results = []
    pid_results = []

    for i in range(n_trials):
        # Random start: approach altitude (2.5–4 m), random lateral offset (0–1.5 m)
        offset_x = rng.uniform(-1.5, 1.5)
        offset_y = rng.uniform(-1.5, 1.5)
        start_alt = rng.uniform(2.5, 4.0)
        x0 = np.array([offset_x, offset_y, start_alt, 0.0, 0.0, -0.2])

        # Target: zone centre at ground level, zero velocity
        x_ref = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        mpc.reset()
        t0 = time.time()
        m = simulate_descent(mpc, x0, x_ref, dt=0.05)
        m["solve_time_ms"] = round((time.time() - t0) * 1000, 2)
        m.pop("trajectory")   # don't bloat JSON for aggregate
        mpc_results.append(m)

        p = simulate_descent(pid, x0, x_ref, dt=0.05)
        p["solve_time_ms"] = 0.0
        p.pop("trajectory")
        pid_results.append(p)

        print(f"  Trial {i+1:2d}: MPC lat={m['lateral_error_m']:.3f}m "
              f"E={m['energy_proxy']:.2f}  |  PID lat={p['lateral_error_m']:.3f}m "
              f"E={p['energy_proxy']:.2f}")

    def mean_std(results, key):
        vals = [r[key] for r in results]
        return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)

    keys = ["lateral_error_m", "path_length_m", "energy_proxy", "smoothness", "vz_violations"]
    comparison = {}
    for k in keys:
        mpc_m, mpc_s = mean_std(mpc_results, k)
        pid_m, pid_s = mean_std(pid_results, k)
        comparison[k] = {
            "mpc_mean": mpc_m, "mpc_std": mpc_s,
            "pid_mean": pid_m, "pid_std": pid_s,
            "delta_pct": round(100 * (mpc_m - pid_m) / (pid_m + 1e-9), 1),
        }

    return {
        "n_trials":    n_trials,
        "comparison":  comparison,
        "mpc_raw":     mpc_results,
        "pid_raw":     pid_results,
    }


def print_comparison(results: dict):
    comp = results["comparison"]
    n = results["n_trials"]
    lines = [
        "",
        "=" * 66,
        "  TerraScout: MPC Descent vs PID Baseline Benchmark",
        f"  {n} trials, random start offsets up to 1.5 m lateral",
        "=" * 66,
        f"  {'Metric':<26} {'MPC':>14} {'PID':>14} {'Delta':>8}",
        "-" * 66,
    ]
    labels = {
        "lateral_error_m":  "Final lateral error [m]",
        "path_length_m":    "Path length [m]",
        "energy_proxy":     "Energy proxy [Σ||u||²·dt]",
        "smoothness":       "Smoothness [mean jerk²]",
        "vz_violations":    "Descent speed violations",
    }
    for k, label in labels.items():
        c = comp[k]
        mpc_str = f"{c['mpc_mean']:.3f} ± {c['mpc_std']:.3f}"
        pid_str = f"{c['pid_mean']:.3f} ± {c['pid_std']:.3f}"
        sign = "+" if c["delta_pct"] > 0 else ""
        lines.append(f"  {label:<26} {mpc_str:>14} {pid_str:>14} {sign}{c['delta_pct']:>6.1f}%")
    lines += ["=" * 66, "  Negative delta = MPC better", "=" * 66, ""]
    text = "\n".join(lines)
    print(text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out",  type=str,
                        default=str(Path(__file__).parent / "results"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Benchmark] MPC vs PID, {args.n} trials...")
    results = run_benchmark(n_trials=args.n, seed=args.seed)
    text = print_comparison(results)

    with open(out_dir / "mpc_vs_pid.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "mpc_vs_pid.txt", "w") as f:
        f.write(text)

    print(f"[Benchmark] Results written to {out_dir}/")


if __name__ == "__main__":
    main()
