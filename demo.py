#!/usr/bin/env python3
"""
demo.py — Non-interactive demonstrations.

Runs without a display; saves plots to /tmp/.

Usage
-----
    python demo.py --demo fk
    python demo.py --demo ik
    python demo.py --demo traj
    python demo.py --demo all
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless rendering
import matplotlib.pyplot as plt

from robot_arm import (
    Robot6DOF,
    ik_jacobian_transpose, ik_pseudoinverse, ik_dls,
    multi_segment_trajectory,
    plot_ik_convergence,
)


# ---------------------------------------------------------------------------
# Demo 1: Forward Kinematics
# ---------------------------------------------------------------------------

def demo_fk():
    print("\n" + "="*50)
    print("  DEMO 1 — Forward Kinematics")
    print("="*50)

    robot = Robot6DOF(preset="ur5_like")

    configs = {
        "Home (all zeros)": np.zeros(6),
        "Elbow up":  np.radians([0,  -45,  90,  0,  45, 0]),
        "Elbow down": np.radians([0,   45, -90,  0, -45, 0]),
        "Reach up":   np.radians([0, -90,   0,  0,   0, 0]),
        "Wrist flex": np.radians([0,   0,   0,  0,  90, 0]),
    }

    for name, q in configs.items():
        T = robot.fk(q)
        pos   = T[:3, 3]
        euler = robot.ee_euler(q)
        man   = robot.manipulability(q)
        sing  = robot.check_singularity(q)

        print(f"\n  Config: {name}")
        print(f"    q   = {np.degrees(q).round(1)} deg")
        print(f"    pos = {pos.round(4)} m")
        print(f"    euler (ZYX) = {np.degrees(euler).round(2)} deg")
        print(f"    manipulability = {man:.5f}")
        print(f"    singular = {sing['singular']}  (σ_min = {sing['sigma_min']:.4e})")


# ---------------------------------------------------------------------------
# Demo 2: Inverse Kinematics — comparison of three solvers
# ---------------------------------------------------------------------------

def demo_ik():
    print("\n" + "="*50)
    print("  DEMO 2 — Inverse Kinematics (3 solvers)")
    print("="*50)

    robot  = Robot6DOF(preset="ur5_like")
    q_true = np.radians([30, -45, 90, 15, 30, -20])
    T_target = robot.fk(q_true)

    print(f"\n  Target position: {T_target[:3,3].round(4)} m")
    print(f"  True q (ground truth): {np.degrees(q_true).round(1)} deg\n")

    q0 = np.radians([10, -20, 45, 5, 10, -10])  # perturbed start

    methods = {
        "transpose": lambda: ik_jacobian_transpose(
            T_target, q0, robot.dh_params, robot.joint_limits,
            alpha=0.5, max_iter=2000,
        ),
        "pinv": lambda: ik_pseudoinverse(
            T_target, q0, robot.dh_params, robot.joint_limits,
            alpha=1.0, max_iter=500,
        ),
        "dls": lambda: ik_dls(
            T_target, q0, robot.dh_params, robot.joint_limits,
            alpha=1.0, damping=0.05, max_iter=500,
        ),
    }

    results = {}
    for name, solver in methods.items():
        result = solver()
        results[name] = result
        print(f"  [{name:>10}] {result.status.value}")
        print(f"              iters={result.iterations:4d}  "
              f"pos_err={result.pos_error*1000:7.3f} mm  "
              f"manip={result.manipulability:.5f}")

    # Save convergence plot
    fig = plot_ik_convergence(results, "IK Solver Comparison")
    path = "/tmp/ik_convergence.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\n  Plot saved → {path}")

    return results


# ---------------------------------------------------------------------------
# Demo 3: Trajectory Planning
# ---------------------------------------------------------------------------

def demo_trajectory():
    print("\n" + "="*50)
    print("  DEMO 3 — Joint-Space Trajectory")
    print("="*50)

    robot = Robot6DOF(preset="ur5_like")
    waypoints = [
        np.zeros(6),
        np.radians([0, -45,  90,   0,  45,  0]),
        np.radians([45, -60, 100,  20,  30, 10]),
        np.radians([90, -30,  60,  45,  15, -20]),
        np.zeros(6),
    ]

    traj = multi_segment_trajectory(waypoints, n_steps=40, smooth=True)
    print(f"  Trajectory: {len(waypoints)} waypoints → {len(traj)} total steps")

    # Compute EE path
    ee_path = np.array([robot.fk(q)[:3, 3] for q in traj])

    # 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot(ee_path[:, 0], ee_path[:, 1], ee_path[:, 2],
            "-", color="#9b59b6", linewidth=2, label="EE path")

    for i, wp in enumerate(waypoints):
        pt = robot.fk(wp)[:3, 3]
        ax.scatter(*pt, s=100, zorder=5, label=f"WP{i+1}")

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("End-Effector Trajectory (joint-space cubic)")
    ax.legend(loc="upper left", fontsize=8)

    path = "/tmp/trajectory.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  Plot saved → {path}")

    # Joint angle plot
    fig2, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    t = np.linspace(0, 1, len(traj))
    for j in range(6):
        axes[j].plot(t, np.degrees(traj[:, j]))
        axes[j].set_title(f"Joint {j+1}")
        axes[j].set_ylabel("deg")
        axes[j].grid(True, alpha=0.3)
    for ax in axes[-2:]:
        ax.set_xlabel("Normalised time")
    fig2.suptitle("Joint Angles Along Trajectory")
    plt.tight_layout()
    path2 = "/tmp/joints_trajectory.png"
    fig2.savefig(path2, dpi=120, bbox_inches="tight")
    print(f"  Plot saved → {path2}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo", choices=["fk", "ik", "traj", "all"], default="all")
    args = p.parse_args()

    if args.demo in ("fk", "all"):
        demo_fk()
    if args.demo in ("ik", "all"):
        demo_ik()
    if args.demo in ("traj", "all"):
        demo_trajectory()

    print("\n  All demos complete.\n")


if __name__ == "__main__":
    main()
