#!/usr/bin/env python3
"""
main.py — Interactive 3D robot arm visualizer.

Usage
-----
    python main.py                    # Default UR5-like preset
    python main.py --preset simple    # Simple educational arm
    python main.py --config q 0 -45 90 0 45 0   # Start at specific angles (degrees)
"""

import argparse
import numpy as np

from robot_arm import Robot6DOF, RobotVisualizer


def parse_args():
    p = argparse.ArgumentParser(description="6-DOF Robot Arm Interactive Simulator")
    p.add_argument("--preset", default="ur5_like",
                   choices=["ur5_like", "simple", "puma560", "stanford"],
                   help="DH parameter preset")
    p.add_argument("--q", nargs=6, type=float, default=None,
                   metavar=("J1","J2","J3","J4","J5","J6"),
                   help="Initial joint angles in degrees")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  6-DOF Robot Arm Simulator")
    print("=" * 60)

    robot = Robot6DOF(preset=args.preset)

    if args.q is not None:
        robot.q = np.radians(args.q)

    print(robot)
    print()
    print("Controls:")
    print("  Sliders → move each joint")
    print("  [Home]   → reset to zero configuration")
    print("  [Random] → random valid configuration")
    print("  [▶ Traj] → play loaded trajectory (if any)")
    print("  Drag 3D view → rotate camera")
    print("=" * 60)

    viz = RobotVisualizer(robot)
    viz.show()


if __name__ == "__main__":
    main()
