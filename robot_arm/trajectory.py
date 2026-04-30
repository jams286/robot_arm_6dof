"""
Trajectory planning utilities.

Joint-space interpolation (cubic spline) and Cartesian straight-line paths.
"""

import numpy as np
from typing import List, Optional, Tuple


def cubic_interpolation(
    q_start: np.ndarray,
    q_end  : np.ndarray,
    n_steps: int = 50,
) -> np.ndarray:
    """
    Cubic (smooth) interpolation between two joint configurations.

    Uses a parameter s ∈ [0, 1] with a cubic smoothstep:
        s_smooth = 3s² - 2s³

    Returns
    -------
    waypoints : (n_steps, n_joints) ndarray
    """
    t = np.linspace(0, 1, n_steps)
    s = 3 * t**2 - 2 * t**3          # smoothstep
    return q_start + s[:, None] * (q_end - q_start)


def linear_interpolation(
    q_start: np.ndarray,
    q_end  : np.ndarray,
    n_steps: int = 50,
) -> np.ndarray:
    """Linear joint-space interpolation (LERP)."""
    t = np.linspace(0, 1, n_steps)
    return q_start + t[:, None] * (q_end - q_start)


def multi_segment_trajectory(
    waypoints : List[np.ndarray],
    n_steps   : int = 30,
    smooth    : bool = True,
) -> np.ndarray:
    """
    Build a trajectory through multiple joint-space waypoints.

    Parameters
    ----------
    waypoints : list of (n,) arrays
    n_steps   : steps per segment
    smooth    : use cubic if True, else linear

    Returns
    -------
    traj : (total_steps, n_joints) ndarray
    """
    interp = cubic_interpolation if smooth else linear_interpolation
    segments = []
    for i in range(len(waypoints) - 1):
        seg = interp(waypoints[i], waypoints[i + 1], n_steps)
        if i < len(waypoints) - 2:
            seg = seg[:-1]   # avoid duplicating junction point
        segments.append(seg)
    return np.vstack(segments)


def cartesian_line(
    robot,
    p_start : np.ndarray,
    p_end   : np.ndarray,
    n_steps : int = 30,
    method  : str = "dls",
) -> Tuple[np.ndarray, List[bool]]:
    """
    Plan a straight Cartesian-space path from p_start to p_end.

    IK is solved at each waypoint, seeding from the previous solution.

    Parameters
    ----------
    robot   : Robot6DOF
    p_start : (3,) start position
    p_end   : (3,) end position
    n_steps : number of intermediate points
    method  : IK solver

    Returns
    -------
    traj    : (n_steps, 6) joint angles
    success : list of bool per step
    """
    points  = np.linspace(p_start, p_end, n_steps)
    traj    = np.zeros((n_steps, robot.n_joints))
    success = []

    q_prev = robot.q
    for i, pt in enumerate(points):
        result = robot.ik_position_only(pt, method=method, initial_q=q_prev)
        traj[i] = result.joint_angles
        success.append(result.success)
        q_prev = result.joint_angles

    return traj, success
