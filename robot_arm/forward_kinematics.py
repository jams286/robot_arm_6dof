"""
Forward Kinematics via Denavit-Hartenberg convention.

Each row of the DH table defines the transform between frame i-1 and frame i:

    T_i = Rot_z(θ_i) · Trans_z(d_i) · Trans_x(a_i) · Rot_x(α_i)

    ┌                                                          ┐
    │  cθ  -sθ·cα   sθ·sα   a·cθ  │
    │  sθ   cθ·cα  -cθ·sα   a·sθ  │
    │   0    sα      cα      d     │
    │   0     0       0      1     │
    └                                                          ┘
"""

import numpy as np
from typing import List, Optional


def dh_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """
    Build the 4×4 homogeneous DH transform for a single joint.

    Parameters
    ----------
    theta : float   Variable joint angle (radians) – revolute joint.
    d     : float   Link offset along Z_{i-1} (metres).
    a     : float   Link length along X_i (metres).
    alpha : float   Twist angle about X_i (radians).

    Returns
    -------
    T : (4, 4) ndarray  Homogeneous transformation matrix.
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct,  -st * ca,   st * sa,   a * ct],
        [st,   ct * ca,  -ct * sa,   a * st],
        [0.0,       sa,        ca,        d],
        [0.0,      0.0,       0.0,      1.0],
    ])


def forward_kinematics(
    joint_angles: np.ndarray,
    dh_params: np.ndarray,
) -> List[np.ndarray]:
    """
    Compute forward kinematics for all joints.

    Parameters
    ----------
    joint_angles : (n,) ndarray   Joint angles θ (radians).
    dh_params    : (n, 4) ndarray Rows are [a, alpha, d, theta_offset].

    Returns
    -------
    transforms : list of (4, 4) ndarray
        transforms[0]  = T_0_1  (base → joint 1)
        transforms[-1] = T_0_n  (base → end-effector)
    """
    n = len(joint_angles)
    T = np.eye(4)
    transforms = []

    for i in range(n):
        a_i, alpha_i, d_i, offset_i = dh_params[i]
        theta_i = joint_angles[i] + offset_i
        T_i = dh_matrix(theta_i, d_i, a_i, alpha_i)
        T = T @ T_i
        transforms.append(T.copy())

    return transforms


def get_end_effector_pose(
    joint_angles: np.ndarray,
    dh_params: np.ndarray,
) -> np.ndarray:
    """
    Return only the end-effector homogeneous transform T_0_n.

    Parameters
    ----------
    joint_angles : (n,) ndarray
    dh_params    : (n, 4) ndarray

    Returns
    -------
    T_ee : (4, 4) ndarray
    """
    transforms = forward_kinematics(joint_angles, dh_params)
    return transforms[-1]


def get_joint_positions(
    joint_angles: np.ndarray,
    dh_params: np.ndarray,
) -> np.ndarray:
    """
    Return XYZ positions of each frame origin (including base and EE).

    Returns
    -------
    positions : (n+1, 3) ndarray  – row 0 is the base (origin).
    """
    transforms = forward_kinematics(joint_angles, dh_params)
    base = np.zeros((1, 3))
    joint_pos = np.array([T[:3, 3] for T in transforms])
    return np.vstack([base, joint_pos])


def rotation_matrix_to_euler_zyx(R: np.ndarray) -> np.ndarray:
    """
    Extract ZYX Euler angles (yaw, pitch, roll) from a rotation matrix.

    Returns
    -------
    euler : (3,) ndarray  [yaw, pitch, roll] in radians.
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll  = np.arctan2( R[2, 1],  R[2, 2])
        pitch = np.arctan2(-R[2, 0],  sy)
        yaw   = np.arctan2( R[1, 0],  R[0, 0])
    else:
        roll  = np.arctan2(-R[1, 2],  R[1, 1])
        pitch = np.arctan2(-R[2, 0],  sy)
        yaw   = 0.0

    return np.array([yaw, pitch, roll])


def pose_error(T_current: np.ndarray, T_target: np.ndarray) -> np.ndarray:
    """
    Compute 6-D pose error [Δx, Δy, Δz, Δroll, Δpitch, Δyaw].

    The orientation error uses the angle-axis representation of
    R_err = R_current^T · R_target.

    Parameters
    ----------
    T_current : (4, 4) ndarray   Current EE pose.
    T_target  : (4, 4) ndarray   Target EE pose.

    Returns
    -------
    error : (6,) ndarray
    """
    # Position error
    dp = T_target[:3, 3] - T_current[:3, 3]

    # Orientation error via skew-symmetric part
    R_cur = T_current[:3, :3]
    R_tar = T_target[:3, :3]
    R_err = R_cur.T @ R_tar

    # Rodrigues angle-axis from R_err
    theta = np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
    if abs(theta) < 1e-8:
        axis = np.zeros(3)
    else:
        axis = np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]) / (2.0 * np.sin(theta))

    dR = R_cur @ (theta * axis)          # map to global frame

    return np.concatenate([dp, dR])
