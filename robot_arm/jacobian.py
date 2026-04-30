"""
Geometric Jacobian for a revolute-joint serial manipulator.

For joint i (revolute), the Jacobian column is:

    J_i = [ z_{i-1} × (p_e - p_{i-1}) ]
          [         z_{i-1}            ]

where
  z_{i-1}  = 3rd column of T_{0,i-1}  (rotation axis in global frame)
  p_{i-1}  = translation of T_{0,i-1}
  p_e      = end-effector position

References
----------
  Siciliano et al. "Robotics: Modelling, Planning and Control", Springer 2010.
"""

import numpy as np
from .forward_kinematics import forward_kinematics


def geometric_jacobian(
    joint_angles: np.ndarray,
    dh_params: np.ndarray,
) -> np.ndarray:
    """
    Compute the (6 × n) geometric Jacobian at the given joint configuration.

    Parameters
    ----------
    joint_angles : (n,) ndarray
    dh_params    : (n, 4) ndarray  [a, alpha, d, theta_offset]

    Returns
    -------
    J : (6, n) ndarray
        Rows 0-2 → linear velocity (translational) part.
        Rows 3-5 → angular velocity (rotational) part.
    """
    transforms = forward_kinematics(joint_angles, dh_params)
    n = len(joint_angles)

    p_e = transforms[-1][:3, 3]   # end-effector position
    J = np.zeros((6, n))

    for i in range(n):
        if i == 0:
            # Frame 0 = world/base frame
            z_prev = np.array([0.0, 0.0, 1.0])
            p_prev = np.zeros(3)
        else:
            T_prev = transforms[i - 1]
            z_prev = T_prev[:3, 2]
            p_prev = T_prev[:3, 3]

        # Linear part: z_{i-1} × (p_e - p_{i-1})
        J[:3, i] = np.cross(z_prev, p_e - p_prev)
        # Angular part: z_{i-1}
        J[3:, i] = z_prev

    return J


def manipulability_index(J: np.ndarray) -> float:
    """
    Yoshikawa manipulability measure: w = sqrt(det(J · Jᵀ)).

    A value near zero indicates a singular (degenerate) configuration.

    Parameters
    ----------
    J : (6, n) ndarray  Full geometric Jacobian.

    Returns
    -------
    w : float ≥ 0
    """
    JJt = J @ J.T
    det = np.linalg.det(JJt)
    return float(np.sqrt(max(det, 0.0)))


def minimum_singular_value(J: np.ndarray) -> float:
    """
    Return the smallest singular value of J.

    Complementary to manipulability: faster to compute for large n
    and directly comparable against a threshold.
    """
    _, sv, _ = np.linalg.svd(J, full_matrices=False)
    return float(sv[-1])


def is_singular(J: np.ndarray, threshold: float = 1e-3) -> bool:
    """True when the arm is near a kinematic singularity."""
    return minimum_singular_value(J) < threshold


def condition_number(J: np.ndarray) -> float:
    """
    Jacobian condition number κ = σ_max / σ_min.

    Large κ → near-singular; κ = 1 → isotropic (ideal).
    """
    _, sv, _ = np.linalg.svd(J, full_matrices=False)
    if sv[-1] < 1e-12:
        return float("inf")
    return float(sv[0] / sv[-1])
