"""
Inverse Kinematics — iterative numerical solvers.

Three methods are implemented, all sharing the same interface:

1. Jacobian Transpose (JT)
   Δθ = α · Jᵀ · e           — simple, always stable.

2. Jacobian Pseudoinverse (J⁺)
   Δθ = J⁺ · e  where  J⁺ = Jᵀ(JJᵀ)⁻¹  via SVD.
   Minimum-norm solution; fast convergence but unstable near singularities.

3. Damped Least Squares (DLS)  ← recommended for production
   J⁺_λ = Jᵀ(JJᵀ + λ²I)⁻¹  — trades accuracy for numerical stability.
   λ (damping) prevents blow-up near singular configurations.

All solvers:
  - Apply joint limits after every step.
  - Detect and report singularities.
  - Return a dataclass with the full solution history.

References
----------
  Buss, S. R. "Introduction to Inverse Kinematics with Jacobian
  Transpose, Pseudoinverse and Damped Least Squares methods." 2004.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum

from .forward_kinematics import get_end_effector_pose, pose_error
from .jacobian import (
    geometric_jacobian,
    manipulability_index,
    is_singular,
    minimum_singular_value,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class IKStatus(Enum):
    CONVERGED        = "converged"
    MAX_ITER         = "max_iterations_reached"
    SINGULAR         = "singular_configuration"
    JOINT_LIMIT_CLIP = "joint_limits_active"


@dataclass
class IKResult:
    joint_angles  : np.ndarray            # Final joint angles
    success       : bool
    status        : IKStatus
    iterations    : int
    pos_error     : float                 # metres
    ori_error     : float                 # radians
    manipulability: float
    history       : list = field(default_factory=list)  # (iter, pos_err, man.)

    def __str__(self) -> str:
        return (
            f"IKResult | {self.status.value}\n"
            f"  Iterations   : {self.iterations}\n"
            f"  Pos error    : {self.pos_error*1000:.4f} mm\n"
            f"  Ori error    : {np.degrees(self.ori_error):.4f}°\n"
            f"  Manipulability: {self.manipulability:.6f}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_joint_limits(
    q: np.ndarray,
    limits: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """Clip joint angles to [min, max]; return clipped array and clipping flag."""
    clipped = np.clip(q, limits[:, 0], limits[:, 1])
    active  = not np.allclose(q, clipped)
    return clipped, active


def _split_error(
    error6: np.ndarray,
    pos_weight: float = 1.0,
    ori_weight: float = 0.3,
) -> Tuple[np.ndarray, float, float]:
    """Return weighted 6-D error and scalar norms for pos/ori."""
    w = np.array([pos_weight]*3 + [ori_weight]*3)
    weighted = w * error6
    pos_err = float(np.linalg.norm(error6[:3]))
    ori_err = float(np.linalg.norm(error6[3:]))
    return weighted, pos_err, ori_err


# ---------------------------------------------------------------------------
# Solver 1 — Jacobian Transpose
# ---------------------------------------------------------------------------

def ik_jacobian_transpose(
    target_pose   : np.ndarray,
    initial_q     : np.ndarray,
    dh_params     : np.ndarray,
    joint_limits  : np.ndarray,
    max_iter      : int   = 1000,
    tol_pos       : float = 1e-4,
    tol_ori       : float = 1e-3,
    alpha         : float = 0.5,
    pos_weight    : float = 1.0,
    ori_weight    : float = 0.3,
    singularity_thresh: float = 1e-3,
) -> IKResult:
    """
    Jacobian Transpose IK.

    Update rule:  Δθ = α · Jᵀ · e_w
    """
    q       = initial_q.copy()
    history = []
    status  = IKStatus.MAX_ITER
    clipped = False

    for i in range(max_iter):
        T_cur   = get_end_effector_pose(q, dh_params)
        e6      = pose_error(T_cur, target_pose)
        J       = geometric_jacobian(q, dh_params)
        man     = manipulability_index(J)

        e_w, pos_err, ori_err = _split_error(e6, pos_weight, ori_weight)
        history.append((i, pos_err, man))

        if pos_err < tol_pos and ori_err < tol_ori:
            status = IKStatus.CONVERGED
            break

        if is_singular(J, singularity_thresh):
            status = IKStatus.SINGULAR
            break

        dq = alpha * J.T @ e_w
        q, clip_active = _apply_joint_limits(q + dq, joint_limits)
        if clip_active:
            clipped = True

    T_final  = get_end_effector_pose(q, dh_params)
    e_final  = pose_error(T_final, target_pose)
    J_final  = geometric_jacobian(q, dh_params)

    if clipped and status == IKStatus.MAX_ITER:
        status = IKStatus.JOINT_LIMIT_CLIP

    return IKResult(
        joint_angles   = q,
        success        = status == IKStatus.CONVERGED,
        status         = status,
        iterations     = i + 1,
        pos_error      = float(np.linalg.norm(e_final[:3])),
        ori_error      = float(np.linalg.norm(e_final[3:])),
        manipulability = manipulability_index(J_final),
        history        = history,
    )


# ---------------------------------------------------------------------------
# Solver 2 — Jacobian Pseudoinverse (SVD-based)
# ---------------------------------------------------------------------------

def ik_pseudoinverse(
    target_pose   : np.ndarray,
    initial_q     : np.ndarray,
    dh_params     : np.ndarray,
    joint_limits  : np.ndarray,
    max_iter      : int   = 1000,
    tol_pos       : float = 1e-4,
    tol_ori       : float = 1e-3,
    alpha         : float = 1.0,
    sv_threshold  : float = 1e-4,
    pos_weight    : float = 1.0,
    ori_weight    : float = 0.3,
    singularity_thresh: float = 1e-3,
) -> IKResult:
    """
    Jacobian Pseudoinverse IK via truncated SVD.

    J⁺ = V · Σ⁺ · Uᵀ   (singular values < sv_threshold are zeroed)
    Update: Δθ = α · J⁺ · e_w
    """
    q       = initial_q.copy()
    history = []
    status  = IKStatus.MAX_ITER
    clipped = False

    for i in range(max_iter):
        T_cur = get_end_effector_pose(q, dh_params)
        e6    = pose_error(T_cur, target_pose)
        J     = geometric_jacobian(q, dh_params)
        man   = manipulability_index(J)

        e_w, pos_err, ori_err = _split_error(e6, pos_weight, ori_weight)
        history.append((i, pos_err, man))

        if pos_err < tol_pos and ori_err < tol_ori:
            status = IKStatus.CONVERGED
            break

        if is_singular(J, singularity_thresh):
            status = IKStatus.SINGULAR
            break

        # SVD-based pseudoinverse
        U, s, Vt = np.linalg.svd(J, full_matrices=False)
        s_inv = np.where(s > sv_threshold, 1.0 / s, 0.0)
        J_pinv = Vt.T @ np.diag(s_inv) @ U.T

        dq = alpha * J_pinv @ e_w
        q, clip_active = _apply_joint_limits(q + dq, joint_limits)
        if clip_active:
            clipped = True

    T_final = get_end_effector_pose(q, dh_params)
    e_final = pose_error(T_final, target_pose)
    J_final = geometric_jacobian(q, dh_params)

    if clipped and status == IKStatus.MAX_ITER:
        status = IKStatus.JOINT_LIMIT_CLIP

    return IKResult(
        joint_angles   = q,
        success        = status == IKStatus.CONVERGED,
        status         = status,
        iterations     = i + 1,
        pos_error      = float(np.linalg.norm(e_final[:3])),
        ori_error      = float(np.linalg.norm(e_final[3:])),
        manipulability = manipulability_index(J_final),
        history        = history,
    )


# ---------------------------------------------------------------------------
# Solver 3 — Damped Least Squares (DLS)  ← recommended
# ---------------------------------------------------------------------------

def ik_dls(
    target_pose   : np.ndarray,
    initial_q     : np.ndarray,
    dh_params     : np.ndarray,
    joint_limits  : np.ndarray,
    max_iter      : int   = 1000,
    tol_pos       : float = 1e-4,
    tol_ori       : float = 1e-3,
    alpha         : float = 1.0,
    damping       : float = 0.05,
    pos_weight    : float = 1.0,
    ori_weight    : float = 0.3,
    singularity_thresh: float = 1e-3,
    adaptive_damping: bool = True,
) -> IKResult:
    """
    Damped Least Squares IK (Levenberg-Marquardt style).

    J⁺_λ = Jᵀ(JJᵀ + λ²I)⁻¹
    Update: Δθ = α · J⁺_λ · e_w

    With adaptive_damping=True, λ scales with the minimum singular value
    to be more aggressive only near singularities.
    """
    q       = initial_q.copy()
    history = []
    status  = IKStatus.MAX_ITER
    clipped = False
    lam     = damping

    for i in range(max_iter):
        T_cur = get_end_effector_pose(q, dh_params)
        e6    = pose_error(T_cur, target_pose)
        J     = geometric_jacobian(q, dh_params)
        man   = manipulability_index(J)

        e_w, pos_err, ori_err = _split_error(e6, pos_weight, ori_weight)
        history.append((i, pos_err, man))

        if pos_err < tol_pos and ori_err < tol_ori:
            status = IKStatus.CONVERGED
            break

        # Adaptive damping: larger λ near singularities
        if adaptive_damping:
            sigma_min = minimum_singular_value(J)
            lam = damping if sigma_min > singularity_thresh else damping * 10.0

        # DLS pseudoinverse
        m = J.shape[0]
        JJt     = J @ J.T
        J_dls   = J.T @ np.linalg.inv(JJt + lam**2 * np.eye(m))

        dq = alpha * J_dls @ e_w
        q, clip_active = _apply_joint_limits(q + dq, joint_limits)
        if clip_active:
            clipped = True

    T_final = get_end_effector_pose(q, dh_params)
    e_final = pose_error(T_final, target_pose)
    J_final = geometric_jacobian(q, dh_params)

    if clipped and status == IKStatus.MAX_ITER:
        status = IKStatus.JOINT_LIMIT_CLIP

    return IKResult(
        joint_angles   = q,
        success        = status == IKStatus.CONVERGED,
        status         = status,
        iterations     = i + 1,
        pos_error      = float(np.linalg.norm(e_final[:3])),
        ori_error      = float(np.linalg.norm(e_final[3:])),
        manipulability = manipulability_index(J_final),
        history        = history,
    )
