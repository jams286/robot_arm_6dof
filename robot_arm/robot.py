"""
Robot6DOF — main class.

Provides a unified interface for:
  • Forward kinematics
  • Inverse kinematics (three solvers)
  • Jacobian analysis
  • Joint-limit management
  • Configuration presets
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Literal

from .config import PRESETS, IK_CONFIG
from .forward_kinematics import (
    forward_kinematics,
    get_end_effector_pose,
    get_joint_positions,
    rotation_matrix_to_euler_zyx,
    pose_error,
)
from .jacobian import (
    geometric_jacobian,
    manipulability_index,
    is_singular,
    condition_number,
    minimum_singular_value,
)
from .inverse_kinematics import (
    ik_jacobian_transpose,
    ik_pseudoinverse,
    ik_dls,
    IKResult,
    IKStatus,
)


class Robot6DOF:
    """
    6-DOF revolute serial manipulator.

    Parameters
    ----------
    preset : str
        One of 'ur5_like' or 'simple'. Loads predefined DH table and limits.
    dh_params : ndarray (optional)
        Custom (6, 4) DH table [a, alpha, d, theta_offset].
    joint_limits : ndarray (optional)
        Custom (6, 2) limits [[min, max], ...] in radians.

    Examples
    --------
    >>> robot = Robot6DOF(preset='ur5_like')
    >>> T = robot.fk([0, -np.pi/4, np.pi/2, 0, np.pi/4, 0])
    >>> result = robot.ik(T, method='dls')
    """

    def __init__(
        self,
        preset      : str = "ur5_like",
        dh_params   : Optional[np.ndarray] = None,
        joint_limits: Optional[np.ndarray] = None,
    ):
        cfg = PRESETS.get(preset, PRESETS["ur5_like"])
        self.name         = cfg["name"]
        self.dh_params    = dh_params    if dh_params    is not None else cfg["dh"].copy()
        self.joint_limits = joint_limits if joint_limits is not None else cfg["joint_limits"].copy()
        self.n_joints     = len(self.dh_params)
        self._q           = np.zeros(self.n_joints)   # current configuration

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def q(self) -> np.ndarray:
        """Current joint angles (radians)."""
        return self._q.copy()

    @q.setter
    def q(self, angles: np.ndarray):
        angles = np.asarray(angles, dtype=float)
        clipped = np.clip(angles, self.joint_limits[:, 0], self.joint_limits[:, 1])
        if not np.allclose(angles, clipped):
            print("[Robot6DOF] Warning: joint angles clipped to limits.")
        self._q = clipped

    # ------------------------------------------------------------------
    # Forward Kinematics
    # ------------------------------------------------------------------

    def fk(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute end-effector pose for joint angles q (or current _q).

        Returns
        -------
        T_ee : (4, 4) homogeneous transform.
        """
        if q is not None:
            self.q = q
        return get_end_effector_pose(self._q, self.dh_params)

    def fk_all_frames(self, q: Optional[np.ndarray] = None):
        """Return list of (4,4) transforms for every frame (base → EE)."""
        if q is not None:
            self.q = q
        return forward_kinematics(self._q, self.dh_params)

    def joint_positions(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """(n+1, 3) array of XYZ joint origins including base."""
        if q is not None:
            self.q = q
        return get_joint_positions(self._q, self.dh_params)

    def ee_position(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """(3,) XYZ end-effector position."""
        T = self.fk(q)
        return T[:3, 3]

    def ee_euler(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """(3,) ZYX Euler angles [yaw, pitch, roll] in radians."""
        T = self.fk(q)
        return rotation_matrix_to_euler_zyx(T[:3, :3])

    # ------------------------------------------------------------------
    # Jacobian
    # ------------------------------------------------------------------

    def jacobian(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """(6, 6) geometric Jacobian at configuration q."""
        if q is not None:
            self.q = q
        return geometric_jacobian(self._q, self.dh_params)

    def manipulability(self, q: Optional[np.ndarray] = None) -> float:
        """Yoshikawa manipulability index w = sqrt(det(J·Jᵀ))."""
        J = self.jacobian(q)
        return manipulability_index(J)

    def condition(self, q: Optional[np.ndarray] = None) -> float:
        """Jacobian condition number κ = σ_max / σ_min."""
        J = self.jacobian(q)
        return condition_number(J)

    def check_singularity(
        self,
        q: Optional[np.ndarray] = None,
        threshold: float = 1e-3,
    ) -> dict:
        """
        Detailed singularity report.

        Returns
        -------
        dict with keys: singular (bool), sigma_min (float),
                        manipulability (float), condition (float).
        """
        J = self.jacobian(q)
        sig_min = minimum_singular_value(J)
        return {
            "singular"        : sig_min < threshold,
            "sigma_min"       : sig_min,
            "manipulability"  : manipulability_index(J),
            "condition"       : condition_number(J),
        }

    # ------------------------------------------------------------------
    # Inverse Kinematics
    # ------------------------------------------------------------------

    def ik(
        self,
        target_pose : np.ndarray,
        method      : Literal["transpose", "pinv", "dls"] = "dls",
        initial_q   : Optional[np.ndarray] = None,
        **kwargs,
    ) -> IKResult:
        """
        Solve IK for a target end-effector pose.

        Parameters
        ----------
        target_pose : (4, 4) homogeneous target transform.
        method      : 'transpose' | 'pinv' | 'dls' (default).
        initial_q   : Starting configuration; defaults to current _q.
        **kwargs    : Override any IK_CONFIG parameter.

        Returns
        -------
        IKResult dataclass.
        """
        q0  = initial_q.copy() if initial_q is not None else self._q.copy()
        cfg = {**IK_CONFIG, **kwargs}

        common = dict(
            target_pose   = target_pose,
            initial_q     = q0,
            dh_params     = self.dh_params,
            joint_limits  = self.joint_limits,
            max_iter      = cfg["max_iter"],
            tol_pos       = cfg["tol_pos"],
            tol_ori       = cfg["tol_ori"],
            pos_weight    = cfg["pos_weight"],
            ori_weight    = cfg["ori_weight"],
            singularity_thresh = cfg["singularity_thresh"],
        )

        if method == "transpose":
            result = ik_jacobian_transpose(**common, alpha=cfg["alpha_transpose"])
        elif method == "pinv":
            result = ik_pseudoinverse(**common, alpha=cfg["alpha_pinv"])
        elif method == "dls":
            result = ik_dls(**common, alpha=cfg["alpha_pinv"], damping=cfg["damping"])
        else:
            raise ValueError(f"Unknown IK method '{method}'. Choose: transpose, pinv, dls.")

        if result.success:
            self._q = result.joint_angles.copy()

        return result

    def ik_position_only(
        self,
        target_xyz : np.ndarray,
        method     : str = "dls",
        initial_q  : Optional[np.ndarray] = None,
        **kwargs,
    ) -> IKResult:
        """
        Convenience: IK targeting only XYZ position (orientation free).
        Builds a target pose using current EE orientation + new position.
        """
        T_cur = self.fk()
        T_target = T_cur.copy()
        T_target[:3, 3] = target_xyz
        return self.ik(T_target, method=method, initial_q=initial_q,
                       ori_weight=0.0, **kwargs)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def random_config(self) -> np.ndarray:
        """Sample a random valid joint configuration."""
        lo, hi = self.joint_limits[:, 0], self.joint_limits[:, 1]
        return np.random.uniform(lo, hi)

    def home(self):
        """Move to the all-zeros (home) configuration."""
        self._q = np.zeros(self.n_joints)

    def __repr__(self) -> str:
        pos = self.ee_position()
        return (
            f"Robot6DOF('{self.name}')\n"
            f"  Joints: {self.n_joints}\n"
            f"  q      = {np.degrees(self._q).round(2)} deg\n"
            f"  EE pos = {pos.round(4)} m\n"
            f"  Manip  = {self.manipulability():.5f}"
        )
