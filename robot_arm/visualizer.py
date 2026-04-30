"""
Interactive 3D Visualizer — Matplotlib.

Features
--------
• 3D arm rendering with joint spheres and link cylinders
• Sliders for each joint angle
• Real-time FK display (EE position + Euler angles)
• Singularity and manipulability indicators
• Target marker with IK solve button
• Trajectory playback
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from typing import Optional, List

from .config import VIZ_CONFIG


class RobotVisualizer:
    """
    Interactive Matplotlib 3D visualizer for a Robot6DOF instance.

    Usage
    -----
    >>> viz = RobotVisualizer(robot)
    >>> viz.show()
    """

    def __init__(self, robot, figsize=None):
        self.robot   = robot
        self.fig_sz  = figsize or VIZ_CONFIG["figure_size"]
        self.cfg     = VIZ_CONFIG
        self._traj   : Optional[np.ndarray] = None
        self._target : Optional[np.ndarray] = None
        self._setup_figure()

    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------

    def _setup_figure(self):
        self.fig = plt.figure(figsize=self.fig_sz, facecolor="#1a1a2e")
        self.fig.canvas.manager.set_window_title("Robot 6-DOF Simulator")

        # --- 3D axes (left 60 %) ---
        self.ax3d = self.fig.add_axes([0.02, 0.08, 0.55, 0.88], projection="3d")
        self.ax3d.set_facecolor("#16213e")
        self.ax3d.set_box_aspect([1, 1, 1])

        # --- Info panel (right 40 %) ---
        self.ax_info = self.fig.add_axes([0.60, 0.50, 0.38, 0.46])
        self.ax_info.set_facecolor("#0f3460")
        self.ax_info.axis("off")

        # --- Manipulability bar ---
        self.ax_man = self.fig.add_axes([0.60, 0.38, 0.38, 0.06])

        # --- Joint sliders ---
        self.sliders: List[Slider] = []
        slider_colors = [
            "#e94560", "#0f3460", "#533483",
            "#e94560", "#0f3460", "#533483",
        ]
        for i in range(self.robot.n_joints):
            lo, hi = self.robot.joint_limits[i]
            ax_s = self.fig.add_axes([0.62, 0.28 - i * 0.045, 0.35, 0.025])
            sl = Slider(
                ax_s,
                f"J{i+1}",
                np.degrees(lo),
                np.degrees(hi),
                valinit=np.degrees(self.robot.q[i]),
                color=slider_colors[i % len(slider_colors)],
            )
            sl.on_changed(self._on_slider_change)
            self.sliders.append(sl)

        # --- Buttons ---
        ax_home = self.fig.add_axes([0.62, 0.04, 0.10, 0.04])
        self.btn_home = Button(ax_home, "Home", color="#533483", hovercolor="#7b52ab")
        self.btn_home.on_clicked(self._on_home)

        ax_rand = self.fig.add_axes([0.74, 0.04, 0.10, 0.04])
        self.btn_rand = Button(ax_rand, "Random", color="#0f3460", hovercolor="#1a5276")
        self.btn_rand.on_clicked(self._on_random)

        ax_traj = self.fig.add_axes([0.86, 0.04, 0.12, 0.04])
        self.btn_traj = Button(ax_traj, "▶ Traj", color="#e94560", hovercolor="#ff6b6b")
        self.btn_traj.on_clicked(self._on_play_trajectory)

        self._draw()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self):
        self.ax3d.cla()
        self.ax3d.set_facecolor("#16213e")

        q = self.robot.q
        positions = self.robot.joint_positions(q)  # (n+1, 3)

        # Determine axis limits from arm reach
        reach = np.sum(self.robot.dh_params[:, 0]) + np.sum(np.abs(self.robot.dh_params[:, 2]))
        lim = max(reach * 0.65, 0.6)

        self.ax3d.set_xlim(-lim, lim)
        self.ax3d.set_ylim(-lim, lim)
        self.ax3d.set_zlim(0, 2 * lim)

        # --- Grid & labels ---
        self.ax3d.set_xlabel("X", color="#aaaaaa", fontsize=8)
        self.ax3d.set_ylabel("Y", color="#aaaaaa", fontsize=8)
        self.ax3d.set_zlabel("Z", color="#aaaaaa", fontsize=8)
        self.ax3d.tick_params(colors="#555555", labelsize=6)
        for pane in [self.ax3d.xaxis.pane, self.ax3d.yaxis.pane, self.ax3d.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#333333")

        # --- Links ---
        for i in range(len(positions) - 1):
            xs = [positions[i, 0], positions[i+1, 0]]
            ys = [positions[i, 1], positions[i+1, 1]]
            zs = [positions[i, 2], positions[i+1, 2]]
            self.ax3d.plot(xs, ys, zs, "-", color=self.cfg["link_color"],
                           linewidth=4, solid_capstyle="round", zorder=3)

        # --- Joint spheres ---
        for i, pos in enumerate(positions[:-1]):
            self.ax3d.scatter(*pos, s=120, c=self.cfg["joint_color"],
                              depthshade=False, zorder=4)

        # --- End-effector ---
        ee = positions[-1]
        self.ax3d.scatter(*ee, s=180, c=self.cfg["ee_color"], marker="*",
                          depthshade=False, zorder=5)

        # --- Target ---
        if self._target is not None:
            self.ax3d.scatter(*self._target[:3, 3], s=200,
                              c=self.cfg["target_color"], marker="x",
                              linewidths=3, zorder=5)

        # --- Trajectory trace ---
        if self._traj is not None and len(self._traj) > 1:
            pts = np.array([self.robot.joint_positions(q_i)[-1]
                            for q_i in self._traj])
            self.ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                           "--", color=self.cfg["trajectory_color"],
                           linewidth=1.5, alpha=0.7)

        # --- Frame axes at EE ---
        T_ee = self.robot.fk(q)
        origin = T_ee[:3, 3]
        scale  = lim * 0.15
        for col, axis_col in zip(range(3), ["#e74c3c", "#2ecc71", "#3498db"]):
            direction = T_ee[:3, col]
            end = origin + scale * direction
            self.ax3d.quiver(*origin, *direction * scale,
                             color=axis_col, linewidth=1.5, alpha=0.9)

        # --- Info panel ---
        self._update_info(q, ee, T_ee)

        self.fig.canvas.draw_idle()

    def _update_info(self, q, ee_pos, T_ee):
        self.ax_info.cla()
        self.ax_info.set_facecolor("#0f3460")
        self.ax_info.axis("off")

        sing_report = self.robot.check_singularity(q)
        man          = sing_report["manipulability"]
        singular     = sing_report["singular"]
        cond         = sing_report["condition"]
        euler        = self.robot.ee_euler(q)

        color_sing = "#e74c3c" if singular else "#2ecc71"
        sing_txt   = "⚠ SINGULAR" if singular else "✓ Normal"

        lines = [
            ("End-Effector Position", "", "#aaaaaa", 11),
            (f"  X = {ee_pos[0]:+.4f} m", "", "#ffffff", 10),
            (f"  Y = {ee_pos[1]:+.4f} m", "", "#ffffff", 10),
            (f"  Z = {ee_pos[2]:+.4f} m", "", "#ffffff", 10),
            ("", "", "#aaaaaa", 10),
            ("EE Orientation (ZYX Euler)", "", "#aaaaaa", 11),
            (f"  Yaw   = {np.degrees(euler[0]):+.2f}°", "", "#ffffff", 10),
            (f"  Pitch = {np.degrees(euler[1]):+.2f}°", "", "#ffffff", 10),
            (f"  Roll  = {np.degrees(euler[2]):+.2f}°", "", "#ffffff", 10),
            ("", "", "#aaaaaa", 10),
            ("Jacobian Analysis", "", "#aaaaaa", 11),
            (f"  Manipulability: {man:.5f}", "", "#f39c12", 10),
            (f"  Condition #:    {cond:.2f}", "", "#f39c12", 10),
            (f"  Singularity:    {sing_txt}", "", color_sing, 10),
        ]

        y = 0.96
        for txt, _, col, fs in lines:
            self.ax_info.text(0.05, y, txt, transform=self.ax_info.transAxes,
                              color=col, fontsize=fs, va="top", family="monospace")
            y -= 0.065

        # Manipulability bar
        self.ax_man.cla()
        max_man = 0.1  # approximate max for visual scaling
        ratio = min(man / max_man, 1.0)
        bar_color = "#e74c3c" if ratio < 0.1 else ("#f39c12" if ratio < 0.4 else "#2ecc71")
        self.ax_man.barh(0, ratio, color=bar_color, height=0.5)
        self.ax_man.barh(0, 1.0 - ratio, left=ratio, color="#333333", height=0.5)
        self.ax_man.set_xlim(0, 1)
        self.ax_man.set_ylim(-0.5, 0.5)
        self.ax_man.axis("off")
        self.ax_man.text(0.5, 0, f"Manipulability: {man:.4f}",
                         ha="center", va="center", color="white",
                         fontsize=9, transform=self.ax_man.transAxes)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_slider_change(self, _):
        q = np.radians([sl.val for sl in self.sliders])
        self.robot.q = q
        self._draw()

    def _on_home(self, _):
        self.robot.home()
        for i, sl in enumerate(self.sliders):
            sl.set_val(np.degrees(self.robot.q[i]))

    def _on_random(self, _):
        q_rand = self.robot.random_config()
        self.robot.q = q_rand
        for i, sl in enumerate(self.sliders):
            sl.set_val(np.degrees(self.robot.q[i]))

    def _on_play_trajectory(self, _):
        if self._traj is None:
            print("[Visualizer] No trajectory loaded. Call set_trajectory() first.")
            return
        for q_i in self._traj:
            self.robot.q = q_i
            for i, sl in enumerate(self.sliders):
                sl.set_val(np.degrees(q_i[i]))
            plt.pause(0.03)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_target(self, pose: np.ndarray):
        """Mark a target end-effector pose on the visualizer."""
        self._target = pose
        self._draw()

    def set_trajectory(self, traj: np.ndarray):
        """Load a (T, 6) joint trajectory for playback."""
        self._traj = traj
        self._draw()

    def show(self):
        """Block and display the interactive window."""
        plt.show()


# ---------------------------------------------------------------------------
# Convergence plot (IK analysis)
# ---------------------------------------------------------------------------

def plot_ik_convergence(results: dict, title: str = "IK Convergence"):
    """
    Plot position error vs iteration for multiple IK methods.

    Parameters
    ----------
    results : dict  {method_name: IKResult}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    colors = {"transpose": "#e74c3c", "pinv": "#3498db", "dls": "#2ecc71"}

    for name, result in results.items():
        if not result.history:
            continue
        iters = [h[0] for h in result.history]
        errs  = [h[1] * 1000 for h in result.history]  # mm
        mans  = [h[2] for h in result.history]
        col   = colors.get(name, "#aaaaaa")

        axes[0].semilogy(iters, errs, label=f"{name} ({result.iterations} iter)", color=col)
        axes[1].plot(iters, mans, label=name, color=col)

    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Position error (mm)")
    axes[0].set_title("Position error convergence")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Manipulability index")
    axes[1].set_title("Manipulability during IK")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
