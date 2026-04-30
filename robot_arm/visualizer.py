"""
Interactive 3D Visualizer — Matplotlib.

Features
--------
• 3D arm rendering with cylindrical links and joint spheres
• Coordinate frames at every joint (RGB = XYZ)
• Joint rotation-axis arrows to clarify DOF orientation
• Ground plane with shadow projection for depth perception
• Sliders for each joint angle
• Real-time FK display (EE position + Euler angles)
• Singularity and manipulability indicators
• Target marker with IK solve button
• Trajectory playback
• Toggle: joint frames / rotation axes / shadow
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, List

from .config import VIZ_CONFIG
from .trajectory import multi_segment_trajectory


# ------------------------------------------------------------------
# Helper — draw a cylinder between two 3D points
# ------------------------------------------------------------------

def _cylinder_between(ax, p0, p1, radius=0.02, n_sides=12, color="#4A90D9", alpha=0.85):
    """Draw a faceted cylinder from p0 to p1."""
    v = np.array(p1) - np.array(p0)
    length = np.linalg.norm(v)
    if length < 1e-6:
        return
    v_hat = v / length

    # Build an arbitrary perpendicular vector
    if abs(v_hat[0]) < 0.9:
        perp = np.cross(v_hat, [1, 0, 0])
    else:
        perp = np.cross(v_hat, [0, 1, 0])
    perp /= np.linalg.norm(perp)
    perp2 = np.cross(v_hat, perp)

    theta = np.linspace(0, 2 * np.pi, n_sides + 1)
    circle = np.array([radius * (np.cos(t) * perp + np.sin(t) * perp2)
                        for t in theta])

    bottom = np.array(p0) + circle
    top    = np.array(p0) + v + circle

    # Side faces
    verts = []
    for i in range(n_sides):
        quad = [bottom[i], bottom[i+1], top[i+1], top[i]]
        verts.append(quad)
    poly = Poly3DCollection(verts, alpha=alpha, linewidths=0.3,
                            edgecolors="#222222")
    poly.set_facecolor(color)
    ax.add_collection3d(poly)


class RobotVisualizer:
    """
    Interactive Matplotlib 3D visualizer for a Robot6DOF instance.

    Usage
    -----
    >>> viz = RobotVisualizer(robot)
    >>> viz.show()
    """

    # Per-link colour gradient (base→EE)
    LINK_COLORS = [
        "#4A90D9", "#5B7FD9", "#6C6ED9",
        "#7D5DD9", "#8E4CD9", "#9F3BD9",
    ]

    def __init__(self, robot, figsize=None):
        self.robot   = robot
        self.fig_sz  = figsize or VIZ_CONFIG["figure_size"]
        self.cfg     = VIZ_CONFIG
        self._traj   : Optional[np.ndarray] = None
        self._target : Optional[np.ndarray] = None
        self._waypoints: List[np.ndarray] = []
        # Toggle states
        self._show_frames  = True
        self._show_axes    = True
        self._show_shadow  = True
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

        # --- Info panel (right side) ---
        self.ax_info = self.fig.add_axes([0.60, 0.55, 0.38, 0.41])
        self.ax_info.set_facecolor("#0f3460")
        self.ax_info.axis("off")

        # --- Check-buttons for toggles (below info panel) ---
        ax_chk = self.fig.add_axes([0.60, 0.45, 0.12, 0.09],
                                   facecolor="#c8d0e0")
        self.chk = CheckButtons(ax_chk,
                                ["Frames", "Rot Axes", "Shadow"],
                                [True, True, True])
        for label in self.chk.labels:
            label.set_fontsize(7)
            label.set_color("#1a1a2e")
        self.chk.on_clicked(self._on_toggle)

        # --- Manipulability bar ---
        self.ax_man = self.fig.add_axes([0.74, 0.45, 0.24, 0.06])

        # --- Joint sliders ---
        self.sliders: List[Slider] = []
        slider_colors = [
            "#e94560", "#0f3460", "#533483",
            "#e94560", "#0f3460", "#533483",
        ]
        for i in range(self.robot.n_joints):
            lo, hi = self.robot.joint_limits[i]
            ax_s = self.fig.add_axes([0.62, 0.34 - i * 0.04, 0.35, 0.025])
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
        ax_home = self.fig.add_axes([0.62, 0.04, 0.07, 0.04])
        self.btn_home = Button(ax_home, "Home", color="#533483", hovercolor="#7b52ab")
        self.btn_home.on_clicked(self._on_home)

        ax_rand = self.fig.add_axes([0.70, 0.04, 0.07, 0.04])
        self.btn_rand = Button(ax_rand, "Random", color="#0f3460", hovercolor="#1a5276")
        self.btn_rand.on_clicked(self._on_random)

        ax_save = self.fig.add_axes([0.78, 0.04, 0.06, 0.04])
        self.btn_save = Button(ax_save, "+ Pt", color="#27ae60", hovercolor="#2ecc71")
        self.btn_save.on_clicked(self._on_save_waypoint)

        ax_clear = self.fig.add_axes([0.845, 0.04, 0.06, 0.04])
        self.btn_clear = Button(ax_clear, "Clear", color="#c0392b", hovercolor="#e74c3c")
        self.btn_clear.on_clicked(self._on_clear_waypoints)

        ax_traj = self.fig.add_axes([0.91, 0.04, 0.07, 0.04])
        self.btn_traj = Button(ax_traj, "\u25b6 Traj", color="#e94560", hovercolor="#ff6b6b")
        self.btn_traj.on_clicked(self._on_play_trajectory)

        # --- IK target input ---
        ax_ik_label = self.fig.add_axes([0.62, 0.13, 0.06, 0.025])
        ax_ik_label.axis("off")
        ax_ik_label.text(0.5, 0.5, "XYZ:", ha="center", va="center",
                         fontsize=9, color="white", fontweight="bold")
        ax_ik_text = self.fig.add_axes([0.68, 0.13, 0.20, 0.025])
        self.txt_ik = TextBox(ax_ik_text, "", initial="0.3, 0.0, 0.5")
        ax_ik_go = self.fig.add_axes([0.89, 0.13, 0.09, 0.025])
        self.btn_ik = Button(ax_ik_go, "IK Go", color="#1abc9c", hovercolor="#2ecc71")
        self.btn_ik.on_clicked(self._on_ik_go)

        # --- Speed slider ---
        ax_speed = self.fig.add_axes([0.62, 0.09, 0.36, 0.02])
        self.slider_speed = Slider(
            ax_speed, "Speed", 0.5, 5.0, valinit=1.0,
            color="#e94560", valstep=0.5,
        )

        # --- Camera view buttons (below 3D viewport) ---
        view_defs = [
            ("Persp",  25, -60),
            ("Front",   0,   0),
            ("Side",    0,  90),
            ("Top",    90, -90),
            ("Back",    0, 180),
        ]
        self._view_buttons = []
        n_views = len(view_defs)
        vbtn_w = 0.55 / n_views
        for vi, (vlabel, velev, vazim) in enumerate(view_defs):
            ax_v = self.fig.add_axes([0.02 + vi * vbtn_w, 0.01, vbtn_w - 0.005, 0.035])
            btn_v = Button(ax_v, vlabel, color="#2a3a5a", hovercolor="#3a5a8a")
            btn_v.label.set_fontsize(8)
            btn_v.label.set_color("white")
            btn_v.on_clicked(lambda _, e=velev, a=vazim: self._on_view(e, a))
            self._view_buttons.append(btn_v)

        self._draw()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self):
        self.ax3d.cla()
        self.ax3d.set_facecolor("#16213e")

        q = self.robot.q
        positions = self.robot.joint_positions(q)  # (n+1, 3)
        frames    = self.robot.fk_all_frames(q)    # list of (4,4) per joint

        # Determine axis limits from arm reach
        reach = np.sum(self.robot.dh_params[:, 0]) + np.sum(np.abs(self.robot.dh_params[:, 2]))
        lim = max(reach * 0.65, 0.6)

        self.ax3d.set_xlim(-lim, lim)
        self.ax3d.set_ylim(-lim, lim)
        self.ax3d.set_zlim(0, 2 * lim)

        # --- Grid & labels ---
        self.ax3d.set_xlabel("X (m)", color="#aaaaaa", fontsize=8)
        self.ax3d.set_ylabel("Y (m)", color="#aaaaaa", fontsize=8)
        self.ax3d.set_zlabel("Z (m)", color="#aaaaaa", fontsize=8)
        self.ax3d.tick_params(colors="#555555", labelsize=6)
        for pane in [self.ax3d.xaxis.pane, self.ax3d.yaxis.pane, self.ax3d.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#333333")

        # --- Ground plane (checkerboard-style grid) ---
        gn = 8
        gs = lim * 2 / gn
        for ix in range(gn):
            for iy in range(gn):
                x0 = -lim + ix * gs
                y0 = -lim + iy * gs
                c = "#1c2a4a" if (ix + iy) % 2 == 0 else "#162040"
                verts = [[(x0, y0, 0), (x0+gs, y0, 0),
                          (x0+gs, y0+gs, 0), (x0, y0+gs, 0)]]
                poly = Poly3DCollection(verts, alpha=0.25)
                poly.set_facecolor(c)
                poly.set_edgecolor("#2a3a5a")
                self.ax3d.add_collection3d(poly)

        # --- Base pedestal ---
        self._draw_base_pedestal(positions[0], lim)

        # --- Cylindrical links (with per-link color) ---
        link_radius = lim * 0.025
        for i in range(len(positions) - 1):
            col = self.LINK_COLORS[i % len(self.LINK_COLORS)]
            _cylinder_between(self.ax3d, positions[i], positions[i+1],
                              radius=link_radius, color=col, alpha=0.85)

        # --- Joint spheres ---
        joint_colors = ["#FF6B35", "#FFD166", "#06D6A0",
                        "#118AB2", "#EF476F", "#FFC43D"]
        for i, pos in enumerate(positions[:-1]):
            c = joint_colors[i % len(joint_colors)]
            self.ax3d.scatter(*pos, s=180, c=c, depthshade=True,
                              zorder=4, edgecolors="#222222", linewidths=0.5)

        # --- End-effector (diamond marker) ---
        ee = positions[-1]
        self.ax3d.scatter(*ee, s=250, c=self.cfg["ee_color"], marker="D",
                          depthshade=False, zorder=5, edgecolors="white",
                          linewidths=1.0)

        # --- Coordinate frames at each joint ---
        T_ee = self.robot.fk(q)
        frame_scale = lim * 0.10
        axis_colors = ["#e74c3c", "#2ecc71", "#3498db"]  # R G B = X Y Z

        if self._show_frames:
            # Base frame
            base_origin = np.array([0, 0, 0])
            for col_idx in range(3):
                d = np.zeros(3)
                d[col_idx] = frame_scale
                self.ax3d.quiver(*base_origin, *d,
                                 color=axis_colors[col_idx],
                                 linewidth=1.8, alpha=0.7, arrow_length_ratio=0.15)

            # Joint frames
            for j, T_j in enumerate(frames):
                origin_j = T_j[:3, 3]
                for col_idx in range(3):
                    d = T_j[:3, col_idx] * frame_scale * 0.7
                    self.ax3d.quiver(*origin_j, *d,
                                     color=axis_colors[col_idx],
                                     linewidth=1.2, alpha=0.6,
                                     arrow_length_ratio=0.18)

        # --- Joint rotation-axis arrows (Z-axis of each frame) ---
        if self._show_axes:
            rot_scale = lim * 0.12
            for j, T_j in enumerate(frames):
                if j == 0:
                    # Joint 1 rotates about the base Z
                    origin_r = positions[0]
                    z_axis   = np.array([0, 0, 1.0])
                else:
                    origin_r = frames[j-1][:3, 3] if j > 0 else positions[0]
                    z_axis   = frames[j-1][:3, 2] if j > 0 else np.array([0, 0, 1.0])
                # Draw a ring/arc to show rotation direction
                self.ax3d.quiver(*origin_r, *(z_axis * rot_scale),
                                 color="#FFD700", linewidth=2.5, alpha=0.8,
                                 arrow_length_ratio=0.25)
                # Label
                label_pos = origin_r + z_axis * rot_scale * 1.15
                self.ax3d.text(*label_pos, f"J{j+1}", color="#FFD700",
                               fontsize=7, ha="center", fontweight="bold")

        # --- Shadow projection on ground plane ---
        if self._show_shadow:
            for i in range(len(positions) - 1):
                xs = [positions[i, 0], positions[i+1, 0]]
                ys = [positions[i, 1], positions[i+1, 1]]
                zs = [0.001, 0.001]  # just above z=0
                self.ax3d.plot(xs, ys, zs, "-", color="#555555",
                               linewidth=2.5, alpha=0.4, zorder=1)
            for pos in positions:
                self.ax3d.scatter(pos[0], pos[1], 0.001,
                                  s=40, c="#444444", alpha=0.3,
                                  marker="o", zorder=1)
            # Dashed drop lines from joints to ground
            for pos in positions:
                self.ax3d.plot([pos[0], pos[0]], [pos[1], pos[1]],
                               [0, pos[2]], ":", color="#444444",
                               linewidth=0.6, alpha=0.3)

        # --- Target ---
        if self._target is not None:
            self.ax3d.scatter(*self._target[:3, 3], s=250,
                              c=self.cfg["target_color"], marker="x",
                              linewidths=3, zorder=5)

        # --- Trajectory trace ---
        if self._traj is not None and len(self._traj) > 1:
            pts = np.array([self.robot.joint_positions(q_i)[-1]
                            for q_i in self._traj])
            self.ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                           "--", color=self.cfg["trajectory_color"],
                           linewidth=1.5, alpha=0.7)

        # --- Frame axes at EE (larger) ---
        origin = T_ee[:3, 3]
        ee_scale = lim * 0.15
        for col_idx in range(3):
            direction = T_ee[:3, col_idx]
            self.ax3d.quiver(*origin, *(direction * ee_scale),
                             color=axis_colors[col_idx],
                             linewidth=2.0, alpha=0.95,
                             arrow_length_ratio=0.15)

        # --- Info panel ---
        self._update_info(q, ee, T_ee)

        self.fig.canvas.draw_idle()

    def _draw_base_pedestal(self, base_pos, lim):
        """Draw a small rectangular pedestal at the base."""
        s = lim * 0.08
        h = lim * 0.03
        x0, y0 = base_pos[0], base_pos[1]
        z0 = 0
        # Top face
        verts_top = [[(x0-s, y0-s, z0), (x0+s, y0-s, z0),
                      (x0+s, y0+s, z0), (x0-s, y0+s, z0)]]
        poly = Poly3DCollection(verts_top, alpha=0.6)
        poly.set_facecolor("#3a3a5a")
        poly.set_edgecolor("#555577")
        self.ax3d.add_collection3d(poly)
        # Bottom
        verts_bot = [[(x0-s, y0-s, z0-h), (x0+s, y0-s, z0-h),
                      (x0+s, y0+s, z0-h), (x0-s, y0+s, z0-h)]]
        poly2 = Poly3DCollection(verts_bot, alpha=0.3)
        poly2.set_facecolor("#2a2a4a")
        self.ax3d.add_collection3d(poly2)

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

    def _on_toggle(self, label):
        if label == "Frames":
            self._show_frames = not self._show_frames
        elif label == "Rot Axes":
            self._show_axes = not self._show_axes
        elif label == "Shadow":
            self._show_shadow = not self._show_shadow
        self._draw()

    def _on_slider_change(self, _):
        q = np.radians([sl.val for sl in self.sliders])
        self.robot.q = q
        self._draw()

    def _on_view(self, elev, azim):
        """Switch the 3D camera to a preset view angle."""
        self.ax3d.view_init(elev=elev, azim=azim)
        self.fig.canvas.draw_idle()

    def _on_home(self, _):
        # Go to midpoint of each joint's range
        mid = np.mean(self.robot.joint_limits, axis=1)
        self.robot.q = mid
        for i, sl in enumerate(self.sliders):
            sl.set_val(np.degrees(self.robot.q[i]))

    def _on_random(self, _):
        q_rand = self.robot.random_config()
        self.robot.q = q_rand
        for i, sl in enumerate(self.sliders):
            sl.set_val(np.degrees(self.robot.q[i]))

    def _on_ik_go(self, _):
        txt = self.txt_ik.text.strip()
        try:
            coords = [float(v) for v in txt.replace(",", " ").split()]
            if len(coords) != 3:
                raise ValueError
            target = np.array(coords)
        except ValueError:
            print("[Visualizer] Invalid input. Enter 3 numbers: x, y, z")
            return
        result = self.robot.ik_position_only(target, method="dls")
        if result.success:
            self.robot.q = result.joint_angles
            for i, sl in enumerate(self.sliders):
                sl.set_val(np.degrees(result.joint_angles[i]))
            self._target = self.robot.fk(result.joint_angles)
            print(f"[Visualizer] IK solved! Error: {result.pos_error:.6f} m")
        else:
            print(f"[Visualizer] IK failed. Position may be unreachable. "
                  f"Error: {result.pos_error:.4f} m")
            self._target = None
        self._draw()

    def _on_save_waypoint(self, _):
        self._waypoints.append(self.robot.q.copy())
        print(f"[Visualizer] Waypoint {len(self._waypoints)} saved: "
              f"{np.degrees(self.robot.q).round(1)} deg")

    def _on_clear_waypoints(self, _):
        self._waypoints.clear()
        self._traj = None
        print("[Visualizer] Waypoints and trajectory cleared.")
        self._draw()

    def _on_play_trajectory(self, _):
        if self._traj is None and len(self._waypoints) >= 2:
            self._traj = multi_segment_trajectory(self._waypoints,
                                                  n_steps=30, smooth=True)
            print(f"[Visualizer] Built trajectory from {len(self._waypoints)} waypoints.")
        if self._traj is None:
            print("[Visualizer] No trajectory. Save at least 2 waypoints with '+ Pt' first.")
            return
        # Disconnect slider callbacks to avoid 6 redraws per frame
        for sl in self.sliders:
            sl.disconnect_events()
        for q_i in self._traj:
            self.robot.q = q_i
            for i, sl in enumerate(self.sliders):
                sl.set_val(np.degrees(q_i[i]))
            self._draw()
            self.fig.canvas.flush_events()
            plt.pause(0.005 / self.slider_speed.val)
        # Reconnect slider callbacks
        for sl in self.sliders:
            sl.on_changed(self._on_slider_change)

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
