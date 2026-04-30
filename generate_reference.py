#!/usr/bin/env python3
"""
generate_reference.py — Generate reference diagrams of each robot arm preset.

Saves annotated 3D views showing:
  • Cylindrical links with per-link colour
  • Joint coordinate frames (RGB = XYZ)
  • Joint rotation-axis arrows (gold)
  • Ground shadow projection
  • DH parameter annotations

Usage
-----
    python generate_reference.py                # all presets
    python generate_reference.py --preset puma560
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from robot_arm import Robot6DOF
from robot_arm.config import PRESETS


LINK_COLORS = [
    "#4A90D9", "#5B7FD9", "#6C6ED9",
    "#7D5DD9", "#8E4CD9", "#9F3BD9",
]
JOINT_COLORS = [
    "#FF6B35", "#FFD166", "#06D6A0",
    "#118AB2", "#EF476F", "#FFC43D",
]
AXIS_COLORS = ["#e74c3c", "#2ecc71", "#3498db"]  # X Y Z


def _cylinder_between(ax, p0, p1, radius=0.02, n_sides=12, color="#4A90D9", alpha=0.85):
    v = np.array(p1) - np.array(p0)
    length = np.linalg.norm(v)
    if length < 1e-6:
        return
    v_hat = v / length
    if abs(v_hat[0]) < 0.9:
        perp = np.cross(v_hat, [1, 0, 0])
    else:
        perp = np.cross(v_hat, [0, 1, 0])
    perp /= np.linalg.norm(perp)
    perp2 = np.cross(v_hat, perp)
    theta = np.linspace(0, 2 * np.pi, n_sides + 1)
    circle = np.array([radius * (np.cos(t) * perp + np.sin(t) * perp2) for t in theta])
    bottom = np.array(p0) + circle
    top = np.array(p0) + v + circle
    verts = []
    for i in range(n_sides):
        verts.append([bottom[i], bottom[i+1], top[i+1], top[i]])
    poly = Poly3DCollection(verts, alpha=alpha, linewidths=0.3, edgecolors="#222222")
    poly.set_facecolor(color)
    ax.add_collection3d(poly)


def render_arm(preset_name, q_deg=None, elev=25, azim=-60):
    """Render a single robot arm preset and return the figure."""
    robot = Robot6DOF(preset=preset_name)
    if q_deg is not None:
        robot.q = np.radians(q_deg)
    else:
        # Use a pose that shows spatial structure clearly
        robot.q = np.radians([30, -35, 60, 20, 30, -15])

    q = robot.q
    positions = robot.joint_positions(q)
    frames = robot.fk_all_frames(q)
    T_ee = robot.fk(q)

    reach = np.sum(robot.dh_params[:, 0]) + np.sum(np.abs(robot.dh_params[:, 2]))
    lim = max(reach * 0.65, 0.6)

    fig = plt.figure(figsize=(14, 10), facecolor="#1a1a2e")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#16213e")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, 2 * lim)
    ax.set_xlabel("X (m)", color="#aaa", fontsize=9)
    ax.set_ylabel("Y (m)", color="#aaa", fontsize=9)
    ax.set_zlabel("Z (m)", color="#aaa", fontsize=9)
    ax.tick_params(colors="#555", labelsize=7)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#333")

    # Ground grid
    gn = 8
    gs = lim * 2 / gn
    for ix in range(gn):
        for iy in range(gn):
            x0 = -lim + ix * gs
            y0 = -lim + iy * gs
            c = "#1c2a4a" if (ix + iy) % 2 == 0 else "#162040"
            verts = [[(x0, y0, 0), (x0+gs, y0, 0), (x0+gs, y0+gs, 0), (x0, y0+gs, 0)]]
            p = Poly3DCollection(verts, alpha=0.25)
            p.set_facecolor(c)
            p.set_edgecolor("#2a3a5a")
            ax.add_collection3d(p)

    # Cylindrical links
    link_radius = lim * 0.025
    for i in range(len(positions) - 1):
        col = LINK_COLORS[i % len(LINK_COLORS)]
        _cylinder_between(ax, positions[i], positions[i+1], radius=link_radius, color=col)

    # Joint spheres
    for i, pos in enumerate(positions[:-1]):
        c = JOINT_COLORS[i % len(JOINT_COLORS)]
        ax.scatter(*pos, s=200, c=c, depthshade=True, zorder=4,
                   edgecolors="#222", linewidths=0.5)

    # End-effector
    ee = positions[-1]
    ax.scatter(*ee, s=300, c="#2ECC71", marker="D", depthshade=False,
              zorder=5, edgecolors="white", linewidths=1.0)

    frame_scale = lim * 0.10

    # Base frame
    base_origin = np.array([0, 0, 0])
    for ci in range(3):
        d = np.zeros(3)
        d[ci] = frame_scale
        ax.quiver(*base_origin, *d, color=AXIS_COLORS[ci],
                  linewidth=2.0, alpha=0.8, arrow_length_ratio=0.15)
    ax.text(frame_scale * 1.15, 0, 0, "X₀", color="#e74c3c", fontsize=8, fontweight="bold")
    ax.text(0, frame_scale * 1.15, 0, "Y₀", color="#2ecc71", fontsize=8, fontweight="bold")
    ax.text(0, 0, frame_scale * 1.15, "Z₀", color="#3498db", fontsize=8, fontweight="bold")

    # Joint frames
    for j, T_j in enumerate(frames):
        origin_j = T_j[:3, 3]
        for ci in range(3):
            d = T_j[:3, ci] * frame_scale * 0.7
            ax.quiver(*origin_j, *d, color=AXIS_COLORS[ci],
                      linewidth=1.2, alpha=0.6, arrow_length_ratio=0.18)

    # Joint rotation-axis arrows (gold)
    rot_scale = lim * 0.12
    for j in range(len(frames)):
        if j == 0:
            origin_r = positions[0]
            z_axis = np.array([0, 0, 1.0])
        else:
            origin_r = frames[j-1][:3, 3]
            z_axis = frames[j-1][:3, 2]
        ax.quiver(*origin_r, *(z_axis * rot_scale),
                  color="#FFD700", linewidth=2.5, alpha=0.85,
                  arrow_length_ratio=0.25)
        label_pos = origin_r + z_axis * rot_scale * 1.2
        ax.text(*label_pos, f"J{j+1}", color="#FFD700", fontsize=8,
                ha="center", fontweight="bold")

    # Shadow on ground
    for i in range(len(positions) - 1):
        ax.plot([positions[i, 0], positions[i+1, 0]],
                [positions[i, 1], positions[i+1, 1]],
                [0.001, 0.001], "-", color="#555", linewidth=2.5, alpha=0.4, zorder=1)
    for pos in positions:
        ax.scatter(pos[0], pos[1], 0.001, s=40, c="#444", alpha=0.3, marker="o", zorder=1)
        ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [0, pos[2]],
                ":", color="#444", linewidth=0.6, alpha=0.3)

    # EE frame (larger)
    origin = T_ee[:3, 3]
    ee_scale = lim * 0.15
    for ci in range(3):
        direction = T_ee[:3, ci]
        ax.quiver(*origin, *(direction * ee_scale),
                  color=AXIS_COLORS[ci], linewidth=2.0, alpha=0.95,
                  arrow_length_ratio=0.15)

    ax.view_init(elev=elev, azim=azim)

    preset_info = PRESETS[preset_name]
    title = f"{preset_info['name']}  —  q = {np.degrees(q).round(1)}°"
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=15)

    # Legend box
    legend_lines = [
        "Red/Green/Blue arrows = X/Y/Z axes",
        "Gold arrows = Joint rotation axes (Jn)",
        "Ground shadow = XY projection",
        "Dashed lines = height reference",
    ]
    fig.text(0.02, 0.02, "\n".join(legend_lines),
             color="#aaa", fontsize=8, family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f3460", alpha=0.8))

    plt.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default=None,
                   choices=list(PRESETS.keys()),
                   help="Render a specific preset (default: all)")
    p.add_argument("--outdir", default="docs",
                   help="Output directory for images")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    presets = [args.preset] if args.preset else list(PRESETS.keys())

    for name in presets:
        print(f"  Rendering {name}...")
        fig = render_arm(name)
        path = os.path.join(args.outdir, f"robot_{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"    Saved → {path}")

    # Also generate a multi-view for the first preset
    name = presets[0]
    fig_multi = plt.figure(figsize=(18, 8), facecolor="#1a1a2e")
    views = [(25, -60), (90, -90), (0, 0), (25, 30)]
    view_names = ["Perspective", "Top (XY)", "Front (XZ)", "Right side"]

    robot = Robot6DOF(preset=name)
    robot.q = np.radians([30, -35, 60, 20, 30, -15])
    q = robot.q
    positions = robot.joint_positions(q)
    frames = robot.fk_all_frames(q)
    reach = np.sum(robot.dh_params[:, 0]) + np.sum(np.abs(robot.dh_params[:, 2]))
    lim = max(reach * 0.65, 0.6)

    for vi, (elev, azim) in enumerate(views):
        ax = fig_multi.add_subplot(1, 4, vi + 1, projection="3d")
        ax.set_facecolor("#16213e")
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(0, 2 * lim)
        ax.tick_params(colors="#555", labelsize=5)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#333")

        link_radius = lim * 0.02
        for i in range(len(positions) - 1):
            _cylinder_between(ax, positions[i], positions[i+1],
                              radius=link_radius,
                              color=LINK_COLORS[i % len(LINK_COLORS)])

        for i, pos in enumerate(positions[:-1]):
            ax.scatter(*pos, s=120, c=JOINT_COLORS[i % len(JOINT_COLORS)],
                       depthshade=True, zorder=4, edgecolors="#222", linewidths=0.4)

        ax.scatter(*positions[-1], s=200, c="#2ECC71", marker="D",
                   depthshade=False, zorder=5, edgecolors="white", linewidths=0.8)

        # Joint rotation axes
        rot_scale = lim * 0.10
        for j in range(len(frames)):
            if j == 0:
                origin_r, z_axis = positions[0], np.array([0, 0, 1.0])
            else:
                origin_r, z_axis = frames[j-1][:3, 3], frames[j-1][:3, 2]
            ax.quiver(*origin_r, *(z_axis * rot_scale),
                      color="#FFD700", linewidth=2, alpha=0.8, arrow_length_ratio=0.25)
            lp = origin_r + z_axis * rot_scale * 1.15
            ax.text(*lp, f"J{j+1}", color="#FFD700", fontsize=6, ha="center", fontweight="bold")

        # Shadow
        for i in range(len(positions) - 1):
            ax.plot([positions[i, 0], positions[i+1, 0]],
                    [positions[i, 1], positions[i+1, 1]],
                    [0.001, 0.001], "-", color="#555", linewidth=2, alpha=0.35, zorder=1)

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(view_names[vi], color="white", fontsize=10, fontweight="bold")

    fig_multi.suptitle(f"{PRESETS[name]['name']} — Multi-View Reference",
                       color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(args.outdir, f"robot_{name}_multiview.png")
    fig_multi.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig_multi)
    print(f"    Multi-view saved → {path}")

    print("\n  Done!")


if __name__ == "__main__":
    main()
