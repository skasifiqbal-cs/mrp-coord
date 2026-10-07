"""Stateless matplotlib renderer — supports circle and box shapes.

Greyscale on purpose. Robots and their goals are told apart by a label, not by hue:
figures survive a black-and-white print, and the number of robots is not capped by the
length of a colour palette. Robot i is ``a{i}``, its goal ``g{i}``, matching the env's
``agent_{i}`` ids. Both labels sit outside the shape they name so they never cover it.
"""
from __future__ import annotations

import io
from typing import Dict, List, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from src.core.collision.shapes import BoxShape, CircleShape, Obstacle

# Neutral greys only: R == G == B. The blue-tinted "greys" this replaced (#F1F3F4,
# #6C757D) survive a mono print but are not actually greyscale, and a figure that is
# 87% off-neutral pixels is a colour figure as far as a journal is concerned.
EDGE_COLOR  = "#2B2B2B"   # robot outline, heading line, and both label texts
BODY_COLOR  = "#C8C8C8"   # robot fill, light enough to keep the outline readable
GOAL_COLOR  = "#5E5E5E"   # goal star and its tolerance ring
TRAIL_COLOR = "#AFAFAF"
OBS_COLOR   = "#8C8C8C"
BG_COLOR    = "#F2F2F2"


def _obb_corners(x: float, y: float, theta: float, w: float, l: float) -> np.ndarray:
    hw, hl = w / 2, l / 2
    local = np.array([[-hw, -hl], [hw, -hl], [hw, hl], [-hw, hl]])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return (R @ local.T).T + np.array([x, y])


def _forward_reach(shape) -> float:
    """Half-extent along the robot's own heading.

    For a box that is ``width``, not ``length``: conf/robot/unicycle_db.yaml defines
    width as the extent along local x (forward) to match dynobench's size[0], and
    _obb_corners lays the box out on that axis. Reading ``length`` here would draw the
    heading line out through the robot's side.
    """
    return shape.radius if isinstance(shape, CircleShape) else shape.width / 2


def _label_fontsize(fig_px: int) -> float:
    """Labels sit OUTSIDE the bodies, so they scale with the figure, not with a radius."""
    return float(np.clip(8.0 * fig_px / 480.0, 6.0, 14.0))


def _label_pos(x: float, y: float, reach: float, world_size: float,
               side: int = 1) -> tuple[float, float]:
    """Label position, diagonally clear of the shape it names.

    ``side`` is +1 (up-right) for robots and -1 (down-left) for goals. Opposite corners
    on purpose: a robot parked on its own goal is the normal end state, and with both
    labels on the same side ``a0`` and ``g0`` land on top of each other exactly when the
    reader most wants to see them. The gap is floored on world size so a small robot in
    a big world still gets one."""
    off = reach + max(0.35 * reach, 0.030 * world_size)
    d = side * off / np.sqrt(2.0)
    return x + d, y + d


def _draw_shape(ax, x, y, theta, shape, color, alpha=1.0, zorder=2, edge=None, lw=0.0):
    if isinstance(shape, CircleShape):
        ax.add_patch(mpatches.Circle((x, y), shape.radius, facecolor=color,
                                     edgecolor=edge, linewidth=lw, alpha=alpha,
                                     zorder=zorder))
    elif isinstance(shape, BoxShape):
        corners = _obb_corners(x, y, theta, shape.width, shape.length)
        ax.add_patch(Polygon(corners, closed=True, facecolor=color, edgecolor=edge,
                             linewidth=lw, alpha=alpha, zorder=zorder))


def _add_reward_bar(fig, ax, step_rewards: Dict[str, float]) -> None:
    """Add per-agent step reward text below the axes."""
    parts = [f"a{i}: {r:+.2f}" for i, (_a, r) in enumerate(sorted(step_rewards.items()))]
    ax.set_xlabel("    ".join(parts), fontsize=8, labelpad=4, family="monospace")


def _finish_frame(fig, ax, dpi: int) -> np.ndarray:
    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    from PIL import Image
    return np.array(Image.open(buf).convert("RGB"))


def render_frame_with_shapes(
    states: List[np.ndarray],
    robot_shapes,
    goals: List[np.ndarray],
    obstacles: List[Obstacle],
    trails: List[List[np.ndarray]],
    world_size: float,
    reached: List[bool],
    step: int,
    fig_px: int = 480,
    step_rewards: Optional[Dict[str, float]] = None,
    goal_radius: float = 0.2,
    title: Optional[str] = None,
    markers: Optional[List[np.ndarray]] = None,
    waypoints: Optional[List[np.ndarray]] = None,
    highlight: Optional[List[bool]] = None,
    guides: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """Render one frame with correct robot shapes (circle or OBB).

    Shapes come from the robot config, so a circular robot draws as a circle and a box
    robot as a box — ``conf/robot/unicycle_v2.yaml`` is a 0.13 m disc while
    ``unicycle_db.yaml`` is dynobench's 0.5 x 0.25 m rectangle.
    """
    dpi = 96
    fig_in = fig_px / dpi
    fig, ax = plt.subplots(figsize=(fig_in, fig_in), dpi=dpi)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, world_size)
    ax.set_ylim(0, world_size)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title if title is not None else f"step {step}", fontsize=9, pad=3)
    fs = _label_fontsize(fig_px)

    for m in (markers or []):
        # Conflict points: where two planned trajectories first violate separation.
        ax.plot(float(m[0]), float(m[1]), marker="x", markersize=9, markeredgewidth=2.0,
                color="#c0392b", linestyle="none", zorder=6)

    for obs in obstacles:
        _draw_shape(ax, obs.x, obs.y, obs.angle, obs.shape, OBS_COLOR, alpha=0.85,
                    zorder=2, edge=EDGE_COLOR, lw=0.8)

    for i, goal in enumerate(goals):
        gx, gy = float(goal[0]), float(goal[1])
        ax.add_patch(mpatches.Circle((gx, gy), goal_radius, facecolor="none",
                                     edgecolor=GOAL_COLOR, linestyle="--",
                                     linewidth=1.1, zorder=3))
        ax.plot(gx, gy, marker="*", markersize=13, markerfacecolor=BODY_COLOR,
                markeredgecolor=EDGE_COLOR, markeredgewidth=0.9, linestyle="none",
                zorder=4)
        lx, ly = _label_pos(gx, gy, goal_radius, world_size, side=-1)
        ax.text(lx, ly, f"g{i}", fontsize=fs, color=GOAL_COLOR, fontweight="bold",
                ha="center", va="center", zorder=4)

    for w in (waypoints or []):
        # Segment boundaries: K-ARC constrains robots to meet these at matching time
        # indices, which is what makes the inter-robot constraints comparable.
        ax.add_patch(mpatches.Circle((float(w[0]), float(w[1])), goal_radius,
                                     facecolor="none", edgecolor=EDGE_COLOR,
                                     linestyle=":", linewidth=1.0, alpha=0.75, zorder=3))

    for guide in (guides or []):
        # The planner's REFERENCE geometry -- a sampled polyline, not something anything
        # drives. Dotted so it never reads as a trajectory: what the robots actually fly is
        # the solid trail, which is a curve fitted through this and then time-parameterised.
        g = np.asarray(guide, float)
        if len(g) > 1:
            ax.plot(g[:, 0], g[:, 1], color=TRAIL_COLOR, linewidth=0.9, linestyle=":",
                    alpha=0.7, zorder=2)

    for trail in trails:
        if len(trail) > 1:
            t = np.array(trail)
            ax.plot(t[:, 0], t[:, 1], color=TRAIL_COLOR, linewidth=1.2, zorder=3)

    for i, state in enumerate(states):
        x, y, theta = float(state[0]), float(state[1]), float(state[2])
        alpha = 0.45 if reached[i] else 1.0
        shape = robot_shapes[i]
        hit = bool(highlight[i]) if highlight is not None else False
        _draw_shape(ax, x, y, theta, shape, "#c0392b" if hit else BODY_COLOR,
                    alpha=alpha, zorder=5, edge=EDGE_COLOR, lw=2.2 if hit else 1.4)
        # Heading inside the body: the label is outside now, so nothing collides here.
        reach = _forward_reach(shape)
        ax.plot([x, x + reach * np.cos(theta)], [y, y + reach * np.sin(theta)],
                color=EDGE_COLOR, lw=1.4, alpha=alpha, solid_capstyle="round", zorder=6)
        lx, ly = _label_pos(x, y, reach, world_size)
        ax.text(lx, ly, f"a{i}", fontsize=fs, color=EDGE_COLOR, fontweight="bold",
                ha="center", va="center", zorder=7)

    if step_rewards is not None:
        _add_reward_bar(fig, ax, step_rewards)

    return _finish_frame(fig, ax, dpi)
