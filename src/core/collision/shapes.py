"""Shapes, obstacles, and collision detection (circle & OBB)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from omegaconf import DictConfig

Pose = Tuple[float, float, float]  # (x, y, theta)


# ── Shape primitives ───────────────────────────────────────────────────────────

@dataclass
class CircleShape:
    radius: float

    @property
    def bounding_radius(self) -> float:
        return self.radius


@dataclass
class BoxShape:
    width: float   # extent along local x-axis
    length: float  # extent along local y-axis

    @property
    def bounding_radius(self) -> float:
        return 0.5 * np.sqrt(self.width**2 + self.length**2)


Shape = Union[CircleShape, BoxShape]


def build_shape(cfg: DictConfig) -> Shape:
    t = cfg.type
    if t == "circle":
        return CircleShape(radius=float(cfg.radius))
    if t == "box":
        return BoxShape(width=float(cfg.width), length=float(cfg.length))
    raise ValueError(f"Unknown shape type: {t!r}")


# ── Obstacle ───────────────────────────────────────────────────────────────────

@dataclass
class Obstacle:
    x: float
    y: float
    shape: Shape
    angle: float = 0.0  # rotation of box obstacles (radians)

    @property
    def pose(self) -> Pose:
        return (self.x, self.y, self.angle)

    @property
    def obs_repr(self) -> np.ndarray:
        """6-D obstacle feature: [rel-position placeholder, hw, hl, sin(a), cos(a)].
        rel-position filled by caller (agent-relative). Shape-unified: circles → [r, r, 0, 1]."""
        if isinstance(self.shape, CircleShape):
            r = self.shape.radius
            return np.array([0.0, 0.0, r, r, 0.0, 1.0], dtype=np.float32)
        hw = self.shape.width / 2
        hl = self.shape.length / 2
        return np.array([0.0, 0.0, hw, hl, np.sin(self.angle), np.cos(self.angle)],
                        dtype=np.float32)


def build_obstacle(cfg: DictConfig) -> Obstacle:
    shape = build_shape(cfg.shape)
    return Obstacle(
        x=float(cfg.x),
        y=float(cfg.y),
        shape=shape,
        angle=float(cfg.get("angle", 0.0)),
    )


# ── Collision detection ────────────────────────────────────────────────────────

def collides(shape_a: Shape, pose_a: Pose, shape_b: Shape, pose_b: Pose) -> bool:
    # Fast broad-phase: bounding circles
    dx, dy = pose_a[0] - pose_b[0], pose_a[1] - pose_b[1]
    if dx*dx + dy*dy > (shape_a.bounding_radius + shape_b.bounding_radius)**2:
        return False

    if isinstance(shape_a, CircleShape) and isinstance(shape_b, CircleShape):
        return _circle_circle(shape_a, pose_a, shape_b, pose_b)
    if isinstance(shape_a, CircleShape) and isinstance(shape_b, BoxShape):
        return _circle_box(shape_a, pose_a, shape_b, pose_b)
    if isinstance(shape_a, BoxShape) and isinstance(shape_b, CircleShape):
        return _circle_box(shape_b, pose_b, shape_a, pose_a)
    if isinstance(shape_a, BoxShape) and isinstance(shape_b, BoxShape):
        return _obb_obb(shape_a, pose_a, shape_b, pose_b)
    raise TypeError(f"Unsupported shapes: {type(shape_a)}, {type(shape_b)}")


def _circle_circle(a: CircleShape, pa: Pose, b: CircleShape, pb: Pose) -> bool:
    dx, dy = pa[0] - pb[0], pa[1] - pb[1]
    return dx*dx + dy*dy < (a.radius + b.radius)**2


def _circle_box(circle: CircleShape, pc: Pose, box: BoxShape, pb: Pose) -> bool:
    """Nearest-point test: circle center → clamped point on OBB."""
    dx, dy = pc[0] - pb[0], pc[1] - pb[1]
    c, s = np.cos(-pb[2]), np.sin(-pb[2])
    lx = c * dx - s * dy
    ly = s * dx + c * dy
    hw, hl = box.width / 2, box.length / 2
    cx = np.clip(lx, -hw, hw)
    cy = np.clip(ly, -hl, hl)
    return (lx - cx)**2 + (ly - cy)**2 < circle.radius**2


def _obb_obb(a: BoxShape, pa: Pose, b: BoxShape, pb: Pose) -> bool:
    """SAT test for two oriented bounding boxes."""
    def corners(box: BoxShape, pose: Pose) -> np.ndarray:
        hw, hl = box.width / 2, box.length / 2
        local = np.array([[-hw, -hl], [hw, -hl], [hw, hl], [-hw, hl]], dtype=np.float64)
        c, s = np.cos(pose[2]), np.sin(pose[2])
        R = np.array([[c, -s], [s, c]])
        return (R @ local.T).T + np.array([pose[0], pose[1]])

    ca, cb = corners(a, pa), corners(b, pb)

    def axes(c):
        result = []
        for i in range(4):
            e = c[(i + 1) % 4] - c[i]
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-10:
                result.append(n / norm)
        return result

    for ax in axes(ca) + axes(cb):
        pa_proj = ca @ ax
        pb_proj = cb @ ax
        if pa_proj.max() < pb_proj.min() or pb_proj.max() < pa_proj.min():
            return False
    return True


def _corners(box: BoxShape, pose: Pose) -> np.ndarray:
    hw, hl = box.width / 2, box.length / 2
    local = np.array([[-hw, -hl], [hw, -hl], [hw, hl], [-hw, hl]], dtype=np.float64)
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return (np.array([[c, -s], [s, c]]) @ local.T).T + np.array([pose[0], pose[1]])


def _pts_to_segs(pts: np.ndarray, poly: np.ndarray) -> float:
    """Smallest distance from any point in `pts` to the boundary of convex polygon `poly`."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a                                          # (E, 2)
    denom = np.einsum("ej,ej->e", ab, ab)
    denom[denom < 1e-18] = 1e-18
    ap = pts[:, None, :] - a[None, :, :]                # (P, E, 2)
    t = np.clip(np.einsum("pej,ej->pe", ap, ab) / denom, 0.0, 1.0)
    closest = a[None, :, :] + t[..., None] * ab[None, :, :]
    return float(np.min(np.linalg.norm(pts[:, None, :] - closest, axis=2)))


def shape_distance(shape_a: Shape, pose_a: Pose, shape_b: Shape, pose_b: Pose) -> float:
    """Shortest distance between two shapes' actual surfaces. 0.0 when they overlap.

    K-ARC Eq. 6 constrains ``||c_i,k - c_j,k|| >= d_min`` where ``c`` is the robot's
    *geometric pose*, and SS V-A states the models plainly: "we use simple polyhedrons for
    both our obstacle and robot models, and we calculate the shortest distances between any
    two objects". Testing centre-to-centre against a sum of bounding radii is a different
    predicate -- for a 0.5 x 0.25 box the circumscribed disc is 0.559 m against a 0.25 m
    half-width, so it reports conflicts at up to 0.3 m of real clearance and hands the
    resolution hierarchy work K-ARC never has to do.

    Two convex polygons that do not overlap realise their distance at a vertex of one
    against an edge of the other, so the vertex-to-edge minimum over both orderings is
    exact here -- no GJK needed for boxes.
    """
    a_box, b_box = isinstance(shape_a, BoxShape), isinstance(shape_b, BoxShape)
    if not a_box and not b_box:
        d = float(np.hypot(pose_a[0] - pose_b[0], pose_a[1] - pose_b[1]))
        return max(0.0, d - shape_a.radius - shape_b.radius)
    if a_box and b_box:
        if _obb_obb(shape_a, pose_a, shape_b, pose_b):
            return 0.0
        ca, cb = _corners(shape_a, pose_a), _corners(shape_b, pose_b)
        return min(_pts_to_segs(ca, cb), _pts_to_segs(cb, ca))
    # One box, one circle: distance from the circle's centre to the box, less its radius.
    box, bpose, circ, cpose = ((shape_a, pose_a, shape_b, pose_b) if a_box
                               else (shape_b, pose_b, shape_a, pose_a))
    centre = np.array([[cpose[0], cpose[1]]], dtype=np.float64)
    inside = _obb_obb(box, bpose, BoxShape(1e-9, 1e-9), (cpose[0], cpose[1], 0.0))
    d = 0.0 if inside else _pts_to_segs(centre, _corners(box, bpose))
    return max(0.0, d - circ.radius)


# ── Ray casting (for lidar) ────────────────────────────────────────────────────

def ray_distance(origin: np.ndarray, direction: np.ndarray,
                 shape: Shape, pose: Pose, max_range: float) -> float:
    """Distance along `direction` from `origin` to shape surface. Returns max_range if no hit."""
    if isinstance(shape, CircleShape):
        return _ray_circle(origin, direction, np.array(pose[:2]), shape.radius, max_range)
    return _ray_obb(origin, direction, pose, shape, max_range)


def _ray_circle(origin, direction, center, radius, max_range):
    oc = center - origin
    t = float(np.dot(oc, direction))
    dist_sq = float(np.dot(oc, oc)) - t * t
    if dist_sq >= radius * radius:
        return max_range
    t_hit = t - np.sqrt(max(0.0, radius * radius - dist_sq))
    return float(t_hit) if t_hit >= 0 else max_range


def _ray_obb(origin, direction, pose, box: BoxShape, max_range):
    """Slab method in OBB local frame."""
    dx, dy = origin[0] - pose[0], origin[1] - pose[1]
    c, s = np.cos(-pose[2]), np.sin(-pose[2])
    lo = np.array([c * dx - s * dy, s * dx + c * dy])
    ld = np.array([c * direction[0] - s * direction[1],
                   s * direction[0] + c * direction[1]])
    hw, hl = box.width / 2, box.length / 2
    slabs = [(-hw, hw, lo[0], ld[0]), (-hl, hl, lo[1], ld[1])]
    t_enter, t_exit = -np.inf, np.inf
    for lo_val, hi_val, orig, dirv in slabs:
        if abs(dirv) < 1e-10:
            if orig < lo_val or orig > hi_val:
                return max_range
        else:
            t1, t2 = (lo_val - orig) / dirv, (hi_val - orig) / dirv
            if t1 > t2:
                t1, t2 = t2, t1
            t_enter = max(t_enter, t1)
            t_exit = min(t_exit, t2)
    if t_exit < 0 or t_enter > t_exit:
        return max_range
    t_hit = t_enter if t_enter >= 0 else t_exit
    return float(t_hit) if 0 <= t_hit <= max_range else max_range


# ── World-boundary collision (Minkowski sum, uniform with collides()) ─────────

def collides_wall(shape: Shape, pose: Pose, world_size: float) -> bool:
    """True if shape at pose intersects any world boundary.

    Uses same Minkowski sum principle as collides(): robot physical shape, not
    point approximation. Circle: exact. Box: exact at any orientation.
    """
    x, y, theta = pose
    if isinstance(shape, CircleShape):
        r = shape.radius
        return x < r or x > world_size - r or y < r or y > world_size - r
    hw, hl = shape.width / 2, shape.length / 2
    ct, st = abs(np.cos(theta)), abs(np.sin(theta))
    dx = hw * ct + hl * st
    dy = hw * st + hl * ct
    return x - dx < 0.0 or x + dx > world_size or y - dy < 0.0 or y + dy > world_size


def clip_to_world(shape: Shape, pose: Pose, world_size: float) -> Tuple[float, float]:
    """Return (x, y) clipped so shape just touches world boundary (theta unchanged)."""
    x, y, theta = pose
    if isinstance(shape, CircleShape):
        r = shape.radius
        return float(np.clip(x, r, world_size - r)), float(np.clip(y, r, world_size - r))
    hw, hl = shape.width / 2, shape.length / 2
    ct, st = abs(np.cos(theta)), abs(np.sin(theta))
    dx = hw * ct + hl * st
    dy = hw * st + hl * ct
    return float(np.clip(x, dx, world_size - dx)), float(np.clip(y, dy, world_size - dy))
