"""
Efficient Spatial Proximity Detection & ORCA Velocity Obstacle.

Two core algorithms:

1. CollisionDetector — O(n log n) proximity queries via cKDTree
   with persistent pair tracking to avoid double-counting.

2. ORCA velocity obstacle — Optimal Reciprocal Collision Avoidance
   for N agents in 2D or 3D. Selects the velocity closest to the
   preferred direction while guaranteeing collision-free motion.
"""

import numpy as np
from scipy.spatial import cKDTree

TOL = 1e-8


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > TOL else v


def _clip_speed(v, max_speed):
    s = np.linalg.norm(v)
    return v / s * max_speed if s > max_speed else v


# ---------------------------------------------------------------------------
# Collision Detection
# ---------------------------------------------------------------------------

class CollisionDetector:
    """N-dimensional proximity detection with persistent pair tracking.

    Uses scipy.spatial.cKDTree for O(n log n) spatial queries.
    Each unique agent pair is counted only once across the entire
    simulation — no double-counting.
    """

    def __init__(self, safe_distance):
        self.safe_dist = safe_distance
        self.seen_pairs = set()
        self.conflict_count = 0
        self.conflict_positions = []

    def reset(self):
        self.seen_pairs.clear()
        self.conflict_count = 0
        self.conflict_positions.clear()

    def detect(self, agents, record_positions=False, group_by='group'):
        """Check active agents for proximity violations.

        agents:  list of objects with .position (np.array), .id, and group attr
        group_by: attribute to group by — only pairs within same group checked.
                  Set to None to check all pairs.
        Returns number of NEW conflicts detected this step.
        """
        active = [a for a in agents if a.is_active]
        if len(active) < 2:
            return 0

        new_count = 0

        if group_by is None:
            groups = {0: active}
        else:
            groups = {}
            for a in active:
                g = getattr(a, group_by, 0)
                groups.setdefault(g, []).append(a)

        for g_agents in groups.values():
            if len(g_agents) < 2:
                continue
            positions = np.array([a.position for a in g_agents])
            ids = [a.id for a in g_agents]
            tree = cKDTree(positions)
            for i, j in tree.query_pairs(self.safe_dist, output_type='set'):
                gi, gj = ids[i], ids[j]
                pair = (gi, gj) if gi < gj else (gj, gi)
                if pair not in self.seen_pairs:
                    self.seen_pairs.add(pair)
                    new_count += 1
                    if record_positions:
                        mid = (positions[i] + positions[j]) / 2.0
                        self.conflict_positions.append(mid)

        self.conflict_count += new_count
        return new_count

    def proximity_count(self, agent, agents, radius=None):
        """Count agents within radius of the given agent."""
        r = radius if radius is not None else self.safe_dist
        others = [a for a in agents if a.id != agent.id and a.is_active]
        if not others:
            return 0
        positions = np.array([a.position for a in others])
        tree = cKDTree(positions)
        return len(tree.query_ball_point(agent.position, r))


# ---------------------------------------------------------------------------
# ORCA Velocity Obstacle
# ---------------------------------------------------------------------------

def orca_velocity(position, velocity, target, neighbors, safe_dist,
                  lookahead=1.0, max_speed=1.0, dim=2):
    """Compute collision-free velocity using ORCA.

    position:   np.array, agent's current position
    velocity:   np.array, agent's current velocity
    target:     np.array, agent's target position
    neighbors:  list of (position, velocity) tuples for nearby agents
    safe_dist:  float, minimum allowed separation
    lookahead:  float, time horizon for collision prediction
    max_speed:  float, maximum speed
    dim:        int, 2 or 3 (XY plane for 3D)

    Returns: velocity vector (np.array)
    """
    pref_speed = np.linalg.norm(velocity)
    if pref_speed < TOL:
        return np.zeros(dim)

    # Preferred velocity: toward target
    target_dir = target - position
    target_dist = np.linalg.norm(target_dir)
    if target_dist < TOL:
        pref_vel = np.zeros(dim)
    else:
        pref_vel = (target_dir / target_dist) * min(pref_speed, target_dist / lookahead)

    orca_lines = []
    for n_pos, n_vel in neighbors:
        rel_pos = n_pos - position
        rel_vel = velocity - n_vel
        dist = np.linalg.norm(rel_pos)
        if dist < TOL:
            continue

        t = lookahead
        w = rel_vel - rel_pos * (safe_dist / (t * dist))
        w_norm = np.linalg.norm(w)
        if w_norm < TOL:
            continue

        n = w / w_norm
        u = (safe_dist / lookahead) * n
        orca_lines.append({'n': n, 'b': np.dot(n, velocity - u)})

    if not orca_lines:
        return _normalize(pref_vel) * min(pref_speed, max_speed)

    # Iterative projection onto feasible velocity set
    v_opt = pref_vel.copy()
    for _ in range(10):
        for line in orca_lines:
            d = np.dot(line['n'], v_opt) - line['b']
            if d < 0:
                v_opt -= line['n'] * d
        v_opt = _clip_speed(v_opt, max_speed)

    return v_opt
