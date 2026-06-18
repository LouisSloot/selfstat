"""Turn noisy per-frame ball detections into a clean, continuous ball track.

Three cleanups, in order:
1. **size gate** — the real ball is small (~15-35 px here); drop big boxes, which
   are the false positives that land on shorts/limbs.
2. **velocity gate** — per frame pick the candidate nearest the previous center
   (within `max_jump`), tie-broken by confidence; rejects ball-shaped distractors
   that flicker far from the trajectory.
3. **gap interpolation** — linearly bridge gaps up to `max_gap` frames so a few
   missed detections mid-flight don't break shot scoring.

Returns `{frame_offset: BallPoint(cx, cy, interpolated)}`.
"""

from dataclasses import dataclass


@dataclass
class BallPoint:
    cx: float
    cy: float
    interpolated: bool


def _center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _max_dim(box):
    x1, y1, x2, y2 = box
    return max(x2 - x1, y2 - y1)


def build_ball_track(detections, max_size=45.0, max_jump=140.0, max_gap=8):
    """detections: list (per frame offset) of `[(xyxy, conf), ...]`."""
    chosen = {}  # offset -> (cx, cy)
    prev = None
    for f, dets in enumerate(detections):
        cands = [(box, conf) for box, conf in dets if _max_dim(box) <= max_size]
        if not cands:
            continue
        if prev is None:
            box, _ = max(cands, key=lambda bc: bc[1])  # highest confidence
        else:
            def dist(bc):
                cx, cy = _center(bc[0])
                return ((cx - prev[0]) ** 2 + (cy - prev[1]) ** 2) ** 0.5
            near = [bc for bc in cands if dist(bc) <= max_jump]
            box, _ = min(near, key=dist) if near else max(cands, key=lambda bc: bc[1])
        cx, cy = _center(box)
        chosen[f] = (cx, cy)
        prev = (cx, cy)

    track = {f: BallPoint(cx, cy, False) for f, (cx, cy) in chosen.items()}
    keys = sorted(chosen)
    for a, b in zip(keys, keys[1:]):
        gap = b - a
        if 1 < gap <= max_gap + 1:
            (ax, ay), (bx, by) = chosen[a], chosen[b]
            for k in range(1, gap):
                t = k / gap
                track[a + k] = BallPoint(ax + t * (bx - ax), ay + t * (by - ay), True)
    return track
