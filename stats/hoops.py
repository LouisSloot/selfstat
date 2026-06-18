"""Rim positions and per-rim scoring zones.

The camera is fixed, so the rims are static — no hand annotation needed. We
cluster the detector's hoop detections over a window into stable rim boxes (median
per cluster), then derive the geometric zones the shot state machine uses:

    up-region   : a band ABOVE the rim (ball must appear here on the way in)
    down-region : a band BELOW the rim (a made ball appears here, through the net)
    x-gate      : the rim x-span widened by a margin (make/miss discriminator)

These gyms have several baskets in frame (left/center/right); each becomes a Rim,
and shots are scored against whichever rim the ball approaches.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Rim:
    id: int
    bbox: tuple        # (x1, y1, x2, y2), median over detections
    n_dets: int

    @property
    def cx(self):
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def cy(self):
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def w(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def h(self):
        return self.bbox[3] - self.bbox[1]

    def zones(self, up_frac=4.0, down_frac=4.0, x_margin=0.6):
        """Return (x_lo, x_hi, up_top, rim_y, down_bot) for scoring."""
        x_lo = self.bbox[0] - x_margin * self.w
        x_hi = self.bbox[2] + x_margin * self.w
        up_top = self.bbox[1] - up_frac * self.h
        down_bot = self.bbox[3] + down_frac * self.h
        return x_lo, x_hi, up_top, self.cy, down_bot


def detect_rims(hoop_dets, min_dets=10, merge_dist=80.0):
    """Cluster hoop detections (list per frame of [(xyxy, conf), ...]) into stable
    rims. Greedy: seed clusters by detection count, absorb nearby centers. Keep
    clusters seen in >= min_dets detections. Returns rims sorted left->right."""
    boxes = [box for frame in hoop_dets for box, _conf in frame]
    if not boxes:
        return []
    boxes = np.array(boxes, dtype=float)
    centers = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                               (boxes[:, 1] + boxes[:, 3]) / 2])

    used = np.zeros(len(boxes), dtype=bool)
    clusters = []
    while not used.all():
        i = np.argmax(~used)
        d = np.linalg.norm(centers - centers[i], axis=1)
        members = np.where((d < merge_dist) & (~used))[0]
        used[members] = True
        if len(members) >= min_dets:
            clusters.append((np.median(boxes[members], axis=0), len(members)))

    clusters.sort(key=lambda c: (c[0][0] + c[0][2]) / 2)  # left -> right
    return [Rim(i, tuple(float(v) for v in b), n) for i, (b, n) in enumerate(clusters)]
