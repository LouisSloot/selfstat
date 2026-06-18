"""MOTChallenge-format I/O and tracks.json -> MOT conversion.

Pure stdlib (json/csv) — no torch, cv2, or motmetrics — so the labeling tool and
converters import cheaply and the metrics dependency stays isolated to metrics.py.

MOTChallenge row: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
(frame and id are 1-based by convention; x/y/z unused -> -1). Boxes are top-left
plus width/height, whereas the pipeline's tracks.json uses xyxy.
"""

import json
import os
from collections import namedtuple

# x, y = top-left corner; w, h = box size (MOT convention)
MOTRow = namedtuple("MOTRow", "frame id x y w h conf")


def write_mot(rows, path):
    """Write MOTRows to a MOTChallenge txt (sorted by frame, then id)."""
    rows = sorted(rows, key=lambda r: (r.frame, r.id))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(f"{int(r.frame)},{int(r.id)},{r.x:.1f},{r.y:.1f},"
                    f"{r.w:.1f},{r.h:.1f},{r.conf:.3f},-1,-1,-1\n")
    return path


def read_mot(path):
    """Parse a MOTChallenge txt into a list of MOTRow (pure; for manipulation)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            rows.append(MOTRow(int(float(p[0])), int(float(p[1])), float(p[2]),
                               float(p[3]), float(p[4]), float(p[5]),
                               float(p[6]) if len(p) > 6 else 1.0))
    return rows


def tracks_json_to_mot(tracks_path, out_path, frame_offset=1, id_offset=1,
                       id_field="player_id"):
    """Convert a pipeline tracks.json -> MOTChallenge txt.

    bbox [x1,y1,x2,y2] -> (left, top, width, height). `frame_offset`/`id_offset`
    shift the 0-based pipeline indices to MOT's 1-based convention (keep the same
    offsets for GT and predictions so they line up). String ids (the optional
    `label` field) are mapped to contiguous ints; int `player_id`s are kept as-is.
    """
    with open(tracks_path) as f:
        data = json.load(f)

    raw_ids = [d.get(id_field, d["player_id"]) for d in data]
    all_int = all(isinstance(r, int) for r in raw_ids)
    id_map = {} if all_int else {v: i for i, v in enumerate(sorted(set(raw_ids), key=str))}

    rows = []
    for d, raw in zip(data, raw_ids):
        x1, y1, x2, y2 = d["bbox"]
        rid = raw if all_int else id_map[raw]
        rows.append(MOTRow(d["frame_idx"] + frame_offset, int(rid) + id_offset,
                           x1, y1, x2 - x1, y2 - y1, float(d.get("conf", 1.0))))
    return write_mot(rows, out_path)
