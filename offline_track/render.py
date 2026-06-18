"""Pass 2: redraw the source video with one labeled box per detection.

Also exports the per-frame tracking table (the `tracks` half of the canonical data
model in CLAUDE.md), which is what phase-2 stat code will consume.
"""

import json
import os

import cv2
import numpy as np


def color_for(pid):
    """Deterministic, well-separated color per player id (BGR)."""
    rng = np.random.RandomState(pid * 9973 + 7)
    c = rng.randint(64, 256, size=3)
    return int(c[0]), int(c[1]), int(c[2])


def _index_by_frame(tracklets, assignment):
    """frame_idx -> list of (xyxy, conf, player_id)."""
    per_frame = {}
    for tid, t in tracklets.items():
        pid = assignment.get(tid)
        if pid is None:
            continue
        for det in t.detections:
            per_frame.setdefault(det.frame_idx, []).append((det.xyxy, det.conf, pid))
    return per_frame


def render_labeled_video(video_path, out_path, tracklets, assignment, max_frames=None,
                         labels=None):
    per_frame = _index_by_frame(tracklets, assignment)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # scale annotation size with resolution so labels stay legible at 1080p+
    font_scale = max(0.5, h / 1080.0 * 0.9)
    thick = max(1, round(h / 1080.0 * 2))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    # mp4v is bundled with OpenCV; avc1/H.264 often isn't and fails silently.
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")

    frame_idx = written = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames is not None and frame_idx >= max_frames):
            break
        for xyxy, _conf, pid in per_frame.get(frame_idx, []):
            x1, y1, x2, y2 = map(int, xyxy)
            color = color_for(pid)
            text = str(labels.get(pid, pid)) if labels else f"Player {pid}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            cv2.putText(frame, text, (x1, max(int(20 * font_scale), y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick)
        out.write(frame)
        frame_idx += 1
        written += 1

    cap.release()
    out.release()
    print(f"[render] wrote {written} frames -> {out_path}")
    return out_path


def export_tracks(tracklets, assignment, path, labels=None):
    """Dump the per-frame tracking table as JSON (sorted by frame, then player)."""
    rows = []
    for tid, t in tracklets.items():
        pid = assignment.get(tid)
        if pid is None:
            continue
        for det in t.detections:
            row = {
                "frame_idx": det.frame_idx,
                "player_id": pid,
                "bbox": [round(v, 1) for v in det.xyxy],
                "conf": round(det.conf, 3),
            }
            if labels:
                row["label"] = str(labels.get(pid, pid))
            rows.append(row)
    rows.sort(key=lambda r: (r["frame_idx"], r["player_id"]))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f)
    print(f"[render] wrote {len(rows)} track rows -> {path}")
    return path
