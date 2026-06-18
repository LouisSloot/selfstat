"""Orchestrates the offline pipeline: video in -> labeled video (+ tracks.json) out.

    collect_tracklets  ->  embed + cluster  ->  render labeled video
"""

import os
import time

# DINOv2/ViT can hit ops MPS doesn't implement; let them fall back to CPU rather
# than crash. Harmless on CUDA/CPU. Must be set before heavy torch usage.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

from .cluster import cluster_tracklets  # noqa: E402
from .embedder import Embedder  # noqa: E402
from .render import export_tracks, render_labeled_video  # noqa: E402
from .tracklets import collect_tracklets  # noqa: E402


def get_device(prefer=None):
    if prefer:
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def auto_seed_boxes(video_path, ref_frame=0, device="cpu", conf=0.4, max_objects=None):
    """Person boxes on the reference frame via YOLO — a stand-in for manual labels,
    handy for testing the SAM 2 path without the GUI. Returns a list of xyxy boxes
    (largest first when capped by `max_objects`)."""
    import cv2
    import numpy as np
    from ultralytics import YOLO

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {ref_frame} of {video_path}")

    r = YOLO("yolo11n.pt").predict(frame, classes=[0], conf=conf,
                                   verbose=False, device=device)[0]
    b = r.boxes.xyxy.cpu().numpy()
    if max_objects and len(b) > max_objects:
        areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        b = b[np.argsort(-areas)[:max_objects]]
    return b.tolist()


def subclip(src, dst, start, n):
    """Write `n` frames of `src` starting at frame `start` to `dst` (mp4). SAM 2
    holds the whole source in memory, so it must be fed a short, bounded clip whose
    first frame is the seed/reference frame. Returns (dst, frames_written)."""
    import cv2

    cap = cv2.VideoCapture(src)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    written = 0
    while n is None or written < n:
        ok, fr = cap.read()
        if not ok:
            break
        out.write(fr)
        written += 1
    cap.release()
    out.release()
    return dst, written


def run_sam2(video_path, out_path, seed_boxes, labels=None, device=None,
             model="sam2_t.pt", imgsz=640, max_frames=None, dump_tracks=True,
             gate=True, gate_iou=0.3):
    """SAM 2 backbone: seed boxes on frame 0 of `video_path`, propagate, render."""
    from .sam_backbone import track_with_sam2

    device = get_device(device)
    print(f"[pipeline] SAM2 backbone  device={device}  model={model}  imgsz={imgsz}  "
          f"seeds={len(seed_boxes)}  gate={gate}")
    t0 = time.time()
    tracklets = track_with_sam2(video_path, seed_boxes, device=device, model=model,
                                imgsz=imgsz, max_frames=max_frames,
                                gate_detector=("yolo11n.pt" if gate else None),
                                gate_iou=gate_iou)
    assignment = {k: k for k in tracklets}
    render_labeled_video(video_path, out_path, tracklets, assignment,
                         max_frames=max_frames, labels=labels)
    if dump_tracks:
        export_tracks(tracklets, assignment,
                      os.path.splitext(out_path)[0] + ".tracks.json", labels=labels)
    print(f"[pipeline] SAM2 total={time.time() - t0:.1f}s")
    return out_path


def run(video_path, out_path, num_ids=None, model_path=None,
        tracker="bytetrack.yaml", conf=0.4, device=None, max_crops=16,
        max_frames=None, distance_threshold=0.25, min_frames=1,
        mask_background=True, dump_tracks=True):
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    device = get_device(device)
    print(f"[pipeline] device={device}  video={video_path}  mask_background={mask_background}")
    t0 = time.time()

    tracklets = collect_tracklets(
        video_path, model_path=model_path, tracker=tracker, conf=conf,
        device=device, max_crops=max_crops, max_frames=max_frames,
        mask_background=mask_background,
    )
    if not tracklets:
        raise RuntimeError("No tracklets found — no people detected/tracked.")
    t1 = time.time()

    embedder = Embedder(device=device)
    assignment = cluster_tracklets(
        tracklets, embedder, num_ids=num_ids,
        distance_threshold=distance_threshold, min_frames=min_frames,
    )
    n_ids = len(set(assignment.values())) if assignment else 0
    t2 = time.time()
    print(f"[pipeline] {len(tracklets)} tracklets -> {n_ids} stable identities")

    render_labeled_video(video_path, out_path, tracklets, assignment,
                         max_frames=max_frames)
    if dump_tracks:
        export_tracks(tracklets, assignment, os.path.splitext(out_path)[0] + ".tracks.json")
    t3 = time.time()

    print(f"[pipeline] timing: track={t1 - t0:.1f}s  embed+cluster={t2 - t1:.1f}s  "
          f"render={t3 - t2:.1f}s  total={t3 - t0:.1f}s")
    return out_path
