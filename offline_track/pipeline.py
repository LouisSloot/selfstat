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


def run(video_path, out_path, num_ids=None, model_path="yolo11n.pt",
        tracker="bytetrack.yaml", conf=0.4, device=None, max_crops=16,
        max_frames=None, distance_threshold=0.25, min_frames=1, dump_tracks=True):
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    device = get_device(device)
    print(f"[pipeline] device={device}  video={video_path}")
    t0 = time.time()

    tracklets = collect_tracklets(
        video_path, model_path=model_path, tracker=tracker, conf=conf,
        device=device, max_crops=max_crops, max_frames=max_frames,
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
