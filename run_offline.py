"""CLI entry point for the offline tracklet-clustering player tracker.

Example:
    python run_offline.py --video clip.mp4 --num-ids 6
    python run_offline.py --video clip.mp4 --out out.mp4 --max-frames 300
"""

import argparse
import os

from offline_track.pipeline import run


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="input video path")
    ap.add_argument("--out", default=None,
                    help="output video path (default: ./annotated_replays/tracked_<name>.mp4)")
    ap.add_argument("--num-ids", type=int, default=None,
                    help="known number of distinct people (closed-set). "
                         "If omitted, count is inferred via --cluster-threshold.")
    ap.add_argument("--model", default="yolo11n.pt", help="YOLO weights (auto-downloads)")
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    help="Ultralytics tracker config (bytetrack.yaml / botsort.yaml)")
    ap.add_argument("--conf", type=float, default=0.4, help="detection confidence threshold")
    ap.add_argument("--device", default=None, help="mps / cuda / cpu (auto if unset)")
    ap.add_argument("--max-frames", type=int, default=None, help="cap frames (debugging)")
    ap.add_argument("--crops-per-tracklet", type=int, default=16,
                    help="crops sampled per tracklet for the appearance embedding")
    ap.add_argument("--cluster-threshold", type=float, default=0.25,
                    help="cosine-distance cut when --num-ids is not given")
    ap.add_argument("--min-frames", type=int, default=1,
                    help="ignore tracklets shorter than this many frames")
    args = ap.parse_args()

    out = args.out
    if out is None:
        base = os.path.splitext(os.path.basename(args.video))[0]
        out = f"./annotated_replays/tracked_{base}.mp4"

    run(args.video, out, num_ids=args.num_ids, model_path=args.model,
        tracker=args.tracker, conf=args.conf, device=args.device,
        max_crops=args.crops_per_tracklet, max_frames=args.max_frames,
        distance_threshold=args.cluster_threshold, min_frames=args.min_frames)


if __name__ == "__main__":
    main()
