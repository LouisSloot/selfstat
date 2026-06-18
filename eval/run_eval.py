"""Orchestrate a tracking eval on one clip: extract -> (label) -> run both
backbones on that same clip -> convert -> score.

Both backbones run on the SAME extracted clip.mp4, so every frame index (GT and
both predictions) is clip-relative 0-based — this is what keeps them aligned.
Do NOT point a backbone at the full source here; that reintroduces the
clip-vs-source frame-offset bug.

    python eval/run_eval.py --name pickup1 --start 39600 --frames 120 --num-ids 6
    # -> if no gt.txt yet, prints the label_gt.py command; re-run after labeling to score
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # eval/

import metrics  # noqa: E402
import mot_io  # noqa: E402
from offline_track.pipeline import (auto_seed_boxes, get_device, run, run_sam2,  # noqa: E402
                                    subclip)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="eval clip name (dir under eval/clips/)")
    ap.add_argument("--source", default="data_dir/gameplay.mp4")
    ap.add_argument("--start", type=int, default=0, help="source frame to start the clip at")
    ap.add_argument("--frames", type=int, default=120, help="clip length (keep short for sam2)")
    ap.add_argument("--num-ids", type=int, default=6, help="roster size (closed-set / sam2 seeds)")
    ap.add_argument("--stride", type=int, default=5, help="GT label stride (for the print-out)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--imgsz", type=int, default=640, help="sam2 inference size")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--backbones", default="tracklets,sam2", help="comma list to run")
    args = ap.parse_args()

    clip_dir = os.path.join("eval", "clips", args.name)
    preds_dir = os.path.join(clip_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)
    clip = os.path.join(clip_dir, "clip.mp4")
    gt = os.path.join(clip_dir, "gt.txt")

    # 1. extract clip (idempotent) — both backbones then share its 0-based index.
    if not os.path.exists(clip):
        _, got = subclip(args.source, clip, args.start, args.frames)
        with open(os.path.join(clip_dir, "clip_meta.json"), "w") as f:
            json.dump({"source": args.source, "start": args.start,
                       "frames": got, "stride": args.stride}, f, indent=2)
        print(f"[eval] extracted {got} frames from {args.source}@{args.start} -> {clip}")
    else:
        print(f"[eval] using existing clip {clip}")

    dev = get_device(args.device)
    backbones = [b for b in args.backbones.split(",") if b]
    pred_paths = {}

    # 2. run each backbone on the clip, convert its tracks.json -> MOT.
    if "tracklets" in backbones:
        out = os.path.join(preds_dir, "tracklets.mp4")
        run(clip, out, num_ids=args.num_ids, device=args.device, max_frames=args.frames)
        pred_paths["tracklets"] = mot_io.tracks_json_to_mot(
            os.path.splitext(out)[0] + ".tracks.json", os.path.join(preds_dir, "tracklets.txt"))
    if "sam2" in backbones:
        out = os.path.join(preds_dir, "sam2.mp4")
        seeds = auto_seed_boxes(clip, ref_frame=0, device=dev, conf=args.conf,
                                max_objects=args.num_ids)
        run_sam2(clip, out, seeds, device=args.device, imgsz=args.imgsz)
        pred_paths["sam2"] = mot_io.tracks_json_to_mot(
            os.path.splitext(out)[0] + ".tracks.json", os.path.join(preds_dir, "sam2.txt"))

    # 3. score (needs hand-labeled GT).
    if not os.path.exists(gt):
        print("\n[eval] Predictions ready, but no ground truth yet.")
        print("Label it locally (needs a display), then re-run this command to score:")
        print(f"    python eval/label_gt.py --clip {clip} --stride {args.stride}")
        return

    summary = metrics.evaluate(gt, pred_paths, iou=args.iou)
    print("\n=== tracking metrics (IDF1 / IDsw are the headline) ===")
    print(metrics.render(summary))
    csv = os.path.join(clip_dir, "results.csv")
    summary.to_csv(csv)
    print(f"\n[eval] wrote {csv}")


if __name__ == "__main__":
    main()
