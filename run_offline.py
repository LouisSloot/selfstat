"""CLI entry point for the offline tracklet-clustering player tracker.

Example:
    python run_offline.py --video clip.mp4 --num-ids 6
    python run_offline.py --video clip.mp4 --out out.mp4 --max-frames 300
"""

import argparse
import os

from offline_track.pipeline import run


def _run_sam2_cli(args, out):
    """Seed players (manually or auto) on a reference frame, then propagate with
    SAM 2 over a short bounded clip whose first frame is the seed frame."""
    from offline_track.pipeline import auto_seed_boxes, get_device, run_sam2, subclip

    n = args.max_frames or 120  # SAM 2 is memory-bound; keep the clip short
    ref = args.ref_frame
    labels = seeds = None

    if args.label:
        from detect import YOLODetector
        from label_players import label_seed_boxes
        sv_ids, seeds, ref = label_seed_boxes(YOLODetector("yolo11n.pt"), args.video)
        if not seeds:
            print("No players labeled — aborting.")
            return
        labels = {k: sv_ids[k] for k in range(len(seeds))}

    clip = "/tmp/selfstat_sam2_clip.mp4"
    _, got = subclip(args.video, clip, ref, n)
    print(f"[run] SAM2 input: {got} frames from ref-frame {ref}")

    if seeds is None:  # auto-seed stand-in for manual labels
        seeds = auto_seed_boxes(clip, ref_frame=0, device=get_device(args.device),
                                conf=args.conf, max_objects=args.max_objects)
        print(f"[run] auto-seeded {len(seeds)} people from frame 0 "
              f"(pass --label to seed manually)")
    if not seeds:
        print("No seeds found — aborting.")
        return

    run_sam2(clip, out, seeds, labels=labels, device=args.device,
             model=args.sam_model, imgsz=args.imgsz)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="input video path")
    ap.add_argument("--out", default=None,
                    help="output video path (default: ./annotated_replays/tracked_<name>.mp4)")
    ap.add_argument("--num-ids", type=int, default=None,
                    help="known number of distinct people (closed-set). "
                         "If omitted, count is inferred via --cluster-threshold.")
    ap.add_argument("--model", default=None,
                    help="YOLO weights (auto-downloads; default yolo11n-seg.pt with "
                         "masking, else yolo11n.pt)")
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
    ap.add_argument("--no-mask", dest="mask_background", action="store_false",
                    help="embed raw crops instead of background-masked crops")
    ap.set_defaults(mask_background=True)
    # --- SAM 2 backbone (seed once on a reference frame, propagate identity) ---
    ap.add_argument("--backbone", choices=["tracklets", "sam2"], default="tracklets",
                    help="tracklets = detect+track+cluster (default); "
                         "sam2 = seed players once and propagate")
    ap.add_argument("--label", action="store_true",
                    help="[sam2] open the manual labeling UI to seed players "
                         "(needs a display)")
    ap.add_argument("--ref-frame", type=int, default=0,
                    help="[sam2 auto-seed] frame to seed from")
    ap.add_argument("--sam-model", default="sam2_t.pt", help="[sam2] SAM 2 checkpoint")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="[sam2] inference image size (lower = less memory)")
    ap.add_argument("--max-objects", type=int, default=None,
                    help="[sam2 auto-seed] cap seeded people (largest first)")
    args = ap.parse_args()

    out = args.out
    if out is None:
        base = os.path.splitext(os.path.basename(args.video))[0]
        out = f"./annotated_replays/tracked_{base}.mp4"

    if args.backbone == "sam2":
        _run_sam2_cli(args, out)
        return

    run(args.video, out, num_ids=args.num_ids, model_path=args.model,
        tracker=args.tracker, conf=args.conf, device=args.device,
        max_crops=args.crops_per_tracklet, max_frames=args.max_frames,
        distance_threshold=args.cluster_threshold, min_frames=args.min_frames,
        mask_background=args.mask_background)


if __name__ == "__main__":
    main()
