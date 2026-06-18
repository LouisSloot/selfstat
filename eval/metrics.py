"""Tracking metrics via py-motmetrics: IDF1, MOTA, ID-switches, and friends.

IDF1 and num_switches are the headline numbers for "does each player keep ONE
id across the clip?". Matching is IoU-Hungarian per frame; ground truth and
predictions live in separate id spaces (motmetrics matches by overlap, then
scores identity consistency), so their id *values* need not coincide.
"""

import argparse

import numpy as np

# motmetrics 1.4.0 still calls np.asfarray, removed in NumPy 2.0. Shim it rather
# than pin numpy down (torch/opencv/ultralytics want numpy 2.x here).
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

import motmetrics as mm  # noqa: E402
import pandas as pd  # noqa: E402

# Reported metrics, in display order.
METRICS = [
    "idf1", "idp", "idr",
    "mota", "motp",
    "num_switches", "num_fragmentations",
    "mostly_tracked", "partially_tracked", "mostly_lost",
    "num_false_positives", "num_misses",
    "num_objects", "num_unique_objects",
]


def _load(path):
    # DataFrame indexed by (FrameId, Id) with X, Y, Width, Height columns.
    return mm.io.loadtxt(path, fmt="mot15-2D")


def _frame_dict(df):
    """frame_id -> (list of ids, Nx4 [X,Y,W,H] array)."""
    out = {}
    for fid, sub in df.groupby(level=0):
        ids = list(sub.index.get_level_values(1))
        boxes = sub[["X", "Y", "Width", "Height"]].to_numpy(dtype=float)
        out[fid] = (ids, boxes)
    return out


def _accumulate(gt_fd, pred_fd, iou, frames):
    acc = mm.MOTAccumulator(auto_id=False)
    for f in frames:
        gids, gboxes = gt_fd.get(f, ([], np.empty((0, 4))))
        pids, pboxes = pred_fd.get(f, ([], np.empty((0, 4))))
        if len(gids) and len(pids):
            # max_iou is a *distance* cutoff (1 - IoU); pairs below `iou` -> unmatchable.
            dists = mm.distances.iou_matrix(gboxes, pboxes, max_iou=1.0 - iou)
        else:
            dists = np.empty((len(gids), len(pids)))
        acc.update(gids, pids, dists, frameid=f)
    return acc


def evaluate(gt_path, pred_paths, iou=0.5, eval_frames_from_gt=True):
    """Score one or more predictions against GT. Returns a combined summary frame.

    pred_paths: {backbone_name: mot_txt_path}. With `eval_frames_from_gt` (default)
    only frames present in GT are scored — essential when GT is labeled sparsely
    (every Nth frame), or prediction-only frames would all count as false positives.
    """
    gt = _load(gt_path)
    gt_fd = _frame_dict(gt)
    gt_frames = sorted(gt_fd)
    mh = mm.metrics.create()

    summaries = []
    for name, path in pred_paths.items():
        pred_fd = _frame_dict(_load(path))
        frames = gt_frames if eval_frames_from_gt else sorted(set(gt_frames) | set(pred_fd))
        acc = _accumulate(gt_fd, pred_fd, iou, frames)
        summaries.append(mh.compute(acc, metrics=METRICS, name=name))
    return pd.concat(summaries)


def render(summary):
    """Pretty MOTChallenge-style table string."""
    mh = mm.metrics.create()
    return mm.io.render_summary(summary, formatters=mh.formatters,
                                namemap=mm.io.motchallenge_metric_names)


def main():
    ap = argparse.ArgumentParser(description="Score tracking predictions vs GT (motmetrics).")
    ap.add_argument("--gt", required=True, help="ground-truth MOT txt")
    ap.add_argument("--pred", action="append", required=True, metavar="NAME=PATH",
                    help="prediction as name=path; repeatable for multiple backbones")
    ap.add_argument("--iou", type=float, default=0.5, help="min IoU to match (default 0.5)")
    ap.add_argument("--all-frames", action="store_true",
                    help="score union of GT+pred frames instead of GT frames only")
    ap.add_argument("--csv", default=None, help="optional path to write the table as CSV")
    args = ap.parse_args()

    preds = {}
    for spec in args.pred:
        name, path = spec.split("=", 1)
        preds[name] = path
    summary = evaluate(args.gt, preds, iou=args.iou, eval_frames_from_gt=not args.all_frames)
    print(render(summary))
    if args.csv:
        summary.to_csv(args.csv)
        print(f"\n[metrics] wrote {args.csv}")


if __name__ == "__main__":
    main()
