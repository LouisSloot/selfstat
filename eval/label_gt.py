"""OpenCV ground-truth labeler for tracking eval. Needs a display — run locally.

Per (sampled) frame it proposes YOLO person boxes and **carries the previous
labeled frame's ids forward by IoU**, so most frames are just confirm-and-advance.
You can correct ids, add people YOLO missed, and delete false positives, so the
result is TRUE ground truth (not merely the detector's output). Writes a
MOTChallenge gt.txt plus a meta.json recording the label stride.

Usage (from repo root, on a machine with a display):
    python eval/label_gt.py --clip eval/clips/<name>/clip.mp4 --stride 5 [--resume]
Keys: left-click=assign/fix id · n=add box (click TL then BR) · right-click=delete
      · r=re-propose YOLO · SPACE/ENTER=commit+next · b=back · q/ESC=save+quit
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # eval/ for mot_io

import mot_io  # noqa: E402
from detect import YOLODetector  # noqa: E402
from label_players import IDInputWindow  # noqa: E402
from utils import findIOU, get_corners, get_person_boxes  # noqa: E402

WIN = "Label GT"


def _yolo_boxes(detector, frame):
    res = detector.detect_frame(frame)
    boxes = []
    for box in get_person_boxes(res):
        x1, y1, x2, y2 = get_corners(box)  # one-shot map -> unpack immediately
        boxes.append([float(x1), float(y1), float(x2), float(y2)])
    return boxes


def _carry_forward(prev, proposals, iou_thr):
    """Pre-assign carried ids to proposed boxes by IoU. prev: [[id, xyxy], ...]
    (committed previous frame). Returns [[id_or_None, xyxy], ...] for proposals."""
    out = [[None, b] for b in proposals]
    if prev and proposals:
        cost = np.ones((len(prev), len(proposals)))
        for i, (_pid, pb) in enumerate(prev):
            for j, b in enumerate(proposals):
                cost[i, j] = 1.0 - findIOU(pb, b)
        for i, j in zip(*linear_sum_assignment(cost)):
            if 1.0 - cost[i, j] >= iou_thr:
                out[j][0] = prev[i][0]
    return out


def _hit(boxes, x, y):
    """Index of the (smallest) box containing (x, y), else -1."""
    best, best_area = -1, float("inf")
    for i, (_id, (x1, y1, x2, y2)) in enumerate(boxes):
        if x1 <= x <= x2 and y1 <= y <= y2:
            area = (x2 - x1) * (y2 - y1)
            if area < best_area:
                best, best_area = i, area
    return best


class GTLabeler:
    def __init__(self, clip, stride, conf, model):
        self.cap = cv2.VideoCapture(clip)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {clip}")
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.stride = stride
        self.frames = sorted(set(list(range(0, self.n, stride)) + [self.n - 1]))
        self.detector = YOLODetector(model, conf=conf)
        self.gt = {}                 # frame_idx -> [[label, xyxy], ...]
        self.cur = 0                 # index into self.frames
        self.add_pt = None           # first corner while adding a box
        self.mode = "edit"           # "edit" | "add"

    def _frame_img(self, fidx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, img = self.cap.read()
        return img if ok else np.zeros((self.h, self.w, 3), np.uint8)

    def _ensure(self, fidx, prev_committed):
        if fidx not in self.gt:
            props = _yolo_boxes(self.detector, self._frame_img(fidx))
            self.gt[fidx] = _carry_forward(prev_committed, props, 0.3)

    def _prev_committed(self, idx):
        for k in range(idx - 1, -1, -1):
            f = self.frames[k]
            if f in self.gt:
                return [[i, b] for i, b in self.gt[f] if i is not None]
        return []

    def _draw(self, img, fidx):
        for label, (x1, y1, x2, y2) in self.gt.get(fidx, []):
            c = (0, 200, 0) if label is not None else (0, 220, 220)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
            txt = str(label) if label is not None else "?"
            cv2.putText(img, txt, (int(x1), max(14, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        lab = sum(1 for i, _ in self.gt.get(fidx, []) if i is not None)
        cv2.putText(img, f"frame {fidx}  ({self.cur + 1}/{len(self.frames)})  labeled={lab}"
                    f"  mode={self.mode}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "click=id  n=add  rclick=del  r=reYOLO  SPACE=next  b=back  q=quit",
                    (8, self.h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _on_mouse(self, event, x, y, flags, _param):
        fidx = self.frames[self.cur]
        boxes = self.gt.setdefault(fidx, [])
        if self.mode == "add" and event == cv2.EVENT_LBUTTONDOWN:
            if self.add_pt is None:
                self.add_pt = (x, y)
            else:
                x1, y1 = self.add_pt
                box = [float(min(x1, x)), float(min(y1, y)), float(max(x1, x)), float(max(y1, y))]
                pid = IDInputWindow().get_player_id(x, y)
                if pid:
                    boxes.append([pid.strip(), box])
                self.add_pt = None
                self.mode = "edit"
        elif event == cv2.EVENT_LBUTTONDOWN:
            i = _hit(boxes, x, y)
            if i >= 0:
                pid = IDInputWindow().get_player_id(x, y)
                if pid:
                    boxes[i][0] = pid.strip()
        elif event == cv2.EVENT_RBUTTONDOWN:
            i = _hit(boxes, x, y)
            if i >= 0:
                boxes.pop(i)

    def run(self, out_path):
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN, self._on_mouse)
        while True:
            fidx = self.frames[self.cur]
            self._ensure(fidx, self._prev_committed(self.cur))
            img = self._frame_img(fidx).copy()
            self._draw(img, fidx)
            cv2.imshow(WIN, img)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in (32, 13):                      # SPACE / ENTER -> next
                self.cur = min(self.cur + 1, len(self.frames) - 1)
            elif key == ord("b"):
                self.cur = max(self.cur - 1, 0)
            elif key == ord("n"):
                self.mode = "add"; self.add_pt = None
            elif key == ord("r"):
                self.gt[fidx] = _carry_forward(self._prev_committed(self.cur),
                                               _yolo_boxes(self.detector, self._frame_img(fidx)), 0.3)
        cv2.destroyAllWindows()
        self.cap.release()
        return self._save(out_path)

    def _save(self, out_path):
        # stable label -> contiguous int id (1-based); ids only need to be consistent.
        labels = sorted({lab for boxes in self.gt.values() for lab, _ in boxes if lab is not None}, key=str)
        id_map = {lab: i + 1 for i, lab in enumerate(labels)}
        rows = []
        for fidx, boxes in self.gt.items():
            for lab, (x1, y1, x2, y2) in boxes:
                if lab is None:
                    continue
                rows.append(mot_io.MOTRow(fidx + 1, id_map[lab], x1, y1, x2 - x1, y2 - y1, 1.0))
        mot_io.write_mot(rows, out_path)
        meta = {"clip_frames": self.n, "fps": self.fps, "w": self.w, "h": self.h,
                "stride": self.stride, "labeled_frames": sorted(self.gt),
                "label_map": id_map}
        with open(os.path.join(os.path.dirname(out_path), "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[label_gt] wrote {len(rows)} GT rows ({len(id_map)} ids) -> {out_path}")
        return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True, help="clip.mp4 to label (frames 0-based)")
    ap.add_argument("--stride", type=int, default=5, help="label every Nth frame")
    ap.add_argument("--conf", type=float, default=0.3, help="YOLO proposal confidence")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--out", default=None, help="gt.txt path (default: alongside the clip)")
    ap.add_argument("--resume", action="store_true", help="reload an existing gt.txt to continue")
    args = ap.parse_args()

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.clip)), "gt.txt")
    labeler = GTLabeler(args.clip, args.stride, args.conf, args.model)
    if args.resume and os.path.exists(out):
        for r in mot_io.read_mot(out):
            labeler.gt.setdefault(r.frame - 1, []).append(
                [str(r.id), [r.x, r.y, r.x + r.w, r.y + r.h]])
        print(f"[label_gt] resumed {out}")
    labeler.run(out)


if __name__ == "__main__":
    main()
