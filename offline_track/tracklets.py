"""Pass 1: detection + tracking over the whole video, collected into tracklets.

We lean on an off-the-shelf tracker (ByteTrack/BoT-SORT via Ultralytics) only for
*short-term* association. The resulting per-track sequences ("tracklets") are the
unit that gets globally re-identified later by clustering. Each tracklet keeps a
bounded, evenly-sampled set of crops (reservoir sampling) so appearance can be
summarized without holding every frame in memory.

By default we use a **segmentation** model and zero out each crop's background
before storing it. Raw person crops in wide/top-down footage are dominated by
shared scene pixels (floor, walls), which collapses appearance features across
different people; masking focuses the embedding on the person. Disable with
`mask_background=False` (falls back to a plain detection model + raw crops).
"""

import random
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO

PERSON_CLASS = 0  # COCO person class id


@dataclass
class Detection:
    frame_idx: int
    xyxy: tuple  # (x1, y1, x2, y2) in pixels
    conf: float


@dataclass
class Tracklet:
    track_id: int
    detections: list = field(default_factory=list)
    crops: list = field(default_factory=list)  # bounded sample of (masked) crops
    _seen: int = 0
    embedding: object = None  # np.ndarray, filled in during clustering

    def add(self, det, crop, max_crops, rng):
        self.detections.append(det)
        # Reservoir sampling keeps a uniform sample of crops across the whole
        # tracklet, so the appearance summary isn't biased to its first frames.
        self._seen += 1
        if len(self.crops) < max_crops:
            self.crops.append(crop)
        else:
            j = rng.randint(0, self._seen - 1)
            if j < max_crops:
                self.crops[j] = crop

    @property
    def num_frames(self):
        return len(self.detections)

    @property
    def first_frame(self):
        return self.detections[0].frame_idx if self.detections else -1


def _clamp(xyxy, w, h):
    x1, y1, x2, y2 = xyxy
    return max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))


def _masked_crop(frame, xyxy, mask=None):
    """Crop the frame to the box; if a full-frame binary mask is given, zero the
    background pixels within the crop. Returns None for a degenerate box."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _clamp(xyxy, w, h)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2].copy()
    if mask is not None:
        mc = mask[y1:y2, x1:x2]
        if mc.shape[:2] != crop.shape[:2]:
            mc = cv2.resize(mc, (crop.shape[1], crop.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        crop[mc == 0] = 0
    return crop


def _full_res_masks(result, shape):
    """Per-detection binary masks at original-frame resolution, in detection order
    (so mask[k] aligns with box k). Returns None if the model produced no masks."""
    if result.masks is None:
        return None
    H, W = shape
    out = []
    for m in result.masks.data.cpu().numpy():  # retina_masks=True -> already (H, W)
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        out.append((m > 0.5).astype(np.uint8))
    return out


def collect_tracklets(video_path, model_path=None, tracker="bytetrack.yaml",
                      conf=0.4, device="cpu", max_crops=16, max_frames=None,
                      seed=0, mask_background=True):
    """Run detection+tracking over the video. Returns {track_id: Tracklet}.

    With `mask_background=True` (default) a segmentation model is used and each
    crop's background is zeroed before storage/embedding.
    """
    if model_path is None:
        model_path = "yolo11n-seg.pt" if mask_background else "yolo11n.pt"

    rng = random.Random(seed)
    model = YOLO(model_path)
    tracklets = {}
    warned_no_mask = False

    track_kwargs = dict(
        source=video_path, stream=True, persist=True, tracker=tracker,
        classes=[PERSON_CLASS], conf=conf, device=device, verbose=False,
    )
    if mask_background:
        track_kwargs["retina_masks"] = True  # masks at native frame resolution
    stream = model.track(**track_kwargs)

    n_frames = 0
    for frame_idx, result in enumerate(stream):
        if max_frames is not None and frame_idx >= max_frames:
            break
        n_frames = frame_idx + 1

        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue  # no confirmed tracks this frame

        frame = result.orig_img
        ids = boxes.id.int().cpu().tolist()
        xyxys = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        masks = _full_res_masks(result, frame.shape[:2]) if mask_background else None
        if mask_background and masks is None and not warned_no_mask:
            print("[track] model produced no masks; embedding raw crops instead "
                  "(use a *-seg model for mask embedding)")
            warned_no_mask = True

        for k, (tid, xyxy, c) in enumerate(zip(ids, xyxys, confs)):
            mask = masks[k] if masks is not None and k < len(masks) else None
            crop = _masked_crop(frame, xyxy, mask)
            if crop is None:
                continue
            det = Detection(
                frame_idx=frame_idx,
                xyxy=tuple(float(v) for v in xyxy),
                conf=float(c),
            )
            tracklets.setdefault(tid, Tracklet(track_id=tid))
            tracklets[tid].add(det, crop, max_crops, rng)

        if frame_idx % 100 == 0:
            print(f"[track] frame {frame_idx}: {len(tracklets)} raw tracklets so far")

    print(f"[track] processed {n_frames} frames -> {len(tracklets)} raw tracklets")
    return tracklets
