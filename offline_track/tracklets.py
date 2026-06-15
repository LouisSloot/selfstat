"""Pass 1: detection + tracking over the whole video, collected into tracklets.

We lean on an off-the-shelf tracker (ByteTrack/BoT-SORT via Ultralytics) only for
*short-term* association, where it's reliable. The resulting per-track sequences
("tracklets") are the unit that gets globally re-identified later by clustering.
Each tracklet keeps a bounded, evenly-sampled set of crops (reservoir sampling)
so appearance can be summarized without holding every frame in memory.
"""

import random
from dataclasses import dataclass, field

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
    crops: list = field(default_factory=list)  # bounded sample of uint8 BGR crops
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


def _crop(frame, xyxy):
    """Clamp a bbox to the frame and return a copied crop, or None if degenerate."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def collect_tracklets(video_path, model_path="yolo11n.pt", tracker="bytetrack.yaml",
                      conf=0.4, device="cpu", max_crops=16, max_frames=None, seed=0):
    """Run detection+tracking over the video. Returns {track_id: Tracklet}."""
    rng = random.Random(seed)
    model = YOLO(model_path)
    tracklets = {}

    stream = model.track(
        source=video_path, stream=True, persist=True, tracker=tracker,
        classes=[PERSON_CLASS], conf=conf, device=device, verbose=False,
    )

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

        for tid, xyxy, c in zip(ids, xyxys, confs):
            crop = _crop(frame, xyxy)
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
