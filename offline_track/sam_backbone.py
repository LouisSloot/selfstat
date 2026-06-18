"""SAM 2 video backbone: seed once, propagate identity through the clip.

This is the Direction-doc upgrade to Pass 1. Instead of collecting fragmented
tracklets and re-merging them by appearance (which is unreliable on small, busy
footage), we label each player once on the reference (first) frame and let SAM 2's
video memory propagate that identity through every later frame. Each seed box
becomes one SAM 2 object, so identity is continuous *by construction* — no
clustering step. Only the seeded players are tracked, so extra people in frame are
ignored automatically.

Precision gate: SAM 2 always emits a full mask for every seeded object — even when
that player is occluded or has left frame — so the mask drifts onto a neighbor or
the background and yields a phantom box (a false positive). We therefore keep a
propagated box only if it overlaps a real per-frame person detection, with at most
one object per detection (which also resolves two objects that merged onto one
person). On a 6-player eval clip this cut false positives 27 -> 4 and lifted IDF1
0.90 -> 0.97 for a small recall cost; disable with `gate_detector=None`.

Constraint: the SAM 2 video predictor holds the whole clip in memory, and cost
scales with objects x frames x resolution. Use it on a **short** clip (seconds,
not minutes) with the players seeded up front. Tune memory with `imgsz`.
"""

import numpy as np

from .tracklets import Detection, Tracklet


def _mask_to_box(mask, min_area=64):
    """Tight xyxy box around a boolean mask, or None if the mask is empty/tiny."""
    ys, xs = np.where(mask)
    if xs.size < min_area:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _gate_keep(obj_boxes, det_boxes, gate_iou):
    """Object ids to keep: each must overlap a person detection (IoU >= gate_iou),
    and at most one object survives per detection (the best match) — which drops
    both background-drift phantoms and merge duplicates."""
    best = {}  # det_idx -> (obj_id, iou)
    for k, b in obj_boxes.items():
        if not det_boxes:
            continue
        iou, di = max((_iou(b, d), i) for i, d in enumerate(det_boxes))
        if iou >= gate_iou and (di not in best or iou > best[di][1]):
            best[di] = (k, iou)
    return {oid for oid, _ in best.values()}


def track_with_sam2(video_path, seed_boxes, device="mps", model="sam2_t.pt",
                    imgsz=640, max_frames=None, min_area=64,
                    gate_detector="yolo11n.pt", gate_iou=0.3):
    """Propagate identities from frame-0 seed boxes through the video.

    Args:
        seed_boxes: list of xyxy boxes on the first frame; the k-th box becomes
            object/player id k (so pass them in player-id order).
        gate_detector: YOLO weights for the per-frame precision gate, or None to
            disable it (keeps every propagated box — higher recall, more phantoms).
    Returns:
        {player_id: Tracklet} — one tracklet per seeded player, ready for render.
    """
    from ultralytics.models.sam import SAM2VideoPredictor

    boxes = [[float(v) for v in b] for b in seed_boxes]
    if not boxes:
        raise ValueError("track_with_sam2 needs at least one seed box")

    overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=imgsz,
                     model=model, verbose=False, device=device, save=False)
    predictor = SAM2VideoPredictor(overrides=overrides)

    gate = None
    if gate_detector:
        from ultralytics import YOLO
        gate = YOLO(gate_detector)

    tracklets = {k: Tracklet(track_id=k) for k in range(len(boxes))}
    # Prompts are applied to the first frame and propagated forward with memory.
    stream = predictor(source=video_path, bboxes=boxes, stream=True)
    n = 0
    for frame_idx, result in enumerate(stream):
        if max_frames is not None and frame_idx >= max_frames:
            break
        n = frame_idx + 1
        if result.masks is None:
            continue
        md = result.masks.data.cpu().numpy()  # (num_obj, H, W) in object/seed order

        obj_boxes = {}
        for k in range(min(len(boxes), md.shape[0])):
            box = _mask_to_box(md[k] > 0.5, min_area=min_area)
            if box is not None:
                obj_boxes[k] = box

        if gate is not None and obj_boxes:
            r = gate.predict(result.orig_img, classes=[0], conf=0.4,
                             verbose=False, device=device)[0]
            det = [tuple(map(float, d)) for d in r.boxes.xyxy.cpu().numpy()]
            keep = _gate_keep(obj_boxes, det, gate_iou)
            obj_boxes = {k: b for k, b in obj_boxes.items() if k in keep}

        for k, box in obj_boxes.items():
            tracklets[k].detections.append(
                Detection(frame_idx=frame_idx, xyxy=box, conf=1.0))

        if frame_idx % 50 == 0:
            print(f"[sam2] propagated through frame {frame_idx}")

    kept = {k: t for k, t in tracklets.items() if t.num_frames > 0}
    print(f"[sam2] {len(boxes)} seeded objects over {n} frames -> "
          f"{len(kept)} with detections (gate={'on' if gate else 'off'})")
    return tracklets
