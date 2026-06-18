"""SAM 2 video backbone: seed once, propagate identity through the clip.

This is the Direction-doc upgrade to Pass 1. Instead of collecting fragmented
tracklets and re-merging them by appearance (which is unreliable on small, busy
footage), we label each player once on the reference (first) frame and let SAM 2's
video memory propagate that identity through every later frame. Each seed box
becomes one SAM 2 object, so identity is continuous *by construction* — no
clustering step. Only the seeded players are tracked, so extra people in frame are
ignored automatically.

Constraint: the SAM 2 video predictor holds the whole clip in memory, and cost
scales with objects x frames x resolution. Use it on a **short** clip (seconds,
not minutes) with the players seeded up front. Tune memory with `imgsz`.
"""

import numpy as np

from .tracklets import Detection, Tracklet


def _mask_to_box(mask, min_area=64):
    """Tight xyxy box around a boolean mask, or None if the object is absent/tiny
    this frame (SAM 2 emits an empty/near-empty mask when it loses an object)."""
    ys, xs = np.where(mask)
    if xs.size < min_area:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def track_with_sam2(video_path, seed_boxes, device="mps", model="sam2_t.pt",
                    imgsz=640, max_frames=None, min_area=64):
    """Propagate identities from frame-0 seed boxes through the video.

    Args:
        seed_boxes: list of xyxy boxes on the first frame; the k-th box becomes
            object/player id k (so pass them in player-id order).
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
        for k in range(min(len(boxes), md.shape[0])):
            box = _mask_to_box(md[k] > 0.5, min_area=min_area)
            if box is None:
                continue  # player not visible / lost this frame
            tracklets[k].detections.append(
                Detection(frame_idx=frame_idx, xyxy=box, conf=1.0)
            )
        if frame_idx % 50 == 0:
            print(f"[sam2] propagated through frame {frame_idx}")

    kept = {k: t for k, t in tracklets.items() if t.num_frames > 0}
    print(f"[sam2] {len(boxes)} seeded objects over {n} frames -> "
          f"{len(kept)} with detections")
    return tracklets
