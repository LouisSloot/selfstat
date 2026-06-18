"""Ball + hoop detection for the shot-chart stage.

Uses a basketball-specific YOLOv8 model (classes `0=Basketball`, `1=Basketball
Hoop`) — `best.pt` from avishah3/AI-Basketball-Shot-Detection-Tracker, saved here
as `basketball.pt`:

    curl -sL -o basketball.pt \\
      https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker/raw/master/best.pt

Validated on the gym footage: it tracks the ball through a full shot arc at
0.7-0.9 confidence and detects the rims, where COCO "sports ball" failed (lost the
ball mid-flight and fired on orange wall objects). Detector-agnostic downstream:
any YOLO with ball/hoop classes drops in via `ball_class`/`hoop_class`.
"""

import cv2

from ultralytics import YOLO


class BallHoopDetector:
    def __init__(self, model="basketball.pt", conf=0.3, device="mps",
                 ball_class=0, hoop_class=1):
        self.model = YOLO(model)
        self.conf = conf
        self.device = device
        self.ball_class = ball_class
        self.hoop_class = hoop_class

    def detect_frame(self, frame):
        """Return (balls, hoops), each a list of (xyxy, conf)."""
        r = self.model.predict(frame, conf=self.conf, verbose=False,
                               device=self.device)[0]
        balls, hoops = [], []
        for b in r.boxes:
            xy = tuple(float(v) for v in b.xyxy.cpu().numpy().reshape(-1))
            c, cls = float(b.conf), int(b.cls)
            if cls == self.ball_class:
                balls.append((xy, c))
            elif cls == self.hoop_class:
                hoops.append((xy, c))
        return balls, hoops

    def detect_window(self, video_path, start=0, n=None, progress=200):
        """Run over [start, start+n). Returns (ball_dets, hoop_dets) — parallel
        lists (one entry per frame) of `[(xyxy, conf), ...]`."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ball_dets, hoop_dets, i = [], [], 0
        while n is None or i < n:
            ok, frame = cap.read()
            if not ok:
                break
            b, h = self.detect_frame(frame)
            ball_dets.append(b)
            hoop_dets.append(h)
            i += 1
            if progress and i % progress == 0:
                print(f"[detect] frame {i}")
        cap.release()
        return ball_dets, hoop_dets
