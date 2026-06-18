"""Shot-chart stats stage: video + tracks.json -> events.json -> points / shot chart.

Decoupled from tracking (consumes a finished `tracks.json` from the offline
tracker, never hooks into pipeline.run). See CLAUDE.md "Phase 2 scoping" and the
canonical `events` data model. Entry point: ../run_stats.py.

Modules:
- ball.py        ball detection (COCO sports-ball via a large YOLO; pluggable)
- ball_track.py  per-frame detections -> a clean ball track (size/velocity gate + gap fill)
- hoops.py       load the one-time rim annotation; derive per-rim scoring zones
- shots.py       trajectory state machine -> shot attempts + make/miss
- attribute.py   shooter = tracked player nearest the ball at release
- events.py      canonical events table + per-player points
- chart.py       (Stage 3) court homography + matplotlib shot chart
"""
