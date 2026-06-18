"""Tracking-evaluation harness for the offline player tracker.

Measures per-player identity quality (IDF1 / MOTA / ID-switches) of the two
backbones (`tracklets`, `sam2`) against hand-labeled ground truth on short clips,
so tracking quality is a number instead of a vibe (see CLAUDE.md "Eval before
tuning"). Modules:

- mot_io.py        pure MOTChallenge I/O + tracks.json -> MOT conversion
- metrics.py       py-motmetrics scoring + comparison table
- label_gt.py      OpenCV ground-truth labeler (needs a display; run locally)
- run_eval.py      orchestrator: extract clip -> label -> run backbones -> score
- verify_metrics.py label-free self-consistency checks for the metric plumbing

Run the scripts from the repo root, e.g. `python eval/verify_metrics.py`.
"""
