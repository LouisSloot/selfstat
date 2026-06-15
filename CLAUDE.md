# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context & current status

SelfStat is an early-stage personal project: an automated stat-tracking system for **amateur** basketball footage (phone-filmed pickup games, players in street clothes — not broadcast NBA video). It is a revival of a stalled summer project.

Development is split into two phases, and the project never cleared phase 1:

- **Phase 1 — persistent player tracking (the blocker).** Given a video, assign a stable unique ID to each player and keep it consistent across the whole clip. This is hard for casual footage and is the reason the project stalled. The original online approach (YOLO + an OSNet re-ID embedding fused with color and motion) reportedly worked on the one available "casual basketball" dataset but was **not believed to generalize**; it has since been replaced by the offline pipeline (see Direction and Architecture).
- **Phase 2 — stat annotation.** Classify on-court actions (currently dribble / layup / shoot) and attribute them to tracked players. The action-recognition model exists (`act_reg_model/`) but is **not yet wired into** the tracking pipeline.

The deeper obstacle is data: there is no good labeled dataset for tracking stats from casual basketball footage. Open ideas being explored include training on NBA footage to test generalization, or finetuning on top of that. Keep this framing in mind — code shaped around "label a few reference frames by hand, then track" exists *because* labeled data is scarce.

**Resources / constraints.** The original work was done under heavy compute limits (paid Colab) by someone newer to ML. That has changed: there is now access to a cluster with **8× H100**, to be used **sparingly**. So compute-heavy directions that were previously off the table — training a detector/tracker on NBA footage, finetuning OSNet, larger action models — are now viable; default to right-sized experiments and scale up deliberately rather than reaching for the full cluster by habit.

The repo was left mid-refactor with several known bugs (the dead/legacy files have since been removed; see **Known bugs** below). Treat the two pipelines as works-in-progress, not a working end-to-end system.

## Direction (decided — offline, closed-set)

The project is committed to an **offline, post-hoc** paradigm: process a finished video and emit stats, not a real-time courtside overlay. This is the foundational architectural decision — build against it, and don't re-litigate it without good reason.

**Why offline.** Per-player stats require *globally consistent* identity so one player's events aggregate correctly. Online per-frame association can't hold identity through the occlusion clusters that pervade basketball, and the old `IdentityManager` (now removed) made it worse: a per-frame Hungarian match plus an EMA that writes the *current* crop into the reference embedding means one bad occlusion swap contaminates both players' references and the error compounds. That structural flaw — not embedding quality or weight tuning — is why phase 1 stalled. Offline removes the constraint: use the whole video, build identity globally, and use lookahead for event attribution.

**Target tracking architecture** (replaces the per-frame matcher):
- **SAM 2** as the tracking backbone, *seeded from the existing reference-frame labeling UI* (`label_players.py` clicks → SAM 2 prompts). Its memory handles occlusion far better than IoU + embedding association.
- Build short, high-confidence **tracklets** (association is easy where there's no occlusion), then **globally cluster tracklets into identities** using strong appearance features (**DINOv2**, not OSNet `x0_25`) averaged over each whole tracklet. Identity is recovered by tracklet→cluster matching, never by propagating an ID across a contaminating frame chain.
- Baseline to beat first: off-the-shelf **BoT-SORT / Deep-OC-SORT** (don't hand-roll a tracker). **SportsMOT** is the relevant sports benchmark/data for pretraining and validating the machinery.

**Closed-set, with a substitution caveat.** Exploit the known, small roster — but cluster into "however many distinct people *appear*," then map clusters → roster. Do **not** hard-code N = players-on-court, or substitutions break it.

**Eval before tuning.** The prior "got fine results but skeptical it generalizes" reflects having had no held-out eval set. Before modeling work, hand-label a few short pickup clips with ground-truth player IDs and track **HOTA / IDF1**. This needs no training data and turns vibes into a metric. Keep training-data needs (minimal on the foundation-model route) separate from eval-data needs (required, cheap).

**Capture constraint.** If footage can be dictated, a fixed, elevated, wide camera (whole court in frame) removes the worst failure modes — camera motion, players leaving frame, scale changes — and drops the problem from research-hard to engineering-hard. Don't anchor feasibility on handheld panning footage.

**Canonical data model** — the contract between tracking and stats:

```
tracks:  frame_idx, t, player_id, bbox, [pose], [has_ball]
events:  t, event_type, actor_player_id, [court_location], [value]
```

Per-player stats are a `GROUP BY player_id` over `events` — a join, not a tracking problem. All difficulty is in *producing* these tables, and offline makes both easier (global identity; plus bidirectional context and global constraints for event attribution and error correction). This is why offline does **not** foreclose per-player stats — it's the best route to them. The only thing offline forecloses is a live in-game scoreboard; an online system is otherwise a *restriction* of the offline one (detector, features, event models, eval set all transfer), so offline-first is low-regret and can gain an online mode later without rebuilding the hard parts.

**Phase 2 scoping.** Don't chase the full box score first. Points / shot attempts (ball + hoop detection + trajectory) → a **shot chart** is the tractable, demoable MVP. Assists/rebounds need robust possession tracking; defer them. For event spotting, evaluate a video **VLM** (e.g. Qwen2.5-VL / Gemini / InternVideo2) before training a bespoke classifier — it sidesteps the data-scarcity problem.

**Keep / replace / drop** (relative to current code):
- *Kept:* `label_players.py` (feeds SAM 2 prompts — the most reusable asset), `detect.py`, `utils.py`, and the action-model `VideoDataset` / `VideoTransform` scaffolding.
- *Done (replaced):* the online `IdentityManager` (per-frame Hungarian + EMA) → offline tracklet clustering in `offline_track/`; OSNet `x0_25` → DINOv2; `create_replay.py` I/O → `offline_track/render.py`.
- *Removed:* the online tracker (`track_players.py`, `reid_manager.py`, `create_replay.py`) and dead files (`assign_IDs.py`, `segmentation.py`, `main.py`) — recover from git history if needed. Salvage for phase 2: ball-possession-by-nearest-player-IoU and the `FrameRecord` structure (from the old `segmentation.py`). Deprioritize the color-histogram path (weak signal in pickup ball — shirts-vs-skins, no jersey numbers).

## Running the code

There is **no build, lint, or test setup** in this repo — no `requirements.txt`/`environment.yml`, no test framework, no CI. Don't go looking for them. Development happens by running scripts directly inside the **`selfstat` conda env** (`/opt/miniconda3/envs/selfstat`, Python 3.12; configured via `.vscode/settings.json`). That env has torch 2.12 (MPS), torchvision, ultralytics 8.4, scipy, opencv, numpy — but **not** `torchreid` or `scikit-learn`, so the legacy OSNet `reid_manager.py` import path no longer resolves. Run things with `/opt/miniconda3/envs/selfstat/bin/python`.

**Active offline pipeline — `run_offline.py` (the `offline_track/` package).** This is the implementation of the committed Direction: video in → per-frame boxes labeled with stable, post-hoc-resolved player IDs out (plus a `tracks.json`).

```bash
python run_offline.py --video clip.mp4                 # infer head count by clustering threshold
python run_offline.py --video clip.mp4 --num-ids 6     # closed-set: known roster size (preferred)
python run_offline.py --video clip.mp4 --max-frames 200  # cap frames while iterating
# output -> ./annotated_replays/tracked_<name>.mp4  +  ./annotated_replays/tracked_<name>.tracks.json
# weights (yolo11n.pt, DINOv2 ViT-S/14) auto-download on first run.
```

Action-recognition finetuning lives under `act_reg_model/` and uses a sibling-module import (`from data_prep import ...`), so it must be run **from inside that directory**:

```bash
cd act_reg_model && python finetune.py
python data_dir/organize_data.py   # one-time: restructure the MultiSubjects dataset splits
```

### Required runtime assets (all gitignored and absent from a fresh clone)

The entry points reference files/dirs that are not in the repo. Before anything runs you must supply:

- `./annotated_replays/` — output dir for the labeled video + `tracks.json` (auto-created by `offline_track/render.py`). The offline pipeline's detector/embedder weights (`yolo11n.pt`, DINOv2) auto-download.
- `./data_dir/train_val_test/{train,val,test}/...` plus `train.txt`/`val.txt` — the MultiSubjects action dataset, for `act_reg_model/`.
- `./act_reg_model/best_models/` — checkpoint output dir for finetuning.

### Inferred dependencies

No manifest exists; install these into the conda env as needed (names as imported): `ultralytics`, `torch`, `torchvision`, `scipy`, `opencv-python` (`cv2`), `numpy`, `tqdm`. Device selection prefers Apple **MPS**, then CUDA, then CPU — this is primarily a Mac dev environment.

## Architecture

One tracking pipeline (the offline one, below), reusable building blocks kept from the removed online prototype, an action-recognition pipeline, and dataset tooling.

### Offline tracking — `offline_track/` (active)

`run_offline.py` → `offline_track.pipeline.run`. Two passes over the video with global identity resolution in between — the committed offline design:

1. **`tracklets.py` (`collect_tracklets`)** — Pass 1. Runs Ultralytics YOLO + a tracker (`bytetrack.yaml` default, `botsort.yaml` available) over the whole video, `classes=[0]` (person). Each tracker ID becomes a `Tracklet` (per-frame `Detection`s + a reservoir-sampled, bounded set of crops, so appearance is summarized without holding every frame in memory). The tracker is trusted only for *short-term* association.
2. **`embedder.py` (`Embedder`)** — appearance features via **DINOv2 ViT-S/14** (torch.hub, `trust_repo=True`), falling back to torchvision ResNet50 if the hub fetch fails. A tracklet is summarized by the mean-pooled, L2-normalized per-crop feature.
3. **`cluster.py` (`cluster_tracklets`)** — the heart of the offline approach. Agglomerative (average-linkage) clustering over cosine distances with a hard **cannot-link constraint**: two tracklets that share any frame are necessarily different people and may never merge (a global constraint only offline can enforce). `--num-ids` cuts to a known roster size (respecting the cannot-link floor); otherwise it cuts at `--cluster-threshold` cosine distance. Player IDs are renumbered by first appearance. scipy only (no sklearn).
4. **`render.py`** — Pass 2. Redraws the source video with one color-coded labeled box per detection (`render_labeled_video`), and dumps the per-frame tracking table to `*.tracks.json` (`export_tracks`) — the `tracks` half of the canonical data model.

**Known quality limitation (verified, not a bug):** within-frame identities are provably consistent (cannot-link guarantees no duplicate ID in a frame), but *cross-time* re-merging is only as good as the appearance features. On hard footage (small, top-down, background-heavy crops) DINOv2 doesn't reliably re-match the same person across time gaps, so one player ID can still span more than one physical person. Next levers: pass `--num-ids` (closed-set); background-suppress crops with segmentation masks before embedding; or swap in re-ID-tuned features. The SAM 2 backbone (Direction doc) is the planned Pass-1 upgrade.

### Retained building blocks (from the removed online prototype)

The online per-frame tracker was removed in the offline pivot (`track_players.py`, `reid_manager.py`/`IdentityManager`, `create_replay.py` — recover from git history if needed). Three reusable pieces were kept:

1. **`label_players.py` (`supervised_label`)** — a human-in-the-loop OpenCV GUI: scrub to a reference frame, click each detected person box, type a label. Returns `(sv_ids, crops, selected_boxes)`. This is the intended seed for SAM 2 prompting and the most reusable asset.
2. **`detect.py` (`YOLODetector`)** — thin Ultralytics YOLO wrapper (`detect_video` streams per-frame results; `detect_frame` returns one). Pairs with the labeling UI.
3. **`utils.py`** — shared helpers (IoU, crop, center/wh, person-box filtering, video metadata, cosine sim, normalize) used by `label_players.py`.

### Pipeline B — action recognition (separate, not integrated)

Under `act_reg_model/`. A **R(2+1)D-18** video classifier (`finetune.py`) finetuned on ~2.5s clips (63 frames @ 25fps) to classify 3 actions: dribble / layup / shoot. `data_prep.py` defines the `VideoDataset` (reads clips, random temporal crop/pad to a fixed length) and `VideoTransform` (resize + ImageNet normalize). Best checkpoint by val accuracy is saved to `act_reg_model/best_models/`. The class set is intended to grow as more stats are added.

### Dataset tooling

`data_dir/organize_data.py` restructures the raw "MultiSubjects" dataset into `train/val/test` folders by action label, parsing the label from the filename convention `xxx_[d/p/s]_x.mp4` (`d`=dribble, `p`=layup, `s`=shoot).

## Conventions & gotchas

- In `label_players.py` / `utils.py`, **bounding boxes are Ultralytics `Boxes` objects, not tuples** — they carry `.xyxy`/`.xywh`/`.cls`; use the `utils.py` accessors. (The `offline_track/` pipeline converts to plain `(x1,y1,x2,y2)` tuples up front instead.)
- **`utils.get_corners(box)` returns a lazy `map` object, not a tuple** — it's single-use. Unpacking it once (`x1,y1,x2,y2 = ...`) is fine; reusing the same return value twice yields empties. `crop_frame(box, frame)` takes box **first**, frame second.- The action-recognition convention uses "**label**" for the string (dribble/layup/shoot) and "**class**" for the int (0/1/2) — kept consistent across `data_prep.py`/`finetune.py`.

## Known bugs

The legacy online tracker (`track_players.py`, `reid_manager.py`, `create_replay.py`) and dead files (`assign_IDs.py`, `segmentation.py`, `main.py`) have been removed. One bug remains in a live file:

- **`act_reg_model/finetune.py`** — `annotation_file` uses `../data_dir/...` but `data_root` uses `../../data_dir/...`; the paths are inconsistent and `data_root` appears to have one `../` too many. Reconcile against where you actually run it before training.
